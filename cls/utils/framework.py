from __future__ import annotations
from typing import *
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

AnyDeepModel = TypeVar("AnyDeepModel")
AnyOptimizer = TypeVar("AnyOptimizer")
AnyScheduler = TypeVar("AnyScheduler")
AnyDataLoader = TypeVar("AnyDataLoader")


class Metric:
    """This is an abstract class for implementing measurements.
    Use this as a superclass to implement subclasses with loss functions, accuracy/f1/precision/recall scores, ...
    """

    def __init__(self) -> None:
        self.default_metric: str = "loss"
        self.default_metric_sign: int = -1

    def __call__(self, *args, **kwargs) -> dict[str, Any]:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("Please implement the forward() method.")


class Framework:
    """This Framework comebines all save, load, train, test, ... methods for training PyTorch models.
    The design is general so that it works with any model and any problem.

    Formats:
        dataset/dataloader = {
            'input_name_X': data_X,
            'input_name_Y': data_Y,
            'input_name_Z': data_Z,
            ...
        }

        data_pipeline_mapper = \n
        (1) dict (multiple input with mapping)\n
        {
            'input_name_X': 'model_param_A',
            'input_name_Y': 'model_param_B',
            'input_name_Z': 'model_param_C',
            ...
        }

        (2) list/tuple (multiple inputs with sequential order)\n
        ['input_name_X', 'input_name_Y', 'input_name_Z', ...]

        (3) str (single input)\n
        'data_input_N'

    Note:
        This framework is an interation-based training approach (not epoch-based).
        For iteration-based approach, the learning rate, evaluation, or saving model is adjusted by the number of iterations.
        It is robust against big dataset because epoch-based approach may take too long to validate or save best models.
    """

    def __init__(
        self,
        model: AnyDeepModel,
        data_pipeline_mapper: str | list[str] | dict[str, str],
        optimizer: str | AnyOptimizer = "auto",
        scheduler: str | AnyScheduler = "auto",
        save_ckpt: Optional[str] = None,
        load_ckpt: Optional[str] = None,
        configs: dict = None,
    ) -> None:

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = model.to(self.device)
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.device = torch.device("cpu")
            self.model = model.to(self.device)
            self.scaler = None

        self.data_pipeline_mapper = data_pipeline_mapper

        if optimizer != "auto":
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(
                    filter(lambda p: p.requires_grad, self.model.parameters())
                ),
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
            )

        if scheduler != "auto":
            self.scheduler = scheduler
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=1000,  # Number of iterations until restart
                T_mult=1,  # Scaling T_0 by T_mult after each restart
                verbose=False,
            )

        self.save_ckpt = save_ckpt
        self.load_ckpt = load_ckpt
        if self.load_ckpt:
            print("Loading:", self.load_ckpt)
            self.saved_stats = self.load(self.load_ckpt)
            print("Statistics:", self.saved_stats)
        else:
            self.saved_stats = None
            self.configs = configs

    def train(
        self,
        criterion: Metric,  # Metrics to compute loss/f1/precision/recall/...
        data_loader: AnyDataLoader,  # Train loader
        val_loader: Optional[
            AnyDataLoader
        ] = None,  # If None, use data_loader as validation set
        num_iters: int = 10000,  # Train for N batches
        val_iters: int = 10,  # Load N batches for validation
        refresh_rate: int = 100,  # Refresh/update results after every N batches
        grad_iters: int = 1,  # Delay loss.backward() for every N batches
        uda_loader: Optional[
            AnyDataLoader
        ] = None,  # External loader for consistency training
        uda_criterion: Optional[Metric] = None,  # UDA consistency loss
    ) -> dict[str, Any]:

        if not val_loader:
            val_loader = data_loader
        if uda_loader:
            uda_loader_iter = iter(uda_loader)

        self.model.train()
        self.optimizer.zero_grad()  # zero_grad before backward
        # Initialize variables
        data_stats = {}
        compare_metric = criterion.default_metric
        metric_sign = criterion.default_metric_sign
        if self.saved_stats:
            best_metric = self.saved_stats[compare_metric]
        else:
            best_metric = 1e9 if metric_sign < 0 else 0
        # Train the model
        bar_format = (
            "[{n_fmt}/{total_fmt} | {elapsed}<{remaining} | {rate_fmt}{postfix}] {desc}"
        )
        prog_bar = tqdm(range(num_iters), bar_format=bar_format)
        cnt_iters = 0
        while cnt_iters < num_iters:
            for data in data_loader:
                if cnt_iters >= num_iters:
                    break
                prog_bar.update()
                data = self.data_to_device(data, self.device)

                if uda_loader:
                    try:
                        uda_data = uda_loader_iter.next()
                    except:
                        uda_loader_iter = iter(uda_loader)
                        uda_data = uda_loader_iter.next()
                    uda_data = self.data_to_device(uda_data, self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.data_to_model(data)
                        # Back-propagate and update weights
                        if uda_loader:
                            aug_output = self.data_to_model(uda_data["aug"])
                            with torch.no_grad():
                                ori_output = self.data_to_model(uda_data["ori"])
                            stats = uda_criterion(
                                output=output,
                                data=data,
                                aug_output=aug_output,
                                aug_target=ori_output,
                                uda_data=uda_data,
                            )
                        else:
                            stats = criterion(output, data)
                        assert isinstance(stats, dict) and "loss" in stats
                        loss = stats["loss"] / grad_iters

                    self.scaler.scale(loss).backward()
                    if (
                        (grad_iters <= 1)
                        or ((cnt_iters + 1) % grad_iters == 0)
                        or (cnt_iters + 1) == num_iters
                    ):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    output = self.data_to_model(data)
                    # Back-propagate and update weights
                    if uda_loader:
                        aug_output = self.data_to_model(uda_data["aug"])
                        with torch.no_grad():
                            ori_output = self.data_to_model(uda_data["ori"])
                        stats = uda_criterion(
                            output=output,
                            data=data,
                            aug_output=aug_output,
                            aug_target=ori_output,
                            uda_data=uda_data,
                        )
                    else:
                        stats = criterion(output, data)
                    assert isinstance(stats, dict) and "loss" in stats
                    loss = stats["loss"] / grad_iters

                    loss.backward()
                    if (
                        (grad_iters <= 1)
                        or ((cnt_iters + 1) % grad_iters == 0)
                        or (cnt_iters + 1) == num_iters
                    ):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                # Update lr_scheduler
                if self.scheduler:
                    if not isinstance(
                        self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step()
                # Mark data as processed
                cnt_iters += 1
                # Update & reset data every refresh_rate
                if cnt_iters % refresh_rate == 0:
                    data_stats = self.test(
                        criterion, val_loader, val_iters, prog_bar=False
                    )
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(metrics=data_stats[compare_metric])
                    if (
                        data_stats[compare_metric] * metric_sign
                        > best_metric * metric_sign
                    ):
                        best_metric = data_stats[compare_metric]
                        if self.save_ckpt:
                            self.save(self.save_ckpt, stats=data_stats)
                prog_bar.set_description_str(
                    self.stats_to_str(data_stats)
                    + (
                        " | Sup Loss: {:.4f}".format(stats["sup_loss"].item())
                        if uda_loader
                        else ""
                    )
                    + (
                        " | Uda Loss: {:.4f}".format(stats["uda_loss"].item())
                        if uda_loader
                        else ""
                    )
                    + (" | Best {}: {:.3f}".format(compare_metric, best_metric))
                )
        return data_stats

    def test(
        self,
        criterion: Metric | None,
        data_loader: AnyDataLoader,  # test loader
        num_iters: int = 1000000000,  # scan through N batches of data
        inspection: bool = False,  # return dictionary for inputs and outputs
        return_inputs: bool = False,  # Set this flag to true to return data inputs
        return_outputs: bool = True,  # Set this flag to true to return computed results
        prog_bar: bool = True,  # Show progress bar
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:

        self.model.eval()
        # Initialize variables
        inputs = []
        outputs = []
        data_stats = {}
        cnt_iters = 0
        num_iters = min(num_iters, len(data_loader))
        # Disable error back-propagation
        with torch.no_grad():
            if prog_bar:
                progress_bar = tqdm(range(num_iters))
            for data in data_loader:
                if cnt_iters < num_iters:
                    if prog_bar:
                        progress_bar.update()
                    data = self.data_to_device(data, self.device)
                    output = self.data_to_model(data)

                    if inspection:
                        if return_inputs:
                            inputs.append(data)
                        if return_outputs:
                            outputs.append(output)
                    if criterion != None:
                        stats = criterion(output, data)
                        assert isinstance(stats, dict)
                        if data_stats:
                            for key in stats:
                                try:
                                    data_stats[key] += float(stats[key] / num_iters)
                                except:
                                    pass
                        else:
                            for key in stats:
                                try:
                                    data_stats[key] = float(stats[key] / num_iters)
                                except:
                                    pass
                    cnt_iters += 1  # Increase the count for processed data
                else:
                    break
        # Reactivate training phase
        self.model.train()

        if inspection:
            return data_stats, {"input": inputs, "output": outputs}
        else:
            return data_stats

    def save(self, ckpt: str, stats: Optional[dict[str, Any]] = None) -> None:
        torch.save(
            {
                # --- Model Statistics ---
                "stats": stats,
                "configs": self.configs,
                # --- Model Parameters ---
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer != None
                else None,
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler != None
                else None,
            },
            ckpt,
        )

    def load(self, ckpt: str) -> dict[str, Any]:
        try:
            checkpoint = torch.load(ckpt, map_location=self.device)
        except Exception as e:
            print(e)
            return None

        # --- Model Statistics ---
        stats = checkpoint["stats"]
        try:
            self.configs = checkpoint["configs"]
            print("Configs loaded.")
        except:
            print("No configs available.")
        # --- Model Parameters ---
        if self.model != None:
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print(e)
                print("Cannot load the model.")
        if self.optimizer != None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:  # Input optimizer doesn't fit the checkpoint one --> should be ignored
                print(e)
                print("Cannot load the optimizer.")
        if self.scheduler != None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:  # Input scheduler doesn't fit the checkpoint one --> should be ignored
                print(e)
                print("Cannot load the scheduler.")
        return stats

    @staticmethod
    def get_configs(load_ckpt: str):
        try:
            if not torch.cuda.is_available():
                checkpoint = torch.load(load_ckpt, map_location="cpu")
            else:
                checkpoint = torch.load(load_ckpt)
            return checkpoint["configs"]
        except Exception as e:
            print(e)
            return None

    def data_to_device(self, data: Any, device: Any) -> Any:
        """This function moves data to CPU/GPU recursively (i.e. Tensor, tuple, list, dict).
        Anything else such as int, float, np.array, etc, will be ignored.
        """
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, tuple):
            data = tuple(self.data_to_device(item, device) for item in data)
        elif isinstance(data, list):
            data = list(self.data_to_device(item, device) for item in data)
        elif isinstance(data, dict):
            data = dict((key, self.data_to_device(data[key], device)) for key in data)
        else:
            pass  # keep as is for other data types
        return data

    def data_to_model(self, data: Any) -> Any:
        """
        Map data inputs to model's parameters.
        """
        if isinstance(data, dict):
            if self.data_pipeline_mapper:
                if isinstance(self.data_pipeline_mapper, dict):
                    kwargs = []
                    for input_key, model_key in self.data_pipeline_mapper.items():
                        kwargs.append((model_key, data[input_key]))
                    kwargs = dict(kwargs)
                    output = self.model(**kwargs)
                elif isinstance(self.data_pipeline_mapper, list) or isinstance(
                    self.data_pipeline_mapper, tuple
                ):
                    args = [data[input_key] for input_key in self.data_pipeline_mapper]
                    output = self.model(*args)
                else:
                    input_key = self.data_pipeline_mapper
                    output = self.model(data[input_key])
                return output
            else:
                raise ValueError("self.data_pipeline_mapper cannot be None!")
        else:
            if self.data_pipeline_mapper:
                if isinstance(self.data_pipeline_mapper, list) or isinstance(
                    self.data_pipeline_mapper, tuple
                ):
                    kwargs = dict(
                        [
                            (model_key, data[i])
                            for i, model_key in enumerate(self.data_pipeline_mapper)
                            if model_key
                        ]
                    )
                    output = self.model(**kwargs)
                else:
                    model_key = self.data_pipeline_mapper
                    kwargs = dict([(model_key, data)])
                    output = self.model(**kwargs)
                return output
            else:
                raise ValueError("self.data_pipeline_mapper cannot be None!")

    def data_unbatchify(self, data: Any, dim: int = 0, transpose_flag=True) -> Any:
        """
        Unbatchify the data.
        (out_1, out_2, out_3, ...) where out_i is a list
        {'out_1': out_1, 'out_2': out_2, ...} where out_i is a list
        [batch_1, batch_2, batch_3, ...]
        """
        if isinstance(data, list):
            if isinstance(data[0], list):
                out_list = []
                for batch_item in data:
                    out_list.extend(batch_item)
                return out_list
            else:  # each element in the list is a batch
                if not transpose_flag:
                    return data
                else:
                    data = self.data_transpose(data, dim)
                    transpose_flag = not isinstance(
                        data, list
                    )  # infinite loop list([1,2,3]) --> tranponse --> list([1,2,3])
                    return self.data_unbatchify(data, dim, transpose_flag)
        elif isinstance(data, tuple):
            return tuple(
                [
                    self.data_unbatchify(data[i], dim, transpose_flag)
                    for i in range(len(data))
                ]
            )
        elif isinstance(data, dict):
            out_dict = {}
            for k in data:
                out_dict[k] = self.data_unbatchify(data[k], dim, transpose_flag)
            data = out_dict
            return data
        else:  # Tensor, int, float, str, ...
            return data

    def data_transpose(self, data: list[Any], dim: int = 0) -> Any:
        """
        Convert list of elements into elements' structure.
        Example:
            (no inner / undefined structure)
            [1,2,3,4,5]
            --> [1,2,3,4,5]

            (inner structure is a list)
            [[1,2],[3,4],[5,6]]
            --> [[1,3,5], [2,4,6]]

            (inner structure is a tuple)
            [(1,2),(3,4),(5,6),...]
            --> ([1,3,5,...], [2,4,6,...])

            (inner structure is a dict)
            [
                {a:1,b:2,c:3},
                {a:2,b:3,c:4},
                {a:3,b:4,c:5},
            ]
            --> {a: [1,2,3], b: [2,3,4], c: [3,4,5]}

            (nested structure)
            [
                (1,(2,3),4),
                (1,[2,3],4),
                (1,[2,3],4)
            ]
            --> (
                    [1, 1, 1],
                    [(2, 3), [2, 3], [2, 3]],
                    [4, 4, 4]
                )
        """
        if isinstance(data[0], dict):
            out_dict = {}
            for k in data[0]:
                out_dict[k] = []
            for item in data:
                for k in item:
                    out_dict[k].append(item[k])
            return out_dict
        elif isinstance(data[0], tuple):
            out_tuple = []
            for col in range(len(data[0])):
                out_col = []
                for item in data:
                    out_col.append(item[col])
                out_tuple.append(out_col)
            out_tuple = tuple(out_tuple)
            return out_tuple
        elif isinstance(data[0], list):
            out_list = []
            for col in range(len(data[0])):
                out_col = []
                for item in data:
                    out_col.append(item[col])
                out_list.append(out_col)
            return out_list
        elif isinstance(data[0], torch.Tensor):
            return torch.cat(data, dim=dim)
        elif isinstance(data[0], np.ndarray):
            return np.concatenate(data, axis=dim)
        else:
            return data

    def stats_to_str(self, stats: dict[str, Any]) -> str:
        return " | ".join(
            ["{}: {:.3f}".format(key, value) for key, value in stats.items()]
        )
