import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.framework import Metric
from sklearn import metrics


class Evaluator:
    def __init__(
        self,
        zero_division=0,
        list_metrics=["macro", "micro", "accuracy", "precision", "recall", "f1"],
    ):
        self.zero_division = zero_division
        self.list_metrics = list_metrics
        self.accuracy_score = metrics.accuracy_score
        self.f1_score = metrics.f1_score
        self.precision_score = metrics.precision_score
        self.recall_score = metrics.recall_score

    def forward(self, output, target):
        stats = {}
        if "accuracy" in self.list_metrics:
            stats["acc"] = self.accuracy_score(target, output)
        if "micro" in self.list_metrics:
            if "precision" in self.list_metrics:
                stats["mic_pr"] = self.precision_score(
                    target, output, average="micro", zero_division=self.zero_division
                )
            if "recall" in self.list_metrics:
                stats["mic_rc"] = self.recall_score(
                    target, output, average="micro", zero_division=self.zero_division
                )
            if "f1" in self.list_metrics:
                stats["mic_f1"] = self.f1_score(
                    target, output, average="micro", zero_division=self.zero_division
                )
        if "macro" in self.list_metrics:
            if "precision" in self.list_metrics:
                stats["mac_pr"] = self.precision_score(
                    target, output, average="macro", zero_division=self.zero_division
                )
            if "recall" in self.list_metrics:
                stats["mac_rc"] = self.recall_score(
                    target, output, average="macro", zero_division=self.zero_division
                )
            if "f1" in self.list_metrics:
                stats["mac_f1"] = self.f1_score(
                    target, output, average="macro", zero_division=self.zero_division
                )
        return stats

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class CELossWithMetric(Metric):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.metrics = Evaluator()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)
        self.default_metric = "acc"
        self.default_metric_sign = 1

    def forward(self, output, data):
        label = data[1]
        logit = output
        loss = self.loss_fn(logit, label)
        label = label.view(-1).cpu()
        pred = logit.max(dim=-1)[1].view(-1).cpu()
        pred = pred.numpy().tolist()
        label = label.numpy().tolist()
        return self.compute_scores(loss, pred, label)

    def compute_scores(self, loss, pred, label):
        stats = self.metrics(pred, label)
        stats["loss"] = loss
        return stats


class KLLossWithMetric(Metric):
    def __init__(self, coeff=1, softmax_temp=0.85, conf_threshold=0.45):
        super().__init__()
        self.loss_fn = nn.KLDivLoss(reduction="none")
        self.coeff = coeff
        self.softmax_temp = softmax_temp
        self.conf_threshold = conf_threshold

    def forward(self, output, target, data=None, use_argmax=False):
        if use_argmax:
            soft_target = torch.softmax(target, dim=-1)  # (B,)
            conf_mask = (
                torch.max(soft_target, dim=-1)[0] > self.conf_threshold
            ).float()  # (B,)
            target = torch.max(soft_target, dim=-1)[1]
            target = F.one_hot(target)
        else:
            soft_target = torch.softmax(target, dim=-1)  # (B,)
            conf_mask = (
                torch.max(soft_target, dim=-1)[0] > self.conf_threshold
            ).float()  # (B,)
            target = torch.softmax(target / self.softmax_temp, dim=-1)  # (B,C)

        output = F.log_softmax(output, dim=-1)  # (B,C)
        # Measure the difference between two distributions

        loss = -target * output
        # loss = self.loss_fn(output, target)
        loss = torch.sum(loss, dim=-1)  # (B,)

        size = torch.sum(conf_mask, dim=-1)  # (1,) elements can be zero
        ones = torch.ones_like(size).to(size.device)  # (1,) all elements are ones
        total = torch.max(size, ones)  # (B,) avoiding zero division
        loss_batch_mean = torch.sum(loss * conf_mask, dim=-1) / total
        return loss_batch_mean * self.coeff


class UDALossWithMetric(Metric):
    def __init__(self, train_iters=2000):
        self.ce_loss = CELossWithMetric(reduction="none")
        self.kl_loss = KLLossWithMetric()
        self.cur_iter = 0
        self.train_iters = train_iters

    def forward(self, output, data, aug_output, aug_target, uda_data=None):
        self.cur_iter += 1
        stats = self.ce_loss(output, data)
        sup_loss = stats["loss"]  # (B,)
        n_classes = output.shape[-1]
        tsa_threshold = self.get_tsa_threshold(
            self.cur_iter, self.train_iters, 1 / n_classes, 1, "linear_schedule"
        )
        unconf_mask = torch.exp(-sup_loss) <= tsa_threshold  # (B,C)
        size = torch.sum(unconf_mask, dim=-1)  # (B,) elements can be zero
        ones = torch.ones_like(size).to(size.device)  # (B,) all elements are ones
        total = torch.max(size, ones)  # (B,) avoiding zero division
        sup_loss = torch.sum(sup_loss * unconf_mask, dim=-1) / total

        uda_loss = self.kl_loss(aug_output, aug_target, uda_data, use_argmax=False)
        stats["sup_loss"] = sup_loss
        stats["uda_loss"] = uda_loss
        stats["loss"] = sup_loss + uda_loss
        return stats

    def get_tsa_threshold(
        self,
        current_train_steps,
        total_train_steps,
        start,
        end=1.0,
        schedule="linear_schedule",
    ):
        train_progress = torch.tensor(
            min(current_train_steps / total_train_steps, 1.0)
        )  # in range (0.0, 1.0)
        if schedule == "linear_schedule":
            threshold = train_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = torch.exp((train_progress - 1) * scale)
        elif schedule == "log_schedule":
            scale = 5
            threshold = 1 - torch.exp((-train_progress) * scale)
        output = start + threshold * (end - start)
        return output


class FixMatchLossWithMetric(Metric):
    def __init__(self, threshold=0.8):
        self.ce_loss = CELossWithMetric()
        self.kl_loss = KLLossWithMetric(conf_threshold=threshold)
        self.threshold = threshold

    def forward(self, output, data, aug_output, aug_target, uda_data=None):
        stats = self.ce_loss(output, data)
        sup_loss = stats["loss"]
        uda_loss = self.kl_loss(aug_output, aug_target, use_argmax=True)

        stats["sup_loss"] = sup_loss
        stats["uda_loss"] = uda_loss
        stats["loss"] = sup_loss + uda_loss
        return stats

