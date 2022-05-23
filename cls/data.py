# %%
import random
import torch.utils.data as data
from tqdm import tqdm
from augment import Augmentor


class Corpus(data.Dataset):
    def __init__(self, dataset, augmentation, return_type="pair"):
        print("==============================")
        print("Initialize Corpus:")
        self.dataset = dataset
        print("- Dataset size: {}".format(len(self.dataset)))
        self.augmentation = augmentation
        print("- Augmentation: {}".format(self.augmentation))
        self.augmentor = Augmentor(augmentation)
        self.augmented_dataset = self.augment_data(self.dataset)
        assert len(self.dataset) == len(self.augmented_dataset)
        self.return_type = return_type

    def augment_data(self, ori_dataset):
        aug_dataset = []
        for (sample, tag) in tqdm(ori_dataset, "Augmenting Data"):
            aug_n = self.augmentation["N"]
            aug_samples = []
            for aug_type in self.augmentation.keys():
                if aug_type == "N":
                    continue
                gen_samples = self.augmentor.generate_augmented_samples(
                    sample=sample, aug_type=aug_type, aug_n=aug_n
                )
                if gen_samples != None:
                    if isinstance(gen_samples, str):
                        aug_samples.extend([(gen_samples, tag)])
                    elif isinstance(gen_samples, list):
                        aug_samples.extend(
                            [(gen_samples[i], tag) for i in range(len(gen_samples))]
                        )
                    else:
                        raise TypeError("Unsupported type for gen_samples")
            aug_dataset.append(aug_samples)
        return aug_dataset

    def __getitem__(self, idx):
        ori = self.dataset[idx]
        aug = random.choice(self.augmented_dataset[idx])
        if self.return_type == "pair":
            return {
                "ori": ori,
                "aug": aug,
            }
        else:
            return random.choice([ori, aug])

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from datasets import IMDB

    random.seed(0)
    my_dataset = IMDB(dataset_size=25)
    my_corpus = Corpus(my_dataset, {"N": 10, "SYN": 0.3})
    print(my_corpus[0])
