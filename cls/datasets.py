# %%
import os
import glob
import random
import torch.utils.data as data
from tqdm import tqdm


class IMDB(data.Dataset):
    def __init__(
        self,
        dir="datasets/aclImdb/train/",
        shuffle=True,
        dataset_size=-1,
        with_unsup=False,
    ):
        print("==============================")
        print("Initialize Dataset: {}".format(dir))
        self.with_unsup = with_unsup
        self.samples = self.__input_data__(dir)
        if shuffle:
            print("- Shuffle dataset: {}".format(shuffle))
            random.shuffle(self.samples)
        if dataset_size > 0:
            print("- Resize: from {} to {}".format(len(self.samples), dataset_size))
            self.samples = self.samples[:dataset_size]
        print("- Dataset size: {}".format(len(self)))

    def __input_data__(self, dir):
        pos_files = glob.glob("{}/pos/*.txt".format(dir))
        neg_files = glob.glob("{}/neg/*.txt".format(dir))
        if os.path.exists("{}/unsup/".format(dir)):
            unsup_files = glob.glob("{}/unsup/*.txt".format(dir))
        else:
            unsup_files = None

        pos_txt = []
        for txt_file in tqdm(pos_files, "Reading positive files"):
            with open(txt_file, "r", encoding="utf-8") as f:
                txt = f.readlines()
                if len(txt) == 1:
                    txt = txt[0]
                pos_txt.append((txt, 1))

        neg_txt = []
        for txt_file in tqdm(neg_files, "Reading negative files"):
            with open(txt_file, "r", encoding="utf-8") as f:
                txt = f.readlines()
                if len(txt) == 1:
                    txt = txt[0]
                neg_txt.append((txt, 0))

        if unsup_files and self.with_unsup:
            unsup_txt = []
            for txt_file in tqdm(unsup_files, "Reading unsup files"):
                with open(txt_file, "r", encoding="utf-8") as f:
                    txt = f.readlines()
                    if len(txt) == 1:
                        txt = txt[0]
                    unsup_txt.append((txt, -1))
        else:
            unsup_txt = []

        samples = pos_txt + neg_txt + unsup_txt
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1]


if __name__ == "__main__":
    train_data = IMDB("datasets/aclImdb/train/")
    test_data = IMDB("datasets/aclImdb/test/")
