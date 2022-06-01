# %%
from tqdm import tqdm

def collate_conll2003(samplelines):
    filelines = [line.split() for line in samplelines]
    words, _, _, tags = zip(*filelines)
    return words, tags


def collate_mitmovie(samplelines):
    filelines = [line.split() for line in samplelines]
    tags, words = zip(*filelines)
    return words, tags


def collate_mitrestaurant(samplelines):
    filelines = [line.split() for line in samplelines]
    tags, words = zip(*filelines)
    return words, tags


def collate_fewnerd(samplelines):
    filelines = [line.split() for line in samplelines]
    words, tags = zip(*filelines)
    return words, tags


def normalize_data(file_path, out_file, collate_fn):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    f_out = open(out_file, "w")

    samplelines = []
    for line in tqdm(lines):
        line = line.strip()
        if line:
            samplelines.append(line)
        else:
            words, tags = collate_fn(samplelines)
            for word, tag in zip(words,tags):
                f_out.writelines('{} {}\n'.format(word,tag))
            f_out.writelines('\n')
            samplelines = []
    

if __name__ == "__main__":
    normalize_data("original_datasets/mit-movie/train.txt", 'datasets/mit-movie/train.txt', collate_mitmovie)
    normalize_data("original_datasets/mit-movie/test.txt", 'datasets/mit-movie/test.txt', collate_mitmovie)

    normalize_data("original_datasets/mit-restaurant/train.txt", 'datasets/mit-restaurant/train.txt', collate_mitrestaurant)
    normalize_data("original_datasets/mit-restaurant/test.txt", 'datasets/mit-restaurant/test.txt', collate_mitrestaurant)

    normalize_data("original_datasets/conll2003/train.txt", 'datasets/conll2003/train.txt', collate_conll2003)
    normalize_data("original_datasets/conll2003/valid.txt", 'datasets/conll2003/valid.txt', collate_conll2003)
    normalize_data("original_datasets/conll2003/test.txt", 'datasets/conll2003/test.txt', collate_conll2003)

