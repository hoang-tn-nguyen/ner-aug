# %%
import torch
import nlpaug.augmenter.word as naw


class Augmentor:
    def __init__(self, augmentation={"BERT": 0.3, "SYNONYM": 0.3}):
        self.augmentation = {
            "BERT": 0.3 if "BERT" not in augmentation else augmentation["BERT"],
            "SYNONYM": 0.3
            if "SYNONYM" not in augmentation
            else augmentation["SYNONYM"],
        }
        self.__init_aug__()

    def __init_aug__(self):
        self.bert_aug = naw.ContextualWordEmbsAug(
            model_path="bert-base-uncased",
            action="substitute",
            aug_p=self.augmentation["BERT"],
            top_k=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.trans_aug = naw.BackTranslationAug(
            from_model_name="facebook/wmt19-en-de",
            to_model_name="facebook/wmt19-de-en",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.syn_aug = naw.SynonymAug(
            aug_src="wordnet", aug_p=self.augmentation["SYNONYM"]
        )

    def generate_augmented_samples(self, sample, aug_type, aug_n=1):
        if aug_type in ["BERT"]:
            return self.generate_samples_by_BERT(sample, aug_n)
        elif aug_type in ["TRANSLATION", "TRANSLATE", "TRANS"]:
            return self.generate_samples_by_translation(sample, aug_n)
        elif aug_type in ["SYNONYM", "SYN"]:
            return self.generate_samples_by_synonym(sample, aug_n)
        else:
            print("Warning: Augmentation {} is unsupported!".format(aug_type))
            return None

    def generate_samples_by_BERT(self, sample, aug_n):
        gen_samples = self.bert_aug.augment(sample, aug_n)
        return gen_samples

    def generate_samples_by_translation(self, sample, aug_n):
        gen_samples = self.trans_aug.augment(sample, aug_n)
        return gen_samples

    def generate_samples_by_synonym(self, sample, aug_n):
        gen_samples = self.syn_aug.augment(sample, aug_n)
        return gen_samples


if __name__ == "__main__":
    import nltk

    nltk.download("averaged_perceptron_tagger")
    text = "This fox can run quickly than any living animal."
    augmentor = Augmentor()
    aug_text = augmentor.generate_augmented_samples(text, "SYN", 10)
    print(aug_text)
