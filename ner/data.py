from __future__ import annotations


class Sample:
    def __init__(self, toks: list[str], tags: list[str]) -> None:
        self.toks = toks
        self.tags = tags
        self.__preprocess__()

    def __preprocess__(self) -> None:
        self.tags = list(map(self.__strip_class_name__, self.tags))

    def __strip_class_name__(self, raw_tag: str) -> str | list[str]:
        # Remove B- and I- from input string
        if raw_tag.startswith("B-") or raw_tag.startswith("I-"):
            raw_tag = raw_tag[2:]
        return raw_tag

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.toks[idx], self.tags[idx]

    def __len__(self) -> int:
        return len(self.toks)

    def get_spans(self) -> dict[str, list[tuple]]:
        class_span = {}
        current_tag = None
        i = 0
        while i < len(self.tags):
            if self.tags[i] != "O":
                start = i
                current_tag = self.tags[start]
                # --- Increase counter until a new tag or empty tag is found ---
                i += 1
                while i < len(self.tags) and self.tags[i] == current_tag:
                    i += 1
                # --------------------------------------------------------------
                end = i
                if current_tag in class_span:
                    class_span[current_tag].append((start, end))
                else:
                    class_span[current_tag] = [(start, end)]
            else:
                i += 1
        return class_span

    def replace_span(
        self,
        replace_start_idx: int,
        replace_end_idx: int,
        replace_toks: list[str],
        replace_tags: list[str] | str,
    ) -> Sample:
        # Align tag sequence with tok sequence
        if isinstance(replace_tags, list):
            if len(replace_tags) == 1:
                replace_tags = [replace_tags[0]] * len(replace_toks)
        if isinstance(replace_tags, str):
            replace_tags = [replace_tags] * len(replace_toks)

        left_toks = self.toks[:replace_start_idx]
        right_toks = self.toks[replace_end_idx:]
        left_tags = self.tags[:replace_start_idx]
        right_tags = self.tags[replace_end_idx:]
        new_toks = left_toks + replace_toks + right_toks
        new_tags = left_tags + replace_tags + right_tags
        return Sample(new_toks, new_tags)

    def replace(
        self, idx: int, replace_tok: str | None = None, replace_tag: str | None = None
    ) -> None:
        if replace_tok:
            self.toks[idx] = replace_tok
        if replace_tag:
            self.tags[idx] = replace_tag

    def __str__(self) -> str:
        newlines = zip(self.toks, self.tags)
        return "\n".join(["\t".join(line) for line in newlines])

    def __repr__(self) -> str:
        return self.__str__()


class Corpus:
    def __init__(self, text_file: str) -> None:
        self.samples = self.get_samples(text_file)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def get_samples(self, text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        samples = []
        sample_lines = []
        for line in lines:
            line = line.strip()
            if line:
                sample_lines.append(line)
            else:
                words, tags = self.__get_data__(sample_lines)
                sample_lines = []
                samples.append(Sample(words, tags))

        if sample_lines:
            words, tags = self.__get_data__(sample_lines)
            samples.append(Sample(words, tags))
        return samples

    def __get_data__(self, sample_lines: list[str]) -> tuple[list[str], list[str]]:
        filelines = [line.split() for line in sample_lines]
        words, tags = zip(*filelines)
        return list(words), list(tags)

    def get_entities(self, as_string=False):
        class_entities = {}
        mentions = set()
        for sample in self.samples:
            class_spans = sample.get_spans()
            for key in class_spans:
                for start_idx, end_idx in class_spans[key]:
                    mention = " ".join(sample.toks[start_idx:end_idx])
                    if mention not in mentions:
                        if key in class_entities:
                            if as_string:
                                class_entities[key].append(mention)
                            else:
                                class_entities[key].append(
                                    sample.toks[start_idx:end_idx]
                                )
                        else:
                            if as_string:
                                class_entities[key] = [mention]
                            else:
                                class_entities[key] = [sample.toks[start_idx:end_idx]]
                    mentions.add(mention)
        return class_entities


if __name__ == "__main__":
    my_corpus = Corpus("datasets/conll2003/train.txt")
    print("Original:", my_corpus[10])
    from augment import (
        generate_sentences_by_shuffle_within_segments,
        generate_samples_by_synonyms,
        generate_samples_by_entities,
    )

    outputs = generate_samples_by_entities(my_corpus[10], my_corpus.get_entities(), 0.3, num_aug_samples=1)
    print("After:", outputs)
    print("Before:", my_corpus[10])
