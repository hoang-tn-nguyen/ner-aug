from __future__ import annotations
from data import Sample
import numpy as np
import random
from nltk.corpus import wordnet, stopwords


def generate_samples_by_entities(
    sample: Sample, class_entities: dict, replace_prob: float, num_aug_samples: int
) -> list[Sample]:
    generated_samples = []
    for _ in range(num_aug_samples):
        generated_sample = sample
        class_spans = generated_sample.get_spans()
        for category in class_spans:
            for i in range(len(class_spans[category])):
                is_replace = np.random.binomial(n=1, p=replace_prob)
                if is_replace:
                    s_id, e_id = class_spans[category][i]
                    replace_toks = random.choice(class_entities[category])
                    generated_sample = generated_sample.replace_span(
                        s_id, e_id, replace_toks, category
                    )
                    class_spans = generated_sample.get_spans()
        generated_samples.append(generated_sample)
    return generated_samples


def generate_samples_by_synonyms(
    sample: Sample, replace_prob: float, num_aug_samples: int
) -> list[Sample]:
    stop_words = stopwords.words("english")
    generated_samples = []
    for _ in range(num_aug_samples):
        generated_sample = sample
        generated_mask = np.random.binomial(
            n=1, p=replace_prob, size=len(generated_sample)
        )
        pos_alignment = 0
        for i in range(len(sample)):
            token, tag = sample[i]
            if generated_mask[i] and token not in stop_words and tag == "O":  # token to be replaced
                synonyms = set()
                for syn in wordnet.synsets(token):
                    for l in syn.lemmas():
                        synonym = l.name().replace("_", " ").replace("-", " ")
                        synonyms.add(synonym)
                if token in synonyms:
                    synonyms.remove(token)
                if len(synonyms) == 0:
                    continue
                synonym = random.choice(list(synonyms)).split()
                # "I [love] dogs because [they] are [cute]"
                # --> "I [have great affection with] dogs because [the dogs] are [lovely]"
                # --> pos_alignment = 0
                # --> if i = 1 (love) --> start_idx = i + pos_alignment = 1 in the new sentence (love --> have great affection with); pos_alginment += 4 - 1 = 3
                # --> if i = 4 (they) --> start_idx = 4 + 3 = 7 in the new sentence (they --> the dogs); pos_alginment += 2 - 1 = 4
                # --> if i = 6 (cute) --> start_idx = 6 + 4 = 10 in the new sentence; pos_algninment += 1 - 1 = 4
                start_idx = i + pos_alignment
                end_idx = i + 1 + pos_alignment
                generated_sample = generated_sample.replace_span(
                    start_idx, end_idx, synonym, tag
                )
                pos_alignment += len(synonym) - 1
        generated_samples.append(generated_sample)
    return generated_samples
