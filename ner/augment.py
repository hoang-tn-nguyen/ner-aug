from __future__ import annotations
from data import Sample
import numpy as np
import random
from nltk.corpus import wordnet, stopwords
from copy import deepcopy


def generate_samples_by_entities(
    sample: Sample, class_entities: dict, replace_prob: float, num_aug_samples: int
) -> list[Sample]:
    generated_samples = []
    for _ in range(num_aug_samples):
        generated_sample = deepcopy(sample)
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
        generated_sample = deepcopy(sample)
        generated_mask = np.random.binomial(
            n=1, p=replace_prob, size=len(generated_sample)
        )
        pos_alignment = 0
        for i in range(len(sample)):
            token, tag = sample[i]
            if (
                generated_mask[i] and token not in stop_words and tag == "O"
            ):  # token to be replaced
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


def _shuffle_within_segments(tags, replace_ratio):
    """
    Given a segmented sentence such as ["O", "O", "PER", "PER", "PER", "ORG", "ORG", "ORG", "ORG"],
    shuffle the token order within each segment
    """
    segments = [0]
    for i in range(len(tags)):
        if i == 0:
            continue
        if tags[i] != tags[i - 1]:
            segments.append(segments[-1] + 1)
        else:
            segments.append(segments[-1])
    # segments: [0 0 1 1 1 2 2 2 2]

    shuffled_idx = []
    start, end = 0, 0
    while start < len(segments) and end < len(segments):
        while end < len(segments) and segments[end] == segments[start]:
            end += 1
        segment = [i for i in range(start, end)]
        if len(segment) > 1 and np.random.binomial(1, replace_ratio, 1)[0] == 1:
            random.shuffle(segment)
        shuffled_idx += segment
        start = end
    return shuffled_idx  # [1,2,5,3,4,6,8,7,9]


def generate_sentences_by_shuffle_within_segments(
    sample: Sample, replace_prob: float, num_aug_samples: int
):
    generated_samples = []
    for _ in range(num_aug_samples):
        generated_sample = deepcopy(sample)
        shuffled_idx = _shuffle_within_segments(sample.tags, replace_prob)
        assert len(shuffled_idx) == len(sample.tags)
        for i in range(len(shuffled_idx)):
            generated_sample.toks[i] = sample.toks[shuffled_idx[i]]
            generated_sample.tags[i] = sample.tags[shuffled_idx[i]]
        generated_samples.append(generated_sample)
    return generated_samples
