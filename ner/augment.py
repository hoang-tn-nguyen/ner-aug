from __future__ import annotations
from data import Sample
import numpy as np
import random
import nlpaug.augmenter.word as naw

def generate_samples_by_entities(sample: Sample, class_entities: dict, replace_prob: float, num_aug_samples: int) -> list[Sample]:
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
                    generated_sample = generated_sample.replace_span(s_id, e_id, replace_toks, category)
                    class_spans = generated_sample.get_spans()
        generated_samples.append(generated_sample)
    return generated_samples