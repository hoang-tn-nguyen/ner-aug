import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer


class WordEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.model = AutoModel.from_pretrained(pretrained_model, config=self.config)

    def forward(self, input, atn_mask=None):
        words = input
        outputs = self.model(
            words, attention_mask=atn_mask, output_hidden_states=True, return_dict=True,
        )
        return outputs


class ClsModel(nn.Module):
    def __init__(self, num_classes=2, pretrained_model="bert-base-uncased", max_length=100, embed_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.encoder = WordEncoder(pretrained_model)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.max_length = max_length

    def forward(self, input_list):
        input, txt_mask, atn_mask = self.tokenize(input_list)
        output = self.encoder(input, atn_mask).pooler_output 
        output = self.fc(output)
        return output

    def tokenize(self, input_list):
        inputs = []
        txt_masks = []
        atn_masks = []
        for input in input_list:
            toks = self.tokenizer.tokenize(input)
            if len(toks) > self.max_length - 2:
                toks = toks[-self.max_length+2:]
            cls_id, sep_id, pad_id = self.tokenizer.convert_tokens_to_ids(
                ["[CLS]", "[SEP]", "[PAD]"]
            )
            toks = self.tokenizer.convert_tokens_to_ids(toks)
            toks = [cls_id] + toks + [sep_id]
            pad_toks = np.ones(self.max_length, dtype=int) * pad_id
            pad_toks[: len(toks)] = toks
            inputs.append(torch.tensor(pad_toks))
            
            txt_mask = np.zeros(self.max_length, dtype=int)
            atn_mask = np.zeros(self.max_length, dtype=int)
            txt_mask[1 : len(toks) - 1] = 1
            atn_mask[: len(toks)] = 1
            txt_masks.append(torch.tensor(txt_mask))
            atn_masks.append(torch.tensor(atn_mask))

        inputs = torch.stack(inputs)
        txt_masks = torch.stack(txt_masks)
        atn_masks = torch.stack(atn_masks)
        if torch.cuda.is_available():
            return inputs.to('cuda'), txt_masks.to('cuda'), atn_masks.to('cuda')
        else:
            return inputs, txt_masks, atn_masks