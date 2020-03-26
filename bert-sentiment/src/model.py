import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        # self.bert_drop = nn.Dropout(0.3)
        # self.out = nn.Linear(768, 1)
        self.bert_drop = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(768, 768)
        self.out = nn.Linear(768, 3)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        # bo = self.tanh(self.fc(bo)) # to be commented if original
        output = self.out(bo)
        return output
