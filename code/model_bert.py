import torch.nn as nn
from transformers import BertModel

class BertED(nn.Module):
    def __init__(self, y_num):
        super(BertED, self).__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, self.y_num)

    def forward(self, data_x, mask_x):
        outputs = self.bert(data_x, attention_mask = mask_x)
        bert_enc = outputs[0]
        logits = self.fc(bert_enc)
        return logits, outputs[0]