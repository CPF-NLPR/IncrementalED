from config import FLAGS
from transformers import BertTokenizer

MAX_LENGTH = 90

class Load:
    def __init__(self):
        self.id2label, self.label2id = self.load_full_labels()
        self.max_length = MAX_LENGTH

    def load_full_labels(self):
        id2label = {}
        label2id = {}
        labels = ['None', 'correspondence', 'endposition', 'meet',
                  'arrestjail', 'die', 'contact', 'broadcast',
                  'transfermoney', 'transportperson', 'attack']
        for label in labels:
            if label not in label2id:
                label2id[label] = len(label2id)
                id2label[len(id2label)] = label
        return id2label, label2id

    def load_data_bert(self, filename):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        data_x = [[], [], [], [], [], [], [], [], [], []]
        data_mask = [[], [], [], [], [], [], [], [], [], []]
        data_y = [[], [], [], [], [], [], [], [], [], []]
        with open(filename, 'r') as infile:
            while True:
                row = infile.readline().replace('\n', '').replace('\r', '')
                if row == '':
                    break
                row = row.split('\t')
                sentence = row[0].split(' ')
                sentence = ['[CLS]'] + sentence
                labels = row[1:]
                for i in range(0, len(labels), 2):
                    label = labels[i + 1]
                    if label in self.label2id:
                        idx = self.label2id[label]
                        input_ids = tokenizer.convert_tokens_to_ids(sentence)
                        input_mask = [1] * len(input_ids)

                        position = int(labels[i])
                        labels_idx = []
                        for i in range(len(input_ids)):
                            labels_idx.append(0)
                        labels_idx[position+1] = idx
                        if len(input_ids) > self.max_length + 1:
                            input_ids = input_ids[:self.max_length + 1]
                            input_mask = input_mask[:self.max_length + 1]
                            labels_idx = labels_idx[:self.max_length + 1]
                        input_ids.extend(tokenizer.convert_tokens_to_ids(['[SEP]']))
                        input_mask.extend([1])
                        labels_idx.extend([0])
                        if len(input_ids) < self.max_length + 2:
                            input_ids.extend([0] * (self.max_length + 2 - len(input_ids)))
                            input_mask.extend([0] * (self.max_length + 2 - len(input_mask)))
                            labels_idx.extend([0] * (self.max_length + 2 - len(labels_idx)))
                        data_x[idx-1].append(input_ids)
                        data_mask[idx-1].append(input_mask)
                        data_y[idx-1].append(labels_idx)
        return data_x, data_mask, data_y


