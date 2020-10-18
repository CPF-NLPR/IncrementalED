import torch.utils.data as D
import numpy as np
import torch

class Exemplar:
    def __init__(self, max_size, total_cls):
        self.ids = []
        self.mask = []
        self.y = []
        self.cur_cls = 0
        self.max_size = max_size
        self.total_classes = total_cls

    def update(self, cls_num, data_x, data_mask, data_y, model, batch_size):
        self.cur_cls += cls_num
        total_store_num = int(self.max_size/self.cur_cls)
        for i in range(len(self.ids)):
            self.ids[i] = self.ids[i][:total_store_num]
            self.mask[i] = self.mask[i][:total_store_num]
            self.y[i] = self.y[i][:total_store_num]
        for i in range(1):
            self.ids.append([])
            self.mask.append([])
            self.y.append([])

        features = []
        dataset = D.TensorDataset(torch.LongTensor(data_x), torch.LongTensor(data_mask))
        dataloader = D.DataLoader(dataset, 2, False, num_workers=5)
        model.eval()
        with torch.no_grad():
            for X, mask in dataloader:
                X = X.cuda()
                mask = mask.cuda()
                _, pooled_output = model(X, mask)
                features.append(pooled_output[:, 0:1, :].view(pooled_output.size(0), pooled_output.size(2)))
        features = torch.cat(features, dim = 0)
        class_mean = torch.mean(features, dim = 0).view(1, features.size(1))
        class_mean.data = class_mean.data.expand(features.size(0), features.size(1))
        dist = torch.sum((class_mean - features) ** 2, dim=1)
        _, neighbor = torch.topk(dist.data, total_store_num, largest=False)
        for i in range(len(neighbor)):
            idx = max(data_y[neighbor[i]])
            if len(self.ids[idx-1]) < total_store_num:
                self.ids[idx-1].append(data_x[neighbor[i]])
                self.mask[idx-1].append(data_mask[neighbor[i]])
                self.y[idx-1].append(data_y[neighbor[i]])


    def get_exemplar_train(self):
        exemplar_ids = []
        exemplar_mask = []
        exemplar_y = []
        for i in range(len(self.ids)):
            exemplar_ids.extend(self.ids[i])
            exemplar_mask.extend(self.mask[i])
            exemplar_y.extend(self.y[i])
        return exemplar_ids, exemplar_mask, exemplar_y

    def get_cur_cls(self):
        return self.cur_cls

