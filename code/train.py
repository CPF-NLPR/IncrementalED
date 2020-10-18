import torch
import torch.utils.data as D
import torch.nn.functional as F
from data_loader_kbp import Load
from config import FLAGS
from model_bert import BertED
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from exemplar import Exemplar
from copy import deepcopy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class Trainer:
    def __init__(self, total_cls):
        self.total_cls = total_cls
        self.seen_cls = 0
        self.model = BertED(total_cls)
        if FLAGS.gpu:
            self.model = self.model.cuda()
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=FLAGS.lr, eps=FLAGS.adam_epsilon)

    def train(self, batch_size, epoches, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        distill_criterion = nn.CosineEmbeddingLoss()
        exemplar = Exemplar(max_size, total_cls)
        self.previous_model = None

        load = Load()
        label2id, id2label = load.label2id, load.id2label
        print('The number of labels:', len(label2id))
        print('load data……')
        data_x_train, data_mask_train, data_y_train = load.load_data_bert(FLAGS.data_path)
        data_x_test, data_mask_test, data_y_test = load.load_data_bert(FLAGS.data_path.replace('train', 'test'))
        print('load successfully!')
        test_xs = []
        test_masks = []
        test_ys = []
        test_fs = []
        for inc_i in range(FLAGS.task_num):
            print("Incremental num : ", inc_i)
            test_xs.extend(data_x_test[inc_i])
            test_masks.extend(data_mask_test[inc_i])
            test_ys.extend(data_y_test[inc_i])

            train_xs, train_masks, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(data_x_train[inc_i])
            train_masks.extend(data_mask_train[inc_i])
            train_ys.extend(data_y_train[inc_i])

            train_xss = torch.LongTensor(train_xs)
            train_maskss = torch.LongTensor(train_masks)
            train_yss = torch.LongTensor(train_ys)
            test_xss = torch.LongTensor(test_xs)
            test_maskss = torch.LongTensor(test_masks)
            test_yss = torch.LongTensor(test_ys)

            train_dataset = D.TensorDataset(train_xss, train_maskss, train_yss)
            train_dataloader = D.DataLoader(train_dataset, batch_size, True, num_workers=5)
            test_dataset = D.TensorDataset(test_xss, test_maskss, test_yss)
            test_dataloader = D.DataLoader(test_dataset, batch_size, False, num_workers=5)

            t_total = int(len(train_dataloader.dataset) / FLAGS.batch_size / FLAGS.gradient_accumulation_steps * FLAGS.n_epochs)
            #self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(t_total * FLAGS.warmup_proportion), num_training_steps=t_total)
            self.seen_cls = exemplar.get_cur_cls()+1
            print("seen cls number : ", self.seen_cls)
            test_f = []
            for epoch in range(epoches):
                print("Epoch", epoch)
                self.model.train()
                if inc_i > 0:
                    self.stage1_distill(train_dataloader, criterion, distill_criterion)
                else:
                    self.stage1(train_dataloader, criterion)
                p, r, f = self.test(test_dataloader)
                test_f.append(f)
            exemplar.update(1, data_x_train[inc_i], data_mask_train[inc_i], data_y_train[inc_i], self.model, batch_size)
            self.previous_model = deepcopy(self.model)
            p, r, f = self.test(test_dataloader)
            test_f.append(f)
            test_fs.append(max(test_f))
            print(test_fs)


    def stage1(self, train_dataloader, criterion):
        total_loss = 0
        for train_X, train_mask, train_Y in train_dataloader:
            if FLAGS.gpu:
                train_X = train_X.cuda()
                train_mask = train_mask.cuda()
                train_Y = train_Y.cuda()
            logits, _ = self.model(train_X, train_mask)
            p = logits[:, :, :self.seen_cls+1]
            loss = criterion(p.view(-1, self.seen_cls+1), train_Y.view(-1))
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            total_loss += loss.item()
        print("Loss: {:.6f}".format(total_loss / len(train_dataloader)))


    def stage1_distill(self, train_dataloader, criterion, distill_criterion):
        total_loss = 0
        T = 2
        alpha = (self.seen_cls - 1) / self.seen_cls
        #print("classification proportion 1-alpha = ", 1 - alpha)
        for train_X, train_mask, train_Y in train_dataloader:
            if FLAGS.gpu:
                train_X = train_X.cuda()
                train_mask = train_mask.cuda()
                train_Y = train_Y.cuda()
            logits, output = self.model(train_X, train_mask)
            normalized_output = F.normalize(output.view(-1, output.size()[2]), p=2, dim=1)
            p = logits[:, :, :self.seen_cls+1]
            with torch.no_grad():
                pre_p, pre_output = self.previous_model(train_X, train_mask)
                normalized_pre_out = F.normalize(pre_output.view(-1, pre_output.size()[2]), p=2, dim=1)
                pre_p = F.softmax(pre_p[:, :, :self.seen_cls] / T, dim=2)
                pre_p = pre_p.view(-1, self.seen_cls)
            logp = F.log_softmax(p[:, :, :self.seen_cls] / T, dim=2).view(-1, self.seen_cls)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = criterion(p.view(-1, self.seen_cls+1), train_Y.view(-1))
            distill_loss = distill_criterion(normalized_output, normalized_pre_out, torch.ones(train_X.size(0)*train_X.size(1)).cuda())
            loss = loss_soft_target * 0.8 + (1 - alpha) * loss_hard_target + 0.7*distill_loss
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            total_loss += loss.item()
        print("Loss: {:.6f}".format(total_loss / len(train_dataloader)))


    def test(self, test_dataloader):
        self.model.eval()
        with torch.no_grad():
            golds = list()
            predicteds = list()
            for test_X, test_mask, test_Y in test_dataloader:
                if FLAGS.gpu:
                    test_X = test_X.cuda()
                    test_mask = test_mask.cuda()
                    test_Y = test_Y.cuda()
                logits, _ = self.model(test_X, test_mask)
                logits = logits[:, :, :self.seen_cls+1]
                predicted_y = torch.argmax(logits, -1).view(-1)
                predicteds.extend(list(predicted_y.cpu().numpy()))
                golds.extend(list(test_Y.view(-1).cpu().numpy()))
            c_predict = 0
            c_correct = 0
            c_gold = 0
            for g, p in zip(golds, predicteds):
                if g != 0:
                    c_gold += 1
                if p != 0:
                    c_predict += 1
                if g != 0 and p != 0 and p == g:
                    c_correct += 1
            p = c_correct / (c_predict + 1e-100)
            r = c_correct / c_gold
            f = 2 * p * r / (p + r + 1e-100)
            print('correct', c_correct, 'predicted', c_predict, 'golden', c_gold)
            return p, r, f


if __name__ == "__main__":
    trainer = Trainer(FLAGS.total_cls)
    trainer.train(FLAGS.batch_size, FLAGS.n_epochs, FLAGS.max_size)


