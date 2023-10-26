import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from sklearn import metrics
from terminaltables import AsciiTable
from model import NeuralNetwork
from dataset import ECGDataset
from loss import ASLMarginLossSmooth, MultiFocalLoss
from evaluate import Evaluate
from checkpoint import Checkpoint

import json

class_name = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14
]

f = open('path.json')
train_path, val_path, test_path = json.load(f).values()
f.close()

class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()
        # --- config ---
        self.to_gpu = True if torch.cuda.is_available() else False
        self.class_name = class_name
        self.max_epoch = 100
        # --- load model ---
        self.model = NeuralNetwork()
        if self.to_gpu:
            self.model = self.model.cuda()

        # --- setup optimizer ---
        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        #optim = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)
        #self.optimizer = optim(self.model)
        self.optimizer = optim
        # --- setup dataset(train + test) ---
        # --- dataset config ---
        transform = Lambda(lambda x: torch.from_numpy(x).to(torch.float32))
        target_transform = Lambda(lambda y: torch.zeros(
            15, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        self.batch_size = 256

        num_workers = 8 if not torch.cuda.is_available() else torch.cuda.device_count() * 4
        train_dataset = ECGDataset(train_path,
                                   transform=transform,
                                   target_transform=target_transform,
                                   class_name=self.class_name)
        val_dataset = ECGDataset(val_path,
                                 transform=transform,
                                 target_transform=target_transform,
                                 class_name=self.class_name)
        test_dataset = ECGDataset(test_path,
                                  transform=transform,
                                  target_transform=target_transform,
                                  class_name=self.class_name)
        self.train_dataloader = DataLoader(dataset=train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           #num_workers=num_workers,
                                           pin_memory=True,
                                           drop_last=True)
        self.val_dataloader = DataLoader(dataset=val_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         #num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=False)
        self.test_dataloader = DataLoader(dataset=test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          #num_workers=num_workers,
                                          pin_memory=True,
                                          drop_last=False)

        train_dataset.compute_class_num()
        val_dataset.compute_class_num()
        test_dataset.compute_class_num()

        # --- setup loss function ---
        self.criterion = nn.CrossEntropyLoss()
        #weight = torch.tensor([1, 10, 10, 10, 100, 40, 20, 50, 60, 150, 60, 30, 10, 30, 100])
        #if self.to_gpu:
        #    weight = weight.cuda()
        #self.criterion = MultiFocalLoss(num_class=15, alpha=weight, gamma=2.0, reduction='mean')
        #self.criterion = ASLMarginLossSmooth(class_weights=self.train_dataloader.dataset.compute_class_weights())

        # --- setup evaluator ---
        self.evaluator = Evaluate(
            dataloader=self.val_dataloader,
            interval=2,
            max_epoch=100,
            criterion=self.criterion,
            class_name=self.class_name,
        )

        # --- setup GradScaler ---
        self.amp_grad_scaler = None
        if torch.cuda.is_available():
            self.amp_grad_scaler = GradScaler()
        # --- setup checkpoint ---
        self.checkpoint = Checkpoint(interval=2,
                                     save_optimizer=True,
                                     out_dir='model_save_2',
                                     save_last=True,
                                     max_epoch=100,
                                     avg_step=20)
        # --- load last trained model ---
        self.start_epoch = self.checkpoint.resume(False,
                                                  self.model,
                                                  self.optimizer,
                                                  self.amp_grad_scaler,
                                                  'last_cp.pth')

    def train_network(self):
        # --- 开始训练 ---
        torch.autograd.set_detect_anomaly(True)  # activate anomaly detection
        cur_iter = 0
        for epoch in range(self.start_epoch + 1, self.max_epoch + 1):
            # self.evaluator.before_every_train_epoch(self.model, 1)
            # train
            loss_list, acc_list = self.train_one_epoch(epoch)
            train_loss = np.array(loss_list).sum() / len(loss_list)
            train_acc = np.array(acc_list).sum() / len(acc_list),
            if np.isnan(train_loss):
                # load last epoch model
                self.start_epoch = self.checkpoint.resume_last_checkpoint(self.model,
                                                                          self.optimizer,
                                                                          self.amp_grad_scaler,
                                                                          max(1,
                                                                              epoch - 5
                                                                              ))
                # retrain net_work
                self.train_network()
                # then return directly and end training process
                return
            # eval
            eval_res = self.evaluator.before_every_train_epoch(
                self.model, epoch - 1)
            if eval_res is not None:
                # save
                save_meta = dict(
                    epoch=epoch - 1,
                    iter=cur_iter,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=eval_res['val_loss'],
                    val_acc=eval_res['val_acc'],
                    f_score=eval_res['f_score'],
                    class_table=eval_res['class_table'],
                    step_loss=loss_list,
                )
                if self.amp_grad_scaler is not None:
                    save_meta['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()
                continue_training = self.checkpoint.after_train_epoch(
                    self.model, self.optimizer, save_meta)
            else:
                continue_training = True

            #self.criterion.reset_class_weights(self.train_dataloader.dataset.compute_class_weights())

            if not continue_training:
                print(
                    f'Model Training Procedure Has Early Stopping At Epoch {epoch}')
                break

        # 最后做一次测试集的测试
        # 读取最好的模型
        best_model_epoch = self.checkpoint.resume_best_model(self.model)
        # 测试
        self.evaluator.switch_data_loader(self.test_dataloader)
        eval_res = self.evaluator.before_every_train_epoch(
            self.model, force_eval=True)
        # 保存结果
        # save
        save_meta = dict(
            epoch=best_model_epoch - 1,
            val_acc=eval_res['val_acc'],
            val_loss=eval_res['val_loss'],
            class_table=eval_res['class_table'],
        )
        self.checkpoint.final_test(save_meta)

    def train_one_epoch(self, epoch):
        bar_dataset = tqdm(self.train_dataloader, ncols=120)
        loss_list = []
        acc_list = []
        pos_acc_list = []
        for i, data_batch in enumerate(bar_dataset):
            cur_iter = (epoch - 1) * len(self.train_dataloader) + i
            torch.cuda.empty_cache()
            self.model.train()
            if self.to_gpu:
                data_batch[0] = data_batch[0].cuda(non_blocking=True) # data
                data_batch[1] = data_batch[1].cuda(non_blocking=True) # label
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                # forward
                out = self.model(data_batch[0])
                _, labels = data_batch[1].max(dim=1)
                loss = self.criterion(out, labels) 
                #loss = self.criterion(out, data_batch[1]) 

            self.amp_grad_scaler.scale(loss).backward()
            self.amp_grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.amp_grad_scaler.step(self.optimizer)
            self.amp_grad_scaler.update()

            loss_list.append(float(loss))
            pred_probab = out.softmax(dim=1)
            y_pred = pred_probab.argmax(1)
            y_true = data_batch[1].argmax(1)

            #for b in range(data_batch[1].shape[0]):
            #    for c in range(data_batch[1].shape[1]):
            #        acc_list.append(int(pred[b, c]) == int(data_batch[1][b, c]))
            #        if data_batch[1][b, c] == 1:
            #            pos_acc_list.append(int(pred[b, c]))

            for b in range(y_true.shape[0]):
                acc_list.append(int(y_pred[b]) == int(y_true[b]))

            bar_dataset.set_description('epoch {:d}/{:d} lr: {:.8f} loss:{:.4f} accuracy:{:.4f}'
                                        .format(epoch,
                                                self.max_epoch,
                                                self.optimizer.param_groups[0]['lr'],
                                                np.array(loss_list).sum(
                                                ) / len(loss_list),
                                                np.array(acc_list).sum(
                                                ) / len(acc_list),
                                                )
                                        )

            # 检测到nan时退出
            if np.isnan(np.array(loss_list).sum() / len(loss_list)):
                return loss_list, -1

        return loss_list, acc_list
    
    def metrics(self, confusion_matrix_=False):    
        self.checkpoint.resume_best_model(self.model)
        self.model.eval()

        pred_list = []
        label_list = []
        all_acc_list = []
        acc_list = []
        for i in range(len(self.class_name) + 1):
            acc_list.append([])

        prog_bar = tqdm(self.test_dataloader, ncols=120)
        for i, data_batch in enumerate(prog_bar):
            torch.cuda.empty_cache()
            label_list.append(data_batch[1].numpy())
            data_batch[0] = data_batch[0].cuda(non_blocking=True)
            data_batch[1] = data_batch[1].cuda(non_blocking=True)

            with torch.no_grad():
                out = self.model(data_batch[0])  # [B, n_cls, 2]

            pred_probab = out.softmax(dim=1)
            pred_list.append(pred_probab.cpu().numpy())
            y_pred_batch = pred_probab.argmax(1)
            y_true_batch = data_batch[1].argmax(1)

            for b in range(y_true_batch.shape[0]):
                all_acc_list.append(int(y_pred_batch[b]) == int(y_true_batch[b]))
                acc_list[int(y_true_batch[b])].append(int(y_pred_batch[b]) == int(y_true_batch[b]))

        y_true = np.concatenate(label_list, axis=0).argmax(axis=1)
        y_pred = np.concatenate(pred_list, axis=0).argmax(axis=1)

        precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_true, y_pred, labels=self.class_name)
        precision_macro, recall_macro, f_score_macro, _ = \
            metrics.precision_recall_fscore_support(y_true, y_pred, labels=self.class_name, average='macro')
        precision_weighted, recall_weighted, f_score_weighted, _ = \
            metrics.precision_recall_fscore_support(y_true, y_pred, labels=self.class_name, average='weighted')
        
        acc = np.array(all_acc_list).sum() / len(all_acc_list)
        acc_per_class = [np.array(acc_per_cls).sum()/max(len(acc_per_cls), 1) for acc_per_cls in acc_list]

        metric_class_table_data = [['Class', 'Count', 'Accuracy', 'Precision', 'Recall', 'F-Score']]
        for cls_name, acc_c, precision_c, recall_c, f_score_c, support_c in zip(self.class_name, acc_per_class, precision, recall, f_score, support):
            metric_class_table_data.append([cls_name,
                                            '{:d}'.format(support_c),
                                            '{:.4f}'.format(acc_c),
                                            '{:.4f}'.format(precision_c),
                                            '{:.4f}'.format(recall_c),
                                            '{:.4f}'.format(f_score_c)])
        metric_class_table_data.append(['accuracy',
                                            '{:d}'.format(np.array(support).sum()),
                                            '{:.4f}'.format(acc),
                                            '\\',
                                            '\\',
                                            '\\'])
        metric_class_table_data.append(['macro avg',
                                            '{:d}'.format(np.array(support).sum()),
                                            '\\',
                                            '{:.4f}'.format(precision_macro),
                                            '{:.4f}'.format(recall_macro),
                                            '{:.4f}'.format(f_score_macro)])
        metric_class_table_data.append(['weighted avg',
                                            '{:d}'.format(np.array(support).sum()),
                                            '\\',
                                            '{:.4f}'.format(precision_weighted),
                                            '{:.4f}'.format(recall_weighted),
                                            '{:.4f}'.format(f_score_weighted)])
        metric_class_table = AsciiTable(metric_class_table_data)
        print('\n' + metric_class_table.table)
        
        if confusion_matrix_:
            _, ax = plt.subplots(figsize=(30, 24))
            cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.class_name)
            cm_display = metrics.ConfusionMatrixDisplay(cm)
            cm_display.plot(ax=ax)
            plt.show()
