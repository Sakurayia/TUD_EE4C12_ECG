import numpy as np
import torch
from torch.utils.data import DataLoader
from terminaltables import AsciiTable
from tqdm import tqdm
from sklearn import metrics

class Evaluate(object):
    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 max_epoch=-1,
                 criterion=None,
                 thresh=0.5,
                 class_name=[],
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')

        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')

        self.dataloader = dataloader
        self.interval = interval
        self.max_epoch = max_epoch
        self.start = start
        self.criterion = criterion
        self.class_name = class_name
        self.thresh = thresh
        self.eval_kwargs = eval_kwargs


    def before_every_train_epoch(self, model, epoch=-1, force_eval=False):
        if not force_eval and (epoch + 1) % self.interval != 0 and epoch + 1 != self.max_epoch:
            return None
        
        val_loss, val_acc, val_acc_per_class,\
        precision, recall, f_score, \
        precision_macro, recall_macro, f_score_macro, \
        precision_weighted, recall_weighted, f_score_weighted = self.every_epoch_evaluation(model)

        metric_class_table_data = [['Class', 'Accuracy', 'Precision', 'Recall', 'F-Score']]
        for cls_name, acc_c, precision_c, recall_c, f_score_c in zip(self.class_name, val_acc_per_class, precision, recall, f_score):
            metric_class_table_data.append([cls_name,
                                            '{:.4f}'.format(acc_c),
                                            '{:.4f}'.format(precision_c),
                                            '{:.4f}'.format(recall_c),
                                            '{:.4f}'.format(f_score_c)])
        metric_class_table_data.append(['accuracy',
                                            '{:.4f}'.format(val_acc),
                                            '\\',
                                            '\\',
                                            '\\'])
        metric_class_table_data.append(['macro avg',
                                            '\\',
                                            '{:.4f}'.format(precision_macro),
                                            '{:.4f}'.format(recall_macro),
                                            '{:.4f}'.format(f_score_macro)])
        metric_class_table_data.append(['weighted avg',
                                            '\\',
                                            '{:.4f}'.format(precision_weighted),
                                            '{:.4f}'.format(recall_weighted),
                                            '{:.4f}'.format(f_score_weighted)])
        metric_class_table = AsciiTable(metric_class_table_data)
        print('\n' + metric_class_table.table)

        eval_res = dict(
            val_loss=val_loss,
            val_acc=sum(val_acc_per_class) / len(val_acc_per_class),
            f_score=f_score,
            class_table=metric_class_table.table,
        )
        return eval_res


    def every_epoch_evaluation(self, model):
        model.eval()
        acc_list = []
        for i in range(len(self.class_name) + 1):
            acc_list.append([])

        loss_list = []
        logits_list = []
        label_list = []
        all_acc_list = []
        prog_bar = tqdm(self.dataloader, ncols=120)
        for i, data_batch in enumerate(prog_bar):
            torch.cuda.empty_cache()
            label_list.append(data_batch[1].numpy())
            data_batch[0] = data_batch[0].cuda(non_blocking=True)
            data_batch[1] = data_batch[1].cuda(non_blocking=True)

            with torch.no_grad():
                out = model(data_batch[0])  # [B, n_cls, 2]

            #pred = out.sigmoid()
            #logits_list.append(pred.cpu().numpy())

            pred_probab = out.softmax(dim=1)
            logits_list.append(pred_probab.cpu().numpy())
            y_pred = pred_probab.argmax(1)
            y_true = data_batch[1].argmax(1)

            #for b in range(data_batch[1].shape[0]):
            #    for c in range(data_batch[1].shape[1]):
            #        all_acc_list.append(int(pred[b, c]) == int(data_batch[1][b, c]))
            #        if data_batch[1][b, c] == 1:
            #            pos_acc_list.append(int(pred[b, c]))
            #            acc_list[c].append(int(pred[b, c]) == int(data_batch[1][b, c]))

            for b in range(y_true.shape[0]):
                all_acc_list.append(int(y_pred[b]) == int(y_true[b]))
                acc_list[int(y_true[b])].append(int(y_pred[b]) == int(y_true[b]))
                    


            val_loss = self.criterion(out, y_true)
            loss_list.append(float(val_loss))
            prog_bar.set_description('[val] loss:{:.4f} accuracy:{:.4f}'
                                        .format(np.array(loss_list).sum() / len(loss_list),
                                                np.array(all_acc_list).sum() / len(all_acc_list),
                                                )
                                    )
        precision, recall, f_score, _ = \
            metrics.precision_recall_fscore_support(np.concatenate(label_list, axis=0).argmax(axis=1), np.concatenate(logits_list, axis=0).argmax(axis=1), labels=self.class_name)
        precision_macro, recall_macro, f_score_macro, _ = \
            metrics.precision_recall_fscore_support(np.concatenate(label_list, axis=0).argmax(axis=1), np.concatenate(logits_list, axis=0).argmax(axis=1), labels=self.class_name, average='macro')
        precision_weighted, recall_weighted, f_score_weighted, _ = \
            metrics.precision_recall_fscore_support(np.concatenate(label_list, axis=0).argmax(axis=1), np.concatenate(logits_list, axis=0).argmax(axis=1), labels=self.class_name, average='weighted')
        acc = np.array(all_acc_list).sum() / len(all_acc_list)
        acc_per_class = [np.array(acc_per_cls).sum()/max(len(acc_per_cls), 1) for acc_per_cls in acc_list]

        return np.array(loss_list).sum()/len(loss_list), acc, acc_per_class, \
               precision, recall, f_score, \
               precision_macro, recall_macro, f_score_macro, \
               precision_weighted, recall_weighted, f_score_weighted


    def switch_data_loader(self, dataloader):
        self.dataloader = dataloader