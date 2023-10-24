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

        self.init_metrics_recorder()


    def init_metrics_recorder(self):
        self.hamming_loss_list = []
        self.coverage_loss_list = []
        self.one_error_loss_list = []
        self.ranking_loss_list = []




    def before_every_train_epoch(self, model, epoch=-1, force_eval=False):
        if not force_eval and (epoch + 1) % self.interval != 0 and epoch + 1 != self.max_epoch:
            return None
        # if (epoch + 1) % self.interval != 0 and epoch + 1 != self.max_epoch:
        #     return None
        val_loss, val_acc_per_class, val_pred_matrix,\
        hamming_loss, coverage_loss, ranking_loss, one_error_loss,\
        precision, recall, f_score = self.every_epoch_evaluation(model)

        # class_table_data = [['Class', 'Accuracy']]
        #
        # for cls_name, acc in zip(self.class_name, val_acc_per_class):
        #     class_table_data.append([cls_name, '{:.2f}%'.format(acc * 100)])
        #
        # table = AsciiTable(class_table_data)
        # print('\n' + table.table)

        # class_matrix_table_data = [['gt\pred', *self.class_name]]
        # for i, cls_name in enumerate(self.class_name):
        #     row = [cls_name]
        #     for j in range(val_pred_matrix.shape[1]):
        #         row.append(int(val_pred_matrix[i, j]))
        #     class_matrix_table_data.append(row)
        #
        # matrix_table = AsciiTable(class_matrix_table_data)
        # print('\n' + matrix_table.table)


        metric_class_table_data = [['Class', 'Precision', 'Recall', 'F-Score']]
        for cls_name, precision_c, recall_c, f_score_c in zip(self.class_name, precision, recall, f_score):
            metric_class_table_data.append([cls_name,
                                            '{:.2f}%'.format(precision_c * 100),
                                            '{:.2f}%'.format(recall_c * 100),
                                            '{:.2f}%'.format(f_score_c * 100)])
        metric_class_table = AsciiTable(metric_class_table_data)
        print('\n' + metric_class_table.table)


        metric_table_data = [['Hamming Loss', 'Coverage Loss', 'Ranking Loss', 'One-Error Loss'],
                            ['{:.5f}'.format(hamming_loss), '{:.5f}'.format(coverage_loss),
                             '{:.5f}'.format(ranking_loss), '{:.5f}'.format(one_error_loss)]]
        metric_table = AsciiTable(metric_table_data)
        print('\n' + metric_table.table)


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

        # 计算每一类被预测成什么
        pred_matrix = np.zeros((len(self.class_name), len(self.class_name))).astype(np.int32)

        loss_list = []
        logits_list = []
        label_list = []
        all_acc_list = []
        pos_acc_list = []
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
            pred = torch.zeros_like(pred_probab)

            for b in range(data_batch[1].shape[0]):
                pred[b][y_pred[b]] = 1
                for c in range(data_batch[1].shape[1]):
                    all_acc_list.append(int(pred[b, c]) == int(data_batch[1][b, c]))
                    if data_batch[1][b, c] == 1:
                        pos_acc_list.append(int(pred[b, c]))

                    if data_batch[1][b, c] == 1:
                        # 只看正类
                        acc_list[c].append(int(pred[b, c]) == int(data_batch[1][b, c]))

                        # 第b张图像,
                        # data_batch['label'][b, c]=1则说明该图像属于第c类
                        # 模型把它预测成第c类的1概率为pred[b, c]
                        for m in range(data_batch[1].shape[1]):
                            pred_matrix[c, m] += 1 if pred[b, m] == 1 else 0




            val_loss = self.criterion(out, data_batch[1])
            loss_list.append(float(val_loss))
            prog_bar.set_description('[val] loss:{:.4f} acc[pos|all]:{:.2f}|{:.2f}'
                                        .format(np.array(loss_list).sum() / len(loss_list),
                                                np.array(pos_acc_list).sum() / len(pos_acc_list) * 100,
                                                np.array(all_acc_list).sum() / len(all_acc_list) * 100,
                                                )
                                    )
        hamming_loss, coverage_loss, ranking_loss, one_error_loss, precision, recall, f_score = \
            self.compute_and_add_to_metrics(np.concatenate(logits_list, axis=0), np.concatenate(label_list, axis=0))
        acc_per_class = [np.array(acc_per_cls).sum()/max(len(acc_per_cls), 1) for acc_per_cls in acc_list]

        return np.array(loss_list).sum()/len(loss_list), acc_per_class, pred_matrix, \
               hamming_loss, coverage_loss, ranking_loss, one_error_loss, precision, recall, f_score


    def compute_and_add_to_metrics(self, logits, target):
        # [B, C]
        pred = (logits > self.thresh)
        pred = pred.astype(np.int32)

        pos_sample = target.sum(axis=1) > 0
        hamming_loss = metrics.hamming_loss(target, pred) # 负样本参与计算
        coverage_loss = metrics.coverage_error(target[pos_sample], logits[pos_sample])
        ranking_loss = metrics.label_ranking_loss(target[pos_sample], logits[pos_sample])

        one_error_loss_list = []
        for b in range(logits.shape[0]):
            if target[b].sum() > 0:
                one_error_loss_list.append(int(target[b, logits[b].argmax()] != 1))
        one_error_loss = sum(one_error_loss_list) / max(len(one_error_loss_list), 1)

        precision, recall, f_score, _ = metrics.precision_recall_fscore_support(target[pos_sample], pred[pos_sample])

        return hamming_loss, coverage_loss, ranking_loss, one_error_loss, precision, recall, f_score


    def switch_data_loader(self, dataloader):
        self.dataloader = dataloader