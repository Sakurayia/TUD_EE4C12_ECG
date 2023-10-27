import os
import torch
from torch.optim import Optimizer
import matplotlib
import matplotlib.pyplot as plt
import sys
import time
import numpy as np

def save_checkpoint(model, filename, optimizer, meta):
    torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'meta': meta
            }, filename)

def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])

    return checkpoint

class Checkpoint(object):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        save_last (bool): Whether to force the last checkpoint to be saved
            regardless of interval.
        sync_buffer (bool): Whether to synchronize buffers in different
            gpus. Default: False.
    """

    def __init__(self,
                 interval=-1,
                 save_optimizer=True,
                 out_dir=None,
                 save_last=True,
                 max_epoch=-1,
                 max_save=10,
                 avg_step=5,
                 early_stopping=15,
                 save_best_model=True,
                 model_judge_metric='f_score',
                 model_judge_class='mean',
                 **kwargs):
        self.interval = interval
        self.max_epoch = max_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.save_last = save_last
        self.avg_step = avg_step
        self.early_stopping = early_stopping
        self.save_best_model = save_best_model
        self.model_judge_metric = model_judge_metric
        assert model_judge_metric in ['train_loss', 'train_acc', 'val_loss', 'f_score', 'val_acc']
        self.model_judge_class = model_judge_class
        self.args = kwargs

        self.basic_key = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'step_loss', 'f_score']
        if self.early_stopping or self.save_best_model:
            self.model_best_performance = 0.0
            if self.early_stopping:
                self.model_best_epoch = 0
        self.init_meta_recorder()

        self.max_save = max_save
        self.log_txt = os.path.join(out_dir, self.get_log_name())
        try:
            os.mkdir(self.out_dir)
        except:
            pass

    def init_meta_recorder(self):
        self.meta = dict()
        for k in self.basic_key:
            self.meta[k] = []
        self.meta['amp_grad_scaler'] = {}

    def get_log_name(self):
        local_time = time.localtime(time.time())
        return "training_log_" + \
            str(local_time.tm_year) + "_" + \
            str(local_time.tm_mon)  + "_" + \
            str(local_time.tm_mday) + "_" + \
            str(local_time.tm_hour) + "_" + \
            str(local_time.tm_min)  + "_" + \
            str(local_time.tm_sec)  + ".txt"
    def get_time_string(self):
        local_time = time.localtime(time.time())
        return str(local_time.tm_year) + "-" + \
               str(local_time.tm_mon)  + "-" + \
               str(local_time.tm_mday) + " " + \
               str.zfill(str(local_time.tm_hour), 2) + ":" + \
               str.zfill(str(local_time.tm_min), 2)  + ":" + \
               str.zfill(str(local_time.tm_sec), 2)
    def after_train_epoch(self, model, optimizer, meta):
        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(meta['epoch'], self.interval) or (self.save_last and self.is_last_epoch(meta['epoch'])):
            # update meta data
            for k in self.basic_key:
                if isinstance(meta[k], list): # loss_list, acc_list
                    self.meta[k].extend(meta[k])
                else:
                    self.meta[k].append(meta[k])


            if meta.get('amp_grad_scaler') is not None:
                # self.meta['amp_grad_scaler'].append(meta['amp_grad_scaler'])
                self.meta['amp_grad_scaler'] = meta['amp_grad_scaler']

            self.meta['best_epoch'] = self.model_best_epoch
            self.meta['best_performance'] = self.model_best_performance

            filename = os.path.join(self.out_dir, 'epoch_{}.pth'.format(meta['epoch'] + 1))
            save_checkpoint(model, filename, optimizer, self.meta)
            filename = os.path.join(self.out_dir, 'last_cp.pth')
            save_checkpoint(model, filename, optimizer, self.meta)

            save_file = [fname for fname in os.listdir(self.out_dir) if
                         fname.startswith("epoch_") and fname.endswith(".pth")]
            if len(save_file) > self.max_save:
                # eg. max_save=5, interval=2, epoch = 10 save epoch_11.pth
                # -> (have 1.pth,3.pth,5.pth,7.pth,9.pth,11.pth saved)remove epoch_1.pth
                need_to_rm = os.path.join(self.out_dir, 'epoch_{}.pth'.format(meta['epoch'] - self.max_save * self.interval + 1))
                if os.path.isfile(need_to_rm):
                    os.remove(need_to_rm)

            self.plot_progress()
            self.write_log(meta)


            if self.early_stopping or self.save_best_model:
                model_performance = meta[self.model_judge_metric]
                if self.model_judge_metric == 'f_score':
                    if self.model_judge_class == 'mean':
                        model_performance = model_performance.mean()
                    else:
                        model_performance = model_performance[self.model_judge_class]
                if self.model_best_performance < model_performance:
                    # update best record
                    self.model_best_performance = model_performance
                    self.model_best_epoch = meta['epoch']
                    # save best model
                    if self.save_best_model:
                        filename = os.path.join(self.out_dir, 'best_model.pth')
                        save_checkpoint(model, filename, optimizer, self.meta)

                # early stop
                return meta['epoch'] - self.model_best_epoch < self.early_stopping
        # continue training
        return True


    def final_test(self, meta):
        # test on test dataset, save meta to txt
        self.write_log(meta, is_final_test=True)


    def every_n_epochs(self, epoch, n):
        return (epoch + 1) % n == 0 if n > 0 else False

    def is_last_epoch(self, epoch):
        return epoch + 1 == self.max_epoch

    def resume(self, continue_train, model, optimizer, amp_grad_scaler=None, continue_file=None):
        filename = os.path.join(self.out_dir, 'last_cp.pth')
        if continue_file is not None:
            if os.path.isfile(os.path.join(self.out_dir, continue_file)):
                filename = os.path.join(self.out_dir, continue_file)
        if continue_train and os.path.isfile(filename):
            checkpoint = load_checkpoint(
                model,
                filename,
                )

            if isinstance(optimizer, Optimizer):
                optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(optimizer, dict):
                for k in optimizer.keys():
                    optimizer[k].load_state_dict(checkpoint['optimizer'][k])

            self.meta = dict()
            for k in self.basic_key:
                self.meta[k] = checkpoint['meta'][k]
            if checkpoint['meta'].get('amp_grad_scaler') is not None and amp_grad_scaler is not None:
                amp_grad_scaler.load_state_dict(checkpoint['meta']['amp_grad_scaler'])

            self.model_best_epoch = checkpoint['meta'].get('best_epoch', 0)
            self.model_best_performance = checkpoint['meta'].get('best_performance', 0.0)

            print(f"--- continue training with epoch {checkpoint['meta']['epoch'][-1] + 1} ---")
            print(f"--- current best epoch is {self.model_best_epoch + 1},"
                  f" best performance is [{self.model_judge_metric}]{self.model_best_performance}")
            return self.meta['epoch'][-1] + 1
        self.init_meta_recorder()

        print("--- start a new training procedure ---")
        return 0

    def resume_last_checkpoint(self, model, optimizer, amp_grad_scaler=None, roll_back_epoch=-1):
        if roll_back_epoch == -1:
            return self.resume(True, model, optimizer, amp_grad_scaler, f'last_cp.pth')
        return self.resume(True, model, optimizer, amp_grad_scaler, f'epoch_{roll_back_epoch}.pth')

    def resume_best_model(self, model):
        if self.save_best_model:
            filename = os.path.join(self.out_dir, 'best_model.pth')
        else:
            filename = os.path.join(self.out_dir, 'last_cp.pth')
        checkpoint = load_checkpoint(model, filename)

        print(f"--- resume best model at epoch {checkpoint['meta']['epoch'][-1] + 1} ---")
        return checkpoint['meta']['epoch'][-1] + 1


    def plot_progress(self):
        font = {'weight': 'normal',
                'size': 18}

        matplotlib.rc('font', **font)

        avg_loss = self.average_list(self.meta['step_loss'], self.avg_step)
        step = [i*self.avg_step for i in range(len(avg_loss))]

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        ax.plot(step, avg_loss, color='b', ls='-', label="loss")
        plt.title("Progress Loss vs Iter", fontsize=18)
        ax.set_xlabel("iter", fontsize=16)
        ax.set_ylabel("loss", fontsize=16)
        plt.margins(0.1)
        ax.xaxis.set_ticks_position('none')
        ax.legend()
        fig.savefig(os.path.join(self.out_dir, "step_progress.pdf"), bbox_inches='tight')
        plt.close()


        avg_loss = self.meta['val_loss']
        step = [i * self.interval for i in range(len(avg_loss))]
        step = np.array(step).astype(dtype=np.int32)
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        ax.plot(step, avg_loss, color='b', ls='-', label="val loss")

        train_loss = self.meta['train_loss']
        ax.plot(step, train_loss, color='g', ls='-', label="train loss")

        plt.title("Loss vs Epoch", fontsize=18)
        ax.set_xlabel("epoch", fontsize=16)
        ax.set_ylabel("loss", fontsize=16)
        plt.margins(0.1)
        ax.xaxis.set_ticks_position('none')
        ax.legend()
        fig.savefig(os.path.join(self.out_dir, "val_loss.pdf"), bbox_inches='tight')
        plt.close()

    def average_list(self, list_data, avg_step):
        avg_length = len(list_data) // avg_step
        avg_list = []
        for start_idx in range(avg_step):
            for i, data_i in enumerate(list_data[start_idx::avg_step]):
                if i < avg_length:
                    if start_idx == 0:
                        avg_list.append(data_i / avg_step)
                    else:
                        avg_list[i] += data_i / avg_step
        return avg_list

    def write_log(self, meta, is_final_test=False):
        if is_final_test:
            epoch = meta['epoch']
            val_acc = meta['val_acc']
            val_loss = meta['val_loss']
            class_table = meta['class_table']
            time = self.get_time_string()
            with open(self.log_txt, "a+") as f:
                f.write('\n------------------------------------------------------------------------------\n')
                f.write(time + ' \n')
                f.write(f'<Model of epoch {epoch + 1} Evaluating On Test Set>\n')
                f.write(f"val loss: {val_loss}\n")
                f.write(f"val acc: {'{:.2f}'.format(val_acc * 100)}%\n")
                f.write("Class Table:\n" + class_table + '\n')
            return


        epoch = meta['epoch']
        train_loss = meta['train_loss']
        train_acc = meta['train_acc']
        val_acc = meta['val_acc']
        val_loss = meta['val_loss']
        class_table = meta['class_table']
        time = self.get_time_string()

        with open(self.log_txt, "a+") as f:
            f.write('\n------------------------------------------------------------------------------\n')
            f.write(time + ' \n')
            f.write(f"epoch: {epoch + 1}\n")
            f.write(f"train loss: {train_loss}\n")
            f.write(f"train acc: {train_acc}\n")
            f.write(f"val loss: {val_loss}\n")
            f.write(f"val acc: {'{:.2f}'.format(val_acc * 100)}%\n")
            f.write("Class Table:\n" + class_table + '\n')

