import utils as ut
import argparse
import copy
import sys, os
import numpy as np
import tensorflow as tf
import datetime
from datetime import timedelta
import model_conf
from model import Model
from tensorflow.keras import regularizers
import time
from sklearn import metrics

class Learner:
    def __init__(self):
        self.model = None

    def set_training_mode(self, enable_training, is_save_model):
        self.model.training = enable_training
        self.model.is_save_model = is_save_model

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, feat, buy_weight=1.0, cat_weight=1.0, click_weight=1.0, ext_weight=1.0):
        model = self.model
        with tf.GradientTape() as tape:
            pred_buy, pred_cat, pred_click, pred_ext = model([feat['fea_ids'], feat['fea_vals']])

            loss_buy = model.loss_bc(tf.expand_dims(feat['cvr_label'], 1), pred_buy)
            loss_cat = model.loss_bc(tf.expand_dims(feat['cat_label'], 1), pred_cat)
            loss_click = model.loss_bc(tf.expand_dims(feat['clk_label'], 1), pred_click)
            loss_ext = model.loss_bc(tf.expand_dims(feat['ext_label'], 1), pred_ext)

            final_loss = loss_buy * buy_weight + loss_cat * cat_weight + loss_click * click_weight + loss_ext * ext_weight

            gradients = tape.gradient(final_loss, model.trainable_weights)
        model.apply_gradients_dual(gradients)
        return loss_buy, loss_cat, loss_click, loss_ext, final_loss, pred_buy, pred_cat, pred_click, pred_ext

    def _date_range(self, start, end):
        """返回 [start, end] 闭区间内的所有天(YYYYMMDD 字符串,升序)"""
        d = datetime.datetime.strptime(start, "%Y%m%d")
        end_d = datetime.datetime.strptime(end, "%Y%m%d")
        days = []
        while d <= end_d:
            days.append(d.strftime("%Y%m%d"))
            d += timedelta(1)
        return days

    def get_day_files(self, data_arg, day):
        """取某一天的数据文件。data_arg 可以是逗号分隔的天目录,也可以是基础目录。"""
        base_paths = [p.rstrip('/') for p in str(data_arg).split(',') if p]
        files = []
        for bp in base_paths:
            if bp.endswith(day):
                files += tf.io.gfile.glob(bp + "/part*")
            else:
                files += tf.io.gfile.glob("%s/%s/part*" % (bp, day))
        return sorted(set(files))

    def train(self, data_arg, start_day, end_day, model_path=None, data_path=None):
        if self.model is None:
            self.model = Model(training=True)
        model = self.model

        train_writer = None
        tensorboard_dir = "./log/tensorboard_data_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if tensorboard_dir:
            train_writer = tf.summary.create_file_writer(
                os.path.join(tensorboard_dir, "train"), max_queue=1000
            )
            model.set_summary_writer(train_writer, histogram_freq=100)
            print('TensorBoard enabled:', tensorboard_dir)

        days = self._date_range(start_day, end_day)
        print("train days [%s, %s], total %d days" % (start_day, end_day, len(days)))

        batch_size = model_conf.batch_size
        shuffle_size = batch_size * 10

        #load ckpt (需要先跑一个 batch 建好变量再 restore)
        ckpt_path = self.get_model_checkpoint_from_file(model_conf.done_file_path)
        if ckpt_path is not None:
            print("load model from checkpoint:", ckpt_path)
            ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer, optimizer_emb=model.optimizer_emb)

            probe_files = self.get_day_files(data_arg, days[0])
            probe_ds = ut.ReadTFRecordV2(probe_files, shuffle_size=1, batch_size=batch_size, fetch_size=1, num_parallel=10)
            first_batch = next(iter(probe_ds))
            _ = model([first_batch['fea_ids'], first_batch['fea_vals']])

            model.warmup_optimizers()

            ckpt.restore(tf.train.latest_checkpoint(ckpt_path)).assert_consumed()
            print("Restored optimizer step: ", model.optimizer.iterations.numpy())
            print("load checkpoint path: ", ckpt_path)

        #每天训练完直接算指标,结果按天写到 metrics 文件(不落 pred/label 明细,省内存/磁盘)
        out_dir = model_conf.local_model_dir
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except Exception:
                pass
        metric_path = os.path.join(out_dir, 'metrics_by_day.txt')
        task_names = ['buy', 'cat', 'click', 'ext']

        mfout = open(metric_path, 'a')
        mfout.write('\t'.join(['day', 'task', 'n', 'auc', 'gauc', 'mae', 'pos_rate']) + '\n')

        #多天累计统计:各任务正样本数 [buy, cat, click, ext] 和总样本数
        self.pos = np.zeros(len(task_names))
        self.cnt = 0
        self.gstep = 0

        print('training...')
        for idx, day in enumerate(days):
            files = self.get_day_files(data_arg, day)
            if not files:
                print(datetime.datetime.now(), "day %s: no files found, skip" % day)
                continue
            print(datetime.datetime.now(), "==== start day %s (%d/%d), %d files ====" % (
                day, idx + 1, len(days), len(files)))

            #train one pass over this day(训练完当天直接算指标写入 metrics 文件)
            self.set_training_mode(True, False)
            ds = ut.ReadTFRecordV2(files, shuffle_size=shuffle_size, batch_size=batch_size, fetch_size=10, num_parallel=10)
            ds = ds.apply(tf.data.experimental.ignore_errors())
            self.train_one_day(ds, day, train_writer, mfout)

            #当天训练的 summary 落盘
            if train_writer is not None:
                train_writer.flush()

            #每 N 天保存一个 ckpt(最后一天也保存)
            is_last = (idx == len(days) - 1)
            if (idx + 1) % model_conf.ckpt_save_days == 0 or is_last:
                self.save_checkpoint(day)

        mfout.close()
        print(datetime.datetime.now(), "metrics written to %s" % metric_path)

        #导出 serving 模型
        self.set_training_mode(False, True)
        self.dump_serving_model(end_day, 0)
        self.set_training_mode(True, False)

    def train_one_day(self, train_data, day, train_writer, mfout=None):
        model = self.model

        uid_index = model_conf.uid_add_info_index
        field_num = model_conf.add_info_field_num
        mod_threshold = model_conf.eval_uid_ratio * 100  # uid % 100 < 此值 则命中(0.01 -> 1)
        label_keys = ['cvr_label', 'cat_label', 'clk_label', 'ext_label']

        #当天采样子集缓存在内存(只保留当天,算完即释放,避免落大文件/OOM)
        eval_uids = []
        eval_labels = [[] for _ in label_keys]
        eval_preds = [[] for _ in label_keys]

        n_sampled = 0
        step = -1
        for step, feat in enumerate(train_data):
            label_arrs = [np.reshape(feat[k].numpy(), [-1]) for k in label_keys]
            self.cnt += label_arrs[0].shape[0]
            self.pos += [a.sum() for a in label_arrs]

            loss_buy, loss_cat, loss_click, loss_ext, final_loss, pred_buy, pred_cat, pred_click, pred_ext = self.train_step(feat)

            #收集 uid 采样子集的 pred/label 到内存,当天训完直接算指标
            if mfout is not None:
                uids = feat['add_info_list'].values.numpy()[uid_index::field_num]
                #向量化:整批一次算出命中下标(避免整批逐样本 Python 循环)
                try:
                    uid_int = uids.astype('S').astype(np.int64)
                    sel = np.where((uid_int % 100) < mod_threshold)[0]
                except Exception:
                    sel = np.array([], dtype=np.int64)
                if sel.size > 0:  # 命中才把 pred 拉到 host
                    pred_arrs = [
                        np.reshape(pred_buy.numpy(), [-1]),
                        np.reshape(pred_cat.numpy(), [-1]),
                        np.reshape(pred_click.numpy(), [-1]),
                        np.reshape(pred_ext.numpy(), [-1]),
                    ]
                    n_sampled += sel.size
                    eval_uids.extend(uids[j].decode('utf-8', 'ignore') for j in sel)
                    for t_idx in range(len(label_keys)):
                        eval_labels[t_idx].extend(label_arrs[t_idx][sel].tolist())
                        eval_preds[t_idx].extend(pred_arrs[t_idx][sel].tolist())

            self.gstep += 1
            if train_writer is not None and self.gstep % 100 == 0:
                global_step = model.optimizer.iterations.numpy()  # 只在打点时同步一次,作 x 轴
                with train_writer.as_default():
                    tf.summary.scalar('loss_buy', tf.reduce_mean(loss_buy), step=global_step)
                    tf.summary.scalar('loss_cat', tf.reduce_mean(loss_cat), step=global_step)
                    tf.summary.scalar('loss_click', tf.reduce_mean(loss_click), step=global_step)
                    tf.summary.scalar('loss_ext', tf.reduce_mean(loss_ext), step=global_step)
                    tf.summary.scalar('loss/total', tf.reduce_mean(final_loss), step=global_step)

                    tf.summary.scalar('data/pos_rate_buy', self.pos[0] / max(self.cnt, 1), step=global_step)
                    tf.summary.scalar('data/pos_rate_click', self.pos[2] / max(self.cnt, 1), step=global_step)

                print(datetime.datetime.now(),
                        "day %s steps: %d, buy loss: %04f, pos: %d, cnt: %d,  cat loss: %04f, cat pos: %d,  click loss: %04f, click pos: %d,  ext loss: %04f, ext pos: %d, sampled: %d" % (
                        day, self.gstep, tf.reduce_mean(loss_buy), self.pos[0], self.cnt, tf.reduce_mean(loss_cat), self.pos[1],
                        tf.reduce_mean(loss_click), self.pos[2], tf.reduce_mean(loss_ext), self.pos[3], n_sampled))

        if step < 0:
            print(datetime.datetime.now(), "day %s finish, no batches" % day)
            return

        #当天训练完成后直接算 auc/gauc/mae,把结果写到 metrics 文件
        if mfout is not None:
            self._write_day_metrics(day, eval_uids, eval_labels, eval_preds, mfout, train_writer)

    def _write_day_metrics(self, day, uids, labels_by_task, preds_by_task, mfout, train_writer=None):
        """用当天内存里的采样子集算 auc/gauc/mae,结果写入 metrics 文件(+TensorBoard)。"""
        task_names = ['buy', 'cat', 'click', 'ext']
        try:
            step = int(day)  # 用日期做 tensorboard step(按天单调递增)
        except Exception:
            step = 0
        for t_idx, t in enumerate(task_names):
            labels = np.asarray(labels_by_task[t_idx], dtype=np.float32)
            preds = np.asarray(preds_by_task[t_idx], dtype=np.float32)
            n = labels.shape[0]
            if n == 0:
                print("METRIC day=%s task=%s: no samples" % (day, t))
                continue
            mae = float(np.mean(np.abs(labels - preds)))
            try:
                auc = metrics.roc_auc_score(labels, preds)
            except Exception as e:
                auc = float('nan')
                print("METRIC day=%s task=%s auc failed: %s" % (day, t, e))
            try:
                gauc, _ = ut.cal_group_auc(labels, preds, uids)
            except Exception:
                gauc = float('nan')
            pos_rate = float(np.mean(labels))
            print("METRIC day=%s task=%s n=%d auc=%.4f gauc=%.4f mae=%.4f pos_rate=%.4f" % (
                day, t, n, auc, gauc, mae, pos_rate))
            mfout.write("%s\t%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\n" % (day, t, n, auc, gauc, mae, pos_rate))
            if train_writer is not None:
                with train_writer.as_default():
                    tf.summary.scalar('eval_auc/%s' % t, auc, step=step)
                    tf.summary.scalar('eval_gauc/%s' % t, gauc, step=step)
                    tf.summary.scalar('eval_mae/%s' % t, mae, step=step)
        mfout.flush()

    def save_checkpoint(self, day):
        model = self.model
        save_dir = "%s/checkpoints/%s/" % (model_conf.local_model_dir, day)
        export_dir = save_dir + "tfmodel"
        ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer, optimizer_emb=model.optimizer_emb)
        ckpt.save(export_dir)

        done_dir = os.path.dirname(model_conf.done_file_path)
        if done_dir and not os.path.exists(done_dir):
            try:
                os.makedirs(done_dir)
            except Exception as e:
                print("Warning: Failed to create done_file directory %s:" % done_dir, e)
        try:
            with open(model_conf.done_file_path, 'a') as f:
                f.write(day + "\t" + save_dir + "\n")
        except Exception as e:
            print("Warning: Failed to write done file %s:" % model_conf.done_file_path, e)
        print(datetime.datetime.now(), "saved checkpoint for day %s -> %s" % (day, save_dir))

    def dump_serving_model(self, end_day, epo):
        if self.model is None:
            return

        train_model = self.model

        fid_keys, fid_values = train_model.fid_table.export()
        fid_keys_ads, fid_values_ads = train_model.fid_table_din_ads.export()

        fid_keys = tf.reshape(fid_keys, [-1])
        fid_values = tf.reshape(fid_values, [-1])

        fid_keys_ads = tf.reshape(fid_keys_ads, [-1])
        fid_values_ads = tf.reshape(fid_values_ads, [-1])

        serve_model = Model(
            training=False,
            pred=True,
            fid_kv=(fid_keys, fid_values),
            fid_ads_kv=(fid_keys_ads, fid_values_ads)
        )
        serve_model.compile(optimizer=self.model.optimizer, loss=self.model.loss_bc, metrics=['mae'])

        serve_model.training = False
        serve_model.is_save_model = True
        dummy_sids = tf.constant([[0] * model_conf.padding_size], tf.uint32)
        dummy_fids = tf.constant([[0] * model_conf.padding_size], tf.uint64)
        _ = serve_model([dummy_sids, dummy_fids])

        serve_model._set_inputs([
            tf.keras.Input(shape=(model_conf.padding_size,), dtype=tf.dtypes.uint32),
            tf.keras.Input(shape=(model_conf.padding_size,), dtype=tf.dtypes.uint64),
        ])

        #serve_model.load_weights(model_conf.model_path)
        serve_model.set_weights(train_model.get_weights())

        serve_model.save(
            'serving_model_%s/%s00' % (epo, end_day),
            save_format='tf'
        )

    def count_parameters(self, verbose=False):
        total_params = 0
        if self.model is None:
            print("Model not initialized")
            return
        for i,weight in enumerate(self.model.trainable_weights):
            shape = weight.shape.as_list()
            num_params = np.prod(shape)
            total_params += num_params
            if verbose:
                print("Layer "+str(i)+":"+str(weight.name)+" | Shape:"+str(shape)+" | params:"+str(num_params))
        print("Total trainable parameters:"+str(total_params))
        return total_params

    def get_model_checkpoint_from_file(self, done_file_path='model.done'):
        if not os.path.exists(done_file_path):
            print("model.done not exit in patch: ",done_file_path)
            return None

        with open(done_file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("model.done is null ")
                return None

            # read last line
            last_line = lines[-1].strip()
            if not last_line:
                print("model.done last line is null")
                return None

            # file format: checkpoint_day\tcheckpoint_path
            parts = last_line.split('\t')
            if len(parts) >= 2:
                ckpt_day = parts[0]
                ckpt_path = parts[1]
                print("load model checkpoint_path=%s, checkpoint_day=%s",ckpt_path, ckpt_day)
                return ckpt_path
            else:
                print("model.done last line format error")
                return None

if __name__ == "__main__":
    # init args and model
    parse = argparse.ArgumentParser(description='get input args')
    parse.add_argument('-data', type=str, help='input data files')
    parse.add_argument('-start_day', type=str, help='train start day')
    parse.add_argument('-end_day', type=str, help='train end day')

    args = parse.parse_args()
    solver = Learner()

    # set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = model_conf.gpu_id
    print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # start training or testing
    if model_conf.train_mode == 'train':
        #按天 for 循环读取数据并训练(数据读取放在 train() 内部,逐天进行)
        print('start training, day by day from %s to %s' % (args.start_day, args.end_day))
        start_time = time.time()
        solver.train(args.data, data_path=args.data, start_day=args.start_day, end_day=args.end_day)
        end_time2 = time.time()
        print('end training, using_time_training: ', end_time2 - start_time)
    else:
        pass
