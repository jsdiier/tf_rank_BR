import utils as ut
import argparse
import copy
import sys, os
import numpy as np
import tensorflow as tf
import datetime
import model_conf
from model import Model
from datetime import timedelta
from tensorflow.keras import regularizers
import time

class Learner:
    def __init__(self):
        self.model = None

    def set_training_mode(self, enable_training):
        self.model.training = enable_training

    def get_files(self, path, start, end):
        d = datetime.datetime.strptime(start, "%Y%m%d")
        end = datetime.datetime.strptime(end, "%Y%m%d")
        files = []
        while d <= end:
            files += tf.io.gfile.glob("{path}/{d}/part*".format(path=path,d=d.strftime("%Y%m%d")))
            d += timedelta(1)
        files = sorted(files)
        return files

    def train(self, train_data, model_path=None, data_path=None):
        if self.model is None:
            self.model = Model(training=True)

        batch_size = model_conf.batch_size
        epoch_num = model_conf.epoch_num

        model = self.model

        #load ckpt
        ckpt_path = self.get_model_checkpoint_from_file(model_conf.done_file_path)
        #ckpt_path="model/checkpoints/20260430_0/"
        if ckpt_path is not None:
            print("load model from checkpoint:", ckpt_path)
            ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

            first_batch = next(iter(train_data))
            _ = model([first_batch['fea_ids'], first_batch['fea_vals']])

            dummy_grad = [tf.zeros_like(v) for v in model.trainable_variables]
            model.optimizer.apply_gradients(zip(dummy_grad, model.trainable_variables))

            #ckpt.restore(tf.train.latest_checkpoint(ckpt_path)).expect_partial()
            ckpt.restore(tf.train.latest_checkpoint(ckpt_path)).assert_consumed()
            print("Restored optimizer step: ", model.optimizer.iterations.numpy())
            print("load checkpoint path: ", ckpt_path)

        buy_weight = 1.0
        cat_weight = 1.0
        click_weight = 1.0
        ext_weight = 1.0
        print('testing...')
        self.set_training_mode(False)
        self.test(train_data)
        self.set_training_mode(True)

    def test(self, test_data, model_path='', ):
        res_buy = []
        res_cat = []
        res_click = []
        res_ext = []
        loss_buy_sum = 0.0
        loss_cat_sum = 0.0
        loss_click_sum = 0.0
        loss_ext_sum = 0.0
        pos_buy = 0
        pos_cat = 0
        pos_click = 0
        pos_ext = 0
        cnt = 0
        low_score_num = 0
        sum_pred_buy = 0.0
        sum_pred_cat = 0.0
        sum_pred_click = 0.0
        sum_pred_ext = 0.0
        for step, feat in enumerate(test_data):
            cnt += feat['cvr_label'].shape[0]
            pred_buy, pred_cat, pred_click, pred_ext = model([feat['fea_ids'], feat['fea_vals']])

            loss_buy = self.model.loss_bc(tf.expand_dims(feat['cvr_label'], 1), pred_buy)
            loss_cat = self.model.loss_bc(tf.expand_dims(feat['cat_label'], 1), pred_cat)
            loss_click = self.model.loss_bc(tf.expand_dims(feat['clk_label'], 1), pred_click)
            loss_ext = self.model.loss_bc(tf.expand_dims(feat['ext_label'], 1), pred_ext)

            loss_buy_sum += tf.reduce_sum(loss_buy, 0)
            loss_cat_sum += tf.reduce_sum(loss_cat, 0)
            loss_click_sum += tf.reduce_sum(loss_click, 0)
            loss_ext_sum += tf.reduce_sum(loss_ext, 0)

            pred_buy = tf.squeeze(pred_buy, 1).numpy().tolist()
            pred_cat = tf.squeeze(pred_cat, 1).numpy().tolist()
            pred_click = tf.squeeze(pred_click, 1).numpy().tolist()
            pred_ext = tf.squeeze(pred_ext, 1).numpy().tolist()

            buy_label = feat['cvr_label'].numpy().reshape(-1).tolist()
            cat_label = feat['cat_label'].numpy().reshape(-1).tolist()
            click_label = feat['clk_label'].numpy().reshape(-1).tolist()
            ext_label = feat['ext_label'].numpy().reshape(-1).tolist()

            recID = [i.decode() for i in feat['add_info_list'].values.numpy()[6:][::24]]
            uid = [i.decode() for i in feat['add_info_list'].values.numpy()[5:][::24]]
            score = [i.decode() for i in feat['add_info_list'].values.numpy()[4:][::24]]
            rank = [i.decode() for i in feat['add_info_list'].values.numpy()[3:][::24]]

            pos_buy += sum(buy_label)
            pos_cat += sum(cat_label)
            pos_click += sum(click_label)
            pos_ext += sum(ext_label)

            sum_pred_buy += sum(pred_buy)
            sum_pred_cat += sum(pred_cat)
            sum_pred_click += sum(pred_click)
            sum_pred_ext += sum(pred_ext)

            res_buy.extend(zip(buy_label, pred_buy, recID, uid, score, rank))
            res_cat.extend(zip(cat_label, pred_cat, recID, uid, score, rank))
            res_click.extend(zip(click_label, pred_click, recID, uid, score, rank))
            res_ext.extend(zip(ext_label, pred_ext, recID, uid, score, rank))
            low_score_num += len(list((filter(lambda x: x < 0.01, pred_buy))))

        print(datetime.datetime.now(), "buy loss:%04f, cat loss:%04f, click loss:%04f, ext loss:%04f" % (
              loss_buy_sum/len(res_buy), loss_cat_sum/len(res_cat), loss_click_sum/len(res_click), loss_ext_sum/len(res_ext)))
        print(datetime.datetime.now(), "pos_buy:%d, pos_cat:%d, pos_click:%d, pos_ext:%d, all_num:%d" % (pos_buy, pos_cat, pos_click, pos_ext, cnt))
        print(datetime.datetime.now(), "buy_rate:%04f, cat_rate:%04f, click_rate:%04f, ext_rate:%04f" % (pos_buy / cnt, pos_cat / cnt, pos_click / cnt, pos_ext / cnt))
        print(datetime.datetime.now(), "res_buy:%04f, res_cat:%04f, res_click:%04f, res_ext:%04f" % (sum_pred_buy / len(res_buy), sum_pred_cat / len(res_cat), sum_pred_click / len(res_click), sum_pred_ext  / len(res_ext)))
        print(datetime.datetime.now(), "low score rate:%f" % (low_score_num / len(res_buy)))

        auc_score, group_auc, u_avg_auc, o_auc_score, o_group_auc, o_u_avg_auc = ut.multi_auc(res_buy)
        print(model_path, "test_buy auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
            auc_score, group_auc, u_avg_auc, len(res_buy), loss_buy_sum / len(res_buy), pos_buy))
        print(model_path, "online_buy auc:%f gauc:%f uauc:%f " % (o_auc_score, o_group_auc, o_u_avg_auc))

        auc_score_cat, group_auc_cat, u_avg_auc_cat, o_auc_score_cat, o_group_auc_cat, o_u_avg_auc_cat = ut.multi_auc(
            res_cat)
        print(model_path, "test_cat auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
            auc_score_cat, group_auc_cat, u_avg_auc_cat, len(res_cat), loss_cat_sum / len(res_cat), pos_cat))
        print(model_path, "online_cat auc:%f gauc:%f uauc:%f " % (o_auc_score_cat, o_group_auc_cat, o_u_avg_auc_cat))

        auc_score_click, group_auc_click, u_avg_auc_click, o_auc_score_click, o_group_auc_click, o_u_avg_auc_click = ut.multi_auc(
            res_click)
        print(model_path, "test_click auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
            auc_score_click, group_auc_click, u_avg_auc_click, len(res_click), loss_click_sum / len(res_click),
            pos_click))
        print(model_path,
              "online_click auc:%f gauc:%f uauc:%f " % (o_auc_score_click, o_group_auc_click, o_u_avg_auc_click))

        auc_score_ext, group_auc_ext, u_avg_auc_ext, o_auc_score_ext, o_group_auc_ext, o_u_avg_auc_ext = ut.multi_auc(
            res_ext)
        print(model_path, "test_ext auc:%f gauc:%f uauc:%f size:%d loss:%f, pos: %d" % (
            auc_score_ext, group_auc_ext, u_avg_auc_ext, len(res_ext), loss_ext_sum / len(res_ext),
            pos_ext))
        print(model_path,
              "online_ext auc:%f gauc:%f uauc:%f " % (o_auc_score_ext, o_group_auc_ext, o_u_avg_auc_ext))

        pass

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
        batch_size = model_conf.batch_size
        shuffle_size = batch_size * 10

        #read data
        print('start read data')
        start_time = time.time()
        files = solver.get_files(args.data, args.start_day, args.end_day)
        print("files: ", files)
        ds = ut.ReadTFRecordV2(files, shuffle_size=shuffle_size, batch_size=batch_size, fetch_size=10, num_parallel=10)
        ds = ds.apply(tf.data.experimental.ignore_errors())
        end_time = time.time()
        using_time = end_time - start_time
        print('end read data, using_time_reading_data: ', using_time)

        #start training
        print('start training')
        solver.train(ds, data_path=args.data)
        end_time2 = time.time()
        using_time2 = end_time2 - end_time
        print('end training, using_time_training: ', using_time2)
    else:
        pass
