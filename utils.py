# coding:utf-8
import sys
import numpy as np
from scipy.sparse import csr_matrix
import yaml
import tensorflow as tf
import datetime
import time
import gc
from sklearn import metrics
from collections import defaultdict

import warnings
import socket
import pynvml

def auc(lis, scale=1e6):
    def add(d, k, n):
        if k in d:
            d[k] += n
        else:
            d[k] = n

    s = {}
    c = {}
    for x in lis:
        label = 1 if x[0] > 0 else 0
        score = x[1]
        q = int(score * scale)
        add(s, q, 1)
        add(c, q, label)

    k = sorted(s.keys(), reverse=True)
    num_pos = 0
    num_neg = 0
    w = 0.0
    for q in k:
        pos = c[q]
        neg = s[q] - c[q]
        w = w + num_pos * neg + 0.5 * pos * neg
        num_pos += pos
        num_neg += neg
    if num_pos == 0 or num_neg == 0:
        return 0
    return w / (num_pos * num_neg)


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""
    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)
    #
    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        if user_id == 0 or len(user_id) < 10: continue
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag
    impression_total = 0
    total_auc = 0
    #
    u_total_auc = 0.0
    flag_count = 0
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = metrics.roc_auc_score(np.asarray(group_truth[user_id]).astype('float'),
                                        np.asarray(group_score[user_id]).astype('float'))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
            u_total_auc += auc
            flag_count += 1
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    u_avg_auc = round(float(u_total_auc) / max(flag_count, 1), 4)
    return group_auc, u_avg_auc


def user_gauc(labels, preds, user_id_list):
    """Calculate group auc"""
    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)
    #
    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        if user_id == 0:
            continue
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag
    impression_total = 0
    total_auc = 0
    #
    u_total_auc = 0.0
    flag_count = 0
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = metrics.roc_auc_score(np.asarray(group_truth[user_id]).astype('float'),
                                        np.asarray(group_score[user_id]).astype('float'))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
            u_total_auc += auc
            flag_count += 1
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    u_avg_auc = round(float(u_total_auc) / max(flag_count, 1), 4)
    return group_auc, u_avg_auc


def multi_auc(lis):
    preds = []
    labels = []
    uids = []
    online_scores = []
    recID_list=[]
    res = []
    for l in lis:
        label = 1 if l[0] > 0 else 0
        pred_score = l[1]
        recID = l[2]
        uid = l[3]
        score = l[4]
        rank = l[5]

        labels.append(label)
        preds.append(pred_score)
        recID_list.append(recID)
        online_scores.append(score)
        uids.append(uid)

    auc_score = metrics.roc_auc_score(np.array(labels).astype('float'), np.array(preds).astype('float'))
    group_auc, u_avg_auc = cal_group_auc(labels, preds, recID_list)

    o_auc_score = metrics.roc_auc_score(np.array(labels).astype('float'), np.array(online_scores).astype('float'))
    o_group_auc, o_u_avg_auc = cal_group_auc(labels, online_scores, recID_list)
    user_weight_gauc,user_average_gauc = user_gauc(labels, preds, uids)
    return auc_score, group_auc, u_avg_auc, o_auc_score, o_group_auc, o_u_avg_auc, user_weight_gauc,user_average_gauc



def load_yaml_file(yaml_file):
    f = open(yaml_file)
    ss = f.read()
    f.close()
    x = yaml.load(ss, Loader=yaml.FullLoader)
    return x


def save_yaml_file(x, yaml_file):
    fw = open(yaml_file, 'w')
    yaml.dump(x, fw)
    fw.close()


def load_conf(conf_file):
    conf = {}
    f = open(conf_file)
    for line in f:
        line = line.strip()
        if line == '':
            continue
        if line[0] == '#':
            continue
        lis = line.split('=')
        if len(lis) < 2:
            continue
        conf[lis[0]] = lis[1]
    f.close()
    return conf


def load_svmlight_stream(fs, zero_based=True):
    indptr = []
    indices = []
    data = []
    label_lis = []
    count = 0
    indptr.append(count)
    for line in fs:
        line = line.split('#', 1)[0].strip()
        if len(line) == 0:
            continue
        lis = line.split(' ')
        label = 0
        for i, x in enumerate(lis):
            if i == 0:
                if int(lis[0]) > 0:
                    label = 1
                if zero_based:
                    indices.append(0)
                    data.append(1.0)
                    count += 1
                continue
            y = x.split(':')
            ind = int(y[0])
            val = float(y[1])

            indices.append(ind)
            data.append(val)
            count += 1
        indptr.append(count)
        label_lis.append(label)

    X = csr_matrix((data, indices, indptr), dtype=float)
    Y = np.array(label_lis)
    return X, Y


def svmlight2dataset(fs, info=None):
    indices = []
    fea_index = []
    fea_value = []
    labels = []
    mx = -1
    NR = 0
    pos_num = 0
    for line in fs:
        NR += 1
        lis = line.split('#', 1)[0].strip(' \n').split(' ')
        label = int(lis[0])
        if label > 0:
            pos_num += 1
        labels.append(label)
        for i, x in enumerate(lis[1:]):
            y = x.split(':')
            ind = int(y[0])
            val = float(y[1])
            indices.append([NR - 1, i])
            fea_index.append(ind)
            fea_value.append(val)
            if i > mx:
                mx = i
    # X_ind=tf.sparse.to_dense(tf.sparse.SparseTensor(indices=indices, values=fea_index, dense_shape=[NR, mx+1]))
    # X_val=tf.sparse.to_dense(tf.sparse.SparseTensor(indices=indices, values=fea_value, dense_shape=[NR, mx+1]))
    print(datetime.datetime.now(), "trans x_ind")
    X_ind = tf.sparse.SparseTensor(indices=indices, values=fea_index, dense_shape=[NR, mx + 1])
    print(datetime.datetime.now(), "trans x_val")
    X_val = tf.sparse.SparseTensor(indices=indices, values=fea_value, dense_shape=[NR, mx + 1])
    print(datetime.datetime.now(), "trans x_label")
    X_lab = tf.constant(labels)
    print(datetime.datetime.now(), "begin dateset")
    dataset = tf.data.Dataset.from_tensor_slices({'fea_ids': X_ind, 'fea_vals': X_val, 'label': X_lab})
    print(datetime.datetime.now(), "end dateset")
    if isinstance(info, dict):
        info.clear()
        info['pos_num'] = pos_num
        info['all_num'] = len(labels)
    del fea_index
    del fea_value
    del labels
    gc.collect()
    print(datetime.datetime.now(), "end gc.collect")
    return dataset


def test_memo(fs, info=None):
    indices = []
    fea_index = []
    fea_value = []
    labels = []
    mx = -1
    NR = 0
    pos_num = 0
    print(datetime.datetime.now(), "begin")
    for line in fs:
        NR += 1
        lis = line.split('#', 1)[0].strip(' \n').split(' ')
        label = int(lis[0])
        if label > 0:
            pos_num += 1
        labels.append(label)
        for i, x in enumerate(lis[1:]):
            y = x.split(':')
            ind = int(y[0])
            val = float(y[1])
            indices.append([NR - 1, i])
            fea_index.append(ind)
            fea_value.append(val)
            if i > mx:
                mx = i
    print(datetime.datetime.now(), "end")
    time.sleep(300)
    print(datetime.datetime.now(), "end1")
    time.sleep(300)
    print(datetime.datetime.now(), "end2")


def parse_line_org(line):
    lis = line.split('#', 1)[0].strip(' \n').split(' ')
    label = int(lis[0])
    fea_index = []
    fea_value = []
    for i, x in enumerate(lis[1:]):
        y = x.split(':')
        ind = int(y[0])
        val = float(y[1])
        fea_index.append(ind)
        fea_value.append(val)
    return fea_index, fea_value, label


def parse_line(line):
    lis = line.split('#', 1)[0].strip(' \n').split(' ')

    cvr_label = int(lis[0])

    add_info_list = line.split('#', 1)[1].strip(' \n').split('\t')
    cat_label = int(add_info_list[9])
    clk_label = int(add_info_list[8])
    ext_label = int(add_info_list[15])
    fea_index = []
    fea_value = []
    for i, x in enumerate(lis[1:]):
        y = x.split(':')
        ind = int(y[0])
        val = float(y[1])
        fea_index.append(ind)
        fea_value.append(val)

    add_info_list = [i.encode() for i in add_info_list]

    return fea_index, fea_value, cvr_label, cat_label, clk_label, ext_label,add_info_list


def WriteTFRecord_org(fs, outfile_name):
    writer = tf.io.TFRecordWriter(outfile_name)
    for line in fs:
        fea_index, fea_value, label = parse_line(line)
        tfrecord_feature = {
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            "fea_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=fea_index)),
            "fea_vals": tf.train.Feature(float_list=tf.train.FloatList(value=fea_value))
        }
        example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
        writer.write(example.SerializeToString())
    writer.close()


def WriteTFRecord(fs, outfile_name):
    writer = tf.io.TFRecordWriter(outfile_name)
    count=0
    for line in fs:
        fea_index, fea_value, cvr_label, cat_label, clk_label,ext_label,add_info_list = parse_line(line)
        tfrecord_feature = {
            "cvr_label": tf.train.Feature(float_list=tf.train.FloatList(value=[cvr_label])),
            "cat_label": tf.train.Feature(float_list=tf.train.FloatList(value=[cat_label])),
            "clk_label": tf.train.Feature(float_list=tf.train.FloatList(value=[clk_label])),
            "ext_label": tf.train.Feature(float_list=tf.train.FloatList(value=[ext_label])),
            "fea_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=fea_index)),
            "fea_vals": tf.train.Feature(float_list=tf.train.FloatList(value=fea_value)),
            "add_infos": tf.train.Feature(bytes_list=tf.train.BytesList(value=add_info_list))
        }
        example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
        writer.write(example.SerializeToString())
        count += 1
    writer.close()
    print("总共写入 %d 条样本到文件: %s" % (count, outfile_name))


def ReadTFRecord_org(files, shuffle_size=1, batch_size=1, fetch_size=0, num_parallel=1):
    def parse_record(value):
        features = {
            "label": tf.io.FixedLenFeature([], tf.float32),
            "fea_ids": tf.io.VarLenFeature(tf.int64),
            "fea_vals": tf.io.VarLenFeature(tf.float32),
        }
        parsed = tf.io.parse_single_example(value, features)
        return parsed

    def parse_record_batch(value):
        if isinstance(value, list):
            return map(parse_record, value)
        return parse_record(value)

    # data_files = tf.io.gfile.glob(data_dir)
    if ',' in files:
        files = files.split(',')
    if isinstance(files, list) or isinstance(files, tuple):
        pass
    else:
        files = [files]
    dataset = tf.data.TFRecordDataset(files)
    # dataset = tf.data.TextLineDataset(data_files)
    dataset = dataset.map(parse_record, num_parallel_calls=num_parallel)
    if shuffle_size > 1:
        dataset = dataset.shuffle(shuffle_size)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if fetch_size > 0:
        dataset = dataset.prefetch(fetch_size)
    return dataset
    # return dataset.map(parse_record, num_parallel_calls=num_parallel)
    # dataset = dataset.map(get_sp_component, num_parallel_calls=2)
    # if shuffle: dataset = dataset.shuffle(buffer_size=batch_size*300)
    # if num_epochs: dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(9)
    # dataset = dataset.prefetch(1000)


def ReadTFRecord(files, shuffle_size=1, batch_size=1, fetch_size=0, num_parallel=1):
    def parse_record(value):
        features = {
            "cvr_label": tf.io.FixedLenFeature([], tf.float32),
            "cat_label": tf.io.FixedLenFeature([], tf.float32),
            "clk_label": tf.io.FixedLenFeature([], tf.float32),
            "ext_label": tf.io.FixedLenFeature([], tf.float32),
            "fea_ids": tf.io.VarLenFeature(tf.int64),
            "fea_vals": tf.io.VarLenFeature(tf.float32),
            "add_infos": tf.io.VarLenFeature(tf.string)
        }
        parsed = tf.io.parse_single_example(value, features)

        return parsed

    def parse_record_batch(value):
        if isinstance(value, list):
            return map(parse_record, value)
        return parse_record(value)

    # data_files = tf.io.gfile.glob(data_dir)
    if ',' in files:
        files = files.split(',')
    if isinstance(files, list) or isinstance(files, tuple):
        pass
    else:
        files = [files]
    dataset = tf.data.TFRecordDataset(files)
    # dataset = tf.data.TextLineDataset(data_files)
    dataset = dataset.map(parse_record, num_parallel_calls=num_parallel)
    if shuffle_size > 1:
        dataset = dataset.shuffle(shuffle_size)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if fetch_size > 0:
        dataset = dataset.prefetch(fetch_size)
    return dataset
    # return dataset.map(parse_record, num_parallel_calls=num_parallel)
    # dataset = dataset.map(get_sp_component, num_parallel_calls=2)
    # if shuffle: dataset = dataset.shuffle(buffer_size=batch_size*300)
    # if num_epochs: dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(9)
    # dataset = dataset.prefetch(1000)


def test1(file_path):
    d = tf.data.TextLineDataset(file_path)
    d = d.map(parse_line)
    for x in d:
        print(d)

def get_available_gpus(threshold_gb=0.5, need_cnt=2):
    """获取空闲显存大于指定阈值的GPU"""
    wait_time = 300
    while True:
        pynvml.nvmlInit()
        available_gpus = []
        for gpu_id in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = info.total / (1024 ** 2)
            used_mb = info.used / (1024 ** 2)
            free_gb = info.free / (1024 ** 3)
            free_mb = info.free / (1024 ** 2)
            utilization = (used_mb / total_mb) * 100
            if free_gb >= threshold_gb:
                available_gpus.append(str(gpu_id))
            print("显存占用情况 GPU:", gpu_id, used_mb, "/", total_mb, "free: ", free_mb)
        if available_gpus:
            print("选中可用GPU: ", ",".join(available_gpus))
            return available_gpus[:need_cnt]
        print("等待中...%d秒后重试" % (wait_time))
        time.sleep(wait_time)
        pynvml.nvmlShutdown()

def ReadTFRecord_new(files, shuffle_size=1, batch_size=1, fetch_size=tf.data.experimental.AUTOTUNE, num_parallel=tf.data.experimental.AUTOTUNE,drop_remainder=True):
    """
    优化后的TFRecord读取函数
    :param files: TFRecord文件路径（支持逗号分隔/列表）
    :param shuffle_size: shuffle buffer大小（按batch数算，建议=100）
    :param batch_size: 批次大小
    :param fetch_size: 预取数量（AUTOTUNE=自动适配）
    :param num_parallel: 并行数（AUTOTUNE=自动适配CPU核心数）
    :return: 优化后的tf.data.Dataset
    """

    # ========== 优化1：批量解析函数（替代单条解析，减少map调用次数） ==========
    def parse_batch(example_batch):
        """批量解析TFRecord，效率比单条解析高3-5倍"""
        features = {
            "cvr_label": tf.io.FixedLenFeature([], tf.float32),
            "cat_label": tf.io.FixedLenFeature([], tf.float32),
            "clk_label": tf.io.FixedLenFeature([], tf.float32),
            "ext_label": tf.io.FixedLenFeature([], tf.float32),
            "fea_ids": tf.io.VarLenFeature(tf.int64),
            "fea_vals": tf.io.VarLenFeature(tf.float32),
            "add_infos": tf.io.VarLenFeature(tf.string)
        }
        # 批量解析（关键：tf.io.parse_example 替代 tf.io.parse_single_example）
        parsed = tf.io.parse_example(example_batch, features)


        return parsed

    if ',' in files:
        files = files.split(',')
    if isinstance(files, list) or isinstance(files, tuple):
        pass
    else:
        files = [files]
    #dataset = tf.data.TFRecordDataset(files)
    dataset = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=num_parallel,  # 并行读取文件（你的版本缺失）
        buffer_size=8 * 1024 * 1024  # 8MB读取缓冲区，减少磁盘IO等待
    )

    # ========== 优化5：调整数据处理顺序（关键！减少内存占用+提升效率） ==========

    # 新顺序：batch → map(批量解析) → shuffle → prefetch → prefetch_to_device（优）
    # 1. 先batch（按批次打包，减少map调用次数）
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)  # drop_remainder避免最后一个小batch
    # 2. 批量解析（替代单条map）
    dataset = dataset.map(parse_batch, num_parallel_calls=num_parallel)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    # 4. shuffle（batch后shuffle，内存占用降低batch_size倍）
    if shuffle_size > 1:
        # shuffle_size按batch数算（比如shuffle_size=100 → 缓存100个batch）
        dataset = dataset.shuffle(buffer_size=shuffle_size)
    # 5. 预取优化（核心：优先预取到GPU）
    if fetch_size > 0:
        dataset = dataset.prefetch(buffer_size=fetch_size)
    return dataset


def ReadTFRecord_GZIP(
        files,
        shuffle_size=1,
        batch_size=1,
        fetch_size=tf.data.experimental.AUTOTUNE,
        num_parallel=tf.data.experimental.AUTOTUNE,
        drop_remainder=True,
        read_buffer_mb=32,
        interleave_files=True,
        cycle_length=16,
        include_add_infos=False,
        shuffle_files=True,
):
    """
    GZIP TFRecord 读取（train_hash）。

    流水线（与常见推荐一致，shuffle 在 parse 前对原始 record 做，省内存、开训更快）::

        打乱文件顺序（可选）
        -> interleave(cycle_length 个文件并行读，逐条交错输出)
        -> shuffle(shuffle_size 条样本，原始序列化 record)
        -> map(parse_single_example，多线程)
        -> batch
        -> prefetch(fetch_size 个 batch)

    :param files: 路径（逗号分隔或 list），支持 hdfs://
    :param shuffle_size: 样本级 shuffle 缓冲条数（不是 batch 数）
    :param fetch_size: 预取的 batch 数；AUTOTUNE 表示交给运行时
    :param cycle_length: interleave 同时读取的文件数（并行度）
    :param include_add_infos: 训练 False；test/predict True
    """
    features = {
        "cvr_label": tf.io.FixedLenFeature([], tf.float32),
        "cat_label": tf.io.FixedLenFeature([], tf.float32),
        "clk_label": tf.io.FixedLenFeature([], tf.float32),
        "ext_label": tf.io.FixedLenFeature([], tf.float32),
        "fea_ids": tf.io.VarLenFeature(tf.int64),
        "fea_vals": tf.io.VarLenFeature(tf.float32),
    }
    if include_add_infos:
        features["add_infos"] = tf.io.VarLenFeature(tf.string)

    def parse_single_record(serialized):
        return tf.io.parse_single_example(serialized, features)

    if ',' in files:
        files = files.split(',')
    if not isinstance(files, (list, tuple)):
        files = [files]
    files = [f for f in files if f]
    if not files:
        raise ValueError("ReadTFRecord_GZIP: no input files")

    read_buffer = int(read_buffer_mb) * 1024 * 1024
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    cycle = min(int(cycle_length), len(files))
    if num_parallel != tf.data.experimental.AUTOTUNE:
        cycle = min(cycle, int(num_parallel))
    cycle = max(cycle, 1)

    def _open_one(path):
        return tf.data.TFRecordDataset(
            path,
            compression_type='GZIP',
            buffer_size=read_buffer,
            num_parallel_reads=1,
        )

    # 1) 多文件：先打乱文件列表（日期/part 顺序），再 interleave 并行读、交错吐样本
    if interleave_files and len(files) > 1:
        file_ds = tf.data.Dataset.from_tensor_slices(files)
        if shuffle_files:
            file_ds = file_ds.shuffle(len(files), reshuffle_each_iteration=True)
        dataset = file_ds.interleave(
            _open_one,
            cycle_length=cycle,
            num_parallel_calls=num_parallel,
            block_length=1,
            deterministic=False,
        )
    else:
        dataset = tf.data.TFRecordDataset(
            files,
            compression_type='GZIP',
            num_parallel_reads=min(cycle, len(files)) if len(files) > 1 else num_parallel,
            buffer_size=read_buffer,
        )

    dataset = dataset.with_options(options)

    # 2) 样本级 shuffle（原始 record，尚未 parse）
    if shuffle_size > 1:
        dataset = dataset.shuffle(
            buffer_size=int(shuffle_size),
            reshuffle_each_iteration=True,
        )

    # 3) 逐条解析（多线程 map）
    dataset = dataset.map(parse_single_record, num_parallel_calls=num_parallel)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    # 4) 组 batch
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # 5) 预取 fetch_size 个 batch（AUTOTUNE 时由 TF 决定深度）
    if fetch_size is not None and fetch_size != 0:
        dataset = dataset.prefetch(buffer_size=fetch_size)

    return dataset


def ReadTFRecordV2(files, shuffle_size=1, batch_size=1, fetch_size=0, num_parallel=10):
    def parse_record(value):
        features = {
            "cvr_label": tf.io.FixedLenFeature([], tf.float32),
            "cat_label": tf.io.FixedLenFeature([], tf.float32),
            "clk_label": tf.io.FixedLenFeature([], tf.float32),
            "ext_label": tf.io.FixedLenFeature([], tf.float32),
            "fea_ids": tf.io.VarLenFeature(tf.int64),
            "fea_vals": tf.io.VarLenFeature(tf.int64),
            "add_info_list": tf.io.VarLenFeature(tf.string)
        }
        parsed = tf.io.parse_single_example(value, features)
        return parsed

    def parse_record_batch(value):
        if isinstance(value, list):
            return map(parse_record, value)
        return parse_record(value)

    if ',' in files:
        files = files.split(',')
    if isinstance(files, list) or isinstance(files, tuple):
        pass
    else:
        files = [files]
    dataset = None
    tf_data_files = tf.data.Dataset.from_tensor_slices(files)
    dataset = tf_data_files.interleave(
        map_func=lambda x: tf.data.TFRecordDataset(
            x,
            buffer_size=100000000,
            num_parallel_reads=4,
            compression_type="GZIP",
        ),
        num_parallel_calls=num_parallel,
        cycle_length=num_parallel,
        block_length=32
    )
    dataset = dataset.map(parse_record, num_parallel_calls=num_parallel)
    if shuffle_size > 1:
        dataset = dataset.shuffle(shuffle_size)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if fetch_size > 0:
        dataset = dataset.prefetch(fetch_size)
    return dataset





if __name__ == '__main__':
    # X,Y = load_svmlight_stream(sys.stdin)
    # print(X.toarray())
    WriteTFRecord(sys.stdin, sys.argv[1])
    # d=ReadTFRecord('ins.record')
    # for i,x in enumerate(d):
    #    if i!=0:
    #        continue
    #    print(i,x["fea_ids"],x["fea_vals"],x['label'])
