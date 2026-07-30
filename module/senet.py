import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from logger import logger


class SENet(tf.keras.layers.Layer):
    retrace_counter = 0  # 类变量，追踪 retrace 次数
    def __init__(self, reduction_ratio=8,**kwargs):
        super(SENet, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.logdir = 'log/tensorboard/SeNet'  # 固定日志路径
        self.writer = tf.summary.create_file_writer(self.logdir)
        logger.info('SENet: reduction_ratio={}'.format(self.reduction_ratio))
        
    def build(self, input_shape):
        # input: [B, n_fea, fea_dim]
        self.n_fea = input_shape[1]
        self.fea_dim = input_shape[2]
        self.reduction_layer = tf.keras.layers.Dense(
            units=max(1, self.n_fea // self.reduction_ratio),
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        )

        self.output_layer = tf.keras.layers.Dense(
            units=self.n_fea,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(1e-4),
            use_bias=False
        )

    @tf.function
    def call(self, inputs,step=None):
        #判断是否在重新构建图
        # SENet.retrace_counter += 1
        # tf.print("[SENet] retrace count:", SENet.retrace_counter)
        z = tf.reduce_mean(inputs, axis=-1)
        # 通道间交互（SENet核心）压缩，还原 — [B, n_fea] → [B, n_fea]
        a = self.reduction_layer(z)
        a = self.output_layer(a)
        a = tf.expand_dims(a, axis=-1)

        if step is not None:
            with self.writer.as_default():
                tf.summary.histogram("SENet/feature_attention", a, step=step)
                tf.summary.scalar("SENet/mean_attention", tf.reduce_mean(a), step=step)
        else:
            tf.print("[SENet] step 未传入或为 None，跳过 TensorBoard 写入")
        # 加权输入 — [B, n_fea, fea_dim]
        return inputs * a


