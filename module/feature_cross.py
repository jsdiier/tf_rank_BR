
"""
@function: 特征交叉
@author: jiazhuo

This module implements various feature crossing techniques commonly used in machine learning and recommendation systems.

The module may include implementations of:

-FiBiNet
-MaskNet
-DCN
-DCN_v2
"""

import tensorflow as tf
from tensorflow.keras import regularizers
from logger import logger

class IGM(tf.keras.layers.Layer):
    """
    Instanc-guided mask:掩码模块，两层全连接
    """
    def __init__(self, reduce_r,mask_input_dim,prefix, **kwargs):
        self.reduce_r = reduce_r
        self.mask_input_dim = mask_input_dim
        self.prefix = prefix
        if reduce_r<10:
            self.aggresion_size=int(reduce_r*mask_input_dim)
        else:
            self.aggresion_size=reduce_r
        logger.info('{}: reduce_r={},mask_input_dim={}'.format(self.prefix,self.reduce_r,self.mask_input_dim))
        super(IGM, self).__init__(**kwargs)

    def build(self,input_shape):
        self.agg_layer=tf.keras.layers.Dense(
            units=self.aggresion_size,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4),
            name=self.prefix+'agg_layer'
        )
        self.proj_layer=tf.keras.layers.Dense(
            units=self.mask_input_dim,
            activation='linear',
            kernel_regularizer=regularizers.l2(1e-4),
            name=self.prefix+'proj_layer'
        )
        super(IGM, self).build(input_shape)

    def call(self,inputs):
        agg_output=self.agg_layer(inputs)
        proj_output=self.proj_layer(agg_output)
        return proj_output

class MaskBlock(tf.keras.layers.Layer):
    """
    Masknet 基本组成模块
    """

    def __init__(self, prefix,reduce_r,mask_input_dim,out_dim, **kwargs):
        self.prefix = prefix
        self.IGM=IGM(reduce_r,mask_input_dim,prefix)
        self.out_dim = out_dim
        super(MaskBlock, self).__init__(**kwargs)

    def build(self,input_shape):
        self.layer_norm2=tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            name=self.prefix+'layer_norm2'
        )

        self.hidden_layer=tf.keras.layers.Dense(
            units=self.out_dim,
            activation='linear',
            kernel_regularizer=regularizers.l2(1e-4),
            name=self.prefix+'_hidden_layer',
            use_bias=False
        )
        super(MaskBlock, self).build(input_shape)

    def call(self,inputs):
        assert isinstance(inputs,list), 'MaskBlock inputs must be a list'
        # IGM输入
        active_input=inputs[0]
        passive_input=inputs[1]
        IGM_output=self.IGM(active_input)
        output=passive_input*IGM_output
        hidden_input=self.hidden_layer(output)
        hidden_output=self.layer_norm2(hidden_input)
        return tf.keras.activations.relu(hidden_output)

class ParallelMaskNet(tf.keras.layers.Layer):
    """
    MaskNet: 并行特征交叉网络

    Args:
        input_dim:输入维度
        igm_reduce_list:每个IGM模块的缩减系数列表
        maskblock_hidden_dim_list:每个MaskBlock模块的隐藏层列表
    """
    def __init__(self,input_mask_dim,igm_reduce_list,maskblock_hidden_dim_list,mlp_hidden_list=None,prefix='masknet', **kwargs):
        self.input_mask_dim = input_mask_dim
        self.igm_reduce_list = igm_reduce_list
        self.maskblock_hidden_dim_list = maskblock_hidden_dim_list
        self.prefix = prefix
        self.mlp_hidden_list = mlp_hidden_list
        logger.info('ParallelMaskNet: input_mask_dim={},igm_reduce_list={},maskblock_hidden_dim_list={}'.format(self.input_mask_dim,self.igm_reduce_list,self.maskblock_hidden_dim_list))
        super(ParallelMaskNet, self).__init__()

    def build(self,input_shape):
        self.maskblock_list=[]
        for i in range(len(self.igm_reduce_list)):
            maskblock=MaskBlock(
                prefix=self.prefix+'_maskblock_'+str(i),
                reduce_r=self.igm_reduce_list[i],
                mask_input_dim=self.input_mask_dim,
                out_dim=self.maskblock_hidden_dim_list[i]
            )
            self.maskblock_list.append(maskblock)
        self.total_maskblock_output_dim = sum(self.maskblock_hidden_dim_list)

        if self.mlp_hidden_list is not None:
            mlp_layers = []
            for i, units in enumerate(self.mlp_hidden_list):
                mlp_layers.append(
                    tf.keras.layers.Dense(
                        units=units,
                        activation='relu',
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="{}_mlp_{}".format(self.prefix, i),
                        use_bias=False
                    )
                )
            self.mlp = tf.keras.Sequential(mlp_layers, name="{}_mlp".format(self.prefix))

        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            name=self.prefix + '_layer_norm'
        )
        super(ParallelMaskNet, self).build(input_shape)

    def call(self,inputs):
        assert isinstance(inputs,list), 'ParallelMaskNet inputs must be a list'
        passive_input=inputs[1]
        
        # 对最后一个轴（特征维度）进行 LayerNormalization
        normalized_passive_input = self.layer_norm(passive_input)
        
        processed_passive_inputs= tf.reshape(normalized_passive_input, [tf.shape(normalized_passive_input)[0], -1])

        mask_output=[maskblock([inputs[0],processed_passive_inputs]) for maskblock in self.maskblock_list]
        out=tf.concat(mask_output,axis=-1)
        if self.mlp_hidden_list is not None:
            out=self.mlp(out)
        return out










