"""
@function: 激活函数
@author: jiazhuo
"""
import tensorflow as tf
import math

class GELU(tf.keras.layers.Layer):
    """
    GELU activation (BERT style)
    Compatible with TF 2.3
    """
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, inputs):
        return 0.5 * inputs * (1.0 + tf.tanh(
            math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))
        ))

    def get_config(self):
        base_config = super(GELU, self).get_config()
        return base_config
