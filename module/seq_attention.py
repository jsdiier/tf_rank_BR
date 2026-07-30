import tensorflow as tf
from tensorflow.keras import regularizers
class DIN_attention_Layer(tf.keras.layers.Layer):
    def __init__(self, att_hidden_units, activation='relu', name=''):
        super(DIN_attention_Layer, self).__init__()
        self.name_prefix = name

        self.att_dense = [tf.keras.layers.Dense(unit, activation=activation, name=f'{name}_att_dense_{i}') for i, unit
                          in enumerate(att_hidden_units)]
        self.att_final_dense = tf.keras.layers.Dense(1, name=f'{name}_att_final_dense')
        self.const_min = -4294967295
        self.q_dense = tf.keras.layers.Dense(32, name=f'{name}_q_dense')
        self.k_dense = tf.keras.layers.Dense(32, name=f'{name}_k_dense')
        self.v_dense = tf.keras.layers.Dense(32, name=f'{name}_v_dense')

    def call(self, inputs):
        q, k, v, mask = inputs
        q = tf.expand_dims(q, axis=1)
        q = tf.tile(q, multiples=[1, tf.shape(k)[1], 1])

        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)

        info = tf.concat([q, k, q - k, q * k], axis=-1)

        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)
        outputs = tf.squeeze(outputs, axis=-1)

        paddings = tf.ones_like(outputs) * self.const_min
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)

        outputs = tf.expand_dims(tf.nn.softmax(logits=outputs), axis=1)
        outputs = tf.squeeze(tf.matmul(outputs, v), axis=1)

        return outputs
