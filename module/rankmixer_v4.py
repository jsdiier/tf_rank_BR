import tensorflow as tf
from .activation import *
from tensorflow.keras import regularizers
from logger import logger


class MultiHeadTokenMixing(tf.keras.layers.Layer):
    def __init__(self, t, token_dim, num_heads=None, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.token_dim = token_dim
        self.num_heads = num_heads or t
        self.head_dim = token_dim // self.num_heads

    def call(self, x):
        # x: [B, T, D]
        B = tf.shape(x)[0]

        x = tf.reshape(x, [B, self.t, self.num_heads, self.head_dim])
        x = tf.transpose(x, [0, 2, 1, 3])      # [B, H, T, D/H]
        x = tf.reshape(x, [B, self.num_heads, self.t * self.head_dim])
        x = tf.reshape(x, [B, self.t, self.token_dim])
        return x


class Gelu_Expert(tf.keras.layers.Layer):
    def __init__(self, token_dim, hidden_dim, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense1 = tf.keras.layers.Dense(hidden_dim)
        self.gelu = GELU()
        self.dense2 = tf.keras.layers.Dense(token_dim)


    def call(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        return x



class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, token_dim, hidden_dim, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.gate_dense = tf.keras.layers.Dense(hidden_dim) # W: D * nD
        self.up_dense = tf.keras.layers.Dense(hidden_dim) # W: D * nD
        self.down = tf.keras.layers.Dense(token_dim)
        # According原论文： xavier初始化方法使残差连接稳定
        # self.down = tf.keras.layers.Dense(token_dim, kernel_initializer=ScaledGlorotUniform(scale=0.01)) # W: nD * D


    def call(self, x):
        gate_out = tf.keras.activations.swish(self.gate_dense(x)) # Swish激活 [B, T, hidden_dim]
        #logger.info("SwishFC-gate的输出形状:{}".format(gate_out.shape))
        up_out = self.up_dense(x) # [B, T, hidden_dim]
        #logger.info("FC-up的输出形状:{}".format(up_out.shape))
        down_out = self.down(gate_out * up_out) # Hadamar积 并映射到[B, T, D]
        return down_out


class PerTokenSwiGLU(tf.keras.layers.Layer):
    def __init__(self, t, token_dim, hidden_ratio=2, prefix='SwiGLU', **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.token_dim = token_dim
        self.hidden_dim = hidden_ratio * token_dim
        self.prefix=prefix
        self.swiglu= [
            SwiGLU(token_dim=self.token_dim, hidden_dim=self.hidden_dim,
                        name=self.prefix + "swiglu_{index}".format(index=i))
            for i in range(self.t)
        ]
        #self.ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def call(self, x):
        """
        x: [B, T, D]
        """
        outputs = []
        for i in range(self.t):
            token_i = x[:, i, :]           # [B, D]
            out_i = self.swiglu[i](token_i)  # [B, D]
            outputs.append(out_i)

        y = tf.stack(outputs, axis=1)      # [B, T, D]
        return y










class RankMixerBlock(tf.keras.layers.Layer):
    """
    Single RankMixer block: Token Mixing + Per-token
    """

    def __init__(self, t, token_dim, num_heads=None, num_experts=16, hidden_ratio=2, l1_coeff=1e-3, training=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.training = training
        self.token_mixing = MultiHeadTokenMixing(t, token_dim, num_heads)

        #self.per_token_moe = PerTokenDTSISparseMoE(t, token_dim, num_experts, hidden_ratio, l1_coeff, self.training)
        #self.per_token_ffn=PerTokenFFN(t, token_dim,hidden_ratio,num_experts)
        self.per_token_swiglu=PerTokenSwiGLU(t, token_dim, hidden_ratio)
        self.token_ln_0=tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

        self.token_ln=tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def call(self, x):
        ori_x=x
        x=self.token_ln_0(x)
        y = self.token_mixing(x)
        ori_y=ori_x+y
        x=self.token_ln(ori_y)
        x = self.per_token_swiglu(x, training=self.training)

        return x+ori_y


class RankMixer(tf.keras.layers.Layer):
    """
    Full RankMixer: multiple stacked RankMixerBlocks + mean pooling + prediction head
    """

    def __init__(self, t, token_dim, num_blocks=2, num_heads=None, num_experts=16, hidden_ratio=2, l1_coeff=0.1,
                 training=False, **kwargs):
        super().__init__(**kwargs)
        self.t=t
        self.training=training
        self.blocks = [
            RankMixerBlock(t, token_dim, num_heads, num_experts, hidden_ratio, l1_coeff, self.training)
            for _ in range(num_blocks)
        ]
        self.token_proj=tf.keras.layers.Dense(token_dim,activation=None,name="token_proj")
        # 添加参数日志输出
        logger.info("RankMixer initialized with parameters: t={}, token_dim={}, "
                    "num_blocks={}, num_heads={}, num_experts={}, "
                    "hidden_ratio={}, l1_coeff={}, training={}".format(
            t, token_dim, num_blocks, num_heads, num_experts,
            hidden_ratio, l1_coeff, training))


    def call(self, x): #【B，D】
        # logger.info("RankMixer call method executed")
        # 划分token,映射
        batch_size = tf.shape(x)[0]
        total_dim = x.shape[-1]
        assert total_dim % self.t == 0
        per_token_raw_dim = total_dim // self.t
        # [B, T, total_dim/T]
        tokens_raw = tf.reshape(
            x,
            [batch_size, self.t, per_token_raw_dim]
        )
        x = self.token_proj(tokens_raw)


        for block in self.blocks:
            x = block(x, training=self.training)
        # Mean pooling over tokens
        out = tf.reduce_mean(x, axis=1)  # [batch, token_dim]
        return out
