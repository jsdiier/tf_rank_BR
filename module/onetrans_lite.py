import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', shape=[int(input_shape[-1])], initializer='ones')

    def call(self, inputs):
        variance = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.epsilon) * self.scale


class OneTransLiteBlock(tf.keras.layers.Layer):
    def __init__(self, token_dim, num_heads, ffn_dim, **kwargs):
        super().__init__(**kwargs)
        self.attention_norm = RMSNorm(name='attention_norm')
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=token_dim // num_heads,
            output_shape=token_dim,
            name='causal_attention')
        self.ffn_norm = RMSNorm(name='ffn_norm')
        self.ffn_up = tf.keras.layers.Dense(ffn_dim, activation=tf.nn.gelu, name='ffn_up')
        self.ffn_down = tf.keras.layers.Dense(token_dim, name='ffn_down')

    def call(self, inputs, attention_mask, training=None):
        normalized = self.attention_norm(inputs)
        attention_output = self.attention(
            normalized,
            normalized,
            attention_mask=attention_mask,
            training=training)
        hidden = inputs + attention_output
        normalized = self.ffn_norm(hidden)
        return hidden + self.ffn_down(self.ffn_up(normalized))


class OneTransLite(tf.keras.layers.Layer):
    """Unified event-sequence and non-sequential feature interaction backbone."""

    def __init__(self, sequence_input_dims, non_sequence_slot_count, embedding_dim,
                 token_dim=128, non_sequence_token_count=16, num_layers=2,
                 num_heads=4, ffn_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.sequence_count = len(sequence_input_dims)
        self.non_sequence_slot_count = non_sequence_slot_count
        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.non_sequence_token_count = non_sequence_token_count

        self.sequence_projections = [
            tf.keras.layers.Dense(token_dim, name='sequence_projection_{}'.format(index))
            for index, _ in enumerate(sequence_input_dims)
        ]
        self.non_sequence_projection = tf.keras.layers.Dense(
            non_sequence_token_count * token_dim,
            name='non_sequence_auto_split')
        self.blocks = [
            OneTransLiteBlock(token_dim, num_heads, ffn_dim,
                              name='onetrans_lite_block_{}'.format(index))
            for index in range(num_layers)
        ]

    def build(self, input_shape):
        self.separator_tokens = self.add_weight(
            name='separator_tokens',
            shape=[self.sequence_count - 1, self.token_dim],
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.sequence_type_embeddings = self.add_weight(
            name='sequence_type_embeddings',
            shape=[self.sequence_count, self.token_dim],
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    @staticmethod
    def _causal_valid_mask(valid_mask):
        token_count = tf.shape(valid_mask)[1]
        causal = tf.linalg.band_part(
            tf.ones([token_count, token_count], dtype=tf.bool), -1, 0)
        query_valid = valid_mask[:, :, None]
        key_valid = valid_mask[:, None, :]
        return causal[None, :, :] & query_valid & key_valid

    def call(self, sequence_inputs, sequence_masks, non_sequence_embeddings, training=None):
        sequence_parts = []
        mask_parts = []
        batch_size = tf.shape(non_sequence_embeddings)[0]

        for index, (events, event_mask, projection) in enumerate(zip(
                sequence_inputs, sequence_masks, self.sequence_projections)):
            tokens = projection(events)
            tokens = tokens + self.sequence_type_embeddings[index][None, None, :]
            sequence_parts.append(tokens)
            mask_parts.append(tf.cast(event_mask, tf.bool))
            if index < self.sequence_count - 1:
                separator = tf.tile(
                    self.separator_tokens[index][None, None, :], [batch_size, 1, 1])
                sequence_parts.append(separator)
                mask_parts.append(tf.ones([batch_size, 1], dtype=tf.bool))

        sequence_tokens = tf.concat(sequence_parts, axis=1)
        sequence_valid = tf.concat(mask_parts, axis=1)

        flat_non_sequence = tf.reshape(
            non_sequence_embeddings,
            [batch_size, self.non_sequence_slot_count * self.embedding_dim])
        non_sequence_tokens = self.non_sequence_projection(flat_non_sequence)
        non_sequence_tokens = tf.reshape(
            non_sequence_tokens,
            [batch_size, self.non_sequence_token_count, self.token_dim])
        non_sequence_valid = tf.ones(
            [batch_size, self.non_sequence_token_count], dtype=tf.bool)

        tokens = tf.concat([sequence_tokens, non_sequence_tokens], axis=1)
        valid_mask = tf.concat([sequence_valid, non_sequence_valid], axis=1)
        attention_mask = self._causal_valid_mask(valid_mask)
        for block in self.blocks:
            tokens = block(tokens, attention_mask=attention_mask, training=training)

        return tf.reduce_mean(tokens[:, -self.non_sequence_token_count:, :], axis=1)
