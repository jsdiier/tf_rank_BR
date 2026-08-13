import tensorflow as tf


class SemanticQueryInitializer(tf.keras.layers.Layer):
    def __init__(self, query_count, semantic_count, dim, **kwargs):
        super().__init__(**kwargs)
        self.query_count = query_count
        self.semantic_count = semantic_count
        self.dim = dim
        self.context_projection = tf.keras.layers.Dense(dim, use_bias=False)
        self.time_projection = tf.keras.layers.Dense(dim, use_bias=False)

    def build(self, input_shape):
        self.query_seed = self.add_weight('query_seed', [self.query_count, self.dim],
                                          initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.route_logits = self.add_weight('route_logits', [self.query_count, self.semantic_count],
                                            initializer='zeros')

    def call(self, inputs):
        semantic_tokens, time_context = inputs
        route = tf.nn.softmax(self.route_logits, axis=-1)
        context = tf.einsum('qk,bkd->bqd', route, semantic_tokens)
        seed = tf.tile(self.query_seed[None, :, :], [tf.shape(semantic_tokens)[0], 1, 1])
        time = self.time_projection(time_context)[:, None, :]
        return seed + self.context_projection(context) + 0.1 * time


class HierarchicalWindowAttention(tf.keras.layers.Layer):
    def __init__(self, query_count, semantic_count, input_dim, dim=256, window_size=5,
                 num_heads=4, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.query_count = query_count
        self.dim = dim
        self.window_size = window_size
        self.num_layers = num_layers
        self.event_projection = tf.keras.layers.Dense(dim)
        self.query_initializer = SemanticQueryInitializer(query_count, semantic_count, dim)
        self.local_attentions = [tf.keras.layers.MultiHeadAttention(num_heads, dim // num_heads)
                                 for _ in range(num_layers)]
        self.global_attentions = [tf.keras.layers.MultiHeadAttention(num_heads, dim // num_heads)
                                  for _ in range(num_layers)]
        self.query_norms = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(num_layers)]

    def _windowize(self, events, mask):
        batch = tf.shape(events)[0]
        length = tf.shape(events)[1]
        pad = tf.math.mod(-length, self.window_size)
        events = tf.pad(events, [[0, 0], [0, pad], [0, 0]])
        mask = tf.pad(tf.cast(mask, tf.bool), [[0, 0], [0, pad]])
        windows = tf.reshape(events, [batch, -1, self.window_size, self.dim])
        window_mask = tf.reshape(mask, [batch, -1, self.window_size])
        return windows, window_mask

    def call(self, inputs):
        events, mask, semantic_tokens, time_context = inputs
        events = self.event_projection(events)
        queries = self.query_initializer([semantic_tokens, time_context])
        states = []
        for local_attention, global_attention, query_norm in zip(
                self.local_attentions, self.global_attentions, self.query_norms):
            windows, window_mask = self._windowize(events, mask)
            batch = tf.shape(windows)[0]
            num_windows = tf.shape(windows)[1]
            flat_windows = tf.reshape(windows, [batch * num_windows, self.window_size, self.dim])
            flat_mask = tf.reshape(window_mask, [batch * num_windows, self.window_size])
            mean_query = tf.reduce_mean(queries, axis=1, keepdims=True)
            local_query = tf.tile(mean_query[:, None, :, :], [1, num_windows, 1, 1])
            local_query = tf.reshape(local_query, [batch * num_windows, 1, self.dim])
            local_mask = flat_mask[:, None, :]
            representatives = local_attention(local_query, flat_windows, attention_mask=local_mask)
            representatives = tf.reshape(representatives, [batch, num_windows, self.dim])
            valid_windows = tf.reduce_any(window_mask, axis=-1)
            delta = global_attention(queries, representatives, attention_mask=valid_windows[:, None, :])
            queries = query_norm(queries + delta)
            states.append(queries)
        return tf.add_n(states) / float(len(states))
