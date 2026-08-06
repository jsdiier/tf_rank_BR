import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import model_conf
import model_conf
from tensorflow.python.framework import sparse_tensor
from module.rankmixer_v4 import *
from logger import logger
from module.seq_attention import *


class InverseTimeDecayWithMin(tf.keras.optimizers.schedules.LearningRateSchedule):
    """InverseTimeDecay + 下限，避免长跑/快衰减后 lr 无限变小。"""

    def __init__(self, initial_learning_rate, decay_steps, decay_rate=1.0, min_learning_rate=1e-4,
                 staircase=False):
        super(InverseTimeDecayWithMin, self).__init__()
        self._schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
        )
        self.min_learning_rate = float(min_learning_rate)

    def __call__(self, step):
        return tf.maximum(self._schedule(step), self.min_learning_rate)

    def get_config(self):
        cfg = self._schedule.get_config()
        cfg['min_learning_rate'] = self.min_learning_rate
        return cfg


class Model(tf.keras.Model):
    def __init__(self, training=False, pred=False, fid_kv=None, fid_ads_kv=None, l2_reg=0.0001):
        super(Model, self).__init__()

        self.is_save_model = False
        self.training = training
        self.pred = pred
        self.dropout_dim = []
        self.use_bn = model_conf.use_bn

        self.lhuc_layers_cache = {}
        self.attention_layers_cache = {}
        self.bottom_layers_cache = {}
        self.task_layers_cache = {}
        self.ads_layers_cache = {}

        self.loss_bc = tf.keras.losses.binary_crossentropy
        # dense：Adam + schedule；sparse emb：Adagrad（更大 lr），照搬 EVE luban_v17 配方
        self.lr_schedule = InverseTimeDecayWithMin(
            initial_learning_rate=model_conf.learning_rate,
            decay_steps=model_conf.lr_decay_steps,
            decay_rate=model_conf.lr_decay_rate,
            min_learning_rate=model_conf.lr_min,
            staircase=False,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999,
                                                  epsilon=1e-07, amsgrad=False, name='AdamDense')
        self.optimizer_emb = tf.keras.optimizers.Adagrad(
            learning_rate=model_conf.emb_learning_rate, name='AdagradEmb')

        # embedding table
        self.emb_fm = tf.keras.layers.Embedding(
            model_conf.feature_size,
            model_conf.lr_emb_size + model_conf.fm_emb_size,
            embeddings_regularizer=regularizers.l2(model_conf.l2_reg))
        # self.emb_lr = tf.keras.layers.Embedding(
        #    model_conf.feature_size,
        #    model_conf.lr_emb_size,
        #    embeddings_regularizer=regularizers.l2(model_conf.l2_reg))
        self.emb_din_ads = tf.keras.layers.Embedding(
            model_conf.ads_fea_size,
            model_conf.din_emb_size,
            embeddings_regularizer=regularizers.l2(model_conf.l2_reg))

        self.slot_id_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(model_conf.all_slot_ids, dtype=tf.int32),
                values=tf.range(len(model_conf.all_slot_ids), dtype=tf.int32)
            ),
            default_value=-1
        )

        if fid_kv is not None:
            self.fid_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=fid_kv[0],
                    values=fid_kv[1]
                ),
                default_value=-1
            )
        else:
            self.fid_table = tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.int64,
                default_value=-1,
                empty_key=0,
                deleted_key=-1,
                initial_num_buckets=model_conf.num_buckets
            )
        self.counter = tf.Variable(1, dtype=tf.int64, trainable=False)

        self.slot_id_table_din_ads = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(model_conf.slot_id_v2, dtype=tf.int32),
                values=tf.range(len(model_conf.slot_id_v2), dtype=tf.int32)
            ),
            default_value=-1
        )
        if fid_ads_kv is not None:
            self.fid_table_din_ads = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=fid_ads_kv[0],
                    values=fid_ads_kv[1]
                ),
                default_value=-1
            )
        else:
            self.fid_table_din_ads = tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.int64,
                default_value=-1,
                empty_key=0,
                deleted_key=-1,
                initial_num_buckets=model_conf.ads_num_buckets
            )
        self.counter_din_ads = tf.Variable(1, dtype=tf.int64, trainable=False)

        # self.bn = tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=1e-3,center=True,scale=False)

        # 初始化底层主网络
        self.rankmixer = RankMixer(t=32, token_dim=768, num_heads=32, num_experts=32, hidden_ratio=2,
                                   training=self.training)
        # 初始化序列网络
        self.seq_click_attention_layer = DIN_attention_Layer([50, 20], 'sigmoid', name='global_click_seq')
        self.seq_pay_attention_layer = DIN_attention_Layer([50, 20], 'sigmoid', name='global_pay_seq')
        self.seq_12h_click_cate_id_attention_layer = DIN_attention_Layer([50, 20], 'sigmoid',
                                                                         name='12h_click_cate_id_seq')
        self.attention_layer_search_long_pay = DIN_attention_Layer([50, 20], 'sigmoid', name='search_pay_seq_long')
        self.attention_layer_search_long_clk = DIN_attention_Layer([50, 20], 'sigmoid', name='search_clk_seq_long')
        self.attention_layer_search_long_query = DIN_attention_Layer([50, 20], 'sigmoid', name='search_query_seq_long')

        # 搜索长序列：先融合多路 embedding，再与 DIN 注意力 + 均值池化残差组合，减轻「高维 concat 噪声」
        seq_token_dim = 32
        self.search_seq_token_dim = seq_token_dim
        seq_reg = regularizers.l2(model_conf.l2_reg)
        self.pay_seq_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.pay_seq_proj = tf.keras.layers.Dense(seq_token_dim, activation=tf.nn.swish, kernel_regularizer=seq_reg)
        self.pay_seq_combine = tf.keras.layers.Dense(seq_token_dim, activation=tf.nn.swish, kernel_regularizer=seq_reg)
        self.clk_seq_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.clk_seq_proj = tf.keras.layers.Dense(seq_token_dim, activation=tf.nn.swish, kernel_regularizer=seq_reg)
        self.clk_seq_combine = tf.keras.layers.Dense(seq_token_dim, activation=tf.nn.swish, kernel_regularizer=seq_reg)
        self.query_seq_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.query_seq_proj = tf.keras.layers.Dense(seq_token_dim, activation=tf.nn.swish, kernel_regularizer=seq_reg)
        self.query_seq_combine = tf.keras.layers.Dense(seq_token_dim, activation=tf.nn.swish,
                                                       kernel_regularizer=seq_reg)
        self.search_seq_dropout = tf.keras.layers.Dropout(0.1)

        # 初始化4个任务塔
        self.buy_tower = tf.keras.Sequential()
        for i, l in enumerate([256, 128]):
            if self.use_bn:
                self.buy_tower.add(tf.keras.layers.BatchNormalization())
            self.buy_tower.add(
                tf.keras.layers.Dense(l, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(model_conf.l2_reg)))

        self.cat_tower = tf.keras.Sequential()
        for i, l in enumerate([256, 128]):
            if self.use_bn:
                self.cat_tower.add(tf.keras.layers.BatchNormalization())
            self.cat_tower.add(
                tf.keras.layers.Dense(l, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(model_conf.l2_reg)))

        self.click_tower = tf.keras.Sequential()
        for i, l in enumerate([256, 128]):
            if self.use_bn:
                self.click_tower.add(tf.keras.layers.BatchNormalization())
            self.click_tower.add(
                tf.keras.layers.Dense(l, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(model_conf.l2_reg)))

        self.ext_tower = tf.keras.Sequential()
        for i, l in enumerate([256, 128]):
            if self.use_bn:
                self.ext_tower.add(tf.keras.layers.BatchNormalization())
            self.ext_tower.add(
                tf.keras.layers.Dense(l, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(l2_reg)))
        self.dense_concat = tf.keras.layers.Dense(1, activation="sigmoid",
                                                  kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat1 = tf.keras.layers.Dense(1, activation="sigmoid",
                                                   kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat2 = tf.keras.layers.Dense(1, activation="sigmoid",
                                                   kernel_regularizer=regularizers.l2(model_conf.l2_reg))
        self.dense_concat3 = tf.keras.layers.Dense(1, activation="sigmoid",
                                                   kernel_regularizer=regularizers.l2(model_conf.l2_reg))

    @staticmethod
    def _is_emb_var(var):
        name = var.name.lower()
        return (
            'emb_fm' in name
            or 'emb_din_ads' in name
            or 'emb_lr' in name
            or '/embeddings:' in name
            or name.endswith('/embeddings')
        )

    def split_trainable_weights(self):
        emb_vars, dense_vars = [], []
        for v in self.trainable_weights:
            if self._is_emb_var(v):
                emb_vars.append(v)
            else:
                dense_vars.append(v)
        return emb_vars, dense_vars

    def apply_gradients_dual(self, gradients, variables=None):
        """gradients 与 variables（默认 trainable_weights）一一对应；emb→Adagrad，其余→Adam。"""
        if variables is None:
            variables = self.trainable_weights
        emb_pairs, dense_pairs = [], []
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            if self._is_emb_var(var):
                emb_pairs.append((grad, var))
            else:
                dense_pairs.append((grad, var))
        if emb_pairs:
            self.optimizer_emb.apply_gradients(emb_pairs)
        if dense_pairs:
            self.optimizer.apply_gradients(dense_pairs)

    def warmup_optimizers(self):
        """构建两个 optimizer 的 slot，供 checkpoint restore 使用。"""
        emb_vars, dense_vars = self.split_trainable_weights()
        if dense_vars:
            self.optimizer.apply_gradients(
                zip([tf.zeros_like(v) for v in dense_vars], dense_vars))
        if emb_vars:
            self.optimizer_emb.apply_gradients(
                zip([tf.zeros_like(v) for v in emb_vars], emb_vars))

    def set_summary_writer(self, writer, histogram_freq=100):
        self.summary_writer = writer
        self.histogram_freq = histogram_freq

    def _write_histograms(self, step, gradients=None):
        with self.summary_writer.as_default():
            grad_map = {}
            if gradients is not None:
                for var, grad in zip(self.trainable_weights, gradients):
                    grad_map[var.name] = grad

            for var in self.trainable_weights:
                name_lower = var.name.lower()
                if any(k in name_lower for k in ["dense", "gate", "emb"]):
                    safe_name = var.name.replace(':', '_')
                    tf.summary.histogram(safe_name, var, step=step)
                    grad = grad_map.get(var.name)
                    if grad is not None:
                        tf.summary.histogram(safe_name + "_grad", grad, step=step)

    def transform(self, sids, fids):
        # # 调试打印：查看原始输入数据的形状
        # tf.print("DEBUG transform - sids type:", type(sids), output_stream=sys.stderr)
        # tf.print("DEBUG transform - fids type:", type(fids), output_stream=sys.stderr)
        # tf.print("DEBUG transform - sids shape:", tf.shape(sids) if hasattr(sids, 'shape') else 'N/A',
        #          output_stream=sys.stderr)
        # tf.print("DEBUG transform - fids shape:", tf.shape(fids) if hasattr(fids, 'shape') else 'N/A',
        #          output_stream=sys.stderr)

        if self.is_save_model:
            sid_list = tf.cast(sids, tf.dtypes.int32)
            fid_list = tf.cast(fids, tf.dtypes.int64)
        else:
            sid_list = tf.sparse.to_dense(sids)
            fid_list = tf.sparse.to_dense(fids)
            sid_list = tf.cast(sid_list, tf.dtypes.int32)

        # # 调试打印：查看转换后的数据形状
        # tf.print("DEBUG transform after - sid_list shape:", tf.shape(sid_list), output_stream=sys.stderr)
        # tf.print("DEBUG transform after - fid_list shape:", tf.shape(fid_list), output_stream=sys.stderr)
        # tf.print("DEBUG transform after - sid_list sample:", sid_list[0, :10], output_stream=sys.stderr)
        # tf.print("DEBUG transform after - fid_list sample:", fid_list[0, :10], output_stream=sys.stderr)

        return sid_list, fid_list

    def fid_lookup_or_insert(self, valid_fids, table_type):
        flat_ids = tf.reshape(valid_fids, [-1])
        unique_ids, idx_map = tf.unique(flat_ids)

        empty_key = 0
        valid_mask = tf.not_equal(unique_ids, empty_key)
        unique_ids = tf.boolean_mask(unique_ids, valid_mask)

        if table_type == 'din_ads_table':
            unique_mapped = self.fid_table_din_ads.lookup(unique_ids)
        else:
            unique_mapped = self.fid_table.lookup(unique_ids)

        mask = tf.equal(unique_mapped, -1)

        def insert_and_update():
            miss_indices = tf.where(mask)
            miss_ids = tf.gather_nd(unique_ids, miss_indices)
            num_new = tf.shape(miss_ids)[0]

            if table_type == 'din_ads_table':
                current_count = self.counter_din_ads.read_value()
                new_values = tf.range(current_count, current_count + tf.cast(num_new, tf.int64))
                self.counter_din_ads.assign_add(tf.cast(num_new, tf.int64))

                self.fid_table_din_ads.insert(miss_ids, new_values)
            else:
                current_count = self.counter.read_value()
                new_values = tf.range(current_count, current_count + tf.cast(num_new, tf.int64))
                self.counter.assign_add(tf.cast(num_new, tf.int64))

                self.fid_table.insert(miss_ids, new_values)

            return tf.tensor_scatter_nd_update(
                unique_mapped, miss_indices, new_values
            )

        final_unique_mapped = tf.cond(
            tf.reduce_any(mask),
            true_fn=insert_and_update,
            false_fn=lambda: unique_mapped
        )

        result = tf.gather(final_unique_mapped, idx_map)
        return tf.reshape(result, tf.shape(valid_fids))

    def fid_lookup(self, valid_fids, table_type):
        flat_ids = tf.reshape(valid_fids, [-1])

        if table_type == 'din_ads_table':
            mapped_ids = self.fid_table_din_ads.lookup(flat_ids)
        else:
            mapped_ids = self.fid_table.lookup(flat_ids)

        result = tf.reshape(mapped_ids, tf.shape(valid_fids))
        return result

    #
    def process_and_pool_fused(self, sid_list, fid_list, table_type='emb_table'):
        batch_size = tf.shape(sid_list)[0]
        n = tf.shape(sid_list)[1]
        sid_list_flat = tf.reshape(sid_list, [-1])
        fid_list_flat = tf.reshape(fid_list, [-1])

        if table_type == 'din_ads_table':
            num_segments = len(model_conf.slot_id_v2)
            mapped_indices = self.slot_id_table_din_ads.lookup(sid_list_flat)
        else:
            num_segments = len(model_conf.all_slot_ids)  # 特征个数
            mapped_indices = self.slot_id_table.lookup(sid_list_flat)

        mask = tf.not_equal(mapped_indices, -1)
        valid_fids = tf.boolean_mask(fid_list_flat, mask)
        valid_indices = tf.boolean_mask(mapped_indices, mask)

        batch_ids = tf.repeat(tf.range(batch_size), n)
        valid_batch_ids = tf.boolean_mask(batch_ids, mask)
        combined_indices = valid_batch_ids * num_segments + valid_indices

        # if self.is_save_model or self.pred:
        #     new_fid_list = self.fid_lookup(valid_fids, table_type)
        # else:
        #     new_fid_list = self.fid_lookup_or_insert(valid_fids, table_type)
        if self.training:
            new_fid_list = self.fid_lookup_or_insert(valid_fids, table_type)
        else:
            new_fid_list = self.fid_lookup(valid_fids, table_type)

        if table_type == 'din_ads_table':
            vocab_size = model_conf.ads_fea_size
        else:
            vocab_size = model_conf.feature_size

        valid_new_fids = (new_fid_list > 0) & (new_fid_list < vocab_size)
        new_fid_list = tf.boolean_mask(new_fid_list, valid_new_fids)
        combined_indices = tf.boolean_mask(combined_indices, valid_new_fids)

        if table_type == 'din_ads_table':
            embeds = self.emb_din_ads(new_fid_list)
        else:
            embeds = self.emb_fm(new_fid_list)
            # lr_embedding = self.emb_lr(new_fid_list)
            # embeds = tf.concat([lr_embedding, fm_embedding], axis=1)

        pooled_flat = tf.math.unsorted_segment_sum(
            data=embeds,
            segment_ids=combined_indices,
            num_segments=batch_size * num_segments
        )

        embedding_dim = tf.shape(embeds)[1]
        pooled_output = tf.reshape(pooled_flat, [batch_size, num_segments, embedding_dim])

        # slot mask
        ones = tf.ones_like(combined_indices, dtype=tf.float32)
        counts_flat = tf.math.unsorted_segment_sum(
            data=ones,
            segment_ids=combined_indices,
            num_segments=batch_size * num_segments
        )
        counts = tf.reshape(counts_flat, [batch_size, num_segments])
        slot_mask = tf.cast(tf.greater(counts, 0), dtype=tf.float32)

        return pooled_output, slot_mask

    def _search_seq_encode_pool_att(self, seq_raw, mask, seq_query, attention_layer, ln_layer, proj_layer,
                                    combine_layer):
        """LayerNorm + 线性投影 → DIN 注意力；与均值池化拼接后再压回固定维度，兼顾全局与候选相关片段。"""
        x = ln_layer(seq_raw)
        x = proj_layer(x)
        x = self.search_seq_dropout(x, training=self.training)
        len_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
        pool = tf.reduce_sum(x * tf.expand_dims(mask, -1), axis=1) / (len_sum + 1e-8)
        att = attention_layer([seq_query, x, x, mask])
        return combine_layer(tf.concat([att, pool], axis=-1))

    def ads_seq_cross_layer(self, name, nn_inputs, ads_emb, ads_hidden_dim=64, ads_output_dim=1):
        # ads_input_dim = nn_inputs.get_shape().as_list()[-1]
        ads_input_dim = tf.shape(nn_inputs)[-1]

        layer_key = name

        if layer_key not in self.ads_layers_cache:
            ads_weight_dims = [ads_hidden_dim, ads_input_dim * ads_output_dim]
            ads_weight_acts = [tf.nn.relu, None]
            ads_weight_layers = []

            for i in range(len(ads_weight_dims)):
                pdim = ads_weight_dims[i]
                pact = ads_weight_acts[i]
                layer = tf.keras.layers.Dense(
                    units=pdim,
                    activation=pact,
                    name=name + '_ads_weight_' + str(i)
                )
                ads_weight_layers.append(layer)

            ads_bias_dims = [ads_hidden_dim, ads_output_dim]
            ads_bias_layers = []
            for i in range(len(ads_bias_dims)):
                pdim = ads_bias_dims[i]
                layer = tf.keras.layers.Dense(
                    units=pdim,
                    activation=tf.nn.relu,
                    name=name + '_ads_bias_' + str(i)
                )
                ads_bias_layers.append(layer)

            self.ads_layers_cache[layer_key] = {
                'weight_layers': ads_weight_layers,
                'bias_layers': ads_bias_layers
            }

        cached_layers = self.ads_layers_cache[layer_key]
        ads_weight_layers = cached_layers['weight_layers']
        ads_bias_layers = cached_layers['bias_layers']

        ads_weight = ads_emb
        for i, layer in enumerate(ads_weight_layers):
            ads_weight = layer(ads_weight)
        ads_weight = tf.reshape(ads_weight, [-1, ads_input_dim, ads_output_dim])

        ads_bias = ads_emb
        for i, layer in enumerate(ads_bias_layers):
            ads_bias = layer(ads_bias)
        ads_bias = tf.expand_dims(ads_bias, 1)

        output = tf.matmul(nn_inputs, ads_weight)
        output = tf.add(output, ads_bias)

        return output

    def attention_din_ads(self, query, key, mask, ads_emb, name, att_hidden_units):
        ads_key = self.ads_seq_cross_layer(name=name + '_ads_layer',
                                           nn_inputs=key,
                                           ads_emb=ads_emb,
                                           ads_hidden_dim=64,
                                           ads_output_dim=tf.shape(key)[-1])

        query_dim = tf.shape(query)[-1]
        query = tf.tile(query, multiples=[1, ads_key.shape[1]])
        query = tf.reshape(query, shape=[-1, ads_key.shape[1], ads_key.shape[2]])

        din_all_output = tf.concat([query, ads_key, query - ads_key, query * ads_key], axis=-1)

        att_hidden_units = att_hidden_units + [query_dim]

        att_layer_key = name

        if att_layer_key not in self.attention_layers_cache:
            att_layers = []
            for i in range(len(att_hidden_units)):
                att_dim = att_hidden_units[i]
                layer = tf.keras.layers.Dense(
                    units=att_dim,
                    activation=tf.nn.relu,
                    name=name + '_tower_' + str(i)
                )
                att_layers.append(layer)
            self.attention_layers_cache[att_layer_key] = att_layers

        att_layers = self.attention_layers_cache[att_layer_key]
        for i, layer in enumerate(att_layers):
            din_all_output = layer(din_all_output)

        key_dim = tf.shape(ads_key)[-1]
        key_dim_float = tf.cast(key_dim, tf.float32)
        scores = din_all_output / (key_dim_float ** 0.5)
        scores = tf.nn.sigmoid(scores)
        outputs = scores * tf.expand_dims(tf.cast(mask, din_all_output.dtype), 2)
        weighted_sum = outputs * ads_key
        weighted_sum = tf.reduce_sum(weighted_sum, axis=1)

        return weighted_sum

    def call(self, inputs, training=None):
        sids, fids = inputs
        step = self.optimizer.iterations

        sid_list, fid_list = self.transform(sids, fids)

        pooled_output, slot_mask = self.process_and_pool_fused(sid_list, fid_list)

        # lr part
        lr_indices = self.slot_id_table.lookup(tf.constant(model_conf.lr_slot_ids, dtype=tf.dtypes.int32))
        lr_emb = tf.gather(pooled_output[:, :, 0], lr_indices, axis=1)
        lr = tf.reduce_sum(lr_emb, axis=1, keepdims=True)

        # fm part
        full_emb = pooled_output[:, :, 1:]
        square_sum_fm_embedding = tf.math.square(tf.reduce_sum(full_emb, 1))
        sum_square_fm_embedding = tf.reduce_sum(tf.math.square(full_emb), 1)
        fm = 0.5 * tf.math.subtract(square_sum_fm_embedding, sum_square_fm_embedding)

        # #embedding part
        # emb_slot_indices = self.slot_id_table.lookup(tf.constant(model_conf.embedding_slot_ids, dtype=tf.dtypes.int32))
        # all_emb = tf.gather(pooled_output, emb_slot_indices, axis=1)
        # all_emb = tf.reshape(all_emb, [tf.shape(all_emb)[0], -1])

        # 获取user_emb
        emb_user_indices = self.slot_id_table.lookup(tf.constant(model_conf.user_fea_list, dtype=tf.dtypes.int32))
        emb_user = tf.gather(pooled_output[:, :, 1:], emb_user_indices, axis=1)
        emb_user = tf.reshape(
            emb_user,
            [tf.shape(emb_user)[0], len(model_conf.user_fea_list) * model_conf.fm_emb_size])
        logger.info("emb_user shape is {}".format(emb_user.shape))

        # 获取shop_emb
        emb_shop_indices = self.slot_id_table.lookup(tf.constant(model_conf.shop_fea_list, dtype=tf.dtypes.int32))
        emb_shop = tf.gather(pooled_output[:, :, 1:], emb_shop_indices, axis=1)
        emb_shop = tf.reshape(
            emb_shop,
            [tf.shape(emb_shop)[0], len(model_conf.shop_fea_list) * model_conf.fm_emb_size])
        logger.info("emb_shop shape is {}".format(emb_shop.shape))

        # 获取interact_emb
        emb_interact_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.interact_fea_list, dtype=tf.dtypes.int32))
        emb_interact = tf.gather(pooled_output[:, :, 1:], emb_interact_indices, axis=1)
        emb_interact = tf.reshape(
            emb_interact,
            [tf.shape(emb_interact)[0], len(model_conf.interact_fea_list) * model_conf.fm_emb_size])
        logger.info("emb_interact shape is {}".format(emb_interact.shape))

        # 获取sequence_emb

        # 获取全局点击，支付sid 序列query-->sid
        global_query_slot_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.global_seq_query_sids, dtype=tf.dtypes.int32))
        global_query_input = tf.gather(pooled_output[:, :, 1:], global_query_slot_indices, axis=1)
        global_query_input = tf.reshape(
            global_query_input,
            [tf.shape(global_query_input)[0], len(model_conf.global_seq_query_sids) * model_conf.fm_emb_size])

        # pooled_output_v2, slot_mask_v2 = self.process_and_pool_fused(sid_list, fid_list, table_type='din_ads_table')
        # ads_slot_indices = self.slot_id_table_din_ads.lookup(tf.constant(model_conf.ads_fea_slots, dtype=tf.dtypes.int32))
        # ads_emb = tf.gather(pooled_output_v2, ads_slot_indices, axis=1)
        # ads_emb = tf.reshape(ads_emb, [tf.shape(ads_emb)[0], -1])

        seq_outputs = []
        for seq_name, seq_sid_ids in model_conf.seq_slot_dict.items():
            seq_slot_indices = self.slot_id_table.lookup(tf.constant(seq_sid_ids, dtype=tf.dtypes.int32))
            seq_input = tf.gather(pooled_output[:, :, 1:], seq_slot_indices, axis=1)
            seq_mask = tf.gather(slot_mask, seq_slot_indices, axis=1)
            if seq_name == 'user_click_seq':
                seq_output = self.seq_click_attention_layer([global_query_input, seq_input, seq_input, seq_mask])
            elif seq_name == 'user_pay_seq':
                seq_output = self.seq_pay_attention_layer([global_query_input, seq_input, seq_input, seq_mask])
            elif seq_name == 'user_12h_click_cateid':
                seq_output = self.seq_12h_click_cate_id_attention_layer(
                    [global_query_input, seq_input, seq_input, seq_mask])
            seq_outputs.append(seq_output)

        # 搜索支付序列
        seq_slot_indices = self.slot_id_table.lookup(tf.constant(model_conf.search_long_pay_seq, dtype=tf.dtypes.int32))
        search_pay_seq_input = tf.gather(pooled_output[:, :, 1:], seq_slot_indices, axis=1)
        search_pay_seq_mask = tf.gather(slot_mask, seq_slot_indices, axis=1)

        seq_slot_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.search_long_pay_catel3_seq, dtype=tf.dtypes.int32))
        search_pay_catel3_seq_input = tf.gather(pooled_output[:, :, 1:], seq_slot_indices, axis=1)
        # search_pay_catel3_seq_mask = tf.gather(slot_mask, seq_slot_indices, axis=1)

        pay_search_long_seq = tf.concat([search_pay_seq_input, search_pay_catel3_seq_input], axis=-1)
        pay_search_long_seq_out = self._search_seq_encode_pool_att(
            pay_search_long_seq, search_pay_seq_mask, emb_shop,
            self.attention_layer_search_long_pay,
            self.pay_seq_ln, self.pay_seq_proj, self.pay_seq_combine)
        seq_outputs.append(pay_search_long_seq_out)

        # 搜索点击序列
        seq_slot_indices = self.slot_id_table.lookup(tf.constant(model_conf.search_long_clk_seq, dtype=tf.dtypes.int32))
        click_seq_input = tf.gather(pooled_output[:, :, 1:], seq_slot_indices, axis=1)
        click_seq_mask = tf.gather(slot_mask, seq_slot_indices, axis=1)

        seq_slot_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.search_long_clk_catel3_seq, dtype=tf.dtypes.int32))
        search_clk_catel3_seq_input = tf.gather(pooled_output[:, :, 1:], seq_slot_indices, axis=1)
        click_search_long_seq = tf.concat([click_seq_input, search_clk_catel3_seq_input], axis=-1)
        clk_search_long_seq_out = self._search_seq_encode_pool_att(
            click_search_long_seq, click_seq_mask, emb_shop,
            self.attention_layer_search_long_clk,
            self.clk_seq_ln, self.clk_seq_proj, self.clk_seq_combine)
        seq_outputs.append(clk_search_long_seq_out)

        # 搜索query序列
        seq_slot_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.search_long_query_catel3_seq, dtype=tf.dtypes.int32))
        query_seq_input = tf.gather(pooled_output[:, :, 1:], seq_slot_indices, axis=1)
        query_cate_mask = tf.gather(slot_mask, seq_slot_indices, axis=1)
        query_search_long_seq_out = self._search_seq_encode_pool_att(
            query_seq_input, query_cate_mask, emb_shop,
            self.attention_layer_search_long_query,
            self.query_seq_ln, self.query_seq_proj, self.query_seq_combine)
        seq_outputs.append(query_search_long_seq_out)

        deep = tf.concat([emb_user, emb_shop, emb_interact] + seq_outputs, axis=-1)

        # 余数补dims
        remain_dims = deep.shape[-1] % (self.rankmixer.t)
        additional_dims = tf.zeros([tf.shape(deep)[0], self.rankmixer.t - remain_dims])
        deep_input = tf.concat([deep, additional_dims], axis=1)
        rankmixer_output = self.rankmixer(deep_input)

        concat = tf.concat([lr, fm, rankmixer_output], axis=1)

        buy_tower_output = self.buy_tower(concat, training=self.training)
        cat_tower_output = self.cat_tower(concat, training=self.training)
        click_tower_output = self.click_tower(concat, training=self.training)
        ext_tower_output = self.ext_tower(concat, training=self.training)

        cvr_pred_org = self.dense_concat(buy_tower_output)
        cat_pred_org = self.dense_concat1(cat_tower_output)
        click_pred = self.dense_concat2(click_tower_output)
        ext_pred = self.dense_concat3(ext_tower_output)

        cat_pred = cat_pred_org

        ctcvr = tf.math.multiply(click_pred, cvr_pred_org)

        if self.is_save_model or self.pred:
            final_pred = ctcvr
            cvr_score = cvr_pred_org
            ctr_score = click_pred
            cat_score = cat_pred_org
            ext_score = ext_pred
            return final_pred, cvr_score, ctr_score, cat_score, ext_score

        return ctcvr, cat_pred, click_pred, ext_pred

