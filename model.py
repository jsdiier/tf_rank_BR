import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import model_conf
import model_conf
from tensorflow.python.framework import sparse_tensor
from module.rankmixer_v4 import *
from module.onetrans_lite import OneTransLite
from logger import logger
from module.seq_attention import *


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
        self.lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(model_conf.learning_rate, decay_steps=1000000,
                                                                          decay_rate=1, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999,
                                                  epsilon=1e-07, amsgrad=False, name='Adam')

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

        # OneTrans-lite unifies the six event sequences and ordinary feature interaction.
        self.onetrans_lite = OneTransLite(
            sequence_input_dims=[8, 8, 8, 16, 16, 8],
            non_sequence_slot_count=len(model_conf.onetrans_flat_slot_ids),
            embedding_dim=model_conf.fm_emb_size,
            token_dim=model_conf.onetrans_token_dim,
            non_sequence_token_count=model_conf.onetrans_non_sequence_token_count,
            num_layers=model_conf.onetrans_num_layers,
            num_heads=model_conf.onetrans_num_heads,
            ffn_dim=model_conf.onetrans_ffn_dim,
            name='onetrans_lite')
        # 初始化4个任务塔
        self.buy_tower = tf.keras.Sequential()
        for i, l in enumerate([256]):
            if self.use_bn:
                self.buy_tower.add(tf.keras.layers.BatchNormalization())
            self.buy_tower.add(
                tf.keras.layers.Dense(l, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(model_conf.l2_reg)))

        self.cat_tower = tf.keras.Sequential()
        for i, l in enumerate([256]):
            if self.use_bn:
                self.cat_tower.add(tf.keras.layers.BatchNormalization())
            self.cat_tower.add(
                tf.keras.layers.Dense(l, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(model_conf.l2_reg)))

        self.click_tower = tf.keras.Sequential()
        for i, l in enumerate([256]):
            if self.use_bn:
                self.click_tower.add(tf.keras.layers.BatchNormalization())
            self.click_tower.add(
                tf.keras.layers.Dense(l, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(model_conf.l2_reg)))

        self.ext_tower = tf.keras.Sequential()
        for i, l in enumerate([256]):
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

        def gather_sequence(slot_ids):
            indices = self.slot_id_table.lookup(tf.constant(slot_ids, dtype=tf.int32))
            return (tf.gather(pooled_output[:, :, 1:], indices, axis=1),
                    tf.gather(slot_mask, indices, axis=1))

        click_events, click_mask = gather_sequence(model_conf.user_click_seq)
        pay_events, pay_mask = gather_sequence(model_conf.user_pay_seq)
        category_events, category_mask = gather_sequence(model_conf.u_12h_click_cateIds)
        search_pay_shop, search_pay_mask = gather_sequence(model_conf.search_long_pay_seq)
        search_pay_category, _ = gather_sequence(model_conf.search_long_pay_catel3_seq)
        search_click_shop, search_click_mask = gather_sequence(model_conf.search_long_clk_seq)
        search_click_category, _ = gather_sequence(model_conf.search_long_clk_catel3_seq)
        search_query_events, search_query_mask = gather_sequence(model_conf.search_long_query_catel3_seq)

        sequence_inputs = [
            click_events,
            pay_events,
            category_events,
            tf.concat([search_pay_shop, search_pay_category], axis=-1),
            tf.concat([search_click_shop, search_click_category], axis=-1),
            search_query_events,
        ]
        sequence_masks = [
            click_mask, pay_mask, category_mask, search_pay_mask,
            search_click_mask, search_query_mask,
        ]
        ordinary_indices = self.slot_id_table.lookup(
            tf.constant(model_conf.onetrans_flat_slot_ids, dtype=tf.int32))
        ordinary_embeddings = tf.gather(
            pooled_output[:, :, 1:], ordinary_indices, axis=1)
        onetrans_output = self.onetrans_lite(
            sequence_inputs,
            sequence_masks,
            ordinary_embeddings,
            training=self.training)

        concat = tf.concat([lr, fm, onetrans_output], axis=1)

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
