# luban

鲁班（EVE）平台 hash 特征排序模型训练代码，模型为 `br_model_hash_v2`（RankMixer + buy/cat/click/ext 四塔）。

**做什么**：从 HDFS 读 GZIP TFRecord → 训练多任务模型 → 导出 serving 模型 / checkpoint → 可选训后评估。

**数据方式**：不走鲁班 `IterableTabularDataset`，与离线脚本一致（`get_files` + `utils.ReadTFRecordV2`）。

**平台适配**：用 `dmlite.train.tensorflow` 的 `load_model` / `save_model` 替代手写 Checkpoint。

---

## 训练

### 入口脚本

| 脚本 | 说明 |
|------|------|
| `train_hash_eve.py` | 仅训练 |
| `train_hash_eve_train_test.py` | 训练 + 可选训后测试 |
| `train_hash_eve_train_test_static.py` | 训练 + 可选训后测试 + TensorBoard |

### 常用参数

| 参数 | 说明 |
|------|------|
| `-data_path` | 训练数据 HDFS 根目录 |
| `--scheduleDt` | 调度日；训练日 = scheduleDt - 1 |
| `-start_day` / `-end_day` | 无 scheduleDt 时手动指定 |
| `--inputs` | 第 1 项为热启 checkpoint（可空） |
| `--outputs` | 模型输出目录（必填） |

### 训练行为

- `train_step` 使用 `@tf.function`
- `training=True`：hash 表 `fid_lookup_or_insert`，四任务 loss 加权反传
- 每个 epoch 结束 `save_model`，导出 serving（5 路输出）+ checkpoint

### 示例

```bash
python luban/train_hash_eve_train_test_static.py \
  -data_path hdfs://path/to/train \
  --scheduleDt 20260304 \
  --outputs hdfs://path/to/output
```

配置见 `model_conf.py`（`batch_size`、`epoch_num`、`learning_rate` 等）。

---

## 测试

### 两种方式

| 方式 | 脚本 | 说明 |
|------|------|------|
| 训后内存直测 | `train_hash_eve_train_test*.py` | 训练完直接用内存模型测，不 reload ckpt |
| 独立评估 | `test_hash_eve.py` | 从 checkpoint 加载后再测（`@tf.function`） |

### 测试参数

| 参数 | 说明 |
|------|------|
| `-enable_test` | `1` 开启训后测试 |
| `-test_data_path` | 测试数据路径 |
| `-test_start_day` / `-test_end_day` | 测试日期 |
| `-max_parts` | 每天最多读 N 个 part |

### 测试行为

- `set_training_mode(False)`：`training=False`，`is_save_model=False`
- TFRecord 特征 `SparseTensor` → `to_dense`
- 只 `fid_lookup`，不 insert、不反向传播
- 输出 AUC / GAUC / loss（buy、cat、click、ext 四任务）

### 示例

```bash
# 训后直测（挂在训练命令后）
-enable_test 1 \
-test_data_path hdfs://path/to/test \
-test_start_day 20260304 -test_end_day 20260304

# 独立评估
python luban/test_hash_eve.py \
  --inputs hdfs://path/to/ckpt \
  -test_data_path hdfs://path/to/test \
  -test_start_day 20260304 -test_end_day 20260304
```

---

## 可视化

仅 `train_hash_eve_train_test_static.py` 支持，通过 `dmlite.tracing` 接入 EVE TensorBoard。

```python
import dmlite.tracing as tracing
tracer = tracing.init()
# summary 写到 tracer.log_dir
```

### 记录指标

| 指标 | 含义 |
|------|------|
| `train/loss_buy` | buy loss |
| `train/loss_cat` | cat loss |
| `train/loss_click` | click loss |
| `train/loss_ext` | ext loss |
| `train/loss_total` | 四任务 loss 之和 |

- 横轴：`optimizer.iterations`（优化步数，跨 epoch 连续）
- 写在 Eager 侧，不参与 `train_step` 计算图，**不影响训练精度**
- 控制台 loss 日志默认每 1000 step 打印一次

在 EVE 实验页打开 TensorBoard 查看曲线。

---

## 版本迭代

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | - | 鲁班适配：`dmlite` load/save，HDFS + `ReadTFRecordV2` 读数 |
| v0.2 | - | `br_model_hash_v2`：RankMixer + 四任务塔 |
| v0.3 | - | `train_hash_eve_train_test.py`：训练 + 训后内存直测 |
| v0.4 | - | 测试修复：`training=False` 只 lookup；`is_save_model=False` 处理 SparseTensor |
| v0.5 | - | `train_hash_eve_train_test_static.py`：测试 `@tf.function` + TensorBoard |
| v0.6 | - | TensorBoard 写入改 Eager 侧；独占 GPU 去掉 `set_memory_growth` |

---

## 依赖

- TensorFlow 2.x
- 平台 `dmlite`（`dmlite.train.tensorflow`、`dmlite.tracing`）
