# GNNExplainer 官方 syn1：服务器运行记录与组会代码导读

实验日期：2026-10-11。已在 fy@10.16.18.199 上完成训练、单节点解释、诊断及输出。只完成第一阶段 GNNExplainer，尚未开展 PGExplainer。Bonnie668/gnn-explainability 未被当作算法代码库，也未修改。

## 1. 代码来源与实际环境

代码来源：[RexYing/gnn-model-explainer](https://github.com/RexYing/gnn-model-explainer/tree/bc984829f4f4829e93c760e9bbdc8e73f96e2cc1)，固定 commit `bc984829f4f4829e93c760e9bbdc8e73f96e2cc1`。下载该 commit 的 GitHub 源码 ZIP，保留原始副本及 SHA256。此处不是完整 Git clone，不含历史。

采用作者原始 `train.py`、`explainer_main.py`、`explainer/explain.py` 和 `models.py`。全部 28 个上游文件与归档内容逐字节一致，无算法补丁。没有调用 PyG 的 Explainer/GNNExplainer 高层 API。新增三个辅助 Python 文件只负责固定随机种子、选择样本和观察/导出结果。

服务器实验根目录：`/data/fy/gnnexplainer-official-syn1`。

| 路径（相对实验根目录） | 用途 |
|---|---|
| `original/` | 未修改的作者源码 |
| `source/` | 实际运行的源码；另有三个辅助脚本 |
| `env/` | 独立复制的 Python 环境，实际解释器与包路径均在这里 |
| `wheels/` | 离线安装依赖及带 SHA256 的下载清单 |
| `ckpt/syn1_base_h20_o20.pth.tar` | 本次训练得到的 GNN 参数及计算图 |
| `runs/train/` | 作者训练产生的 TensorBoard 事件 |
| `runs/explain/` | 首次原始解释器运行输出的 `.npy` |
| `runs/diagnostics/` | 损失、梯度、mask、节点映射、重要边与图 |
| `logs/` | 带命令的阶段日志、依赖快照和源码完整性验证 |

实测服务器：Linux 3.10.0-1160.108.1.el7.x86_64；2 × RTX 2080 Ti；驱动 510.54；`gnn-base` 为 Python 3.10.13、PyTorch 1.13.1+cu116、PyG 2.3.1，CUDA 可用。两张 GPU 有其他用户任务，故本次 CPU 运行，Torch/OMP/MKL 线程数限制为 2。

作者要求的 Python 3.7 / Torch 1.6.0 等旧依赖未直接安装。采用现有环境的独立文件副本，保留 PyTorch 1.13.1，添加 NetworkX 2.6.3、pandas 1.5.3、matplotlib 3.7.5、seaborn 0.12.2、tensorboardX 2.6.2.2、protobuf 3.20.3、opencv-python-headless 4.8.1.78。完整版本见 `logs/experiment-environment.txt`。`pip check` 通过。此路径不是作者历史软件栈的严格复现，而是作者代码在兼容环境中的实际运行。

原环境未执行安装、升级或卸载。安装前后的 `pip freeze` 完全相同，保存为 `gnn-base.before.txt` / `gnn-base.after.txt`。实验解释器的 sys.path 不指向 gnn-base。复制的 Torch 文件 inode 与原环境不同，非硬链接；只修正独立副本 `bin/` 内文本启动脚本的路径。后续请使用实验目录 `env/bin/python`，不要用基础环境的 pip。

## 2. 阶段命令、文件、输出与概念

以下 Linux 命令均在服务器执行；Windows 连接使用：

```powershell
ssh -o BatchMode=yes -i "$env:USERPROFILE\.ssh\gnn_server_ed25519" -p 22 fy@10.16.18.199
```

**阶段 A：检查与环境准备。**

实际先执行 `hostname; whoami; pwd; nvidia-smi`、`ls`、`df -h /data/fy`，并用基础环境 Python 导入 Torch/PyG 查询版本。初始输出为 `fy`、`/home/fy`、CUDA 可用、磁盘可用约 6.4 TB。可重复的只读检查保存在 `check.sh`；`logs/00-server-check.log` 是收尾时重复检查的记录。

创建目录和复制环境时执行：

```bash
mkdir -p /data/fy/gnnexplainer-official-syn1/logs
/data/fy/dl-research/envs/gnn-base/bin/python -m pip freeze > /data/fy/gnnexplainer-official-syn1/logs/gnn-base.before.txt
cp -a /data/fy/dl-research/envs/gnn-base /data/fy/gnnexplainer-official-syn1/env
bash /data/fy/gnnexplainer-official-syn1/prepare.sh
```

服务器直连 GitHub/PyPI 超时，因此由 Windows Python 下载固定源码 ZIP 和 Linux cp310 wheels，经 scp 上传；wheel 下载脚本校验 PyPI 提供的 SHA256。`prepare.sh` 解压源码、创建原始与运行副本，在副本里离线安装包，并执行 `train.py --help`。该脚本是首次安装记录，已安装后不应再次原样执行。

输出：`No broken requirements found.`；作者训练 CLI 可以导入和显示帮助。对应概念：建立可追踪的运行前提，本阶段尚未优化模型或解释。

**阶段 B：训练被解释的 GNN。**

```bash
cd /data/fy/gnnexplainer-official-syn1/source
export CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONHASHSEED=42 EXPERIMENT_SEED=42
../env/bin/python -u seeded_run.py train.py --dataset syn1 --epochs 1000 --ckptdir ../ckpt --logdir ../runs/train
```

已通过 `train.sh` 执行，命令与标准输出见 `logs/02-train.log`。`seeded_run.py` 设置 Python/NumPy/Torch 随机种子后，以 `runpy` 原样运行作者入口。网络为作者默认的三层 GCN，hidden/output dim 均为 20，10 维常数节点特征，4 个结构角色类别，不启用 BN。训练 Adam 默认 lr=0.001。

数据：300 个 BA 节点 + 80 个五节点 house，总计 700 节点，加入扰动后 2055 条无向边。训练/测试节点 560/140。这是同一张图上的节点划分；消息传播使用整张图，属于传导式节点分类。

检查点重新计算的准确率：训练 **99.4643%（557/560）**，测试 **97.1429%（136/140）**。训练日志里的最后一次逐轮指标来自该轮更新前的 logits，与保存前重新前向的指标可能不同；报告使用检查点中的最终 logits。

论文概念：先获得已训练的预测函数，再解释它。组会至少应解释 `train.py:264–301` 的节点划分、前向、交叉熵与模型参数更新，以及 `352–361` 保存的 `adj/feat/label/pred/train_idx` 为什么能供解释阶段使用。

**阶段 C：选目标节点并运行原始解释器。**

```bash
../env/bin/python select_node.py
../env/bin/python -u seeded_run.py explainer_main.py --dataset syn1 --explain-node 339 --epochs 100 --ckptdir ../ckpt --logdir ../runs/explain --no-writer
```

已通过 `explain.sh` 执行，日志 `logs/03-explain.log`。样本规则事先固定为“编号最小、预测正确的测试集 house 屋顶节点”，不根据解释分数挑选。得到节点 **339**，真实标签和原模型预测均为 **3**；原图目标类别概率 **0.887163**。

作者取三跳计算邻域，得到 **30 节点、48 条无向边**；全图节点 339 映射为局部节点 **29**。命令显式传 `--explain-node`，因此不会进入作者默认的批量解释分支。

100 轮 Adam，lr=0.1。首轮总损失 **7.170422**；第 99 轮更新前总损失 **0.748411**。输出为：

```text
runs/explain/masked_adj_syn1_base_h20_o20_explainnode_idx_339graph_idx_-1.npy
```

这是 30×30 的软加权邻接矩阵，不是离散边列表。`graph_idx=-1` 是入口参数默认值；在 batch 只有一张图时，-1 和 0 都指该图，不表示节点分类变成图分类。

论文概念：对一个实例的计算图做连续 mask 优化。必读 `explainer_main.py:257–258`、`explainer/explain.py:74–117`、`492–501`。

**阶段 D：观察优化过程、核验、导出。**

```bash
bash /data/fy/gnnexplainer-official-syn1/diagnostics.sh
```

主要执行 `../env/bin/python -u inspect_run.py`。该脚本用 Python trace 在原始 loss 返回时读取局部变量，用 autograd hook 记录梯度；不替换 loss、forward 或 optimizer。对相同 seed 重跑后，保存的软 mask 与阶段 C **逐元素完全相同**。

实际断言通过：优化器参数恰好只有边 mask 和特征 mask；GNN state_dict 全部不变；mask 确有更新；100 行损失/梯度数值均有限；28 个上游文件全部不变。没有为修正图像布局而更改任何算法；诊断脚本的绘图布局经过检查。

输出：`loss_trace.csv`（100 轮各项损失和两种梯度范数）、`edge_ranking.csv`（48 条真实无向边排序）、`masks_and_mapping.npz`（初始/最终 logits、特征 mask、全局节点映射、保存时与最终一步后的软邻接、top-6 二值图）、`summary.json`、`syn1_explanation.png`。

## 3. 组会核心代码路线

下面行号对应固定 commit，位于服务器 `/data/fy/gnnexplainer-official-syn1/source/`。算法文件未修改，所以也可在固定版本 GitHub 页面逐行核对。

| 问题 | 文件与行号 | 必须说清楚的内容 |
|---|---|---|
| 图和结构标签从何而来 | `gengraph.py:106–137`；`utils/synthetic_structsim.py:178–203` | BA 图接上 house；角色类别不是节点 ID |
| 解释谁 | `explainer_main.py:257–258` | 由 `--explain-node 339` 指定；样本筛选在辅助脚本 `select_node.py` |
| 为什么局部编号变了 | `explainer/explain.py:492–501` | neighbors 保存局部到全局映射；邻接和特征一起切片 |
| mask 的形状与初始化 | `explainer/explain.py:607–622,633–663` | 边 logits 是 30×30 Parameter；正态初始化均值 1、std=sqrt(2/30)；特征 logits 为长度10的零向量，sigmoid 后为0.5 |
| mask 怎样影响边 | `explainer/explain.py:665–678` | sigmoid、对称化、逐元素乘邻接、去掉对角线；本次 mask_bias=False，不增加不存在的边 |
| mask 怎样影响特征和预测 | `explainer/explain.py:685–715` | 同一长度10的特征 mask 广播到所有节点，随后送进原 GNN，只取目标节点的类别概率 |
| 掩码怎样影响消息传递 | `models.py:58–81`，尤其70–71 | `adj @ x` 后乘权重；软邻接改变邻居消息的权重 |
| 总 loss 到底有哪些项 | `explainer/explain.py:740–820`，尤其808 | 必须看最后相加的表达式，不能只看前面定义了什么变量 |
| 谁被更新 | `explainer/explain.py:613–622,135–146`；`utils/train_utils.py:7–10` | Adam 接收mask参数列表；backward求梯度，step更新mask；GNN未在此优化器中 |
| 输出如何得到 | `explainer/explain.py:207–221`；辅助 `inspect_run.py` | 原始输出软邻接；导出只选上三角真实边，排序后取top-6，并映射回全图ID |

最少应能读懂这条计算链：

```text
edge logits M (30×30)
  → sigmoid(M)
  → (sigmoid(M) + sigmoid(M).T) / 2
  → A ⊙ symmetric_mask（对角置零）
feature logits f (10)
  → X ⊙ sigmoid(f)
  → 原 GNN → 节点29的softmax概率 → loss.backward()
  → 只包含 M 和 f 的 Adam.step()
```

本次总目标严格按源码为：

```text
L = -log p(y_true)
    + 0.005 * sum(sigmoid(M))
    + mean(binary_entropy(sigmoid(M)))
    + mean(sigmoid(f))
    + y_hat^T (D - A_masked) y_hat / 900
```

其中 y_hat 是邻域内原模型预测的类别编号向量；最后一项是该实现的拉普拉斯项。size 让 mask 更稀疏，binary entropy 让软 mask 更接近0或1；二者作用不同。正则在完整30×30矩阵上计算，包含非边与对角位置，并非只对48条无向真实边计算。邻接乘法才会屏蔽非边。

| 已乘系数的损失项 | 第0轮（更新前） | 第99轮（更新前） |
|---|---:|---:|
| 预测损失 | 2.820485 | 0.202556 |
| 边 size | 3.258665 | 0.097307 |
| 边 entropy | 0.582505 | 0.056062 |
| 特征 size | 0.500000 | 0.381675 |
| Laplacian | 0.008766 | 0.010811 |
| 总和 | 7.170422 | 0.748411 |

边 mask 梯度范数从0.188131到0.020490，特征 mask 梯度范数从4.453788到1.068762。边 logits 最大绝对改变量为6.592366。这里不是只看到loss下降，而是同时确认梯度存在、参数改变、GNN权重不变。

## 4. 实际输出怎么解释

第99轮的日志概率为 **0.816641**，对应最后一次更新前的边和特征 mask。100次更新全部完成后重新前向，目标概率为 **0.826332**，预测仍为类别3。最终软边权总和/原邻接总和为 **0.133439**；这个density是权重比例，不能说成“恰好保留13.34%的边”。

作者保存的邻接矩阵来自最后一次更新前的forward缓存；与最后一次step后重算的邻接最大差0.001320。两种版本均保存在NPZ中，未悄悄修正作者输出。下面排名使用作者原始保存结果。

| 排名 | 全图边 | 分数 | 目标house内部边 |
|---|---|---:|---|
| 1 | 335—339 | 0.908726 | 是 |
| 2 | 335—336 | 0.897618 | 是 |
| 3 | 336—337 | 0.883958 | 是 |
| 4 | 336—339 | 0.878085 | 是 |
| 5 | 21—335 | 0.850619 | 否，连接BA图 |
| 6 | 335—338 | 0.847663 | 是 |
| 7 | 337—338 | 0.538868 | 是，house的第6条边 |

top-6是人为选择的可视化预算，因为一个house有6条边，不是算法自动学出的k。top-6恢复5/6条目标house边；不能声称完美找回house，更不能据此声称达到论文整体性能。第7名是缺失的house底边。

进一步检查二值top-6图：使用原始特征时，类别3概率0.417695；使用学到的特征mask时为0.795160，两种情况下argmax仍为3。说明本次解释共同依赖边与特征mask，不能把它讲成单纯“删除边”的实验。syn1特征为常数，因此特征维度排名本身没有业务语义。

## 5. 与论文目标对照时必须说明的实现细节

论文将解释看作紧凑子图/特征与模型预测之间的互信息优化；工程中通过连续mask与正则实现可微优化。参考：[论文](https://arxiv.org/abs/1903.03894)、[固定版本的解释器](https://github.com/RexYing/gnn-model-explainer/blob/bc984829f4f4829e93c760e9bbdc8e73f96e2cc1/explainer/explain.py)。

1. **损失用真实标签。** `pred_label_node` 被取出，但第752行实际索引 `pred[gt_label_node]`。本次选择预测正确节点，两种类别一致；推广到误分类节点时不能忽略差别。原始实现没有在本实验中修改。
2. **特征熵未加入总损失。** 第777行计算了它，第808行没有相加；日志的特征熵tag也误写成边熵。诊断CSV直接读取真正的局部变量，避免依赖错误tag。
3. **eval不是冻结梯度。** 作者代码可能仍计算GNN参数梯度，但解释器优化器没有GNN参数，因此权重不更新。已逐项核验；本次bn=False、dropout=0，子模块train/eval切换没有造成随机dropout行为。
4. **软mask不等于二值子图。** 排名、top-k/threshold、是否保留最大连通分量都是后处理选择。本次不额外裁剪连通分量，公开保留排序结果。
5. **最后一次更新与保存不同步。** loss、pred和保存的mask来自前向时刻，日志density则在step后重新计算；解读日志时应分清时点。

组会可以这样概述：“我在独立环境中运行了作者固定版本的syn1训练和单节点解释。对预测正确的测试节点339，在三跳计算图上优化30×30边mask与10维特征mask。mask通过邻接和输入特征进入原GNN；反向传播只由mask优化器更新解释参数。最后导出软边权及top-6子图，恢复5条house边，并观察到一条BA连接边。这个实验验证了算法执行链路，不是完整论文性能复现。”

## 6. 后续复跑

阅读现有结果不需重新训练。要重新解释339并保留原结果，请使用新log目录：

```bash
cd /data/fy/gnnexplainer-official-syn1/source
mkdir -p ../runs/manual-node339
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=42 EXPERIMENT_SEED=42 \
  ../env/bin/python seeded_run.py explainer_main.py --dataset syn1 --explain-node 339 --epochs 100 --ckptdir ../ckpt --logdir ../runs/manual-node339 --no-writer
```

报告、原始源码、辅助脚本、检查点、日志和结果另打包为 `GNNExplainer-syn1-evidence.zip`；归档不包含4.1GB环境和wheel二进制，但保留环境版本及wheel来源/哈希清单。服务器上环境与wheels均保留。
