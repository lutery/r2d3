# #codebase 结合代码分别讲解一下reward_multisteps、lstm_ful_input_length、input_sequence、burnin_length这几个变量的含义、作用和使用场景
下面结合你仓库里的实现（主要在 r2d3.py 与 env_play.py / rainbow.py）分别说明 `reward_multisteps`、`lstm_ful_input_length`、`input_sequence`、`burnin_length` 的**含义 / 作用 / 使用场景**，以及它们之间如何配合。

---

## 0. 先给一个总览：这四个量在“序列经验”里怎么排布

在 **Stateful LSTM** 路线下，你的实现本质是在存一段更长的序列，用于：

- 前面 `burnin_length` 步：只“烧入/预热”LSTM 状态（不算 loss）
- 后面训练用的 `lstm_ful_input_length` 步：真正做 `train_on_batch`
- 同时每个训练步的 TD target 还要用到 `reward_multisteps` 步后的 bootstrap Q 值
- `input_sequence` 则决定“每个时间点的网络输入 state”是由多少帧/多少步观测堆叠成的

你可以把每个 time step 的输入理解成：`state_t = [obs_{t-input_sequence+1} ... obs_t]`。

---

## 1. `input_sequence`：每次喂给网络的“观测堆叠长度”（输入序列长度）

### 含义
`input_sequence` 决定模型输入张量里“时间/帧堆叠”这一维的长度。它在你项目里既承担了**帧堆叠（frame stack）**的角色，也在启用 LSTM 时成为 LSTM 的 time_steps 维度（Stateful 情况下又会按步喂入）。

### 代码里怎么用
- 建模时决定输入层 shape：见 [`build_compile_model`](src/r2d3.py) 里对 `Input(...)` 的构造（`shape=(input_sequence,) + input_shape` 等）  
- Actor 侧维护最近观测窗口：见 [`ActorRunner.reset_states`](src/r2d3.py) 与 [`ActorRunner.forward`](src/r2d3.py)
  - `self._state1 = self.recent_observations[-self.input_sequence:]`（取最近 `input_sequence` 个观测组成一个 state）见 `ActorRunner.forward`
- Demo/回放加载同样按这个长度组装：见 `add_memory`

### 作用
- **非图像**：等价于把最近 N 步状态拼起来（有助于部分可观测时补信息）
- **图像**：
  - 不用 LSTM 时（`LstmType.NONE`）你这里对 `GRAY_2ch` 会把 `input_sequence` 当作“通道数”用（`Permute((2,3,1))`），见 `build_compile_model`
  - 用 LSTM 时则是 time_steps，配合 `TimeDistributed` 提取每帧特征再交给 LSTM（同文件）

### 使用场景建议
- Atari：常用 4（你示例也是 4，见 atari_breakout.py）
- 低维状态：可以更长一些（如 MountainCar 示例 12，见 mountaincar.py），本质是“多步状态拼接”。

---

## 2. `reward_multisteps`：N-step 回报的 N（多步 TD 的跨度）

### 含义
`reward_multisteps = n` 表示使用 **n-step return**：用接下来 n 步的累积奖励，加上 `γ^n * Q(s_{t+n}, a*)` 作为目标（bootstrap）。

### 代码里怎么用
1) Actor 侧计算 n-step 累积奖励（并可选 rescaling）：
- 在 [`ActorRunner.backward`](src/r2d3.py) 中累积最近 `reward_multisteps` 步奖励并存到 `recent_rewards_multistep`  
- Demo 加载时也同样计算：见 `add_memory`

2) Learner 侧计算 TD-error / 目标：
- 无 LSTM 或非 stateful 的训练：见 `LearnerRunner.train_model`  
  `td_error = reward + (gamma ** reward_multisteps) * maxq - q0`
- Stateful LSTM 训练：见 `LearnerRunner.train_model_ful`  
  同样通过 `seq_i + reward_multisteps` 取“n 步后”的 Q 估计来做 bootstrap

### 作用
- **更快传播奖励信号**（尤其稀疏奖励）
- 但 **n 太大** 会增加估计方差，且对 episode 终止处理更敏感（你这里还有 `enable_terminal_zero_reward` 的补丁逻辑，见 `LearnerRunner.train` 与 Actor 侧发送经验片段处）

### 使用场景建议
- 稀疏/延迟奖励：适当加大（如 3、5）
- 奖励噪声大：别太大，避免方差过高

---

## 3. `lstm_ful_input_length`：Stateful LSTM 的“训练段长度”（要算 loss 的步数）

> 这个变量只在 `LstmType.STATEFUL` 这条分支里真正发挥核心作用；在 `STATELESS` 情况下它更多是配置保留/兼容。

### 含义
`lstm_ful_input_length = L` 表示每次从 replay 里采样到的“序列经验”中，Learner 会对连续 **L 个时间步**执行训练更新（即做 L 次 `train_on_batch`），而不是只训一个时间点。

### 代码里怎么用
- Actor 侧：
  - 维护动作/奖励缓冲区长度与 `multi_len` 相关：  
    `multi_len = reward_multisteps + lstm_ful_input_length - 1`  
    见 `ActorRunner.reset_states`
  - 发送给 Learner 的经验里动作/回报序列长度就是 `lstm_ful_input_length`：  
    `self.recent_actions[0:self.lstm_ful_input_length]`、`self.recent_rewards_multistep[:]`  
    见 `ActorRunner.forward`

- Learner 侧（Stateful）：
  - 外层先跑：`burnin_length + reward_multisteps + lstm_ful_input_length` 个时间点做预测并缓存（含 burn-in 与 bootstrap 所需的未来步）  
    见 `LearnerRunner.train_model_ful`
  - 真正训练循环是：`for seq_i in range(self.lstm_ful_input_length): ... train_on_batch(...)`  
    同函数

### 作用
- 让 RNN 在一个 batch 内看到更长的连续片段，学习“时间依赖”
- 同时也使 PER 的 priority 变成“序列级别”的（你后面用 max/mean 混合聚合多个时间步 priority）

### 使用场景建议
- 需要更强记忆（POMDP 明显、观测不全）：可以增大 L
- 算力/吞吐受限：L 越大每次更新越慢

---

## 4. `burnin_length`：Stateful LSTM 的“烧入/预热步数”（不训练，只推进 hidden state）

### 含义
对 **Stateful LSTM + 经验回放**而言，随机采样到序列中间一段时，LSTM 的隐状态如果从 0 开始会不匹配。`burnin_length` 就是：在开始计算 loss 前，先用一段历史观测把 LSTM hidden state “烧到”更接近采样时的真实状态。

### 代码里怎么用
- 只有 `lstm_type == LstmType.STATEFUL` 才保留 burn-in，否则会被强制设为 0：见 [`R2D3.__init__`](src/r2d3.py)（`if lstm_type != STATEFUL: burnin_length = 0`）
- Actor 侧为 burn-in 预留更长的缓冲区：
  - `tmp = burnin_length + input_sequence + multi_len` 等，见 `ActorRunner.reset_states`
  - 选择 `self._state0 = recent_observations_wrap[-burnin_length - 1]` 作为“对齐 burn-in 起点”的 state：见 `ActorRunner.forward`
- Learner 侧执行 burn-in（只 predict 更新 state，不训练）：
  - `if seq_i < burnin_length: continue` 这种逻辑，见 `LearnerRunner.train_model_ful`

### 作用
- 提高训练时 hidden state 与采样时的一致性，稳定 RNN 学习
- burn-in 越长越“对齐”，但越耗算力

### 使用场景建议
- 仅 **Stateful LSTM** 需要（你代码也是这么做的）
- Episode 很长、依赖较远历史：适当增大 burn-in
- 资源紧张：burn-in 取小一些，但可能会降低效果

---

## 5. 四者的典型组合（按你的实现习惯）

### A) 图像 + Stateless LSTM（你示例 Breakout）
- `input_sequence=4`：4 帧输入
- `lstm_type=STATELESS`：每次序列独立训练
- `burnin_length` 会被置 0（因为不是 STATEFUL）
- `reward_multisteps=3`：3-step return

见 atari_breakout.py 与建模逻辑 `build_compile_model`。

### B) 低维状态 + Stateless LSTM（MountainCar）
- `input_sequence=12`：拼更长的状态窗口
- `reward_multisteps=3`
- `burnin_length` 同样会被置 0

见 mountaincar.py。

### C) Stateful LSTM（真正用到 burn-in / ful_input_length 的地方）
- 需要同时调：`burnin_length`、`lstm_ful_input_length`、`reward_multisteps`
- 并且经验在 replay 里是“序列片段 + hidden state”的形式（你代码就是这么存的）

核心路径在：
- Actor 组装并发送序列经验：`ActorRunner.forward`
- Learner 按时间步 burn-in + 多步 bootstrap + 分步训练：`LearnerRunner.train_model_ful`

---

如果你愿意，我可以按你当前正在用的某个具体配置（例如 atari_breakout.py 或你自己的 kwargs）把这四个参数代入，画出**一次采样序列的索引对齐图**（哪些步是 burn-in、哪些步参与 loss、哪些步仅用于 `reward_multisteps` 的 bootstrap），这样最直观。