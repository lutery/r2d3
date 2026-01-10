import tensorflow as tf
from keras import backend as K
import numpy as np

import enum
import math
import os
import random


def clipped_error_loss(y_true, y_pred):
    '''
    这是 DQN 及其变体中常用的损失函数，也叫 Huber Loss，用于解决传统 MSE 损失对异常值过于敏感的问题。
    Docstring for clipped_error_loss
    
    :param y_true: Description
    :param y_pred: Description
    '''
    err = y_true - y_pred  # 计算预测的误差 todo 动作q值的误差？那么y_ture哪里来的？
    L2 = 0.5 * K.square(err) # 平方误差的一半，用于误差较小时的损失计算
    L1 = K.abs(err) - 0.5 # 线性误差减去0.5，用于误差较大时的损失计算 ，可以理解，如果误差比较大，直接用减就好，没必要平方

    # エラーが[-1,1]区間ならL2、それ以外ならL1を選択する。 如果误差在[-1,1]区间内则使用L2，否则使用L1
    loss = tf.where((K.abs(err) < 1.0), L2, L1)   # Keras does not cover where function in tensorflow :-(
    return K.mean(loss) # 返回平均损失

def rescaling(x, epsilon=0.001):
    '''
    对输入的x进行压缩：它通常作用在 n-step 累积奖励（以及由此构成的 TD target / Q 值目标） 上，用来让数值范围更温和、训练更稳定。
    对大数值：sqrt(|x|+1)-1 让增长从“线性”变成“次线性”（压缩大回报/大 TD 误差的影响），减少梯度爆炸或不稳定。
    对小数值：近似保持线性，避免太强的失真。
    epsilon*x：加一个很小的线性项，常见目的是让变换更“平滑/近似可逆”，避免在 0 附近或某些区间信息损失过大。
    # epsilon*x是可以去掉的，但是不建议，
    可以去掉，但一般**不建议**。

    `rescaling(x) = sign(x)(sqrt(|x|+1)-1) + εx` 里这项 `εx`（默认 `ε=0.001`）的主要作用是：

    - **防止“大数值区间梯度趋近于 0”**：  
    `sqrt(|x|+1)-1` 的导数是 `1/(2*sqrt(|x|+1))`，当 `|x|` 很大时会越来越接近 0，容易出现“值被强烈压缩后，反传梯度很小”的情况。  
    加上 `εx` 后，整体导数在大 `|x|` 时至少还能接近 `ε`，**不会完全“饱和”**。
    - **保留一点线性信息**：让缩放在极端回报/TD target 下不至于过分扁平，有助于学习信号稳定。

    如果你把 `εx` 去掉：
    - 优点：缩放更强，数值更“收敛”。
    - 风险：对大回报/大 TD-error 的样本，学习信号可能更弱（更慢或更不稳定，视环境奖励尺度而定）。

    建议：除非你确认回报范围不大或已经有别的机制（例如 reward clipping/更小学习率/更强归一化）保证稳定，否则保持 `εx` 更稳。你也可以做个 A/B 测试：同样配置下比较收敛速度和是否更容易出现训练停滞。
    '''
    n = math.sqrt(abs(x)+1) - 1
    return np.sign(x)*n + epsilon*x


class InputType(enum.Enum):
    VALUES = 1    # 画像無し 这应该指的是非图像输入，比如RAM状态
    GRAY_2ch = 3  # (width, height)
    GRAY_3ch = 4  # (width, height, 1)
    COLOR = 5     # (width, height, ch)

# todo 这几个参数的含义
class LstmType(enum.Enum):
    NONE = 0 # 不使用LSTM
    STATELESS = 1 # 不包含上一个状态的信息，即隐藏状态
    STATEFUL = 2 # 包含上一个状态的信息，有状态LSTM

class DuelingNetwork(enum.Enum):
    AVERAGE = 0
    MAX = 1
    NAIVE = 2


class LoggerType(enum.Enum):
    TIME = 1
    STEP = 2

# copy from https://qiita.com/okotaku/items/8d682a11d8f2370684c9
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #session_conf = tf.compat.v1.ConfigProto(
    #    intra_op_parallelism_threads=1,
    #    inter_op_parallelism_threads=1
    #)
    #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #tf.compat.v1.keras.backend.set_session(sess)

