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

