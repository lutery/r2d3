import gym
from keras.optimizers import Adam

import traceback

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.common import seed_everything
seed_everything(43)

from src.r2d3 import Actor
from src.policy import EpsilonGreedy, AnnealingEpsilonGreedy
from src.image_model import DQNImageModel
from src.memory import PERRankBaseMemory, PERProportionalMemory
from src.common import InputType, LstmType, DuelingNetwork, LoggerType

from Lib import run_gym_rainbow, run_gym_r2d3, run_play, run_replay


ENV_NAME = "MountainCar-v0"
episode_save_dir = "tmp_{}.".format(ENV_NAME)


def create_parameter():
    '''
    Docstring for create_parameter
    创建并返回用于训练的参数字典，其中环境的参数通过临时make一个环境获得
    todo 补齐训练参数
    '''

    env = gym.make(ENV_NAME) # 构建环境
    
    # ゲーム情報
    print("action_space      : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    print("reward_range      : " + str(env.reward_range))

    processor = None
    input_shape = env.observation_space.shape
    input_type = InputType.VALUES
    image_model = None
    enable_rescaling = False
    warmup = 50_000

    kwargs = {
        "input_shape": input_shape,  # 环境的输入shape
        "input_type": input_type, # 输入的类型，例如RAM或者图像等
        "nb_actions": env.action_space.n, # 环境的动作数量
        "optimizer": Adam(lr=0.001),
        "metrics": [],

        "image_model": image_model, # 可能是支持自定义检测模型 图像处理模型，主要作用是对图像序列进行特征提取（一个一个序列图片进行卷积）
        "input_sequence": 12,         # 入力フレーム数 这个是输入的帧数类似于帧堆叠，在LSTM中就是时间步数
        "dense_units_num": 32,       # dense層のユニット数 特征采集完成后的全连接层单元数
        "enable_dueling_network": True, # 是否启用dueling network 双DQN, 一条预测状态价值，一条预测动作优势Q值
        "dueling_network_type": DuelingNetwork.AVERAGE,  # dueling networkで使うアルゴリズム 这里定义双dqn最终计算Q值的方法,是平均、最大值还是加法
        "lstm_type": LstmType.STATELESS,           # 使用するLSTMアルゴリズム LSTM的类型，这里是无状态LSTM，也可以设置为有状态LSTM
        "lstm_units_num": 32,             # LSTMのユニット数
        "lstm_ful_input_length": 2,       # ステートフルLSTMの入力数

        # train/action関係
        "memory_warmup_size": warmup,    # 初期のメモリー確保用step数(学習しない)
        "target_model_update": 1000,  # target networkのupdate間隔
        "action_interval": 1,       # アクションを実行する間隔
        "batch_size": 32,     # batch_size
        "gamma": 0.99,        # Q学習の割引率
        "enable_double_dqn": True,
        "enable_rescaling": enable_rescaling,   # rescalingを有効にするか
        "rescaling_epsilon": 0.001,  # rescalingの定数
        "priority_exponent": 0.9,   # priority優先度
        "burnin_length": 1,        # burn-in期間
        "reward_multisteps": 3,    # multistep reward
        "enable_terminal_zero_reward": False,

        # その他
        "processor": processor,
        "memory": PERRankBaseMemory(
            capacity=500_000,
            alpha=0.8,           # PERの確率反映率
            beta_initial=0.0,    # IS反映率の初期値
            beta_steps=warmup+50_000,  # IS反映率の上昇step数
            enable_is=True,     # ISを有効にするかどうか
        ),
        "demo_memory": PERProportionalMemory(100_000, alpha=0.8),
        "demo_episode_dir": episode_save_dir,
        "demo_ratio_initial": 1.0,
        "demo_ratio_final": 1.0/512.0,
        "demo_ratio_steps": warmup+50_000,

        "episode_memory": PERProportionalMemory(2_000, alpha=0.8),
        "episode_ratio": 1.0/8.0,
        
    }

    env.close()
    return kwargs


def run_rainbow(enable_train):
    kwargs = create_parameter()
    kwargs["train_interval"] = 1
    kwargs["action_policy"] = AnnealingEpsilonGreedy(
        initial_epsilon=0.3,      # 初期ε
        final_epsilon=0.01,        # 最終状態でのε
        exploration_steps=kwargs["memory_warmup_size"] + 10_000  # 初期→最終状態になるまでのステップ数
    )
    
    run_gym_rainbow(enable_train, ENV_NAME, kwargs,
        nb_steps=kwargs["memory_warmup_size"] + 100_000,
        log_interval1=2000,
        is_load_weights=False,
        skip_movie_save=True,
    )
    


class MyActor(Actor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1) # 返回一个epsilon-greedy动作策略

    def fit(self, index, agent):
        '''
        这个方法是调用agent的fit方法与环境进行交互，在本代码中主要是用于数据的采集
        
        :param self: Description
        :param index: Description
        :param agent: Description
        '''
        env = gym.make(ENV_NAME)
        agent.fit(env, visualize=False, verbose=0)
        env.close()

class MyActor1(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)

class MyActor2(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.2)

def run_r2d3(enable_train):
    '''
    Docstring for run_r2d3
    
    :param enable_train: 运行模式，True表示训练，False表示测试
    '''
    kwargs = create_parameter() # 构建训练的参数

    #kwargs["actors"] = [MyActor1]
    kwargs["actors"] = [MyActor1, MyActor2] # 多个actor，不同的actor使用不同的动作选择策略，但是做这里只是不同的epsilon值
    kwargs["gamma"] = 0.997 # 折扣因子，应该是用于计算回报的时候使用
    kwargs["actor_model_sync_interval"] = 50  # learner から model を同期する間隔 这个看起来是有点像同步到target net的间隔

    run_gym_r2d3(enable_train, ENV_NAME, kwargs,
        nb_trains=100_000,
        test_actor=MyActor,
        log_warmup=60,
        log_interval1=60,
        is_load_weights=False,
        skip_movie_save=True,
    )



if __name__ == '__main__':
    kwargs = create_parameter()
    env = gym.make(ENV_NAME)
    
    #run_play(env, episode_save_dir, kwargs["processor"])
    #run_replay(episode_save_dir)

    # run_rainbow(enable_train=True)
    #run_rainbow(enable_train=False)  # test only

    run_r2d3(enable_train=True)
    #run_r2d3(enable_train=False)  # test only





