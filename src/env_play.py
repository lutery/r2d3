import gym
import pygame
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.animation

import os
import pickle
from tkinter import Tk
from tkinter import messagebox
import glob
import time

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.processor import PendulumProcessorForDQN
from src.common import LstmType, rescaling

class _PlayWindow():
    """ copy from: https://github.com/openai/gym/blob/master/gym/utils/play.py """

    def __init__(self, font=None):
        self.font = font

        self.min_height = 600
        self.info_width = 200

    def play(self, fps=30, zoom=None, keys_to_action=None):
        pygame.init()
        clock = pygame.time.Clock()

        self.fps = fps
        self.msgs = {
            "key": [
                "Available keys:",
                "ESC: exit",
                "1-6: size change"
            ],
        }
        font1 = pygame.font.SysFont(self.font, 22)

        self.org_size = self.on_play_before()
        self.screen = pygame.display.set_mode(self.org_size)
        self.resize(zoom)

        self.running = True

        while self.running:
            self.screen.fill((0, 0, 0, 0))

            # process pygame events
            for event in pygame.event.get():
                self.on_event_loop(event)
            
            self.on_loop()

            # draw text
            y = 5
            for v in self.msgs.values():
                for s in v:
                    self.screen.blit(font1.render(s, False, (255,255,255)), (5+ self.video_size[0], y))
                    y += 25
                y += 10
            
            pygame.display.flip()
            clock.tick(self.fps)
        pygame.quit()
        self.on_play_end()

    def on_play_before(self):
        raise NotImplementedError()

    def on_play_end(self):
        pass

    def on_event_loop(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == 27:  # ESC
                self.running = False
            elif event.unicode == '1':
                self.resize(0.5)
            elif event.unicode == '2':
                self.resize(1.0)
            elif event.unicode == '3':
                self.resize(1.5)
            elif event.unicode == '4':
                self.resize(2.0)
            elif event.unicode == '5':
                self.resize(3.0)
            elif event.unicode == '6':
                self.resize(4.0)
        elif event.type == pygame.QUIT:
            self.running = False

    def on_loop(self):
        pass
    
    def resize(self, zoom):
        if zoom is None:
            zoom = 1
        h = self.org_size[1] * zoom
        self.video_size = int(self.org_size[0] * zoom), int(h)
        if h < self.min_height:
            h = self.min_height
        window_size = int(self.video_size[0] + self.info_width), int(h)
        pygame.display.set_mode(window_size)
        self.set_msg(["size: {}".format(self.video_size)])

    def display_arr(self, arr):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        pyg_img = pygame.transform.scale(pyg_img, self.video_size)
        self.screen.blit(pyg_img, (0,0))

    def add_key_msg(self, msg):
        self.msgs["key"].append(msg)

    def set_msg(self, msgs):
        print("\n".join(msgs))
        self.msgs["msg"] = msgs

    def set_info(self, msgs):
        self.msgs["info"] = msgs



class EpisodeSave(_PlayWindow):
    def __init__(self, env, processor=None, episode_save_dir=None, keys_to_action=None, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.processor = processor
        self.episode_save_dir = episode_save_dir
        self.keys_to_action = keys_to_action
        if self.episode_save_dir is not None:
            os.makedirs(self.episode_save_dir, exist_ok=True)


    def on_play_before(self):
        self.env.reset()
        rendered = self.env.render( mode='rgb_array')
        env_size = [rendered.shape[1], rendered.shape[0]]
        
        if self.keys_to_action is None:
            if self.processor is not None and hasattr(self.processor, 'get_keys_to_action'):
                self.keys_to_action = self.processor.get_keys_to_action()
            elif hasattr(self.env, 'get_keys_to_action'):
                self.keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, 'get_keys_to_action'):
                self.keys_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                assert False, self.env.spec.id + " does not have explicit key to action mapping, " + \
                            "please specify one manually"
        self.relevant_keys = set(sum(map(list, self.keys_to_action.keys()),[]))
        super().add_key_msg("-/+: speed change")
        super().add_key_msg("f: frameadvance")
        super().add_key_msg("p: Pause/Unpouse")
        super().add_key_msg("Use keys:")
        for k in self.relevant_keys:
            super().add_key_msg(" {}".format(pygame.key.name(k)))

        self.pressed_keys = []
        self.env_done = True
        self.episode_count = 0
        self.is_frameadvance = False

        return env_size

    

    def on_event_loop(self, event):
        super().on_event_loop(event)

        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
            elif event.unicode == 'p':
                self.env_pause = False if self.env_pause else True
            elif event.unicode == 'f':
                self.is_frameadvance = True
            elif event.unicode == '+':
                self.fps += 10
                super().set_msg(["fps: {}".format(self.fps)])
            elif event.unicode == '-':
                self.fps -= 10
                if self.fps < 10:
                    self.fps = 10
                super().set_msg(["fps: {}".format(self.fps)])
        elif event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)

    def on_loop(self):
        super().on_loop()

        if self.env_done:
            self.env_done = False
            self.env_pause = True
            self.obs = self.env.reset()
            self.episode_count += 1
            self.total_reward = 0
            self.step = 0
            self.action = 0
            self.reward = 0
            self.states1 = []
            self.states2 = []
            super().set_msg(["start new episode {}.".format(self.episode_count)])
        
        f = True
        if self.env_pause:
            f = False
        if self.is_frameadvance:
            f = True
            self.env_pause = True
            self.is_frameadvance = False
        
        if f:
            self.action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)
            if self.processor is not None:
                _action = self.processor.process_action(self.action)
            else:
                _action = self.action
            self.obs, self.reward, self.env_done, info = self.env.step(_action)
            if self.processor is not None:
                self.obs, self.reward, self.env_done, info = \
                    self.processor.process_step(self.obs, self.reward, self.env_done, info)
            self.total_reward += self.reward
            self.step += 1

            self.states1.append({
                "action": self.action,
                "observation": self.obs,
                "reward": self.reward,
                "done": self.env_done,
            })
            self.states2.append({
                "step": self.step,
                "reward_total": self.total_reward,
                "info": info,
                "rgb": self.env.render(mode='rgb_array'),
            })

            if self.env_done:
                Tk().wm_withdraw() #to hide the main window
                if messagebox.askyesno("Save?", "Do you want to save the episode?"):
                    path1 = os.path.join(self.episode_save_dir, "episode{}.dat".format(self.episode_count))
                    path2 = os.path.join(self.episode_save_dir, "episode{}.dat.display".format(self.episode_count))
                    super().set_msg(["save: {}".format(path1)])
                    with open(path1, 'wb') as f:
                        pickle.dump(self.states1, f)

                    d = {
                        "episode": self.episode_count,
                        "rgb_size": self.org_size,
                        "states": self.states2,
                    }
                    with open(path2, 'wb') as f:
                        pickle.dump(d, f)
                else:
                    self.episode_count -= 1

        if self.obs is not None:
            rendered = self.env.render(mode='rgb_array')
            super().display_arr(rendered)

        # env info
        super().set_info([
            "episode: {}".format(self.episode_count),
            "action : {}".format(self.action),
            "step   : {}".format(self.step),
            "reward : {}".format(self.reward),
            "total  : {}".format(self.total_reward),
        ])



class EpisodeReplay(_PlayWindow):
    def __init__(self, episode_save_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.episode_save_dir = episode_save_dir

        path = os.path.join(self.episode_save_dir, "episode1.dat")
        assert os.path.isfile(path), "episode is not found: {}".format(path)


    def on_play_before(self):

        super().add_key_msg("down: prev episode")
        super().add_key_msg("up: next episode")
        super().add_key_msg("left: prev frame")
        super().add_key_msg("right: next frame")
        super().add_key_msg("p: Pause/Unpouse")

        self.episode = 1
        self.set_episode()

        return self.org_size
    
    def on_event_loop(self, event):
        super().on_event_loop(event)
        
        if event.type == pygame.KEYDOWN:
            if event.unicode == 'p':
                self.env_pause = False if self.env_pause else True
            elif event.key == 275:  # right
                self.step += 1
            elif event.key == 276:  # left
                self.step -= 1
            elif event.key == 273:  # up
                self.episode += 1
                self.set_episode()
            elif event.key == 274:  # down
                self.episode -= 1
                self.set_episode()
        

    def on_loop(self):
        super().on_loop()

        if self.step < 0:
            self.step = 0
            self.env_pause = True
        if self.step >= len(self.states1):
            self.step = len(self.states1) - 1
            self.env_pause = True
        state1 = self.states1[self.step]
        state2 = self.states2[self.step]
        
        super().display_arr(state2["rgb"])
        super().set_info([
            "episode: {}".format(self.episode),
            "action : {}".format(state1["action"]),
            "step   : {} / {}".format(state2["step"], len(self.states1)),
            "reward : {}".format(state1["reward"]),
            "total  : {}".format(state2["reward_total"]),
            "done   : {}".format(state1["done"]),
        ])
        
        if not self.env_pause:
            self.step += 1
    
    def set_episode(self):
        if self.episode < 1:
            self.episode = 1

        episode_file = os.path.join(self.episode_save_dir, "episode{}.dat".format(self.episode))
        path1 = episode_file
        path2 = episode_file + ".display"
        if not os.path.isfile(path1):
            super().set_msg(["episode file is not found: {}".format(path1)])
            return
        with open(path1, 'rb') as f:
            self.states1 = pickle.load(f)
        if os.path.isfile(path2):
            with open(path2, 'rb') as f:
                d = pickle.load(f)
            self.org_size = d["rgb_size"]
            self.states2 = d["states"]
        else:
            print("display file is not found: {}".format(path2))

        self.env_pause = True
        self.step = 0
        super().set_msg(["episode{} is load.".format(self.episode)])
    

    def save(self,
            episode,
            start_frame=0,
            end_frame=0,
            gifname="",
            mp4name="",
            interval=200,
            fps=30
        ):
        #--- episode を読み込む
        self.episode = episode
        self.set_episode()
        self.frames = []
        for s in self.states2:
            self.frames.append(s["rgb"])
        
        assert start_frame<len(self.frames), "start frame is over frames({})".format(len(self.frames))
        if end_frame == 0:
          end_frame = len(self.frames)
        elif end_frame > len(self.frames):
            end_frame = len(self.frames)
        self.start_frame = start_frame
        self.t0 = time.time()
        
        self.patch = plt.imshow(self.frames[0])
        plt.axis('off')
        ani = matplotlib.animation.FuncAnimation(plt.gcf(), self._plot, frames=end_frame - start_frame, interval=interval)

        if gifname != "":
            ani.save(gifname, writer="pillow", fps=fps)
            #ani.save(gifname, writer="imagemagick", fps=fps)
        if mp4name != "":
            ani.save(mp4name, writer="ffmpeg")
    
    def _plot(self, frame):
        if frame % 50 == 0:
            print("{}f {:.2f}m".format(frame, (time.time()-self.t0)/60))
        
        #plt.imshow(self.frames[frame + self.start_frame])
        self.patch.set_data(self.frames[frame + self.start_frame])


def add_memory(episode_save_dir, memory, agent):
    '''
    Docstring for add_memory
    与其叫add不如叫load更合适一些，这个函数是从保存的demo数据中加载数据到memory中
    存储的记忆格式是一个episode一文件，文件中存储了一个list，list的每个元素是一个dict，包含
    "observation", "action", "reward", "done"等字段
    
    :param episode_save_dir: 记忆存储的路径
    :param memory: Description
    :param agent: Description


    burnin_length：在训练前使用固定长度的数据预热LSTM状态 todo 是如何预热的
    input_sequence：在普通模型中是帧堆叠，在LSTM中就是时间步数
    reward_multisteps：N-step 回报的 N（多步 TD 的跨度），所以采集抽样用于训练的样本树要额外增加一个reward_multisteps长度，这样才可以计算最后一个时间步的多步回报
    lstm_ful_input_length：只在 LstmType.STATEFUL 这条分支里真正发挥核心作用；在 STATELESS 情况下它更多是配置保留/兼容。这里是在有状态的LSTM下训练的帧长度
    '''

    for fn in glob.glob(os.path.join(episode_save_dir, "episode*.dat")):
        print("load: {}".format(fn))
        with open(fn, 'rb') as f:
            epi_states = pickle.load(f) # 记忆文件使用pickle存储的
        if len(epi_states) <= 0:
            continue
        
        # 防御式编程，检查episode的shape是否和agent的input shape匹配
        if agent.input_shape != epi_states[0]["observation"].shape:
            print("episode shape is not match. input_shape{} != epi_shape{}".format(
                agent.input_shape,
                epi_states[0]["observation"].shape
            ))
            continue

        # init 初始化缓冲区
        if agent.lstm_type == LstmType.STATEFUL:
            multi_len = agent.reward_multisteps + agent.lstm_ful_input_length - 1 # 有状态LSTM的多步长度，包含多步奖励和ful_input长度
            recent_actions = [ 0 for _ in range(multi_len + 1)] # 最近的动作缓冲区
            recent_rewards = [ 0 for _ in range(multi_len)] # 最近的奖励缓冲区
            recent_rewards_multistep = [ 0 for _ in range(agent.lstm_ful_input_length)] # 最近的多步奖励缓冲区
            tmp = agent.burnin_length + agent.input_sequence + multi_len # todo 这三段的作用，为什么会比之前的缓冲区长
            recent_observations = [
                np.zeros(agent.input_shape) for _ in range(tmp)
            ] # 最近的观察缓冲区 可以用于存储最近的观察值，包装成LSTM需要的格式，比如从最近的观察中一次性取出帧堆叠的数据
            tmp = agent.burnin_length + multi_len + 1 # 这个长度的每段的作用 看函数注释
            recent_observations_wrap = [
                [np.zeros(agent.input_shape) for _ in range(agent.input_sequence)] for _ in range(tmp)
            ] # 最近的观察缓冲区，包装成LSTM需要的格式
            # 注意recent_observations_wrap是一个数组的数组的格式，即：[[agent.input_sequence], [agent.input_sequence], [agent.input_sequence]... tmp]

            # hidden_state: [(batch_size, lstm_units_num), (batch_size, lstm_units_num)]
            tmp = agent.burnin_length + multi_len + 1+1 # todo 这个长度的每段的作用
            agent.model.reset_states() # 这个是tf官方的lstm状态重置，清空状态，设置为0
            recent_hidden_states = [
                [K.get_value(agent.lstm.states[0]), K.get_value(agent.lstm.states[1])] for _ in range(tmp)
            ] # 创建一个存储LSTM近期的隐藏状态的缓冲区，存储每input_sequence个观察运行后的隐藏层状态

        else: 
            # 如果是无状态的，则每次的起步都一样，隐藏状态都是从0开始
            recent_actions = [ 0 for _ in range(agent.reward_multisteps+1)] # 最近的动作缓冲区
            recent_rewards = [ 0 for _ in range(agent.reward_multisteps)] # 最近的奖励缓冲区
            recent_rewards_multistep = 0 # todo 为啥没有最近的多步奖励缓冲区为0？
            recent_observations = [
                np.zeros(agent.input_shape) for _ in range(agent.input_sequence + agent.reward_multisteps)
            ] # 最近的观察缓冲区

        # episode
        total_reward = 0 # 一局游戏最大的奖励
        # 读取存储的一局完整的游戏数据
        for epi_state in epi_states: # 遍历每一步的数据
            # forward
            recent_observations.pop(0) # 弹出最早的观察（因为可能存在存储的游戏数据超过了缓冲区的大小）
            recent_observations.append(epi_state["observation"]) # 保存最新的游戏观察

            if agent.lstm_type == LstmType.STATEFUL:
                # 有状态的LSTM，则需要进一步处理recent_observations_wrap
                recent_observations_wrap.pop(0) # 先弹出最早的观察
                recent_observations_wrap.append(recent_observations[-agent.input_sequence:]) # 取出recent_observations最新input_sequence长度的观察然后作为一个组合送入recent_observations_wrap缓冲区
            
            # hidden states update todo 如果是有状态的LSTM，需要通过真实的观察送入模型，拿到该时刻的隐藏层状态存储起来
            if agent.lstm_type == LstmType.STATEFUL: # 有状态的LSTM
                agent.lstm.reset_states(recent_hidden_states[-1]) # 提出最近的一次隐藏层状态，然后设置到lstm层中

                state = np.asarray(recent_observations[-agent.input_sequence:]) # 提取出最近的input_sequence长度个观察
                state = np.full((agent.batch_size,) + state.shape, state) # 创建一个shape = （batch_size， state.shape）的数组，内容全部是state
                agent.model.predict(state, batch_size=agent.batch_size) # 将最近的观察送入模型进行一次预测 # todo 作用？难道是为了预测拿到真实的隐藏层状态？

                # 获取真实的隐藏层状态保存到recent_hidden_states
                hidden_state = [K.get_value(agent.lstm.states[0]), K.get_value(agent.lstm.states[1])] 
                recent_hidden_states.pop(0)
                recent_hidden_states.append(hidden_state)

            # add memory
            # 将准备好的数据添加到memory缓冲区中
            if agent.lstm_type == LstmType.STATEFUL:
                # todo 为什么有状态和无状态塞入memory中的数据不同？
                memory.add((
                    recent_observations_wrap[:],
                    recent_actions[0:agent.lstm_ful_input_length],
                    recent_rewards_multistep[:],
                    recent_hidden_states[0]))
            else:
                memory.add((
                    recent_observations[:agent.input_sequence],
                    recent_actions[0],
                    recent_rewards_multistep, 
                    recent_observations[-agent.input_sequence:]))
            
            # action 更新最近执行动作缓冲区的数据
            recent_actions.pop(0)
            recent_actions.append(epi_state["action"])

            # reward 更新最近获得奖励缓冲区的数据
            recent_rewards.pop(0)
            recent_rewards.append(epi_state["reward"])
            total_reward += epi_state["reward"]

            # multi step learning の計算
            _tmp = 0
            # 到倒数第agent.reward_multisteps个便利到倒数第一个
            # 计算最近reward_multisteps步的累计奖励
            for i in range(-agent.reward_multisteps, 0):
                r = recent_rewards[i]
                _tmp += r * (agent.gamma ** i) # todo 这里的计算是否存在问题？i是负数吧

            # rescaling
            if agent.enable_rescaling:
                _tmp = rescaling(_tmp) # 缩放计算的多步奖励

            # todo 为什么有状态的LSTM是将计算的累积奖励存储到缓冲区中，而无状态的LSTM则直接覆盖？
            if agent.lstm_type == LstmType.STATEFUL:
                recent_rewards_multistep.pop(0)
                recent_rewards_multistep.append(_tmp)
            else:
                recent_rewards_multistep = _tmp
        

        # 最後の状態も追加
        if agent.lstm_type == LstmType.STATEFUL:
            memory.add((
                recent_observations_wrap[:],
                recent_actions[0:agent.lstm_ful_input_length],
                recent_rewards_multistep[:],
                recent_hidden_states[0]))
        else:
            memory.add((
                recent_observations[:agent.input_sequence],
                recent_actions[0],
                recent_rewards_multistep, 
                recent_observations[-agent.input_sequence:]))
        

        # 最後の報酬が0じゃないなら0報酬を追加
        # todo enable_terminal_zero_reward 这个的作用是什么？
        if agent.enable_terminal_zero_reward and recent_rewards[-1] != 0:
            # add memory todo 如果最后的奖励不是0，对于有状态的LSTM则添加最后一次计算的累积奖励，但是为啥无状态的添加的是0？
            if agent.lstm_type == LstmType.STATEFUL:
                memory.add((
                    recent_observations_wrap[:],
                    recent_actions[0:agent.lstm_ful_input_length],
                    recent_rewards_multistep[:],
                    recent_hidden_states[0]))
            else:
                memory.add((
                    recent_observations[:agent.input_sequence],
                    recent_actions[0],
                    0, 
                    recent_observations[-agent.input_sequence:]))


    print("demo replay loaded, on_memory: {}, total reward: {}".format(len(memory), total_reward))

