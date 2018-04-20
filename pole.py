import gym
from dqn import DQN

env = gym.make('CartPole-v0')   # 立杆子游戏
env = env.unwrapped

N_ACTIONS = env.action_space.n  # 杆子能做的动作
N_STATES = env.observation_space.shape[0]   # 杆子能获取的环境信息数

dqn = DQN(32, N_STATES, N_ACTIONS) # 定义 DQN 系统
i = 0
for i_episode in range(400):
    s = env.reset()
    while True:
        i += 1
        print('train_step', i)
        env.render()    # 显示实验动画
        a = dqn.choose_action(s)

        # 选动作, 得到环境反馈
        s_, r, done, info = env.step(a)

        # 修改 reward, 使 DQN 快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存记忆
        dqn.store_sars_(s, a, r, s_, done)

        if dqn.memory_counter > dqn.memory_size:
            dqn.train_net() # 记忆库满了就进行学习

        if done:    # 如果回合结束, 进入下回合
            break

        s = s_