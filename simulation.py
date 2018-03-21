import gym
import numpy as np
from neural_net import LearningModel
import os
from threading import Thread
import time
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)

learning = True
model_file = "last_session.hdf"


def stop_learning():
    global learning
    global model
    while True:
        a = input("Press 'q' to interrupt, 's' to save")
        if a == 'q':
            learning = False
            break
        if a == 's':
            model.save_to_file(model_file)
        if a == 't':
            print("Learning duration:", learning_time)
        if a == 'p':
            draw_graphics()


def draw_graphics():
    t = np.linspace(0, (len(th)-1)*0.05, len(th))
    plt.figure(1)
    plt.subplot(311)
    plt.title("Угол отклонения маятника от вертикальной оси")
    plt.xlabel('t, с')
    plt.ylabel('θ, рад')
    plt.grid(True)
    plt.plot(t, th, 'b')

    plt.subplot(312)
    plt.title("Угловая скорость маятника")
    plt.xlabel('t, с')
    plt.ylabel('ω, рад/с')
    plt.grid(True)
    plt.plot(t, av, 'g')

    plt.subplot(313)
    plt.title("Внешний момент силы, воздействующий на маятник")
    plt.xlabel('t, с')
    plt.ylabel('u, Н•м')
    plt.plot(t, u, 'r')
    plt.grid(True)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.525,
                        wspace=0.35)

    plt.show()


MAX_EP_STEPS = 200

env = gym.make('Pendulum-v0')
env.seed(1)  # reproducible
env = env.unwrapped

S_LEN = env.observation_space.shape[0]
A_BOUND = env.action_space.high

model = LearningModel()  # Нейросетевая модель

if os.path.exists(model_file):
    model.load_from_file(model_file)
    POINTS = model.get_outputs_count()
    model.eps = 0
else:
    POINTS = 2
    model.new_nn(S_LEN, (30, 14, POINTS), ("relu", "sigmoid", "linear"))

model.gamma = 0.8
fi = np.linspace(-A_BOUND, A_BOUND, POINTS)

t = Thread(target=stop_learning)
t.start()

i_episode = 0
start_time = time.clock()
learning_time = -1

while learning:
    thetas = []
    angle_velocities = []
    actions = []

    s = env.reset()
    s[2] /= 8
    t = 0
    ep_rs = []

    while True:
        env.render()

        a = model.compute(s)
        action = np.array([sum(a*fi)])

        s_, r, done, info = env.step(action)
        s_[2] /= 8

        model.training(s, a, r, s_)

        r /= 10
        s = s_

        thetas.append(np.arctan2(s[1], s[0]))
        angle_velocities.append(s[2]*8)
        actions.append(action)

        t += 1
        ep_rs.append(r)
        if t > MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            print("episode:", i_episode, "  reward:", int(running_reward))
            if int(running_reward) >= -50 and learning_time == -1:
                finish_time = time.clock()
                learning_time = finish_time - start_time
                print("Learning duration:", learning_time)
            th = np.array(thetas)
            av = np.array(angle_velocities)
            u = np.array(actions)
            i_episode += 1
            del s
            break

model.save_to_file(model_file)