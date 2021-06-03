#!/usr/bin/env python3

import gym
import numpy as np
import time
from cart_pole_nn import create_a_nn, use_trained_nn
from statistics import mean

def generate_data(episodes):
    threshold = 100
    env = gym.make('CartPole-v0')
    num_data_points = 0
    prev_obs = []
    all_data = []
    rewards = []
    for episode in range(episodes):
        observation = env.reset()
        episode_data = []
        reward = 0
        if not episode % 100:
            print("Episode ", episode)
        #     print("Data points:", num_data_points * 100)
        for t in range(200):
            # env.render()
            # time.sleep(0.05)
            prev_obs = observation.copy()
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)  # take a random action
            if action == 0:
                action = -1

            prev_obs = np.append(prev_obs, action)

            episode_data.append(prev_obs)
            # observation is now [x, xdot, theta, thetadot, action (0 or 1)]

            if done:
                reward = t + 1
                # print("Random: {} time steps".format(reward))
                rewards.append(reward)
                break

        if reward >= threshold:
            num_data_points += 1
            for obs in episode_data:
                all_data.append(obs)
                # print(len(all_data))
            # print(all_data)
        env.close()

    all_data = np.array(all_data)
    print("Average score over {} episodes: {} timesteps".format(episodes, mean(rewards)))
    # print(all_data)
    # np.save('cart-pole-data-{}'.format(time.strftime("%Y%m%d-%H%M%S")), all_data)


def use_nn_for_cartpole(network, episodes):
    env = gym.make('CartPole-v0')

    final_times = []
    for episode in range(episodes):
        observation = env.reset()
        observation = np.reshape(observation, (1, 4))

        if not episode % 100:
            print("Episode ", episode)
            # try:
            #     print("Average time:", mean(final_times))
            # except:
            #     pass
        for t in range(500):
            # env.render()
            # time.sleep(0.05)
            action = network.predict(observation)
            if action > 0.0:
                action = 1
            else:
                action = 0

            observation, reward, done, info = env.step(action)  # take a random action
            observation = np.reshape(observation, (1, 4))

            if done:
                # print("Network achieved {} time steps".format(t+1))
                final_times.append(t+1)
                break
        env.close()

    print("Average time:", mean(final_times))
    # np.save('cart-pole-data-{}'.format(time.strftime("%Y%m%d-%H%M%S")), all_data)


def test_nn():
    nn_weights_file = "weights-20210521-124015"
    network = use_trained_nn(nn_weights_file)
    use_nn_for_cartpole(network, 100)


if __name__ == "__main__":
    # generate_data(100000)
    test_nn()
