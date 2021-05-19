#!/usr/bin/env python3

import gym
import numpy as np
import time
from cart_pole_nn import create_a_nn, use_trained_nn


def generate_data(episodes):
    threshold = 90
    env = gym.make('CartPole-v0')
    num_data_points = 0

    all_data = []
    for episode in range(episodes):
        observation = env.reset()
        episode_data = []
        reward = 0
        if not episode % 1000:
            print("Episode ", episode)
            print("Data points:", num_data_points)
        for t in range(200):
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)  # take a random action
            if action == 0:
                action = -1
            observation = np.append(observation, action)
            # observation is now [x, xdot, theta, thetadot, action (0 or 1)]
            episode_data.append(observation)

            if done:
                reward = t + 1
                break

        if reward >= threshold:
            num_data_points += 1
            for obs in episode_data:
                all_data.append(obs)
                # print(len(all_data))
            # print(all_data)
        env.close()

    all_data = np.array(all_data)
    print(all_data)
    np.save('cart-pole-data-{}'.format(time.strftime("%Y%m%d-%H%M%S")), all_data)


def use_nn_for_cartpole(network, episodes):
    env = gym.make('CartPole-v0')

    all_data = np.array([])
    for episode in range(episodes):
        observation = env.reset()
        observation = np.reshape(observation, (1, 4))

        if not episode % 1000:
            print("Episode ", episode)

        for t in range(200):
            # env.render()
            action = network.predict(observation)
            if action > 0.0:
                action = 1
            else:
                action = 0

            observation, reward, done, info = env.step(action)  # take a random action
            observation = np.reshape(observation, (1, 4))

            if done:
                print("Network achieved {} time steps".format(t+1))

                break
        env.close()

    np.save('cart-pole-data-{}'.format(time.strftime("%Y%m%d-%H%M%S")), all_data)


def test_nn():
    nn_weights_file = "weights-20210519-143022"
    network = use_trained_nn(nn_weights_file)
    use_nn_for_cartpole(network, 100)


if __name__ == "__main__":
    # generate_data(500000)
    test_nn()
