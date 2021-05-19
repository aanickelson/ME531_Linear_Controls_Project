#!/usr/bin/env python3

import gym
import numpy as np
import time


def generate_data(episodes):
    env = gym.make('CartPole-v0')

    all_data = np.array([])
    for episode in range(episodes):
        observation = env.reset()
        episode_data = []
        if not episode % 100:
            print("Episode ", episode)
        for t in range(200):
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)  # take a random action

            observation = np.append(observation, action)
            # observation is now [x, xdot, theta, thetadot, action (0 or 1)]
            episode_data.append(observation)

            if done:
                reward = np.array([[t + 1] for _ in range(t+1)])
                episode_data = np.array(episode_data)
                episode_data = np.append(episode_data, reward, axis=1)
                if episode == 0:
                    all_data = episode_data.copy()
                # print(episode_data)
                break
        if episode > 0:
            all_data = np.vstack((all_data, episode_data))
        env.close()

    np.save('cart-pole-data-{}'.format(time.strftime("%Y%m%d-%H%M%S")), all_data)


if __name__ == "__main__":
    generate_data(100000)
