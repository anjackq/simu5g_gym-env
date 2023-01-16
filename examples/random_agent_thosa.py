#!/usr/bin/env python3
"""
Random example agent.

Samples a random action for one episode.
"""

import gym
import lib.simu5g_gym

def main():
    """Run the random agent."""

    gym.register(
        id="simu5g-v1",
        entry_point="lib.simu5g_gym:Simu5gEnv",
        kwargs={
            "scenario_dir": "../scenario/ai4mobile_thosa",
            "simu5g_root_dir": "/home/anjie/Desktop/simu5g_gym-env/lib/simu5g/bin"
        },
    )

    env = gym.make("simu5g-v1")

    env.reset()
    done = False
    rewards = []
    while not done:
        random_action = env.action_space.sample()
        observation, reward, done, info = env.step(random_action)
        rewards.append(reward)
    print("Number of steps taken:", len(rewards))
    print("Mean reward:", sum(rewards) / len(rewards))


if __name__ == "__main__":
    main()
