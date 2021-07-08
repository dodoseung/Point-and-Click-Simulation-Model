# Random Policy

import deep_q_network as dqn
import tensorflow as tf
import numpy as np

from point_and_click_env import Env
import time

env = Env()

TRAINED_MODEL = True
N_EPISODE = 100
MAX_STEP_1_EPISODE = 100
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

if TRAINED_MODEL:
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./trained_model/'))
        sess.run(tf.global_variables_initializer())
        
        for i_episode in range(N_EPISODE):
            observation = env.reset()

            for t in range(MAX_STEP_1_EPISODE):
                # Render the animation
                time.sleep(0.1)
                env.render()

                # Select the action
                q_values = mainDQN.predict(observation)
                action = np.argmax(q_values)

                # Next step
                observation, reward, done, info = env.step(action)

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

else:
    for i_episode in range(N_EPISODE):
        observation = env.reset()

        for t in range(MAX_STEP_1_EPISODE):
            # Render the animation
            time.sleep(0.1)
            env.render()

            # Select the action
            action = env.action_space.sample()

            # Next step
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

env.close()
