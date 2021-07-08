"""
This code is the modified code from https://github.com/hunkim/ReinforcementZeroToAll/
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)
    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
    Loss: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃
"""
import os
import numpy as np
import tensorflow as tf
import random
from collections import deque
import deep_q_network as dqn
from point_and_click_env import Env
from score_logger import ScoreLogger
from typing import List

env = Env()
score_logger = ScoreLogger('mouse model', 1000, 100000)

# Constants defining our neural network
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.95
REPLAY_MEMORY = 100000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 1000
MAX_EPISODES = 4000000
SAVE_PERIOD = 10000
LOG_PERIOD = 10000
E_DECAY = 0.9998
E_MIN = 0.05

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

"""
Trains `mainDQN` with target Q values given by `targetDQN`
Args:
    mainDQN (dqn.DQN): Main DQN that will be trained
    targetDQN (dqn.DQN): Target DQN that will predict Q_target
    train_batch (list): Minibatch of replay memory
        Each element is (s, a, r, s', done)
        [(state, action, reward, next_state, done), ...]
Returns:
    float: After updating `mainDQN`, it returns a `loss`
"""
def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states
    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)

"""
Creates TF operations that copy weights from `src_scope` to `dest_scope`
Args:
    dest_scope_name (str): Destination weights (copy to)
    src_scope_name (str): Source weight (copy from)
Returns:
    List[tf.Operation]: Update operations are created and returned
"""

def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def main():
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        step_count = 0
        e = 1

        for episode in range(MAX_EPISODES + 1):
            if e > E_MIN: e *= E_DECAY
            done = False
            score = 0
            count = 0
            loss = 0
            q_value = 0
            state = env.reset()

            while not done:
                # Get the q table
                q_values = mainDQN.predict(state)
                q_value += np.mean(q_values)

                # Get the action
                action = np.argmax(q_values)

                if np.random.rand() < e:
                    action = env.action_space.sample()

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss_temp, _ = replay_train(mainDQN, targetDQN, minibatch)
                    loss += loss_temp
                    count += 1

                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                score += reward
                state = next_state
                step_count += 1

            # Log the data
            if count == 0:
                score_logger.add_csv(loss, q_value, score, env.time, env.effort, env.click, episode)
            else:
                score_logger.add_csv(loss / count, q_value / count, score, env.time, env.effort, env.click, episode)

            if episode % LOG_PERIOD == 0 or os.path.exists('./check'):
                _, _, _, ave = score_logger.score_show()
                _, _, _, ave_loss = score_logger.loss_show()
                _, _, _, ave_q = score_logger.q_value_show()
                time_mean = sum(env.time_mean) / len(env.time_mean)
                time_std = (sum([((x - time_mean) ** 2) for x in env.time_mean]) / len(env.time_mean)) ** 0.5
                error_rate = 1 - (sum(env.error_rate) / len(env.error_rate))
                print("Episode: {:}, Reward: {:.4}, Loss: {:.4}, Q Value: {:.4}, Time: {:.4} (SD: {:.4}), ER: {:.4}".format(
                    episode, float(ave), float(ave_loss), float(ave_q), float(time_mean), float(time_std), float(error_rate)))

            # Save the model
            if episode % SAVE_PERIOD == 0 and episode >= SAVE_PERIOD:
                _, score_ave, _, _ = score_logger.score_show()
                _, loss_ave, _, _ = score_logger.loss_show()
                mainDQN.save(episode, score_ave, loss_ave)
                print("Saved the model", episode, score_ave, loss_ave)


if __name__ == "__main__":
    main()