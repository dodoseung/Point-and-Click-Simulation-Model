from collections import deque
import os
import csv
import shutil


CSV_PATH = "./outputs/"

class ScoreLogger:

    def __init__(self, env_name, ave_num, save_period):
        self.scores = deque(maxlen=save_period)
        self.loss = deque(maxlen=save_period)
        self.q_value = deque(maxlen=save_period)
        self.scores_short = deque(maxlen=ave_num)
        self.loss_short = deque(maxlen=ave_num)
        self.q_value_short = deque(maxlen=ave_num)
        self.env_name = env_name
        self.save_period = save_period
        self.ave_num = ave_num

        if os.path.exists(CSV_PATH):
            shutil.rmtree(CSV_PATH)
            
        os.mkdir(CSV_PATH)

    def add_csv(self, loss, q_value, score, time, effort, click, run):
        path = "./outputs/output" + str(run // self.ave_num) + ".csv"
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a", newline='')
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([loss, q_value, score, time, effort, click])

        self.scores.append(score)
        self.scores_short.append(score)
        self.loss.append(loss)
        self.loss_short.append(loss)
        self.q_value.append(q_value)
        self.q_value_short.append(q_value)

    def score_show(self):
        min_value = min(self.scores)
        max_value = max(self.scores)
        ave_value = sum(self.scores) / len(self.scores)
        ave_value_short = sum(self.scores_short) / len(self.scores_short)
        return float(min_value), float(ave_value), float(max_value), float(ave_value_short)

    def loss_show(self):
        min_value = min(self.loss)
        max_value = max(self.loss)
        ave_value = sum(self.loss) / len(self.loss)
        ave_value_short = sum(self.loss_short) / len(self.loss_short)
        return float(min_value), float(ave_value), float(max_value), float(ave_value_short)

    def q_value_show(self):
        min_value = min(self.q_value)
        max_value = max(self.q_value)
        ave_value = sum(self.q_value) / len(self.q_value)
        ave_value_short = sum(self.q_value_short) / len(self.q_value_short)
        return float(min_value), float(ave_value), float(max_value), float(ave_value_short)