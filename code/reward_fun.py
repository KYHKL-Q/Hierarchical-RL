import numpy as np
import math

def reward_fun(current_state, next_state):
    if np.sum(current_state[:, 2]) + np.sum(current_state[:, 3]) < np.sum(next_state[:, 2]) + np.sum(next_state[:, 3]):
        reward1 = 1 / (np.sum(next_state[:, 2]) + np.sum(next_state[:, 3]) - np.sum(current_state[:, 2]) - np.sum(current_state[:, 3]) + 1)
    else:
        reward1=1

    if np.sum(current_state[:, 1]) < np.sum(next_state[:, 1]):
        reward2 = 1 / (np.sum(next_state[:, 1]) - np.sum(current_state[:, 1]) + 1)

    else:
        reward2=1

    return math.log((reward1 + reward2) / 2) + 10