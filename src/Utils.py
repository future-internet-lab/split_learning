import numpy as np
import random


def change_state_dict(state_dicts, i):
    def change_name(name):
        parts = name.split(".", 1)
        number = int(parts[0]) + i
        name = f"{number}" + "." + parts[1]
        return name
    new_state_dict = {}
    for key, value in state_dicts.items():
        new_key = change_name(key)
        new_state_dict[new_key] = value
    return new_state_dict


def non_iid_rate(num_data, rate):
    result = []
    for _ in range(num_data):
        if rate < random.random():
            result.append(0)
        else:
            result.append(1)
    return np.array(result)


def num_client_in_cluster(client_cluster_label):
    max_val = max(client_cluster_label)
    count_list = [0] * (max_val + 1)
    for num in client_cluster_label:
        count_list[num] += 1
    count_list = [[x] for x in count_list]
    return count_list

