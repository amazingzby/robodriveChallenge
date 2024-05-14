import numpy as np
import torch
import pickle

if __name__ == "__main__":
    nusc_class_frequencies = np.array([
        1689094172,  # free
        # 1202967, #noise,忽略
        499256, 30190, 474913, 4078450, 327017, 40273, 579250, 104073, 599519, 1375497, 25841748, 734070, 8296492,
        11371339, 27194524, 28476243
    ])
    class_weights = 1 / np.log(nusc_class_frequencies + 0.001)
    class_mean = np.mean(class_weights)
    print(class_weights/class_mean)