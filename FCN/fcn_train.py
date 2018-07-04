from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


seed = 1337

tf.set_random_seed(seed)
np.random.seed(seed)


results = {
    'output': './gen_img/',
    'model': './model/FCN-model.ckpt'
}

train_step = {
    'batch_size': 16,
    'global_step': 100001,
    'logging_interval': 1000,
}


def main():
    pass


if __name__ == '__main__':
    main()
