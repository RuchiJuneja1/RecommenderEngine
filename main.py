import random
import tensorflow as tf

gflags = tf.app.flags

# Engine
gflags.DEFINE_boolean('engine', 'rnn', 'which recommender engine to use')

# Environment
gflags.DEFINE_string('env_name', 'RecommenderEngineEnv', 'name of environment to use')

# Misc
gflags.DEFINE_boolean('use_gpu', True, 'use gpu or not')
gflags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
gflags.DEFINE_integer('random_seed', 123, 'Value of random seed')

flags = gflags.FLAGS

# Set random seed
tf.set_random_seed(flags.random_seed)
random.seed(flags.random_seed)

if __name__ == '__main__':
    pass