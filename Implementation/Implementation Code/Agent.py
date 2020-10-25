import numpy as np
import tensorflow as tf
import random
from collections import deque


random.seed(6)
np.random.seed(6)



class DQN:
    def __init__(
            self,
            action_num,
            features_num,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=1000,
            batch_size=30,
            e_greedy_increment=0.002,
            
    ):
        self.action_num = action_num  
        self.features_num = features_num
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.01 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0  # total learning step
        self.replay_buffer = deque()  # init experience replay [s, a, r, s_, done]

        # [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        w_initializer = tf.random_normal_initializer(0., 0.3, 5)  # (mean=0.0, stddev=1.0, seed=None)
        # w_initializer = tf.random_normal_initializer(0., 0.3)  # no seed
        b_initializer = tf.constant_initializer(0.1)
        n_l1 = 20  # config of layers

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.features_num], name='s')  # input
        with tf.variable_scope('eval_net', reuse=tf.AUTO_REUSE):
            # collections to store variables
            collection_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.features_num, n_l1], initializer=w_initializer, collections=collection_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=collection_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.action_num], initializer=w_initializer, collections=collection_names)
                b2 = tf.get_variable('b2', [1, self.action_num], initializer=b_initializer, collections=collection_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        # --------------------calculate loss---------------------
        self.action_input = tf.placeholder("float", [None, self.action_num])
        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')  # for calculating loss
        q_evaluate = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, q_evaluate))
            # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  # better than RMSProp

            # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.features_num], name='s_')  # input
        with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):
            # collection_names(collections_names) are the collections to store variables
            collection_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.features_num, n_l1], initializer=w_initializer, collections=collection_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=collection_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.action_num], initializer=w_initializer, collections=collection_names)
                b2 = tf.get_variable('b2', [1, self.action_num], initializer=b_initializer, collections=collection_names)
                self.q_next = tf.matmul(l1, w2) + b2

            

    def choose_action(self, state):
        pro = np.random.uniform()
        if pro < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
            action = np.argmax(actions_value)
            print(' q-values:', actions_value, '  best_action:', action)
            print('  best_action:', action)
        else:
            action = np.random.randint(0, self.action_num)
            print('  rand_action:', action)
            print('  rand_action:', action)
        return action

    def choose_best_action(self, state):
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
        action = np.argmax(actions_value)
        return action

    def store_transition(self, s, a, r, s_):
        one_hot_action = np.zeros(self.action_num)
        one_hot_action[a] = 1
        self.replay_buffer.append((s, one_hot_action, r, s_))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            

        # sample batch memory from all memory: [s, a, r, s_]
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # calculate target q-value batch
        q_next_batch = self.sess.run(self.q_next, feed_dict={self.s_: next_state_batch})
        q_real_batch = []
        for i in range(self.batch_size):
            q_real_batch.append(minibatch[i][2] + self.gamma * np.max(q_next_batch[i]))
            

        # train eval network
        self.sess.run(self._train_op, feed_dict={
            self.s: state_batch,
            self.action_input: action_batch,
            self.q_target: q_real_batch
        })

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1