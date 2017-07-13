import itertools
import numpy as np
import os
import pandas as pd
import pickle
import sys
sys.path.append('../')

from datetime import datetime
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib import rnn
from tensorflow.contrib import slim
from time import time
from tqdm import tqdm
import Model

def read_data(file_path = './babi_preprocessed'):
    with open(file_path+'/train_dataset_masked.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(file_path+'/test_dataset_masked.pkl', 'rb') as f:
        test = pickle.load(f)
    with open(file_path+'/val_dataset_masked.pkl', 'rb') as f:
        val = pickle.load(f)

    [train_q, train_a, train_c, train_l, train_c_real_len, train_q_real_len] = train
    [val_q, val_a, val_c, val_l, val_c_real_len, val_q_real_len] = val
    [test_q, test_a, test_c, test_l, test_c_real_len, test_q_real_len] = test

    return ([train_q, train_a, train_c, train_l, train_c_real_len, train_q_real_len],\
           [val_q, val_a, val_c, val_l, val_c_real_len, val_q_real_len],\
           [test_q, test_a, test_c, test_l, test_c_real_len, test_q_real_len])


def batch_iter(c, q, l, a, c_real_len, q_real_len, batch_size, num_epochs, shuffle=True,
               is_training=True):
    """
    Generates a batch iterator for a dataset.
    """
    c = np.array(c)
    q = np.array(q)
    l = np.array(l)
    a = np.array(a)
    c_real_len = np.array(c_real_len)
    q_real_len = np.array(q_real_len)
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        # if is_training:
        #     alarm.send_message('RN training...epoch {}'.format(epoch + 1))
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            c_shuffled = c[shuffle_indices]
            q_shuffled = q[shuffle_indices]
            l_shuffled = l[shuffle_indices]
            a_shuffled = a[shuffle_indices]
            c_real_len_shuffled = c_real_len[shuffle_indices]
            q_real_len_shuffled = q_real_len[shuffle_indices]
        else:
            c_shuffled = c
            q_shuffled = q
            l_shuffled = l
            a_shuffled = a
            c_real_len_shuffled = c_real_len
            q_real_len_shuffled = q_real_len

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index < data_size:
                c_batch, q_batch, l_batch, a_batch, c_real_len_batch, q_real_len_batch = c_shuffled[start_index:end_index], \
                                                                                         q_shuffled[start_index:end_index], \
                                                                                         l_shuffled[start_index:end_index], \
                                                                                         a_shuffled[start_index:end_index], \
                                                                                         c_real_len_shuffled[start_index:end_index], \
                                                                                         q_real_len_shuffled[start_index:end_index]
            else:
                end_index = data_size
                start_index = end_index - batch_size
                c_batch, q_batch, l_batch, a_batch, c_real_len_batch, q_real_len_batch = c_shuffled[start_index:end_index], \
                                                                                         q_shuffled[start_index:end_index], \
                                                                                         l_shuffled[start_index:end_index], \
                                                                                         a_shuffled[start_index:end_index], \
                                                                                         c_real_len_shuffled[start_index:end_index], \
                                                                                         q_real_len_shuffled[start_index:end_index]
            yield list(zip(c_batch, q_batch, l_batch, a_batch, c_real_len_batch, q_real_len_batch))

def parse_config(string):
    #parsing txt file
    output = string
    return output

#flags setting
flags = tf.app.flags

# flags.Define
flags.DEFINE_string('save_dir', 'path', 'description')  # './babi_result/lookup_table/%s'
# flags.DEFINE_string('save_summary_path', 'path', 'description')  # os.path.join(save_dir, 'model_summary')
# flags.DEFINE_string('save_variable_path', 'path', 'description')  # os.path.join(save_dir, 'model_variables')
FLAGS = flags.FLAGS

def main():
    config = parse_config(open('config.txt', 'r'))
    date = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
    model_id = "RN" + date

    save_dir = flags.save_dir
    save_summary_path = os.path.join(save_dir, 'model_summary')
    save_variable_path = os.path.join(save_dir, 'model_variables')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_summary_path)
        os.makedirs(save_variable_path)

    (train, val, test) = read_data()
    # rain_q, train_a, train_c, train_l, train_c_real_len, train_q_real_len

    model = Model(config)

    start_time = time()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=4)
        summary_writer = tf.summary.FileWriter(save_summary_path, sess.graph)
        print("====training====")
        batch_train = batch_iter(c=train[2], q=train[0], l=train[3], a=train[1], c_real_len=train[4], q_real_len=train[5])
        for train in batch_train:
            c_batch, q_batch, l_batch, a_batch, c_real_len_batch, q_real_len_batch = zip(*train)
            feed_dict = {c: c_batch, q: q_batch, l: l_batch, a: a_batch, c_real_len: c_real_len_batch,
                         q_real_len: q_real_len_batch, training_phase: True}
            current_step = sess.run(global_step, feed_dict=feed_dict)
            optimizer.run(feed_dict=feed_dict)
            train_summary = sess.run(train_summary_ops, feed_dict=feed_dict)
            summary_writer.add_summary(train_summary, current_step)
            if current_step % (display_step) == 0:
                print("step: {}".format(current_step))
                print("====validation start====")
                batch_val = batch_iter(val_c, val_q, val_l, val_a, val_c_real_len, val_q_real_len, num_epochs=1)
                accs = []
                for val in batch_val:
                    c_val, q_val, l_val, a_val, c_real_len_val, q_real_len_val = zip(*val)
                    feed_dict = {c: c_val, q: q_val, l: l_val, a: a_val, c_real_len: c_real_len_val,
                                 q_real_len: q_real_len_val, training_phase: False}
                    acc = accuracy.eval(feed_dict=feed_dict)
                    accs.append(acc)
                    val_summary = sess.run(val_summary_ops, feed_dict=feed_dict)
                    summary_writer.add_summary(val_summary, current_step)
                print("Mean accuracy=" + str(sum(accs) / len(accs)))
                saver.save(sess, save_path=save_summary_path, global_step=current_step)
                print("====training====")
    end_time = time()

if __name__ == '__main__' :
    main()