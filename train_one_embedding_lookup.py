import numpy as np
import os
import pickle

from datetime import datetime
import tensorflow as tf
from time import time

from model_one_embedding_lookup import Model

def read_data(file_path):
    with open(os.path.join(file_path, 'train_dataset.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(file_path, 'val_dataset.pkl'), 'rb') as f:
        val = pickle.load(f)

    [train_q, train_a, train_a_num, train_c, train_l, train_c_real_len, train_q_real_len] = train
    [val_q, val_a, val_a_num, val_c, val_l, val_c_real_len, val_q_real_len] = val

    return ([train_q, train_a, train_a_num, train_c, train_l, train_c_real_len, train_q_real_len],\
           [val_q, val_a, val_a_num, val_c, val_l, val_c_real_len, val_q_real_len])


def batch_iter(c, q, l, a, a_num, c_real_len, q_real_len, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    c = np.array(c)
    q = np.array(q)
    l = np.array(l)
    a = np.array(a)
    a_num = np.array(a_num)
    c_real_len = np.array(c_real_len)
    q_real_len = np.array(q_real_len)
    data_size = len(q)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            c_shuffled = c[shuffle_indices]
            q_shuffled = q[shuffle_indices]
            l_shuffled = l[shuffle_indices]
            a_shuffled = a[shuffle_indices]
            a_num_shuffled = a_num[shuffle_indices]
            c_real_len_shuffled = c_real_len[shuffle_indices]
            q_real_len_shuffled = q_real_len[shuffle_indices]
        else:
            c_shuffled = c
            q_shuffled = q
            l_shuffled = l
            a_shuffled = a
            a_num_shuffled = a_num
            c_real_len_shuffled = c_real_len
            q_real_len_shuffled = q_real_len

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index < data_size:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch = c_shuffled[start_index:end_index], \
                                                                                         q_shuffled[start_index:end_index], \
                                                                                         l_shuffled[start_index:end_index], \
                                                                                         a_shuffled[start_index:end_index], \
                                                                                         a_num_shuffled[start_index:end_index], \
                                                                                         c_real_len_shuffled[start_index:end_index], \
                                                                                         q_real_len_shuffled[start_index:end_index]
                yield list(zip(c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch))

def parse_config(string):
    #parsing txt file
    result = dict()
    args = string.split("\t")
    result['c_max_len'] = int(args[0])
    result['s_max_len'] = int(args[1])
    result['q_max_len'] = int(args[2])
    result['babi_processed'] = args[3] # path
    result['batch_size'] = int(args[4])
    result['s_hidden'] = int(args[5])
    result['q_hidden'] = int(args[5])
    result['learning_rate'] = 2e-4
    result['iter_time'] = 150
    result['display_step'] = 100
    return result

#flags setting
flags = tf.app.flags



def main():
    global config
    with open('config.txt', 'r') as f:
        config = parse_config(f.readline())
    date = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
    model_id = "RN-one-embedding-lookup/" + date

    save_dir = "./result/" + model_id
    save_summary_path = os.path.join(save_dir, 'model_summary')
    save_variable_path = os.path.join(save_dir, 'model_variables')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_summary_path)
        os.makedirs(save_variable_path)

    (train_dataset, val_dataset) = read_data(config['babi_processed'])

    with tf.Graph().as_default():
        sess = tf.Session()
        start_time = time()
        with sess.as_default():
            rn = Model(config)

            # Define Training procedure
            global_step = tf.Variable(0, name='global_step', trainable = False)
            opt = tf.train.AdamOptimizer(config['learning_rate'])
            optimizer = opt.minimize(rn.loss, global_step = global_step)

            loss_train = tf.summary.scalar("loss_train", rn.loss)
            accuracy_train = tf.summary.scalar("accuracy_train", rn.accuracy)
            train_summary_ops = tf.summary.merge([loss_train, accuracy_train])

            loss_val = tf.summary.scalar("loss_val", rn.loss)
            accuracy_val = tf.summary.scalar("accuracy_val", rn.accuracy)
            val_summary_ops = tf.summary.merge([loss_val, accuracy_val])

            saver = tf.train.Saver(tf.global_variables(),max_to_keep=4)
            sess.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(save_summary_path, sess.graph)
            batch_train = batch_iter(c=train_dataset[3],
                                     q=train_dataset[0],
                                     l=train_dataset[4],
                                     a=train_dataset[1],
                                     a_num=train_dataset[2],
                                     c_real_len=train_dataset[5],
                                     q_real_len=train_dataset[6],
                                     num_epochs=config['iter_time'],
                                     batch_size= config['batch_size'])
            for train in batch_train:
                c_batch, q_batch, l_batch, a_batch, a_num_batch, c_real_len_batch, q_real_len_batch = zip(*train)
                feed_dict = {rn.context: c_batch,
                             rn.question: q_batch,
                             rn.label: l_batch,
                             rn.answer: a_batch,
                             rn.answer_num: a_num_batch,
                             rn.context_real_len: c_real_len_batch,
                             rn.question_real_len: q_real_len_batch,
                             rn.is_training: True}
                current_step = sess.run(global_step, feed_dict=feed_dict)
                optimizer.run(feed_dict=feed_dict)
                train_summary = sess.run(train_summary_ops, feed_dict=feed_dict)
                summary_writer.add_summary(train_summary, current_step)
                if current_step % (config['display_step']) == 0:
                    print("step: {}".format(current_step))
                    print("====validation start====")
                    batch_val = batch_iter(c = val_dataset[3],
                                           q = val_dataset[0],
                                           l = val_dataset[4],
                                           a = val_dataset[1],
                                           a_num = val_dataset[2],
                                           c_real_len=val_dataset[5],
                                           q_real_len=val_dataset[6],
                                           num_epochs=1,
                                           batch_size=config['batch_size'])
                    accs = []
                    for val in batch_val:
                        c_val, q_val, l_val, a_val, a_num_val, c_real_len_val, q_real_len_val = zip(*val)
                        feed_dict = {rn.context: c_val,
                                     rn.question: q_val,
                                     rn.label: l_val,
                                     rn.answer: a_val,
                                     rn.answer_num: a_num_val,
                                     rn.context_real_len: c_real_len_val,
                                     rn.question_real_len: q_real_len_val,
                                     rn.is_training: False}
                        acc = rn.accuracy.eval(feed_dict=feed_dict)
                        accs.append(acc)
                        val_summary = sess.run(val_summary_ops, feed_dict=feed_dict)
                        summary_writer.add_summary(val_summary, current_step)
                    print("Mean accuracy=" + str(sum(accs) / len(accs)))
                    saver.save(sess, save_path=save_summary_path, global_step=current_step)
                    print("====training====")
        end_time = time()
        print("Training finished in {}sec".format(end_time-start_time))

if __name__ == '__main__' :
    main()