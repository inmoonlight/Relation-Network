import itertools
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib import rnn

class Model(object):

    def __init__(self,
                 config,
                 is_train=True,
                 seed=9,
                 c_word_embed=32,
                 q_word_embed=32,
                 context_vocab_size = 124,
                 question_vocab_size = 88,
                 answer_vocab_size=159):

        self.batch_size = config.batch_size
        self.seed = seed
        self.c_max_len = config.c_max_len  # 20
        self.s_max_len = config.s_max_len  # 12
        self.q_max_len = config.q_max_len  # 12
        self.mask_index = 0
        self.s_input_step = config.s_max_len
        self.s_hidden = config.s_hidden  # 32
        self.q_input_step = config.q_max_len
        self.q_hidden = config.q_hidden  # 32
        self.c_word_embed = c_word_embed
        self.q_word_embed = q_word_embed
        self.context_vocab_size = context_vocab_size + 1  # consider masking
        self.question_vocab_size = question_vocab_size + 1  # consider masking
        self.answer_vocab_size = answer_vocab_size
        self.context = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.c_max_len, self.s_max_len],
            name="context"
        )
        self.context_real_len = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.c_max_len],
            name="context_real_length"
        )
        self.sentence = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.s_max_len],
            name="sentence"
        )
        self.question = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size, self.q_max_len],
            name="question"
        )
        self.question_real_len = tf.placeholder(
            dtype=tf.int32,
            shape=[self.batch_size],
            name="question_real_length"
        )
        self.label = tf.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, self.c_max_len, self.c_max_len],
            name="label"
        )
        self.answer = tf.placeholder(
            dtype=tf.float32,
            shape=[self.batch_size, self.answer_cand],
            name="answer"
        )
        self.is_training = tf.placeholder(
            dtype=tf.bool,
            name="is_training"
        )
        self.embed_matrix()
        self.pred = self.build(is_train=is_train)

        self.correct = tf.equal(tf.argmax(self.pred , axis=1), tf.argmax(self.answer, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.answer))

    def embed_matrix(self):
        self.c_word_embed_matrix = tf.Variable(
            tf.random_uniform(shape=[self.c_vocab_size, self.c_word_embed],
                              minval=-1,
                              maxval=1,
                              seed= self.seed))
        self.q_word_embed_matrix = tf.Variable(
            tf.random_uniform(shape=[self.q_vocab_size, self.q_word_embed],
                              minval=-1,
                              maxval=1,
                              seed=self.seed))

    def contextLSTM(c, l, c_real_len, reuse = False, scope = "ContextLSTM"):

        def sentenceLSTM(s,
                         s_real_len,
                         reuse = reuse,
                         scope = "sentenceLSTM"):
            """
            embedding sentence

            Arguments
                s: sentence (word index list), shape = [batch_size*20, 12]
                s_real_len: length of the sentence before zero padding, int32

            Returns
                embedded_s: embedded sentence, shape = [batch_size*20, 32]
            """
            embedded_sentence_word = tf.nn.embedding_lookup(self.c_word_embed_matrix, s)
            s_input = tf.unstack(embedded_sentence_word, num = self.s_max_len, axis = 1)
            lstm_cell = rnn.BasicLSTMCell(self.s_hidden, reuse = reuse)
            outputs, _ = rnn.static_rnn(lstm_cell, s_input, dtype = tf.float32, scope = scope)

            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1,0,2])
            index = tf.range(0, self.batch_size* self.c_max_len) * (self.s_max_len) + (s_real_len - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.s_hidden]), index)
            return outputs

        """
        Args
            c: list of sentences, shape = [batch_size, 20, 12]
            l: list of labels, shape = [batch_size, 20, 20]
            c_real_len: list of real length, shape = [batch_size, 20]

        Returns
            tagged_c_objects: list of embedded sentence + label, shape = [batch_size, 52] 20개
            len(tagged_c_objects) = 20
        """
        sentences = tf.reshape(c, shape = [-1, self.s_max_len])
        real_lens = tf.reshape(c_real_len, shape= [-1])
        labels = tf.reshape(l, shape = [-1, self.c_max_len])

        s_embedded = sentenceLSTM(sentences, real_lens, reuse = reuse)
        c_embedded = tf.concat([s_embedded, labels], axis=1)
        c_embedded = tf.reshape(c_embedded, shape = [self.batch_size, self.c_max_len, self.c_max_len + self.c_word_embed])
        tagged_c_objects = tf.unstack(c_embedded, axis=1)
        return tagged_c_objects

    def questionLSTM(q, q_real_len, reuse = False, scope= "questionLSTM"):
        """
        Args
            q: zero padded qeustions, shape=[batch_size, q_max_len]
            q_real_len: original question length, shape = [batch_size, 1]

        Returns
            embedded_q: embedded questions, shape = [batch_size, q_hidden(32)]
        """
        embedded_q_word = tf.nn.embedding_lookup(self.q_word_embed_matrix, q)
        q_input = tf.unstack(embedded_q_word, num = self.q_max_len, axis=1)
        lstm_cell = rnn.BasicLSTMCell(self.q_hidden, reuse = reuse)
        outputs, _ = rnn.static_rnn(lstm_cell, q_input, dtype = tf.float32, scope = scope)

        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1,0,2])
        index = tf.range(0, self.batch_size) * (self.q_max_len) + (q_real_len - 1)
        return outputs

    def convert_to_RN_input(embedded_c, embedded_q):
        """
        Args
            embedded_c: output of contextLSTM, 20 length list of embedded sentences
            embedded_q: output of questionLSTM, embedded question

        Returns
            RN_input: input for RN g_theta, shape = [batch_size*190, (52+52+32)]
            considered batch_size and all combinations
        """
        # 20 combination 2 --> total 190 object pairs
        object_pairs = list(itertools.combinations(embedded_c, 2))
        # concatenate with question
        RN_inputs = []
        for object_pair in object_pairs:
            RN_input = tf.concat([object_pair[0], object_pair[1], embedded_q], axis=1)
            RN_inputs.append(RN_input)

        return tf.concat(RN_inputs, axis=0)

    def batch_norm_relu(inputs, output_shape, phase = True, scope = None, activation = True):
        with tf.variable_scope(scope):
            h1 = fully_connected(inputs, output_shape, activation_fn= None, scope ="dense")
            h2 = batch_norm(h1, decay = 0.95, center = True, scale = True,
                            is_training= phase, scope = 'bn', updates_collections=None)
            if activation:
                out = tf.nn.relu(h2, 'relu')
            else:
                out = h2
            return out

    def g_theta(RN_input, scope='g_theta', reuse = True, phase = True):
        """
        Args
            RN_input: [o_i, o_j, q], shape = [batch_size*190, 136]

        Returns
            g_output: shape = [190, batch_size, 256]
        """
        input_dim = RN_input.shape[1]
        g_units = [256,256,256,256]
        with tf.variable_scope(scope, reuse = reuse) as scope:
            g_1 = batch_norm_relu(RN_input, g_units[0], scope= 'g_1', phase = phase)
            g_2 = batch_norm_relu(g_1, g_units[1], scope='g_2', phase=phase)
            g_3 = batch_norm_relu(g_2, g_units[2], scope='g_3', phase=phase)
            g_4 = batch_norm_relu(g_3, g_units[3], scope='g_4', phase=phase)
        g_output = tf.reshape(g_4, shape= [-1, self.batch_size, g_units[3]])
        return g_output

    def f_phi(g, scope = "f_phi", reuse = True , phase = True):
        """
        Args
            g: g_theta result, shape = [190, batch_size, 256]

        Returns
            f_output: shape = [batch_size, 159]
        """
        f_input = tf.reduce_sum(g, axis=0)
        f_units = [256,512,159]
        with tf.variable_scope(scope, reuse = reuse) as scope:
            f_1 = batch_norm_relu(f_input, f_units[0], scope = "f_1", phase = phase)
            f_2 = batch_norm_relu(f_1, f_units[1], scope = "f_2", phase = phase)
            f_3 = batch_norm_relu(f_2, f_units[2], scope = "f_3", phase = phase)
        return f_3

    def build(self, is_train = True):
        embedded_c = self.contextLSTM(self.context, self.label, self.context_real_len, reuse = None)
        embedded_q = self.questionLSTM(self.question, self.question_real_len, reuse = None)
        RN_input = self.convert_to_RN_input(embedded_c, embedded_q)
        f_input = self.g_theta(RN_input, reuse = None, phase = self.is_training)
        pred = self.f_phi(f_input, reuse = None, phase = self.is_training)
        return pred