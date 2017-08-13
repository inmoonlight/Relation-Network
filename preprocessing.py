import numpy as np
import os
import pickle
import re
import sys
import argparse

class Preprocess():


    def __init__(self, path_to_babi):
        # path_to_babi example: '././babi_original'
        self.path_to_babi = os.path.join(path_to_babi, "tasks_1-20_v1-2/en-valid-10k")
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None
        self.path_to_processed = "./babi_processed"
        self._c_word_set = set()
        self._q_word_set = set()
        self._a_word_set = set()
        self._cqa_word_set = set()
        self.c_max_len = 20
        self.s_max_len = 0
        self.q_max_len = 0
        self.mask_index = 0

    def set_path(self):
        """
        set list of train, val, and test dataset paths

        Returns
            train_paths: list of train dataset paths for all task 1 to 20
            val_paths: list of val dataset paths for all task 1 to 20
            test_paths: list of test dataset paths for all task 1 to 20
        """
        train_paths = []
        val_paths = []
        test_paths= []
        for dirpath, dirnames, filenames in os.walk(self.path_to_babi):
            for filename in filenames:
                if 'train' in filename:
                    train_paths.append(os.path.join(dirpath, filename))
                elif 'val' in filename:
                    val_paths.append(os.path.join(dirpath, filename))
                else:
                    test_paths.append(os.path.join(dirpath, filename))
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths

    def _split_paragraphs(self, path_to_file):
        """
        split into paragraphs as babi dataset consists of multiple 1~n sentences

        Args
            file_path: path of the data

        Returns
            paragraphs: list of paragraph
        """
        with open(path_to_file, 'r') as f:
            babi = f.readlines()
        paragraph = []
        paragraphs = []
        alphabet = re.compile('[a-zA-Z]')
        for d in babi:
            if d.startswith('1 '):
                if paragraph:
                    paragraphs.append(paragraph)
                paragraph = []
            mark = re.search(alphabet, d).span()[0]
            paragraph.append(d[mark:])
        return paragraphs

    def _split_clqa(self, paragraphs, show_print= True):
        """
        for each paragraph, split into context, label, question and answer

        Args
            paragraphs: list of paragraphs

        Returns
            context: list of contexts
            label: list of labels
            question: list of questions
            answer: list of answers
        """
        context = []
        label = []
        question = []
        answer = []
        for paragraph in paragraphs:
            for i, sent in enumerate(paragraph):
                if '?' in sent:
                    related_para = [para.strip().lower() for para in paragraph[:i] if '?' not in para][::-1]
                    if len(related_para) > 20:
                        related_para = related_para[:20]
                    context.append(related_para)
                    label.append([i for i in range(len(related_para))])
                    q_a_ah = sent.split('\t')
                    question.append(q_a_ah[0].strip().lower())
                    answer.append(q_a_ah[1].strip().lower())
        # check
        if show_print:
            if (len(question) == len(answer)) & (len(answer) == len(context)) & (len(context) == len(label)):
                print("bAbI is well separated into question, answer, context, and label!")
                print("total: {}".format(len(label)))
            else:
                print("Something is missing! check again")
                print("the number of questions: {}".format(len(question)))
                print("the number of answers: {}".format(len(answer)))
                print("the number of contexts: {}".format(len(context)))
                print("the number of labels: {}".format(len(label)))
        return context, label, question, answer

    def split_all_clqa(self, paths, show_print= True):
        """
        merge all 20 babi tasks into one dataset

        Args
            paths: list of path of 1 to 20 task dataset

        Returns
            contexts: list of contexts of all 20 tasks
            labels: list of labels of all 20 tasks
            questions: list of questions of all 20 tasks
            answers: list of answers of all 20 tasks
        """
        if paths == None:
            print('path is None, run set_path() first!')
        else:
            contexts = []
            labels = []
            questions = []
            answers = []
            for path in paths:
                if show_print:
                    print('=================')
                paragraphs = self._split_paragraphs(path)
                if show_print:
                    print("data: {}".format(os.path.basename(path)))
                context, label, question, answer = self._split_clqa(paragraphs, show_print=show_print)
                contexts.extend(context)
                labels.extend(label)
                questions.extend(question)
                answers.extend(answer)
            return contexts, labels, questions, answers

    def _set_word_set(self):
        c_word_set = set()
        q_word_set = set()
        a_word_set = set()
        train_context, train_label, train_question, train_answer = self.split_all_clqa(self.train_paths, show_print=False)
        val_context, val_label, val_question, val_answer = self.split_all_clqa(self.val_paths, show_print=False)
        test_context, test_label, test_question, test_answer = self.split_all_clqa(self.test_paths, show_print=False)
        list_of_context = [train_context, val_context, test_context]
        list_of_question = [train_question, val_question, test_question]
        list_of_answer = [train_answer, val_answer, test_answer]
        for list_ in list_of_context:
            for para in list_:
                for sent in para:
                    sent = sent.replace(".", " .")
                    sent = sent.replace("?", " ?")
                    sent = sent.split()
                    c_word_set.update(sent)
        for list_ in list_of_question:
            for sent in list_:
                sent = sent.replace(".", " .")
                sent = sent.replace("?", " ?")
                sent = sent.split()
                q_word_set.update(sent)
        for answers in list_of_answer:
            for answer in answers:
                answer = answer.split(',')
                a_word_set.update(answer)
        a_word_set.add(',')
        self._c_word_set = c_word_set
        self._q_word_set = q_word_set
        self._a_word_set = a_word_set
        self._cqa_word_set = c_word_set.union(q_word_set).union(a_word_set)

    def _index_context(self, contexts):
        c_word_index = dict()
        for i, word in enumerate(self._c_word_set):
            c_word_index[word] = i+1 # index 0 for zero padding
        indexed_cs = []
        for context in contexts:
            indexed_c = []
            for sentence in context:
                sentence = sentence.replace(".", " .")
                sentence = sentence.replace("?", " ?")
                sentence = sentence.split()
                indexed_s = []
                for word in sentence:
                    indexed_s.append(c_word_index[word])
                indexed_c.append(indexed_s)
            indexed_cs.append(np.array(indexed_c))
        return indexed_cs

    def _index_label(self, labels):
        indexed_ls = []
        for label in labels:
            indexed_ls.append(np.eye(self.c_max_len)[label])
        return indexed_ls

    def _index_question(self, questions):
        q_word_index = dict()
        for i, word in enumerate(self._q_word_set):
            q_word_index[word] = i+1 # index 0 for zero padding
        indexed_qs = []
        for sentence in questions:
            sentence = sentence.replace(".", " .")
            sentence = sentence.replace("?", " ?")
            sentence = sentence.split()
            indexed_s = []
            for word in sentence:
                indexed_s.append(q_word_index[word])
            indexed_qs.append(np.array(indexed_s))
        return indexed_qs

    def _index_answer(self, answers):
        a_word_index = dict()
        a_word_dict = dict()
        for i, word in enumerate(self._cqa_word_set):
            a_word_dict[i] = word
            if word in self._a_word_set:
                answer_one_hot = np.zeros(len(self._cqa_word_set), dtype=np.float32)
                answer_one_hot[i] = 1
                a_word_index[word] = answer_one_hot
        indexed_as = []
        for answer in answers:
            if ',' in answer:
                multiple_answer = [a_word_index[',']]
                for a in answer.split(','):
                    indexed_a = a_word_index[a]
                    multiple_answer.append(indexed_a)
                indexed_as.append(np.sum(multiple_answer, axis=0))
            else:
                indexed_a = a_word_index[answer]
                indexed_as.append(indexed_a)

        if not os.path.exists(self.path_to_processed):
            os.makedirs(self.path_to_processed)

        with open(os.path.join(self.path_to_processed, 'answer_word_dict.pkl'), 'wb') as f:
            pickle.dump(a_word_dict, f)
        return indexed_as

    def masking(self, context_index, label_index, question_index):
        context_masked = []
        question_masked = []
        label_masked = []
        context_real_len = []
        question_real_len = []
        # cs: one context
        for cs, l, q in zip(context_index, label_index, question_index):
            context_masked_tmp = []
            context_real_length_tmp = []
            # cs: many sentences
            for context in cs:
                context_real_length_tmp.append(len(context))
                diff = self.s_max_len - len(context)
                if (diff > 0):
                    context_mask = np.append(context, [self.mask_index]*diff, axis=0)
                    context_masked_tmp.append(context_mask.tolist())
                else:
                    context_masked_tmp.append(context)
            diff_c = self.c_max_len - len(cs)
            context_masked_tmp.extend([[0]*self.s_max_len]*diff_c)
            context_masked.append(context_masked_tmp)

            diff_q = self.q_max_len - len(q)
            question_real_len.append(len(q))
            question_masked_tmp = np.array(np.append(q, [self.mask_index]*diff_q, axis=0))
            question_masked.append(question_masked_tmp.tolist())
            
            diff_l = self.c_max_len - len(l)
            label_masked_tmp = np.append(l, np.zeros((diff_l, self.c_max_len)), axis= 0)
            label_masked.append(label_masked_tmp.tolist())
            context_real_length_tmp.extend([0]*diff_l)
            context_real_len.append(context_real_length_tmp)
        return context_masked, question_masked, label_masked, context_real_len, question_real_len


    def load(self, mode):
        if mode == 'train':
            path = self.train_paths
        elif mode == 'val':
            path = self.val_paths
        else:
            path = self.test_paths

        contexts, labels, questions, answers = self.split_all_clqa(path)
        context_index = self._index_context(contexts)
        label_index = self._index_label(labels)
        question_index = self._index_question(questions)
        answer_index = self._index_answer(answers)

        if mode == 'train':
            # check max sentence length
            for context in context_index:
                for sentence in context:
                    if len(sentence) > self.s_max_len:
                        self.s_max_len = len(sentence)
            # check max question length
            for question in question_index:
                if len(question) > self.q_max_len:
                    self.q_max_len = len(question)

        context_masked, question_masked, label_masked, context_real_len, question_real_len = self.masking(context_index, label_index, question_index)
        # check masking
        cnt = 0
        for c, q, l in zip(context_masked, question_masked, label_masked):
            for context in c :
                if (len(context) != self.s_max_len) | (len(q) != self.q_max_len) | (len(l) != self.c_max_len):
                    cnt += 1
        if cnt == 0:
            print("Masking success!")
        else:
            print("Masking process error")
        dataset = (question_masked, answer_index, context_masked, label_masked, context_real_len, question_real_len)
        if not os.path.exists(self.path_to_processed):
            os.makedirs(self.path_to_processed)
        with open(os.path.join(self.path_to_processed, mode + '_dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

def get_args_parser():
    """
    python preprocessing.py --path ../ --batch_size 64 --hidden_units 32 --learning_rate 2e-4 --iter_time 150 --display_step 100
    :return:
    """
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--path', '--path_to_babi')
    _parser.add_argument('--batch_size', '--batch_size')
    _parser.add_argument('--hidden_units', '--hidden_units')
    _parser.add_argument('--learning_rate', '--learning_rate')
    _parser.add_argument('--iter_time', '--iter_time')
    _parser.add_argument('--display_step', '--display_step')
    return _parser

def default_write(f, string, default_value):
    if string == None:
        f.write(str(default_value) + "\t")
    else:
        f.write(str(string) + "\t")

def main():
    args = get_args_parser().parse_args()
    preprocess = Preprocess(args.path)
    preprocess.set_path()
    preprocess._set_word_set()
    preprocess.load(mode='train')
    preprocess.load(mode='val')
    preprocess.load(mode='test')

    with open(os.path.join('config.txt'), 'w') as f:
        f.write(str(preprocess.c_max_len)+"\t")
        f.write(str(preprocess.s_max_len)+"\t")
        f.write(str(preprocess.q_max_len)+"\t")
        f.write(str(preprocess.path_to_processed)+'\t')
        default_write(f, args.batch_size, 64)
        default_write(f, args.hidden_units, 32)
        default_write(f, args.learning_rate, 2e-4)
        default_write(f, args.iter_time, 150)
        default_write(f, args.display_step, 100)
if __name__ == '__main__':
    main()
