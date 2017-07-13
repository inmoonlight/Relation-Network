import numpy as np
import os
import pickle
import re

class Preprocess():


	def __init__(self, path_to_babi):
		# path_to_babi example: '././babi_origianl'
		self.path_to_babi = os.path.join(path_to_babi, "tasks_1-20_v1-2/en-valid-10k")
		self.train_paths = None
		self.val_paths = None
		self.test_paths = None
		self.path_to_processed = os.path.join(path_to_babi, "babi_processed")
		self._c_word_set = set()
		self._q_word_set = set()
		self._a_word_set = set()
		self._cqa_word_set = set()
		self._max_context_len = 20

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

	def _split_clqa(self, paragraphs):
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

    def split_all_clqa(self, paths):
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
    	contexts = []
    	labels = []
    	questions = []
    	answers = []
    	if paths == None:
			print('path is None, run set_path() first!')
		else:
			for path in paths:
			    print('=================')    
			    paragraphs = self._split_paragraphs(path)
			    print("data: {}".format(os.path.basename(path)))
			    context, label, question, answer = self._split_clqa(paragraphs)
			    contexts.extend(context)
			    labels.extend(label)
			    questions.extend(question)
			    answers.extend(answer)
	    return contexts, labels, questions, answers

    def _set_word_set(self):
   		c_word_set = set()
   		q_word_set = set()
   		a_word_set = set()
   		train_context, train_label, train_question, train_answer = self.split_all_clqa(self.train_paths)
		val_context, val_label, val_question, val_answer = self.split_all_clqa(self.val_paths)
		test_context, test_label, test_question, test_answer = self.split_all_clqa(self.test_paths)
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
			indexed_ls.append(np.eye(self._max_context_len)[label])
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
		with open(os.path.join(self.path_to_processed, 'answer_word_dict.pkl', 'wb')) as f:
		    pickle.dump(a_word_dict, f)
	    return indexed_as

   	def load_train(self):
   		train_context, train_label, train_question, train_answer = self.split_all_clqa(self.train_paths)
   		train_context_index = self._index_context(train_context)
   		train_label_index = self._index_label(train_label)
   		train_question_index = self._index_question(train_question)
   		train_answer_index = self._index_answer(train_answer)
		train_dataset = (train_question_index, train_answer_index, train_context_index, train_label_index)
		with open(os.path.join(self.path_to_processed, 'train_dataset.pkl', 'wb')) as f:
    		pickle.dump(train_dataset, f)

	def load_val(self):
		val_context, val_label, val_question, val_answer = self.split_all_clqa(self.val_paths)
   		val_context_index = self._index_context(val_context)
   		val_label_index = self._index_label(val_label)
   		val_question_index = self._index_question(val_question)
   		val_answer_index = self._index_answer(val_answer)
		val_dataset = (val_question_index, val_answer_index, val_context_index, val_label_index)
		with open(os.path.join(self.path_to_processed, 'val_dataset.pkl', 'wb')) as f:
    		pickle.dump(val_dataset, f)

	def load_test(self):
		test_context, test_label, test_question, test_answer = self.split_all_clqa(self.test_paths)
		with open(os.path.join(self.path_to_processed, 'test_context.pkl', 'wb')) as f:
		    pickle.dump(test_context, f)
		with open(os.path.join(self.path_to_processed, 'test_question.pkl', 'wb')) as f:
		    pickle.dump(test_question, f)
		with open(os.path.join(self.path_to_processed, 'test_answer.pkl', 'wb')) as f:
		    pickle.dump(test_answer, f)
	    test_context_index = self._index_context(test_context)
   		test_label_index = self._index_label(test_label)
   		test_question_index = self._index_question(test_question)
   		test_answer_index = self._index_answer(test_answer)
		test_dataset = (test_question_index, test_answer_index, test_context_index, test_label_index)
		with open(os.path.join(self.path_to_processed, 'test_dataset.pkl', 'wb')) as f:
    		pickle.dump(test_dataset, f)

