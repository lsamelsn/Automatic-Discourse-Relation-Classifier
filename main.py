import string
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pdtb2.pdtb2 import CorpusReader, Datum
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# dictionary to convery relations to ints
relation2int = {"Explicit": 0, "Implicit": 1, "AltLex": 2, "EntRel": 3, "NoRel": 4}

# confusion matrix
cmatrix = [[0 for i in range(5)] for j in range(5)]

stopwords = set(stopwords.words("english"))
learning_model = None
s = " "


def pre_process(raw_text):
	global stopwords
	global s

	custom_del_chars = ["``", "''", "--"]

	retval = ""

	# first make sure contractions are delt with, delete apost
	raw_text = raw_text.replace("'", "")

	# tokenize sentence
	retval_toked = word_tokenize(raw_text)

	# filter bas stuff out token by token
	for i, tok in enumerate(retval_toked):
		# remove punctuation, stopwords, and numbers
		if tok not in string.punctuation and tok not in stopwords and \
						not tok.isdigit() and tok not in custom_del_chars:
			# lowercase
			tok = tok.lower()

			# Lemmatize words
			#tok = WordNetLemmatizer().lemmatize(tok)

			if i == 0:
				retval += tok
			else:
				retval += s + tok

	return retval


def get_corpus():
	retval = []

	iterator = CorpusReader('./pdtb2/pdtb2.csv').iter_data(display_progress=False)

	# iterate through entire corpus, sentence by sentence
	while True:
		# stop
		d = next(iterator, None)
		if d == None:
			break
		else:
			# get metadata
			raw_text_arg1 = d.Arg1_RawText.encode('ascii', 'ignore').decode('ascii')
			raw_text_arg2 = d.Arg2_RawText.encode('ascii', 'ignore').decode('ascii')
			relation = d.Relation
			connective = d.Connective_RawText

			# pre processing stage
			cleaned_arg1 = pre_process(raw_text_arg1)
			cleaned_arg2 = pre_process(raw_text_arg2)

			# add to retval as dict
			retval.append({\
				"arg1": cleaned_arg1, \
				"arg2": cleaned_arg2, \
				"relation": relation, \
				"connective": connective\
			 })

	return retval


def convert_to_sklearn_input(corpus_dict):
	global s

	corpus_array = []
	relations_array = []

	for metadata in corpus_dict:
		arg1 = metadata["arg1"]
		arg2 = metadata["arg2"]
		relation = metadata["relation"]
		connective = metadata["connective"]

		if connective == None:
			input_text = arg1 + s + arg2
		else:
			input_text = arg1 + s + connective + s + arg2

		corpus_array.append(input_text)
		relations_array.append(relation2int[relation])

	return corpus_array, relations_array


def convert_data_to_tfdif_format(x_train, x_test, y_train, y_test):
	# function: tfdif vector converter
	cv = TfidfVectorizer(min_df=4)

	# convert to tfdif
	x_traincv = cv.fit_transform(x_train)
	x_testcv = cv.transform(x_test)

	y_train = y_train
	y_test = y_test

	return x_traincv, x_testcv, y_train, y_test


def evaluate(num_total, num_correct):
	global cmatrix

	print(np.matrix(cmatrix))

	# TruePos / (TruePos + FalsePos)
	precision = None

	# TruePos / (TruePos + FalseNeg)
	recall = None

	# total_correct / total_correct_negative
	accuracy = float(num_correct/num_total)

	return precision, recall, accuracy


def run_model_predictions(x_testcv, y_test):
	global learning_model
	global cmatrix

	precision = None
	recall = None
	accuracy = None	

	predictions = learning_model.predict(x_testcv)

	num_correct = 0	
	for i in range(len(predictions)):
		prediction_int = predictions[i]
		actual_int = y_test[i]

		# POSTIVE
		if prediction_int == actual_int:
			num_correct += 1
			cmatrix[actual_int][actual_int] += 1
		# NEGATIVE
		else:
			cmatrix[actual_int][prediction_int] += 1

	return len(predictions), num_correct


def	run_model_on_data(x_traincv, y_train):
	global learning_model 

	learning_model = MultinomialNB()

	learning_model.fit(x_traincv, y_train)


def main():
	# get the corpus and put into dictionary and org by metadata
	corpus_dict = get_corpus()

	# get into sklearn format
	corpus_array, relations_array = convert_to_sklearn_input(corpus_dict)

	# split my data
	x_train, x_test, y_train, y_test = \
		train_test_split(corpus_array, relations_array, test_size=0.2, random_state=0)

	# convert my data into tfdif format
	x_traincv, x_testcv, y_train, y_test = \
		convert_data_to_tfdif_format(x_train, x_test, y_train, y_test)

	# run the model off the converted data
	run_model_on_data(x_traincv, y_train)

	# run model on the test data
	num_total, num_correct = run_model_predictions(x_testcv, y_test)

	# evaluate how the model performs
	precision, recall, accuracy = evaluate(num_total, num_correct)

	print("Precision: ", precision)
	print("Recall: ", recall)
	print("Accuracy: ", accuracy)


main()
