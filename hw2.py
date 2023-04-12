#Assignment 2: Naives Bayes Sentiment Analysis

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist, LaplaceProbDist
import nltk
import math
import json
import pandas as pd


# Read in the data CSV files and turning it into data frame
train_df = pd.read_csv('train.csv', header=None, names=['id', 'review', 'label'])
test_df = pd.read_csv('test.csv', header=None, names=['id', 'review', 'label'])

#turn data frame into list 
train_set=train_df.values.tolist()
test_set=test_df.values.tolist()

#opening vocab and tokenizing
with open('vocab.txt', 'r') as f:
	# Read the lines of the file into a list
	text = f.read()

vocab=word_tokenize(text.lower())

#tokenizing trainset reviews into word, label tupple
def tokenize_reviews():
	positive_words=[]
	negative_words=[]
	for list in train_set:
		review_tokenized=word_tokenize(list[1])
		if list[2]=='pos':
			for word in review_tokenized:
				positive_words.append(word.lower())
		else:
			for word in review_tokenized:
				negative_words.append(word.lower())

#returns seperate list of positive and neg words that are lowercased
	return positive_words,negative_words


positive_words, negative_words=tokenize_reviews()


#total negative and pos words
total_pos=len(positive_words)
total_neg=len(negative_words)


#word count in positive and negative
pos_freq=FreqDist(positive_words)
neg_freq=FreqDist(negative_words)



#getting laplace probabilities for words in vocab. 
def probability():
	pos_prob={}
	neg_prob={}
	for word in vocab:
		poscount=pos_freq.get(word.lower())
		negcount=neg_freq.get(word.lower())
		#if words in vocab are not in training set
		if poscount is None:
			poscount=0
		if negcount is None:
			negcount=0
		
		posprob=(poscount+1)/(len(vocab)+total_pos)  #p(w|pos)=count word in pos+1/total positive words+total vocab
		negprob=(negcount+1)/(len(vocab)+total_neg) #p(w|neg)=count word in neg+1/total neg words+total vocab
		pos_prob[word]=posprob
		neg_prob[word]=negprob
	return pos_prob, neg_prob

"""
#saving into json file to save run time
pos_prob, neg_prob = probability()
with open('pos_prob.json', 'w') as f:
    json.dump(pos_prob, f)
with open('neg_prob.json', 'w') as f:
    json.dump(neg_prob, f)
"""
#opening probability dictionaries
with open('pos_prob.json', 'r') as f:
    pos_prob=json.load( f)
with open('neg_prob.json', 'r') as f:
    neg_prob= json.load( f)



#classifying reviews from test set
def classify():
	classified_review=[]

	#tokenizing reviews from test set then finding log probilities of words in them
	for l in test_set:
		#setting intial log probability to the log of 1/2
		totalpos=math.log(0.5)
		totalneg=math.log(0.5)
		review=l[1]
		words = word_tokenize(review.lower())

		for word in words:
			posprob=pos_prob.get(word.lower())
			negprob=neg_prob.get(word.lower())
			#accounting for unknown words int
			if posprob is None: 
				posprob = 1 / (totalpos + len(vocab))
			if negprob is None:
				negprob = 1 / (totalneg + len(vocab))
			else:
				totalpos+=math.log(posprob)
				totalneg+=math.log(negprob)
		
		if totalneg>totalpos:
			classified_review.append('neg')
		else:
			classified_review.append('pos')
	
	return classified_review


"""

# Saving classified review into a JSON file to reduce compute time(so i dont have to keep running the classify method)
classified_review = classify()
with open('classified_review.json', 'w') as f:
    json.dump(classified_review, f)


"""


with open('classified_review.json', 'r') as f:
    classified_review= json.load(f)





#finding precision, recall, and accuracy of entire model
def evaluate():
	actual_labels = test_df['label'].tolist()
	predicted_labels=classified_review
	correct_predictions = 0
	true_positives = 0
	false_positives = 0
	true_negatives = 0
	false_negatives = 0

	for i in range(len(actual_labels)):
		if actual_labels[i] == predicted_labels[i]: #checking for accuracy, true positives and negatives
			correct_predictions += 1
			if actual_labels[i] == 'pos':
				true_positives += 1
			else:
				true_negatives += 1
		else: #if not the same, checking for false positive and negatives
			if actual_labels[i] == 'pos': 
				false_negatives += 1
			else:
				false_positives += 1
	#recall=tp/tp+fn= tp/totalpositives
	recall=true_positives/(true_positives+false_negatives)*100
	#Precision=TP / (TP + FP)
	precision=true_positives/(true_positives+false_positives)*100
	accuracy=(correct_predictions/len(actual_labels) )*100

	return recall, precision, accuracy


#finding precision, recall, and accuracy for positive class
def evaluate_pos_class():
	actual_labels = test_df['label'].tolist()
	predicted_labels=classified_review
	correct_predictions = 0
	true_positives = 0
	false_positives = 0
	true_negatives = 0
	false_negatives = 0

	for i in range(12001):
		if actual_labels[i] == predicted_labels[i]: #checking for accuracy, true positives and negatives
			correct_predictions += 1
			if actual_labels[i] == 'pos':
				true_positives += 1
			else:
				true_negatives += 1
		else: #if not the same, checking for false positive and negatives
			if actual_labels[i] == 'pos': 
				false_negatives += 1
			else:
				false_positives += 1
	#recall=tp/tp+fn= tp/totalpositives
	recall=true_positives/(true_positives+false_negatives)*100
	#Precision=TP / (TP + FP)
	precision=true_positives/(true_positives+false_positives)*100
	accuracy=(correct_predictions/len(actual_labels) )*100

	return recall, precision, accuracy


print(evaluate_pos_class())





