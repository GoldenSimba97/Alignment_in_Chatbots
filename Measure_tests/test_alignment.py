# All code in this file is obtained from: https://bitbucket.org/sagieske/project_cop/src/master/IAC/iac_v1.1/research_code/qrpairs.py
# This has been done with permission from Sharon Gieske. The code has been changed to fit the purpose of this Thesis.

import csv
import json
import os
import sys
from collections import defaultdict
import numpy as np
import operator


class QRPair:
    # Add question text to object
	def add_question(self, question):
		self.question = question

    # Add response text to object
	def add_response(self, response):
		self.response = response

    # Compute alignment score of a question-response pair. The baseline of the user history is determined to ensure
    # that preferences for certain markers do not influence the result. If the question is empty, if none of the
    # features is found in text or if the feature count divided by the length is zero, there is no alignment.
    # The actual alignment between question and response can be calculated using four different alternatives.
	def compute_alignment(self, features, history, questions_list, smoothing, weight_vector, alternative="matching"):
        # Determine baseline of user history
		base_prob = get_baseline(features, questions_list)
		other_base_prob = other_baseline(features, questions_list)
		conditional_prob = self.response_prob(features)

        # Used for testing purposes only
		# conditional_prob = self.response_prob_binary(features)

		value_vector = np.array([other_base_prob, base_prob])
		scalarized = np.dot(weight_vector, value_vector)

        # Get needed information from question, response and user history
		response_text = getWords(self.response)
		self.response_length = len(response_text)
		question_text = getWords(self.question)
		self.question_length = len(question_text)
		history_text = getWords(history)

		# If question is empty there is no alignment
		if len(question_text) == 0:
			self.alignment = 0
			return self.alignment

        # If none of the features is found in the text there is no alignment
		r_count = feature_count(history_text, features)
		if r_count == 0 and features[0] != "number_posts":
			self.alignment = 0
			return self.alignment

        # If the feature count divided by the length is zero there is no alignment
		q_m = (r_count + smoothing) / (len(question_text) + smoothing)
		if q_m == 0:
			self.alignment = 0
			return self.alignment

        # Calculate alignment using four different alternatives.
		r_m = (feature_count(response_text, features) + smoothing) / (len(response_text) + smoothing)
		match = self.matching(q_m, r_m)
		if alternative == "alternative":
			self.alignment = r_m - base_prob
		elif alternative == "matching":
			self.alignment = match - base_prob
		elif alternative == "original":
			self.alignment = conditional_prob - base_prob
		elif alternative == "final":
			self.alignment = conditional_prob - scalarized
		return self.alignment

    # Match question and response
	def matching(self, q_m, r_m):
		if q_m == r_m:
			return 1
		else:
			return min(q_m, r_m)

    # Calculate the frequency of features that occur in the response text
	def response_prob(self, features):
		total_markers = 0.0
		response_text = getWords(self.response)
		for feature in features:
			if feature in response_text:
				total_markers += 1
		return total_markers/len(response_text)

    # See if the features appear in the utterance at all (binary)
    # Used for testing purposes only.
	def response_prob_binary(self, features):
		response_text = getWords(self.response)
		total_markers = False
		for feature in features:
			if feature in response_text:
				total_markers = True
		return int(total_markers)

# Compute the baseline probabilities for all questions and features.
def other_baseline(features, questions_list):
	exists = False
	total_exists = 0.0
	counter = 0
	# Check number of questions which hold marker
	for question in questions_list:
		text = getWords(question)
		counter += len(text)
		# Loop over list of features, if 1 marker is found in text, go to next response
		for feature in features:
			if feature in text:
				exists = True
				total_exists += 1
	return total_exists/counter

# Compute the baseline probabilities for all questions and features. Count the number of features
# that appear (binary) in the question and divide by the number of utterances.
def get_baseline(features, questions_list):
	total_markers = 0.0
	# Check number of responses which hold marker
	for question in questions_list:
		text = getWords(question)
		# Loop over list of features, if 1 marker is found in text, go to next response
		for feature in features:
			if feature in text:
				total_markers += 1
				break
		total_posts = 2
	return total_markers/total_posts

# Count the total number of features found in the text
def feature_count(text, features):
	feature_count = 0.0
	for word in text:
		if word in features:
			feature_count += 1.0
	return feature_count

# Written by Julian. Proces the text and return the words.
def getWords(post_text):
	text = post_text.split(" ")
	text2 = []
	for word in text:
		word = word.strip()
		# Remove non-words:
		if not word in ['#','/','[',']','}','--',',','-/','+','-','((','))']:
			if not (word.startswith('{') or
					word.startswith('<') or word.endswith('>')
					or word.startswith('(')):
				if word:
					text2.append(word)
	text2 = [word.lower() for word in text2]
	text2 = [word.strip().strip(',.-#') for word in text2]
	return text2

# Read a list of feature words from file_name and return a list of all the words.
def get_features_from_file(file_name):
	feature_list = []
	f = open(file_name, 'r')
	for line in f:
		feature_list.append(line.split("\n")[0])
	f.close()
	return feature_list



# Used for testing purposes only
if __name__=="__main__":
	weight_vector = np.array([0.5, 0.5])
	# weight_vector = np.array([0.8, 0.2])
	markers = ['adverbs', 'articles', 'auxiliaryverbs', 'conjunctions', 'impersonalpronouns', 'personalpronouns', 'prepositions', 'quantifiers', 'number_posts']
	smoothing = 0.1
	alignment_scores = {}
	scores = 0
	for mark in markers:
		if mark != "number_posts" :
			features = get_features_from_file('coordination_markers/'+mark+'.txt')
		else:
			features = [mark]

		response_list = ['See you later my friend.', 'Looking forward to seeing you again soon.', 'Have a great journey until next time.', 'Looking forward to our next time together.', 'Goodbye', "I don't like to say goodbye.", "It's been a pleasure to be in your company."]
		question = "bye"
		questions_list = ["Hi! My name is Kim and I am 21 years old. I work as a receptionist at a physical therapy firm. I hate doing dishes, because they are so very dirty. I love playing volleyball, reading, watching tv series and shopping.", "you cunt, why won't you tell me something", "do you love me", "do you have a sense of humour", "bye"]
		history = "Hi! My name is Kim and I am 21 years old. I work as a receptionist at a physical therapy firm. I hate doing dishes, because they are so very dirty. I love playing volleyball, reading, watching tv series and shopping.. you cunt, why won't you tell me something. do you love me. do you have a sense of humour. bye"

		pair_dictionary = {}
		for response in response_list:
			qr_list = []
			for question in questions_list:
				qrpair = QRPair()
				qrpair.add_question(question)
				qrpair.add_response(response)
				qr_list.append(qrpair)
			pair_dictionary[response] = qr_list

		alignment_dict = {}
		# Loop through all qrpairs again to compute alignment
		for k, qr_list in pair_dictionary.items():
			alignment_list = []
			for qrpair in qr_list:
				alignment_list.append(qrpair.compute_alignment(features, history, questions_list, smoothing, weight_vector, alternative="final"))
			alignment_dict[k] = sum(alignment_list)

		for key, score in alignment_dict.items():
			alignment_scores.setdefault(key, []).append(score)

	total_alignment = {key: sum(alignment_scores[key]) for key in alignment_scores}
	print(total_alignment)
	sorted_alignment = sorted(total_alignment.items(), key=operator.itemgetter(1), reverse=True)
	print(sorted_alignment[0][0])
