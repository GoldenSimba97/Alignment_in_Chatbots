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
    # The actual alignment between question and response can be calculted using four different alternatives.
	def compute_alignment(self, features, questions_list, smoothing, weight_vector, alternative="matching"):
        # Determine baseline of user history
		base_prob = get_baseline(features, questions_list)
		other_base_prob = other_baseline(features, questions_list)
		conditional_prob = self.response_prob(features)

		value_vector = np.array([other_base_prob, base_prob])
		scalarized = np.dot(weight_vector, value_vector)

        # Get needed information from question and response
		question_text = getWords(self.question)
		response_text = getWords(self.response)
		self.response_length = len(response_text)
		question_text = getWords(self.question)
		self.question_length = len(question_text)

		# If question is empty there is no alignment
		if len(question_text) == 0:
			self.alignment = 0
			return self.alignment

        # If none of the features is found in the text there is no alignment
		q_count = feature_count(question_text, features)
		if q_count == 0 and features[0] != "number_posts":
			self.alignment = 0
			return self.alignment

        # If the feature count divided by the length is zero there is no alignment
		q_m = (q_count + smoothing) / (len(question_text) + smoothing)
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

    # See if the features appear in the utterance at all (binary)
	def response_prob(self, features):
		response_text = getWords(self.response)
		# question_text = getWords(self.question)
		total_markers = False
		for feature in features:
			# if feature in question_text and feature in response_text:
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

		# response_dict = {72.22222222222221: 'Swearing is like using the horn on your car.', 50.0: 'Swearing is often cathartic.'}
		# question = "you cunt, why won't you tell me something"
		# questions_list = ["Hi! My name is Kim and I am 21 years old. I work as a receptionist at a physical therapy firm. I hate doing dishes, because they are so very dirty. I love playing volleyball, reading, watching tv series and shopping.", "you cunt, why won't you tell me something"]

		# response_dict = {50.0: 'I am deeply appreciative of who you are.', 68.75: 'I can see your great capacities and gifts.'}
		# question = "do you love me"
		# questions_list = ["Hi! My name is Kim and I am 21 years old. I work as a receptionist at a physical therapy firm. I hate doing dishes, because they are so very dirty. I love playing volleyball, reading, watching tv series and shopping.", "you cunt, why won't you tell me something", "do you love me"]

		response_dict = {77.77777777777779: 'I do have a fabulous computer sense of humor.', 78.57142857142858: "I'm not aware that I've a SENSE OF HUMOUR at this time."}
		question = "do you have a sense of humour"
		questions_list = ["Hi! My name is Kim and I am 21 years old. I work as a receptionist at a physical therapy firm. I hate doing dishes, because they are so very dirty. I love playing volleyball, reading, watching tv series and shopping.", "you cunt, why won't you tell me something", "do you love me", "do you have a sense of humour"]

		pair_dictionary = {}
		for question in questions_list:
			for key, response in response_dict.items():
				qrpair = QRPair()
				qrpair.add_question(question)
				qrpair.add_response(response)
				pair_dictionary[key] = qrpair

		alignment_dict = {}
		# Loop through all qrpairs again to compute alignment
		for k, qrpair in pair_dictionary.items():
			# some qrpairs do not have the author dict and response dict
			try:
				alignment_dict[k] = qrpair.compute_alignment(features, questions_list, smoothing, weight_vector, alternative="final")
			except:
				pass

		print(alignment_dict)
		for key, score in alignment_dict.items():
			alignment_scores.setdefault(key, []).append(score)

	print(alignment_scores)
	total_alignment = {key: sum(alignment_scores[key]) for key in alignment_scores}
	print(total_alignment)
	score = list(total_alignment.values())[0]
	if all(value == score for value in total_alignment.values()) == True:
		print("Return response with closest formality score to user history")
	else:
		print(response_dict[max(total_alignment.items(), key=operator.itemgetter(1))[0]])
