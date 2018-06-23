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
    # The actual alignment between question and response can be calculated by subtracting the conditional probability
    # by the prior probability.
	def compute_alignment(self, features, history, questions_list, smoothing, weight_vector):
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
		q_count = feature_count(history_text, features)
		if q_count == 0 and features[0] != "number_posts":
			self.alignment = 0
			return self.alignment

        # If the feature count divided by the length is zero there is no alignment
		q_m = (q_count + smoothing) / (len(question_text) + smoothing)
		if q_m == 0:
			self.alignment = 0
			return self.alignment

        # Calculate alignment
		self.alignment = conditional_prob - scalarized
		return self.alignment

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
