import csv
import json
import os
import sys
from collections import defaultdict
import numpy as np
import operator


class QRPair:
	def add_question(self, question):
		'''
		Add question text to object
		'''
		self.question = question

	def add_response(self, response):
		'''
		Add response text to object
		'''
		self.response = response

	def compute_alignment(self, features, questions_list, smoothing, weight_vector, alternative="matching"):
		# base_prob, autor_dict = get_baseline(self.response_author, features, author_dict, response_dict)
		# base_prob = get_baseline(features, response_dict)
		base_prob = get_baseline(features, questions_list)

		# other_base_prob, other_author_dict = other_baseline(self.response_author, features, other_author_dict, response_dict)
		# other_base_prob = other_baseline(features, response_dict)
		other_base_prob = other_baseline(features, questions_list)
		conditional_prob = self.response_prob(features)

		value_vector = np.array([other_base_prob, base_prob])

		scalarized = np.dot(weight_vector, value_vector)

		question_text = getWords(self.question)
		response_text = getWords(self.response)
		self.response_length = len(response_text)
		question_text = getWords(self.question)
		self.question_length = len(question_text)

		# question is empty -> no alignment
		if len(question_text) == 0:
			self.alignment = 0
			return self.alignment

		q_count = feature_count(question_text, features)
		if q_count == 0 and features[0] != "number_posts":
			self.alignment = 0
			return self.alignment
		q_m = (q_count + smoothing) / (len(question_text) + smoothing)

		if q_m == 0:
			self.alignment = 0
			return self.alignment

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

	def matching(self, q_m, r_m):
		if q_m == r_m:
			return 1
		else:
			return min(q_m, r_m)

	def response_prob(self, features):
		'''
		See if the features appear in the utterance at all (binary)
		'''
		response_text = getWords(self.response)
		total_markers = False
		for feature in features:
			if feature in response_text:
				total_markers = True
		return int(total_markers)

def other_baseline(features, questions_list):
	exists = False
	total_exists = 0.0
	counter = 0
	# check number of responses which hold marker
	for question in questions_list:
		text = getWords(question)
		counter += len(text)
		# Loop over list of features, if 1 marker is found in text, go to next response
		for feature in features:
			if feature in text:
				exists = True
				total_exists += 1
	return total_exists/counter

def get_baseline(features, questions_list):
	'''
	Compute the baseline probabilities for this author and
	features: count the number of features that appear (binary)
	in the author's utterances, divide by number of utterances
	'''
	total_markers = 0.0
	# check number of responses which hold marker
	for question in questions_list:
		text = getWords(question)
		# Loop over list of features, if 1 marker is found in text, go to next response
		for feature in features:
			if feature in text:
				total_markers += 1
				break

		total_posts = 2
	return total_markers/total_posts

def feature_count(text, features):
	feature_count = 0.0
	for word in text:
		if word in features:
			feature_count += 1.0
	return feature_count

def getWords(post_text):
	'''
	Written by Julian
	'''
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


# if __name__=="__main__":
# 	weight_vector = np.array([0.8, 0.2])
# 	markers = ['adverbs', 'articles', 'auxiliaryverbs', 'conjunctions', 'impersonalpronouns', 'personalpronouns', 'prepositions', 'quantifiers', 'number_posts']
# 	smoothing = 0.1
# 	alignment_scores = {}
# 	scores = 0
# 	for mark in markers:
# 		if mark != "number_posts" :
# 			features = get_features_from_file('coordination_markers/'+mark+'.txt')
# 		else:
# 			features = [mark]
#
# 		response_dict = {40.0: "You're making your point.", 50.0: 'Swearing is often cathartic.'}
# 		question = "you cunt, why won't you tell me something"
# 		questions_list = ["you cunt, why won't you tell me something", "i love you"]
#
# 		pair_dictionary = {}
# 		for question in questions_list:
# 			for key, response in response_dict.items():
# 				qrpair = QRPair()
# 				qrpair.add_question(question)
# 				qrpair.add_response(response)
# 				pair_dictionary[str(key)] = qrpair
#
# 		alignment_dict = {}
# 		# Loop through all qrpairs again to compute alignment
# 		for k, qrpair in pair_dictionary.items():
# 			# some qrpairs do not have the author dict and response dict
# 			try:
# 				alignment_dict[k] = qrpair.compute_alignment(features, questions_list, smoothing, weight_vector, alternative="final")
# 			except:
# 				pass
#
# 		print(alignment_dict)
# 		for key, score in alignment_dict.items():
# 			alignment_scores.setdefault(key, []).append(score)
#
# 	print(alignment_scores)
# 	total_alignment = {key: sum(alignment_scores[key]) for key in alignment_scores}
# 	print(total_alignment)
# 	score = list(total_alignment.values())[0]
# 	if all(value == score for value in total_alignment.values()) == True:
# 		print(max(total_alignment.items(), key=operator.itemgetter(1))[0])
# 	else:
# 		print("hello")
