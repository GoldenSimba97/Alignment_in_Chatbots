import csv
import json
import os
import sys
from collections import defaultdict
import numpy as np
import operator

# try:
# 	import cPickle as pickle
# except:
# 	import pickle

class QRPair:
	def __init__(self, key):
	# def __init__(self, key, discussion_id, response_post_id, question_post_id):
		self.key = key
		# self.discussion_id = discussion_id
		# self.response_post_id = response_post_id
		# self.question_post_id = question_post_id


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

	# def add_annotations(self, annotations):
	# 	'''
	# 	Add dictionary of annotations to object,
	# 	get agreement
	# 	'''
	# 	self.annotations = annotations
	# 	self.agreement = annotations['agreement']
	# 	self.sarcasm = annotations['sarcasm']
	# 	self.nicenasty = annotations['nicenasty']


	# def update_authorratio(self, author_ratio_dict):
	# 	"""
	# 	Update dictionary that saves ratio of disagreement and agreement responses for response author
	# 	"""
    #
	# 	if float(self.agreement) > 1.0:
	# 		agreement = 'agreement'
	# 	elif float(self.agreement) > -1.0:
	# 		agreement = 'disagreement'
	# 	elif float(self.agreement) <= 1.0 or float(self.agreement) >= -1.0:
	# 		agreement = 'neutral'
	# 	# invalid agreement level?
	# 	else:
	# 		return author_ratio_dict
    #
	# 	try:
	# 		author_ratio_dict[self.response_author][agreement] += 1
	# 	except KeyError:
	# 		author_ratio_dict[self.response_author][agreement] = 1
    #
	# 	return author_ratio_dict


	# def update_authordict(self, author_dict, response_dict):
	# 	"""
	# 	Update number of posts of response author in author dict
	# 	"""
	# 	# Set response and question author for qrpair object
	# 	self.response_author = self.get_author_by_id(self.response_post_id)
	# 	self.question_author = self.get_author_by_id(self.question_post_id)
    #
	# 	# update number of posts for this response author (for computation baseline)
	# 	try:
	# 		author_dict[self.response_author]["number_posts"] += 1
	# 	except KeyError:
	# 		author_dict[self.response_author]= {"number_posts": 1}
	# 	# save responses to dict
	# 	try:
	# 		response_dict[self.response_author].append(self.response)
	# 	except KeyError:
	# 		response_dict[self.response_author]= [self.response]
	# 	# Also increase number of posts for the author of the question.
	# 	#try:
	# 	#	author_dict[self.question_author]["number_posts"] += 1
	# 	#except KeyError:
	# 	#	author_dict[self.question_author] = {"number_posts": 1}
	# 	return author_dict, {}, response_dict


	def compute_alignment(self, features, questions_list, smoothing, weight_vector, alternative="matching"):
		#question_author = self.get_author_by_id(self.question_post_id)
		#self.response_author = self.get_author_by_id(self.response_post_id)

		# OLD NEW VERSION USING LENGTH
		#base_prob, autor_dict = get_new_baseline(self.response_author, features, author_dict)
		# NEW OLD VERSION USING BINARY
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
			#print "zero case"
			self.alignment = 0
			return self.alignment
		q_m = (q_count + smoothing) / (len(question_text) + smoothing)
		#print q_count

		if q_m == 0:
			self.alignment = 0
			return self.alignment
		#print "Q_M", q_m

		r_m = (feature_count(response_text, features) + smoothing) / (len(response_text) + smoothing)
		#print "R_M", r_m
		match = self.matching(q_m, r_m)
		if alternative == "alternative":
			self.alignment = r_m - base_prob
		elif alternative == "matching":
			self.alignment = match - base_prob
		elif alternative == "original":
			self.alignment = conditional_prob - base_prob
		elif alternative == "final":
			self.alignment = conditional_prob - scalarized
			#if self.alignment > 1.0 or self.alignment < - 1.0:
			#	print "alignment", conditional_prob, base_prob, self.alignment
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


	# def get_author_by_id(self,post_id):
	# 	'''
	# 	Retrieve the author's username, using
	# 	the discussion and post id and the corresponding
	# 	json files.
	# 	'''
	# 	author = ""
	# 	if self.discussion_id != "discussion_id":
	# 		json_data=open('../data/fourforums/discussions/'+ self.discussion_id +".json").read()
	# 		data = json.loads(json_data)
	# 		for post in data[0]:
	# 			try:
	# 				if int(post[0]) == int(post_id):
	# 					author = post[2]
	# 			except ValueError:
	# 				continue
	# 	return author

def other_baseline(features, questions_list):
	# try:
	# 	baseline = author_dict[author][features[0]]
	# 	return baseline, author_dict
    #
	# except KeyError:
	exists = False
	total_exists = 0.0
	counter = 0
	# check number of responses which hold marker
	# for key, response in response_dict.items():
	for question in questions_list:
		text = getWords(question)
		counter += len(text)
		# Loop over list of features, if 1 marker is found in text, go to next response
		for feature in features:
			if feature in text:
				exists = True
				total_exists += 1


			# try:
			# 	author_dict[author][features[0]] = total_exists/counter
			# except KeyError:
			# 	author_dict[author] = {features[0]: total_exists/counter}
	return total_exists/counter #, author_dict

def get_baseline(features, questions_list):
	'''
	Compute the baseline probabilities for this author and
	features: count the number of features that appear (binary)
	in the author's utterances, divide by number of utterances
	'''
	# # Check if baseline is already known, do not compute again
	# try:
	# 	baseline = author_dict[author][features[0]]
	# 	return baseline, author_dict
	# except KeyError:
	# 	#all_text = []
	# 	#f = open(os.path.join('authors/', author+".txt"), 'r')
	# 	#total_length = 1.0
	# 	#new_utt = False
	# 	"""
	# 	for line in f:
	# 		line_text = getWords(line)
	# 		#print "Len",  len(line_text)
	# 		if new_utt == True:
	# 			for feature in features:
	# 				if feature in line_text:
	# 					total_markers += 1
	# 					break
	# 		#If we are at a new line, we are moving into a new utterance
	# 		if len(line_text) == 0:
	# 			total_length += 1
	# 			new_utt = True
	# 		else:
	# 			new_utt = False
	# 	"""
	total_markers = 0.0

	# check number of responses which hold marker
	# for key, response in response_dict.items():
	for question in questions_list:
		text = getWords(question)
		# Loop over list of features, if 1 marker is found in text, go to next response
		for feature in features:
			if feature in text:
				total_markers += 1
				break

		# total_posts = float(author_dict[author]['number_posts'])
		total_posts = 2
		# update author dict to hold baseline value
		# try:
		# 	author_dict[author][features[0]] = total_markers/total_posts
		# except KeyError:
		# 	author_dict[author] = {features[0]: total_markers/total_posts}
        #
		# """
		# old using total_length which is created using newlines, faulty
		# try:
		# 	author_dict[author][features[0]] = total_markers/total_length
		# except KeyError:
		# 	author_dict[author] = {features[0]: total_markers/total_length}
		# """
	return total_markers/total_posts #, author_dict

# def get_new_baseline(author, features, author_dict):
# 	'''
# 	NOT USED
# 	Compute the baseline probabilities for this author and
# 	features: count the number of features that appear (binary)
# 	in the author's utterances, divide by number of utterances
# 	'''
# 	try:
# 		baseline = author_dict[author][features[0]]
# 		return baseline, author_dict
# 	except KeyError:
# 		post_count = 0
# 		all_text = []
# 		lines = ""
# 		zero_case = 0
# 		f = open(os.path.join('authors/', author+".txt"), 'r')
# 		for line in f:
# 			if line=="\n":
# 				post_count += 1
# 			line_text = getWords(line)
# 			lines += line
# 			#print "Len",  len(line_text)
# 			all_text += line_text
# 		num_words = len(all_text)
# 		#print "Total text",
# 		f_count = feature_count(all_text, features)
# 		feature_value = f_count / num_words
# 		#print "Feature count", f_count
# 		try:
# 			author_dict[author][features[0]] = feature_value
# 		except KeyError:
# 			author_dict[author] = {features[0]: feature_value}
# 		return feature_value, author_dict

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


# def read_csv_data(features, smoothing, weight_vector, alternative="matching"):
# 	'''
# 	Adds all qr_pairs in qr_meta to a dictionary
# 	of qr_pair objects, including their annotations.
# 	'''
# 	pair_dictionary = {}
# 	#Get meta-data:
# 	author_dict = {}
# 	response_dict = {}
#
# 	# list to save qrpairs to update on alignment later on
# 	qrpair_list = []
# 	# Loop through data to get all qrpairs
#
# 	pickle_dict = {}
#
# 	print "Reading meta data..."
# 	with open('../data/fourforums/annotations/mechanical_turk/qr_meta.csv', 'rb') as meta_file:
# 		reader = csv.reader(meta_file, delimiter=',', questionchar='"')
# 		count = 0
# 		for row in reader:
# 			#if count > 10:
# 				#break
# 			key = row[0]
# 			discussion_id = row[1]
# 			response_post_id = row[2]
# 			question_post_id = row[3]
# 			question = row[8]
# 			response = row[9]
# 			qrpair = QRPair(key, discussion_id, response_post_id, question_post_id)
# 			qrpair.add_question(question)
# 			qrpair.add_response(response)
# 			if discussion_id != "discussion_id":
# 				author_dict, other_author_dict, response_dict = qrpair.update_authordict( author_dict, response_dict)
# 				#qrpair_list.append(qrpair)
# 			pair_dictionary[str(key) + str(discussion_id)] = qrpair
# 			count +=1
#
# 			# save to dict to pickle for use in other scripts
# 			pickle_dict[str(key) + str(discussion_id)] = [qrpair.question, qrpair.response]
#
# 	pickle_dump_qrpair(pickle_dict)
#
# 	# Loop through all qrpairs again to compute alignment
# 	for k, qrpair in pair_dictionary.iteritems():
# 		# some qrpairs do not have the author dict and response dict
# 		try:
# 			qrpair.compute_alignment(features, author_dict, other_author_dict, response_dict, smoothing, weight_vector, alternative)
# 		except:
# 			pass
#
# 	author_ratio_dict = defaultdict(lambda : defaultdict(int))
#
# 	#Get annotation data:
# 	print "Reading annotation data..."
# 	with open('../data/fourforums/annotations/mechanical_turk/qr_averages.csv', 'rb') as average_file:
# 		reader = csv.reader(average_file, delimiter=',', questionchar='"')
# 		for row in reader:
# 			try:
# 				corresponding_pair = pair_dictionary[str(row[0]) + str(row[1])]
# 				annotations = row[2:]
# 				annotations_dictionary = list_to_dictionary(annotations)
# 				corresponding_pair.add_annotations(annotations_dictionary)
# 				# update ratio of number of agreeing/disagreeing posts by response author
# 				try:
# 					author_ratio_dict = corresponding_pair.update_authorratio(author_ratio_dict)
# 				# sometimes qrpair does not contain a response author, so pass
# 				except:
# 					pass
# 			except KeyError:
# 				pass
#
#
# 	return pair_dictionary, author_dict, response_dict, author_ratio_dict


# def pickle_dump_qrpair(pickle_dict):
# 	"""
# 	Dump qrpairs to file
# 	key: str(key) + str(discussion_id),  value: [question_text, response_text]
# 	"""
# 	pickle.dump( pickle_dict, open( "qr_texts.pickle", "wb" ) )

# def list_to_dictionary(annotations):
# 	'''
# 	Turns a row of annotation values from qr_averages into a dictionary for
# 	easy annotation lookup.
# 	'''
# 	key_list = ["agree-disagree","agreement","agreement_unsure","attack","attack_unsure","defeater-undercutter","defeater-undercutter_unsure","fact-feeling","fact-feeling_unsure","negotiate-attack","negotiate-attack_unsure","nicenasty","nicenasty_unsure","personal-audience","personal-audience_unsure","questioning-asserting","questioning-asserting_unsure","sarcasm","sarcasm_unsure"]
# 	annotation_dictionary = {}
# 	for i in range(0, len(key_list)):
# 		annotation_dictionary[key_list[i]] = annotations[i]
# 	return annotation_dictionary

# def group_by_agreement(pair_dictionary):
# 	"""
# 	note: agreement is set on an 11 point scale [-5,5] with a slider
# 	"""
# 	agreement_list = []
# 	disagreement_list = []
# 	for pair, value in pair_dictionary.iteritems():
# 		try:
# 			if float(value.agreement) >= 0 :
# 				#print "agree", value.agreement
# 				agreement_list.append(value)
# 			else:
# 				#print "disagree", value.agreement
# 				disagreement_list.append(value)
# 		except ValueError:
# 			continue
# 	return agreement_list, disagreement_list




# def write_data_qrtext(pair_dictionary,features, feature_name):
# 	f = open(os.path.join('data/', "withtext_" + feature_name +".txt"), 'w')
# 	for qr_pair in pair_dictionary:
# 		pair = pair_dictionary[qr_pair]
# 		if pair.discussion_id != 'discussion_id':
# 			pair.compute_alignment(features)
# 			f.write(str(pair.question.split()) + " ; " + str(pair.response.split()) + " ; " + str(pair.agreement) + " ; " + str(pair.sarcasm) + " ; " + str(pair.nicenasty) + " ; " + str(pair.alignment) + "\n")
# 	f.close()

def get_features_from_file(file_name):
    """
    Read a list of feature words from file_name,
    and return a list of all the words
    """
    feature_list = []
    f = open(file_name, 'r')
    for line in f:
        feature_list.append(line.split("\n")[0])
    f.close()
    return feature_list


# if __name__=="__main__":
# 	#'adverbs', 'articles', 'auxiliaryverbs', 'conjunctions',
# 	#markers = ['adverbs']#
# 	weight_vector = np.array([0.8, 0.2])
# 	markers = ['adverbs', 'articles', 'auxiliaryverbs', 'conjunctions', 'impersonalpronouns', 'personalpronouns', 'prepositions', 'quantifiers', 'number_posts']
# 	#markers = ['number_posts']
# 	smoothing = 0.1
# 	total_alignment = {}
# 	alignment_scores = {}
# 	scores = 0
# 	for mark in markers:
# 		print("Getting features for", mark)
# 		if mark != "number_posts" :
# 			features = get_features_from_file('coordination_markers/'+mark+'.txt')
# 		else:
# 			features = [mark]
#
# 		response_dict = {40.0: "You're making your point.", 50.0: 'Swearing is often cathartic.'}
# 		# align = Align("you cunt, why won't you tell me something", responses_dict)
# 		question = "you cunt, why won't you tell me something"
# 		questions_list = ["you cunt, why won't you tell me something", "i love you"]
#
# 		# question_post = Post(question, "user")
#         # for key, response in responses_dict.items():
#         #     response_post = Post(response, "chatbot")
#         #     utterance = Utterance(question_post, response_post, key)
#         #     author1 = Author(utterance.response_object.author)
#         #     self.author_dict[utterance.response_object.author] = author1
#         #     author2 = Author(utterance.question_object.author)
#         #     self.author_dict[utterance.question_object.author] = author2
#         #     author1.add_utterance(utterance)
#         #     author2.add_utterance(utterance)
#         #     self.utterance_dict[str(key)] = utterance
#
# 		pair_dictionary = {}
# 		for question in questions_list:
# 			for key, response in response_dict.items():
# 				qrpair = QRPair(key)
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
# 		# for key, res in response_dict.items():
# 		# 	# print("KEY", type(k))
# 		# 	for k, score in alignment_dict.items():
# 		# 		# print("K", type(key))
# 		# 		if key == float(k):
# 		# 			print("hello")
# 		# 			# print(k)
# 		# 			if score != None:
# 		# 				print(score)
# 		# 				scores += score
# 		# 				print(scores)
# 		# 		total_alignment[k] = scores
#
# 		# years_dict = dict()
#
#
# 		# for k, line in alignment_dict.items():
# 		# 	print(line)
# 		#     if k, line[0] in total_alignment.items():
# 		#         # append the new number to the existing array at this slot
# 		# 		total_alignment[line[0]].append(line[1])
# 		#     else:
# 		#         # create a new array in this slot
# 		# 		total_alignment[line[0]] = [line[1]]
#
#
#
# 	# total_alignment = {}
# 	# scores = 1
# 	# for k, scores in alignment_scores.items():
# 	# 	# print(scores)
# 	# 	for score in scores:
# 	# 		if score != 0.0:
# 	# 			print(type(score))
# 	# 			scores *= int(score)
# 	# 	total_alignment[k] = scores
# 	# print(total_alignment)
# 	# result = {k:v for k,v in alignment_scores.items() if v != None}
# 	# print("Hello", result)
# 	print(alignment_scores)
# 	# print({k:sum(map(int, v)) for k, v in alignment_scores.items()})
# 	total_alignment = {key: sum(alignment_scores[key]) for key in alignment_scores}
# 	print(total_alignment)
# 	# print(max(total_alignment.items(), key=operator.itemgetter(1))[0])
# 	# val = list(total_alignment.values())[0]
# 	# print(all(value == val for value in total_alignment.values()))
# 	score = list(total_alignment.values())[0]
# 	if all(value == score for value in total_alignment.values()) == True:
# 		print(max(total_alignment.items(), key=operator.itemgetter(1))[0])
# 	else:
# 		print("hello")


		# qrpair = QRPair(key)
		# qrpair.add_question(question)
		# qrpair.add_response(response)

		# print "Reading data from file..."
		# pair_dictionary, author_dict,response_dict, author_ratio_dict = read_csv_data(features, smoothing, weight_vector, alternative="final")
		#print "KY", pair_dictionary.keys()
		# print "Getting agreement groups..."
		# agreements, disagreements = group_by_agreement(pair_dictionary)
		# print "writing to file for %s" %(mark)
		# write_data(pair_dictionary,features, mark, author_dict, response_dict, author_ratio_dict, smoothing, weight_vector, alternative="final")
