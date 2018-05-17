import numpy as np

class Author:
	def __init__(self, author_id):
		self.author_id = author_id
		self.utterances = {}
		self.num_posts = 0
		self.baseline_dict = {}
		self.marker_dict = {}

	def get_baseline(self, marker_name):
		return self.marker_dict[marker_name]

	def add_utterance(self, utterance_object):
		'''Add an utterance object to the utterance dict
		'''
		try:
			self.utterances[utterance_object.quote_object.author].append(utterance_object)
		except KeyError:
			self.utterances[utterance_object.quote_object.author] = [utterance_object]
		self.num_posts += 1

	def get_all_utterances(self):
		'''Return a flattened list of all Author's utterances
		'''
		return [item for sublist in self.utterances.values() for item in sublist]

	def get_quotes(self):
		'''Get all utterances where Author is the quote author
		'''
		return self.utterances[self.author_id]

	def get_responses(self, quote_author):
		'''Return the flattened list of utterance objects
		where Author has responded to quote_author
		'''
		if quote_author == "":
			#If no author, return all responses, so no quotes
			response_list = [value for key, value in self.utterances.items() if key not in [self.author_id]]
			return [item for sublist in response_list for item in sublist]
		else:
			try:
				return self.utterances[quote_author]
			except KeyError:
				print("No author found by id", quote_author, "returning empty list...")
				return []

	def compute_baseline(self, marker_name, marker_list):
		'''Compute baseline use of the marker in Author's responses
		'''
		response_list = self.get_all_utterances()
		post_count = 0.0
		word_count = 0.0
		word_appearance_count = 0.0
		for utterance in response_list:
			appearance_count = utterance.compute_appearance(marker_list)
			word_appearance_count += appearance_count
			word_count += utterance.response_object.num_words
			if appearance_count:
				post_count += 1.0
		frequency_in_posts = post_count / self.num_posts
		frequency_in_words = word_appearance_count / word_count
		baseline_vector = np.array([frequency_in_posts, frequency_in_words])
		self.marker_dict[marker_name] = baseline_vector
