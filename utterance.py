import numpy as np

class Utterance:
	def __init__(self, quote_object, response_object, key):
		self.quote_object = quote_object
		self.response_object = response_object

		self.key = key
		# self.discussion_id = discussion_id

		self.alignment_dict = {}
		self.alignment_vector_dict = {}

	def add_alignment(self, alignment, marker_name):
		self.alignment_dict[marker_name] = alignment

	def get_alignment(self, marker_name):
		return self.alignment_dict[marker_name]

	def add_alignment_vector(self, vector, marker_name):
		self.alignment_vector_dict[marker_name] = vector

	def get_alignment_vector(self, marker_name):
		return self.alignment_vector_dict[marker_name]

	def in_quote(self, marker_list):
		'''Check if the marker is in the
		quote text
		'''
		text = self.quote_object.words

		for word in text:
			for marker in marker_list:
				if word == marker:
					return True
		return False

	def compute_appearance(self, marker_list):
		text = self.response_object.words
		count = 0.0

		for word in text:
			for marker in marker_list:
				if word == marker:
					count += 1
		return count
