import json

class Post:
	def __init__(self, text, author):
		# self.discussion_id = discussion_id
		# self.post_id = post_id
		self.author = author
		self.text = text
		self.words, self.num_words = self.get_words(text)

	def get_words(self, post_text):
		'''
		Written by Julian
		'''
		text = post_text.split(" ")
		text2 = []
		count = 0
		for word in text:
			word = word.strip()
			# Remove non-words:
			if not word in ['#','/','[',']','}','--',',','-/','+','-','((','))']:
				if not (word.startswith('{') or
						word.startswith('<') or word.endswith('>')
						or word.startswith('(')):
					if word:
						count += 1
						text2.append(word)
		text2 = [word.lower() for word in text2]
		text2 = [word.strip().strip(',.-#') for word in text2]
		return text2, count

	# def get_author_by_id(self):
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
	# 				if int(post[0]) == int(self.post_id):
	# 					author = post[2]
	# 			except ValueError:
	# 				continue
	# 	return author
