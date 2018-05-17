import pandas as pd
import csv
import numpy as np
import time

start_time = time.time()

words = pd.read_table("glove.6B/glove.6B.50d.1.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

mid_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

def vec(w):
  return words.loc[w].as_matrix()

words_matrix = words.as_matrix()

# If we want retrieve more than one closest words here is the function:
def find_N_closest_word(v, N, words):
  Nwords = []
  for w in range(N):
     diff = words.as_matrix() - v
     delta = np.sum(diff * diff, axis=1)
     i = np.argmin(delta)
     Nwords.append(words.iloc[i].name)
     words = words.drop(words.iloc[i].name, axis=0)
  return Nwords[1:]

print(find_N_closest_word(vec('the'), 6, words))

print("--- %s seconds ---" % (time.time() - start_time))




# CODE DIE NIET WERKTE/TE LANGZAAM WAS/NUET NODIG WAS

# print(vec('hello'))    #this will print same as print (model['hello'])  before

# words = words.drop("table", axis=0)
# words = words.drop("tables", axis=0)

# def find_closest_word(v):
#   diff = words_matrix - v
#   delta = np.sum(diff * diff, axis=1)
#   i = np.argmin(delta)
#   return words.iloc[i].name


# print(find_closest_word(vec('the')))
#output:  place

# glove = open("glove.840B.300d.txt", "r")
# glove_vectors = glove.readlines()
# glove_vec = glove_vectors[:1]
# print(type(glove_vec))

# # with open("glove.6B/glove.6B.50d.3.txt", "r") as f:
# #     with open("glove.6B/glove.6B.50d.1.txt","w") as output:
# #         # glove_vectors = {}
# #         for line in f:
# #             vals = line.rstrip().split(' ')
# #             word = vals[0]
# #             if word[0].isalpha():
# #                 # print(word[0], word)
# #                 output.write(line)
#             # if line!="nickname_to_delete"+"\n":
#             #     output.write(line)
#
# with open("glove.6B/glove.6B.50d.1.txt", "r") as f:
#     glove_vectors = {}
#     for line in f:
#         vals = line.rstrip().split(' ')
#         glove_vectors[vals[0]] = np.asarray([float(x) for x in vals[1:]])
# # # print(dict(list(glove_vectors.items())[0:2]))
#
# mid_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))
#
# # print(Glove(glove_vectors).most_similar('man'))
#
# # model = word2vec.load_word2vec_format("glove.6B/glove.6B.50d.txt", binary=False)
# # print(model.most_similar(["the"]))
#
# # glove = Glove(no_components=100, learning_rate=0.05)
# # glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
# # glove.add_dictionary(glove_vectors)
# # print(glove.most_similar('the'))
# # print(glove_vectors['the'])
#
# # def loadGloveModel(gloveFile):
# #     print("Loading Glove Model")
# #     f = open(gloveFile,'r')
# #     model = {}
# #     for line in f:
# #         splitLine = line.split()
# #         word = splitLine[0]
# #         embedding = np.array([float(val) for val in splitLine[1:]])
# #         model[word] = embedding
# #     print("Done.",len(model)," words loaded!")
# #     return model
# #
# # glove = loadGloveModel("glove.6B/glove.6B.50d.txt")
# # print(glove.most_similar("the"))
#
# # def similarity_query(word_vec, number):
# #     dst = (np.dot(glove_vectors, word_vec)
# #             / np.linalg.norm(glove_vectors, axis=1)
# #             / np.linalg.norm(glove_vectors[word_vec]))
# #     word_ids = np.argsort(-dst)
# #
# #     return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
# #             if x in self.inverse_dictionary]
#
# words = pd.read_table("glove.6B/glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
#
# def vec(w):
#   return words.loc[w].as_matrix()
#
# glove_matrix = words.as_matrix()
# # print(glove_matrix)
# # print(glove_matrix.shape)
#
# def find_closest_word(v):
#   diff = glove_matrix - v
#   delta = np.sum(diff * diff, axis=1)
#   i = np.argmin(delta)
#   return words.iloc[i].name
#
# # print(find_closest_word(glove_vectors["the"]))
#
# def calculate_most_similar(word_vector, glove_vectors):
#     result = Counter()
#     for key,value in glove_vectors.items():
#         # print(value.reshape(-1, 1).shape ,word_vector.reshape(-1, 1).shape)
#         temp = 0
#         temp = cosine_similarity(value.reshape(1, -1) ,word_vector.reshape(1, -1))
#         # print(temp)
#         result[key] = temp
#
#     # print(glove_matrix.shape, np.asarray(word_vector).shape)
#     # similarity = cosine_similarity(glove_matrix.reshape(-1, 1), np.asarray(word_vector).reshape(-1, 1))
#     # return similarity
#     return result.most_common(6)[1:]
#
# # print(similarity_query('the', 5))
# # print(np.linalg.norm(glove_vectors['the']))
# # print(np.linalg.norm(glove_vectors, axis=1))
# print(calculate_most_similar(glove_vectors["the"], glove_vectors))
# print("--- %s seconds ---" % (time.time() - start_time))
