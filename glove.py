import pandas as pd
import csv
import numpy as np
import time

start_time = time.time()

# # Remove everything that is not a word and create a new file
# # Unzip glove.6B.50.txt before running this code
# with open("glove.6B.50d.txt", "r") as f:
#     with open("glove.6B.50d.1.txt","w") as output:
#         for line in f:
#             vals = line.rstrip().split(' ')
#             word = vals[0]
#             if word[0].isalpha():
#                 output.write(line)

# Read GloVe pre trained vectors and make a matrix of it
# Code used from: http://ai.intelligentonlinetools.com/ml/convert-word-to-vector-glove-python/
# Unzip glove.6B.50.1.txt before running this code
words = pd.read_table("glove.6B.50d.1.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
words_matrix = words.as_matrix()
word_list = words.index

mid_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

# Make vector for certain word
# Code used from: http://ai.intelligentonlinetools.com/ml/convert-word-to-vector-glove-python/
def vec(word):
    return words.loc[word].as_matrix()

# Retrieve N closest (most similar) words
# Code used from: http://ai.intelligentonlinetools.com/ml/convert-word-to-vector-glove-python/
def find_N_closest_words(vector, N, words):
    Nwords = []
    for w in range(N):
        diff = words.as_matrix() - vector
        delta = np.sum(diff * diff, axis=1)
        i = np.argmin(delta)
        Nwords.append(words.iloc[i].name)
        words = words.drop(words.iloc[i].name, axis=0)
    return Nwords

print(find_N_closest_words(vec('the'), 6, words))
print("--- %s seconds ---" % (time.time() - mid_time))




# CODE THAT WORKED, BUT WAS VERY SLOW

# Unzip glove.6B.50.1.txt before running this code
# start_time = time.time()
# glove = open("glove.6B.50d.1.txt", "r")
# glove_vectors = glove.readlines()
#
# mid_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))
#
# # Calculate cosine similarity of a word vector with all other vectors to find the 5 most similar words
# def calculate_most_similar(word_vector, glove_vectors):
#     result = Counter()
#     for key,value in glove_vectors.items():
#         temp = 0
#         temp = cosine_similarity(value.reshape(1, -1) ,word_vector.reshape(1, -1))
#         result[key] = temp
#     return result.most_common(6)[1:]
#
# print(calculate_most_similar(glove_vectors["the"], glove_vectors))
# print("--- %s seconds ---" % (time.time() - start_time))
