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
# Code obtained from: http://ai.intelligentonlinetools.com/ml/convert-word-to-vector-glove-python/
# Unzip glove.6B.50.1.txt before running this code
words = pd.read_table("glove.6B.50d.1.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
words_matrix = words.as_matrix()
word_list = words.index

mid_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

# Make vector for certain word
# Code obtained from: http://ai.intelligentonlinetools.com/ml/convert-word-to-vector-glove-python/
def vec(word):
    return words.loc[word].as_matrix()

# Retrieve N closest (most similar) words
# Code obtained from: http://ai.intelligentonlinetools.com/ml/convert-word-to-vector-glove-python/
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
