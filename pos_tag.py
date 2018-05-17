# nltk.download()
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_treebank_pos_tagger')

import nltk
import pandas as pd
from heapq import nlargest
from heapq import nsmallest
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import gensim
from gensim.models import word2vec
import csv
import time




# NN    Noun                  table
# NNS
# NNP
# NNPS
# JJ    adjective             Green
# JJR
# JJS
# IN    preposition           in, of, LIKE
# DT    article, determiner   the, a, an
# PRP   pronoun               I, he, it
# PRP$
# WP
# WP$
# VB    verb                  are, going, like
# VBD
# VBG
# VBN
# VBP
# VBZ
# RB    adverb                completely, however, usually
# RBR
# RBS
# WRB
# UH    interjection          uhhuhhuhh


# F score = (noun frequency + adjective freq + preposition freq
# + article freq - pronoun freq - verb freq - adverb freq
# - interjection freq + 100)/2

# Open formal and informal word lists
# formal_file = open("FormalityLists/formal_seeds_100.txt", "r")
# formal = formal_file.read()
#
# informal_file = open("FormalityLists/informal_seeds_100.txt", "r")
# informal = informal_file.read()

start_time = time.time()

formal_file = open("FormalityLists/formal_seeds_100.txt", "r")
formal = []
for line in formal_file:
    line = line.replace("\r\n", "")
    line = line.replace("\t", "")
    formal.append(line)

informal_file = open("FormalityLists/informal_seeds_100.txt", "r")
informal = []
for line in informal_file:
    line = line.replace("\r\n", "")
    line = line.replace("\t", "")
    informal.append(line)

text_file = open("FormalityLists/CTRWpairsfull.txt", "r")
for line in text_file:
    lines = line.split("/")
    informal.append(lines[0])
    formal.append(lines[1])


test = pd.read_csv("fii_annotations/mturk_experiment_2.csv", sep=',', encoding = "ISO-8859-1")
# print(test.shape)
# print(test.head())
test_formality = test["Formality"]
test_sentences = test["Sentence"]

# Read GloVe pre trained vectors
words = pd.read_table("glove.6B/glove.6B.50d.1.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
# print(words[words.columns[0]].head())
word_list = words.index
print(words.index)
# if "the" in word_list:
    # print("Hoi")

# x[x.columns[0]]
# Make a matrix of all word vectors
words_matrix = words.as_matrix()

mid_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))


# Determine the total formality score of the input sentence
def determine_formality(sentence):
    # POS tag the input sentence
    sentence = sentence.lower()
    # s_len = float(len(sentence.split()))
    text = nltk.word_tokenize(sentence)
    s_len = float(len(text))
    print(s_len)
    tagged = nltk.pos_tag(text)
    NN_count = JJ_count = IN_count = DT_count = PRP_count = VB_count = RB_count = UH_count = 0# formality = 0
    formality = 1

    # Get the counts needed to determine frequencys for calculation of F score.
    # If a punctuation is encountered, decrease the length of the sentence by 1.
    for tag in tagged:
        if tag[1] == "NN" or tag[1] == "NNS" or tag[1] == "NNP" or tag[1] == "NNS":
            NN_count += 1
        elif tag[1] == "JJ" or tag[1] == "JJR" or tag[1] == "JJS":
            JJ_count += 1
        elif tag[1] == "IN":
            IN_count += 1
        elif tag[1] == "DT":
            DT_count += 1
        elif tag[1] == "VB" or tag[1] == "VBD" or tag[1] == "VBG" or tag[1] == "VBN" or tag[1] == "VBP" or tag[1] == "VBZ":
            VB_count += 1
        elif tag[1] == "RB" or tag[1] == "RBR" or tag[1] == "RBS" or tag[1] == "WRB":
            RB_count += 1
        elif tag[1] == "UH":
            UH_count += 1
        elif tag[1] == "." or tag[1] == ":" or tag[1] == "," or tag[1] == "(" or tag[1] == ")":
            s_len -= 1

    # Increase formality score if a formal word is encountered and decrease it if an informal word is encountered
    for tag in tagged:
        if tag[0] in formal:
            formality *= 1.1 #+= 10
        elif tag[0] in informal:
            formality *= 0.9 #-= 10

    return formality * f_score(NN_count/s_len*100, JJ_count/s_len*100, IN_count/s_len*100, DT_count/s_len*100,
            PRP_count/s_len*100, VB_count/s_len*100, RB_count/s_len*100, UH_count/s_len*100)


# Calculation of the F score
def f_score(NN_freq, JJ_freq, IN_freq, DT_freq, PRP_freq, VB_freq, RB_freq, UH_freq):
    return ((NN_freq + JJ_freq + IN_freq + DT_freq - PRP_freq - VB_freq - RB_freq - UH_freq + 100)/2)

# Make vector for certain word
def vec(w):
  return words.loc[w].as_matrix()

# Retrieve N closest (most similar) words
def find_N_closest_words(v, N, words):
  Nwords = []
  for w in range(N):
     diff = words.as_matrix() - v
     delta = np.sum(diff * diff, axis=1)
     i = np.argmin(delta)
     Nwords.append(words.iloc[i].name)
     words = words.drop(words.iloc[i].name, axis=0)
  return Nwords

# print(find_N_closest_words(vec('the'), 6, words))

# print("--- %s seconds ---" % (time.time() - start_time))

# Calculate MSE and MAE to compare the true formality scores from mturk_experiment_2 and the calculated scores.
# Scale the calculated scores to the range from 1-7 used in mturk_experiment_2.
def test_formality_score():
    formality_score = []
    for test_sentence in test_sentences:
        score = determine_formality(test_sentence)
        new_score = ((score * 6) / 100) + 1
        formality_score.append(round(new_score,1))
    test["Formality_score"] = formality_score
    return mean_squared_error(test_formality, formality_score), mean_absolute_error(test_formality, formality_score)#, r2_score(test_formality, formality_score)

# print(test_formality_score())


print(determine_formality("100"))
# print(determine_formality("Ha ha ha."))
# print(determine_formality("Why?"))
# print(determine_formality("hello. why will not you cry. you cunt, why won't you tell me something?."))
# print(determine_formality("hello why will not you cry you cunt, why won't you tell me something? "))
# print(determine_formality("Just wipe the Mac OS X partition when u install the dapper.")) # 1.2
# print(determine_formality("Water sports and golf are abundant- and we have some of the greatest cycling in the world, we will be hosting the Ironman competition while you are here.")) # 3.2
# print(determine_formality("At the Shuttle Landing Facility at NASA's Kennedy Space Center in Florida, hardware that will be used in the launch of the Ares I-X rocket is offloaded from a C-5 aircraft.")) # 5.8
# #
# print(determine_formality("A few companies have decided to buck the trend by not offering any employment contracts.")) # 1.2
# print(determine_formality("A few of President Obama's top advisers, as well as one or two rare guests, sit down on the network sofas this Sunday.")) # 3.6
# print(determine_formality("Although the Chinese government has not taken action against Yuan or the publisher, a nongovernmental organization, the Chinese Assn. for the Promotion of Olympic Culture said last week it would file a civil lawsuit against Yuan's publisher, Beijing Fonghong Media Co., to prevent publication of any copies beyond the 200,000 in print in China.")) # 6.2
# print(determine_formality("And part of the subtext of the Afghanistan debate is that as a matter of bureaucratic warfare, it makes enormous sense for the currently ascendant COIN faction to try to press its advantages - to exaggerate the extent of what was achieved in Iraq in 2007, and to overstate the strategic significance of achieving some kind of comprehensive success in Afghanistan.")) # 6.4
# print(determine_formality("China will impose five-year anti-dumping tariffs, ranging from 5 percent to 35.4 percent, on imports of adipic acid from the United States, the European Union and the Republic of Korea, the Ministry of Commerce (MOC) said on Sunday.")) # 6.6
# print(determine_formality("CIT spokesman Curt Ritter declined to comment yesterday.")) # 6.8
# print(determine_formality("Thanks...Michael"))
