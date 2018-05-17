import pandas as pd
import csv

words = pd.read_table("glove.6B/glove.6B.50d.1.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(w):
  return words.loc[w].as_matrix()

print(vec('hello'))    #this will print same as print (model['hello'])  before

words = words.drop("table", axis=0)
words = words.drop("tables", axis=0)

words_matrix = words.as_matrix()

def find_closest_word(v):
  diff = words_matrix - v
  delta = np.sum(diff * diff, axis=1)
  i = np.argmin(delta)
  return words.iloc[i].name


print(find_closest_word(model['table']))
#output:  place

#If we want retrieve more than one closest words here is the function:

def find_N_closest_word(v, N, words):
  Nwords=[]
  for w in range(N):
     diff = words.as_matrix() - v
     delta = np.sum(diff * diff, axis=1)
     i = np.argmin(delta)
     Nwords.append(words.iloc[i].name)
     words = words.drop(words.iloc[i].name, axis=0)

  return Nwords


print(find_N_closest_word(model['table'], 10, words)) 

#Output:
#['table', 'tables', 'place', 'sit', 'set', 'hold', 'setting', 'here', 'placing', 'bottom']
