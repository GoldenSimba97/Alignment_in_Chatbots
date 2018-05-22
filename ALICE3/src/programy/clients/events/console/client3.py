"""
Copyright (c) 2016-2018 Keith Sterling http://www.keithsterling.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import nltk
from heapq import nlargest
import numpy as np
import time
import operator
import time

from programy.utils.logging.ylogger import YLogger

from programy.clients.events.client import EventBotClient
from programy.clients.events.console.config import ConsoleConfiguration
from programy.clients.events.alignment import QRPair
from programy.clients.render.text import TextRenderer

# Create formal and informal word lists
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

# # Read GloVe pre trained vectors and make a matrix of it
# # Used to test the influence of using GloVe vectors
# # Unzip glove.6B.50.txt before running this code
# words = pd.read_table("glove.6B/glove.6B.50d.1.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
# words_matrix = words.as_matrix()
# word_list = words.index

class ConsoleBotClient(EventBotClient):

    def __init__(self, argument_parser=None):
        self.running = False
        EventBotClient.__init__(self, "Console", argument_parser)
        self._renderer = TextRenderer(self)

        # Variables added for Thesis
        self.user_history = ""
        self.user_his_list = []
        self.question_number = 0
        self.user_formality = 0

    def get_description(self):
        return 'ProgramY AIML2.0 Console Client'

    def get_client_configuration(self):
        return ConsoleConfiguration()

    def add_client_arguments(self, parser=None):
        return

    def parse_args(self, arguments, parsed_args):
        return

    def get_question(self, client_context, input_func=input):
        ask = "%s " % self.get_client_configuration().prompt
        return input_func(ask)

    # Displays the startup message which consists of introductory questions to learn more about
    # the language use of a user.
    def display_startup_messages(self, client_context):
        self.process_response(client_context, client_context.bot.get_version_string(client_context))
        # initial_question = client_context.bot.get_initial_question(client_context)

        # Changed initial question and created introductory questions
        initial_question = ("Hi! In order for me to be able to have a better conversation with you,"
        " could you please tell me something about yourself? What is your name? How old are you? Do you work?"
        " If so, as what? What is something you hate doing and why? What do you do to have fun and why?"
        " Please only use the enter key after you've answered all questions. Pressing crtl-c will enable you"
        " to stop talking with ALICE.")
        self._renderer.render(client_context, initial_question)

    def process_question(self, client_context, question):
        # Returns a response
        return client_context.bot.ask_question(client_context , question, responselogger=self)

    def render_response(self, client_context, response):
        # Calls the renderer which handles RCS context, and then calls back to the client to show response
        self._renderer.render(client_context, response)

    # All answers between random tag are split by 11039841. Formality of the user is compared to the
    # formality of the random responses. The two responses with a formality furthest from that of the user
    # are used to find the least linguistically aligned response, which is returned. If both responses
    # have the same alignment score the response which has a formality score furthest from that of the user
    # is returned. If there is just a single possible answer, it gets directly returned.
    # Also makes sure "@" is replaced with "at" and "<3" with "heart".
    def process_response(self, client_context, response):
        if "11039841" in response:
            if "@" in response:
                response = response.replace("@", " at ")
            if "<3" in response:
                response = response.replace("<3", " heart")
            responses_formality = {}
            worst_responses = {}
            two_worst_responses = []

            # Split by 11039841 and remove empty items from list
            sentences = response.split("11039841")
            sentences = [x for x in sentences if x]

            # Determine formality for all possible responses
            # start_time = time.time() # Used to test the influence of using GloVe vectors
            for sentence in sentences:
                responses_formality[self.determine_formality(sentence)] = sentence
            #   # Used to test the influence of using GloVe vectors
            #     glove = 1
            #     formality = self.determine_formality(sentence)
            #     difference = self.user_formality - formality
            #     question = self.question.split()
            #     for word in question:
            #         if word in word_list:
            #             closest = self.find_N_closest_words(self.vec(word), 3, words)
            #             for close in closest:
            #                 if close in sentence:
            #                     if difference > 0:
            #                         glove *= 1.1
            #                     elif difference < 0:
            #                         glove *= 0.9
            #     responses_formality[formality * glove] = sentence

            for key, val in responses_formality.items():
                 print(key, "=>", val)

            # Find the two responses with furthest formality from user formality
            two_worst_formality = nlargest(2, responses_formality, key=lambda x:abs(x-self.user_formality))
            two_worst_responses.extend((responses_formality[two_worst_formality[0]], responses_formality[two_worst_formality[1]]))

            for item in two_worst_formality:
                worst_responses[item] = responses_formality[item]
            print(worst_responses)

            print(two_worst_responses)
            # print("--- %s seconds ---" % (time.time() - start_time)) # Used to test the influence of using GloVe vectors

            # Determine alignment score of the two worst responses
            total_alignment = self.determine_alignment(worst_responses)
            print(total_alignment)
            score = list(total_alignment.values())[0]
            # If scores are equal, print worst formality response. Else print minimum alignment response
            if all(value == score for value in total_alignment.values()) == True:
                print(two_worst_responses[0])
            else:
                print(worst_responses[min(total_alignment.items(), key=operator.itemgetter(1))[0]])
        else:
            if "@" in response:
                response = response.replace("@", " at ")
            if "<3" in response:
                response = response.replace("<3", " heart")
            print(response)

    # Add question to the user history and determines the formality over the whole user history. If A.L.I.C.E. responds
    # to the user for the first time, it will not repond according to its AIML, but will give fixed response.
    def process_question_answer(self, client_context):
        question = self.get_question(client_context)

        # Expand user history
        self.user_history = self.user_history + question + ". "
        self.user_his_list.append(question)

        # Determine user formality over whole user history
        self.user_formality = self.determine_formality(self.user_history)
        print(self.user_formality)

        # Make sure A.L.I.C.E. doesn't try to respond to the introductory questions
        if self.question_number == 1:
            response = "Thank you very much for telling me something about yourself! How can I help you today?"
        else:
            response = self.process_question(client_context, question)
        self.render_response(client_context, response)

    # Keep track of the number of questions a user asks. This is useful for making sure A.L.I.C.E. does not respond to the
    # introductory questions and for evaluating the conversation length.
    def wait_and_answer(self):
        running = True
        try:
            self.question_number += 1
            client_context = self.create_client_context(self._configuration.client_configuration.default_userid)
            self.process_question_answer(client_context)
        except KeyboardInterrupt as keye:
            running = False
            client_context = self.create_client_context(self._configuration.client_configuration.default_userid)
            # self._renderer.render(client_context, client_context.bot.get_exit_response(client_context))

            # Changed exit response. Can be further changed to accomodate the specific user
            exit_response = "\nBye!"
            self._renderer.render(client_context, exit_response)
        except Exception as excep:
            YLogger.error(self, "Oops something bad happened !")
            YLogger.exception(self, excep)
        return running

    def prior_to_run_loop(self):
        client_context = self.create_client_context(self._configuration.client_configuration.default_userid)
        self.display_startup_messages(client_context)

    # Determine the total formality score of the input by computing the frequency of nouns, adjectives, prepositions
    # articles, pronouns, verbs, adverbs and interjections. If a word of the input exists in the formal words list
    # the formality score will be increased with 10% and if a word of the input exists in the informal words list
    # the formality score will be decreased with 10%. The total formality score will be returned. The higher the score
    # the more formal the input.
    def determine_formality(self, sentence):
        sentence = sentence.lower()

        # POS tag the input sentence
        text = nltk.word_tokenize(sentence)
        s_len = float(len(text))
        tagged = nltk.pos_tag(text)
        NN_count = JJ_count = IN_count = DT_count = PRP_count = VB_count = RB_count = UH_count = 0
        formality = 1

        # Get the counts needed to determine frequencys for calculation of F-score
        # If punctuation is encountered, decrease the length of the sentence by 1
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
                formality *=1.1
            elif tag[0] in informal:
                formality *0.9

        return formality * self.f_score(NN_count/s_len*100, JJ_count/s_len*100, IN_count/s_len*100, DT_count/s_len*100,
                PRP_count/s_len*100, VB_count/s_len*100, RB_count/s_len*100, UH_count/s_len*100)

    # Calculate the F-score
    def f_score(self, NN_freq, JJ_freq, IN_freq, DT_freq, PRP_freq, VB_freq, RB_freq, UH_freq):
        return ((NN_freq + JJ_freq + IN_freq + DT_freq - PRP_freq - VB_freq - RB_freq - UH_freq + 100)/2.0)

    # Make vector for certain word
    # Used to test the influence of using GloVe vectors
    def vec(self, word):
        return words.loc[word].as_matrix()

    # Retrieve N closest (most similar) words
    # Used to test the influence of using GloVe vectors
    def find_N_closest_words(self, vector, N, words):
        Nwords = []
        for w in range(N):
            diff = words.as_matrix() - vector
            delta = np.sum(diff * diff, axis=1)
            i = np.argmin(delta)
            Nwords.append(words.iloc[i].name)
            words = words.drop(words.iloc[i].name, axis=0)
        return Nwords

    # Determine the total alignment score of the responses of the chatbot to the whole user history. Alignment will be calculated
    # for each coordination marker and question-response pair. The sum of all alignment scores will be returned for each response.
    def determine_alignment(self, response_dict):
        weight_vector = np.array([0.8, 0.2])
        markers = ['adverbs', 'articles', 'auxiliaryverbs', 'conjunctions', 'impersonalpronouns', 'personalpronouns', 'prepositions', 'quantifiers', 'number_posts']
        smoothing = 0.1
        alignment_scores = {}
        scores = 0
        for mark in markers:
            if mark != "number_posts" :
                features = self.get_features_from_file('coordination_markers/'+mark+'.txt')
            else:
                features = [mark]

            # Create question-reponse pairs
            pair_dictionary = {}
            for question in self.user_his_list:
                for key, response in response_dict.items():
                    qrpair = QRPair()
                    qrpair.add_question(question)
                    qrpair.add_response(response)
                    pair_dictionary[key] = qrpair

            alignment_dict = {}
    		# Loop through all qrpairs again to compute alignment
            for k, qrpair in pair_dictionary.items():
                alignment_dict[k] = qrpair.compute_alignment(features, self.user_his_list, smoothing, weight_vector, alternative="final")

            # Append all scores for the same respons
            for key, score in alignment_dict.items():
                alignment_scores.setdefault(key, []).append(score)

    	# Return total alignment scores
        return {key: sum(alignment_scores[key]) for key in alignment_scores}

    # Read a list of feature words from file_name and return a list of all the words
    def get_features_from_file(self, file_name):
        feature_list = []
        f = open(file_name, 'r')
        for line in f:
            feature_list.append(line.split("\n")[0])
        f.close()
        return feature_list



if __name__ == '__main__':

    def run():
        print("Loading, please wait...")
        console_app = ConsoleBotClient()
        console_app.run()

    run()
