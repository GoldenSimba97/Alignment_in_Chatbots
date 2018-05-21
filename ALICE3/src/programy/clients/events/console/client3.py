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
# from heapq import nsmallest
import numpy as np
import time
import operator

from programy.utils.logging.ylogger import YLogger

from programy.clients.events.client import EventBotClient
from programy.clients.events.console.config import ConsoleConfiguration
from programy.clients.render.text import TextRenderer

from programy.clients.events.alignment import QRPair

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

class ConsoleBotClient(EventBotClient):

    def __init__(self, argument_parser=None):
        self.running = False
        EventBotClient.__init__(self, "Console", argument_parser)
        self._renderer = TextRenderer(self)

        # Added user history variable
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
    # formality of the random responses. The two responses with a formality that are the most different
    # to that of the user are returned. If there is just a single possible answer, it gets directly returned.
    # Also made sure "@" is replaced with "at".
    def process_response(self, client_context, response):
        if "11039841" in response:
            if "@" in response:
                response = response.replace("@", " at ")
            if "<3" in response:
                response = response.replace("<3", " heart")
            responses_formality = {}
            best_responses = {}
            two_best_responses = []
            sentences = response.split("11039841")
            sentences = [x for x in sentences if x]

            # Determine formality for all possible responses
            start_time = time.time()
            for sentence in sentences:
                responses_formality[self.determine_formality(sentence)] = sentence

            for key, val in responses_formality.items():
                 print(key, "=>", val)

            # Find the two responses with most different formality to user formality
            # two_best_formality = nlargest(2, responses_formality, key=lambda x:abs(x-self.user_formality))
            two_best_formality = nlargest(2, responses_formality, key=lambda x:abs(x-self.user_formality))
            two_best_responses.extend((responses_formality[two_best_formality[0]], responses_formality[two_best_formality[1]]))
            # print(two_best_responses)
            # print("--- %s seconds ---" % (time.time() - start_time))
            for item in two_best_formality:
                best_responses[item] = responses_formality[item]
            print(best_responses)

            print(two_best_responses)
            # print("--- %s seconds ---" % (time.time() - start_time))
            total_alignment = self.determine_alignment(best_responses)
            print(total_alignment)
            score = list(total_alignment.values())[0]
            if all(value == score for value in total_alignment.values()) == True:
                # print("hello")
                print(two_best_responses[0])
            else:
                # print("bye")
                print(best_responses[min(total_alignment.items(), key=operator.itemgetter(1))[0]])
        else:
            if "@" in response:
                response = response.replace("@", " at ")
            if "<3" in response:
                response = response.replace("<3", " heart")
            print(response)

    def process_question_answer(self, client_context):
        question = self.get_question(client_context)

        # Expands user history
        self.user_history = self.user_history + question + ". "
        self.user_his_list.append(question)
        # print(self.user_history)

        # Determines user formality over whole user history
        self.user_formality = self.determine_formality(self.user_history)
        print(self.user_formality)

        # Makes sure A.L.I.C.E. doesn't try to respond to the introductory questions
        if self.question_number == 1:
            response = "Thank you very much for telling me something about yourself! How can I help you today?"
        else:
            response = self.process_question(client_context, question)
        self.render_response(client_context, response)

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

    # Determine the total formality score of the input sentence
    def determine_formality(self, sentence):
        sentence = sentence.lower()

        # POS tag the input sentence
        text = nltk.word_tokenize(sentence)
        s_len = float(len(text))
        tagged = nltk.pos_tag(text)
        NN_count = JJ_count = IN_count = DT_count = PRP_count = VB_count = RB_count = UH_count = 0#formality = 0
        formality = 1

        # Get the counts needed to determine frequencys for calculation of F score
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
                formality *=1.1#+= 10
            elif tag[0] in informal:
                formality *0.9#-= 10

        # return formality/s_len + self.f_score(NN_count/s_len*100, JJ_count/s_len*100, IN_count/s_len*100, DT_count/s_len*100,
        #         PRP_count/s_len*100, VB_count/s_len*100, RB_count/s_len*100, UH_count/s_len*100)

        return formality * self.f_score(NN_count/s_len*100, JJ_count/s_len*100, IN_count/s_len*100, DT_count/s_len*100,
                PRP_count/s_len*100, VB_count/s_len*100, RB_count/s_len*100, UH_count/s_len*100)

    # Calculation of the F score
    def f_score(self, NN_freq, JJ_freq, IN_freq, DT_freq, PRP_freq, VB_freq, RB_freq, UH_freq):
        return ((NN_freq + JJ_freq + IN_freq + DT_freq - PRP_freq - VB_freq - RB_freq - UH_freq + 100)/2.0)

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
    			# some qrpairs do not have the author dict and response dict
                try:
                    alignment_dict[k] = qrpair.compute_alignment(features, self.user_his_list, smoothing, weight_vector, alternative="final")
                except:
                    pass

            for key, score in alignment_dict.items():
                alignment_scores.setdefault(key, []).append(score)

    	# Return total alignment scores
        return {key: sum(alignment_scores[key]) for key in alignment_scores}

    def get_features_from_file(self, file_name):
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



if __name__ == '__main__':

    def run():
        print("Loading, please wait...")
        console_app = ConsoleBotClient()
        console_app.run()

    run()
