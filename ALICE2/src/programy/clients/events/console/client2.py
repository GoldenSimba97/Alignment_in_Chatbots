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

from programy.utils.logging.ylogger import YLogger

from programy.clients.events.client import EventBotClient
from programy.clients.events.console.config import ConsoleConfiguration

# Open formal and informal word lists
formal = []
formal_file = open("FormalityLists/formal_list", "r")
for line in formal_file:
    line = line.replace("\n", "")
    formal.append(line)

informal = []
informal_file = open("FormalityLists/informal_list", "r")
for line in informal_file:
    line = line.replace("\n", "")
    informal.append(line)

class ConsoleBotClient(EventBotClient):

    def __init__(self, argument_parser=None):
        self.running = False
        EventBotClient.__init__(self, "Console", argument_parser)

        # Variables added for Thesis
        self.user_history = ""
        self.question_number = 0
        self.user_formality = 0
        self.filename = "user_results_v2.txt"

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
        initial_question = client_context.bot.get_initial_question(client_context)
        self._renderer.render(client_context, initial_question)

    def process_question(self, client_context, question):
        # Returns a response
        return client_context.bot.ask_question(client_context , question, responselogger=self)

    def render_response(self, client_context, response):
        # Calls the renderer which handles RCS context, and then calls back to the client to show response
        self._renderer.render(client_context, response)

    # Added that the results will now be written to a file
    def process_response(self, client_context, response):
        print(response)
        if self.question_number > 0 and response != "\nBye!":
            self.write_results_to_file2(response)

    def process_question_answer(self, client_context):
        question = self.get_question(client_context)

        # Expand user history
        self.user_history = self.user_history + question + ". "

        # Determine user formality over whole user history
        self.user_formality = self.determine_formality(self.user_history)

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

            # To anounce the end of the conversation
            with open(self.filename, "a") as output:
                output.write("--------------------------------------------------------------------------------" + "\n")

            # Changed exit response. Can be further changed to accomodate the specific user
            exit_response = "\nBye!"
            self._renderer.render(client_context, exit_response)
        except Exception as excep:
            YLogger.exception(self, "Oops something bad happened !", e)
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

        # Get the counts needed to determine frequencies for calculation of F-score
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
            elif tag[1] == "PRP" or tag[1] == "PRP$" or tag[1] == "WP" or tag[1] == "WP$":
                PRP_count += 1
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
                formality *= 1.1
            elif tag[0] in informal:
                formality *= 0.9

        return formality * self.f_score(NN_count/s_len*100, JJ_count/s_len*100, IN_count/s_len*100, DT_count/s_len*100,
                PRP_count/s_len*100, VB_count/s_len*100, RB_count/s_len*100, UH_count/s_len*100)

    # Calculate the F-score
    def f_score(self, NN_freq, JJ_freq, IN_freq, DT_freq, PRP_freq, VB_freq, RB_freq, UH_freq):
        return ((NN_freq + JJ_freq + IN_freq + DT_freq - PRP_freq - VB_freq - RB_freq - UH_freq + 100)/2)

    # Write results to file
    def write_results_to_file2(self, response):
        with open(self.filename, "a") as output:
            output.write(str(self.question_number) + "\n")
            output.write(str(self.user_formality) + "\n")
            output.write(str(response) + " => " + str(self.determine_formality(str(response))) + "\n")


if __name__ == '__main__':

    def run():
        print("Loading, please wait...")
        console_app = ConsoleBotClient()
        console_app.run()

    run()
