# All answers between random tag are split by 11039841. Formality of the user is compared to the
# formality of the random responses. The two responses with a formality closest to that of the user
# are used to find the most linguistically aligned response, which is returned. If both responses
# have the same alignment score the response which has a formality score closest to that of the user
# is returned. If there is just a single possible answer, it gets directly returned.
# Also makes sure "@" is replaced with "at" and "<3" with "heart".
def process_response(self, client_context, response):
    if "11039841" in response:
        if "@" in response:
            response = response.replace("@", " at ")
        if "<3" in response:
            response = response.replace("<3", " heart")
        responses_formality = {}
        responses_glove = {}
        best_responses = {}
        two_best_responses = []
        responses_ranking = []
        glove_ranking = []

        # Split by 11039841 and remove empty items from list
        sentences = response.split("11039841")
        sentences = [x for x in sentences if x]
        # print(sentences)
        n_sentences = len(sentences)
        # print(n_sentences)

        # Determine formality for all possible responses
        # start_time = time.time() # Used to test the influence of using GloVe vectors
        for sentence in sentences:
            # print(sentence)
            # responses_formality[self.determine_formality(sentence)] = sentence
            responses_formality[sentence] = self.determine_formality(sentence)
            # print(self.determine_formality(sentence))
          # Used to test the influence of using GloVe vectors
            glove = 1
            formality = self.determine_formality(sentence)
            difference = self.user_formality - formality
            question = self.question.split()
            for word in question:
                if word in word_list:
                    closest = self.find_N_closest_words(self.vec(word), 3, words)
                    for close in closest:
                        if close in sentence:
                            if difference > 0:
                                glove *= 1.1
                            elif difference < 0:
                                glove *= 0.9
            responses_glove[sentence] = formality * glove

        # print(responses_formality)
        for key, val in responses_formality.items():
             print(val, "=>", key)

        temp_dict = responses_formality.copy()
        temp_glove = responses_glove.copy()
        # print(temp_dict)

        for i in range(0,n_sentences):
            # print("hello")
            min_val = min(temp_dict.values(), key=lambda x:abs(x-self.user_formality))
            # print(min_val)
            # print(min_val)
            for key, val in responses_formality.items():
                if val == min_val:
                    if key in temp_dict.keys():
                        # print("hello")
                        responses_ranking.append(key)
                        temp_dict.pop(key)
                        break
            # print(responses_formality)
            # print(temp_dict)
        # print(responses_formality)
        print(responses_ranking)
        print(responses_ranking[0])

        for i in range(0,n_sentences):
            min_glove = min(temp_glove.values(), key=lambda x:abs(x-self.user_formality))
            for key, val in responses_glove.items():
                if val == min_val:
                    if key in temp_glove.keys():
                        # print("hello")
                        glove_ranking.append(key)
                        temp_glove.pop(key)
                        break

        print(glove_ranking)
        print(glove_ranking[0])

        # for i in range(0,n_sentences):
        #     min_val = min(temp_dict.values(), key=lambda x:abs(x-self.user_formality))
        #     # print(min_val)
        #     for key, val in responses_formality.copy().items():
        #         if val == min_val:
        #             responses_ranking.append(key)
        #             temp_dict.pop(key)
        #         print(temp_dict)
        # # print(responses_ranking)
        # print(responses_ranking[0])

        # Find the two responses with closest formality to user formality
        # two_best_formality = nsmallest(2, responses_formality, key=lambda x:abs(x-self.user_formality))
        # two_best_responses.extend((responses_formality[two_best_formality[0]], responses_formality[two_best_formality[1]]))

        # formality_ranking = nsmallest(n_sentences, responses_formality, key=lambda x:abs(x-self.user_formality))
        # # print(formality_ranking)
        # for i in range(0,n_sentences):
        #     print(i)
        #     responses_ranking.append(responses_formality[formality_ranking[i]])
        # print(responses_ranking)
        # print(responses_ranking[0])

        with open("results_formality2.txt", "a") as output:
            output.write(str(self.question_number - 1) + "\n")
            output.write(self.user_history)
            output.write(str(self.user_formality) + "\n")
            output.write(str(responses_formality) + "\n")
            output.write(str(responses_ranking) + "\n")
            output.write(str(responses_ranking[0]) + "\n")


        with open("results_glove.txt", "a") as output:
            output.write(str(self.question_number - 1) + "\n")
            output.write(self.user_history)
            output.write(str(self.user_formality) + "\n")
            output.write(str(responses_glove) + "\n")
            output.write(str(glove_ranking) + "\n")
            output.write(str(glove_ranking[0]) + "\n")

        # total_alignment = self.determine_alignment(sentences)
        # # print(total_alignment)
        # sorted_alignment = sorted(total_alignment.items(), key=operator.itemgetter(1), reverse=True)
        # print(sorted_alignment)
        # print(sorted_alignment[0][0])
        #
        # with open("results_linguistic2.txt", "a") as output:
        #     output.write(str(self.question_number - 1) + "\n")
        #     output.write(self.user_history)
        #     output.write(str(self.user_formality) + "\n")
        #     output.write(str(n_sentences) + "\n")
        #     output.write(str(sorted_alignment) + "\n")
        #
        # for item in two_best_formality:
        #     best_responses[item] = responses_formality[item]
        # print(best_responses)
        #
        # print(two_best_responses)
        # # print("--- %s seconds ---" % (time.time() - start_time)) # Used to test the influence of using GloVe vectors
        #
        # # Determine alignment score of the two best responses
        # total_alignment = self.determine_alignment(best_responses)
        # print(total_alignment)
        # score = list(total_alignment.values())[0]
        # # If scores are equal, print closest formality response. Else print maximum alignment response
        # if all(value == score for value in total_alignment.values()) == True:
        #     print(two_best_responses[0])
        # else:
        #     print(best_responses[max(total_alignment.items(), key=operator.itemgetter(1))[0]])
    else:
        if "@" in response:
            response = response.replace("@", " at ")
        if "<3" in response:
            response = response.replace("<3", " heart")
        print(response)
