import os
import csv
import sys
import math
import numpy as np
# import cPickle as pickle
# import matplotlib.pyplot as plt
from collections import defaultdict

from utterance import Utterance
from post import Post
from author import Author

# import krippendorff_alpha

class Align:
    def __init__(self, quote, responses_dict):
        self.utterance_dict = {}
        self.author_dict = {}
        # self.author = ""
        self.marker_dict = {}
        # author_dict niet gebruiken. User compute_baseline, maar chatbot niet

        quote_post = Post(quote, "user")
        for key, response in responses_dict.items():
            response_post = Post(response, "chatbot")
            utterance = Utterance(quote_post, response_post, key)
            author1 = Author(utterance.response_object.author)
            self.author_dict[utterance.response_object.author] = author1
            author2 = Author(utterance.quote_object.author)
            self.author_dict[utterance.quote_object.author] = author2
            author1.add_utterance(utterance)
            author2.add_utterance(utterance)
            self.utterance_dict[str(key)] = utterance

        # self.marker_dict = self.read_markers("coordination_markers")
        # self.markers()
        self.marker_dict = self.read_markers("coordination_markers")
        print(self.marker_dict)
        print(self.author_dict)

        for a, author in self.author_dict.items():
            for marker_name, marker_list in self.marker_dict.items():
                author.compute_baseline(marker_name, marker_list)

    # def create_author_dict(self):
    #     '''Create dict of author objects
    #     '''
    #     print("Creating dictionary of authors from utterances...")
    #     count = 0
    #     for utterance in self.utterance_dict.values():
    #         sys.stdout.write('Utterance: %s\r' % count)
    #         sys.stdout.flush()
    #         try:
    #             author = self.author_dict[utterance.response_object.author]
    #         except KeyError:
    #             author = Author(utterance.response_object.author)
    #             self.author_dict[utterance.response_object.author] = author
    #         author.add_utterance(utterance)
    #         count += 1
    #     sys.stdout.write('Utterance: %s\n' % count)
        # print "Dumping author dict..."
        # pickle.dump(self.author_dict, open("author_dict.p", "wb"))



    # def compute_baselines(self):
    #     '''Compute baselines for each author
    #     and each set of markers
    #     '''
    #     print("Computing baselines for markers for each author...")
    #     for author_name, author in self.author_dict.iteritems():
    #         sys.stdout.write('Author: %s\r' % author_name)
    #         sys.stdout.flush()
    #         for marker_name, marker_list in self.marker_dict.iteritems():
    #             author.compute_baseline(marker_name, marker_list)
    #     sys.stdout.write('Author: %s\n' % author_name)


    def compute_alignments(self, weight_vector):
        print("Computing alignments for markers for each author...")
        diff_f_p_dict = {}
        diff_f_w_dict = {}
        a_f_w_dict = {}
        a_f_p_dict = {}
        b_f_p_dict = {}
        b_f_w_dict = {}
        for a, author in self.author_dict.items():

            for marker_name, marker_list in self.marker_dict.items():
                for utterance in author.get_responses(""):

                    if utterance.in_quote(marker_list):
                        appearance_count = utterance.compute_appearance(marker_list)
                        marker_frequency = appearance_count / utterance.response_object.num_words
                        if appearance_count:
                            post_appearance = 1.0
                        else:
                            post_appearance = 0.0

                        alignment_vector = np.array([post_appearance, marker_frequency])
                        try:
                            a_f_p_dict[marker_name].append(post_appearance)
                        except KeyError:
                            a_f_p_dict[marker_name] = [post_appearance]
                        try:
                            a_f_w_dict[marker_name].append(marker_frequency)
                        except KeyError:
                            a_f_w_dict[marker_name] = [marker_frequency]


                        #print "AV",alignment_vector
                        baseline_vector = author.get_baseline(marker_name)
                        try:
                            b_f_p_dict[marker_name].append(baseline_vector[0])
                        except KeyError:
                            b_f_p_dict[marker_name] = [baseline_vector[0]]
                        try:
                            b_f_w_dict[marker_name].append(baseline_vector[1])
                        except KeyError:
                            b_f_w_dict[marker_name] = [baseline_vector[1]]
                        #print "BV", baseline_vector
                        diff_vector = alignment_vector-baseline_vector
                        #print "DV", diff_vector
                        utterance.add_alignment_vector(alignment_vector, marker_name)

                        try:
                            diff_f_p_dict[marker_name].append(diff_vector[0])
                        except KeyError:
                            diff_f_p_dict[marker_name] = [diff_vector[0]]
                        try:
                            diff_f_w_dict[marker_name].append(diff_vector[1])
                        except KeyError:
                            diff_f_w_dict[marker_name] = [diff_vector[1]]
        # with open('val_dir/vals' + str(weight_vector[0]) + '.txt', 'wb') as f :
        #     for marker_name in b_f_w_dict:
        #         f.write(marker_name + ":" + "b_f_w:" + str(max(b_f_w_dict[marker_name])) + ";" + str(min(b_f_w_dict[marker_name])) +  "\n")
        #         f.write(marker_name + ":" + "b_f_p:" + str(max(b_f_p_dict[marker_name])) + ";" + str(min(b_f_p_dict[marker_name])) +  "\n")
        #         f.write(marker_name + ":" + "a_f_w:" + str(max(a_f_w_dict[marker_name])) + ";" + str(min(a_f_w_dict[marker_name])) +  "\n")
        #         f.write(marker_name + ":" + "a_f_p:" + str(max(a_f_p_dict[marker_name])) + ";" + str(min(a_f_p_dict[marker_name])) +  "\n")

        for a, author in self.author_dict.items():
            for marker_name, marker_list in self.marker_dict.items():
                baseline_vector = author.get_baseline(marker_name)
                max_a_f_p = max(a_f_p_dict[marker_name])
                min_a_f_p = min(a_f_p_dict[marker_name])
                max_a_f_w = max(a_f_w_dict[marker_name])
                min_a_f_w = min(a_f_w_dict[marker_name])

                max_b_f_p = max(b_f_p_dict[marker_name])
                min_b_f_p = min(b_f_p_dict[marker_name])
                max_b_f_w = max(b_f_w_dict[marker_name])
                min_b_f_w = min(b_f_w_dict[marker_name])


                for utterance in author.get_responses(""):
                    print("HELLO")
                    if utterance.in_quote(marker_list):
                        alignment_vector = utterance.get_alignment_vector(marker_name)
                        #print "DV", diff_vector
                        normalized_alignment_vector = self.normalize(alignment_vector, max_a_f_p, min_a_f_p, max_a_f_w, min_a_f_w) #alignment_vector
                        normalized_baseline_vector = self.normalize(baseline_vector, max_b_f_p, min_b_f_p, max_b_f_w, min_b_f_w) #baseline_vector
                        print(normalized_alignment_vector.shape)
                        #print "NDV", normalized_diff_vector
                        normalized_vec = np.dot(normalized_alignment_vector - normalized_baseline_vector, weight_vector)
                        #print "NAV", normalized_alignment_vector
                        #print "align:", current_use - baseline_use,"current:", current_use, "base:", baseline_use, "len:", utterance.response_object.num_words
                        utterance.add_alignment(normalized_vec, marker_name)
                    else:
                        utterance.add_alignment(None, marker_name)

        return normalized_alignment_vector


    def normalize(self, vector, first_max, first_min, second_max, second_min):
        #print "Normalizing", vector
        normalized_vec = [0,0]
        f_p = vector[0]
        f_w = vector[1]
        #print "first_max", first_max, "first_min", first_min
        subtract_max = (f_p - first_min)
        max_min = (first_max - first_min)
        normed = subtract_max / max_min
        #print subtract_max, "/", max_min, "=", normed
        normalized_vec[0] = normed
        #print "f_p", f_p, "normalized", normalized_vec[0]
        #print "second_max", second_max, "second_min", second_min
        subtract_max = (f_w - second_min)
        max_min = (second_max - second_min)
        normed = subtract_max / max_min
        #print subtract_max, "/", max_min, "=", normed
        normalized_vec[1] = normed
        #print "f_w", f_w, "normalized", normalized_vec[1]
        #print "to", normalized_vec
        return np.array(normalized_vec)


    def read_markers(self, marker_folder):
        file_list = os.listdir(marker_folder)
        for marker_file in file_list:
            marker_list = self.get_features_from_file(os.path.join(marker_folder, marker_file))
            marker_name = marker_file.split(".txt")[0]
            self.marker_dict[marker_name] = marker_list
        return self.marker_dict


    # def count_markers(self):
    #     count_dict = defaultdict(int)
    #     presence_dict = defaultdict(int)
    #     for a, author in self.author_dict.iteritems():
    #         for utterance in author.get_responses(""):
    #
    #             presence_dict['posts']  += 1
    #             words = utterance.response_object.words
    #             for marker_name, marker_list in self.marker_dict.iteritems():
    #                 for marker in marker_list:
    #                     for word in words:
    #                         count_dict['words'] +=1
    #                         if word == marker:
    #                            count_dict[marker] += 1
    #                 for marker in marker_list:
    #                     if marker in words:
    #                         #print marker, marker_name
    #                         presence_dict[marker_name] += 1
    #                         #print presence_dict[marker_name]
    #                         break
    #
    #     return count_dict, presence_dict


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

responses_dict = {40.0: "You're making your point.", 50.0: 'Swearing is often cathartic.'}
align = Align("you cunt, why won't you tell me something", responses_dict)
# align.read_markers("coordination_markers")
# print(align.marker_dict)
# align.create_author_dict()
# print(align.author_dict)
print(align.compute_alignments([0,0]))


# print(align.count_markers)

# if __name__ == '__main__':
#
#     def run():
#         print("Read Markers, please wait...")
#         console_app = ConsoleBotClient()
#         console_app.run()
#
#     run()
