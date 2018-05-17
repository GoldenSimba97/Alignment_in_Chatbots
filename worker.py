import os
import csv
import sys
import math
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from collections import defaultdict

from utterance import Utterance
from post import Post
from author import Author

import krippendorff_alpha

class Worker:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.temp_dict = {}
        self.author_dict = {}
        self.marker_dict = {}

    def read_data(self, load=False):
        if load:
            print "Loading author dict from file..."
            self.author_dict = pickle.load(open("author_dict.p", "rb"))
        else:
            #create utterance object
            self.create_utterances()
            #get or create author object for utterance
            self.create_author_dict()

    def create_utterances(self):
        print "Creating utterances from", self.root_folder + "qr_meta.csv..."
        with open(os.path.join(self.root_folder, 'qr_meta.csv'), 'rb') as meta_file:
            reader = csv.reader(meta_file, delimiter=',', quotechar='"')
            count = 0
            for row in reader:
                sys.stdout.write('Row: %s\r' % count)
                sys.stdout.flush()
                key = row[0]
                discussion_id = row[1]
                response_post_id = row[2]
                quote_post_id = row[3]
                quote = row[8]
                response = row[9]

                quote_post = Post(quote, discussion_id, quote_post_id)
                response_post = Post(response, discussion_id, response_post_id)
                utterance = Utterance(quote_post, response_post, key, discussion_id)

                self.temp_dict[str(key) + str(discussion_id)] = utterance

                count += 1
        sys.stdout.write('Row: %s\n' % count)

    def create_author_dict(self):
        '''Create dict of author objects
        '''
        print "Creating dictionary of authors from utterances..."
        count = 0
        for utterance in self.temp_dict.values():
            sys.stdout.write('Utterance: %s\r' % count)
            sys.stdout.flush()
            try:
                author = self.author_dict[utterance.response_object.author]
            except KeyError:
                author = Author(utterance.response_object.author)
                self.author_dict[utterance.response_object.author] = author
            author.add_utterance(utterance)
            count += 1
        sys.stdout.write('Utterance: %s\n' % count)
        print "Dumping author dict..."
        pickle.dump(self.author_dict, open("author_dict.p", "wb"))

    def compute_baselines(self):
        '''Compute baselines for each author
        and each set of markers
        '''
        print "Computing baselines for markers for each author..."
        for author_name, author in self.author_dict.iteritems():
            sys.stdout.write('Author: %s\r' % author_name)
            sys.stdout.flush()
            for marker_name, marker_list in self.marker_dict.iteritems():
                author.compute_baseline(marker_name, marker_list)
        sys.stdout.write('Author: %s\n' % author_name)

    def compute_alignments(self, weight_vector):
        print("Computing alignments for markers for each author...")
        diff_f_p_dict = {}
        diff_f_w_dict = {}
        a_f_w_dict = {}
        a_f_p_dict = {}
        b_f_p_dict = {}
        b_f_w_dict = {}
        for a, author in self.author_dict.iteritems():

            for marker_name, marker_list in self.marker_dict.iteritems():
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
        with open('val_dir/vals' + str(weight_vector[0]) + '.txt', 'wb') as f :
            for marker_name in b_f_w_dict:
                f.write(marker_name + ":" + "b_f_w:" + str(max(b_f_w_dict[marker_name])) + ";" + str(min(b_f_w_dict[marker_name])) +  "\n")
                f.write(marker_name + ":" + "b_f_p:" + str(max(b_f_p_dict[marker_name])) + ";" + str(min(b_f_p_dict[marker_name])) +  "\n")
                f.write(marker_name + ":" + "a_f_w:" + str(max(a_f_w_dict[marker_name])) + ";" + str(min(a_f_w_dict[marker_name])) +  "\n")
                f.write(marker_name + ":" + "a_f_p:" + str(max(a_f_p_dict[marker_name])) + ";" + str(min(a_f_p_dict[marker_name])) +  "\n")

        for a, author in self.author_dict.iteritems():
            for marker_name, marker_list in self.marker_dict.iteritems():
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
                    if utterance.in_quote(marker_list):
                        alignment_vector = utterance.get_alignment_vector(marker_name)
                        #print "DV", diff_vector
                        normalized_alignment_vector = alignment_vector #self.normalize(alignment_vector, max_a_f_p, min_a_f_p, max_a_f_w, min_a_f_w)
                        normalized_baseline_vector = baseline_vector #self.normalize(baseline_vector, max_b_f_p, min_b_f_p, max_b_f_w, min_b_f_w)
                        #print "NDV", normalized_diff_vector
                        normalized_vec = np.dot(normalized_alignment_vector - normalized_baseline_vector, weight_vector)
                        #print "NAV", normalized_alignment_vector
                        #print "align:", current_use - baseline_use,"current:", current_use, "base:", baseline_use, "len:", utterance.response_object.num_words
                        utterance.add_alignment(normalized_vec, marker_name)
                    else:
                        utterance.add_alignment(None, marker_name)

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

    def count_markers(self):
        count_dict = defaultdict(int)
        presence_dict = defaultdict(int)
        for a, author in self.author_dict.iteritems():
            for utterance in author.get_responses(""):

                presence_dict['posts']  += 1
                words = utterance.response_object.words
                for marker_name, marker_list in self.marker_dict.iteritems():
                    for marker in marker_list:
                        for word in words:
                            count_dict['words'] +=1
                            if word == marker:
                               count_dict[marker] += 1
                    for marker in marker_list:
                        if marker in words:
                            #print marker, marker_name
                            presence_dict[marker_name] += 1
                            #print presence_dict[marker_name]
                            break

        return count_dict, presence_dict



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



    def inner_default_dict_to_dict(self, default_dictionary):
        for k, v in default_dictionary.iteritems():
            default_dictionary[k] = dict(v)

        return dict(default_dictionary)



    def write_csv(self, file_name):
        print "Writing to file " + file_name
        with open(file_name, 'wb') as f:
            f.write('key, discussion_id, adverbs, articles, auxiliaryverbs, conjunctions, impersonal pronouns, personal pronouns, prepositions, quantifiers,quotelength, responselength, number_posts\n')
            for a, author in self.author_dict.iteritems():

                    for ut in author.get_responses(""):
                        #If the utterance is eligible
                        if ut.available:
                            ut_key = '"' + ut.key + '",' + ut.discussion_id + ","
                            f.write(ut_key)
                            f.write(str(ut.agreement) + ", " + str(ut.nicenasty) + ", " + str(ut.sarcasm) + ", ")
                            f.write(str(ut.get_alignment('adverbs')) + ", " )
                            f.write(str(ut.get_alignment('articles')) + ", " )
                            f.write(str(ut.get_alignment('auxiliaryverbs')) + ", " )
                            f.write(str(ut.get_alignment('conjunctions')) + ", " )
                            f.write(str(ut.get_alignment('impersonalpronouns')) + ", " )
                            f.write(str(ut.get_alignment('personalpronouns')) + ", " )
                            f.write(str(ut.get_alignment('prepositions')) + ", " )
                            f.write(str(ut.get_alignment('quantifiers')) + ", " )
                            f.write(str(ut.quote_object.num_words) + ", ")
                            f.write(str(ut.response_object.num_words) + ", ")
                            f.write(str(author.num_posts) + ", ")


    def transform_dict(self, dictionary):
        new_dict = defaultdict(int)
        for key, val in dictionary.iteritems():
            for marker_name, marker_list in self.marker_dict.iteritems():
                if key in marker_list:
                    new_dict[marker_name] += (val) #/dictionary['words']

        return new_dict



# def ordinal_dist(v1, v2):
#     """
#     Quick function for ordinal distance (see: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha#Difference_functions). and http://repository.upenn.edu/cgi/viewcontent.cgi?article=1043&context=asc_papers
#     Distance function needs to return a float between 0.0 and 1.0 to be correctly used in nltk's AnnotationTask
#     note: ugly quick code, sorry, how can i not find a python function for this..?
#     """
#     # values are the same, distance is 0
#     if v1 == v2:
#         return 0.0

#     # need to loop between values, so for easiness find the upper and lower limit
#     elif v1 < v2:
#         lower_lim = v1
#         upper_lim = v2
#     else:
#         lower_lim = v2
#         upper_lim = v1

#     total = 0.0

#     # convert rank -5 to 5 to scale of 0-10
#     ranking = range(-5, 6)
#     lower_lim_index = ranking.index(int(lower_lim))
#     upper_lim_index = ranking.index(int(upper_lim))

#     # min and max used rank (0 and 10), not quite sure if the min and max are really -5 and 5 in the data set after filtering
#     normalizer = float(len(ranking) - (0+10.0)/2)

#     # not quite sure where the summation ends.. i think after the n_g.
#     # if it ends after the brackets:
#     # constant =  float(lower_lim_index + upper_lim_index) /2.0
#     # for n_g in range(int(lower_lim), int(upper_lim)+1):
#     #     print ranking.index(n_g) - constant
#     #     total += ranking.index(n_g) - constant

#     # if it ends after n_g:
#     n_g = sum(range(int(lower_lim_index), int(upper_lim_index)+1))
#     total = n_g - float(lower_lim_index + upper_lim_index) /2.0


#     # TODO!! NEEDS NORMALIZING PROBABLY..

#     #print "ord distance: ", math.pow(total/normalizer,2)
#     #return math.pow(total/normalizer, 2)\
#     #print math.pow((upper_lim-lower_lim)/11.0,2)

#     return math.pow((upper_lim-lower_lim)/11.0,2)



if __name__=="__main__":
    import optparse
    optParser = optparse.OptionParser()
    optParser.add_option("--w0", help="Get weight of first element in weight vector", default=0.5)
    options, args = optParser.parse_args()

    print "Selected w0=", options.w0
    root_folder = '../data/fourforums/annotations/mechanical_turk/'
    worker_bee = Worker(root_folder)
    print "NOTE: Loading data from file! Make sure author_dict.p is up to date, otherwise set load=False manually.."
    worker_bee.read_data(load=True)
    marker_folder = 'coordination_markers/'
    worker_bee.read_markers(marker_folder)
    start_d, presence_d = worker_bee.count_markers()
    print "words", start_d['words']
    D = worker_bee.transform_dict(start_d)
    #E = worker_bee.transform_dict(presence_d)

    # plt.bar(range(len(D)), D.values(), align='center')
    # plt.xticks(range(len(D)), D.keys())
    # print D
    # plt.show()
    # plt.clf()
    # plt.bar(range(len(presence_d)), presence_d.values(), align='center')
    # plt.xticks(range(len(presence_d)), presence_d.keys())
    # print presence_d
    # plt.show()

    worker_bee.compute_baselines()

    w0 = float(options.w0)
    weight_vector = np.array([w0, 1-w0])
    print weight_vector
    worker_bee.compute_alignments(weight_vector)
    file_stream = os.path.join(root_folder, 'qr_worker_answers_task1.csv')
    import nltk


    # task_sarcasm = nltk.metrics.agreement.AnnotationTask(data=worker_bee.selected_list_sarcasm)
    # # The problem here is the distance value. By default it is set to binary, however the paper uses 'ordinal distance' i think: "assuming an ordinal scale for all measures except sarcasm;"
    # # this is why the alphas for agreement and nicenastiness are lower than the paper,  it doesnt take into account ranks or that  -4 and -5 are almost similar, it just sees it as no interannotator agreement
    # task_agreement = nltk.metrics.agreement.AnnotationTask(data=worker_bee.selected_list_agreement, distance=ordinal_dist)
    # task_nicenasty = nltk.metrics.agreement.AnnotationTask(data=worker_bee.selected_list_nicenasty, distance=ordinal_dist)




    file_name = 'csv/unnormalized_alignment/test_file' + str(w0) + '.csv'
    worker_bee.write_csv(file_name)
    #worker_bee.test()
    #worker_bee.test()'''
