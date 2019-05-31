from os import listdir
import pickle
from read_dataset.read_pereira import PereiraReader
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class RSA(object):
    def __init__(self, user_dir, paradigm):        
        self.paradigm = paradigm
        
        self.user_dir = user_dir
        self.data_dir = "/datastore/pereira_data/"
        self.subject_ids = [file for file in listdir(self.data_dir)]
        
        self.save_embedding_dir = user_dir + "embeddings/already_embedded/"
        
        # where the analysis will be stored
        self.save_dir = user_dir + "analyses/already_analysed/"
        
    def load_RSA(self, selection = ""):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        if not self.selection == "searchlight":
            accuracy_file = self.save_dir + "RSA_" + paradigms[paradigm_index] + ".pickle"
        else:
            accuracy_file = self.save_dir + "RSA_" + paradigms[paradigm_index] + "_" + self.selection + ".pickle"
            
        
        if os.path.isfile(accuracy_file):
            print("Loading accuracies from " + accuracy_file)
            with open(accuracy_file, 'rb') as handle:
                correlations = pickle.load(handle)
            return correlations
        else:
            return {}
        
    def read_data(self):
        # get information of all participants
        print("reader start for paradigm " + str(self.paradigm))
        pereira_reader = PereiraReader(self.data_dir, paradigm = self.paradigm)
        pereira_data = pereira_reader.read_all_events() 
        blocks, xyz_voxels = pereira_data
        self.blocks = blocks
        print("reader done...")
        
        # all participants get the same words (and concreteness), therefore I arbitrarily chose M01
        self.words = []
        self.abstract_words = []
        self.concrete_words = []
        for block in self.blocks["M01"]:
            self.words.append([word for word in block.sentences][0][0])
            if block.concreteness_id == "abstract":
                self.abstract_words.append([word for word in block.sentences][0][0])
            else:
                self.concrete_words.append([word for word in block.sentences][0][0])
                
            
        
    def save_correlations(self, data, concreteness, reduced_random = False): 
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        
        if concreteness == "total":
            self.correlations["total"] = data
        elif concreteness == "abstract":
            self.correlations["abstract"] = data
        elif concreteness == "concrete":
            self.correlations["concrete"] = data
            
        if self.selection == "searchlight":
            minimum = 2
        else:
            minimum = 2
            
        if len(self.correlations) == minimum:         
            if self.selection == "searchlight":
                if reduced_random == True:
                    accuracy_file = self.save_dir + "RSA_" + paradigms[paradigm_index] + "_" + self.selection + "_reducedrandom.pickle"
                else:
                    accuracy_file = self.save_dir + "RSA_" + paradigms[paradigm_index] + "_" + self.selection + ".pickle"
                    
            else:
                accuracy_file = self.save_dir + "RSA_" + paradigms[paradigm_index] + ".pickle"
            
            os.makedirs(os.path.dirname(accuracy_file), exist_ok=True)
            with open(accuracy_file, 'wb') as handle:
                pickle.dump(self.correlations, handle)
            print("Accuracies stored in file: " + accuracy_file)
        
        return self.correlations
        
    def runRSA(self, data, labels, concreteness):
        
        x, C = self.get_dists(data, labels, concreteness)
        self.compute_distance_over_dists(x, C, labels, concreteness)
    
    
    def get_dists(self, data, labels=[], concreteness = ""):
        print("Calculating dissimilarity matrix for " + concreteness)
        x = {}
        C = {}
    
        # For each list of vectors
        for i in np.arange(len(data)):
            x[i] = np.asarray(data[i])
            
            # Calculate distances between vectors
            print("Calculating Pearson for: " + labels[i] + " " + str(i))
            C[i] = sp.spatial.distance.cdist(x[i], x[i], 'cosine')
    
# =============================================================================
#         for i in C:
#             if not labels[i] == "RandomEncoder":
#                 print("Start plotting")
#                 self.plot(C[i], [], [x for x in range(1, len(C[i]+1))], labels[i], concreteness)
# =============================================================================

        return x, C
    
    def plot(self, data, p_values, labels, title ="", concreteness = "", cbarlabel = "Pearson Distance", between_matrices = False):
        plt.rcParams["axes.grid"] = False
        plt.interactive(False)
    
        fig, ax = plt.subplots(figsize=(50, 50))
        
        if between_matrices == False:
            if concreteness == "total":
                axis_labels = self.words
            elif concreteness == "abstract":
                axis_labels = self.abstract_words
            elif concreteness == "concrete":
                axis_labels = self.concrete_words
        else:
            axis_labels = labels
            
        im, cbar = self.heatmap(data, p_values, axis_labels, axis_labels, ax=ax,
                           cmap="BuPu_r", vmin = 0, vmax = 1.5, title = title, cbarlabel = cbarlabel)
    
        fig.tight_layout()
        
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        if self.selection == "searchlight":
            plot_name = self.save_dir + "RSA_" + paradigms[self.paradigm -1] + "_" + self.selection + "_" + concreteness + "_" + title + ".png" 
        else:
            plot_name = self.save_dir + "RSA_" + paradigms[self.paradigm -1] + "_roi_stable_" + concreteness + "_" + title + ".png" 
            
        plt.savefig(plot_name)
        
        print("Show plot")
        plt.show(block=True)
        print("Done")
        

    
    def heatmap(self, data, p_values, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", title = "",  **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.
        Arguments:
            data       : A 2D numpy array of shape (N,M)
            row_labels : A list or array of length N with the labels
                         for the rows
            col_labels : A list or array of length M with the labels
                         for the columns
        Optional arguments:
            ax         : A matplotlib.axes.Axes instance to which the heatmap
                         is plotted. If not provided, use current axes or
                         create a new one.
            cbar_kw    : A dictionary with arguments to
                         :meth:`matplotlib.Figure.colorbar`.
            cbarlabel  : The label for the colorbar
        All other arguments are directly passed on to the imshow call.
        """
    
        if not ax:
            ax = plt.gca()
    
        # Plot the heatmap
        im = ax.imshow(data, **kwargs)
        ax.set_title(title, pad =50.0)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
    
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
                 rotation_mode="anchor")
    
        # Turn spines off and create white grid.
        # for edge, spine in ax.spines.items():
        #    spine.set_visible(False)
    
        # what is happening here?
        ax.set_xticks(np.arange(0, data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(0, data.shape[0] + 1) - 0.5, minor=True)
        
        if not p_values == []:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if i == data.shape[0] - 1 or j == data.shape[1] - 1:
                        continue
                    else:
                        if p_values[i][j] < 0.05:
                            #ax.text(i, j, str(round(data[i][j], 2)) + "*", ha="center", va="center", color="w")
                            ax.text(i, j, "*", ha="center", va="center", color="w")
# =============================================================================
#                         else:
#                             ax.text(i, j, round(data[i][j], 2), ha="center", va="center", color="w")
# 
# =============================================================================
    
        return im, cbar

    def compute_distance_over_dists(self, x, C, labels, concreteness):
        
        
        keys = np.asarray(list(x.keys()))

        # makes sure you only get average random in plot, but run accuracy against distribution
        randoms = labels.count("RandomEncoder")
        spearman = {}
        spearman["p-values"] = np.zeros((len(keys) - randoms, len(keys) - randoms))
        spearman["labels"] = np.zeros((len(keys) - randoms, len(keys))).astype(str)
        random_cor = {}
        
        if not self.selection == "searchlight":
            spearman["correlations"] = np.zeros((len(keys) - randoms, len(keys)))
            
            # not compare the randoms against each other, but compare rest against all randoms
            for i in np.arange(len(keys) - randoms):
                random_cor[i] = []
                for j in np.arange(len(keys)):
                    print("Calculating distance between matrices: " + labels[i] + " and " + labels[j])
                    corr_s = []
                    for a, b in zip(C[keys[i]], C[keys[j]]):
                        s, _ = sp.stats.spearmanr(a, b)
                        corr_s.append(s)
                    spearman["correlations"][i][j] = np.mean(corr_s)
                    
                    # save the correlations against random in variable somewhere
                    if labels[j] == "RandomEncoder":
                        random_cor[i].append(np.mean(corr_s))
                        
            for i in np.arange(len(keys) - randoms):
                random_distribution = [spearman["correlations"][i][random_value] for random_value in range((len(keys) - randoms), len(keys))]
                for j in np.arange(len(keys) - randoms):
                    spearman["p-values"][i][j] = self.calculate_p_value(spearman["correlations"][i][j], random_distribution)
                    
            print(spearman)
    
            self.save_correlations(spearman, concreteness)
            
            # make plot with average random correlation instead of distribution
            average_random_spearman = np.zeros((len(keys) - randoms + 1, len(keys) - randoms + 1))
            for i in np.arange(len(keys) - randoms + 1):
                for j in np.arange(len(keys) - randoms + 1):
                    if j == len(keys) - randoms and i == len(keys) - randoms:
                        average_random_spearman[i][j] = 1
                    elif j == len(keys) - randoms:
                        average_random_spearman[i][j] = np.mean(random_cor[i])
                    elif i == len(keys) - randoms:
                        average_random_spearman[i][j] = np.mean(random_cor[j])
                    else:
                        average_random_spearman[i][j] = spearman["correlations"][i][j]
                    
            labels = labels[:len(keys) - randoms + 1]
            # average over the random ones
            
            
            # self.plot(average_random_spearman, spearman["p-values"], labels, title="RDM Comparison Spearman", concreteness = concreteness, cbarlabel="Spearman Correlation", between_matrices = True)
        
        else:
            spearman["correlations"] = np.zeros((len(keys) - randoms, len(keys) - randoms + 1))
            done_subjects = []
            for i in np.arange(len(keys) - randoms):
                if "_" in labels[i]:
                    subject_id = labels[i][:labels[i].index("_")]
                    if not subject_id in done_subjects:
                        done_subjects.append(subject_id)
                        print("calculating spearman for subject " + subject_id)
                    voxel_id = labels[i][labels[i].index("_")+ 1:]
                    if voxel_id in self.random_voxels[subject_id]:
                        random_cor[subject_id].append([])
                        print("calculating random for voxel selection " + str(self.random_voxels[subject_id].index(voxel_id) + 1) + "/3")
                        for j in np.arange(len(keys)):
                            if labels[j] == "RandomEncoder":
                                for a, b in zip(C[keys[i]], C[keys[j]]):
                                    s, _ = sp.stats.spearmanr(a, b)
                                    corr_s.append(s)
                                random_cor[subject_id][self.random_voxels[subject_id].index(voxel_id)].append(np.mean(corr_s))

                for j in np.arange(len(keys)):
                    spearman["labels"][i][j] = labels[i] + "_" + labels[j]
                    if labels[j] == "PereiraEncoder" or labels[j] == "ImageEncoder" or labels[j] == "CombiEncoder":
                        corr_s = []
                        for a, b in zip(C[keys[i]], C[keys[j]]):
                            if np.isnan(a).all() or np.isnan(b).all():
                                corr_s.append(0)
                            else:
                                s, _ = sp.stats.spearmanr(a, b)
                                corr_s.append(s)
                        spearman["correlations"][i][j] = np.mean(corr_s)
                    else:
                        continue
                        
            random_distribution = {}
            for i in np.arange(len(keys) - randoms):
                if "_" in labels[i]:
                    subject_id = labels[i][:labels[i].index("_")]
                    if not subject_id in random_distribution:
                        random_distribution[subject_id] = np.mean(random_cor[subject_id], axis = 0)
                    spearman["correlations"][i][len(keys) - randoms] = np.mean(random_distribution[subject_id])
                else:
                    continue
            
            
                
            for i in np.arange(len(keys) - randoms):
                if "_" in labels[i]:
                    subject_id = labels[i][:labels[i].index("_")]
                    for j in np.arange(len(keys)):
                        if labels[j] == "PereiraEncoder" or labels[j] == "ImageEncoder" or labels[j] == "CombiEncoder":
                            spearman["p-values"][i][j] = self.calculate_p_value(spearman["correlations"][i][j], random_distribution[subject_id])
                else:
                    continue
                
            print(spearman["correlations"])
            print(spearman["p-values"])
            print(spearman["labels"])
            self.save_correlations(spearman, concreteness, reduced_random = True)

            
        
    def calculate_p_value(self, accuracy, random):
    
        # adapted this from https://github.com/scikit-learn/scikit-learn/blob/7b136e92acf49d46251479b75c88cba632de1937/sklearn/model_selection/_validation.py
        # best possible = 1/1001 --> 0.0099
        n_permutations = len(random)
        counter_random_better = 0
        for random_score in random:
            if random_score >= accuracy:
                counter_random_better += 1
        p_value = (counter_random_better + 1) / (n_permutations + 1)
        
        return p_value
