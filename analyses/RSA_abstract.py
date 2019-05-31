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
        self.data_dir = "DATADIR"
        self.subject_ids = [file for file in listdir(self.data_dir)]
        
        self.save_embedding_dir = user_dir + "embeddings/already_embedded/"
        
        # where the analysis will be stored
        self.save_dir = user_dir + "analyses/already_analysed/"
        
    def load_RSA(self, selection = ""):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        accuracy_file = self.save_dir + "RSA_" + paradigms[paradigm_index] + ".pickle"            
        
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
        
            
        if len(self.correlations) == 3:                   
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
            print("Calculating cosine for: " + labels[i] + " " + str(i))
            C[i] = sp.spatial.distance.cdist(x[i], x[i], 'cosine')
    
# =============================================================================
#         for i in C:
#             if not labels[i] == "RandomEncoder":
#                 print("Start plotting")
#                 self.plot(C[i], [x for x in range(1, len(C[i]+1))], labels[i], concreteness)
# =============================================================================

        return x, C
    
    def plot(self, data, labels, title ="", concreteness = "", cbarlabel = "Cosine", between_matrices = False):
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
            
        im, cbar = self.heatmap(data, axis_labels, axis_labels, ax=ax,
                           cmap="BuPu_r", vmin = 0, vmax = 1.5, title = title, cbarlabel = cbarlabel)
    
        fig.tight_layout()
        
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        plot_name = self.save_dir + "RSA_" + paradigms[self.paradigm -1] + "_roi_stable_" + concreteness + "_" + title + ".png" 
            
        plt.savefig(plot_name)
        
        print("Show plot")
        plt.show(block=True)
        print("Done")
        

    
    def heatmap(self, data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", title = "",  **kwargs):
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
    
        return im, cbar

    def compute_distance_over_dists(self, x, C, labels, concreteness):
        
        
        keys = np.asarray(list(x.keys()))

        # makes sure you only get average random in plot, but run accuracy against distribution
        randoms = labels.count("RandomEncoder")
        spearman = {}
        spearman["labels"] = np.zeros((len(keys) - randoms, len(keys))).astype(str)
        random_cor = {}
        
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


        self.plot(average_random_spearman, labels, title="RDM Comparison Spearman", concreteness = concreteness, cbarlabel="Spearman Correlation", between_matrices = True)
