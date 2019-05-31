from result_analyses.result_analyses_abstract import ResultAnalysis
import matplotlib.pyplot as plt
import numpy as np
import result_analyses.table_roi_separate as table
import seaborn as sns
from pandas import DataFrame, melt, concat
import pandas as pd
import scipy.io
import pickle
import os
from nilearn import datasets, plotting
import nibabel as nib
from matplotlib import cm
from statistics import mean, stdev 
from os import listdir
from matplotlib.lines import Line2D
from heapq import nlargest, nsmallest


class EncodingResults(ResultAnalysis):
    
    def __init__(self, user_dir):
         super(EncodingResults, self).__init__(user_dir)
         self.analysis_method = "encoding"
         
    def violin_plot_mix(self, sign_part_decoding = True):
        
        sns.set(style="whitegrid")
            
        for paradigm in [1,2,3]:
            
            self.paradigm = paradigm
                
            long_formats =[]
            
            for selection in ["roi", "stable"]:
                
                if sign_part_decoding == True:
                    signif_part = self.participant_selection_decoding(paradigm, selection)
                 

                if selection == "stable":
                    accuracy_file = self.load_accuracies(selection)

                    for embedding in accuracy_file.keys():  
                        if embedding == "combi":
                            continue
                        
                        # work with averages of random
                        if embedding == "random":
                            for subject_id in accuracy_file[embedding].keys():
                                for concreteness in accuracy_file[embedding][subject_id].keys():
                                    accuracy_file[embedding][subject_id][concreteness] = mean(accuracy_file[embedding][subject_id][concreteness])
                        
                        long_format = DataFrame.from_dict(accuracy_file[embedding], orient='index')
                        long_format = long_format.drop(columns = "total")
                        
                        # only include the participants that were significant during decoding
                        if sign_part_decoding == True:
                            for subject_id in accuracy_file[embedding].keys():
                                if not subject_id in signif_part:
                                    long_format = long_format.drop(subject_id)
                        
                        long_format['new_col'] = range(1, len(long_format) + 1)
                        long_format = long_format.reset_index().drop(columns= "new_col")                   
                        long_format = melt(long_format, id_vars = ['index'], var_name = "concreteness", value_name = "accuracy")
                        
                        if embedding == "linguistic":
                            embedding = "textual"
                        elif embedding == "non-linguistic":
                            embedding = "visual"
                        elif embedding == "random":
                            embedding = "random"

                        long_format['embedding'] = embedding
                        long_format["area"] = "stable"
                        long_formats.append(long_format)

                else:
                    accuracy_file = self.load_accuracies(selection)
                    
                    new_dict = {}
                    # TODO: alter dict so average takes the place of random
                    for embedding in accuracy_file.keys():  
                        if embedding =="combi":
                            continue
                        new_dict[embedding] = {}
                        for subject_id in accuracy_file[embedding].keys():
                            for area in accuracy_file[embedding][subject_id].keys():
                                if area == "post_cing":
                                    continue
                                elif area == "precuneus":
                                    continue
                                elif area == "paraHIP":
                                    continue
                                
                                if not area in new_dict[embedding]:
                                    new_dict[embedding][area] = {}
                                new_dict[embedding][area][subject_id] = {}
                                for concreteness in accuracy_file[embedding][subject_id][area].keys():
                                    if embedding == "random":
                                        new_dict[embedding][area][subject_id][concreteness] = mean(accuracy_file[embedding][subject_id][area][concreteness])
                                    else:
                                        new_dict[embedding][area][subject_id][concreteness] = accuracy_file[embedding][subject_id][area][concreteness]
    
                        for area in new_dict[embedding].keys():

                            
                            long_format = DataFrame.from_dict(new_dict[embedding][area], orient='index')
                            long_format = long_format.drop(columns = "total")
                            
                            # only include the participants that were significant during decoding
                            if sign_part_decoding == True:
                                for subject_id in accuracy_file[embedding].keys():
                                    if not subject_id in signif_part[area]:
                                        long_format = long_format.drop(subject_id)
                            
                            long_format['new_col'] = range(1, len(long_format) + 1)
                            long_format = long_format.reset_index().drop(columns= "new_col")      
                            
                                        
                            long_format = melt(long_format, id_vars = ['index'], var_name = "concreteness", value_name = "accuracy")
                            
                            if embedding == "linguistic":
                                row_embedding = "textual"
                            elif embedding == "non-linguistic":
                                row_embedding = "visual"
                            elif embedding == "random":
                                row_embedding = "random"
                            long_format['embedding'] = row_embedding
                            
                            
                            
                            long_format["area"] = area
                            long_formats.append(long_format)
                            
            long_format = concat(long_formats)
            print(long_format)
            
            fig, axs = plt.subplots(ncols=2, nrows=2,  sharex='row', sharey="row", figsize = (10,10))
            plt.subplots_adjust(wspace=0.05)
            
            
            keys = []
            for area in new_dict["linguistic"].keys():
                keys.append(area)
            keys.append("stable")
            mean_dict = {}
            mean_dict["text"] = {}
            mean_dict["visual"] = {}
            mean_dict["random"] = {}
            
            for embedding in mean_dict.keys():
                for concreteness in ["concrete", "abstract"]:
                    mean_dict[embedding][concreteness] = {}
                    for area in keys:
                        mean_dict[embedding][concreteness][area] = []
            
            counter1 = 0
            counter2 = 0
            
            for area in keys:
                
                axs[counter1][counter2].set_ylim([0.28,0.88])
                
                axs[counter1][counter2].set_title(area, weight='bold')
                
                # make the violin plot   
                plot = sns.violinplot(x = "embedding", y="accuracy", hue="concreteness", hue_order = ["concrete","abstract"], palette = "muted", data=long_format[long_format.area == area], split=True, ax = axs[counter1][counter2], inner = None,linewidth = 0)
                
                
                # indicate the means in the plot
                data = long_format[long_format.area == area]
                text = data[data.embedding == "textual"]
                visual = data[data.embedding == "visual"]
                random = data[data.embedding == "random"]
                mean_dict["text"]["concrete"][area] = mean(text.accuracy[text.concreteness == "concrete"])
                mean_dict["text"]["abstract"][area]  = mean(text.accuracy[text.concreteness == "abstract"])
                mean_dict["visual"]["concrete"][area]  = mean(visual.accuracy[visual.concreteness == "concrete"])
                mean_dict["visual"]["abstract"][area] = mean(visual.accuracy[visual.concreteness == "abstract"])
                mean_dict["random"]["concrete"][area]  = mean(random.accuracy[random.concreteness == "concrete"])
                mean_dict["random"]["abstract"][area] = mean(random.accuracy[random.concreteness == "abstract"])
                df = pd.DataFrame({
                        'x': [-0.1,0,0.9,1, 1.9,2],
                        'y': [mean_dict[embedding][concreteness][area] for embedding in mean_dict for concreteness in mean_dict[embedding]],
                        })
    
                
                means = sns.regplot(data=df, x="x", y="y", fit_reg=False, marker=1, color="black", ax = axs[counter1][counter2])
                for line in range(0,df.shape[0]):
                    if df.x[line] % 1 == 0:
                        means.text(df.x[line]+0.1, df.y[line], round(df.y[line],2), horizontalalignment='left', size='small', color='black', weight='semibold')
                    else:
                         means.text(df.x[line]-0.1, df.y[line], round(df.y[line],2), horizontalalignment='right', size='small', color='black', weight='semibold')
                
                if area == "MTG" and paradigm == 3:
                    plot.get_legend().set_visible(True)
                else:
                    plot.get_legend().set_visible(False)
                plot.set(xlabel="", ylabel="") 
                          
                      
                if self.paradigm == 1 and counter2 == 0:
                    axs[counter1][counter2].set_ylabel("Accuracy", weight = "bold")  
                    axs[counter1][counter2].set_yticklabels(["",0.3,0.4,0.5,0.6,0.7,0.8])
                else:
                    axs[counter1][counter2].set_ylabel("")
                if counter1 == 1:
                    axs[counter1][counter2].set_xticklabels(["textual", "visual", "random"], weight="bold")

                else:
                    axs[counter1][counter2].set_xticklabels("")
                    
                
                
                
                
                
                if counter1 == 0 and counter2 == 0:
                    counter2 = 1
                elif counter1 == 0 and counter2 == 1:
                    counter1 = 1
                    counter2 = 0
                elif counter1 == 1 and counter2 == 0:
                    counter2 = 1    


  
    def participant_selection_decoding(self, paradigm, selection_method):
        self.analysis_method = "classification"
        accuracy_file = self.load_accuracies(selection_method)
        self.analysis_method = "encoding"

        accuracy_file = self.transform_randoms(accuracy_file, selection_method)
        
        
        
        if selection_method == "stable": 
            significant_participants = []
            for subject_id in accuracy_file["accuracy"].keys():
                random_scores = accuracy_file["random"][subject_id]
                accuracy_score = accuracy_file["accuracy"][subject_id]
                p_value = self.calculate_p_value(accuracy_score, random_scores)
                if p_value < 0.05:
                    significant_participants.append(subject_id)
                    print("accuracy and p-value for participant " + subject_id + " are: " + str(accuracy_score) + " and " + str(p_value))
                    
        else:
            significant_participants = {}
            for area in accuracy_file["accuracy"]["M01"].keys():
                significant_participants[area] = []
            
            for subject_id in accuracy_file["accuracy"].keys():
                for area in accuracy_file["accuracy"][subject_id].keys():
                    random_scores = accuracy_file["random"][subject_id][area]
                    accuracy_score = accuracy_file["accuracy"][subject_id][area]
                    p_value = self.calculate_p_value(accuracy_score, random_scores)
                    if p_value < 0.05:
                        significant_participants[area].append(subject_id)
            
        return significant_participants
    
    
    def transform_randoms(self, accuracy_file, selection_method):
        
        # make sure random accuracies are given as one array instead of 1 array with frequencies and 1 array with values
        # this is easier for plotting and calculating p-values
        random_distribution = {}
        if selection_method == "stable":
            for subject_id in accuracy_file["random"].keys():
                random_accuracy = accuracy_file["random"][subject_id][0]
                frequencies = accuracy_file["random"][subject_id][1]
                random_distribution[subject_id] = {}
                
                random_distr = []
                index = 0
                for frequency in frequencies:
                    for times in range(0,int(frequency)):
                        random_distr.append(random_accuracy[index])
                    index = index + 1
                random_distribution[subject_id] = random_distr
                
        
        else:
            for subject_id in accuracy_file["random"].keys():
                random_distribution[subject_id] = {}
                for area in accuracy_file["random"][subject_id].keys():
                    random_accuracy = accuracy_file["random"][subject_id][area][0]
                    frequencies = accuracy_file["random"][subject_id][area][1]
                    random_distribution[subject_id][area] = {}
                    
                    random_distr = []
                    index = 0
                    for frequency in frequencies:
                        for times in range(0,int(frequency)):
                            random_distr.append(random_accuracy[index])
                        index = index + 1
                    random_distribution[subject_id][area] = random_distr
                
            
        accuracy_file["random"] = random_distribution

        return accuracy_file
    

        
        
        
