#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:10:08 2019

@author: evihendrikx
"""

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
                
# =============================================================================
#                 axs[counter1][counter2].set_ylabel('')    
#                 ax[counter1][counter2].set_xlabel('')
# =============================================================================
                
                
                      
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

                
# =============================================================================
#                 if counter1 == 0 and counter2 == 0:
#                     counter2 = 1
#                     axs[counter1][counter2].set_xticklabels("")
#                     axs[counter1][counter2].set_yticklabels("")
#                 elif counter1 == 0 and counter2 == 1:
#                     axs[counter1][counter2].set_xticklabels("")
#                     axs[counter1][counter2].set_yticklabels(["",0.3,0.4,0.5,0.6,0.7,0.8], weight="bold")
#                     counter1 = 1
#                     counter2 = 0
#                 elif counter1 == 1 and counter2 == 0:
#                     counter2 = 1
# =============================================================================
            
                
                


            
# =============================================================================
#     def make_table(self):
# 
#         # put information accuracy file in table
#         # TODO: alter dict so average takes the place of random
#         # TODO: include calculating significance 
# 
#         mean_accuracies = {}
#         p_value = {}
#         sign_counter = {}
#         
#         for selection_method in ["roi", "stable"]: 
#             p_value[selection_method] = {}
#             if selection_method == "roi":
#                 mean_accuracies[selection_method] = {}
#                 p_value[selection_method] = {}
#                 sign_counter[selection_method] = {}
# 
#                 # Combine info from various files
#                 for paradigm in [1,2,3]:
#                     self.paradigm = paradigm
#                     accuracy_file = self.load_accuracies(selection_method)
#                     p_value[selection_method][paradigm] = {}
#                     mean_accuracies[selection_method][paradigm] = {}
#                     sign_counter[selection_method][paradigm] = {}
#                     
#                     # all participants have the same areas in all embeddings
#                     areas = [area for  area in accuracy_file["linguistic"]["M01"].keys()]
#                     for area in areas:
#                         mean_accuracies[selection_method][paradigm][area] = {}
#                         sign_counter[selection_method][paradigm][area] = {}
#                         p_value[selection_method][paradigm][area] = {}
#                         for embedding in accuracy_file.keys():
#                             mean_accuracies[selection_method][paradigm][area][embedding] = {}
#                             if not embedding == "random":
#                                 p_value[selection_method][paradigm][area][embedding] = {}
#                                 sign_counter[selection_method][paradigm][area][embedding] = {}
#                                 collect_total_abstract_concrete = []
#                                 
#                                 for concreteness in accuracy_file[embedding]["M01"]["IFG"].keys():
#                                     sign_counter[selection_method][paradigm][area][embedding][concreteness] = 0
#                                 
#                                 for subject_id in accuracy_file[embedding].keys():
#                                     collect_total_abstract_concrete.append(np.array(list(accuracy_file[embedding][subject_id][area].values())))
#                                     p_value[selection_method][paradigm][area][embedding][subject_id] = {}
#                                     for concreteness in accuracy_file[embedding][subject_id][area].keys():
#                                         pvalue = self.calculate_p_value(accuracy_file[embedding][subject_id][area][concreteness], accuracy_file["random"][subject_id][area][concreteness])
#                                         p_value[selection_method][paradigm][area][embedding][subject_id][concreteness] = pvalue
#                                         if pvalue < 0.05:
#                                             sign_counter[selection_method][paradigm][area][embedding][concreteness] += 1 
#                                 mean_accuracy_collection = np.mean(collect_total_abstract_concrete, axis = 0)
#                                 
#                                 # same for everyone, I picked a random subject
#                                 for concreteness in accuracy_file[embedding]["M01"]["IFG"].keys():
#                                     mean_accuracies[selection_method][paradigm][area][embedding][concreteness] = str(round(mean_accuracy_collection[list(accuracy_file[embedding]["M01"]["IFG"].keys()).index(concreteness)] * 100,1)) + "%"
#                             
#                             # calculate mean of random distributions to put in the table 
#                             else:
#                                 random = {}
#                                 
#                                 # same for everyone, I picked a random subject
#                                 for concreteness in accuracy_file[embedding]["M01"]["IFG"].keys():
#                                     random[concreteness] = []
#                                     
#                                 # collect all average random distributions
#                                 for subject_id in accuracy_file[embedding].keys():
#                                     for concreteness in accuracy_file[embedding]["M01"]["IFG"].keys():
#                                         random[concreteness].append(mean(accuracy_file[embedding][subject_id][area][concreteness]))
# 
#                                                            
#                                 # same for everyone, I picked a random subject
#                                 for concreteness in accuracy_file[embedding]["M01"]["IFG"].keys():
#                                     
#                                     # average over participants
#                                     mean_accuracies[selection_method][paradigm][area][embedding][concreteness] = str(round(mean(random[concreteness])*100,1)) + "%"
#                                 
#             else:
#                 mean_accuracies[selection_method] = {}
#                 p_value[selection_method] = {}
#                 sign_counter[selection_method] = {}
#                 for paradigm in [1,2,3]:
#                     self.paradigm = paradigm
#                     mean_accuracies[selection_method][paradigm] = {}
#                     p_value[selection_method][paradigm] = {}
#                     sign_counter[selection_method][paradigm] = {}
#                     accuracy_file = self.load_accuracies(selection_method)
#                     for embedding in accuracy_file.keys():
#                         mean_accuracies[selection_method][paradigm][embedding] = {}
#                         if not embedding == "random":
#                             collect_total_abstract_concrete = []
#                             sign_counter[selection_method][paradigm][embedding] = {}
#                             p_value[selection_method][paradigm][embedding] = {}
#                             
#                             # same for all participants
#                             for concreteness in accuracy_file[embedding]["M01"].keys():
#                                 sign_counter[selection_method][paradigm][embedding][concreteness] = 0
# 
#                             for subject_id in accuracy_file[embedding].keys():
#                                 collect_total_abstract_concrete.append(np.array(list(accuracy_file[embedding][subject_id].values())))
#                                 p_value[selection_method][paradigm][embedding][subject_id] = {}
#                                 for concreteness in accuracy_file[embedding][subject_id].keys():
#                                     pvalue = self.calculate_p_value(accuracy_file[embedding][subject_id][concreteness], accuracy_file["random"][subject_id][concreteness])
#                                     p_value[selection_method][paradigm][embedding][subject_id][concreteness] = pvalue
#                                     if pvalue < 0.05:
#                                         sign_counter[selection_method][paradigm][embedding][concreteness] += 1    
# 
#                             mean_accuracy_collection = np.mean(collect_total_abstract_concrete, axis = 0)
#                             
#                             # same for everyone, I picked a random subject
#                             for concreteness in accuracy_file[embedding]["M01"].keys():
#                                 mean_accuracies[selection_method][paradigm][embedding][concreteness] = str(round(mean_accuracy_collection[list(accuracy_file[embedding]["M01"].keys()).index(concreteness)]*100,1)) + "%"
#                         else:
#                             random = {}
#                             
#                             # same for everyone, I picked a random subject
#                             for concreteness in accuracy_file[embedding]["M01"].keys():
#                                 random[concreteness] = []
#                                 
#                             # collect all average random distributions
#                             for subject_id in accuracy_file[embedding].keys():
#                                 for concreteness in accuracy_file[embedding]["M01"].keys():
#                                     random[concreteness].append(mean(accuracy_file[embedding][subject_id][concreteness]))
# 
#                                                        
#                             # same for everyone, I picked a random subject
#                             for concreteness in accuracy_file[embedding]["M01"].keys():
#                                 
#                                 # average over participants
#                                 mean_accuracies[selection_method][paradigm][embedding][concreteness] = str(round(mean(random[concreteness])* 100,1)) + "%"
#                                     
#         
#         self.save_p_value(p_value)                    
#         table.generate_table(mean_accuracies, sign_counter)
#         
# =============================================================================
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
    

    def create_map(self, selection_method, paradigm):
        self.paradigm = paradigm 
        
        # I ran all embeddings and subject separately,, so this is necessary, comment out if this is not necessary
        self.rearrange_pickles()
        
        if selection_method == "stable" or selection_method == "roi":
            print("no map for stable or ROI, try barplots")
        
        else:            
            accuracy_file = self.load_accuracies(selection_method)
            areas = {}
            for concreteness in ["total", "abstract", "concrete"]:
                areas[concreteness] = {}
                for embedding in accuracy_file.keys():
                    if embedding == "random":
                        continue
                    areas[concreteness][embedding] = {}
            
            for subject_id in ["M02"]: #TODO [file for file in listdir(self.data_dir)]: 
                print(subject_id)
                paradigms = ["sentences", "pictures", "wordclouds", "average"]
                paradigm_index = paradigm - 1
                datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_" + paradigms[paradigm_index] + ".mat")
                area_number = datafile["meta"]["roiMultimask"][0][0][0][0]
                
                areas_subject = {}
                for concreteness in ["total"]: # TODO:  "abstract", "concrete"]:
                    areas_subject[concreteness] = {}
                    for embedding in ["linguistic", "non-linguistic"]: # TODO accuracy_file.keys():
                        if embedding == "random":
                            continue
                        areas_subject[concreteness][embedding] = {}
                        
                        random_score = accuracy_file["random"][subject_id][concreteness]
    
                        for voxel_index in range(10000): # TODO: len(accuracy_file[embedding][subject_id][concreteness])):
                            if voxel_index % 1000 ==  0:
                                print(str(voxel_index) + "/" + str(len(accuracy_file[embedding][subject_id][concreteness])))
                            accuracy_score = accuracy_file[embedding][subject_id][concreteness][voxel_index]
                        
                            p_value = self.calculate_p_value(accuracy_score, random_score)

                            if p_value < 0.05:
                                if area_number[voxel_index][0] == 0:
                                    continue
                                else:
                                    area_index = area_number[voxel_index][0] - 1
                                    area = datafile["meta"]["rois"][0][0][0][0][area_index][0][0]
                                    if not area in areas_subject:
                                        areas_subject[concreteness][embedding][area] = {}
                                        areas_subject[concreteness][embedding][area]["score"] = 1
                                        areas_subject[concreteness][embedding][area]["index"] = area_index
                                    else:
                                        areas_subject[concreteness][embedding][area]["score"] = areas_subject[concreteness][embedding][area]["score"] + 1
                
                for concreteness in ["total"]: # TODO: , "abstract", "concrete"]: 
                    for embedding in ["linguistic", "non-linguistic"]: #TODO accuracy_file.keys():
                        if embedding == "random":
                            continue
                        for area in areas_subject[concreteness][embedding]:
                            total_voxels_in_area = datafile["meta"]["roiColumns"][0][0][0][0][areas_subject[concreteness][embedding][area]["index"]][0]
                            proportion = areas_subject[concreteness][embedding][area]["score"] / len(total_voxels_in_area)
                    
                            if area in areas[concreteness][embedding]:
                                # len of areas[area] is how many participants have it
                                areas[concreteness][embedding][area].append(proportion)
                            else:
                                areas[concreteness][embedding][area] = [proportion]
            
            for concreteness in ["total"]: # TODO:, "abstract", "concrete"]:
                for embedding in ["linguistic", "non-linguistic"]: # TODO accuracy_file.keys():
                    if embedding == "random":
                        continue
                    for area in areas[concreteness][embedding]:
                        average_proportion = sum(areas[concreteness][embedding][area]) / 1 # TODO len(accuracy_file["random"].keys())
                        areas[concreteness][embedding][area] = average_proportion

            
            for concreteness in ["total"]: # TODO: , "abstract", "concrete"]:
                self.map_brain(areas[concreteness], concreteness)
                
                
    def map_brain(self, average_proportions, concreteness, paradigm):
        dataset = datasets.fetch_atlas_aal()
        aal_img = nib.load(dataset["maps"])
        aal_data = aal_img.get_data()
        
        aal_new = {}
        aal_new["rank"] = {}
        aal_new["proportions"] = {}
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = paradigm - 1
        
        for embedding in ["linguistic", "non-linguistic", "combi"]: 
            new_data_rank = np.zeros(aal_img.shape, dtype='>i2')
            new_data_proportion = np.zeros(aal_img.shape, dtype='>i2')
            for x in range(len(aal_data)):
                print(x)
                for y in range(len(aal_data[x])):
                    for z in range(len(aal_data[x][y])):
                        if str(aal_data[x][y][z]) in dataset.indices:
            
                            # get the index in indices and look which area it is in labels
                            roi = dataset.labels[dataset.indices.index(str(aal_data[x][y][z]))]
                            if roi in average_proportions[concreteness][embedding].keys():
                                new_rank_value = average_proportions[concreteness][embedding][roi]["rank"]["mean"]
                                new_proportion_value = average_proportions[concreteness][embedding][roi]["percentage_of_max"]["mean"]
                            else:
                                new_rank_value = 0
                                new_proportion_value = 0
                            new_data_rank[x][y][z] = new_rank_value
                            new_data_proportion[x][y][z] = new_proportion_value
            
            aal_new["rank"][embedding] = nib.Nifti1Image(new_data_rank, aal_img.affine)
            aal_new["proportions"][embedding] = nib.Nifti1Image(new_data_proportion, aal_img.affine)
            
            nib.save(aal_new["rank"][embedding], self.save_dir + "encoding_searchlight_rank_" + paradigms[paradigm_index] + "_" + concreteness + "_" + embedding + ".nii")
            nib.save(aal_new["proportions"][embedding], self.save_dir + "encoding_searchlight_rank_" + paradigms[paradigm_index] + "_" + concreteness + "_" + embedding + ".nii")

        # draw contours on top of each other

        
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        
        display = plotting.plot_anat()
        
        for embedding in ["linguistic", "non-linguistic", "combi"]: 
            img = nib.load("/home/evihendrikx/Documents/brain-lang-master/pereira_code/" + "result_analyses/final_results/" + "encoding_searchlight_rank_" + paradigms[paradigm_index] + "_" + concreteness + "_" + embedding + ".nii")
            if embedding == "linguistic":
                color = cm.get_cmap('Purples')
            elif embedding == "non-linguistic":
                color = cm.get_cmap('Blues')
            else:
                color = cm.get_cmap('Greens')
            
            # Do the colours represent the values?!
            display.add_contours(img, cmap=color, colorbar=True, vmin = 1, vmax = 58, output_file = "/home/evihendrikx/Documents/brain-lang-master/pereira_code/" + "result_analyses/final_results/" + "encoding_brain_map_" + paradigms[paradigm_index] + "_" + concreteness +".png") #TODO: aal_new[embedding],, self.paradigm - 1,, self.save_dir
            
        
        plotting.show()
        

        
    def rearrange_pickles(self):
        
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        my_dict_final = {}
        for embedding in ["linguistic", "non-linguistic", "combi", "random"]:
            my_dict_final[embedding] = {}
            for subject_id in [file for file in listdir(self.data_dir)]:
                file = "/media/evihendrikx/PACKARDBELL1/stage_evi/encoding_searchlight/encoding_" + paradigms[self.paradigm - 1] + "_searchlight_" + embedding + "_" + subject_id + ".pickle" 
                with open(file, 'rb') as f:
                    my_dict_final[embedding].update(pickle.load(f)[embedding])
        
        accuracy_file = self.accuracy_dir + "encoding_" + paradigms[self.paradigm - 1] + "_searchlight.pickle"
        os.makedirs(os.path.dirname(accuracy_file), exist_ok=True)
        with open(accuracy_file, 'wb') as handle:
            pickle.dump(my_dict_final, handle)
        print("Accuracies stored in file: " + accuracy_file)
        
        
        
        
        
        
    def create_bar_plots(self, paradigm):
        
        self.paradigm = paradigm
        paradigms = ["sentences","pictures","wordclouds","average"]
        paradigm_index = self.paradigm - 1
        
        # make bar plots
        for concreteness in ["total", "abstract", "concrete"]: 
            x_labels = []
            x_pos_emb_area = {}
            start_bars = 0
            bandwidth = 0.3
            x_pos_emb_area["linguistic"] = [start_bars]
            x_pos_emb_area["non-linguistic"] = [start_bars + bandwidth+ 0.1]
            x_pos_emb_area["combi"] = [start_bars + (bandwidth + 0.1) * 2]
            
            mean_accuracies = {}
            
            # make sure every participant has his/ her own color
            colors = {}
            colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
            colorst = [colormap(i) for i in np.linspace(0, 0.9,16)] 
            subject_ids = [file for file in listdir("/media/evihendrikx/PACKARDBELL/stage_evi/pereira_data/")]
            legend_elements = []
            for counter, subject_id in enumerate(subject_ids):
                colors[subject_id] = colorst[counter]
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor = colors[subject_id], label=subject_id))

        
            for selection_method in ["stable", "roi"]:
                accuracy_file = self.load_accuracies(selection_method)    
                
                if selection_method == "stable":      
                    x_labels.append("stable")
                    
                    for embedding in accuracy_file.keys():
                        if not embedding == "random":
                            if not embedding in mean_accuracies.keys():
                                mean_accuracies[embedding] = []
                            all_accuracies = []
                            x_pos_marker = x_pos_emb_area[embedding][-1] - 0.08
                            
                            # keep track of significance 
                            count_sign = 0
                            
                            for subject_id in accuracy_file[embedding].keys():    
                                random_scores = accuracy_file["random"][subject_id][concreteness]
                                accuracy_score = accuracy_file[embedding][subject_id][concreteness]
                                all_accuracies.append(accuracy_score)
                                
                                p_value = self.calculate_p_value(accuracy_score, random_scores)
                                if p_value < 0.05:
                                    plt.plot(x_pos_emb_area[embedding][-1], accuracy_score, color = colors[subject_id], marker = 'o')
                                    count_sign += 1
                                else:
                                    plt.plot(x_pos_emb_area[embedding][-1], accuracy_score, color = colors[subject_id], marker = 'o', markerfacecolor='w')
                                    
                                x_pos_marker = x_pos_marker + 0.08
                                if x_pos_marker > x_pos_emb_area[embedding][-1] + 0.08:
                                    x_pos_marker = x_pos_emb_area[embedding][-1] - 0.08
                                
                            x_pos_emb_area[embedding].append(x_pos_emb_area[embedding][-1] + 1.5)
                            
    
                            mean_accuracies[embedding].append(np.mean(all_accuracies))
                            print("Mean accuracy stable, embedding " + embedding + " : " + str(np.mean(all_accuracies)) + ", significant participants: " + str(count_sign))
                       
                    
                else:
                    accuracy_file = self.load_accuracies(selection_method)
                
                    for embedding in accuracy_file.keys():
                        if not embedding == "random":
                            if not embedding in mean_accuracies.keys():
                                mean_accuracies[embedding] = []
                        
                            # all subjects have the same areas, arbitrarily chose M01
                            for area in accuracy_file[embedding]["M01"].keys():
                                x_pos_marker = x_pos_emb_area[embedding][-1] - 0.08 
                                count_sign = 0
                                
                                if area == "precuneus":
                                    x_labels.append("prec")
                                elif area == "post_cing":
                                    x_labels.append("post. cing.")
                                elif area == "paraHIP":
                                    x_labels.append("parahip.")
                                else:
                                    x_labels.append(area)
            
                                all_accuracies = []
                                for subject_id in accuracy_file[embedding].keys():
                                    random_scores = accuracy_file["random"][subject_id][area][concreteness]
                                    accuracy_score = accuracy_file[embedding][subject_id][area][concreteness]
                                    all_accuracies.append(accuracy_score)
                                    
                                    p_value = self.calculate_p_value(accuracy_score, random_scores)
                                    if p_value < 0.05:
                                        plt.plot(x_pos_emb_area[embedding][-1], accuracy_score, color = colors[subject_id], marker = 'o') 
                                        count_sign += 1
            
                                    else:
                                        plt.plot(x_pos_emb_area[embedding][-1], accuracy_score, color = colors[subject_id], marker = 'o', markerfacecolor='w') 
                                    
                                    x_pos_marker = x_pos_marker + 0.08
                                    if x_pos_marker > x_pos_emb_area[embedding][-1] + 0.08:
                                        x_pos_marker = x_pos_emb_area[embedding][-1] - 0.08
                                        
                                mean_accuracies[embedding].append(np.mean(all_accuracies))
                                x_pos_emb_area[embedding].append(x_pos_emb_area[embedding][-1] + 1.5)
                                print("Mean accuracy " + area + ": " + str(np.mean(all_accuracies)) + ", significant participants: " + str(count_sign))
                
            
            plt.bar(x_pos_emb_area["linguistic"][:-1], mean_accuracies["linguistic"], color = 'white', edgecolor = 'black', label = "linguistic", width = bandwidth, linewidth = 2)
            plt.bar(x_pos_emb_area["non-linguistic"][:-1], mean_accuracies["non-linguistic"], color = 'white', edgecolor = 'red', label = "non-linguistic", width = bandwidth, linewidth = 2)
            plt.bar(x_pos_emb_area["combi"][:-1], mean_accuracies["combi"], color = 'white', edgecolor = 'blue', label = "combi", width = bandwidth, linewidth = 2)

            plt.xticks(x_pos_emb_area["non-linguistic"][:-1], x_labels)
            if paradigms[paradigm_index] == "sentences":
                title = concreteness + " - sentence"
            elif paradigms[paradigm_index] == "pictures":
                title = concreteness + " - picture"
            elif paradigms[paradigm_index] == "wordclouds":
                title = concreteness + " - word cloud"
                
            plt.title(title)
            first_legend = plt.legend()
            ax = plt.gca().add_artist(first_legend)
            
            if concreteness == "concrete":
                plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.ylim([0,1.1])
            if concreteness == "total":
                plt.ylabel("accuracy")
            plt.xlabel("voxel selection")
            plt.savefig(self.save_dir + self.analysis_method + "_barplot_roi_stable_" + paradigms[paradigm_index] + "_" + concreteness + ".png")
            plt.show()
    
        
        
    def create_map_ranks(self, selection_method, paradigm):
        
        # TODO: if ROI is significant: draw average accuracies
                 
        if selection_method == "stable" or selection_method == "roi":
            print("no map for stable or ROI, try individual_ or group_analysis")
            
        else:
            
            paradigms = ["sentences", "pictures", "wordclouds", "average"]
            paradigm_index = paradigm - 1
            self.paradigm = paradigm    
            self.rearrange_pickles()                
            accuracy_file = self.load_accuracies(selection_method)
            print("accuracy file loaded")
            
            self.paradigm = paradigms[paradigm_index]
            areas = {}
            for embedding in accuracy_file.keys():
                print(embedding)
                if embedding == "random":
                    continue
                areas[embedding] = {}
                for subject_id in self.subject_ids:
                    datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_" + self.paradigm + ".mat")
                    area_number = datafile["meta"]["roiMultimask"][0][0][0][0]
                    
                    areas_subject = {}
                       
                    for voxel_index in range(len(accuracy_file[embedding][subject_id]["total"])):
                        if area_number[voxel_index][0] == 0:
                            continue
                        area_index = area_number[voxel_index][0] - 1
                        area = datafile["meta"]["rois"][0][0][0][0][area_index][0][0]
                        
                        abstract_score = accuracy_file[embedding][subject_id]["abstract"][voxel_index]
                        concrete_score = accuracy_file[embedding][subject_id]["concrete"][voxel_index]
                        if not area in areas_subject:
                            areas_subject[area] = {}
                            areas_subject[area]["abstract"] = {}
                            areas_subject[area]["abstract"]["all"] = []
                            areas_subject[area]["abstract"]["all"].append(abstract_score)
                            
                            areas_subject[area]["concrete"] = {}
                            areas_subject[area]["concrete"]["all"] = []
                            areas_subject[area]["concrete"]["all"].append(concrete_score)
                        else:
                            areas_subject[area]["abstract"]["all"].append(abstract_score)
                            areas_subject[area]["concrete"]["all"].append(concrete_score)
                    
                    for area in areas_subject:
                        areas_subject[area]["abstract"]["mean"] = mean(areas_subject[area]["abstract"]["all"])
                        areas_subject[area]["concrete"]["mean"] = mean(areas_subject[area]["concrete"]["all"])
                        
                    areas[embedding][subject_id] = areas_subject
                
            print(areas.keys())
            print(areas["linguistic"].keys())
            
            self.save_accuracy_searchlight(areas)
            self.analyse_area_accuracies(areas, paradigm)
            
        
    
    
    def analyse_area_accuracies(self, accuracies, paradigm):
        
        paradigms = ["sentences","pictures","wordclouds","average"]
        paradigm_index = paradigm - 1
        self.paradigm = paradigms[paradigm_index]
        
        score = {}
        for concreteness in ["abstract", "concrete"]:
            score[concreteness] = {}
            for embedding in accuracies.keys():
                score[concreteness][embedding] = {}
                for area in accuracies["linguistic"]["M01"].keys():
                    score[concreteness][embedding][area] = {}
                    score[concreteness][embedding][area]["rank"] = {}
                    score[concreteness][embedding][area]["percentage_of_max"] = {}

          
       
        for embedding in accuracies.keys():
            for subject_id in accuracies[embedding].keys(): 
                for concreteness in ["abstract", "concrete"]: 
                    mean_accuracies = [accuracies[embedding][subject_id][area][concreteness]["mean"] for area in accuracies[embedding][subject_id].keys()]
                    max_accuracy = max(mean_accuracies)
                    
                    ordered = sorted(mean_accuracies, reverse=True)
                                  
                    
                    for area in accuracies[embedding][subject_id].keys():
                        score[concreteness][embedding][area]["percentage_of_max"][subject_id] = accuracies[embedding][subject_id][area][concreteness]["mean"] / max_accuracy * 100
                        accuracy_area = accuracies[embedding][subject_id][area][concreteness]["mean"]
                        index_acc_area = ordered.index(accuracy_area)
                        score[concreteness][embedding][area]["rank"][subject_id] = index_acc_area + 1
                        
        for concreteness in score:
            for embedding in score[concreteness]:
                for area in score[concreteness][embedding]:
                    mean_score = mean(score[concreteness][embedding][area]["percentage_of_max"].values())
                    dv_score = stdev(score[concreteness][embedding][area]["percentage_of_max"].values())
                    score[concreteness][embedding][area]["percentage_of_max"]["mean"] = mean_score
                    score[concreteness][embedding][area]["percentage_of_max"]["sd"] = dv_score
                    mean_rank = mean(score[concreteness][embedding][area]["rank"].values())
                    dv_rank = stdev(score[concreteness][embedding][area]["rank"].values())
                    score[concreteness][embedding][area]["rank"]["mean"] = mean_rank
                    score[concreteness][embedding][area]["rank"]["sd"] = dv_rank
                
        self.save_ranking_areas(score)
        
        Top10_rank = {}
        Top10_mean = {}
        for embedding in score["abstract"].keys():
            Top10_rank[embedding] = {}
            Top10_mean[embedding] = {}
            for concreteness in score.keys():
                
                mean_per_area = {}
                rank_per_area = {}
                for area in score[concreteness][embedding].keys():
                    

                    mean_per_area[area] = score[concreteness][embedding][area]["percentage_of_max"]["mean"]
                    rank_per_area[area] = score[concreteness][embedding][area]["rank"]["mean"]
        
                print("TOP 10 " + concreteness + ", " + embedding + " embeddings: ")
                top_mean = nlargest(10, mean_per_area, key = mean_per_area.get)
                top_rank = nsmallest(10, rank_per_area, key = rank_per_area.get)
                
                Top10_rank[embedding][concreteness] = top_rank
                Top10_rank[embedding][concreteness] = top_mean
        
                print("Ranks: " + str(top_rank))
                print("Proportions: " + str(top_mean))
            
            
        
        return score
# =============================================================================
#         max_acc = max([score[area]["percentage_of_max"]["mean"] for area in score.keys()])
#         print(max_acc)
#         print([area for area in score.keys() if score[area]["percentage_of_max"]["mean"] == max_acc])
#         
#         min_rank = min([score[area]["rank"]["mean"] for area in score.keys()])
#         print(min_rank)
#         print([area for area in score.keys() if score[area]["rank"]["mean"] == min_rank])        
#             
# =============================================================================
        
               
            
            