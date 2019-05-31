from result_analyses.result_analyses_abstract import ResultAnalysis
from matplotlib import cm
from os import listdir
from matplotlib.lines import Line2D
from pandas import DataFrame, melt, concat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statistics

class RSAResults(ResultAnalysis):
    
    def __init__(self, user_dir):
        super(RSAResults, self).__init__(user_dir)
        self.analysis_method = "RSA"
        
# =============================================================================
#     def create_bar_plots(self, paradigm):
#         self.paradigm = paradigm
#         paradigms = ["sentences", "pictures", "wordclouds", "average"]
#         paradigm_index = paradigm - 1
#             
#         cor_data = self.collect_data_bar()
#         
#         for concreteness in cor_data.keys():
#             mean_cor = {}
#             
#             # create color legend for participants
#             colors = {}
#             colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
#             colorst = [colormap(i) for i in np.linspace(0, 0.9,16)] 
#             subject_ids = [file for file in listdir(self.data_dir)]
#             legend_elements = []
#             for counter, subject_id in enumerate(subject_ids):
#                 colors[subject_id] = colorst[counter]
#                 legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor = colors[subject_id], label=subject_id))
#     
#             
#             # determine position of the bars
#             start_bars = 0
#             bandwidth = 0.3
#             x_pos_emb_area = {}
#             x_pos_emb_area["linguistic"] = [start_bars]
#             x_pos_emb_area["non-linguistic"] = [start_bars + bandwidth+ 0.1]
#             x_pos_emb_area["combi"] = [start_bars + (bandwidth + 0.1) * 2]
#             x_labels = []
#         
#         
#             for embedding in cor_data[concreteness].keys():
#                 mean_cor[embedding] = []
#                 for area in cor_data[concreteness][embedding].keys():
#                     if area == "precuneus":
#                         x_labels.append("prec")
#                     elif area == "post_cing":
#                         x_labels.append("post. cing.")
#                     elif area == "paraHIP":
#                         x_labels.append("parahip.")
#                     else:
#                         x_labels.append(area)
#                     
#                     count_sign = 0
#                     mean_cor_area = []
#                     for participant in cor_data[concreteness][embedding][area].keys():
#                         score = cor_data[concreteness][embedding][area][participant]["cor"]
#                         mean_cor_area.append(score)
#                         if cor_data[concreteness][embedding][area][participant]["p"] < 0.05:
#                             plt.plot(x_pos_emb_area[embedding][-1], score, color = colors[participant], marker = 'o')
#                             count_sign += 1
#                         else:
#                             plt.plot(x_pos_emb_area[embedding][-1], score, color = colors[participant], marker = 'o', markerfacecolor='w')
#                         
#                     mean_cor[embedding].append(statistics.mean(mean_cor_area))
#                     x_pos_emb_area[embedding].append(x_pos_emb_area[embedding][-1] + 1.5)
#                 
#                     print("Mean accuracy area " + area + ", embedding " + embedding + " : " + str(statistics.mean(mean_cor_area)) + ", significant participants: " + str(count_sign))
# 
#             plt.bar(x_pos_emb_area["linguistic"][:-1], mean_cor["linguistic"], color = 'white', edgecolor = 'black', label = "linguistic", width = bandwidth, linewidth = 2)
#             plt.bar(x_pos_emb_area["non-linguistic"][:-1], mean_cor["non-linguistic"], color = 'white', edgecolor = 'red', label = "non-linguistic", width = bandwidth, linewidth = 2)
#             plt.bar(x_pos_emb_area["combi"][:-1], mean_cor["combi"], color = 'white', edgecolor = 'blue', label = "combi", width = bandwidth, linewidth = 2)
#             
#             plt.xticks(x_pos_emb_area["non-linguistic"][:-1], x_labels)
#             if paradigms[paradigm_index] == "sentences":
#                 title = concreteness + " - sentence"
#             elif paradigms[paradigm_index] == "pictures":
#                 title = concreteness + " - picture"
#             elif paradigms[paradigm_index] == "wordclouds":
#                 title = concreteness + " - word cloud"
#                 
#             plt.title(title)
#             first_legend = plt.legend()
#             ax = plt.gca().add_artist(first_legend)
#             
#             plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
#             plt.ylim([-0.05,0.25])
#             plt.ylabel("accuracy")
#             plt.xlabel("voxel selection")
#             plt.savefig(self.save_dir + self.analysis_method + "_barplot_RSA_" + paradigms[paradigm_index] + "_" + concreteness + ".png")
#             plt.show()
# =============================================================================

    def violin_plot_mix(self, sign_part_decoding = True):
        sns.set(style="whitegrid")
        
        for paradigm in [1,2,3]:
            self.paradigm = paradigm
            long_formats =[]
            

                
            new_dict = self.collect_data_violin()

            for embedding in new_dict.keys():
                if embedding == "combi":
                    continue
                
                for area in new_dict[embedding].keys():
                    if area == "precuneus":
                        continue
                    elif area == "paraHIP":
                        continue
                    elif area == "post_cing":
                        continue
                    
                    long_format = DataFrame.from_dict(new_dict[embedding][area], orient='index')
                    
                    if sign_part_decoding == True:
                        signif_part = self.participant_selection_decoding(paradigm)
                        for subject_id in new_dict[embedding][area].keys():
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
                    else:
                        row_embedding = "com"
                    long_format['embedding'] = row_embedding
                    
                    long_format["area"] = area
                    long_formats.append(long_format)
                            
            long_format = concat(long_formats)
            print(long_format)

            
            
            
            
            fig, axs = plt.subplots(ncols=2, nrows =2,  sharex='row', sharey="row", figsize = (10,10))
            plt.subplots_adjust(wspace=0.05)
            
            
            keys = []
            for area in ["IFG","MTG", "FFG", "stable"]:
                keys.append(area)
                
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
                axs[counter1][counter2].set_ylim([-0.12,0.48])
                axs[counter1][counter2].set_title(area, weight='bold')
                plot = sns.violinplot(x = "embedding", y="accuracy", hue="concreteness", hue_order = ["concrete", "abstract"], palette = "muted", data=long_format[long_format.area == area], split=True, inner = None, ax = axs[counter1][counter2], linewidth = 0)

                # indicate the means
                data = long_format[long_format.area == area]
                text = data[data.embedding == "textual"]
                visual = data[data.embedding == "visual"]
                random = data[data.embedding == "random"]
                mean_dict["text"]["concrete"][area] = statistics.mean(text.accuracy[text.concreteness == "concrete"])
                mean_dict["text"]["abstract"][area]  = statistics.mean(text.accuracy[text.concreteness == "abstract"])
                mean_dict["visual"]["concrete"][area]  = statistics.mean(visual.accuracy[visual.concreteness == "concrete"])
                mean_dict["visual"]["abstract"][area] = statistics.mean(visual.accuracy[visual.concreteness == "abstract"])
                mean_dict["random"]["concrete"][area]  = statistics.mean(random.accuracy[random.concreteness == "concrete"])
                mean_dict["random"]["abstract"][area] = statistics.mean(random.accuracy[random.concreteness == "abstract"])
                
                df = pd.DataFrame({
                        'x': [-0.1,0,0.9,1, 1.9,2],
                        'y': [mean_dict[embedding][concreteness][area] for embedding in mean_dict for concreteness in mean_dict[embedding]]
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
                    axs[counter1][counter2].set_ylabel("Spearman correlation", weight = "bold")
                    axs[counter1][counter2].set_yticklabels(["",-0.1,0.0,0.1,0.2,0.3,0.4])                    
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
#     def collect_data_bar(self):
#         
#         dict_cors = {}
#         self.paradigm = 1
#         correlations = self.load_accuracies("stable")
#         print(correlations)
#         
#                     
#         participants = ["M01", "M02","M03","M04","M05","M06","M07","M08","M09","M10", "M13","M14","M15","M16","M17","P01"]
#         areas = ["IFG", "MTG", "post_cing", "precuneus", "FFG", "paraHIP"]
#         embeddings = ["linguistic", "non-linguistic", "combi"]
#         for concreteness in ["total","abstract","concrete"]:
#             dict_cors[concreteness] = {}
#             for embedding in embeddings:
#                 dict_cors[concreteness][embedding] = {}
#                 for area in areas:
#                     dict_cors[concreteness][embedding][area] = {}
#                     for participant in participants:
#                         dict_cors[concreteness][embedding][area][participant] = {}
#                         i = areas.index(area) +  len(areas) * participants.index(participant)
#                         j = (len(areas) + 1) * len(participants) + embeddings.index(embedding)
#                         
#                         dict_cors[concreteness][embedding][area][participant]["cor"] = correlations[concreteness]["correlations"][i][j]
#                         dict_cors[concreteness][embedding][area][participant]["p"] = correlations[concreteness]["p-values"][i][j]
# 
#             for embedding in embeddings:
#                 for area in ["stable"]:
#                     dict_cors[concreteness][embedding][area] = {}
#                     for participant in participants:
#                         dict_cors[concreteness][embedding][area][participant] = {}
#                         i = len(areas) * len(participants) + participants.index(participant)
#                         j = (len(areas) + 1) * len(participants) + embeddings.index(embedding)
#                         dict_cors[concreteness][embedding][area][participant]["cor"] = correlations[concreteness]["correlations"][i][j]
#                         dict_cors[concreteness][embedding][area][participant]["p"] = correlations[concreteness]["p-values"][i][j]
#         return dict_cors
# =============================================================================
                    
    def collect_data_violin(self):
        
        dict_cors = {}
        correlations = self.load_accuracies("stable")
        
        print(correlations.keys())
                    
        participants = ["M01", "M02","M03","M04","M05","M06","M07","M08","M09","M10", "M13","M14","M15","M16","M17","P01"]
        areas = ["IFG", "MTG", "post_cing", "precuneus", "FFG", "paraHIP"]
        embeddings = ["linguistic", "non-linguistic", "combi","random"]
        concretenesses = ["abstract","concrete"]
        for embedding in embeddings:
            dict_cors[embedding] = {}
            for area in areas:
                dict_cors[embedding][area] = {}
                for participant in participants:
                    dict_cors[embedding][area][participant] = {}
                    for concreteness in concretenesses:
                        dict_cors[embedding][area][participant][concreteness] = []
                        i = areas.index(area) +  len(areas) * participants.index(participant)
                        
                        # take stable into account (+1)
                        j = (len(areas) + 1) * len(participants) + embeddings.index(embedding)
                        
                        if embedding == "random":
                            dict_cors[embedding][area][participant][concreteness] = statistics.mean(correlations[concreteness]["correlations"][i][j:])
                        else:
                            dict_cors[embedding][area][participant][concreteness] = correlations[concreteness]["correlations"][i][j]
                        

        for embedding in embeddings:
            for area in ["stable"]:
                dict_cors[embedding][area] = {}
                for participant in participants:
                    dict_cors[embedding][area][participant] = {}
                    for concreteness in concretenesses:
                        dict_cors[embedding][area][participant][concreteness] = []
                        i = len(areas) * len(participants) + participants.index(participant)
                        j = (len(areas) + 1) * len(participants) + embeddings.index(embedding)
                        if embedding == "random":
                            dict_cors[embedding][area][participant][concreteness] = statistics.mean(correlations[concreteness]["correlations"][i][j:])
                        else:
                            dict_cors[embedding][area][participant][concreteness] = correlations[concreteness]["correlations"][i][j]
        return dict_cors
    
    def participant_selection_decoding(self, paradigm):
        
        
        significant_participants = {}
        for selection_method in ["roi","stable"]:
            self.analysis_method = "classification"
            accuracy_file = self.load_accuracies(selection_method)
            self.analysis_method = "RSA"
    
            accuracy_file = self.transform_randoms(accuracy_file, selection_method)
            
            if selection_method == "roi": 
                for area in accuracy_file["accuracy"]["M01"].keys():
                    significant_participants[area] = []
            else:
                significant_participants["stable"] = []
            
            for subject_id in accuracy_file["accuracy"].keys():
                
                if selection_method == "stable":
                    random_scores = accuracy_file["random"][subject_id]
                    accuracy_score = accuracy_file["accuracy"][subject_id]
                    p_value = self.calculate_p_value(accuracy_score, random_scores)
                    if p_value < 0.05:
                        significant_participants["stable"].append(subject_id)                                       
                
                else:
                    for area in accuracy_file["accuracy"][subject_id].keys():
                        random_scores = accuracy_file["random"][subject_id][area]
                        accuracy_score = accuracy_file["accuracy"][subject_id][area]
                        p_value = self.calculate_p_value(accuracy_score, random_scores)
                        if p_value < 0.05:
                            significant_participants[area].append(subject_id)
                
        return significant_participants
# =============================================================================
#                     
#                     for area in ["IFG", "MTG", "post_cing", "precuneus", "FFG", "paraHIP"]:
#                 
#             RSA = RSA_ROI_Stable(user_dir, paradigm) 
#             paradigms = ["sentences","pictures","wordclouds"]
#             file_path = "/home/evihendrikx/Documents/brain-lang-master/pereira_code/analyses/already_analysed/RSA_" + paradigms[paradigm - 1] + "_stable.pickle"
#             
#             print("Loading voxel selections from " + file_path)
#             with open(file_path, 'rb') as handle:
#                 spearman = pickle.load(handle)
#                 
#             keys = spearman[concreteness]["correlations"].shape[1]
#             print(keys)
#             randoms = 1000
#             
#             # make plot with average random correlation instead of distribution
#             average_random_spearman = np.zeros((keys - randoms + 1, keys - randoms + 1))
#             for i in np.arange(keys - randoms + 1):
#                 for j in np.arange(keys - randoms + 1):
#                     if j == keys - randoms and i == keys - randoms:
#                         average_random_spearman[i][j] = 1
#                     elif j == keys - randoms:
#                         average_random_spearman[i][j] = np.mean(spearman[concreteness]["correlations"][i][j:])
#                     elif i == keys - randoms:
#                         average_random_spearman[i][j] = np.mean(spearman[concreteness]["correlations"][j][i:])
#                     else:
#                         average_random_spearman[i][j] = spearman[concreteness]["correlations"][i][j]
#                         
#             labels = []
#             for participant in ["M01", "M02","M03","M04","M05","M06","M07","M08","M09","M10", "M13","M14","M15","M16","M17","P01"]:
#                 for area in :
#                     labels.append(participant + "_" + area)
#             for participant in ["M01", "M02","M03","M04","M05","M06","M07","M08","M09","M10", "M13","M14","M15","M16","M17","P01"]:
#                 labels.append(participant + "_stable")
#             encoder = 
#             for i in range(1000):
#                 encoder.append("RandomEncoder")
#             for embedding in encoder:
#                 labels.append(embedding)
#     
#     
#         RSA.plot(average_random_spearman, spearman[concreteness]["p-values"], labels, title="RDM Comparison Spearman", concreteness = concreteness, cbarlabel="Spearman Correlation", between_matrices = True)
#         
# =============================================================================
        ################################################
# =============================================================================
#                 if not ax:
#             ax = plt.gca()
#     
#         # Plot the heatmap
#         im = ax.imshow(data, **kwargs)
#         ax.set_title(title, pad =50.0)
#         # create an axes on the right side of ax. The width of cax will be 5%
#         # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)
#         cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#     
#         # We want to show all ticks...
#         ax.set_xticks(np.arange(data.shape[1]))
#         ax.set_yticks(np.arange(data.shape[0]))
#         # ... and label them with the respective list entries.
#         ax.set_xticklabels(col_labels)
#         ax.set_yticklabels(row_labels)
#     
#         # Let the horizontal axes labeling appear on top.
#         ax.tick_params(top=True, bottom=False,
#                        labeltop=True, labelbottom=False)
#     
#         # Rotate the tick labels and set their alignment.
#         plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
#                  rotation_mode="anchor")
#     
#         # Turn spines off and create white grid.
#         # for edge, spine in ax.spines.items():
#         #    spine.set_visible(False)
#     
#         # what is happening here?
#         ax.set_xticks(np.arange(0, data.shape[1] + 1) - 0.5, minor=True)
#         ax.set_yticks(np.arange(0, data.shape[0] + 1) - 0.5, minor=True)
#         
#         if not p_values == []:
#             for i in range(data.shape[0]):
#                 for j in range(data.shape[1]):
#                     if i == data.shape[0] - 1 or j == data.shape[1] - 1:
#                         continue
#                     else:
#                         if p_values[i][j] < 0.05:
#                             #ax.text(i, j, str(round(data[i][j], 2)) + "*", ha="center", va="center", color="w")
#                             ax.text(i, j, "*", ha="center", va="center", color="w")
# # =============================================================================
# #                         else:
# #                             ax.text(i, j, round(data[i][j], 2), ha="center", va="center", color="w")
# # 
# # =============================================================================
#     
#         return im, cbar
# =============================================================================
