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
