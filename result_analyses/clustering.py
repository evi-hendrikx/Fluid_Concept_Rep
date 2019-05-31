from result_analyses.result_analyses_abstract import ResultAnalysis
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pickle
from nilearn import datasets, plotting
import nibabel as nib
from matplotlib import cm
from os import listdir
from matplotlib.lines import Line2D

class ClusteringResults(ResultAnalysis):
    
    def __init__(self, user_dir):
         super(ClusteringResults, self).__init__(user_dir)
         self.analysis_method = "clustering"
    
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

    def create_bar_plots(self, paradigm):

        # TODO: adapt to ROI selection 
        
        self.paradigm = paradigm
        paradigms = ["sentences","pictures","wordclouds","average"]
        paradigm_index = self.paradigm - 1
        
        x_labels = []
        x_position = 0
        mean_accuracies = []
        colors = {}
        
        colormap = plt.cm.gist_ncar 
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
                all_accuracies = []
                
                x_position_area = x_position
                x_pos_marker = x_position_area - 0.02
                count_sign = 0
                for subject_id in accuracy_file.keys():    
                    
                    
                    random_scores = accuracy_file[subject_id]["total"]["random"]
                    accuracy_score = accuracy_file[subject_id]["total"]["accuracy"]
                    all_accuracies.append(accuracy_score)
                    
                    p_value = self.calculate_p_value(accuracy_score, random_scores)
                    
                    if p_value < 0.05:
                        plt.plot(x_pos_marker, accuracy_score, color = colors[subject_id], marker = 'o')
                        count_sign += 1
                    else:
                        plt.plot(x_pos_marker, accuracy_score, color = colors[subject_id], marker = 'o', markerfacecolor='w')
                        
                    x_pos_marker = x_pos_marker + 0.1
                    if x_pos_marker > x_position_area + 0.2:
                        x_pos_marker = x_position_area - 0.2
                        
                mean_accuracies.append(np.mean(all_accuracies))
                x_position += 1
                print("Mean accuracy stable: " + str(np.mean(all_accuracies)) + ", significant participants: " + str(count_sign))
               
                
            else:
                accuracy_file = self.load_accuracies(selection_method)
            
                # all subjects have the same areas, arbitrarily chose M01
                for area in accuracy_file["M01"].keys():
                    x_position_area = x_position
                    x_pos_marker = x_position_area - 0.2
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
                    for subject_id in accuracy_file.keys():
                        random_scores = accuracy_file[subject_id][area]["total"]["random"]
                        accuracy_score = accuracy_file[subject_id][area]["total"]["accuracy"]
                        all_accuracies.append(accuracy_score)
                        
                        p_value = self.calculate_p_value(accuracy_score, random_scores)
                        if p_value < 0.05:
                            plt.plot(x_pos_marker, accuracy_score, color = colors[subject_id], marker = 'o') 
                            count_sign += 1

                        else:
                            plt.plot(x_pos_marker, accuracy_score, color = colors[subject_id], marker = 'o', markerfacecolor='w') 
                        
                        x_pos_marker = x_pos_marker + 0.1
                        if x_pos_marker > x_position_area + 0.2:
                            x_pos_marker = x_position_area - 0.2
                            
                    mean_accuracies.append(np.mean(all_accuracies))
                    x_position += 1
                    print("Mean accuracy " + area + ": " + str(np.mean(all_accuracies)) + ", significant participants: " + str(count_sign))
               
        
        plt.bar(range(x_position), mean_accuracies, color = 'white', edgecolor = 'black') # TODO: grid is off
        plt.xticks(range(x_position), x_labels)
        if paradigms[paradigm_index] == "sentences":
            title = "sentence paradigm"
        elif paradigms[paradigm_index] == "pictures":
            title = "picture paradigm"
        elif paradigms[paradigm_index] == "wordclouds":
            title = "word cloud paradigm"
            
        plt.title(title)
        if self.paradigm == 3:
            plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim([-0.01,0.25])
        if self.paradigm == 1:
            plt.ylabel("accuracy")
        plt.xlabel("voxel selection")
        plt.savefig(self.save_dir + self.analysis_method + "_barplot_roi_stable_" + paradigms[paradigm_index] +".png")
        plt.show()


    def create_map(self, paradigm):
        print("create map")
        
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = paradigm - 1
        self.paradigm = paradigms[paradigm_index]
        
        selection_method = "searchlight"

        
        
        areas = self.load_proportions()

        if not len(areas) == len([1,2,3]):
            
            print("load accuracies")
            
            accuracy_file = self.load_accuracies(selection_method)
            print(accuracy_file)
                
            for subject_id in accuracy_file.keys():
                print(subject_id)
                datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_" + self.paradigm + ".mat")
                area_number = datafile["meta"]["roiMultimask"][0][0][0][0]
                    
                random_scores = accuracy_file[subject_id]["total"]["random"]
    
                areas_subject = {}
                for voxel_index in range(len(accuracy_file[subject_id]["total"]["accuracy"])):
                    accuracy_score = accuracy_file[subject_id]["total"]["accuracy"][voxel_index]
                    p_value = self.calculate_p_value(accuracy_score, random_scores)
                    
                    if p_value < 0.0000094844: 
                        if area_number[voxel_index][0] == 0:
                            continue
                        else:
                            area_index = area_number[voxel_index][0] - 1
                            area = datafile["meta"]["rois"][0][0][0][0][area_index][0][0]
                            if not area in areas_subject:
                                areas_subject[area] = {}
                                areas_subject[area]["score"] = 1
                                areas_subject[area]["index"] = area_index
                            else:
                                areas_subject[area]["score"] = areas_subject[area]["score"] + 1
                                
                for area in areas_subject:
                    total_voxels_in_area = datafile["meta"]["roiColumns"][0][0][0][0][areas_subject[area]["index"]][0]
                    proportion = areas_subject[area]["score"] / len(total_voxels_in_area)
                    
                    if area in areas:
                        # len of areas[area] is how many participants have it
                        areas[area].append(proportion)
                    else:
                        areas[area] = [proportion]
                            
            self.save_proportions(areas)
                
        for area in areas.keys():
            average_proportion = sum(areas[area]) / len([file for file in listdir(self.data_dir)])
            areas[area] = average_proportion
                
        print(areas)
        self.map_brain(areas, selection_method)
        
    
    def map_brain(self, average_proportions, selection_method):
                
        dataset = datasets.fetch_atlas_aal()
        aal_img = nib.load(dataset["maps"])
        aal_data = aal_img.get_data()
        
        new_data = np.zeros(aal_img.shape, dtype='>i2')
        for x in range(len(aal_data)):
            print(x)
            for y in range(len(aal_data[x])):
                for z in range(len(aal_data[x][y])):

                    if str(aal_data[x][y][z]) in dataset.indices:
        
                        # get the index in indices and look which area it is in labels
                        roi = dataset.labels[dataset.indices.index(str(aal_data[x][y][z]))]
                        if roi in average_proportions.keys():
                            new_value = average_proportions[roi] * 100
                        else:
                            new_value = 0
                        new_data[x][y][z] = new_value
                
            
        aal_new = nib.Nifti1Image(new_data, aal_img.affine)
        hot = cm.get_cmap('hot_r')
        vmin = 0
        vmax = 55
        
        plotting.plot_roi(aal_new, cmap=hot, colorbar=True, vmin = vmin, vmax = vmax, output_file = self.save_dir + self.analysis_method + "_brain_map_" + selection_method + "_" + self.paradigm +".png")
        plotting.show()
