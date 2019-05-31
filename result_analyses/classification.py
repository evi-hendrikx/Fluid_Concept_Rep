from result_analyses.result_analyses_abstract import ResultAnalysis
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pickle
from nilearn import datasets, plotting, image
import nibabel as nib
from matplotlib import cm
from os import listdir, path
from matplotlib.lines import Line2D
from statistics import mean, stdev
from prettytable import PrettyTable
from heapq import nlargest, nsmallest

class ClassificationResults(ResultAnalysis):
    
    def __init__(self, user_dir):
         super(ClassificationResults, self).__init__(user_dir)
         self.analysis_method = "classification"
         
    
    
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

         
    def individual_analysis(self, selection_method, paradigm):
        # TODO: adapt to ROI selection 
        
        self.paradigm = paradigm
        
        if selection_method == "searchlight":
            print("No individual searchlight analysis, try create_map")
        
        else:
            # relevant for plotting results

            paradigms = ["sentences","pictures","wordclouds","average"]
            paradigm_index = self.paradigm - 1
            
            if selection_method == "stable":  
                all_random_scores = []
                x_label = 0
                
                accuracy_file = self.load_accuracies(selection_method)
                accuracy_file = self.transform_randoms(accuracy_file, selection_method)

                for subject_id in accuracy_file["accuracy"].keys():                
                    random_scores = accuracy_file["random"][subject_id]
                    all_random_scores.append(random_scores)
                    
                    # compares accuracy to random distribution
                    x_label = x_label + 1
                    accuracy_score = accuracy_file["accuracy"][subject_id]
                    p_value = self.calculate_p_value(accuracy_score, random_scores)
                    print("accuracy and p-value for participant " + subject_id + " are: " + str(accuracy_score) + " and " + str(p_value))
                    if p_value < 0.05:
                        plt.plot(x_label, accuracy_score, 'P', color = "green")
                    else:
                        plt.plot(x_label, accuracy_score, 'P', color = "red")
               
                # relevant for plotting results
                plt.boxplot(all_random_scores)
                plt.title(selection_method + " " + paradigms[paradigm_index])
                plt.ylim([0,1])
                plt.show()
                
            else:
                accuracy_file = self.load_accuracies(selection_method)
                accuracy_file = self.transform_randoms(accuracy_file, selection_method)
            
                # all subjects have the same areas, arbitrarily chose M01
                for area in accuracy_file["accuracy"]["M01"].keys():
                    all_random_scores = []
                    x_label = 0
                    for subject_id in accuracy_file["accuracy"].keys():    
                        random_scores = accuracy_file["random"][subject_id][area]
                        all_random_scores.append(random_scores)
                        
                        # compares accuracy to random distribution
                        x_label = x_label + 1
                        accuracy_score = accuracy_file["accuracy"][subject_id][area]
                        p_value = self.calculate_p_value(accuracy_score, random_scores)
                        print("accuracy and p-value for participant " + subject_id + " are: " + str(accuracy_score) + " and " + str(p_value))
                        if p_value < 0.05:
                            plt.plot(x_label, accuracy_score, 'P', color = "green")
                        else:
                            plt.plot(x_label, accuracy_score, 'P', color = "red")
               
                    # relevant for plotting results
                    plt.boxplot(all_random_scores)
                    plt.title(selection_method + " " + paradigms[paradigm_index] + " " + area)
                    plt.ylim([0,1])
                    plt.show()
                        

    def group_analysis(self, selection_method, paradigm):
        self.paradigm = paradigm
        paradigm_index = self.paradigm - 1
        paradigms = ["sentences","pictures","wordclouds","average"]
        
        if selection_method == "stable":
            accuracy_file = self.load_accuracies(selection_method)
            accuracy_file = self.transform_randoms(accuracy_file, selection_method)
            
            # pick arbitrary subject, they all had equally many permutations
            mean_distribution = []
            for index in range(len(accuracy_file["random"]["M01"])):
                all_random = []
                for subject_id in accuracy_file["random"].keys():
                    all_random.append(accuracy_file["random"][subject_id][index])
                mean_distribution.append(np.mean(all_random))
            
            plt.hist(mean_distribution, width = 0.05)
            plt.xlim([0, 1])
            plt.ylim([0, 300])
        
            all_accuracies = []
            for subject_id in accuracy_file["accuracy"].keys():
                all_accuracies.append(accuracy_file["accuracy"][subject_id])
            mean_accuracy = np.mean(all_accuracies)
            
            plt.axvline(x=mean_accuracy, color = "red", linestyle = "--")
            plt.title(selection_method + " " + paradigms[paradigm_index])
            plt.show()
            
            p_value = self.calculate_p_value(mean_accuracy, mean_distribution)
            print("The mean accuracy and p-value are: " + str(mean_accuracy) + " and " + str(p_value))
            
        elif selection_method == "roi":
            accuracy_file = self.load_accuracies(selection_method)
            accuracy_file = self.transform_randoms(accuracy_file, selection_method)
            
            # pick arbitrary subject, they all had equally many permutations
            mean_distribution = []
            
            # all have the same areas
            for area in accuracy_file["random"]["M01"].keys():
                mean_distribution = []
                for index in range(len(accuracy_file["random"]["M01"]["IFG"])):
                    all_random = []
                    for subject_id in accuracy_file["random"].keys():
                        all_random.append(accuracy_file["random"][subject_id][area][index])
                    mean_distribution.append(np.mean(all_random))
                
                plt.hist(mean_distribution, width = 0.05)
                plt.xlim([0, 1])
                plt.ylim([0, 300])
        
                all_accuracies = []
                for subject_id in accuracy_file["accuracy"].keys():
                    all_accuracies.append(accuracy_file["accuracy"][subject_id][area])
                mean_accuracy = np.mean(all_accuracies)
                
                plt.axvline(x=mean_accuracy, color = "red", linestyle = "--")
                plt.title(selection_method + " " + paradigms[paradigm_index] + " " + area)
                plt.show()
            
            # p_value = self.calculate_p_value(mean_accuracy, mean_distribution)
            print("The mean accuracy is: " + str(mean_accuracy))
            
        elif selection_method == "searchlight":
            print("no analysis yet on group level")
            
    def create_map(self, selection_method, paradigm):
        
        # TODO: if ROI is significant: draw average accuracies
                 
        if selection_method == "stable" or selection_method == "roi":
            print("no map for stable or ROI, try individual_ or group_analysis")
            
        else:
            paradigms = ["sentences", "pictures", "wordclouds", "average"]
            paradigm_index = paradigm - 1
            self.paradigm = paradigms[paradigm_index]
                     
            accuracy_file = self.load_accuracies(selection_method)
            print("accuracy file loaded")
                
            areas = {}
            for subject_id in self.subject_ids:
                print(subject_id)
                datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_" + self.paradigm + ".mat")
                area_number = datafile["meta"]["roiMultimask"][0][0][0][0]
      
                areas_subject = {}
                for voxel_index in range(len(accuracy_file["accuracy"][subject_id])):
                    if area_number[voxel_index][0] == 0:
                        continue
                    area_index = area_number[voxel_index][0] - 1
                    area = datafile["meta"]["rois"][0][0][0][0][area_index][0][0]
                    accuracy_score = accuracy_file["accuracy"][subject_id][voxel_index]
                    if not area in areas_subject:
                        areas_subject[area] = {}
                        areas_subject[area]["all"] = []
                        areas_subject[area]["all"].append(accuracy_score)
                    else:
                        areas_subject[area]["all"].append(accuracy_score)
                
                for area in areas_subject:
                    areas_subject[area]["mean"] = mean(areas_subject[area]["all"])
                    
                areas[subject_id] = areas_subject
                self.save_accuracy_searchlight(areas)
                
            self.analyse_area_accuracies(areas, paradigm)
                    
            
    def map_brain(self, average_proportions, selection_method, paradigm):
        
        self.paradigm = paradigm
                
        dataset = datasets.fetch_atlas_aal()
        aal_img = nib.load(dataset["maps"])
        aal_data = aal_img.get_data()               
        roi_ids = []
        
        new_data_rank = np.zeros(aal_img.shape, dtype='>i2')
        new_data_proportion = np.zeros(aal_img.shape, dtype='>i2')
        for x in range(len(aal_data)):
            print(x)
            for y in range(len(aal_data[x])):
                for z in range(len(aal_data[x][y])):

                    if str(aal_data[x][y][z]) in dataset.indices:
                        
        
                        # get the index in indices and look which area it is in labels
                        roi = dataset.labels[dataset.indices.index(str(aal_data[x][y][z]))]
                        if roi in average_proportions.keys():
                            
                            if average_proportions[roi]["accuracy"]["mean"] > 0.52:
                                if not roi in roi_ids:
                                    roi_ids.append(roi)
                                print(roi)
                                new_rank_value = average_proportions[roi]["rank"]["mean"]
                                print(new_rank_value)
                                new_proportion_value = average_proportions[roi]["percentage_of_max"]["mean"]
                            else:
                                new_rank_value = 0
                                new_proportion_value = 0
                        else:
                            new_rank_value = 0
                            new_proportion_value = 0
                        new_data_rank[x][y][z] = new_rank_value
                        new_data_proportion[x][y][z] = new_proportion_value
                     
        print(roi_ids)
        for roi in roi_ids:
            roi_id = dataset.indices[dataset.labels.index(roi)]
            roi_map = image.math_img('img == %s' % roi_id, img=dataset.maps)
            plotting.plot_roi(roi_map, title=roi)
                
        
        aal_new = nib.Nifti1Image(new_data_rank, aal_img.affine)
        hot = cm.get_cmap('hot')        
        plotting.plot_roi(aal_new, cmap=hot, colorbar=True, output_file = self.save_dir + self.analysis_method + "_brain_map_rank_" + selection_method + "_" + self.paradigm +"_>.52.png")
        plotting.show()
        
    def export_legend(self, legend, filename="legend.png"):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(self.save_dir + self.analysis_method + "_barplot_roi_stable_legend.png", dpi="figure", bbox_inches=bbox)


    def create_bar_plots(self, paradigm):
        
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        
        self.paradigm = paradigm
        paradigms = ["sentences","pictures","wordclouds","average"]
        paradigm_index = self.paradigm - 1
        
        x_labels = []
        x_position = 0
        mean_accuracies = []
        colors = {}
        
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
        colorst = [colormap(i) for i in np.linspace(0, 0.9,16)] 
        subject_ids = [file for file in listdir(self.data_dir)]
        
        legend_elements = []
        for counter, subject_id in enumerate(subject_ids):
            colors[subject_id] = colorst[counter]
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor = colors[subject_id], label=subject_id))

        
        for selection_method in ["roi","stable"]:
            accuracy_file = self.load_accuracies(selection_method)
            accuracy_file = self.transform_randoms(accuracy_file, selection_method)
                        
            
            if selection_method == "stable":  
                
                x_labels.append("stable")
                all_accuracies = []
                
                x_position_area = x_position
                x_pos_marker = x_position_area - 0.04
                count_sign = 0
                for subject_id in accuracy_file["accuracy"].keys():    
                    
                    
                    random_scores = accuracy_file["random"][subject_id]
                    accuracy_score = accuracy_file["accuracy"][subject_id]
                    all_accuracies.append(accuracy_score)
                    
                    p_value = self.calculate_p_value(accuracy_score, random_scores)
                    #print("accuracy and p-value for participant " + subject_id + " are: " + str(accuracy_score) + " and " + str(p_value))
                    if p_value < 0.05:
                        plt.plot(x_pos_marker, accuracy_score, color = colors[subject_id], marker = 'o')
                        count_sign += 2
                    else:
                        plt.plot(x_pos_marker, accuracy_score, color = colors[subject_id], marker = 'o', markerfacecolor='w')
                        
                    x_pos_marker = x_pos_marker + 0.2
                    if x_pos_marker > x_position_area + 0.4:
                        x_pos_marker = x_position_area - 0.4
                        
                mean_accuracies.append(np.mean(all_accuracies))
                x_position += 2
                print("Mean accuracy stable: " + str(np.mean(all_accuracies)) + ", significant participants: " + str(count_sign))
               
                
            else:
                accuracy_file = self.load_accuracies(selection_method)
                accuracy_file = self.transform_randoms(accuracy_file, selection_method)
                
                    
            
                # all subjects have the same areas, arbitrarily chose M01
                for area in ["IFG", "MTG", "FFG", "post_cing", "precuneus", "paraHIP"]:
                    x_position_area = x_position
                    x_pos_marker = x_position_area - 0.4
                    count_sign = 0
                    
                    if area == "precuneus":
                        x_labels.append("PCUN")
                    elif area == "post_cing":
                        x_labels.append("PCC")
                    elif area == "paraHIP":
                        x_labels.append("PHG")
                    else:
                        x_labels.append(area)

                    all_accuracies = []
                    for subject_id in accuracy_file["accuracy"].keys():
                        random_scores = accuracy_file["random"][subject_id][area]
                        accuracy_score = accuracy_file["accuracy"][subject_id][area]
                        all_accuracies.append(accuracy_score)
                        
                        p_value = self.calculate_p_value(accuracy_score, random_scores)
                        if p_value < 0.05:
                            plt.plot(x_pos_marker, accuracy_score, color = colors[subject_id], marker = 'o') 
                            count_sign += 1

                        else:
                            plt.plot(x_pos_marker, accuracy_score, color = colors[subject_id], marker = 'o', markerfacecolor='w') 
                        
                        x_pos_marker = x_pos_marker + 0.2
                        if x_pos_marker > x_position_area + 0.4:
                            x_pos_marker = x_position_area - 0.4
                            
                    mean_accuracies.append(np.mean(all_accuracies))
                    x_position += 2
                    print("Mean accuracy " + area + ": " + str(np.mean(all_accuracies)) + ", significant participants: " + str(count_sign))
                    if area == "paraHIP":
                        x_position += 1
               
        
        #plt.bar(range(x_position), mean_accuracies, color = 'white', edgecolor = 'black')
        plt.bar([0,2,4,6,8,10,13], mean_accuracies, color = 'white', edgecolor = 'black', width = 1.6)
        plt.xticks([0,2,4,6,8,10,13], x_labels)
        if paradigms[paradigm_index] == "sentences":
            title = "sentence paradigm"
        elif paradigms[paradigm_index] == "pictures":
            title = "picture paradigm"
        elif paradigms[paradigm_index] == "wordclouds":
            title = "word cloud paradigm"
            
        
        plt.title(title)
        # plt.legend(handles = legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
          
        

        plt.ylim([0,1])
        if self.paradigm == 1:
            plt.ylabel("accuracy")
        plt.xlabel("voxel selection")
        plt.savefig(self.save_dir + self.analysis_method + "_barplot_roi_stable_" + paradigms[paradigm_index] +".png")
        plt.show()
        
        
        if self.paradigm == 3:
            plt.rcParams['axes.edgecolor'] = 'white'
            plt.rcParams['ytick.color'] = 'white'
            legend = plt.legend(handles = legend_elements,loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol = 16,fancybox=True)
            self.export_legend(legend)  
        
        
    def analyse_area_accuracies(self, accuracies, paradigm):
        
        paradigms = ["sentences","pictures","wordclouds","average"]
        paradigm_index = paradigm - 1
        self.paradigm = paradigms[paradigm_index]
        
        score = {}
        for area in accuracies["M01"].keys():
            score[area] = {}
            score[area]["rank"] = {}
            score[area]["percentage_of_max"] = {}
            score[area]["accuracy"] = {}

            
        for subject_id in accuracies.keys():            
            mean_accuracies = [accuracies[subject_id][area]["mean"] for area in accuracies[subject_id].keys()]
            max_accuracy = max(mean_accuracies)
            
            ordered = sorted(mean_accuracies, reverse=True)
            
           
            
            for area in accuracies[subject_id].keys():
                score[area]["percentage_of_max"][subject_id] = accuracies[subject_id][area]["mean"] / max_accuracy * 100
                accuracy_area = accuracies[subject_id][area]["mean"]
                rank_area = ordered.index(accuracy_area) + 1
                score[area]["rank"][subject_id] = rank_area
                score[area]["accuracy"][subject_id] = accuracies[subject_id][area]["mean"]
                
        
        for area in score:
            mean_score = mean(score[area]["percentage_of_max"].values())
            dv_score = stdev(score[area]["percentage_of_max"].values())
            score[area]["percentage_of_max"]["mean"] = mean_score
            score[area]["percentage_of_max"]["sd"] = dv_score
            
            mean_rank = mean(score[area]["rank"].values())
            dv_rank = stdev(score[area]["rank"].values())
            score[area]["rank"]["mean"] = mean_rank
            score[area]["rank"]["sd"] = dv_rank
            
            mean_acc = mean(score[area]["accuracy"].values())
            dv_acc = stdev(score[area]["accuracy"].values())
            score[area]["accuracy"]["mean"] = mean_acc
            score[area]["accuracy"]["sd"] = dv_acc
            
        self.save_ranking_areas(score)
        
        self.map_brain(score, "searchlight")
        
        
        t = PrettyTable(['Area','Mean % of max', 'Mean ranking'])
        for area in score:
            t.add_row([area, str(round(score[area]["percentage_of_max"]["mean"],2))+ " +/- " + str(round(score[area]["percentage_of_max"]["sd"],2)), str(round(score[area]["rank"]["mean"],2)) + " +/- " + str(round(score[area]["rank"]["sd"],2))])
            
        # print(t)
        
        
        mean_per_area = {}
        rank_per_area = {}
        for area in score.keys():
            mean_per_area[area] = score[area]["percentage_of_max"]["mean"]
            rank_per_area[area] = score[area]["rank"]["mean"]
        
        Top10_mean = nlargest(10, mean_per_area, key = mean_per_area.get)
        Top10_rank = nsmallest(10, rank_per_area, key = rank_per_area.get)
        
        print(Top10_rank)
        print(Top10_mean)
        
        return score

