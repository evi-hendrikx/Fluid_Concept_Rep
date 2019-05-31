from analyses.classification import ClassifyStable, ClassifyROI, ClassifySearchlight
from analyses.encoding import EncodingStable, EncodingROI, EncodingSearchlight
from analyses.RSA import RSA_ROI_Stable, RSA_Searchlight
from analyses.clustering import ClusterROI, ClusterStable, ClusterSearchlight
from os import path, makedirs, listdir
import pickle

class ResultAnalysis(object):
    def __init__(self, user_dir):
        self.user_dir = user_dir
        self.data_dir = "/media/evihendrikx/PACKARDBELL1/stage_evi/pereira_data/"
        self.accuracy_dir = self.user_dir + "pereira_code/" + "analyses/already_analysed/"
        self.save_dir = self.user_dir + "pereira_code/" + "result_analyses/final_results/" 
        self.subject_ids = [file for file in listdir(self.data_dir)]
        
    def load_accuracies(self, selection_method):
        if self.analysis_method == "classification":
            if selection_method == "stable":
                classification = ClassifyStable(self.user_dir, self.paradigm)
            elif selection_method == "roi":
                classification = ClassifyROI(self.user_dir, self.paradigm)
            elif selection_method == "searchlight":
                print("load accuracies searchlight")
                paradigm = ["sentences","pictures","wordclouds"].index(self.paradigm) + 1
                classification = ClassifySearchlight(self.user_dir, paradigm)
                
            accuracy_file = classification.classify()
            
            
                
        elif self.analysis_method == "encoding":
            if selection_method == "stable":
                encoding = EncodingStable(self.user_dir, self.paradigm)
            elif selection_method == "roi":
                encoding = EncodingROI(self.user_dir, self.paradigm)
            elif selection_method == "searchlight":
                encoding = EncodingSearchlight(self.user_dir, self.paradigm)
                
            accuracy_file = encoding.encode()
            
        elif self.analysis_method == "RSA":
            if selection_method == "roi" or selection_method == "stable":
                print(self.paradigm)
                rsa = RSA_ROI_Stable(self.user_dir, self.paradigm)
            else:
                rsa = RSA_Searchlight(self.user_dir, self.paradigm)
                
            accuracy_file = rsa.run_RSA()
            
        elif self.analysis_method == "clustering":
            if selection_method =="roi":
                clustering = ClusterROI(self.user_dir, self.paradigm)
            elif selection_method == "stable":
                clustering = ClusterStable(self.user_dir, self.paradigm)
            else:
                paradigm = ["sentences","pictures","wordclouds"].index(self.paradigm) + 1
                clustering = ClusterSearchlight(self.user_dir, paradigm)
            accuracy_file = clustering.cluster_fmri()
            
        
        return accuracy_file
    
    def save_proportions(self, proportions):
        file_path = self.save_dir + "area_proportions_" + self.analysis_method + "_" + self.paradigm + ".pickle"          
        makedirs(path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as handle:
            pickle.dump(proportions, handle)
            
    def save_accuracy_searchlight(self, accuracies):
        file_path = self.save_dir + "accuracies_per_area_" + self.analysis_method + "_" + self.paradigm + ".pickle"          
        makedirs(path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as handle:
            pickle.dump(accuracies, handle)
            
    def save_ranking_areas(self, rankings):
        file_path = self.save_dir + "ranking_" + self.analysis_method + "_" + self.paradigm + ".pickle"          
        makedirs(path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as handle:
            pickle.dump(rankings, handle)
    
    def load_ranking_areas(self, rankings):
        file_path = self.save_dir + "ranking_" + self.analysis_method + "_" + self.paradigm + ".pickle"          
        
        if path.isfile(file_path):
            print("Loading accuracies from " + file_path)
            with open(file_path, 'rb') as handle:
                ranking = pickle.load(handle)
            return ranking
        else:
            print("sorry that file does not exist")
            
    def load_proportions(self):
        file_path = self.save_dir + "area_proportions_" + self.analysis_method + "_" + self.paradigm + ".pickle"
        
        if path.isfile(file_path):
            print("Loading accuracies from " + file_path)
            with open(file_path, 'rb') as handle:
                areas = pickle.load(handle)
            return areas
        else:
            return {}
        
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
        
    def save_p_value(self, p_value):
        file_path = self.save_dir + "p-values_encoding.pickle"          
        makedirs(path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as handle:
            pickle.dump(p_value, handle)
    
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

        
  
