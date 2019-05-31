from select_voxels.selection_methods import SelectROI
from select_voxels.selection_methods import SelectSearchlight
from select_voxels.select_stable import SelectStable
from analyses.classification_abstract import Classification
import scipy.io
import numpy as np

class ClassifyROI(Classification):  
    
    def __init__(self, user_dir, paradigm):
        super(ClassifyROI, self).__init__(user_dir, paradigm)
    
    def classify(self):
        self.selection = "roi"
         
        # see if file already exists
        accuracies = self.load_classifications(self.selection)
        if len(accuracies) == len(["accuracy", "random"]):
            return(accuracies)
            
        # if it doesn't, create a file with random and actual accuracies
        else:
            
            # get indices from voxels in the region of interest
            # Format: {participant 1: [indices for voxels in ROIs] participant 2: [indices for voxels in ROIs] ... participant n: ...}
            roi_selection = SelectROI(self.user_dir)
            voxel_selection = roi_selection.select_voxels()
                        
            accuracies["accuracy"] = {}
            accuracies["random"] = {}
            processed_subject_ids = []
            
            # read data of all participants
            self.read_data()
            
            for subject_id in self.blocks.keys():
                accuracies["accuracy"][subject_id] = {}
                accuracies["random"][subject_id] = {}
                
                for area in voxel_selection[subject_id].keys():
                    print("Going to classify data from: " + subject_id + ", area: " + area)
                    scans = []
                
                    # per word, fetch corresponding activities from the requested voxels
                    for block in self.blocks[subject_id]:
                        selected_voxel_activities = []
                        all_voxels = [event.scan for event in block.scan_events][0]
                        for voxel_index in voxel_selection[subject_id][area]:
                            selected_voxel_activities.append(all_voxels[voxel_index])
                        scans.append(selected_voxel_activities) 
                
                    accuracies["accuracy"][subject_id][area] = self.run_svc(scans)
                    accuracies["random"][subject_id][area] = self.random_generator(scans)
                
                processed_subject_ids.append(subject_id)
                self.save_classifications(self.selection, accuracies, processed_subject_ids, intermediate_save = True)
                
            return accuracies 
        
            
class ClassifySearchlight(Classification):
    def __init__(self, user_dir, paradigm):
        super(ClassifySearchlight, self).__init__(user_dir, paradigm)
    
    def classify(self):
        self.selection = "searchlight"
        self.reduced_random = reduced_random

        # see if file already exists
        accuracies = self.load_classifications(self.selection)
        if len(accuracies) == len(["accuracy"]):
            return(accuracies)
        
        # if it doesn't, create a file with random and actual accuracies
        else:
            # get indices from neighbors for every voxel
            # Format: {participant 1: [indices voxel 1 as center with neighbors]...[indices vox. k as center with neighbors] 
                # participant 2: [indices vox.1 as center with neighbors]...[indices vox. m as center with neighbors] ... participant n:....}
            searchlight_selection = SelectSearchlight(self.user_dir)
            voxel_selection = searchlight_selection.select_voxels()
            
            accuracies_so_far = {}
            accuracies_so_far["accuracy"] = {}
            processed_subject_ids = []
            
            self.read_data()
            for subject_id in self.blocks.keys():
                print("Going to classify data from: " + subject_id)
        
                accuracies_so_far["accuracy"][subject_id] = []
                

                # determine classification accuracy per voxel (with surroundings)
                counter = 1
                for voxel_indices in voxel_selection[subject_id]:
                    print("determine accuracies for voxel: " + str(counter) + "/" + str(len(voxel_selection[subject_id])))  
                    counter += 1

                    # for every word, fetch center and neighbor voxels
                    scans = []
                    for block in self.blocks[subject_id]:
                        selected_voxel_activities = []
                        all_voxels = [event.scan for event in block.scan_events][0]
                        for voxel_index in voxel_indices:
                            selected_voxel_activities.append(all_voxels[voxel_index])
                        scans.append(selected_voxel_activities)

                    accuracy = self.run_svc(scans)
   
                    # collect accuracies for all voxels as center
                    accuracies_so_far["accuracy"][subject_id].append(accuracy)
              
                # save in between runs just in case
                processed_subject_ids.append(subject_id)
                self.save_classifications(self.selection, accuracies_so_far, processed_subject_ids, intermediate_save = True)

            return accuracies_so_far

                     
class ClassifyStable(Classification):  
     
    def __init__(self, user_dir, paradigm):
        super(ClassifyStable, self).__init__(user_dir, paradigm)
    
    def classify(self):
        self.selection = "stable"
        
        # see if file already exists
        accuracies = self.load_classifications(self.selection)
        if len(accuracies) == len(["accuracy", "random"]):
            return(accuracies)
            
        else:
            accuracies["accuracy"] = {}
            accuracies["random"] = {}
            
            # get indices stable voxels per training set
            # Format: {participant 1: {first_last_word_of_1st_test_fold:[indices stable voxels], ...., first_last_word_of_kth_test_fold:[indices stable voxels]}
                # participant 2: {first_last_word_of_1st_test_fold:[indices stable voxels], ... first_last_word_of_kth_test_fold:[indices stable voxels]}... participant n: ... }
            stable_selection = SelectStable(self.user_dir, "classification")
            voxel_selection = stable_selection.select_voxels()
            
            self.read_data()
            for subject_id in self.blocks.keys():
                print("Going to classify data from: " + subject_id)             
                
                # for every word and fold ("group"), fetch activations of stable voxels
                scans = {}
                index_first_word = 0
                for group in range(0, self.amount_of_folds):
                    index_last_word = int(index_first_word + len(self.words) / self.amount_of_folds - 1)
                    key = self.words[index_first_word] + "_" + self.words[index_last_word]
                    scans[key] = []
                    
                    for block in self.blocks[subject_id]:
                        selected_voxel_activities = []
                        all_voxels = [event.scan for event in block.scan_events][0]
                        for voxel_index in voxel_selection[subject_id][key]:
                            selected_voxel_activities.append(all_voxels[voxel_index])
                        scans[key].append(selected_voxel_activities)
    
                    index_first_word = int(index_last_word + 1)
                            
                accuracies["accuracy"][subject_id] = self.run_svc(scans)
                accuracies["random"][subject_id] = self.random_generator(scans)

            self.save_classifications(self.selection, accuracies)

            return accuracies

