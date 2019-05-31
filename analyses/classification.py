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
    
    def classify(self, reduced_random = False, only_random = False):
        self.selection = "searchlight"
        self.reduced_random = reduced_random

        # see if file already exists
        accuracies = self.load_classifications(self.selection)
        if len(accuracies) == len(["accuracy", "random"]):
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
            accuracies_so_far["random"] = {}
            processed_subject_ids = []
            
            self.read_data()
            for subject_id in self.blocks.keys():
                print("Going to classify data from: " + subject_id)
        
                accuracies_so_far["accuracy"][subject_id] = []
                accuracies_so_far["random"][subject_id] = []
                
                if only_random == False:
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
                        if self.reduced_random == False:
                            random_accuracies = self.random_generator(scans)
                    
                        # collect accuracies for all voxels as center
                        accuracies_so_far["accuracy"][subject_id].append(accuracy)
                        if self.reduced_random == False:
                            accuracies_so_far["random"][subject_id].append(random_accuracies)

                # select three voxels in different locations
                if self.reduced_random == True:
                    voxel_selection_random = self.select_three_voxels(voxel_selection[subject_id], subject_id)
                    for voxel_indices in voxel_selection_random:
                        scans = []
                        for block in self.blocks[subject_id]:
                            selected_voxel_activities = []
                            all_voxels = [event.scan for event in block.scan_events][0]
                            for voxel_index in voxel_indices:
                                selected_voxel_activities.append(all_voxels[voxel_index])
                            scans.append(selected_voxel_activities)

                        random_accuracies = self.random_generator(scans)
                        accuracies_so_far["random"][subject_id].append(random_accuracies)
                    accuracies_so_far["random"][subject_id] = np.mean(accuracies_so_far["random"][subject_id], axis = 0)
                
                # save in between runs just in case
                processed_subject_ids.append(subject_id)
                self.save_classifications(self.selection, accuracies_so_far, processed_subject_ids, intermediate_save = True)

            return accuracies_so_far

    def select_three_voxels(self, voxel_selection, subject_id):
        selected_voxels = []

        # select areas form matlab file that fall in these lobes, selection is based on the name of the lobes in the region of the AAL atlas
        # this is the same for all subjects
        frontal = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,23,24,25,26]; frontal_done = False
        occipital = [49,50,51,52,53,54]; occipital_done = False
        temporal = [81,82,83,84,85,86,87,88,89,90]; temporal_done = False

        datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_pictures.mat")
        voxel_to_region_mapping = datafile["meta"]["roiMultimask"][0][0][0][0]
        for voxels in np.random.permutation(voxel_selection):
            if all(voxel_to_region_mapping[voxel] in frontal for voxel in voxels) and frontal_done == False:
                selected_voxels.append(voxels)
                frontal_done = True
            elif all(voxel_to_region_mapping[voxel] in occipital for voxel in voxels) and occipital_done == False:
                selected_voxels.append(voxels)
                occipital_done = True
            elif all(voxel_to_region_mapping[voxel] in temporal for voxel in voxels) and temporal_done == False:
                selected_voxels.append(voxels)
                temporal_done = True
            else:
                continue

            if frontal_done == True and occipital_done == True and temporal_done == True:
                break

        return selected_voxels
                  
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

