import numpy as np
from scipy.stats import pearsonr
import scipy.io
from sklearn import neighbors
from select_voxels.select_voxels_abstract import SelectVoxels
from  read_dataset.read_pereira import PereiraReader
import heapq

# Note: If pickle files exists in the embedding dir, the encoder automatically reads them.
# Make sure to delete them, if you want new selections.


class SelectROI(SelectVoxels):

    def __init__(self, user_dir):
        super(SelectROI, self).__init__(user_dir)
        
    def select_voxels(self):
        print("Selecting roi voxels")
        method = "roi"
        roi_voxels = self.load_voxel_selection(method)
 
        # select for each participant voxel indices within ROI 
        if len(roi_voxels) == len(self.subject_ids):
            return roi_voxels
        else:
            rois = {}
            
            # For the regions of interest, the indexes in the AAL atlas are used. These are: 
                # IFG:[11, 12, 13, 14, 15, 16]      # MTG: [85, 86]         # posterior Cing: [35, 36]  
                # precuneus: [67, 68]       # FusiForm G: [55, 56]  # ParaHip.: [39, 40]
            # regions_of_interest = [11, 12, 13, 14, 15, 16, 85, 86, 35, 36, 67, 68, 55, 56, 39, 40]
            # I only used left hemispheric areas, since these were only significant in Wang et al., 2010
            areas = {"IFG": [11,13,15], "MTG":[85],"post_cing":[35],"precuneus": [67],"FFG": [55],"paraHIP":[39]}
            
            for subject_id in self.subject_ids:
                rois[subject_id] = {} 
                # regions remain constant over all paradigms, so I arbitrarily chose pictures
                datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_pictures.mat")
                
                for area in areas.keys():
                    for index in areas[area]:
                        voxels = [datafile['meta']['roiColumns'][0][0][0][0][index - 1][0][voxel_index][0] for voxel_index in range(len(datafile['meta']['roiColumns'][0][0][0][0][index - 1][0]))]
                        if area in rois[subject_id]:
                            rois[subject_id][area].extend(voxels)
                        else:
                            rois[subject_id][area] = voxels
            
            self.save_voxel_selection(rois, method)
        return rois

class SelectSearchlight(SelectVoxels):
    
    # right now this is done with xyz coordinates of the participants,, this might need to be in MNI space later(?)
    # it would be interesting to see which brain areas have a higher accuracy across participants, but I am not sure how to compare 
    # between participants if I do xyz and keep it statistically okay... Need to think about this
    
    # Right now, I just have no correct transformation to MNI, so I didn't really have a choice
    
    def __init__(self, user_dir):
        super(SelectSearchlight, self).__init__(user_dir)
        
    def select_voxels(self):
        print("Selecting searchlight voxels")
        method = "searchlight"
        searchlight_voxels = self.load_voxel_selection(method)
        
        # select for each participant
        if not len(searchlight_voxels) == len(self.subject_ids):
            
            # acquire xyz dimensions of subjects
            pereira_reader = PereiraReader(self.data_dir)
            pereira_data = pereira_reader.read_all_events()
            blocks, xyz_voxels = pereira_data
                
            for subject_id in self.subject_ids:
                
                voxels_subject = xyz_voxels[subject_id]
        
                # make every voxel a center and find its neighbors (and itself)
                print("Selecting searchlight voxels for " + subject_id)  
                
                # radius is expressed in number of voxels, 2 voxels == 4 mm
                radius = 2 
                info_neighbors = neighbors.NearestNeighbors(radius).fit(voxels_subject).radius_neighbors_graph(voxels_subject, radius)
                indices_all_neighbors = []
                for voxel in range(0,len(voxels_subject)):
                    index_neighbors = np.where(info_neighbors[voxel].toarray()==1)[1]
                    indices_all_neighbors.append(index_neighbors)
                    
                searchlight_voxels[subject_id] = indices_all_neighbors
            
            self.save_voxel_selection(searchlight_voxels, method)
            
        return searchlight_voxels
    
class SelectStable(SelectVoxels):
    
    def __init__(self, user_dir, analysis):
        super(SelectStable, self).__init__(user_dir) 
        
        self.analysis = analysis
        self.number_of_paradigms = 3
              
    def select_voxels(self):
        print("selecting stable voxels")
        method = "stable"
        stable_voxels = self.load_voxel_selection(method, self.analysis)

    
        if not len(stable_voxels) == len(self.subject_ids):
            
            # WARNING: paradigms need to be in the same order as in the reader
            paradigm_indices = [0, 1, 2]
            paradigms = ["sentences", "pictures", "wordclouds", "average"]
            all_blocks = {}
            
            # collect information of all participants in all paradigms
            print("reader for all paradigms start...")
            for paradigm_index in paradigm_indices:
                pereira_reader = PereiraReader(self.data_dir, paradigm = paradigm_index + 1)
                pereira_data = pereira_reader.read_all_events()
                blocks, xyz_voxels = pereira_data
                all_blocks[paradigms[paradigm_index]] = blocks
            print("reader for all paradigms done...")
            
            # select info per participant over all paradigms and calculate stability

            for subject_id in self.subject_ids:
                print("collecting presentations for " + subject_id)
                words2scans = {}
                for paradigm_index in paradigm_indices:
                    scan = []
                    for block in all_blocks[paradigms[paradigm_index]][subject_id]:
                        scan = [event.scan for event in block.scan_events][0]
                        word = [word for word in block.sentences][0][0]
                        scans = []
                        if word in words2scans:
                            scans = words2scans[word]
                        scans.append(scan)
                        words2scans[word] = scans
                        
                # Build a matrix with all activations for each subject                
                all_words = [word for word in words2scans.keys()]
                
                # Number of voxels differs for each subject, but is the same for every stimulus
                # I am simply checking the number of voxels for the example word "ability" here.
                total_number_voxels = len(words2scans["ability"][0])
                paradigm_matrix = np.zeros((total_number_voxels, self.number_of_paradigms, len(all_words)))
                for voxel_index in np.arange(total_number_voxels):
                    for word_index in np.arange(len(all_words)):
                        word = all_words[word_index]
                        for paradigm_index in np.arange(self.number_of_paradigms):
                            voxel_value = words2scans[word][paradigm_index][voxel_index]
                            paradigm_matrix[voxel_index][paradigm_index][word_index] = voxel_value
                        
                # stability measures vary per analysis method, since classification and encoding both have training data (12-leave out and 2-leave-out respectively)
                # for which stablity will be calculated, while clustering and RSA do not (and so stability is calculated over the entire set)
                if self.analysis == "classification" or self.analysis == "encoding":
                    stable_voxels[subject_id] = self.calculate_stable_voxels_per_group(words2scans, all_words, total_number_voxels, paradigm_matrix)
                elif self.analysis == "RSA" or self.analysis == "clustering":
                    stable_voxels[subject_id] = self.calculate_stable_voxels(words2scans, total_number_voxels, paradigm_matrix) 
                else:
                    print("please fill in a valid selection method: classification, RSA, clustering, or encoding")
            
            
            self.save_voxel_selection(stable_voxels, method, analysis = self.analysis)
            
        return stable_voxels

    def calculate_stable_voxels_per_pair(self, words2scans, all_words, total_number_voxels, paradigm_matrix, number_of_selected_voxels = 500):
        print("calculating stable voxels")
                
        # For each voxel, calculate the correlation of the activation over all words in the trainingset between pairs of paradigms
        stability_matrix = np.zeros(total_number_voxels)
        stable_voxels_per_pair = {}
        words = list(words2scans.keys())
        for word1 in range(0, len(all_words)):
            for word2 in range(word1 + 1, len(all_words)):  
                for voxel_index in np.arange(total_number_voxels):
                    paradigm_correlation_pairs = []
                    for paradigm_index1 in np.arange(self.number_of_paradigms):
                        for paradigm_index2 in np.arange(paradigm_index1, self.number_of_paradigms):
                            data1 = paradigm_matrix[voxel_index][paradigm_index1]
                            data2 = paradigm_matrix[voxel_index][paradigm_index2]
    
                            # exclude the data for the two test words
                            slice1_1 = data1[0:word1]
                            slice1_2 = data1[word1 + 1:word2]
                            slice1_3 = data1[word2 + 1:]
                            vector1 = []
                            for slice in [slice1_1, slice1_2, slice1_3]:
                                if len(slice) > 0:
                                    vector1.extend(slice)
                            slice2_1 = data2[0:word1]
                            slice2_2 = data2[word1 + 1:word2]
                            slice2_3 = data2[word2 + 1:]
                            vector2 = []
                            for slice in [slice2_1, slice2_2, slice2_3]:
                                if len(slice) > 0:
                                    vector2.extend(slice)
    
                            correlation_for_paradigm_pair = pearsonr(vector1, vector2)[0]
                            paradigm_correlation_pairs.append(correlation_for_paradigm_pair)
                            
                    stability_matrix[voxel_index] = np.mean(paradigm_correlation_pairs)
                
                # Find the voxels with the highest mean pearson correlation and return the indices 
                # heapq finds the largests n values. Enumerate makes sure the original index is remembered
                stable_voxel_ids = [i for x, i in heapq.nlargest(number_of_selected_voxels, ((x, i) for i, x in enumerate (stability_matrix)))]

                key = words[word1] + "_" + words[word2]
                stable_voxels_per_pair[key] = stable_voxel_ids
                print("Saving stable voxels to dict for pair: "+ str(key))
                print("Number of keys saved already: " +str(len(stable_voxels_per_pair.keys())))
                
        return stable_voxels_per_pair 

    def calculate_stable_voxels(self, words2scans, total_number_voxels, paradigm_matrix, number_of_selected_voxels = 500):
        print("calculating stable voxels")
        
        # For each voxel, calculate the correlation of the activation over all words between pairs of paradimgs
        stability_matrix = np.zeros(total_number_voxels)
        for voxel_index in np.arange(total_number_voxels):
            correlations_paradigms = []
            for paradigm_index1 in np.arange(self.number_of_paradigms):
                for paradigm_index2 in np.arange(paradigm_index1, self.number_of_paradigms):
                    data1 = paradigm_matrix[voxel_index][paradigm_index1]
                    data2 = paradigm_matrix[voxel_index][paradigm_index2]
                    correlation_for_paradigm_pair = pearsonr(data1, data2)[0]
                    correlations_paradigms.append(correlation_for_paradigm_pair)
            stability_matrix[voxel_index] = np.mean(correlations_paradigms)
                   
        # Find the voxels with the highest mean pearson correlation and return the indices 
        # heapq finds the largests n values. Enumerate makes sure the original index is remembered
        stable_voxel_ids = [i for x, i in heapq.nlargest(number_of_selected_voxels, ((x, i) for i, x in enumerate (stability_matrix)))]        
        
        return stable_voxel_ids
    
    def calculate_stable_voxels_per_group(self, words2scans, all_words, total_number_voxels, paradigm_matrix, number_of_selected_voxels = 500, amount_of_groups = 11):      
        print("calculating stable voxels")
        
        stability_matrix = np.zeros(total_number_voxels) 
        stable_voxels_per_group = {} 
        
        # select stable voxels over all training sets and save it under the name of the leave-out-group (test set)
        index_first_word = 0
        for group in range(0, amount_of_groups):
            index_last_word = int(index_first_word + len(all_words) / amount_of_groups - 1)
            
            for voxel_index in np.arange(total_number_voxels):
                paradigm_correlation_pairs = []
                for paradigm_index1 in np.arange(self.number_of_paradigms):
                    for paradigm_index2 in np.arange(paradigm_index1, self.number_of_paradigms):
                        data1 = paradigm_matrix[voxel_index][paradigm_index1]
                        data2 = paradigm_matrix[voxel_index][paradigm_index2]

                        # exclude the data for the two test words
                        slice1_1 = data1[0:index_first_word]
                        slice1_2 = data1[index_last_word + 1:]
                        vector1 = []
                        for slice in [slice1_1, slice1_2]:
                            if len(slice) > 0:
                                vector1.extend(slice)
                        slice2_1 = data2[0:index_first_word]
                        slice2_2 = data2[index_last_word + 1:]
                        vector2 = []
                        for slice in [slice2_1, slice2_2]:
                            if len(slice) > 0:
                                vector2.extend(slice)

                        correlation_for_paradigm_pair = pearsonr(vector1, vector2)[0]
                        paradigm_correlation_pairs.append(correlation_for_paradigm_pair)
                        
                stability_matrix[voxel_index] = np.mean(paradigm_correlation_pairs)
            
            # Find the voxels with the highest mean pearson correlation and return the indices 
            # heapq finds the largests n values. Enumerate makes sure the original index is remembered
            key = all_words[index_first_word] + "_" + all_words[index_last_word]
            stable_voxel_ids = [i for x, i in heapq.nlargest(number_of_selected_voxels, ((x, i) for i, x in enumerate (stability_matrix)))]
            stable_voxels_per_group[key] = stable_voxel_ids
            
            print("Saving stable voxels to dict for pair: "+ str(key))
            print("Number of keys saved already: " +str(len(stable_voxels_per_group.keys())))
            
            index_first_word = index_last_word + 1
            
               
        return stable_voxels_per_group
 
    
