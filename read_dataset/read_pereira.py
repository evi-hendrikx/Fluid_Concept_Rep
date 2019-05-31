import numpy as np
import scipy.io
from .scan_elements import Block, ScanEvent
from .read_fmri_data_abstract import FmriReader
from os import listdir
from nilearn import datasets, image

# =============================================================================
#   Some participants have missing data; 
#   probably didn't matter for the authors since they selected the 5000 best voxels
# =============================================================================

# This class reads the word and sentence data from Pereira et al., 2018
# Paper: https://www.nature.com/articles/s41467-018-03068-4
# Data: https://evlab.mit.edu/sites/default/files/documents/index.html
# Make sure to also check the supplementary material.


class PereiraReader(FmriReader):

    def __init__(self, data_dir, paradigm=1):
        super(PereiraReader, self).__init__(data_dir)
        self.paradigm = paradigm
        

    def read_all_events(self, subject_ids=None): 

        # Collect scan events
        blocks = {}
        mni_voxels = {}
        xyz_voxels = {}


        if subject_ids == None:
            subject_ids = [file for file in listdir(self.data_dir)]
                
        for subject_id in subject_ids:
            print(subject_id)
            blocks_for_subject = []
                            
            # possible paradigms
            paradigms = ["sentences", "pictures", "wordclouds", "average"]
            paradigm_index = self.paradigm - 1
            if paradigm_index not in range(len(paradigms)):
                raise ValueError("please fill in 1 (for sentences), 2 (for pictures), 3 (for clouds), or 4 (for average)")
            
            # make blocks for sentences, pictures, or clouds
            elif paradigm_index in [0, 1, 2]:
                datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_" + paradigms[paradigm_index] + ".mat")
    
                # time stamp 
                n = 1 
                
                for i in range(0, len(datafile["examples"])):
                    word = datafile["keyConcept"][i][0][0]
                    
                    # only select words with right concreteness measures
                    # In this study mean + 0.5 sd and mean - 0.5 sd are used as boundaries
                    if datafile["labelsConcreteness"][i][0] > 4.0292 or datafile["labelsConcreteness"][i][0] < 2.9463:
                        if datafile["labelsConcreteness"][i][0] > 4.0292:
                            concreteness_id = "concrete"
                        else:
                            concreteness_id = "abstract"
                        scan = datafile["examples"][i]
                        scan_event = ScanEvent(subject_id, [(0, 0)], n, scan)
                        block = Block(subject_id, n, concreteness_id, [[word]], [scan_event], None)
                        blocks_for_subject.append(block)
                        n+=1
                blocks[subject_id] = blocks_for_subject                 
            
            # make blocks for average
            else:
                all_scans = []
                words = []
                
                # collect scans of all paradigms
                for paradigm in range(paradigm_index):   
                    datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_" + paradigms[paradigm] + ".mat")
                    all_scans.append([])
                    for i in range(0, len(datafile["examples"])):
                        
                        # don't make this an if-statement if you want to select all words (instead of words with a high or low concreteness rating)
                        if datafile["labelsConcreteness"][i][0] > 4.0292 or datafile["labelsConcreteness"][i][0] < 2.9463:
                            if datafile["labelsConcreteness"][i][0] > 4.0292:
                                concreteness_id = "concrete"
                            else:
                                concreteness_id = "abstract"
                            scan = datafile["examples"][i]
                            all_scans[paradigm].append(scan)
                            
                            # make word list (remains the same over paradigms)
                            if paradigm == 0:
                                word = datafile["keyConcept"][i][0][0]
                                words.append(word)                         
                        
                # average voxel values over all paradigms
                mean_scan = np.mean(all_scans, axis = 0)
                
                # time stamp increments with 1 for words 
                n = 1 
                
                for i in range(0, len(words)):
                    scan_event = ScanEvent(subject_id, [(0, 0)], n, mean_scan[i]) 
                    block = Block(subject_id, n, concreteness_id, [[words[i]]], [scan_event], None)
                    blocks_for_subject.append(block)
                    n+=1
                blocks[subject_id] = blocks_for_subject
                
            # get brain shape for every participant
            xyz_voxels[subject_id] = self.get_voxel_to_xyz_mapping(subject_id)
                 
        return blocks, xyz_voxels
    
    
    def get_voxel_to_xyz_mapping(self, subject_id):
        metadata = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_sentences.mat")["meta"] 
        coordinates_of_nth_voxel = metadata[0][0][5]
        voxel_to_xyz = []
        for voxel in range(0, coordinates_of_nth_voxel.shape[0]):
            voxel_to_xyz.append(coordinates_of_nth_voxel[voxel])
        return voxel_to_xyz
   
