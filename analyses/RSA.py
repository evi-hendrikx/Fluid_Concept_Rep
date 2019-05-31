from select_voxels.selection_methods import SelectROI, SelectStable, SelectSearchlight
from embeddings.all_embeddings import PereiraEncoder, ImageEncoder, CombiEncoder, RandomEncoder
from analyses.RSA_abstract import RSA
import scipy
import numpy as np


class RSA_ROI_Stable(RSA):  
    
    def __init__(self, user_dir, paradigm):
        super(RSA_ROI_Stable, self).__init__(user_dir, paradigm)
        self.selection = ""

    def run_RSA(self):
        
        # see if file exists
        correlations = self.load_RSA()
        if list(correlations.keys()) == ["total", "abstract", "concrete"]:
            return correlations
        
        else:
            self.read_data()
            labels = []
            all_scans = []
            all_abs_scans = []
            all_conc_scans = []
                
            for selection in ["roi", "stable"]:
                self.selection = selection
                
                # get voxel selections
                if self.selection == "roi":
                    selection = SelectROI(self.user_dir)
                else:
                    selection = SelectStable(self.user_dir, "RSA")
                voxel_selection = selection.select_voxels()
                
    
                # get scans of the selected voxels and appropriate words
                if self.selection == "stable":
                    for subject_id in self.blocks.keys():
                        scans = []
                        abstract_scans = []
                        concrete_scans = []
                        for block in self.blocks[subject_id]:
                            selected_voxel_activities = []
                            all_voxels = [event.scan for event in block.scan_events][0]
                            for voxel_index in voxel_selection[subject_id]:
                                selected_voxel_activities.append(all_voxels[voxel_index])
                            scans.append(selected_voxel_activities)
                            if block.concreteness_id == "abstract":
                                abstract_scans.append(selected_voxel_activities)
                            else:
                                concrete_scans.append(selected_voxel_activities)
                                
                        all_scans.append(scans)
                        all_abs_scans.append(abstract_scans)
                        all_conc_scans.append(concrete_scans)
                        labels.append(subject_id + "_stable")
                else:
                    for subject_id in self.blocks.keys():
                        for area in voxel_selection[subject_id].keys():
                            scans = []
                            abstract_scans = []
                            concrete_scans = []
                            for block in self.blocks[subject_id]:
                                selected_voxel_activities = []
                                all_voxels = [event.scan for event in block.scan_events][0]
                                for voxel_index in voxel_selection[subject_id][area]:
                                    selected_voxel_activities.append(all_voxels[voxel_index])
                                scans.append(selected_voxel_activities)
                                if block.concreteness_id == "abstract":
                                    abstract_scans.append(selected_voxel_activities)
                                else:
                                    concrete_scans.append(selected_voxel_activities)
                                    
                            all_scans.append(scans)
                            all_abs_scans.append(abstract_scans)
                            all_conc_scans.append(concrete_scans)
                            labels.append(subject_id + "_" + area)
                        
            # get embeddings for to be used words
            all_embeddings = []
            all_abs_embs = []
            all_conc_embs = []
            for embedder in (PereiraEncoder(self.save_embedding_dir), ImageEncoder(self.save_embedding_dir), CombiEncoder(self.save_embedding_dir), RandomEncoder(self.save_embedding_dir)):

                if embedder.__class__.__name__ ==  "RandomEncoder":
                    counter = 0
                else:
                    counter =999
                
                while counter < 1000:
                    embeddings_unsorted = embedder.get_embeddings(self.words)
                    embeddings = []
                    abstract_embs = []
                    concrete_embs = []
                    
                    # get embeddings in alphabetical order and without the actual words
                    for word in self.words:
                        embeddings.append(embeddings_unsorted[word])
                        if word in self.abstract_words:
                            abstract_embs.append(embeddings_unsorted[word])
                        else:
                            concrete_embs.append(embeddings_unsorted[word])
                    
                    
                    all_embeddings.append(embeddings)
                    all_abs_embs.append(abstract_embs)
                    all_conc_embs.append(concrete_embs)
                    
                    labels.append(embedder.__class__.__name__)
                    
                    counter +=1
            
            # combine all info to one file
            data = [x for x in all_scans]
            data.extend([x for x in all_embeddings])
            
            abs_data = [x for x in all_abs_scans]
            abs_data.extend([x for x in all_abs_embs])
            
            conc_data = [x for x in all_conc_scans]
            conc_data.extend([x for x in all_conc_embs])
            
            print(len(data))
            print(len(all_scans))
            print(len(all_embeddings))
            
            
            
            # run RSAs
            self.correlations = {}
            #self.runRSA(data, labels, "total")
            self.runRSA(abs_data, labels, "abstract")
            self.runRSA(conc_data, labels, "concrete")
            
            
            return self.correlations
        
class RSA_Searchlight(RSA):
    
    def __init__(self, user_dir, paradigm):
        super(RSA_Searchlight, self).__init__(user_dir, paradigm)
        self.selection = "searchlight"
        
    def run_RSA(self):
        
        print("run_RSA RSA")
          # see if file exists
        correlations = self.load_RSA()
        if len(correlations) == len(["total", "abstract", "concrete"]):
            return correlations
        
        else:
            
            selection = SelectSearchlight(self.user_dir)
            voxel_selection = selection.select_voxels()
            
            self.read_data()
            labels = []
            
            # get scans for to be used words                    
            all_scans = []
            all_abs_scans = []
            all_conc_scans = []

            # get scans of the selected voxels and appropriate words
            self.random_voxels = {}
            for subject_id in self.blocks.keys():
                random_voxel_indices = []
                print("run subject " + subject_id)
                voxel_index = 0

                random_voxels = self.select_three_voxels(voxel_selection[subject_id], subject_id)
                
                # can not use .index in np arrays, so need to transform
                voxel_storage = []
                for voxel_indices in voxel_selection[subject_id]:
                    voxel_storage.append(list(voxel_indices))
                voxel_selection[subject_id] = voxel_storage

                # select acivities of random voxels
                for random_voxel in random_voxels:
                    print(random_voxel)
                    index = (voxel_selection[subject_id]).index(list(random_voxel))
                    random_voxel_indices.append(index)
                self.random_voxels[subject_id] = random_voxel_indices
                    
                # select activities for all voxels
                for voxel_indices in voxel_selection[subject_id]:
                    scans = []
                    abstract_scans = []
                    concrete_scans = []
                    
                    for block in self.blocks[subject_id]:
                        selected_voxel_activities = []
                        all_voxels = [event.scan for event in block.scan_events][0]
                        for voxel in voxel_indices:
                            selected_voxel_activities.append(all_voxels[voxel])
                        scans.append(selected_voxel_activities)
                        if block.concreteness_id == "abstract":
                            abstract_scans.append(selected_voxel_activities)
                        else:
                            concrete_scans.append(selected_voxel_activities)
                    
                            
                    all_scans.append(scans)
                    all_abs_scans.append(abstract_scans)
                    all_conc_scans.append(concrete_scans)
                    labels.append(subject_id + "_" + str(voxel_index))
                    voxel_index += 1
                    
            # create vectors with the embeddings for each word
            all_embeddings = []
            all_abs_embs = []
            all_conc_embs = []
            for embedder in (PereiraEncoder(self.save_embedding_dir), ImageEncoder(self.save_embedding_dir), CombiEncoder(self.save_embedding_dir), RandomEncoder(self.save_embedding_dir)):
                print("run embedder")
                
                # random will be tested 106000 times for the three specifically selected voxels
                if embedder.__class__.__name__ ==  "RandomEncoder":
                    counter = 0
                else:
                    counter =105999
                
                while counter < 106000:
                    embeddings_unsorted = embedder.get_embeddings(self.words)
                    embeddings = []
                    abstract_embs = []
                    concrete_embs = []
                    
                    # get embeddings in alphabetical order and without the actual words
                    for word in self.words:
                        embeddings.append(embeddings_unsorted[word])
                        if word in self.abstract_words:
                            abstract_embs.append(embeddings_unsorted[word])
                        else:
                            concrete_embs.append(embeddings_unsorted[word])
                    
                    
                    all_embeddings.append(embeddings)
                    all_abs_embs.append(abstract_embs)
                    all_conc_embs.append(concrete_embs)
                    
                    labels.append(embedder.__class__.__name__)
                    
                    counter +=1
                    if counter % 100 == 0:
                        print("run random " + str(counter))
           
            # combine all info to one file
            data = [x for x in all_scans]
            data.extend([x for x in all_embeddings])
            
            abs_data = [x for x in all_abs_scans]
            abs_data.extend([x for x in all_abs_embs])
            
            conc_data = [x for x in all_conc_scans]
            conc_data.extend([x for x in all_conc_embs])
            
            print(len(data))
            print(len(all_scans))
            print(len(all_embeddings))  
            
            # run RSAs
            self.correlations = {}
            #self.runRSA(data, labels, "total")
            self.runRSA(abs_data, labels, "abstract")
            self.runRSA(conc_data, labels, "concrete")
            
            
            return self.correlations

    def select_three_voxels(self, voxel_selection, subject_id):
        selected_voxels = []
        
        # select areas from matlab file that fall in these lobes, selection is based on the name of the lobe in the region of AAL atlas
        # this is the same for all subjects
        frontal = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]; frontal_done = False
        occipital = [49, 50, 51, 52, 53, 54]; occipital_done = False
        temporal = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90]; temporal_done = False
        
        paradigms = ["sentences", "pictures", "wordclouds", "average"]

        # regions remain constant over all paradigms, so I arbitrarily chose pictures
        datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_pictures.mat")
        voxel_to_region_mapping = datafile["meta"]["roiMultimask"][0][0][0][0]
        
        possible_files = [scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_" + paradigm + ".mat") for paradigm in paradigms[:-1]]
        
        for voxels in np.random.permutation(voxel_selection):
            
            # voxels should not all be 0, because then they are skipped
            for file in possible_files:
                if all(all(activities == 0 for activities in (file["examples"][word][voxel] for word in range(len(file["examples"])))) for voxel in voxels):
                    continue
 
            if all(voxel_to_region_mapping[voxel] in frontal for voxel in voxels) and frontal_done  == False:
                selected_voxels.append(voxels)
                frontal_done = True
                print("found voxel for frontal")
            elif all(voxel_to_region_mapping[voxel] in occipital for voxel in voxels) and occipital_done  == False:
                selected_voxels.append(voxels)
                occipital_done = True
                print("found voxel for occipital")
            elif all(voxel_to_region_mapping[voxel] in temporal for voxel in voxels) and temporal_done  == False:
                selected_voxels.append(voxels)
                temporal_done = True
                print("found voxel for temporal")
            else:
                continue
            
            if frontal_done == True and occipital_done == True and temporal_done == True:
                break
        
        if not len(selected_voxels) == 3:
            print("WE HAVE A PROBLEM MORE OR LESS VOXELS SELECTED THAN 3")
        
        return selected_voxels
