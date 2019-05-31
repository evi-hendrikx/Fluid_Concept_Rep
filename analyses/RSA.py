from select_voxels.selection_methods import SelectROI, SelectStable
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
            self.runRSA(data, labels, "total")
            self.runRSA(abs_data, labels, "abstract")
            self.runRSA(conc_data, labels, "concrete")
            
            
            return self.correlations
