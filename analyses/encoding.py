from select_voxels.selection_methods import SelectROI, SelectSearchlight, SelectStable
from analyses.encoding_abstract import Encoding
from embeddings.all_embeddings import ImageEncoder, PereiraEncoder, CombiEncoder, RandomEncoder
import scipy.io
import numpy as np


class EncodingROI(Encoding):  
    
    def __init__(self, user_dir, paradigm):
        super(EncodingROI, self).__init__(user_dir, paradigm)
    
    def encode(self):
        self.selection = "roi"
        
        # open file if it exists
        accuracies = self.load_encodings(self.selection)
        if len(accuracies) == len(self.embeddings):
            return(accuracies)
            
        else:
            roi_selection = SelectROI(self.user_dir)
            voxel_selection = roi_selection.select_voxels()
            
            self.read_data()
            for embedding in self.embeddings:
                accuracies[embedding] = {}
                
                # get the right embeddings
                if embedding == "linguistic":
                    encoder = PereiraEncoder(self.save_embedding_dir)
                elif embedding == "non-linguistic":
                    encoder = ImageEncoder(self.save_embedding_dir)
                elif embedding == "combi":
                    encoder = CombiEncoder(self.save_embedding_dir)
                else:
                    encoder = RandomEncoder(self.save_embedding_dir)
                embeddings = encoder.get_embeddings(self.words)
                
                for subject_id in voxel_selection.keys():
                    accuracies[embedding][subject_id] = {}
                    for area in voxel_selection[subject_id].keys():
                        accuracies[embedding][subject_id][area] = {}
                        scans = []
                        print("Going to enode data from " + subject_id + ", area " + area + " with " + embedding + " embeddings")
                    
                        # per word, fetch corresponding activities from the requested voxels
                        for block in self.blocks[subject_id]:
                            selected_voxel_activities = []
                            all_voxels = [event.scan for event in block.scan_events][0]
                            for voxel_index in voxel_selection[subject_id][area]:
                                selected_voxel_activities.append(all_voxels[voxel_index])
                            scans.append(selected_voxel_activities) 

                        if embedding == "random":
                            accuracies[embedding][subject_id][area]["total"] = []
                            accuracies[embedding][subject_id][area]["abstract"] = []
                            accuracies[embedding][subject_id][area]["concrete"] = []
                            for i in range(1000):
                                embeddings = encoder.get_embeddings(self.words)
                                if i % 10 == 0:
                                    print("repetition " + str(i) + " from " + subject_id + ", area " + area)
                                total, abstract, concrete = self.run_encoding(scans, embeddings, embedding)
                                accuracies[embedding][subject_id][area]["total"].append(total)
                                accuracies[embedding][subject_id][area]["abstract"].append(abstract)
                                accuracies[embedding][subject_id][area]["concrete"].append(concrete)
                        else:
                            accuracies[embedding][subject_id][area]["total"], accuracies[embedding][subject_id][area]["abstract"], accuracies[embedding][subject_id][area]["concrete"] = self.run_encoding(scans, embeddings, embedding)

                self.intermediate_save_encodings(self.selection, accuracies, embedding)
            return accuracies 
    
        
class EncodingSearchlight(Encoding):
    def __init__(self, user_dir, paradigm):
        super(EncodingSearchlight, self).__init__(user_dir, paradigm)
    
    def encode(self):
        self.selection = "searchlight"
        
        accuracies = self.load_encodings(self.selection)
        if len(accuracies) == len(self.embeddings):
            return(accuracies)

        else:
            # get indices from neighbors for every voxel
            # Format: {participant 1: [indices voxel 1 as center with neighbors]...[indices vox. k as center with neighbors] 
            # participant 2: [indices vox.1 as center with neighbors]...[indices vox. m as center with neighbors] ... participant n:....}
            searchlight_selection = SelectSearchlight(self.user_dir)
            
            self.read_data()
            for embedding in self.embeddings:
                accuracies[embedding] = {}
                
                voxel_selection = searchlight_selection.select_voxels()

                # get the right embeddings
                if embedding == "linguistic":
                    encoder = PereiraEncoder(self.save_embedding_dir)
                elif embedding == "non-linguistic":
                    encoder = ImageEncoder(self.save_embedding_dir)
                elif embedding == "combi":
                    encoder = CombiEncoder(self.save_embedding_dir)
                else:
                    encoder = RandomEncoder(self.save_embedding_dir)
                embeddings = encoder.get_embeddings(self.words)
                
                for subject_id in self.subject_ids:
                    print("Going to encode data from: " + subject_id + " with " + embedding + " embeddings")
                    accuracies[embedding][subject_id] = {}
                    accuracies[embedding][subject_id]["total"] = []
                    accuracies[embedding][subject_id]["abstract"] = []
                    accuracies[embedding][subject_id]["concrete"] = []

                    index = 0
                    skipped = 0 
                    datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_sentences.mat")["meta"]
                    if embedding == "random":
                        voxel_selection[subject_id] = self.select_three_voxels(voxel_selection[subject_id], subject_id)

                    for voxel_indices in voxel_selection[subject_id]:
                        if embedding == "random":
                            accuracies[embedding][subject_id]["total"].append([])
                            accuracies[embedding][subject_id]["abstract"].append([])
                            accuracies[embedding][subject_id]["concrete"].append([])                       

                        if datafile["roiMultimask"][0][0][0][0][index][0] == 0 and not embedding == "random":
                            total = 0; abstract = 0; concrete = 0 
                            skipped += 1
                            accuracies[embedding][subject_id]["total"].append(total)
                            accuracies[embedding][subject_id]["abstract"].append(abstract)
                            accuracies[embedding][subject_id]["concrete"].append(concrete)
                      
                        # for every word, fetch center and neighbor voxels
                        else:
                            scans = []
                            for block in self.blocks[subject_id]:
                                selected_voxel_activities = []
                                all_voxels = [event.scan for event in block.scan_events][0]
                                for voxel_index in voxel_indices:
                                    selected_voxel_activities.append(all_voxels[voxel_index])
                                scans.append(selected_voxel_activities)
                            
                            if all(all(voxels == 0 for voxels in scan) for scan in scans):
                                total = 0; abstract = 0; concrete = 0
                                skipped += 1
                                accuracies[embedding][subject_id]["total"].append(total)
                                accuracies[embedding][subject_id]["abstract"].append(abstract)
                                accuracies[embedding][subject_id]["concrete"].append(concrete)
                            else:
                                if embedding == "random":
                                    for i in range(106000):
                                        embeddings = encoder.get_embeddings(self.words)
                                        print(str(i) + ", voxel " + str(index + 1) + " / 3")
                                        total, abstract, concrete = self.run_encoding(scans, embeddings, embedding)
                                        accuracies[embedding][subject_id]["total"][index].append(total)
                                        accuracies[embedding][subject_id]["abstract"][index].append(abstract)
                                        accuracies[embedding][subject_id]["concrete"][index].append(concrete)
                            
                                else:
                                    total, abstract, concrete = self.run_encoding(scans,embeddings, embedding)                                    
                                    accuracies[embedding][subject_id]["total"].append(total)
                                    accuracies[embedding][subject_id]["abstract"].append(abstract)
                                    accuracies[embedding][subject_id]["concrete"].append(concrete)
                        
                        index += 1
                    
                    for concreteness in ["total","abstract", "concrete"]:
                        accuracies["random"][subject_id][concreteness] = np.mean(accuracies["random"][subject_id][concreteness], axis = 0)
                         
               
                    self.intermediate_save_encodings(self.selection, accuracies, embedding, subject_id)
                    
        return accuracies
        
    
class EncodingStable(Encoding):  
    
    def __init__(self, user_dir, paradigm):
        super(EncodingStable, self).__init__(user_dir, paradigm)
    
    def encode(self):
        self.selection = "stable"
        
        # open file if it exists
        accuracies = self.load_encodings(self.selection)
        if len(accuracies) == len(self.embeddings):
            return(accuracies)

        else:

            # get indices stable voxels per training set
            # Format: {participant 1: {first_last_word_of_1st_test_fold:[indices stable voxels], ...., first_last_word_of_kth_test_fold:[indices stable voxels]}
            # participant 2: {first_last_word_of_1st_test_fold:[indices stable voxels], ... first_last_word_of_kth_test_fold:[indices stable voxels]}... participant n: ... }
            stable_selection = SelectStable(self.user_dir, analysis = "encoding")
            voxel_selection = stable_selection.select_voxels()
            
            self.read_data()
            for embedding in self.embeddings:
                accuracies[embedding] = {}
                
                # get the right embeddings
                if embedding == "linguistic":
                    encoder = PereiraEncoder(self.save_embedding_dir)
                elif embedding == "non-linguistic":
                    encoder = ImageEncoder(self.save_embedding_dir)
                elif embedding == "combi":
                    encoder = CombiEncoder(self.save_embedding_dir)
                else:
                    encoder = RandomEncoder(self.save_embedding_dir)
                embeddings = encoder.get_embeddings(self.words)

                for subject_id in voxel_selection.keys():
                    print("Going to encode data from: " + subject_id + " with " + embedding + " embeddings")
                    scans = {}
                    
                    index_first_word = 0
                    for fold in range(0, self.amount_of_folds):
                        index_last_word = int(index_first_word + len(self.words) / self.amount_of_folds - 1)
                        key = self.words[index_first_word] + "_" + self.words[index_last_word]
                        scans[key] = []
                    
                        # per word, fetch corresponding activities from the requested voxels
                        for block in self.blocks[subject_id]:
                            selected_voxel_activities = []
                            all_voxels = [event.scan for event in block.scan_events][0]
                            for voxel_index in voxel_selection[subject_id][key]:
                                selected_voxel_activities.append(all_voxels[voxel_index])
                            scans[key].append(selected_voxel_activities) 
                        
                        index_first_word = int(index_last_word + 1)
                    
                    accuracies[embedding][subject_id] = {}
                    if embedding == "random":
                        accuracies[embedding][subject_id]["total"] = []
                        accuracies[embedding][subject_id]["abstract"] = []
                        accuracies[embedding][subject_id]["concrete"] = []
                        for i in range(1000):
                            embeddings = encoder.get_embeddings(self.words)
                            total, abstract, concrete = self.run_encoding(scans, embeddings, embedding)
                            accuracies[embedding][subject_id]["total"].append(total)
                            accuracies[embedding][subject_id]["abstract"].append(abstract)
                            accuracies[embedding][subject_id]["concrete"].append(concrete)
                    else:
                        accuracies[embedding][subject_id]["total"], accuracies[embedding][subject_id]["abstract"], \
                        accuracies[embedding][subject_id]["concrete"] = self.run_encoding(scans, embeddings, embedding)
            
                self.intermediate_save_encodings(self.selection, accuracies, embedding)
            
            return accuracies 
