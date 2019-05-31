from os import listdir, path, makedirs
import pickle
from read_dataset.read_pereira import PereiraReader
import numpy as np
#from os import path
from embeddings.all_embeddings import PereiraEncoder, ImageEncoder, CombiEncoder, RandomEncoder
from sklearn.cluster import KMeans



class Clustering(object):
    def __init__(self, user_dir, paradigm):        
        self.paradigm = paradigm
        
        self.user_dir = user_dir
        self.data_dir = "DATADIR"
        self.subject_ids = [file for file in listdir(self.data_dir)]
        
        self.save_embedding_dir = user_dir + "embeddings/already_embedded/"
        
        # where the analysis will be stored
        self.save_dir = user_dir + "analyses/already_analysed/"
                
        self.n_total_clusters = 2
        self.n_concreteness_clusters = 11
        
        
    def read_data(self):
        # get information of all participants
        print("reader start for paradigm " + str(self.paradigm))
        pereira_reader = PereiraReader(self.data_dir, paradigm = self.paradigm)
        pereira_data = pereira_reader.read_all_events()
        blocks, xyz_voxels = pereira_data
        self.blocks = blocks
        print("reader done...")
        
        # all participants get the same words (and concreteness), therefore I arbitrarily chose M01
        self.words = []
        self.abstract_words = []
        self.concrete_words = []
        for block in self.blocks["M01"]:
            self.words.append(block.sentences[0][0])
            if block.concreteness_id == "abstract":
                self.abstract_words.append(block.sentences[0][0])
            else:
                self.concrete_words.append(block.sentences[0][0])
                
                
                
    def create_embeddding_lists(self):
        
        for embedder in (PereiraEncoder(self.save_embedding_dir), ImageEncoder(self.save_embedding_dir), CombiEncoder(self.save_embedding_dir), RandomEncoder(self.save_embedding_dir)):
            
            embeddings_unsorted = embedder.get_embeddings(self.words)
            self.embeddings = []
            self.abstract_embs = []
            self.concrete_embs = []
            
            # get embeddings in alphabetical order and without the actual words
            self.concrete_index = np.zeros(len(self.words))
            for word in self.words:
                self.embeddings.append(embeddings_unsorted[word])
                if word in self.abstract_words:
                    self.abstract_embs.append(embeddings_unsorted[word])
                else:
                    self.concrete_embs.append(embeddings_unsorted[word])
                    self.concrete_index[self.words.index(word)] = 1
                    
    def save_clusters(self, data):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        accuracy_file = self.save_dir + "clustering_clusters_" + paradigms[paradigm_index] + "_" + self.selection + ".pickle"
            
        makedirs(path.dirname(accuracy_file), exist_ok=True)
        with open(accuracy_file, 'wb') as handle:
            pickle.dump(data, handle)
        print("Accuracies stored in file: " + accuracy_file)
        
    def load_clusters(self):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        file_path = self.save_dir + "clustering_clusters_" + paradigms[paradigm_index] + "_" + self.selection + ".pickle"
            
        
        if path.isfile(file_path):
            print("Loading clusters from " + file_path)
            with open(file_path, 'rb') as handle:
                clusters = pickle.load(handle)
            return clusters
        else:
            return {}

    def save_ARI(self, data):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        accuracy_file = self.save_dir + "clustering_ARI_" + paradigms[paradigm_index] + "_" + self.selection + ".pickle"
            
        makedirs(path.dirname(accuracy_file), exist_ok=True)
        with open(accuracy_file, 'wb') as handle:
            pickle.dump(data, handle)
        print("Accuracies stored in file: " + accuracy_file)
        
    def load_ARI(self):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        file_path = self.save_dir + "clustering_ARI_" + paradigms[paradigm_index] + "_" + self.selection + ".pickle"
            
        
        if path.isfile(file_path):
            print("Loading accuracies from " + file_path)
            with open(file_path, 'rb') as handle:
                ARI = pickle.load(handle)
            return ARI
        else:
            return {}
        
    def get_scans_and_clusters(self, voxel_selection, subject_id):
        
        # get scans of the selected voxels and appropriate words

        scan_total_clusters = []
        scan_abstract_clusters = []
        scan_concrete_clusters = []
        
        scans = []
        abstract_scans = []
        concrete_scans = []

        for block in self.blocks[subject_id]:
            selected_voxel_activities = []
            all_voxels = [event.scan for event in block.scan_events][0]
            selected_voxel_activities = [all_voxels[voxel_index] for voxel_index in voxel_selection]
            scans.append(selected_voxel_activities)
            if block.concreteness_id == "abstract":
                abstract_scans.append(selected_voxel_activities)
            else:
                concrete_scans.append(selected_voxel_activities)
            
                
        kmeans = KMeans(n_clusters=self.n_total_clusters)
        kmeans.fit(scans)                        
        labels_total = kmeans.labels_
        print(labels_total)
        

        for cluster in range(self.n_total_clusters):
            scan_total_clusters.append([self.words[index] for index in np.where(labels_total == cluster)[0]])
            
        kmeans = KMeans(n_clusters=self.n_concreteness_clusters)
        kmeans.fit(abstract_scans)
        labels_abstract = kmeans.labels_

        
        for cluster in range(self.n_concreteness_clusters):
            scan_abstract_clusters.append([self.abstract_words[index] for index in np.where(labels_abstract == cluster)[0]])
    
        kmeans.fit(concrete_scans)
        labels_concrete = kmeans.labels_
        
        for cluster in range(self.n_concreteness_clusters):                     
            scan_concrete_clusters.append([self.concrete_words[index] for index in np.where(labels_concrete == cluster)[0]])
                
        return scan_total_clusters, scan_abstract_clusters, scan_concrete_clusters, labels_total, labels_abstract, labels_concrete

