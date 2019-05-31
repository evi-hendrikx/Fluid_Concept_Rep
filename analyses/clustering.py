import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from analyses.clustering_abstract import Clustering
import scipy.io
from embeddings.all_embeddings import PereiraEncoder, ImageEncoder, CombiEncoder, RandomEncoder
from select_voxels.selection_methods import SelectROI, SelectStable, SelectSearchlight
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from statistics import mean, stdev



class ClusterROI(Clustering):
    
    def __init__(self, user_dir, paradigm):
        super(ClusterROI, self).__init__(user_dir, paradigm)
        self.selection = "roi"
        
    def cluster_fmri(self):
        
        ARI = self.load_ARI()
        if len(ARI) == len(self.subject_ids):
            return ARI
        
        else:
            self.read_data()
            self.create_embeddding_lists()
            
            selection = SelectROI(self.user_dir)
            voxel_selection = selection.select_voxels()
                
            clusters = {}
            ARI = {}
            for subject_id in voxel_selection.keys():
                ARI[subject_id] = {}
                for area in voxel_selection[subject_id].keys():
                    ARI[subject_id][area] = {}
                    if not area in clusters.keys():
                        clusters[area] = {}
                    clusters[area][subject_id] = {}
                    clusters[area][subject_id]["total"], clusters[area][subject_id]["abstract"], clusters[area][subject_id]["concrete"] = {}, {},  {}
                    clusters[area][subject_id]["total"]["clusters"], clusters[area][subject_id]["abstract"]["clusters"], clusters[area][subject_id]["concrete"]["clusters"], clusters[area][subject_id]["total"]["labels"], clusters[area][subject_id]["abstract"]["labels"], clusters[area][subject_id]["concrete"]["labels"] = self.get_scans_and_clusters(voxel_selection[subject_id][area], subject_id)
    
                    for concreteness in ["total"]: # TODO? ,"abstract","concrete"]:
                        ARI[subject_id][area][concreteness] = {}
                        labels = clusters[area][subject_id][concreteness]["labels"]
                        
                        if concreteness == "total":
                            true_index = self.concrete_index
    
                        ARI[subject_id][area][concreteness]["accuracy"] = adjusted_rand_score(true_index, labels)
                        ARI[subject_id][area][concreteness]["random"] = []
                        for i in range(1000):
                            np.random.shuffle(labels)
                            ARI[subject_id][area][concreteness]["random"].append(adjusted_rand_score(self.concrete_index, labels))
                        
            self.save_clusters(clusters)
            self.save_ARI(ARI)
            
            return ARI
        
    def bar_graph_clusters(self):
        clusters = self.load_clusters()
        
        if not len(clusters) == 6:
            self.cluster_fmri()
        
        else:
            self.read_data()
            self.create_embeddding_lists()
            proportion_clusters = {}
            for area in clusters.keys():
                proportion_clusters[area] = {}
                for subject_id in clusters[area].keys():
                    proportion_clusters[area][subject_id] = {}
                    proportion_clusters[area][subject_id]["abstract_proportion"] = {}
                    proportion_clusters[area][subject_id]["concrete_proportion"] = {}
                    proportion_clusters[area][subject_id]["total_abstracts"] = {}
                    proportion_clusters[area][subject_id]["total_concretes"] = {}
                    cluster1 = clusters[area][subject_id]["total"]["clusters"][0]
                    cluster2 = clusters[area][subject_id]["total"]["clusters"][1]
                    
                    cluster_index = 0
                    for cluster in [cluster1, cluster2]:
                        abstract_word_counter = 0
                        concrete_word_counter = 0
                        for word in cluster:
                            if word in self.abstract_words:
                                abstract_word_counter += 1
                            else:
                                concrete_word_counter += 1    
                        proportion_clusters[area][subject_id]["abstract_proportion"][cluster_index] = abstract_word_counter / len(cluster)
                        proportion_clusters[area][subject_id]["concrete_proportion"][cluster_index] = concrete_word_counter / len(cluster)
                        proportion_clusters[area][subject_id]["total_abstracts"][cluster_index] = abstract_word_counter
                        proportion_clusters[area][subject_id]["total_concretes"][cluster_index] = concrete_word_counter

                        if not abstract_word_counter / len(cluster) + concrete_word_counter / len(cluster) == 1:
                            print("SOMETHING WENT WRONG,, PROPORTIONS DON'T ADD UP TO 1")
                            
                        cluster_index += 1
                        
            # print(proportion_clusters)
            
            for area in clusters.keys():
                for subject in clusters[area].keys():
                    N = 2
                    abstract_prop = []
                    concrete_prop = []
                    for cluster_index in [0, 1]:
                        abstract_prop.append(proportion_clusters[area][subject_id]["abstract_proportion"][cluster_index])
                        concrete_prop.append(1 - proportion_clusters[area][subject_id]["abstract_proportion"][cluster_index])
# =============================================================================
#                     ind = np.arange(N)    # the x locations for the groups
#                     width = 0.35       # the width of the bars: can also be len(x) sequence
#             
#                     plt.bar(ind, abstract_prop, width)
#                     plt.bar(ind, concrete_prop, width,
#                                  bottom=abstract_prop)
#                     
#                     plt.ylabel('proportion')
#                     plt.title('proportion abstract vs. concrete in clusters')
#                     plt.xticks(ind, ('cluster1', 'cluster2'))
#                     
#                     plt.show()
# =============================================================================
                print(area + " mean proportion abstract cluster 1: " + str(mean([proportion_clusters[area][subject_id]["abstract_proportion"][0] for subject_id in proportion_clusters[area].keys()])))
                print(area + " sd proportion abstract cluster 1: " + str(stdev([proportion_clusters[area][subject_id]["abstract_proportion"][0] for subject_id in proportion_clusters[area].keys()])))
                print(area + " mean proportion concrete cluster 1: " + str(mean([proportion_clusters[area][subject_id]["concrete_proportion"][0] for subject_id in proportion_clusters[area].keys()])))
                print(area + " sd proportion concrete cluster 1: " + str(stdev([proportion_clusters[area][subject_id]["concrete_proportion"][0] for subject_id in proportion_clusters[area].keys()])))
                                
                print(area + " mean proportion abstract cluster 2: " + str(mean([proportion_clusters[area][subject_id]["abstract_proportion"][1] for subject_id in proportion_clusters[area].keys()])))
                print(area + " sd proportion abstract cluster 2: " + str(stdev([proportion_clusters[area][subject_id]["abstract_proportion"][1] for subject_id in proportion_clusters[area].keys()])))
                print(area + " mean proportion concrete cluster 2: " + str(mean([proportion_clusters[area][subject_id]["concrete_proportion"][1] for subject_id in proportion_clusters[area].keys()])))
                print(area + " sd proportion concrete cluster 2: " + str(stdev([proportion_clusters[area][subject_id]["concrete_proportion"][1] for subject_id in proportion_clusters[area].keys()])))
                 
                print(area + " mean abstract cluster 1: " + str(mean([proportion_clusters[area][subject_id]["total_abstracts"][0] for subject_id in proportion_clusters[area].keys()])))
                print(area + " sd abstract cluster 1: " + str(stdev([proportion_clusters[area][subject_id]["total_abstracts"][0] for subject_id in proportion_clusters[area].keys()])))
                print(area + " mean concrete cluster 1: " + str(mean([proportion_clusters[area][subject_id]["total_concretes"][0] for subject_id in proportion_clusters[area].keys()])))
                print(area + " sd concrete cluster 1: " + str(stdev([proportion_clusters[area][subject_id]["total_concretes"][0] for subject_id in proportion_clusters[area].keys()])))
                                
                print(area + " mean abstract cluster 2: " + str(mean([proportion_clusters[area][subject_id]["total_abstracts"][1] for subject_id in proportion_clusters[area].keys()])))
                print(area + " sd abstract cluster 2: " + str(stdev([proportion_clusters[area][subject_id]["total_abstracts"][1] for subject_id in proportion_clusters[area].keys()])))
                print(area + " mean concrete cluster 2: " + str(mean([proportion_clusters[area][subject_id]["total_concretes"][1] for subject_id in proportion_clusters[area].keys()])))
                print(area + " sd concrete cluster 2: " + str(stdev([proportion_clusters[area][subject_id]["total_concretes"][1] for subject_id in proportion_clusters[area].keys()])))
                 
            return proportion_clusters
        
                            
        
        
        
    # TODO: get embeddings
    
class ClusterEmbeddings(Clustering):
    
    def __init__(self, user_dir, paradigm):
        super(ClusterEmbeddings, self).__init__(user_dir, paradigm)
            
        
    def cluster_embeddings(self): 
        self.read_data()
        self.create_embeddding_lists()
        
        for embedder in (PereiraEncoder(self.save_embedding_dir), ImageEncoder(self.save_embedding_dir), CombiEncoder(self.save_embedding_dir), RandomEncoder(self.save_embedding_dir)):
            
            # TODO: maybe PCA
            
            # look at clusters that all words form --> are abstract more alike and concrete more alike?
            
            print("EMBEDDER: " + embedder.__class__.__name__)
            
            kmeans = KMeans(n_clusters=self.n_total_clusters)
            kmeans.fit(self.embeddings)                        
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            print("Clusters within total: ")
            
            # print the various clusters that were created
            for cluster in range(self.n_total_clusters):
                print([self.words[index] for index in np.where(labels == cluster)[0]])
                
            # see whether it distinguishes abstract and concrete or not
            print("extra infomation:" )
            compare = np.asarray([labels == self.abstract_index])
            disagreements = [self.words[index] for index in np.where(compare == False)[1]]
            print("labels and concreteness do not agree: " + str(disagreements))
            datafile = scipy.io.loadmat(self.data_dir + "M01/data_180concepts_sentences.mat")
            
            # find the concreteness of disagreements in the MATLAB file
            # criteria right now: > 4.0292 or  < 2.9463
            all_words = [datafile["keyConcept"][index][0][0] for index in range(len(datafile["examples"]))]
            print("Corresponding concretenesses: " + str([datafile["labelsConcreteness"][index_word][0] for index_word in [all_words.index(word) for word in disagreements]]))       
                
            
            
            # look at clusters that form within abstract and concrete
            for embeddings_concreteness in [self.abstract_embs, self.concrete_embs]:
                kmeans = KMeans(n_clusters=self.n_concreteness_clusters)
                kmeans.fit(embeddings_concreteness)
                labels = kmeans.labels_
                
                if embeddings_concreteness == self.abstract_embs:
                    print("Clusters within abstract: ")
                else: 
                    print("Clusters within concrete: ")
                    
                for cluster in range(self.n_concreteness_clusters):
                    if embeddings_concreteness == self.abstract_embs:
                        print([self.abstract_words[index] for index in np.where(labels == cluster)[0]])
                    else:
                        print([self.concrete_words[index] for index in np.where(labels == cluster)[0]])
        
    
class ClusterStable(Clustering):
    
    def __init__(self, user_dir, paradigm):
        super(ClusterStable, self).__init__(user_dir, paradigm)       
        self.selection = "stable"
    
    def cluster_fmri(self):
        
        ARI = self.load_ARI()
        if len(ARI) == len(self.subject_ids):
            return ARI
        else:
            self.read_data()
            self.create_embeddding_lists()
            
            
            selection = SelectStable(self.user_dir, "clustering")
            voxel_selection = selection.select_voxels()
                
            clusters = {}
            ARI = {}
            for subject_id in voxel_selection.keys():
                ARI[subject_id] = {}
                clusters[subject_id] = {}
                clusters[subject_id]["total"], clusters[subject_id]["abstract"], clusters[subject_id]["concrete"] = {}, {},  {}
                clusters[subject_id]["total"]["clusters"], clusters[subject_id]["abstract"]["clusters"], clusters[subject_id]["concrete"]["clusters"], clusters[subject_id]["total"]["labels"], clusters[subject_id]["abstract"]["labels"], clusters[subject_id]["concrete"]["labels"] = self.get_scans_and_clusters(voxel_selection[subject_id], subject_id)
                for concreteness in ["total"]: # TODO ,"abstract","concrete"]:
                    ARI[subject_id][concreteness] = {}
                    labels = clusters[subject_id][concreteness]["labels"]
                    ARI[subject_id][concreteness]["accuracy"] = adjusted_rand_score(self.concrete_index, labels)
                    ARI[subject_id][concreteness]["random"] = []
                    for i in range(1000):
                        np.random.shuffle(labels)
                        ARI[subject_id][concreteness]["random"].append(adjusted_rand_score(self.concrete_index, labels))
                        
            self.save_clusters(clusters)
            self.save_ARI(ARI)
            return ARI
        
    def bar_graph_clusters(self):
        clusters = self.load_clusters()
        
        if not len(clusters) == 16:
            self.cluster_fmri()
        
        else:
            self.read_data()
            self.create_embeddding_lists()
            proportion_clusters = {}
            for subject_id in clusters.keys():
                proportion_clusters[subject_id] = {}
                proportion_clusters[subject_id]["abstract_proportion"] = {}
                proportion_clusters[subject_id]["concrete_proportion"] = {}
                proportion_clusters[subject_id]["total_abstracts"] = {}
                proportion_clusters[subject_id]["total_concretes"] = {}
                cluster1 = clusters[subject_id]["total"]["clusters"][0]
                cluster2 = clusters[subject_id]["total"]["clusters"][1]
                    
                cluster_index = 0
                for cluster in [cluster1, cluster2]:
                    abstract_word_counter = 0
                    concrete_word_counter = 0
                    for word in cluster:
                        if word in self.abstract_words:
                            abstract_word_counter += 1
                        else:
                            concrete_word_counter += 1    
                    proportion_clusters[subject_id]["abstract_proportion"][cluster_index] = abstract_word_counter / len(cluster)
                    proportion_clusters[subject_id]["concrete_proportion"][cluster_index] = concrete_word_counter / len(cluster)
                    proportion_clusters[subject_id]["total_abstracts"][cluster_index] = abstract_word_counter
                    proportion_clusters[subject_id]["total_concretes"][cluster_index] = concrete_word_counter

                    if not abstract_word_counter / len(cluster) + concrete_word_counter / len(cluster) == 1:
                        print("SOMETHING WENT WRONG,, PROPORTIONS DON'T ADD UP TO 1")
                        
                    cluster_index += 1
                        
            # print(proportion_clusters)
            
            for subject in clusters.keys():
                N = 2
                abstract_prop = []
                concrete_prop = []
                for cluster_index in [0, 1]:
                    abstract_prop.append(proportion_clusters[subject_id]["abstract_proportion"][cluster_index])
                    concrete_prop.append(1 - proportion_clusters[subject_id]["abstract_proportion"][cluster_index])
# =============================================================================
#                     ind = np.arange(N)    # the x locations for the groups
#                     width = 0.35       # the width of the bars: can also be len(x) sequence
#             
#                     plt.bar(ind, abstract_prop, width)
#                     plt.bar(ind, concrete_prop, width,
#                                  bottom=abstract_prop)
#                     
#                     plt.ylabel('proportion')
#                     plt.title('proportion abstract vs. concrete in clusters')
#                     plt.xticks(ind, ('cluster1', 'cluster2'))
#                     
#                     plt.show()
# =============================================================================
                print(" mean proportion abstract cluster 1: " + str(mean([proportion_clusters[subject_id]["abstract_proportion"][0] for subject_id in proportion_clusters.keys()])))
                print(" sd proportion abstract cluster 1: " + str(stdev([proportion_clusters[subject_id]["abstract_proportion"][0] for subject_id in proportion_clusters.keys()])))
                #print(area + " mean proportion concrete cluster 1: " + str(mean([proportion_clusters[area][subject_id]["concrete_proportion"][0] for subject_id in proportion_clusters[area].keys()])))
                #print(area + " sd proportion concrete cluster 1: " + str(stdev([proportion_clusters[area][subject_id]["concrete_proportion"][0] for subject_id in proportion_clusters[area].keys()])))
                                
                print(" mean proportion abstract cluster 2: " + str(mean([proportion_clusters[subject_id]["abstract_proportion"][1] for subject_id in proportion_clusters.keys()])))
                print(" sd proportion abstract cluster 2: " + str(stdev([proportion_clusters[subject_id]["abstract_proportion"][1] for subject_id in proportion_clusters.keys()])))
                #print(area + " mean proportion concrete cluster 2: " + str(mean([proportion_clusters[area][subject_id]["concrete_proportion"][1] for subject_id in proportion_clusters[area].keys()])))
                #print(area + " sd proportion concrete cluster 2: " + str(stdev([proportion_clusters[area][subject_id]["concrete_proportion"][1] for subject_id in proportion_clusters[area].keys()])))
                 
# =============================================================================
#                 print(area + " mean abstract cluster 1: " + str(mean([proportion_clusters[area][subject_id]["total_abstracts"][0] for subject_id in proportion_clusters[area].keys()])))
#                 print(area + " sd abstract cluster 1: " + str(stdev([proportion_clusters[area][subject_id]["total_abstracts"][0] for subject_id in proportion_clusters[area].keys()])))
#                 print(area + " mean concrete cluster 1: " + str(mean([proportion_clusters[area][subject_id]["total_concretes"][0] for subject_id in proportion_clusters[area].keys()])))
#                 print(area + " sd concrete cluster 1: " + str(stdev([proportion_clusters[area][subject_id]["total_concretes"][0] for subject_id in proportion_clusters[area].keys()])))
#                                 
#                 print(area + " mean abstract cluster 2: " + str(mean([proportion_clusters[area][subject_id]["total_abstracts"][1] for subject_id in proportion_clusters[area].keys()])))
#                 print(area + " sd abstract cluster 2: " + str(stdev([proportion_clusters[area][subject_id]["total_abstracts"][1] for subject_id in proportion_clusters[area].keys()])))
#                 print(area + " mean concrete cluster 2: " + str(mean([proportion_clusters[area][subject_id]["total_concretes"][1] for subject_id in proportion_clusters[area].keys()])))
#                 print(area + " sd concrete cluster 2: " + str(stdev([proportion_clusters[area][subject_id]["total_concretes"][1] for subject_id in proportion_clusters[area].keys()])))
#                  
# =============================================================================
            return proportion_clusters
        
        
class ClusterSearchlight(Clustering):

    def __init__(self, user_dir, paradigm):
        super(ClusterSearchlight, self).__init__(user_dir, paradigm)
        self.selection = "searchlight"
        
        
    def cluster_fmri(self):
        
        ARI = self.load_ARI()
        if len(ARI) == len(self.subject_ids):
            return ARI
            
        else:
            self.read_data()
            self.create_embeddding_lists()
            selection = SelectSearchlight(self.user_dir)
            voxel_selection = selection.select_voxels()
            clusters = {}
            ARI = {}
            
            for subject_id in voxel_selection.keys():
                print(subject_id)
                ARI[subject_id] = {}
                clusters[subject_id] = {}
                
                for concreteness in ["total","abstract","concrete"]:
                    ARI[subject_id][concreteness] = {}
                
                for score in ["accuracy", "random"]:  
                    print(subject_id + score)
                    clusters[subject_id][score] = {}                 
                    clusters[subject_id][score]["total"], clusters[subject_id][score]["abstract"], clusters[subject_id][score]["concrete"] = {}, {}, {}
                    clusters[subject_id][score]["total"]["clusters"], clusters[subject_id][score]["abstract"]["clusters"], clusters[subject_id][score]["concrete"]["clusters"], clusters[subject_id][score]["total"]["labels"], clusters[subject_id][score]["abstract"]["labels"], clusters[subject_id][score]["concrete"]["labels"] = [], [], [], [], [], []
                    
                    for concreteness in ["total","abstract","concrete"]:
                        ARI[subject_id][concreteness][score] = []
                    
                    if score == "random":
                        voxel_selection[subject_id] = self.select_three_voxels(voxel_selection[subject_id], subject_id)
                        
                    voxel_index = 0
                    for voxel in voxel_selection[subject_id]:
                        if score == "accuracy":
                            if voxel_index % 1000 == 0:
                                print(str(voxel_index) + "/" + str(len(voxel_selection[subject_id])))
                        else:
                            print(str(voxel_index) + "/3")

                        total_clusters, abstract_clusters, concrete_clusters, total_labels, abstract_labels, concrete_labels = self.get_scans_and_clusters(voxel, subject_id)
                        clusters[subject_id][score]["total"]["clusters"].append(total_clusters); clusters[subject_id][score]["abstract"]["clusters"].append(abstract_clusters); clusters[subject_id][score]["concrete"]["clusters"].append(concrete_clusters)
                        clusters[subject_id][score]["total"]["labels"].append(total_labels); clusters[subject_id][score]["abstract"]["labels"].append(abstract_labels); clusters[subject_id][score]["concrete"]["labels"].append(concrete_labels)
                        
                        for concreteness in ["total"]:
                            labels = clusters[subject_id][score][concreteness]["labels"][voxel_index]
                            if score == "accuracy":
                                ARI[subject_id][concreteness][score].append(adjusted_rand_score(self.concrete_index, labels))
                            else:
                                ARI[subject_id][concreteness][score].append([])
                                for i in range(10):
                                    if i% 1000 == 0:
                                        print(i)
                                    np.random.shuffle(labels)
                                    ARI[subject_id][concreteness][score][voxel_index].append(adjusted_rand_score(self.concrete_index, labels))
                            voxel_index += 1
                        if voxel_index == 10:
                            break
                for concreteness in ARI[subject_id].keys():
                    ARI[subject_id][concreteness]["random"] = np.mean(ARI[subject_id][concreteness]["random"], axis = 0)
    
            self.save_clusters(clusters)
            self.save_ARI(ARI)
            
            return ARI
            
    def select_three_voxels(self, voxel_selection, subject_id):
        selected_voxels = []
        
        # select areas from matlab file that fall in these lobes, selection is based on the name of the lobe in the region of AAL atlas
        # this is the same for all subjects
        frontal = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]; frontal_done = False
        occipital = [49, 50, 51, 52, 53, 54]; occipital_done = False
        temporal = [81, 82, 83, 84, 85, 86, 87, 88, 89, 90]; temporal_done = False
        
        # regions remain constant over all paradigms, so I arbitrarily chose pictures
        datafile = scipy.io.loadmat(self.data_dir + subject_id + "/data_180concepts_pictures.mat")
        voxel_to_region_mapping = datafile["meta"]["roiMultimask"][0][0][0][0]
        for voxels in np.random.permutation(voxel_selection):
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
            
        

