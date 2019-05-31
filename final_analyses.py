from result_analyses.classification import ClassificationResults
from result_analyses.encoding_rois_separate import EncodingResults
from result_analyses.RSA import RSAResults
from result_analyses.clustering import ClusteringResults
from analyses.clustering_abstract import Clustering
from analyses.clustering import ClusterROI, ClusterStable
import pickle 


user_dir = "/home/evihendrikx/Documents/brain-lang-master/"

        
        
        
# =============================================================================
# # =============================================================================
# # results = EncodingResults(user_dir)
# # # results.make_table()
# # results.violin_plot_mix()
# # =============================================================================
# 
# 
# # =============================================================================
# # results = RSAResults(user_dir)
# # results.violin_plot_mix()
# # =============================================================================
# 
# for paradigm in [1]: #,2,3]:
#     brain_map = results.create_map("searchlight", paradigm)
# =============================================================================


results = EncodingResults(user_dir)
#results = ClassificationResults(user_dir)

results = RSAResults(user_dir)

for paradigm in [3]:
    paradigm_index = paradigm - 1
    paradigms = ["sentences", "pictures", "wordclouds"]
    
# =============================================================================
#     for voxel_selection in ["searchlight"]:
#         brain_map = results.create_map_ranks(voxel_selection, paradigm)
# =============================================================================
    
    results.violin_plot_mix()
        
# =============================================================================
#     file_path = "/home/evihendrikx/Documents/brain-lang-master/pereira_code/result_analyses/final_results/accuracies_per_area_classification_" + paradigms[paradigm_index] + ".pickle"
# 
#     with open (file_path, 'rb') as handle:
#         searchlight = pickle.load(handle)
#     print(searchlight.keys())
# =============================================================================
    
# =============================================================================
#     #score = results.create_map("searchlight", paradigm)
#     file = "/home/evihendrikx/Documents/brain-lang-master/pereira_code/result_analyses/final_results/ranking_classification_wordclouds.pickle"
# 
#     with open (file, 'rb') as handle:
#         score = pickle.load(handle)
# 
#     results.map_brain(score, "searchlight", "wordclouds")
# =============================================================================
    
    



    #results.create_bar_plots(paradigm)

# =============================================================================
# results = RSA(user_dir)
# results.create_bar_plots(3)
# =============================================================================


# =============================================================================
# for paradigm in [1,2,3]:
#     results = ClusterStable(user_dir, paradigm)
#     # results.create_bar_plots(paradigm)
#     # results.create_map(paradigm)
#     results.bar_graph_clusters()
# =============================================================================
 
    
    
