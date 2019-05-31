from result_analyses.classification import ClassificationResults
from result_analyses.encoding_rois_separate import EncodingResults
from result_analyses.RSA import RSAResults
from result_analyses.clustering import ClusteringResults
from analyses.clustering_abstract import Clustering
from analyses.clustering import ClusterROI, ClusterStable
import pickle 


user_dir = "USERDIR"

        
results = ClassificationResults(user_dir)
results.create_bar_plots(paradigm)
for paradigm in [1,2,3]:
    for voxel_selection in ["searchlight"]:
        brain_map = results.create_map_ranks(voxel_selection, paradigm)


for results in (EncodingResults(user_dir), RSAResults(user_dir)):
    results.violin_plot_mix()
 

for paradigm in [1,2,3]:
     results = ClusterStable(user_dir, paradigm)
     # results.create_bar_plots(paradigm)
     # results.create_map(paradigm)
     results.bar_graph_clusters()
 
    
    
