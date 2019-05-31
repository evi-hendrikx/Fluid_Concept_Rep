from analyses.classification import ClassifyStable, ClassifyROI, ClassifySearchlight
from analyses.encoding import EncodingStable, EncodingROI, EncodingSearchlight

user_dir = "/home/USERDIR/"

# I chose index 1, 2, or 3, because these are also used in the reader.
# They correspond to respectively "sentences", "pictures", "wordclouds"
for classification in (ClassifyROI(user_dir, paradigm), ClassifyStable(user_dir, paradigm), ClassifySearchlight(user_dir, paradigm):
    accuracies = classification.classify()
    print(len(accuracies))
    print(len(accuracies["accuracy"]))
    
  
for paradigm in [1,2,3]:
    for encoding in (EncodingROI(user_dir, paradigm), EncodingStable(user_dir, paradigm)):
        accuracies = encoding.encode()
        print(accuracies.keys())
                       
                       
for paradigm in [1,2,3]:
    for clustering in (ClusterROI(user_dir, paradigm), ClusterStable(user_dir, paradigm)):
        clusters = clustering.cluster_fmri()
        
for paradigm in [1,2,3]:
    RSA = RSA_ROI_Stable(user_dir, paradigm)
    correlations = RSA.run_RSA()
    print(len(correlations))
