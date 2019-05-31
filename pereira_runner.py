from analyses.classification import ClassifyStable, ClassifyROI, ClassifySearchlight
from analyses.encoding import EncodingStable, EncodingROI, EncodingSearchlight

user_dir = "/home/10978089/"

# I chose index 1, 2, or 3, because these are also used in the reader.
# They correspond to respectively "sentences", "pictures", "wordclouds"
for classification in (ClassifyROI(user_dir, paradigm), ClassifyStable(user_dir, paradigm), ClassifySearchlight(user_dir, paradigm):
    accuracies = classification.classify()
    print(len(accuracies))
    print(len(accuracies["accuracy"]))
    
  
for paradigm in [1,2,3]:
    for encoding in (EncodingROI(user_dir, paradigm), EncodingStable(user_dir, paradigm), EncodingSearchlight(user_dir, paradigm)):
        accuracies = encoding.encode()
        print(accuracies.keys())


