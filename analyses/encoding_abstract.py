from os import listdir
import pickle
from read_dataset.read_pereira import PereiraReader
import numpy as np
import os
from mapping_models.ridge_regression_mapper import RegressionMapper
import scipy as sp
import scipy.io

class Encoding(object):
    def __init__(self, user_dir, paradigm):        
        self.paradigm = paradigm
        
        self.user_dir = user_dir
        self.data_dir = "DATADIR"
        self.subject_ids = [file for file in listdir(self.data_dir)]
        
        # where the analysis will be stored
        self.save_dir = user_dir + "/analyses/already_analysed/"

        # pay attention! this should be the same as in the stable voxel selection
        self.amount_of_folds = 11
        
        self.embeddings = ["linguistic", "non-linguistic", "combi", "random"]
        
        # where embeddings are stored
        self.save_embedding_dir = user_dir + "/embeddings/already_embedded/"
        
        
    def load_encodings(self, voxel_selection):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        accuracy_file = self.save_dir + "encoding_" + paradigms[paradigm_index] + "_" + voxel_selection + ".pickle"
        
        if os.path.isfile(accuracy_file):
            print("Loading accuracies from " + accuracy_file)
            with open(accuracy_file, 'rb') as handle:
                accuracies = pickle.load(handle)
            return accuracies
        else:
            return {}
        
        
    def intermediate_save_encodings(self, voxel_selection, accuracies, embedding, subject_id = ""):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        
        # If I save intermediately, I want the filename to show all present participants
        embs = ""
        index_embedding = self.embeddings.index(embedding)
        embeddings_so_far = self.embeddings[:index_embedding + 1]
        for embedding in embeddings_so_far:
            embs = embs + "_" + embedding

        accuracy_file = self.save_dir + "encoding_" + paradigms[paradigm_index] + "_" + voxel_selection + embs + ".pickle"

        # if I am saving intermediately, remove old file so I don't save the same information twice
        if not index_embedding == 0:
            len_last_emb = len(embeddings_so_far[-1])
            old_embs = embs[: - (len_last_emb + 1)]
            old_accuracy_file = accuracy_file.replace(embs, old_embs)
            print("Remove file: " + old_accuracy_file)
            os.remove(old_accuracy_file)

        # the last save should be able to be automatically loaded
        if not len(self.embeddings) == 1:
            if index_embedding == len(self.embeddings) - 1:
                accuracy_file = accuracy_file.replace(embs, "")
        
        os.makedirs(os.path.dirname(accuracy_file), exist_ok=True)
        with open(accuracy_file, 'wb') as handle:
            pickle.dump(accuracies, handle)
        print("Accuracies stored in file: " + accuracy_file)   
        
        
    def read_data(self):
        # get information of all participants
        print("reader start for paradigm " + str(self.paradigm))
        pereira_reader = PereiraReader(self.data_dir, paradigm = self.paradigm)
        pereira_data = pereira_reader.read_all_events()
        blocks, xyz_voxels = pereira_data
        self.blocks = blocks
        print("reader done...")
        
        # all participants get the same words (and concreteness) in the same order, therefore I arbitrarily chose M01
        self.words = []
        self.concreteness = []
        for block in self.blocks["M01"]:
            self.words.append([word for word in block.sentences][0][0])
            concreteness_id = block.concreteness_id
            self.concreteness.append(concreteness_id)      
            
          
    # I didn't run the usual svc (which requires waaay less code), since I wanted to make sure the right voxels of the stable voxels were
    # selected with the right training sets.
    def run_encoding(self, scans, embeddings, embedding):
        
        accuracy_folds = []
        correct = []
        correct_abstract = []
        correct_concrete = []
        total_abstract = []
        total_concrete = []

        input_scans = scans

        # create folds and run classification
        fold_info = {}
        index_first_word = 0
        for fold in range(0, self.amount_of_folds):
            index_last_word = int(index_first_word + len(self.words) / self.amount_of_folds - 1)
            key = self.words[index_first_word] + "_" + self.words[index_last_word]
        
            # if this is the stable voxel selection, make sure to use appropriate selected voxels
            # (stable voxels were determined over the training set; key represents first and last word of the test set)
            if len(input_scans) == self.amount_of_folds: 
                scans = input_scans[key]
            
            self.scans = scans
        
            # select training and test groups for each fold
            scan_slice1_1 = self.scans[0:index_first_word]
            scan_slice1_2 = self.scans[index_last_word + 1:]
            words_slice1_1 = self.words[0:index_first_word]
            words_slice1_2 = self.words[index_last_word + 1:]

            train_scans = []
            train_words = []
            for slice in [scan_slice1_1, scan_slice1_2]:
                if len(slice) > 0:
                    train_scans.extend(slice)
            for slice in [words_slice1_1, words_slice1_2]:
                if len(slice) > 0:
                    train_words.extend(slice)
                
            # translate words into embeddings
            train_embeddings = []
            for word in train_words:
                train_embeddings.append(embeddings[word])
         
            fold_info["train_embeddings"] = train_embeddings
            fold_info["train_scans"] = train_scans
           
            test_scans = self.scans[index_first_word:index_last_word + 1]
            test_words = self.words[index_first_word:index_last_word + 1]
            test_concreteness = self.concreteness[index_first_word:index_last_word + 1]
            
            # translate words into embeddings
            test_embeddings = []
            for word in test_words:
                test_embeddings.append(embeddings[word])
         
            fold_info["test_scans"] = test_scans
            fold_info["test_embeddings"] = test_embeddings
          
            # form predictive scan
            cor_pred, cor_abs_pred, cor_con_pred, all_abs_pred, all_con_pred = self.make_predictions(fold_info, test_concreteness, test_words)
            correct.append(cor_pred)
            correct_abstract.append(cor_abs_pred)
            correct_concrete.append(cor_con_pred)
            total_abstract.append(all_abs_pred)
            total_concrete.append(all_con_pred)

            index_first_word = int(index_last_word + 1)
      
        total_predictions = len(self.words) * (len(self.words) - 1)
        total_accuracy = sum(correct)/total_predictions
        abstract_accuracy = sum(correct_abstract)/sum(total_abstract)
        concrete_accuracy = sum(correct_concrete)/sum(total_concrete)
        
        return total_accuracy, abstract_accuracy, concrete_accuracy
    
    def make_predictions(self, fold_info, test_concreteness, test_words):
        
         # predict
         mapper = RegressionMapper(alpha=1.0)
         mapper.train(fold_info["train_embeddings"], fold_info["train_scans"])
         predictions = mapper.map(inputs=fold_info["test_embeddings"])
         
         # determine accuracy predictions by comparing correlations
         correct_predictions = 0
         correct_abstract_predictions = 0
         correct_concrete_predictions = 0
         all_abstract_predictions = 0
         all_concrete_predictions = 0
         for word1 in range(len(test_words)):
             if test_concreteness[word1] == "abstract":
                 all_abstract_predictions +=1 * (len(self.words) - 1)
             elif test_concreteness[word1] == "concrete":
                 all_concrete_predictions +=1 * (len(self.words) - 1)
                 
             actual_correlation = sp.spatial.distance.cosine(predictions[word1], fold_info["test_scans"][word1]) 
             for word2 in range(len(self.words)):
                 if test_words[word1] == self.words[word2]:
                     continue
                 other_correlation = sp.spatial.distance.cosine(predictions[word1], self.scans[word2])
                 if actual_correlation < other_correlation:
                     correct_predictions += 1
                     if test_concreteness[word1] == "abstract":
                         correct_abstract_predictions  += 1
                     elif test_concreteness[word1] == "concrete":
                         correct_concrete_predictions += 1
         
         # Induce error if something went wrong
         if not correct_predictions == correct_abstract_predictions + correct_concrete_predictions:
             print("Something is wrong, abstract and concrete predictions don't add up to the total")
             all_abstract_predictions = None

         return correct_predictions, correct_abstract_predictions, correct_concrete_predictions, all_abstract_predictions, all_concrete_predictions
