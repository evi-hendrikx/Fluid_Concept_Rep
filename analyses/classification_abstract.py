from os import listdir
import pickle
from sklearn.svm import SVC
from read_dataset.read_pereira import PereiraReader
import numpy as np
import os


class Classification(object):
    def __init__(self, user_dir, paradigm):
        
        self.paradigm = paradigm
        
        self.user_dir = user_dir
        self.data_dir = "/datastore/pereira_data/"
        self.subject_ids = [file for file in listdir(self.data_dir)]
        
        # where the analysis will be stored
        self.save_dir = user_dir + "analyses/already_analysed/"

        # pay attention! this should be the same as the "groups" in the stable voxel selection
        self.amount_of_folds = 11
        
        
    def load_classifications(self, voxel_selection):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        accuracy_file = self.save_dir + "classification_" + paradigms[paradigm_index] + "_" + voxel_selection + ".pickle"
        
        if os.path.isfile(accuracy_file):
            print("Loading accuracies from " + accuracy_file)
            with open(accuracy_file, 'rb') as handle:
                accuracies = pickle.load(handle)
            return accuracies
        else:
            return {}
        
        
    def save_classifications(self, voxel_selection, accuracies, subject_ids = "", intermediate_save = False):
        paradigms = ["sentences", "pictures", "wordclouds", "average"]
        paradigm_index = self.paradigm - 1
        
        # If I save intermediately, I want the filename to show all present participants
        ids = ""
        if not subject_ids == "":
            for subject in subject_ids:
                ids = ids + "_" + subject
        
        accuracy_file = self.save_dir + "classification_" + paradigms[paradigm_index] + "_" + voxel_selection + ids + ".pickle"
        
        # if I am saving intermediately, remove old file so I don't save the same information twice
        if intermediate_save == True and len(subject_ids) > 1:
            len_last_id = len(subject_ids[-1])
            old_ids = ids[: - (len_last_id + 1)]
            old_accuracy_file = accuracy_file.replace(ids, old_ids)
            print("Remove file: " + old_accuracy_file)
            os.remove(old_accuracy_file)
        
        # the last save should be able to be automatically loaded
        if subject_ids == list(self.blocks.keys()):
            accuracy_file = accuracy_file.replace(ids, "")
        
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
        
        # all participants get the same words (and concreteness), therefore I arbitrarily chose M01
        self.words = []
        self.concreteness = []
        for block in self.blocks["M01"]:
            self.words.append([word for word in block.sentences][0][0])
            concreteness_id = block.concreteness_id
            self.concreteness.append(concreteness_id)      
            
          
    # I didn't run the usual svc (which requires waaay less code), since I wanted to make sure the right voxels of the stable voxels were
    # selected with the right training sets.
    # Also, it made it easier to implement a random distribution in which only the training labels for concreteness are shuffled
    # A lot of studies run their tests against a chance of 0.5, but I don't think this is fair since 
        # your classifier might be biased
        # the labels don't have an exact 50/50 ratio (but 63/69)
    def run_svc(self, scans, random_concreteness = False):
        if len(scans) > 6000:
            svc = SVC(kernel = "linear")
        else:
            svc = SVC(gamma = "scale")
        
        index_first_word = 0
        predictions = []
        actual_concretenesses = []
        
        input_scans = scans
        
        # create folds and run classification
        for fold in range(0, self.amount_of_folds):
            index_last_word = int(index_first_word + len(self.words) / self.amount_of_folds - 1)
            
            # if this is the stable voxel selection, make sure to use appropriate selected voxels
            # (stable voxels were determined over the training set; key represents first and last word of the test set)
            if len(input_scans) == 11: 
                key = self.words[index_first_word] + "_" + self.words[index_last_word]
                scans = input_scans[key]
            
            # select training and test groups for each fold
            scan_slice1_1 = scans[0:index_first_word]
            scan_slice1_2 = scans[index_last_word + 1:]
            concrete_slice1_1 = self.concreteness[0:index_first_word]
            concrete_slice1_2 = self.concreteness[index_last_word + 1:] 
            
            train_scans = []
            train_concreteness = []
            for slice in [scan_slice1_1, scan_slice1_2]:
                if len(slice) > 0:
                    train_scans.extend(slice)
            for slice in [concrete_slice1_1, concrete_slice1_2]:
                if len(slice) > 0:
                    train_concreteness.extend(slice)
            
            # shuffle concreteness labels (creates different randomization every time)
            if random_concreteness == True:
                np.random.shuffle(train_concreteness)
            
            test_scans = scans[index_first_word:index_last_word + 1]
            test_concreteness = self.concreteness[index_first_word:index_last_word + 1] 
            actual_concretenesses.extend(test_concreteness)
                         
            # caluculate accuracies of prediction
            svc.fit(train_scans, train_concreteness)
            predictions.extend(svc.predict(test_scans))            
            index_first_word = int(index_last_word + 1)
        
        # Lisa adviced me to calculate the accuracy in the end instead of averaging over folds
        # (I think it gives similar answers now since all my folds are equally large)
        accuracy = (np.array(predictions) == np.array(actual_concretenesses)).sum() / float(len(actual_concretenesses))
        
        return accuracy
    
    
    def random_generator(self, scans):
        print("selecting random accuracies")
        random_accuracies = []

        if self.selection == "searchlight":
            for i in range(0, 106000):
                random_accuracy = self.run(scans, random_concreteness = True)
                random_accuracies.append(random_accuracy)
                if i % 1000 == 0:
                    print(str(i) + " random accuracies collected")
            if self.reduced_random == False:
                unique_elements, count_elements = np.unique(np.asarray(random_accuracies),return_counts = True)
                random_distribution = np.asarray((unique_elements, count_elements))
            else:
                random_distribution = random_accuracies
        else:
            for i in range(0,1000):
                random_accuracy = self.run_svc(scans, random_concreteness = True)
                random_accuracies.append(random_accuracy)
                if i % 100 == 0:
                    print(str(i) + " random accuracies collected")
        # instead of saving all values, save their frequencies
            unique_elements, counts_elements = np.unique(np.asarray(random_accuracies), return_counts=True)
            random_distribution = np.asarray((unique_elements, counts_elements))
        
        return random_distribution
