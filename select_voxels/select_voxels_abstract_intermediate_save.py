import os
import pickle


class SelectVoxels(object):
    def __init__(self, user_dir):
        self.user_dir = user_dir
        self.data_dir =  "/datastore/pereira_data/"
        self.save_dir = user_dir + "select_voxels/already_selected/"
        self.subject_ids = [file for file in os.listdir(self.data_dir)]

    def load_voxel_selection(self, selection_method, analysis = ""):
        # clustering and RSA both employ stable voxels over the entire set of words
        if analysis == "RSA"  or analysis =="clustering":
            analysis = "_clustering_RSA"
        elif analysis == "classification" or analysis == "encoding":
            analysis = "_classification_encoding"
        
        # check if file exists and return its content
        file_path = self.save_dir + selection_method + analysis + "_voxels_all.pickle"
        if os.path.isfile(file_path):
            print("Loading voxel selections from " + file_path)
            with open(file_path, 'rb') as handle:
                voxel_selections = pickle.load(handle)
            return voxel_selections
        else:
            return {}


    def save_voxel_selection(self, indices, selection_method, analysis = "", subject_id = "", intermediate_save = False):
        # clustering and RSA both employ stable voxels over the entire set of words
        if analysis == "RSA"  or analysis =="clustering":
            analysis = "_clustering_RSA"
        elif analysis == "classification" or analysis == "encoding":
            analysis = "_classification_encoding"
            
        # save the file
        if intermediate_save == True:
            file_path = self.save_dir + selection_method + analysis + "_voxels_" + subject_id + ".pickle" 
        else:      
            file_path = self.save_dir + selection_method + analysis + "_voxels_all.pickle"
            for subject_id in self.subject_ids:
                remove_file = self.save_dir + selection_method + analysis + "_voxels_" + subject_id + ".pickle" 
                if os.path.isfile(remove_file):
                    os.remove(remove_file)
       
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as handle:
            pickle.dump(indices, handle)
        
        print("Voxel selections stored in file: " + file_path)

            
        
