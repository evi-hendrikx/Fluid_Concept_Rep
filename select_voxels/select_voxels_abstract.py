from os import path, listdir, makedirs
import pickle


class SelectVoxels(object):
    def __init__(self, user_dir):
        self.user_dir = user_dir
        self.data_dir = "/datastore/pereira_data/"
        self.save_dir = user_dir + "select_voxels/already_selected/"
        self.subject_ids = [file for file in listdir(self.data_dir)]

    def load_voxel_selection(self, selection_method, analysis = ""):
        # clustering and RSA both employ stable voxels over the entire set of words
        if analysis == "RSA"  or analysis =="clustering":
            analysis = "_clustering_RSA"
        elif analysis == "classification" or analysis == "encoding":
            analysis = "_classification_encoding"
        
        # check if file exists and return its content
        file_path = self.save_dir + selection_method + analysis + "_voxels_all.pickle"
        if path.isfile(file_path):
            print("Loading voxel selections from " + file_path)
            with open(file_path, 'rb') as handle:
                voxel_selections = pickle.load(handle)
            return voxel_selections
        else:
            return {}


    def save_voxel_selection(self, indices, selection_method, analysis = ""):
        # clustering and RSA both employ stable voxels over the entire set of words
        if analysis == "RSA"  or analysis =="clustering":
            analysis = "_clustering_RSA"
        elif analysis == "classification" or analysis == "encoding":
            analysis = "_classification_encoding"
        
        # save the file
        file_path = self.save_dir + selection_method + analysis + "_voxels_all.pickle"          
        makedirs(path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as handle:
            pickle.dump(indices, handle)
        
        print("Voxel selections stored in file: " + file_path)

            
        
