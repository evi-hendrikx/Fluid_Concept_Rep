import os
import pickle

class TextEncoder(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
        # images can be found on https://osf.io/crwz7/
        self.img_dir = "/" #TODO: right now I ran this on my own computer and just added the embeddings in already embedded
        self.corpus_dir = "/" #TODO: right now I ran this on my own computer and just added the embeddings in already embedded

    def load_embeddings(self, embedding_file):
        if os.path.isfile(embedding_file):
            print("Loading embeddings from " + embedding_file)
            with open(embedding_file, 'rb') as handle:
                embeddings = pickle.load(handle)
            return embeddings
        else:
            return {}


    def save_embeddings(self, embedding_file, embeddings):
        os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
        with open(embedding_file, 'wb') as handle:
            pickle.dump(embeddings, handle)
        print("Embeddings stored in file: " + embedding_file)
