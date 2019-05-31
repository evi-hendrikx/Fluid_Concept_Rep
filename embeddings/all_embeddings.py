import numpy as np
from embeddings.abstract_text_encoder import TextEncoder
import cv2
from keras.applications.resnet50 import ResNet50
from os import listdir
import scipy
from sklearn.decomposition import PCA

# This class generates random embeddings.
# Every word is assigned a random (but fixed) vector of the requested dimensionality.
class RandomEncoder(TextEncoder):
    def __init__(self, save_dir, dimensions=2048):
        super(RandomEncoder, self).__init__(save_dir)
        self.dimensions = dimensions
        self.dict = {}
        
        # Set the seed to make experiments reproducible.
        self.seed = 5
        np.random.seed(self.seed)

    def get_embeddings(self,  words):
        print("Creating random embeddings")
        for word in words:
            embedding = np.random.rand(self.dimensions)
            self.dict[word] = embedding

        return self.dict
    
    
# This class generates Glove embeddings (like Pereira uses)
class PereiraEncoder(TextEncoder):

    def __init__(self, save_dir):
        super(PereiraEncoder, self).__init__(save_dir)
        
        
    def get_embeddings(self, words):
        
        embedding_file = self.save_dir + "pereira_embeddings.pickle"
        pereira_embeddings = self.load_embeddings(embedding_file)

        # there is no such file, create one
        if not (len(pereira_embeddings) == len(words)):
            
            # pick the dimensions of the wanted vectors (50, 100, 200, or 300)
            dimension_vectors = 300
            corpus_dir = self.corpus_dir 
            
            # collect the required word embeddings
            with open(corpus_dir + "glove.42B." + str(dimension_vectors) + "d.txt", 'rb') as f:
                for l in f:
                    line = l.decode().split()
                    word_glove = line[0]
                    if word_glove in words:
                        pereira_embeddings[word_glove] = np.array([float(val) for val in line[1:]])
            
            # there might be words not stored in the glove embeddings, we need to know 
            # It seems that required words are accounted for in the 42B file
            for word in words:
                if word not in pereira_embeddings:
                    print("There is no Glove vector for: " + word)
                    pereira_embeddings[word] = []
                        
            self.save_embeddings(embedding_file, pereira_embeddings)
             
        return pereira_embeddings



# This class generates image embeddings from the pretrained ResNet
class ImageEncoder(TextEncoder):

    def __init__(self, save_dir):
        super(ImageEncoder, self).__init__(save_dir)
        
        
    def get_embeddings(self, words):
        
        embedding_file = self.save_dir + "mean_image_embeddings.pickle"
        mean_image_embeddings = self.load_embeddings(embedding_file)

        # there is no such file, create one
        if not (len(mean_image_embeddings) == len(words)):
            print("Collecting image embeddings from images")
            resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            all_image_embeddings = {}
            
            # collect image embeddings for the 6 used pictures per concept
            for word in words:
                folder_name = word.capitalize() 
                img_dir = self.img_dir + folder_name + "/"
                embeddings_for_this_word = []
                for file in listdir(img_dir):
                    img = cv2.imread(img_dir + file) 
                    img = scipy.misc.imresize(img, 224.0 / img.shape[0])
                    img = img.reshape((1,) + img.shape)
                   
                    embeddings_for_this_word.append(resnet.predict(img)[0])
                    embeddings_for_this_word.append([])
                
                # I want to save all 6 separate embeddings and an embedding representing the mean of these 6
                all_image_embeddings[word] = embeddings_for_this_word
                
                # compute average and reduce its dimension to 300 (like linguistic embeddings)
                mean_image_embeddings[word] = np.mean(embeddings_for_this_word, axis = 0)
                print("Image embedding for " + word + " collected")
            
            self.save_embeddings(embedding_file.replace("mean_image_embeddings.pickle", "all_image_embeddings.pickle"), all_image_embeddings)
            self.save_embeddings(embedding_file, mean_image_embeddings)
            print("Image embeddings stored")

        return mean_image_embeddings
    
    

# This class generates a combined image and text embedding
class CombiEncoder(TextEncoder):

    def __init__(self, save_dir):
        super(CombiEncoder, self).__init__(save_dir)
        
    def get_embeddings(self, words):
        
        embedding_file = self.save_dir + "combi_embeddings.pickle"
        combi_embeddings = self.load_embeddings(embedding_file)

        # there is no such file, create one
        if not (len(combi_embeddings) == len(words)):
            print("Combining image and linguistic embeddings")
            image_encoder = ImageEncoder(self.save_dir)
            image_embeddings = image_encoder.get_embeddings(words)
            pereira_encoder = PereiraEncoder(self.save_dir)
            pereira_embeddings = pereira_encoder.get_embeddings(words)
            
            matrix_embeddings = []            
            for word in words:
                concatenation = np.concatenate((image_embeddings[word], pereira_embeddings[word]))
                matrix_embeddings.append(concatenation)

            # PCA uses SVD to reduce dimensions. Maximum dimensions is 132 (since I have 132 words)
            pca = PCA(n_components=132)
            pca_result = pca.fit_transform(matrix_embeddings)
            
            # get the embeddings in the right format
            index = 0
            for word in words:
                combi_embeddings[word] = pca_result[index]
                index = index + 1

            self.save_embeddings(embedding_file, combi_embeddings)
        return combi_embeddings
