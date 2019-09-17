from matplotlib.image import imread
import numpy as np
import os
import random
from tqdm import tqdm

# callable class that allows user to load batch image by its folder name

class Reader(object):
    def __init__(self, folder_name):
        self.train_set = []
        self.validation_set = []
        self.test_set = []
        dir_dict = {"train":self.train_set, "validation":self.validation_set, "test":self.test_set}

        for dir, storage in dir_dict.items():
            images_list = os.listdir(folder_name+"/"+dir)
            for img in tqdm(images_list):
                target_image = np.asarray(imread(folder_name+"/"+dir+"/"+img))
                storage.append(target_image)

        print(folder_name, "has", len(self.train_set), "train sets,", len(self.validation_set), "validation sets, and ", len(self.test_set), "test sets.")

    def get_next_train(self, batch_size):
        next_batch = random.sample(self.train_set, batch_size)
        return (next_batch)

    def get_next_validation(self, batch_size):
        next_batch = random.sample(self.validation_set, batch_size)
        return (next_batch)

    def get_next_test(self, batch_size):
        next_batch = random.sample(self.test_set, batch_size)
        return (next_batch)