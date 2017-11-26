import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

class PreprocessImage():
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'
    IMG_SIZE = 64
    LR = 1e-3

    MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match

    @staticmethod
    def label_img(img):
        word_label = img.split('.')[0]
        # conversion to one-hot array [cat,dog]
        #                            [much cat, no dog]
        if word_label == 'cat': return 1
        #                             [no cat, very doggo]
        elif word_label == 'dog': return 0

    @staticmethod
    def process_data(mode):
        data = []
        if mode == "train":
            dir = TRAIN_DIR
        else:
            dir = TEST_DIR

        for img in tqdm(os.listdir(dir)):
            label = label_img(img)
            path = os.path.join(dir,img)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            try:
                img = cv2.resize(img, (1, IMG_SIZE**2))
            except cv2.error as e:
                continue
            img = np.divide(img, 255)
            data.append([np.array(img),np.array(label)])
        shuffle(data)
        np.save('data/{}_data.npy'.format(mode), data)
        return data

if __name__ == '__main__':
   PreprocessImage.process_data('train')
   PreprocessImage.process_data('test')
