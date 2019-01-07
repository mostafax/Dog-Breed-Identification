import pandas as pd
import cv2
from os import listdir


def image_resize():
    """
    Resize the Input Images to (224,224) and save them in other folder
    """
    train_data = 'test/'
    images = listdir(train_data)
    new_folder = 'train_edited/{}'
    for image in images:
        img = cv2.imread(train_data+image)
        resized_image = cv2.resize(img, (224, 224))
        cv2.imwrite(new_folder.format(image), resized_image)


def data_load():
    """
    Load train data
    :return:
    image_id (numpy array): the Images ID
    image_label(numpy array) : the corresponding label to the Image ID
    train_images(list) : list that contains images names WITH extensions
    """

    train_images = listdir('train_edited')
    text_data = pd.read_csv('labels.csv')
    image_id = text_data['id'].values
    image_label = text_data['breed'].values
    return image_id, image_label, train_images
