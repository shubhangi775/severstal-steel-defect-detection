import os
import json
import gc

import cv2
import keras
from keras import backend as K
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from tensorflow.keras.layers import Concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('./input/train.csv')
# train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
# train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

print(train_df.shape)
print(train_df.head())


mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
print(mask_count_df.shape)
print(mask_count_df.head())

sub_df = pd.read_csv('./input/sample_submission.csv')
sub_df['ImageId'] = sub_df['ImageId']
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
print(test_imgs.head())

non_missing_train_idx = mask_count_df[mask_count_df['hasMask'] > 0]
print(non_missing_train_idx.head())




def load_img(code, base, resize=True):
    path = f'{base}/{code}'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, (256, 256))
    
    return img

def validate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


BATCH_SIZE = 64
def create_test_gen():
    return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
        test_imgs,
        directory='../input/test_images',
        x_col='ImageId',
        class_mode=None,
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

test_gen = create_test_gen()

remove_model = load_model('../input/severstal-predict-missing-masks/model.h5')
remove_model.summary()