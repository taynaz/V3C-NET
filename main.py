import csv
import os
import numpy as np
from PIL import Image
from keras.models import Sequential, load_model, Model
from keras.layers import Conv3D, BatchNormalization, Dropout
from keras.layers.convolutional import MaxPooling3D, ZeroPadding3D
from keras.layers.core import Dense, Flatten
from keras import optimizers
from natsort import natsorted
from keras.utils import np_utils
import random
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

train_data_dir = '/home/dgxstation1/Desktop/Tayyba/Counting_Dataset/'  # path

from keras import backend as K

K.set_image_dim_ordering('tf')


def Counting_NN():
    model = Sequential()
    backend = 'tf'
    if backend == 'tf':
        input_shape = (25, 84, 84, 3)  # l, h, w, c
    else:
        input_shape = (3, 25, 84, 84)  # c, l, h, w
    model.add(Conv3D(64, (3, 3, 3), padding='same', strides=(1, 1, 1), input_shape=input_shape,
                     activation='relu',
                     name='conv1'))  # (batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))

    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1)))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))

    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))

    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a', strides=(1, 1, 1)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b', strides=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 2, 2)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))

    model.add(Flatten())

    print(model.summary())

    return model


def Counting_NN_FC_Layers():
    counting_model = Sequential()

    # FC layers group
    counting_model.add(Dense(4096,input_shape = (8192,), activation='relu', name='fc6'))
    counting_model.add(Dense(4096, activation='relu', name='fc7'))
    counting_model.add(Dense(4096, activation='relu', name='fc8'))
    counting_model.add(Dense(4096, activation='relu', name='fc9'))
    counting_model.add(Dense(4096, activation='relu', name='fc10'))
    counting_model.add(Dense(512, activation='relu', name='fc11'))
    counting_model.add(Dense(1, activation='linear', name='fc12'))

    return counting_model


def read_csv(folderpath):
    subfolder_path = []
    count = []
    with open(folderpath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['folder'], row['count'])
            subfolder_path.append(row['count'])
            count.append(row['folder'])
    return subfolder_path, count


def get_list_images(subfolder_name):
    list_images = []
    subfolder_path = os.path.join(train_data_dir, subfolder_name)
    print(subfolder_path)
    for images in os.listdir(subfolder_path):
        if images.endswith('.jpg'):
            list_images.append(images)

    list_images = natsorted(list_images)
    return list_images


def read_images(folder_path, array_image_file_path):
    Image_array_data = []
    index = 0
    reading_ith_frame = 10  # 25 frames
    subfolder_full_path = os.path.join(train_data_dir, folder_path)
    num_frame = 0
    while (index < 250):
        localImageName = str(array_image_file_path[index]).strip()

        original_image = Image.open(subfolder_full_path + "/" + localImageName)
        resized_image = original_image.resize((84, 84))
        inputImage = np.asarray(resized_image).astype('float32')
        Image_array_data.append(inputImage)
        index += reading_ith_frame
        num_frame += num_frame + 1
    return Image_array_data


def main():
    
    filepath = '/home/dgxstation1/Desktop/Tayyba/train_sorted.csv'
    count, subfolder_path = read_csv(filepath)
    feature_model = Counting_NN()
    
    feature_model = load_model('/home/dgxstation1/Desktop/Models/Untitled Folder/pre_trained model.h5')
    feature_model.summary()

    counting_model =Counting_NN_FC_Layers()

    counting_model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    counting_model.load_weights('/home/dgxstation1/Desktop/Models/Untitled Folder/17dec2018_Weights_id_16740_.h5')

    for model_epochs in range(1, 25000):
        features = []
        i = random.randint(1, 2170)
        
        list_images = get_list_images(subfolder_path[i])
        images_array = read_images(subfolder_path[i], list_images)
        image = np.array(images_array).astype('float32')
        train_image = image.reshape(1,25, 84, 84,3)

        output = count[i]
        output_count = np.array(output)
        train_label = output_count.reshape((-1, 1))
        
        features= feature_model.predict(train_image)
        reshape_features = features.reshape(1,4,4,512)
        flatten_array = reshape_features.flatten()
        reshaped_flattened_array = flatten_array.reshape(1,8192)

        counting_model.train_on_batch(np.array(reshaped_flattened_array), np.array(train_label), sample_weight=None, class_weight=None)
        Loss_training = counting_model.evaluate(np.array(reshaped_flattened_array), np.array(train_label), batch_size=None, verbose=1,
                                       sample_weight=None, steps=None)
       
        if ((Loss_training < 0.005) or (model_epochs%30== 0) ):
            file_name = "05feb2019_id_" + str(model_epochs) + "_"  + ".h5"
            file_name_weights = "05feb2019_Weights_id_" + str(model_epochs) + "_" + ".h5"
            Weight_file_Path = '/home/dgxstation1/Desktop/Models/pretrained_feature_model/'
            counting_model.save(Weight_file_Path + file_name)
            counting_model.save_weights(Weight_file_Path +file_name_weights)
            testing(model_epochs, file_name)

    
#TESTING
def testing(model_epochs, filename):
   
    feature_model = load_model('/home/dgxstation1/Desktop/Models/Untitled Folder/pre_trained model.h5')
    filepath_test = '/home/dgxstation1/Desktop/Models/pretrained_feature_model/'
    model_test = load_model(filepath_test + filename)


    filepath = '/home/dgxstation1/Desktop/Tayyba/test_sorted.csv'
    count, subfolder_path = read_csv(filepath)
    for i in range(1, 930):
        list_images_test = get_list_images(subfolder_path[i])
        images_array_test = read_images(subfolder_path[i], list_images_test)
        image_test = np.array(images_array_test).astype('float32')
        train_image_test = image_test.reshape(1, 25, 84, 84, 3)
        output_test = count[i]
        output_count_test = np.array(output_test)
        test_label = output_count_test.reshape((-1, 1))
        features_test = feature_model.predict(train_image_test)
        reshape_features_test = features_test.reshape(1, 4, 4, 512)
        print(reshape_features_test.shape)
        flatten_array_test = reshape_features_test.flatten()
        reshaped_flattened_array_test = flatten_array_test.reshape(1, 8192)
        estimated_output = model_test.predict(reshaped_flattened_array_test, batch_size=1)

        with open("testingresult_05feb2019.txt", "a") as f:
            f.writelines("Epoch:")
            f.writelines(str(model_epochs) + ' ')
            f.writelines("actual: ")
            f.writelines(str(output_test) + ' ')
            f.writelines("estimated")
            f.writelines(str(np.array(estimated_output[0])) + '\n')

    return

if __name__ == '__main__':
    main()