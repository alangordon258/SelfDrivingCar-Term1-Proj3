import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import argparse
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
# I had to comment out the call to plot when running on Ubuntu server because pyplot is missing in the carnd virtualenv
from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Construct the argument parser and parse the arguments
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--datadirs", required=False,type=str,default='./recordeddata',
                    help="Path to the training data. Use commas to separate multiple paths.")
    ap.add_argument("-m", "--model", required=False, type=str, default='model.h5',
                    help="Model filename to write and read if reload model is specified.")
    ap.add_argument("-e", "--epochs", required=False,type=int,default=20,
                    help="Number of epochs.")
    ap.add_argument("-g", "--generator", required=False, default='True',type=str,
                    help="Use a generator instead of loading all images in memory.")
    ap.add_argument("-l", "--reloadmodel", required=False, default='False', type=str,
                    help="Reload specified model instead of starting from scratch.")
    ap.add_argument("-v", "--visualization", required=False, default='False', type=str,
                    help="Visualize data instead of training.")
    ap.add_argument("-r", "--regularization", required=False, default='True', type=str,
                    help="Use regularization.")
    ap.add_argument("-p", "--droppercentage", required=False, type=int, default=50,
                    help="Percentage of small steering angles to drop")
    args = vars(ap.parse_args())
    return args

def get_boolean_arg(args,arg_name):
    if args[arg_name] == "True" or args[arg_name] == "true":
        boolean_arg=True
    elif args[arg_name] == "False" or args[arg_name] == "false":
        boolean_arg=False
    return boolean_arg

def view_measurements(data,plot_name,file_name):
    bin_range = np.arange(-1.0, 1.0, 0.1, dtype=np.float32)
    fig = plt.figure(figsize=(8, 5))
    plt.hist(data,bins=21)
    plt.title(plot_name)
    plt.ylabel('# of Measurements')
    plt.xticks(bin_range)
    plt.xlim(-1.0, 1.0)
    plt.show()
    fig.savefig(file_name)

def view_images(center_paths,left_paths,right_paths,file_name_before,file_name_after):
    indx = random.randint(0, (len(center_paths)-1))

    center_img=cv2.imread(center_paths[indx])
    left_img = cv2.imread(left_paths[indx])
    right_img = cv2.imread(right_paths[indx])

    center_img_rgb = cv2.cvtColor(center_img,cv2.COLOR_BGR2RGB)
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(9, 3))
    subfig = fig.add_subplot(1,3,1)
    subfig.imshow(left_img_rgb)
    subfig.axis('off')
    subfig = fig.add_subplot(1,3,2)
    subfig.imshow(center_img_rgb)
    subfig.axis('off')
    subfig = fig.add_subplot(1,3,3)
    subfig.imshow(right_img_rgb)
    subfig.axis('off')
    fig.tight_layout()
    fig.show()
    fig.savefig(file_name_before)

    center_img = preprocess_image(center_img)
    left_img = preprocess_image(left_img)
    right_img = preprocess_image(right_img)
    center_img_rgb = cv2.cvtColor(center_img, cv2.COLOR_YUV2RGB)
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_YUV2RGB)
    right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_YUV2RGB)

    fig = plt.figure(figsize=(9, 3))
    subfig = fig.add_subplot(1, 3, 1)
    subfig.imshow(left_img_rgb)
    subfig.axis('off')
    subfig = fig.add_subplot(1, 3, 2)
    subfig.imshow(center_img_rgb)
    subfig.axis('off')
    subfig = fig.add_subplot(1, 3, 3)
    subfig.imshow(right_img_rgb)
    subfig.axis('off')
    fig.tight_layout()
    fig.show()
    fig.savefig(file_name_after)

def preprocess_image(img):
    # crop the top 50 and bottom 20 pixels from the image
    processed_img = img[50:140,:,:]
    # apply a slight blur
    processed_img = cv2.GaussianBlur(processed_img, (3,3), 0)
    # scale to 66x200x3 to match nVidia
    processed_img = cv2.resize(processed_img,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV color space per the nVidia paper
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2YUV)
    return processed_img

def generate_data(image_data_paths, steering_angles, batch_size=128):
    half_batch_size=batch_size/2
    j=0
    num_samples=len(steering_angles)
    while True:
        X, y = ([], [])
        for i in range(64):
            # choose a random index
       #     index = random.randint(0, (len(steering_angles)-1))
            if j==num_samples:
                j=0

            path=image_data_paths[j]
            img = cv2.imread(image_data_paths[j])

            new_img = preprocess_image(img)
            steering_angle = steering_angles[j]
            X.append(new_img)
            y.append(steering_angle)

            img_flipped = np.fliplr(new_img)
            steering_angle_flipped = -steering_angle
            X.append(img_flipped)
            y.append(steering_angle_flipped)
            j=j+1
        Features=np.array(X)
        Labels=np.array(y)
        yield (Features,Labels )

def fake_generate_data(image_data_paths, steering_angles):
    X,y = ([],[])
    for i in range(len(steering_angles)):
        img = cv2.imread(image_data_paths[i])
        steering_angle = steering_angles[i]
        new_img = preprocess_image(img)
        X.append(new_img)
        y.append(steering_angle)
        img_flipped = np.fliplr(new_img)
        steering_angle_flipped = -steering_angle
        X.append(img_flipped)
        y.append(steering_angle_flipped)
    return (np.array(X), np.array(y))

def build_model(keep_prob):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))
#    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

def build_model_with_regularization():
    model = Sequential()
    # Normalize
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))

    # Add three 5x5 convolution layers (with output depths: 24, 36, and 48), with 2x2 stride and regularization
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add two 3x3 convolution layers with output depth 64
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers with output depths: 100, 50, 10, ELU activation with regularization
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add fully connected output layer
    model.add(Dense(1))
    model.summary()
    return model

def train_model_with_generator(model,train_gen,val_gen,test_gen):
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.fit_generator(train_gen, validation_data=val_gen,
                        samples_per_epoch=2*num_train_samples,
                        nb_val_samples=2*num_validation_samples, nb_epoch=num_epochs)
    print('Test Loss:', model.evaluate_generator(test_gen, 128))
    return model

def train_model_in_memory(model,X_train,y_train):
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.fit(X_train, y_train, validation_split=0.2, batch_size=128,shuffle=True, nb_epoch=num_epochs)
    return model

# Coalesce the left and right images along with their adjusted steering values to account for
# the offset of the camera into a single list of paths to images and label values (measurements)
def collapse_left_right(new_steerings,new_center_paths,new_left_paths,new_right_paths):
# The steering correction to apply to the left and right images
    correction = 0.25
    for steering_center, center_path, left_path, right_path in zip(new_steerings,
                                                                   new_center_paths, new_left_paths, new_right_paths):
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        image_paths.append(center_path)
        measurements.append(steering_center)

        image_paths.append(left_path)
        measurements.append(steering_left)

        image_paths.append(right_path)
        measurements.append(steering_right)

def load_data_files(root_path,drop_percentage):
# steering angles less than significant_value are considered small steering angles and are candidates
# to be dropped based on the specified drop_percentage
    significant_value = 0.05
    lines = []
    image_data_path = root_path + '/IMG/'
    with open(root_path+'/driving_log.csv') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    for line in lines:
        steering_center = float(line[3])
        if abs(steering_center) < significant_value:
            rand_num = random.randint(0, 100)
            if rand_num < drop_percentage:
                continue
        steerings.append(steering_center)
        center_path=line[0]
        center_filename = center_path.split('/')[-1]
        center_final_path = image_data_path + center_filename
        center_paths.append(center_final_path)

        left_path=line[1]
        left_filename = left_path.split('/')[-1]
        left_final_path = image_data_path + left_filename
        left_paths.append(left_final_path)

        right_path=line[2]
        right_filename = right_path.split('/')[-1]
        right_final_path = image_data_path + right_filename
        right_paths.append(right_final_path)

def flatten_histogram(steerings,center_paths,left_paths,right_paths):
# Get histogram and use to smooth data
# Read all the steering value and image file paths into memory
    flattened_steerings=[]
    flattened_center_paths=[]
    flattened_left_paths=[]
    flattened_right_paths=[]
    bin_range=np.arange(-1.0,1.0,0.1,dtype=np.float32)
    hist, bins = np.histogram(steerings, bins=bin_range)
    print(bins)
    print(hist)
    avg=np.mean(hist)
    print(avg)
    above_avg_const=1.0
    above_avg=int(above_avg_const*math.ceil(avg))
    for i, val in enumerate(bin_range):
        low=val
        high=val+0.1

        band=np.where((steerings > low) & (steerings <= high))
        num_values_in_band=len(band[0])

        if num_values_in_band == 0:
            continue
        elif num_values_in_band <= above_avg:
            flattened_steerings.extend(steerings[band])
            flattened_center_paths.extend(center_paths[band])
            flattened_left_paths.extend(left_paths[band])
            flattened_right_paths.extend(right_paths[band])
        else:
        # calculate percentage above average
            reduced_band=np.random.choice(band[0], above_avg)
            flattened_steerings.extend(steerings[reduced_band])
            flattened_center_paths.extend(center_paths[reduced_band])
            flattened_left_paths.extend(left_paths[reduced_band])
            flattened_right_paths.extend(right_paths[reduced_band])
    return flattened_steerings, flattened_center_paths, flattened_left_paths, flattened_right_paths

# Create empty arrays to load data
image_paths=[]
measurements=[]
steerings=[]
center_paths=[]
left_paths=[]
right_paths=[]

# Read command line arguments
args=get_arguments()
recorded_data_paths=args["datadirs"]
num_epochs=args["epochs"]
model_file_name=args["model"]
percentage_of_small_steering_angles_to_drop=args["droppercentage"]
use_generator=get_boolean_arg(args,"generator")
reload_model=get_boolean_arg(args,"reloadmodel")
do_visualization=get_boolean_arg(args,"visualization")
use_regularization=get_boolean_arg(args,"regularization")

# Print to command line so user knows we are running
if do_visualization:
    print("Visualization started:")
    print("Data paths={}".format(recorded_data_paths))
    print("Percentage of small angles that will be dropped={}".format(percentage_of_small_steering_angles_to_drop))
else:
    print("Training started:")
    print("Data paths={}".format(recorded_data_paths))
    print("Percentage of small angles that will be dropped={}".format(percentage_of_small_steering_angles_to_drop))
    print("Number of epochs={}".format(num_epochs))
    print("Use generator={}".format(use_generator))
    print("Use regularization={}".format(use_regularization))
    print("Model file={}".format(model_file_name))
    print("Reload model={}".format(reload_model))

parsed_paths=recorded_data_paths.split(',')
for path in parsed_paths:
    load_data_files(path,percentage_of_small_steering_angles_to_drop)
    print("Data read from path: {}".format(path))
    if do_visualization:
        loaded_filename = path.split('/')[-1]
        view_measurements(steerings, "Measurements from {}".format(loaded_filename), "./visualization/"+
                          loaded_filename+".jpg")
        view_images(center_paths, left_paths, right_paths, "./visualization/"+loaded_filename+"_images_before.jpg",
                    "./visualization/"+loaded_filename+"_images_after.jpg")

if do_visualization:
    view_measurements(steerings, "Combined Measurements", "./visualization/combined_measurements.jpg")

# Convert everything to numpy
steerings=np.array(steerings)
center_paths=np.array(center_paths,dtype=object)
left_paths=np.array(left_paths,dtype=object)
right_paths=np.array(right_paths,dtype=object)

new_steerings,new_center_paths,new_left_paths, new_right_paths=flatten_histogram(steerings,center_paths,left_paths,right_paths)

if do_visualization:
    view_measurements(new_steerings, "Flattened Histogram", "./visualization/AfterHistogramFlattening.jpg")

collapse_left_right(new_steerings,new_center_paths,new_left_paths,new_right_paths)

image_paths=np.array(image_paths)
measurements=np.array(measurements)

if do_visualization:
    view_measurements(measurements, "Left and Right Images Added", "./visualization/AfterLeftRightAdded.jpg")

if do_visualization:
    X_train_data, y_train_data = fake_generate_data(image_paths, measurements)
    view_measurements(y_train_data, "Flipped Images Added", "./visualization/FlippedImagesAdded.jpg")
else:
    if use_generator:
        image_paths, measurements = shuffle(image_paths, measurements)
        image_paths_train, image_paths_valid_test, measurements_train, measurements_valid_test = train_test_split(
            image_paths, measurements,
            test_size=0.20, random_state=42)
        image_paths_valid, image_paths_test, measurements_valid, measurements_test = train_test_split(
            image_paths_valid_test, measurements_valid_test,
            test_size=0.50, random_state=42)

        train_gen_data = generate_data(image_paths_train, measurements_train, batch_size=128)
        val_gen_data = generate_data(image_paths_valid, measurements_valid, batch_size=128)
        test_gen_data = generate_data(image_paths_test, measurements_test, batch_size=128)

        num_train_samples=len(image_paths_train)
        num_validation_samples=len(image_paths_valid)
    else:
        X_train_data,y_train_data=fake_generate_data(image_paths, measurements)

if reload_model:
    model=load_model(model_file_name)
else:
    if use_regularization:
        model=build_model_with_regularization()
    else:
        model=build_model(0.6)

if do_visualization:
# I had to comment out the call to plot when running on Ubuntu server because pyplot is missing in the carnd virtualenv
    plot(model, to_file='./visualization/model.jpg',show_shapes=True,show_layer_names=False)
#    print("Whatever")

if not do_visualization:
    if use_generator:
        model=train_model_with_generator(model,train_gen_data,val_gen_data,test_gen_data)
    else:
        model=train_model_in_memory(model,X_train_data,y_train_data)

    model.save(model_file_name)