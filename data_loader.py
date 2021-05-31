import csv
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from progressbar import ProgressBar
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def preprocessing_pipe(image_file=None, image=None, img_rescale=100):
    if image is None:
        center_image = cv2.imread(image_file)
    else:
        center_image = image
    # Crop image
    x_pos = [0, center_image.shape[1]]
    y_pos = [75, 125]
    center_image = center_image[y_pos[0]:y_pos[1], x_pos[0]:x_pos[1], :]
    if img_rescale < 100:
        # Rescale image by half
        scale_percent = img_rescale
        # calculate the 50 percent of original dimensions
        width = int(center_image.shape[1] * scale_percent / 100)
        height = int(center_image.shape[0] * scale_percent / 100)
        dsize = (width, height)

        # rescale + normalize image
        center_image = cv2.resize(center_image, dsize)

        center_image = (center_image / 127.5) - 1.
    return center_image


def generator(samples, batch_size=32, image_folder="./IMG/", img_rescale=100):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                file_path = batch_sample[0].replace('/', os.sep).replace('\\', os.sep)
                name = os.path.join(image_folder, file_path.split(os.sep)[-1])
                center_image = preprocessing_pipe(image_file=name, img_rescale=img_rescale)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def load_training_data_as_generator(file_path):
    samples = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


def load_training_data(file_path, image_folder="IMG/", test=False):
    lines = []
    counter = 0
    data_folder = file_path.rsplit('/', 1)[0]
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            if test:
                counter += 1
                if counter > 1000:
                    break
            lines.append(line)
    images = []
    measurements = []
    with ProgressBar(max_value=len(lines * 3)) as bar:
        counter = 0
        for line in lines:
            for i in range(3):
                bar.update(counter)
                counter += 1
                source_path = line[i]
                source_path = source_path.replace("\\", os.sep).replace("/", os.sep)
                file_name = source_path.rsplit(os.sep, 1)[-1]
                current_path = os.path.join(data_folder, image_folder, file_name)
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                if i == 0:
                    measurements.append(measurement)
                elif i == 1:
                    # steering left image to right
                    measurements.append((measurement + 0.2))
                elif i == 2:
                    # steering right image to left
                    measurements.append((measurement - 0.2))

    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train


def augment_images(images, measurements):
    augmented_images, augmented_measurements = [], []
    counter = 0
    with ProgressBar(max_value=len(images)) as bar:
        for image, measurement in zip(images, measurements):
            bar.update(counter)
            counter += 1
            augmented_images.append(image)
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurements.append(measurement)
            augmented_measurements.append(-measurement)
    return np.array(augmented_images), np.array(augmented_measurements)


# Save to pickel file for later testing

def save_to_pickle_file(save_file: str, data):
    print("[Saving] To file {FILE}".format(save_file))
    if os.path.exists(save_file):
        print("[Warning] Overwrite old pickle file!")
        os.remove(save_file)
    with open(save_file, 'ab') as f:
        pickle.dump(data, f)


def load_from_pickle_file(load_file: str):
    print("[Loading] Try to load {FILE}".format(load_file))
    if os.path.exists(load_file):
        with open(load_file, 'rb') as file:
            print("[Loading] Found file storage. Load existing one")
            return pickle.loads(file.read())
    print("[Loading] Failed!")


def data_resampling(csv_file):
    """
    Generate a new csv with left and right images added as center with a steering offset
    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file, skiprows=1, names=["center", "left", "right", "steering", "throttle", "brake", "speed"])
    new_df = pd.DataFrame(columns=["center", "left", "right", "steering", "throttle", "brake", "speed"])
    counter = 0
    with ProgressBar(max_value=len(df) * 3) as bar:
        for pos, row in df.iterrows():
            bar.update(counter)
            # add image
            new_df.loc[counter, :] = {
                "center": row["center"],
                "left": '',
                "right": '',
                "steering": row["steering"],
                "throttle": row["throttle"],
                "brake": row["brake"],
                "speed": row["speed"]
            }
            counter += 1
            # add left image
            new_df.loc[counter, :] = {
                "center": row["left"],
                "left": '',
                "right": '',
                "steering": row["steering"] + 0.25,
                "throttle": row["throttle"],
                "brake": row["brake"],
                "speed": row["speed"]
            }
            counter += 1
            # add right image
            new_df.loc[counter, :] = {
                "center": row["right"],
                "left": '',
                "right": '',
                "steering": row["steering"] - 0.25,
                "throttle": row["throttle"],
                "brake": row["brake"],
                "speed": row["speed"]
            }
            counter += 1
    save_file = csv_file.rsplit(".", 1)[0] + "_to_center_only.csv"
    new_df.to_csv(save_file, index=False)
    return new_df

