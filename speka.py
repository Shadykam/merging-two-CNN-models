import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from IPython.display import display, Audio

import io
import csv
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
from pathlib import Path
from IPython.display import display, Audio

from keras import Model
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, Conv1D, MaxPooling2D, Reshape, Concatenate, concatenate, Dropout , MaxPooling1D, Flatten
from keras.layers import Dense, Input

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D , Conv1D
from keras.layers import Conv2D, Conv1D,MaxPooling2D ,BatchNormalization, Reshape, Multiply, Dropout , MaxPooling1D

# Optimizers
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop

# ----------------------- Accent Data ----------------------

SEED = 1337
EPOCHS = 5
BATCH_SIZE = 64
VALIDATION_RATIO = 0.1
MODEL_NAME = "uk_irish_accent_recognition"

# Location where the dataset will be downloaded.
# By default (None), keras.utils.get_file will use ~/.keras/ as the CACHE_DIR
#CACHE_DIR = None

# The location of the dataset
#URL_PATH = "https://www.openslr.org/resources/83/"

# List of datasets compressed files that contain the audio files
zip_files = {
    0: "irish_english_male",
    1: "midlands_english_female",
    2: "midlands_english_male",
    3: "northern_english_female",
    4: "northern_english_male",
    5: "scottish_english_female",
    6: "scottish_english_male",
    7: "southern_english_female",
    8: "southern_english_male",
    9: "welsh_english_female",
    10: "welsh_english_male",


}

# We see that there are 2 compressed files for each accent (except Irish):
# - One for male speakers
# - One for female speakers
# However, we will using a gender agnostic dataset.

# List of gender agnostic categories
gender_agnostic_categories = [
    "ir",  # Irish
    "mi",  # Midlands
    "no",  # Northern
    "sc",  # Scottish
    "so",  # Southern
    "we",  # Welsh

]

class_names = [
    "Irish",
    "Midlands",
    "Northern",
    "Scottish",
    "Southern",
    "Welsh",
    "Not a speech",
]

# Set all random seeds in order to get reproducible results
keras.utils.set_random_seed(SEED)

# Where to download the dataset
DATASET_DESTINATION = os.path.dirname(os.getcwd()) + '/data';

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# CSV file that contains information about the dataset. For each entry, we have:
# - ID
# - wav file name
# - transcript
line_index_file = keras.utils.get_file(
    fname="line_index_file", origin=DATASET_DESTINATION + "line_index_all.csv"
)

# Download the list of compressed files that contains the audio wav files
for i in zip_files:
    fname = zip_files[i].split(".")[0]
    url = DATASET_DESTINATION + zip_files[i]

    #zip_file = keras.utils.get_file(fname=fname, origin=url, extract=True)
    #os.remove(zip_file)

dataframe = pd.read_csv(
    line_index_file, names=["id", "filename", "transcript"], usecols=["filename"]
)
dataframe.head()

# The purpose of this function is to preprocess the dataframe by applying the following:
# - Cleaning the filename from a leading space
# - Generating a label column that is gender agnostic i.e.
#   welsh english male and welsh english female for example are both labeled as
#   welsh english
# - Add extension .wav to the filename
# - Shuffle samples
def preprocess_dataframe(dataframe):
    # Remove leading space in filename column
    dataframe["filename"] = dataframe.apply(lambda row: row["filename"].strip(), axis=1)

    # Create gender agnostic labels based on the filename first 2 letters
    dataframe["label"] = dataframe.apply(
        lambda row: gender_agnostic_categories.index(row["filename"][:2]), axis=1
    )

    # Add the file path to the name
    dataframe["filename"] = dataframe.apply(
        lambda row: os.path.join(DATASET_DESTINATION, row["filename"] + ".wav"), axis=1
    )

    # Shuffle the samples
    dataframe = dataframe.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return dataframe


dataframe = preprocess_dataframe(dataframe)
dataframe.head()

split = int(len(dataframe) * (1 - VALIDATION_RATIO))
train_df = dataframe[:split]
valid_df = dataframe[split:]

print(
    f"We have {train_df.shape[0]} training samples & {valid_df.shape[0]} validation ones"
)

@tf.function
def load_16k_audio_wav(filename):
    # Read file content
    file_content = tf.io.read_file(filename)

    # Decode audio wave
    audio_wav, sample_rate = tf.audio.decode_wav(file_content, desired_channels=1)
    audio_wav = tf.squeeze(audio_wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    # Resample to 16k
    audio_wav = tfio.audio.resample(audio_wav, rate_in=sample_rate, rate_out=16000)

    return audio_wav


def filepath_to_embeddings(filename, label):
    # Load 16k audio wave
    audio_wav = load_16k_audio_wav(filename)

    # Get audio embeddings & scores.
    # The embeddings are the audio features extracted using transfer learning
    # while scores will be used to identify time slots that are not speech
    # which will then be gathered into a specific new category 'other'
    scores, embeddings, _ = yamnet_model(audio_wav)

    # Number of embeddings in order to know how many times to repeat the label
    embeddings_num = tf.shape(embeddings)[0]
    labels = tf.repeat(label, embeddings_num)

    # Change labels for time-slots that are not speech into a new category 'other'
    labels = tf.where(tf.argmax(scores, axis=1) == 0, label, len(class_names) - 1)

    # Using one-hot in order to use AUC
    return (embeddings, tf.one_hot(labels, len(class_names)))


def dataframe_to_dataset(dataframe, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["filename"], dataframe["label"])
    )

    dataset = dataset.map(
        lambda x, y: filepath_to_embeddings(x, y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).unbatch()

    return dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


train_ds = dataframe_to_dataset(train_df)
valid_ds = dataframe_to_dataset(valid_df)

keras.backend.clear_session()


# ----------------------- Speaker Data ----------------------

# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
# and save it to the 'Downloads' folder in your HOME directory
DATASET_ROOT = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

# Percentage of samples to use for validation
VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The sampling rate to use.
# This is the one used in all of the audio samples.
# We will resample all of the noise to this sampling rate.
# This will also be the output size of the audio wave samples
# (since all samples are of 1 second long)
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZ = 128
EPOCH = 5

# If folder `audio`, does not exist, create it, otherwise do nothing
if os.path.exists(DATASET_AUDIO_PATH) is False:
    os.makedirs(DATASET_AUDIO_PATH)

# If folder `noise`, does not exist, create it, otherwise do nothing
if os.path.exists(DATASET_NOISE_PATH) is False:
    os.makedirs(DATASET_NOISE_PATH)

for folder in os.listdir(DATASET_ROOT):
    if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
        if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
            # If folder is `audio` or `noise`, do nothing
            continue
        elif folder in ["other", "_background_noise_"]:
            # If folder is one of the folders that contains noise samples,
            # move it to the `noise` folder
            shutil.move(
                os.path.join(DATASET_ROOT, folder),
                os.path.join(DATASET_NOISE_PATH, folder),
            )
        else:
            # Otherwise, it should be a speaker folder, then move it to
            # `audio` folder
            shutil.move(
                os.path.join(DATASET_ROOT, folder),
                os.path.join(DATASET_AUDIO_PATH, folder),
            )

# Get the list of all noise files
noise_paths = []
for subdir in os.listdir(DATASET_NOISE_PATH):
    subdir_path = Path(DATASET_NOISE_PATH) / subdir
    if os.path.isdir(subdir_path):
        noise_paths += [
            os.path.join(subdir_path, filepath)
            for filepath in os.listdir(subdir_path)
            if filepath.endswith(".wav")
        ]

print(
    "Found {} files belonging to {} directories".format(
        len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
    )
)

command = (
    "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
    "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
    "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
    "$file | grep sample_rate | cut -f2 -d=`; "
    "if [ $sample_rate -ne 16000 ]; then "
    "ffmpeg -hide_banner -loglevel panic -y "
    "-i $file -ar 16000 temp.wav; "
    "mv temp.wav $file; "
    "fi; done; done"
)

os.system(command)

# Split noise into chunks of 16000 each
def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None


noises = []
for path in noise_paths:
    sample = load_noise_sample(path)
    if sample:
        noises.extend(sample)
noises = tf.stack(noises)

print(
    "{} noise files were split into {} noise samples where each is {} sec. long".format(
        len(noise_paths), noises.shape[0], noises.shape[-1] // SAMPLING_RATE
    )
)

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
       # Create a random tensor of the same size as audio ranging from
       # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

       # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)
        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Get the list of audio file paths along with their corresponding labels

class_name = os.listdir(DATASET_AUDIO_PATH)
print("Our class names: {}".format(class_names,))

audio_paths = []
labels = []
for label, name in enumerate(class_name):
    print("Processing speaker {}".format(name,))
    dir_path = Path(DATASET_AUDIO_PATH) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)

# Shuffle
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]

print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

# Create 2 datasets, one for training and the other for validation
train_dse = paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_dse = train_dse.shuffle(buffer_size=BATCH_SIZ * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZ
)

valid_dse = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_dse = valid_dse.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)


# Add noise to the training set
train_dse = train_dse.map(
    lambda x, y: (add_noise(x, noises, scale=SCALE), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Transform audio wave to the frequency domain using `audio_to_fft`
train_dse = train_dse.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
train_dse = train_dse.prefetch(tf.data.AUTOTUNE)

valid_dse = valid_dse.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
valid_dse = valid_dse.prefetch(tf.data.AUTOTUNE)

# ----------------------- Accent CNN ----------------------
in_1D = Sequential()

in_1D = keras.layers.Input(shape=(1024), name="embedding")
# 1
x = keras.layers.Dense(256, activation="relu", name="dense_1")(in_1D)
x = keras.layers.Dropout(0.15, name="dropout_1")(x)

x = keras.layers.Dense(384, activation="relu", name="dense_2")(x)
x = keras.layers.Dropout(0.2, name="dropout_2")(x)

x = keras.layers.Dense(192, activation="relu", name="dense_3")(x)
x = keras.layers.Dropout(0.25, name="dropout_3")(x)

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(384, activation="relu", name="dense_4")(x)
x = keras.layers.Dropout(0.2, name="dropout_4")(x)

#class name = 7
outputs = keras.layers.Dense(len(class_names), activation="softmax", name="ouput")(
    x
)

x = keras.Model(inputs=in_1D, outputs=outputs, name="accent_recognition")

x.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1.9644e-5),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

# ----------------------- Speaker CNN ----------------------

model = Sequential()

def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

in_2D = keras.layers.Input((SAMPLING_RATE // 2, 1), len(class_name))
print(in_2D, in_1D)
x = residual_block(in_2D, 16, 2)
x = residual_block(x, 32, 2)
x = residual_block(x, 64, 3)
x = residual_block(x, 128, 3)
x = residual_block(x, 128, 3)

x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dense(128, activation="relu")(x)

#change num classes to 50
x = keras.layers.Dense(len(class_name), activation="softmax", name="output")(x)

model = x

model_save_filename = "model.h5"

earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True)

# ----------------------- Merged CNN ----------------------

merged = Multiply()([model, x])
output = Dense(7, activation='softmax')(merged)

model_combined = Model(inputs=[in_1D, in_2D], outputs=[output])

model_combined.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = model_combined

model.summary()


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_combined.png')

# ----------------------- training ----------------------

model = Sequential()
model.add(Dense(8, input_shape=(10,), activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="linear"))

inputs = Input(shape=(10,))
x = Dense(8, activation="relu")(inputs)
x = Dense(4, activation="relu")(x)
x = Dense(1, activation="linear")(x)
model = Model(inputs, x)

# define two sets of inputs
inputA = Input(shape=(1))
inputB = Input(shape=(1))

# the first branch operates on the first input
x = Dense(8, activation="relu")(inputA)
x = Dense(4, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)

# the second branch opreates on the second input
y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(4, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)

# combine the output of the two branches
combined = concatenate([x.output, y.output])

# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(2, activation="relu")(combined)
z = Dense(1, activation="linear")(z)

# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model1 = tf.keras.Sequential(name="input_3")
model2 = tf.keras.Sequential(name="input_4")


x = tf.data.Dataset.from_tensors(({"input_3": [[(1024)], [(1)]],
                                     "input_4": [[(8000)], [(1)]]}))

print(model.evaluate(x))

history = model.fit(
    x,
    validation_data=x,
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)



