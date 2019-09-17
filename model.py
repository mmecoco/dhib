import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from matplotlib.image import imread
from DataReader import Reader
import random
import matplotlib.pyplot as plt
import matplotlib.image as mping

# macros
set_lists = ["train", "validation", "test"]
LEARNING_RATE = 0.5
BATCH_SIZE = 100
EPOCH = 3

# list of the element
labels = list(filter(os.path.isdir, os.listdir(".")))
if ("__pycache__" in labels):
    labels.remove("__pycache__")
label_len = len(labels)
# convert idx to name and inverse
idx_2_label = {idx:name for idx, name in enumerate(labels)}
label_2_idx = {name:idx for idx, name in enumerate(labels)}
# data storage as "filename":Reader class
data_dict = {}
for label in labels:
    data_dict.update({label:Reader(label)})

# functions that allows to make next batches
def one_hot(list):
    label_2_idx = {lab:num for num, lab in enumerate(labels)}
    encoded = np.zeros((len(list), label_len))
    for idx, element in enumerate(list):
        encoded[idx][label_2_idx[element]] = 1
    return (encoded)

def get_next_batch(batch_size, type):
    next = []
    if (type == "train"):
        for label, reader in data_dict.items():
            curr = reader.get_next_train(batch_size)
            encoded_array = one_hot([label] * batch_size)
            for img, encod in zip(curr, encoded_array):
                next.append((img, encod))
    elif (type == "validation"):
        for label, reader in data_dict.items():
            curr = reader.get_next_validation(batch_size)
            encoded_array = one_hot([label] * batch_size)
            for img, encod in zip(curr, encoded_array):
                next.append((img, encod))
    else:
        for label, reader in data_dict.items():
            curr = reader.get_next_test(batch_size)
            encoded_array = one_hot([label] * batch_size)
            for img, encod in zip(curr, encoded_array):
                next.append((img, encod))
    for i in range(3):
        random.shuffle(next)
    img_array = []
    encoded_label = []
    for elm in next:
        ia, en = elm
        img_array.append(ia)
        encoded_label.append(en)
    return ((np.asarray(img_array), np.asarray(encoded_label)))

# tensorflow variables and placeholders
def get_weight(shape):
    get = tf.truncated_normal(shape, stddev=0.1)
    return (tf.Variable(get))

def get_bias(shape):
    get = tf.constant(0.1, shape=shape)
    return (tf.Variable(get))

# Bunch of images in numpy array form
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
# One hot encoded answers to the result of the images
answers = tf.placeholder(tf.int32, [None, label_len])
drop_prob = tf.placeholder(tf.float32)

# Variables for 5 conv layers and one dense layer + lelu hidden layers

#   first conv layer with max pooling layer at the end
w_conv_1 = get_weight([5, 5, 3, 64])
b_conv_1 = get_bias([64])
h_conv_1 = tf.nn.relu(tf.nn.conv2d(images, w_conv_1, strides=[1, 1, 1, 1], padding="SAME") + b_conv_1)
pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

#   second conv layer
w_conv_2 = get_weight([5, 5, 64, 64])
b_conv_2 = get_bias([64])
h_conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, w_conv_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv_2)
pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

#   third conv layer
w_conv_3 = get_weight([3, 3, 64, 128])
b_conv_3 = get_bias([128])
h_conv_3 = tf.nn.relu(tf.nn.conv2d(pool_2, w_conv_3, strides=[1, 1, 1, 1], padding="SAME") + b_conv_3)
pool_3 = tf.nn.max_pool(h_conv_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

#   fourth conv layer
w_conv_4 = get_weight([3, 3, 128, 128])
b_conv_4 = get_bias([128])
h_conv_4 = tf.nn.relu(tf.nn.conv2d(pool_3, w_conv_4, strides=[1, 1, 1, 1], padding="SAME") + b_conv_4)
pool_4 = tf.nn.max_pool(h_conv_4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

#   last conv layer
w_conv_5 = get_weight([3, 3, 128, 128])
b_conv_5 = get_bias([128])
h_conv_5 = tf.nn.relu(tf.nn.conv2d(pool_4, w_conv_5, strides=[1, 1, 1, 1], padding="SAME") + b_conv_5)

#   remaining the last conved layer
flat = tf.reshape(h_conv_5, [-1, 14 * 14 * 128])

#   dense layer 1
w_dense_1 = get_weight([14 * 14 * 128, 384])
b_dense_1 = get_bias([384])
h_dense_1 = tf.nn.relu(tf.matmul(flat, w_dense_1) + b_dense_1)
drop_1 = tf.nn.dropout(h_dense_1, drop_prob)

#   dense layer 2
w_dense_2 = get_weight([384, label_len])
b_dense_2 = get_bias([label_len])
logits = tf.matmul(drop_1, w_dense_2) + b_dense_2
y_pred = tf.nn.softmax(logits)

#   loss, optimization and accuracy of the model
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=answers, logits=logits))
opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(answers, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCH):
        img, ans = get_next_batch(BATCH_SIZE, "train")
        print("Epoch", epoch)
        pred = sess.run(h_conv_5, feed_dict={images:img, answers:ans, drop_prob:0.8})
        print(epoch, ": pred:", pred)
        print(pred.shape)