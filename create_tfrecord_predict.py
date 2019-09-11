import cv2
import tensorflow as tf
import numpy as np
import glob
from random import shuffle
import os
import sys

labels=list()
addrs=list()
shuffle_data = True

print(os.getcwd())
test_path="Data_dir/Predict/*.jpg"

addrs = glob.glob(test_path)
for addr in addrs:
    if "baseball" in addr:
        labels.append(0)
    elif "golf" in addr:
        labels.append(1)
    elif "soccer" in addr:
        labels.append(2)
    elif "skiing" in addr:
        labels.append(3)
    elif "rowing" in addr:
        labels.append(4)
    elif "bmx" in addr:
        labels.append(5)
    else:
        labels.append(6)


# train_addrs = addrs[0:int(0.6*len(addrs))]
# train_labels = labels[0:int(0.6*len(labels))]
# print(len(train_labels))
# test_addrs = addrs[int(0.6*len(addrs)):]
# test_labels = labels[int(0.6*len(labels)):]
# print(len(test_labels))
def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = 'predict.tfrecords'
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(addrs)):
    # print how many images are saved every 1000 images
    if not i % 100:
        print ('Predicted data: {}/{}'.format(i, len(addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(addrs[i])
    
    # Create a feature
    feature = {'predict/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()