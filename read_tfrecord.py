import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def train_me():
    data_path = 'train.tfrecords'
    with tf.Session() as sess:
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['train/image'], tf.float32)
        
        label = tf.cast(features['train/label'], tf.int32)
        image = tf.reshape(image, [100, 100, 3])
        image = tf.image.rgb_to_grayscale(image)
        
        images, labels = tf.train.shuffle_batch([image, label], batch_size=100, capacity=30, num_threads=1, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for batch_index in range(7):
            img, lbl = sess.run([images, labels])
            img = img.astype(np.uint8)
            one_hot_lbl=np.eye(6)[lbl]
            # for j in range(6):
            #     plt.subplot(2, 3, j+1)
            #     plt.imshow(img[j, ...])
            #     if lbl[j]==0:
            #         plt.title('baseball')
            #     elif lbl[j]==1:
            #         plt.title('golf')
            #     elif lbl[j]==2:
            #         plt.title('soccer')
            #     elif lbl[j]==3:
            #         plt.title('skiing')
            #     elif lbl[j]==4:
            #         plt.title('rowing')
            #     elif lbl[j]==5:
            #         plt.title('bmx')
                
            # plt.show()
        coord.request_stop()
        coord.join(threads)
        sess.close()
        img=np.reshape(img, (len(img), 10000))
        return img, one_hot_lbl

def try_me():
    data_path = 'train.tfrecords'
    with tf.Session() as sess:
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['train/image'], tf.float32)
        
        label = tf.cast(features['train/label'], tf.int32)
        image = tf.reshape(image, [100, 100, 3])
        image = tf.image.rgb_to_grayscale(image)
        
        images, labels = tf.train.shuffle_batch([image, label], batch_size=720, capacity=30, num_threads=1, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        one_hot_lbl=np.eye(6)[lbl]
        coord.request_stop()
        coord.join(threads)
        sess.close()
        img=np.reshape(img, (len(img), 10000))
        return img, one_hot_lbl, lbl


def test_me():
    data_path = 'test.tfrecords'
    with tf.Session() as sess:
        feature = {'test/image': tf.FixedLenFeature([], tf.string),
                   'test/label': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['test/image'], tf.float32)
        
        label = tf.cast(features['test/label'], tf.int32)
        image = tf.reshape(image, [100, 100, 3])
        image = tf.image.rgb_to_grayscale(image)
        
        images, labels = tf.train.shuffle_batch([image, label], batch_size=480, capacity=30, num_threads=1, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        one_hot_lbl=np.eye(6)[lbl]
        coord.request_stop()
        
        coord.join(threads)
        sess.close()
        #print(len(img))
        img=np.reshape(img, (len(img), 10000))
        return img, one_hot_lbl, lbl

#d1,x=train_me()

# print(len(d1))
# # print(len(d1[0]))
# # print(d1[0])

# # print(len(z))
# print(len(x))


def predict_me():
    data_path = 'predict.tfrecords'
    with tf.Session() as sess:
        feature = {'predict/image': tf.FixedLenFeature([], tf.string)}
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        image = tf.decode_raw(features['predict/image'], tf.float32)

        image = tf.reshape(image, [100, 100, 3])
        image = tf.image.rgb_to_grayscale(image)
        
        images = tf.train.shuffle_batch([image], batch_size=19, capacity=30, num_threads=1, min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img = sess.run([images])
        print(type(img))
        print(img)
        img = np.array(img)
        img = img.astype(np.uint8)
        coord.request_stop()
        #print(img)
        coord.join(threads)
        sess.close()
        #print(len(img))
        img=np.reshape(img, 19, 10000)
        return img

#x_predict = predict_me()
#print(type(x_predict))
#print(x_predict)