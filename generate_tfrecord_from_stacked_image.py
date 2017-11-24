import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import linecache


dest_path = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/stacked.tfrecords'
writer = tf.python_io.TFRecordWriter(dest_path)
image_folder = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/stacked_deleted/'
label_csv = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/stacked_img_label.csv'

def make_tfrecord():

    print ("There are {} images in the folder").format(len(os.listdir(image_folder)))
    l = 2
    f = os.listdir(image_folder)
    f.sort()
    for img in f[:-1]:
        # print ("Image name is: {}".format(img))
        print (l)
        img_path = image_folder + img
        # print (img_path)
        img = Image.open(img_path)
        img_binary = img.tobytes()

        # data = tf.train.Example(features=tf.train.Features(feature={
        #     'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_binary])),
        #     "label_x": tf.train.Feature(float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[1])])),
        #     "label_y": tf.train.Feature(float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[2])])),
        #     "label_z": tf.train.Feature(float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[3])]))
        # }))

        data = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[img_binary])),
            "p_x": tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[1])])),
            "p_y": tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[2])])),
            "p_z": tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[3])])),
            "q_w": tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[4])])),
            "q_x": tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[5])])),
            "q_y": tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[6])])),
            "q_z": tf.train.Feature(
                float_list=tf.train.FloatList(value=[float(linecache.getline(label_csv, l).split(',')[7])]))
        }))

        writer.write(data.SerializeToString())
        l += 1

        if (l-1) %10 == 0:
            if float(l-1)/len(os.listdir(image_folder)) == 1:
                print ("dataset generation finished !")
            else:
                print ("{} percent finished ...").format(float(l-1)/len(os.listdir(image_folder)))

    writer.close()

def decode_from_tfrecord(filename, num_epoch=None):

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    # example = tf.parse_single_example(serialized, features={
    #     'image': tf.FixedLenFeature([], tf.string),
    #     'label_x': tf.FixedLenFeature([], tf.float32),
    #     'label_y': tf.FixedLenFeature([], tf.float32),
    #     'label_z': tf.FixedLenFeature([], tf.float32)
    # })

    example = tf.parse_single_example(serialized, features={
        'image': tf.FixedLenFeature([], tf.string),
        'p_x': tf.FixedLenFeature([], tf.float32),
        'p_y': tf.FixedLenFeature([], tf.float32),
        'p_z': tf.FixedLenFeature([], tf.float32),
        'q_w': tf.FixedLenFeature([], tf.float32),
        'q_x': tf.FixedLenFeature([], tf.float32),
        'q_y': tf.FixedLenFeature([], tf.float32),
        'q_z': tf.FixedLenFeature([], tf.float32)
    })

    p_x = tf.cast(example['p_x'], tf.int32)
    p_y = tf.cast(example['p_y'], tf.int32)
    p_z = tf.cast(example['p_z'], tf.int32)
    q_w = tf.cast(example['q_w'], tf.int32)
    q_x = tf.cast(example['q_x'], tf.int32)
    q_y = tf.cast(example['q_y'], tf.int32)
    q_z = tf.cast(example['q_z'], tf.int32)

    # label_y = tf.cast(example['label_y'], tf.int32)
    # label_z = tf.cast(example['label_z'], tf.int32)
    image = tf.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, [752, 960, 1])
    image = tf.cast(image, tf.float32)


    # with tf.Session() as sess:
    #     print (sess.run(image))
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     # for i in range(20):
    #     print ("in the loop")
    #     # example, l1,l2,l3 = sess.run([image, label_x, label_y, label_z])
    #     example, l1 = sess.run([image, label])
    #     print ("img_savinddddg")
    #     img = Image.fromarray(example, 'RGB')
    #     print ("img_saving")
    #     img.save("/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/" + str(i) + '_''Label_' + '.png')
    #     print ("img_saving finished")
    #     print(example, l1)
    #     coord.request_stop()
    #     coord.join(threads)

    return image, p_x, p_y, p_z, q_w, q_x, q_y, q_z

if __name__ == '__main__':
    make_tfrecord()
    # image, label_x = decode_from_tfrecord(filename="/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/euroc_v102_medium_cam0.tfrecords",num_epoch=1)
    # print (image)
    # print (label_x)

    # print (label_y)
    # print (label_z)

