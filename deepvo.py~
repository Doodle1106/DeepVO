# encoding=utf-8
import tensorflow as tf
# import re
import os
import sys
import argparse
FLAGS = None

cell_dim = 1000
batch_size = 32

def conv_net(input):
    # input = tf.reshape(input, [-1, 480, 752, 1])
    conv1 = tf.layers.conv2d(input, 64, 7, (2, 2), activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 128, 5, (2, 2), activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, 256, 5, (2, 2), activation=tf.nn.relu)
    conv3_1 = tf.layers.conv2d(conv3, 256, 3, (1, 1), activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3_1, 512, 3, (2, 2), activation=tf.nn.relu)
    conv4_1 = tf.layers.conv2d(conv4, 512, 3, (1, 1), activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv4_1, 512, 3, (2, 2), activation=tf.nn.relu)
    conv5_1 = tf.layers.conv2d(conv5, 512, 3, (1, 1), activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(conv5_1, 1024, 3, (2, 2), activation=tf.nn.relu)
    # fc1 = tf.contrib.layers.flatten(conv6)
    # logits = tf.layers.dense(fc1, 7)
    print ('Last CNN layer shape : {} '.format(conv6))
    return conv6


def lstm(input, batch_size):

    # lstm_cell_1 = BasicConvLSTMCell.BasicConvLSTMCell([4, 8], [3, 3], 1024)
    # initial_state = lstm_cell_1.zero_state(batch_size, tf.float32)
    lstm_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[12, 8, 1024], kernel_shape=[3, 3], output_channels=1)

    hidden = lstm_cell.zero_state(batch_size, tf.float32)
    y_1, _ = lstm_cell(input, hidden)
    y_1_flattened = tf.layers.Flatten()(y_1)
    output = tf.layers.dense(y_1_flattened, 7)

    # stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(2)])
    # initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
    # print ("-------")
    # print (initial_state)
    # print (input)
    # outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, input,
    #                                          initial_state=initial_state, scope='LSTM')
    # # output = tf.reshape(outputs, [-1, cell_dim])
    # output = tf.reshape(outputs, [-1, 7])
    return output

def read_images():
    folder = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/data/'
    imagepaths, labels = list(), list()
    for img in os.listdir(folder):
        print (img)
        imagepaths.append(folder + img)
        labels.append(1)
    return imagepaths, labels

def read_and_decode(file_name):
    filename_queue = tf.train.string_input_producer(file_name)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'p_x': tf.FixedLenFeature([], tf.float32),
                                           'p_y': tf.FixedLenFeature([], tf.float32),
                                           'p_z': tf.FixedLenFeature([], tf.float32),
                                           'q_w': tf.FixedLenFeature([], tf.float32),
                                           'q_x': tf.FixedLenFeature([], tf.float32),
                                           'q_y': tf.FixedLenFeature([], tf.float32),
                                           'q_z': tf.FixedLenFeature([], tf.float32)
                                       })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [960, 752, 3])
    image = tf.cast(image, tf.float32)
    label = [features['p_x'],
             features['p_y'],
             features['p_z'],
             features['q_w'],
             features['q_x'],
             features['q_y'],
             features['q_z']]

    # print (image)
    # print (label)
    # print (np.shape(image))
    return image, label


def main(self):

    print (tf.__version__)
    filename = [FLAGS.data_dir]
    print (filename)
    image, label = read_and_decode(filename)
    print ('---------------------------')
    print (image)
    print (label)
    print ('---------------------------')

    image_batch, label_batch = tf.train.batch(
        [image, label], batch_size=32, capacity=1000+64)

    print ('batch fetched successfully')

    print (image_batch)
    print (label_batch)

    X = tf.placeholder("float", [None, 960, 752, 3])
    Y = tf.placeholder("float", [None, 7])

    conv_output = conv_net(X)
    logits = lstm(conv_output, 32)
    print ("logits is {}".format(logits))

    loss_op = tf.reduce_mean(tf.square(tf.subtract(Y, logits)))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op)
    epoch_num = 500
    batch_num = 100
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init)
        for _ in range(epoch_num):
            for i in range(int(1710/32)):
                feature, label = sess.run([image_batch, label_batch])
                # feed batch
                _, c = sess.run([train_op, loss_op], feed_dict={X: feature,
                                                                Y: label})
                print ("Current Training Error is {}".format(c))
        coord.request_stop()
        coord.join(threads)

        save_path = saver.save(sess, FLAGS.model_dir)
        print ("Training Finished! To path {}".format(save_path))
        print("Run the command line:\n" \
              "--> tensorboard --logdir=/tmp/tensorflow_logs " \
              "\nThen open http://0.0.0.0:6006/ into your web browser")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='',
                        help='input data path')
    parser.add_argument('--model_dir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
    

    # init_op = tf.group(tf.global_variables_initializer(),
    #                    tf.local_variables_initializer())


    # with tf.Session() as sess:
    #
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #     print ('In the session')
    #     # coord = tf.train.Coordinator()
    #     # threads = tf.train.start_queue_runners(coord=coord)
    #
    #     model = tf.estimator.Estimator(model_fn)
    #     # print ('Fetching data')
    #     # img, lbl = sess.run([image_batch, label_batch])
    #     print ('fetched data')
    #
    #     input_fn = tf.estimator.inputs.numpy_input_fn(
    #         x={'image': image_batch},
    #         y=label_batch,
    #         batch_size=32, num_epochs=1, shuffle=False)
    #     print ('During Training')
    #
    #     model.train(input_fn)
    #
    #     input_fn = tf.estimator.inputs.numpy_input_fn(
    #         x={'image': image_batch},
    #         y=label_batch,
    #         batch_size=32, num_epochs=1, shuffle=False)
    #     # Use the Estimator 'evaluate' method
    #     e = model.evaluate(input_fn)
    #     print("Testing Accuracy:", e['accuracy'])
    #     # coord.request_stop()
    #     #
    #     # # Wait for threads to stop
    #     # coord.join(threads)
    #     sess.close()

    # logits_train = conv_net(image_batch)
    # print (logits_train)
    #
    # loss_op = tf.reduce_mean(tf.square(logits_train - label_batch))
    # print (loss_op)
    #
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # print (optimizer)
    #
    # train_op = optimizer.minimize(loss_op)
    #
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    #
    # print ('before entering a session')
    #
    # with tf.Session() as sess:
    #
    #     # Run the initializer
    #     print ('sess.run')
    #     sess.run(init)
    #
    #     # # Start the data queue
    #     # print ('start queue runner')
    #     # tf.train.start_queue_runners()
    #
    #     # Training cycle
    #     print ('start training ...')
    #     for step in range(1, 101):
    #         print ('in the loop')
    #         sess.run(train_op)





    # ----------------------------uncomment it if you want to visualize fetched bathces---------------------
    # Initialize all global and local variables

    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     # Create a coordinator and run all QueueRunner objects
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     for batch_index in range(5):
    #         img, lbl = sess.run([image_batch, label_batch])
    #
    #         img = img.astype(np.uint8)
    #
    #         for j in range(6):
    #             plt.subplot(2, 3, j + 1)
    #             plt.imshow(img[j, ...])
    #             # plt.title('cat' if lbl[j] == 0 else 'dog')
    #
    #         plt.show()
    #
    #     # Stop the threads
    #     coord.request_stop()
    #
    #     # Wait for threads to stop
    #     coord.join(threads)
    #     sess.close()

