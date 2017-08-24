"""Module for training FCN

Usage as follows:

    # Build a new model and train it
    fcn = FCN()
    fcn.run()

"""

# -*- coding: utf-8 -*-
import math
import os.path
import tensorflow as tf
import helper
import tests
from tqdm import tqdm

class FCN(object):

    '''
    Constructor for setting params
    '''
    def __init__(self, options={}):
        self.learning_rate = options.get('learning_rate', 0.00001)
        self.dropout = options.get('dropout', 0.5)
        self.epochs = options.get('epochs', 100)
        self.batch_size = options.get('batch_size', 4)
        self.init_sd = options.get('init_sd', 0.01)
        self.training_images = options.get('training_images', 289)
        self.num_classes = options.get('num_classes', 2)
        self.image_shape = options.get('image_shape', (160, 576))
        self.data_dir = options.get('data_dir', 'data')
        self.runs_dir = options.get('runs_dir', 'runs')
        self.training_subdir = options.get('training_subdir', 'data_road/training')
        self.save_location = options.get('save_location', 'data/fcn/')

        self.vgg_path = os.path.join(self.data_dir, 'vgg')
        self.training_path = os.path.join(self.data_dir, self.training_subdir)


    '''
    Main training routine
    '''
    def run(self):

        helper.check_compatibility()
        tests.test_for_kitti_dataset(self.data_dir)
        helper.maybe_download_pretrained_vgg(self.data_dir)

        # Define the batching function
        get_batches_fn = helper.gen_batch_function(self.training_path, self.image_shape)

        # TensorFlow session
        with tf.Session() as sess:

            # Placeholders
            self.learning_rate_holder = tf.placeholder(dtype = tf.float32)
            self.correct_label_holder = tf.placeholder(dtype = tf.float32, shape = (None, None, None, self.num_classes))

            # Define network and training operations
            self.load_vgg(sess)
            self.layers()
            self.optimize_cross_entropy()

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # Train the model
            self.train_nn(sess)

            # Save images using the helper
            helper.save_inference_samples(
                    self.runs_dir,
                    self.data_dir,
                    sess,
                    self.image_shape,
                    self.logits,
                    self.keep_prob,
                    self.input_image)

            # Save the model
            self.save_model(sess)

    '''
    Load the VGG16 model
    '''
    def load_vgg(self, session):

        # Load the saved model
        tf.saved_model.loader.load(session, ['vgg16'], self.vgg_path)

        # Get the relevant layers for constructing the skip-layers out of the graph
        graph = tf.get_default_graph()

        self.input_image = graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        self.layer_3     = graph.get_tensor_by_name('layer3_out:0')
        self.layer_4     = graph.get_tensor_by_name('layer4_out:0')
        self.layer_7     = graph.get_tensor_by_name('layer7_out:0')


    '''
    Truncated norm to make layer initialization readable
    '''
    def tf_norm(self):
        return tf.truncated_normal_initializer(stddev=self.init_sd)

    '''
    Define the layers
    '''
    def layers(self):

        # 1x1 convolutions of the three layers
        l7 = tf.layers.conv2d(self.layer_7, self.num_classes, 1, 1, kernel_initializer=self.tf_norm())
        l4 = tf.layers.conv2d(self.layer_4, self.num_classes, 1, 1, kernel_initializer=self.tf_norm())
        l3 = tf.layers.conv2d(self.layer_3, self.num_classes, 1, 1, kernel_initializer=self.tf_norm())

        # Upsample layer 7 and add to layer 4
        layers = tf.layers.conv2d_transpose(l7, self.num_classes, 4, 2, 'SAME', kernel_initializer=self.tf_norm())
        layers = tf.add(layers, l4)

        # Upsample the sum and add to layer 3
        layers = tf.layers.conv2d_transpose(layers, self.num_classes, 4, 2, 'SAME', kernel_initializer=self.tf_norm())
        layers = tf.add(layers, l3)

        # Upsample the total and return
        self.layers = tf.layers.conv2d_transpose(layers, self.num_classes, 16, 8, 'SAME', kernel_initializer=self.tf_norm())


    '''
    Optimizer based on cross entropy
    '''
    def optimize_cross_entropy(self):

        # Reshape logits and label for computing cross entropy
        self.logits   = tf.reshape(self.layers, (-1, self.num_classes), name='logits')
        correct_label = tf.reshape(self.correct_label_holder, (-1, self.num_classes))

        # Compute cross entropy and loss
        cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.correct_label_holder)
        self.cross_entropy_loss = tf.reduce_mean(cross_entropy_logits)

        # Define a training operation using the Adam optimizer
        self.train_op = tf.train.AdamOptimizer(self.learning_rate_holder).minimize(self.cross_entropy_loss)


    '''
    Define training op
    '''
    def train(self, session):

        # Iterate over epochs
        for epoch in range(1, self.epochs+1):
            print("Epoch: " + str(epoch) + "/" + str(epochs))

            # Iterate over the batches using the batch generation function
            total_loss = []

            get_batches_fn = helper.gen_batch_function(self.training_path, self.image_shape)
            batch = get_batches_fn(self.batch_size)
            size = math.ceil(self.training_images / self.batch_size)

            for i, d in tqdm(enumerate(batch), desc="Batch", total=size):

                # Create the feed dictionary
                image, label = d

                feed_dict = {
                    input_image   : image,
                    correct_label : label,
                    keep_prob     : self.dropout,
                    learning_rate : self.learning_rate_holder
                }

                # Train and compute the loss
                _, loss = session.run([self.train_op, self.cross_entropy_loss], feed_dict=feed_dict)

                total_loss.append(loss)

            # Compute mean epoch loss
            mean_loss = sum(total_loss) / size
            print("Loss:  " + str(loss) + "\n")


    '''
    Save the model
    '''
    def save_model(self):
        saver = tf.train.Saver()
        saver.save(self.session, self.save_location + 'variables/saved_model')
        tf.train.write_graph(self.session.graph_def, self.save_location, "saved_model.pb", False)


    '''
    Run the tests
    '''
    def run_tests(self):
        tests.test_load_vgg(self.load_vgg, tf)
        tests.test_layers(self.layers)
        tests.test_optimize(self.optimize_cross_entropy())
        tests.test_train_nn(self.train)


