"""Module for training FCN

Usage as follows:

    # Build a new model and train it
    fcn = FCN(session)
    fcn.load_vgg()

"""

# -*- coding: utf-8 -*-
import math
import os.path
import tensorflow as tf
import helper
import tests
from tqdm import tqdm

# from dataset import *

class FCN(object):

    '''
    Constructor for setting params
    '''
    def __init__(self, session, options={}):
        self.session = session

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

        # self.dataset = Dataset()

    '''
    Load the VGG16 model
    '''
    def load_vgg(self):

        # Load the saved model
        tf.saved_model.loader.load(self.session, ['vgg16'], self.vgg_path)

        # Get the relevant layers for constructing the skip-layers out of the graph
        graph = tf.get_default_graph()

        self.input_image = graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        self.layer_3     = graph.get_tensor_by_name('layer3_out:0')
        self.layer_4     = graph.get_tensor_by_name('layer4_out:0')
        self.layer_7     = graph.get_tensor_by_name('layer7_out:0')


