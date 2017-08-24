# Semantic Segmentation
Self-Driving Car Engineer Nanodegree Program

By: [Eqbal Eki](http://www.eqbalq.com)

### Introduction

The objective of this project is to build a Fully Connected Convolutional Neural Net `FCN` to identify the road in a collection of pictures. The encoding of the FCN will be provided by a pre-trained `VGG16` model and the decoder will be built using 1x1 convolutions, upscaling and layer skipping.


### File structure

- `fcn.py` contains FCN class which is responsible of training, load pretained VGG model, build the model layers and even save the trained model. The main public function is `run`. It can be used with something like:

	```
    fcn = FCN()
    fcn.run()

	``` 
- `helpers.py` it has all the helpers functions include: checking compatibility and GPU device, Progress bar, check if the VGG model installed and if not download the model, generate training batches and saving inference samples.

- `main.py` is a wrapper for `FCN` so we can use it to run the model. 

- `tests.py` includes the basic tests like testing the layers and training.

- `playground` is a python notebook so I can test the components and document training the model. 

- `process_video.py` handles videos, segment the images and then process each using the trained model we generated. 

### Running the project
The project can be run with ease by invoking `python main.py` or by simply use the notebook and run `fcn.run` The way it is set up in this repo, using a GTX 1060 it takes about 10-15 minutes to train.


### Checklist
- [x] Ensure you've passed all the unit tests
- [x] Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
- [x] Handle a video sample.

### Running the tests

When run, the project invokes some tests around building block functions that help ensure tensors of the right size and description are being used.

I cleaned up the test code and added more debugging. The project has been restructured in a class to facilitate better passing of hyperparameters for training. You can use the default options but not passing anything to `FCN` init (ex: `FCN()`) or custom params by passing a hash with the needed params ex: 

	```
    params = {
        'learning_rate':   0.00001,
        'dropout':         0.5,
        'epochs':          100,
        'batch_size':      4,
        'init_sd':         0.01,
        'training_images': 289,
        'num_classes':     2,
        'image_shape':     (160, 576),
        'data_dir':        'data',
        'runs_dir':        'runs',
        'training_subdir': 'data_road/training',
        'save_location':   'data/fcn/'
    }
    fcn = FCN(params)
    fcn.run_tests()
    fcn.run()
	```

Once the tests are complete, the code checks for the existence of a base trained VGG16 model. If it is not found locally it downloads it. Once it is available, it is used as the basis for being reconnected with skip layers and recapping the model with an alternative top end for semantic segmentation.


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
