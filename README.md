# Semantic Segmentation
Self-Driving Car Engineer Nanodegree Program

By: [Eqbal Eki](http://www.eqbalq.com)

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

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
