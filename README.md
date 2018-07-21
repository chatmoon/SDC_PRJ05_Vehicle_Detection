# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal is to write a software pipeline to detect vehicles in a video.  


Installation
---

Clone the Github repository and change the directory path in the `main.py` file in line 576. Namely, it should be the project repository path in your local:

```python
def main():
    ## parameter
    directory = 'D:/tmp/CarND-Vehicle-Detection'
    args      = PARSE_ARGS(path=directory)
    var       = parameters()

    ## dectect vehicle:
    #  {0: "test_video.mp4", 1: "project_video.mp4"}[mp4]
    tracker_cars(args, mp4=1)
```

Usage
---

Edit the `tracker_cars()` function in the `main.py` file in line 581, and select the `mp4` attribut value such as:
- `mp4=0` : "test_video.mp4"
- `mp4=1` : "project_video.mp4"

By default, the video called `project_video.mp4` is the video used by the pipeline. The video ouput is saved into the root directory.   


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing the pipeline on single frames are located in the `test_images` folder.  

To help, the output from each stage of the pipeline are saved in the folder called `ouput_images`, and they are included in the writeup for the project.    

The video called `project_video.mp4` is the video the pipeline works well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!


