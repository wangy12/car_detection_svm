## Vehicle Detection and Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.

Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg

[image31]: ./output_images/heatmap.png
[image32]: ./output_images/heatmap2.png
[image33]: ./output_images/heatmap3.png
[image34]: ./output_images/heatmap4.png
[image35]: ./output_images/heatmap5.png
[image36]: ./output_images/heatmap6.png
[image37]: ./output_images/heatmap6_furthersearch.png

[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

 

---


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images (the 10th code cell of the IPython notebook).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

The code for hog feature extraction is contained in the 5th code cell of the IPython notebook, which is a function `get_hog_features()`. The functions for binned color features and color histogram features are also defined and used, which can be found in the 4th and the 8th code cells. 

I tuned the parameters (`orientations`, color spaces, `pixels_per_cell`) in `skimage.hog()` while training the SVM classifier, and found that the test accuracy is greater when `color_space` is `HLS`, `HSV`, `YUV`, or `YCrCb`, and when the `orientations` is large.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled the HOG parameters:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
```

The more features extracted, the more accuracy the classifier. But there is a tradeoff between number of features and the processing time. When I was working on the fine tuning of HOG parameters, for example, I found that the vehicle detection performance is almost the same when `orient = 9` comparing to `orient = 12`, but the computational time (e.g., second per iteration) is greater, so I set `orient = 9`.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM (the 12th code cell) using HOG features, binned color features and color histogram features. The performance of classifier with different features is shown in the following table with the above-mentioned parameters.


| HOG feature   | Binned color feature   | Color histogram feature  | Number of features  | Accuracy |
| :-----------: |:----------:|:--------:|:---------:| :-----:|
| True      | False | False | 5292 |  0.981 |
| False      | True      |   True | 3168  | 0.897 |
| True | True      |   True | 8460 |  0.988 |

### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the `Hog Sub-sampling Window Search` method (defined in the 8th code cell). The window size hard-coded in `find_car()` is 64x64. By using a `scale_list`, I search for cars in different scales. The `scale_list` for testing is [1.3, 1.5, 1.8], which corresponds to window size of [83, 96, 115].

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier? Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image:

![alt text][image31]
![alt text][image32]
![alt text][image33]
![alt text][image34]
![alt text][image35]
![alt text][image36]

The test results show that YCrCb color space works better than HSV in detecting vehicles.

In the pipeline, I search for vehicles in different ROI using different scales for different scenarios. For example, when cars are not detected in the previous frame, `scale_list=[1.3, 1.5, 1.8]` for ROI (which is the road area); when cars are detected in the previous frame, `scale_list=[1.5]` for ROI (which is the road area), `scale_list=[1.3, 1.5, 1.8]` for where cars appear in the previous frame whose central pixel of `y` axes is smaller than 500, and `scale_list=[1.5, 1.8]` for where cars appear in the previous frame whose central pixel of `y` axes is not smaller than 500. This is in the 16th code cell.

Here is a test result of searching cars in ROI where cars are detected in the previous frame:

![alt text][image37]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/77DjaYa5k-E)

Here's a [link to my video result of both lane finding and car detection](https://youtu.be/V1aF9ymvK5Q)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. At first, I used the small set of training data (.jpeg), which has 1196 car images and 1125 non-car images. But the performance of vechicle detection is really poor because most of the car images are captured from similar angles and the training size is small. When I load the training data set (.png) which has 8792 images of cars and 8968 images of non-cars, the detection performance by the same SVM classifier can be guaranteed.

2. The centers of boxes containing detected cars are computed and recorded. In each frame, the detection results of the previous frame are used to narrow down a search area (ROI) for already detected vehicles. New targets (cars) are also searched from the whole road area.

3. The bounding boxes (tracking results) are smoothed by summation and thresholding the bounding boxes in previous frames.

4. In the result video, there are some frames (less than 4) that two detections are generated from the same vehicle. 

5. More object tracking methods should be explored, such as track initialization, data association, track ID management, etc. 




