# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./test_images/13.jpg "13"
[image2]: ./test_images/17.jpg "17"
[image3]: ./test_images/25.jpg "25"
[image4]: ./test_images/36.jpg "36"
[image5]: ./test_images/39.jpg "39"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12360
* The shape of a traffic sign image is 32X32 pixels
* The number of unique classes/labels in the data set is 43



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to gray-scale because the traffic data features are more dependent on shapes and wanted my algorithm to give more weight to shapes. To calculate the gray-scale, I used the average of the RGB channels


As a last step, I normalized the image data with mean 0 and range(-1,1) to make it a well conditioned problem.

The amount of data/ label was highly uneven. Therefore, duplicated data to reduce unevenness. Did not make it exactly equal t leave some uncertainty.
If the number of images for  label was below 1000, I padded them until they were close to 1000. 
I did not augment the images in any way to make the algorithm robust. Instead used the model architecture to reduce over-fitting which will take care of this problem to certain extent.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Fully connected 		| Reshape to  400 								|
| Dropout				|												|
| Fully connected 		| Reshape to  120    							|
| RELU					|												|
| Fully connected 		| Reshape to 84     							|
| RELU					|												|
| Dropout				|												|
| Fully connected 		| Reshape to 43     							|


 This architecture is very similar to the LeNet for MNIST data with some additional Dropout and RELU layers


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with initial learning rate of 0.001 and exponential learning decay rate of 0.9999.
For EPOCHS, I chose 30 so that the model doesn't start over-fitting due to the Validation data seeping through.
For Batch size, I used 300 since I wanted the batch to have a good probability of having all the labels.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.8% 
* test set accuracy of 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used the LeNet architecture from out previous assignment for classification.
* What were some problems with the initial architecture?
The accuracy was limited to 90% and over-fitting of data causing it to drop with increasing number of epochs.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over-fitting or under-fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Adding dropout layers increased the accuracy considerably by eliminating over-fitting.
Also added an exponential decay term to the optimizer so that it continues to perform optimization at higher accuracy levels.
* Which parameters were tuned? How were they adjusted and why?
Hyper parameters tuned were
- Learning rate decay
- Epochs
- initial learning rate
- Batch size
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I chose to add dropout layers at alternate intervals between layers because there a was a good chance of overfitting due the highly repeatable dataset. With a droput probability of 0.5 half the values in each layer were dropped. This improved the ccuracy significantly.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]
These images were chosen from a test by another Udacity student to compare result. 
https://pechyonkin.me/portfolio/traffic-signs-classification/

These images were sourced from google maps data and can serve as a good comparision for my algortihm.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Yield					| Yield											|
| No entry      		| No entry   									|
| Road work     		| Road work 									|
| Go straight or right	| Go straight or right							|
| Keep left 			| Keep left 									|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.
The incorrect classification of the Keep left sigh was primarily due to a faded color and smooth edges on the arrow. However the softmax probabilyt for the corrct classification were a close second in this case.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Yield											|
| .99     				| No entry   									|
| .99					| Road work 									|
| .99	      			| Go straight or right							|
| .29				    | Keep left     								|

The keep left prediction is not robust due to the softmax probailities spilling over to Keep Right and Roundabout images(labels 38 and 40)


