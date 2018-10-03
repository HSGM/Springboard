
### Introduction

Vacation Time!! I wish!! As the summer comes to an end I think back to our family vacation by the beach. In planning for this much valued time together, I spent several hours scoring through the web for vacation rentals, looking for the idyllic place. I compared homes, their price, the location, the setting, the rooms, all with my dear family in mind. Whilst engaged thus, I wondered how one would build a visual image classifier to classify the different rooms in a home. It would probably help not only vacation home renters and rental agencies but also in real estate. Instead of laboriously manually labeliing every photo, with a image classifier the photos would automatically be sorted into the right category to be displayed as needed. So here is how i went about buildng an image classifier to label the photos of romms in a home.

Images:

The data set: The images used for this project is a subset of the images from here.

http://web.mit.edu/torralba/www/indoor.html

150 images for each of the following categories was included in this project. Dining room Kitchen Bathroom Bedroom Staircase Corridor Pantry.
Twnety of these images were  reserved in a folder to test the model and the other 130 images per room were used to build the model. 

Here are a few of the images :
(Include a few images here)


Building a custom deep learning model from scratch, requires extensive computation resources and lots of training data. However there are models already built that perform pretty well in classifying images from various categories. One such model is the Inception V3 model built by the Google Brain team for the ImageNet Visual Recognition Challenge. This model is available in Keras and I will use this model not as a classifier but to extract the features of the images. Once the features are extracted I will fit the following machine learning models.

Logistic
Random Forest
Support Vector Machines
XGBOOSt
Voting Classifier.

#### Feature Extraction Using Inception V3:

We first use the following code to get the model, along with the pre-trained weights and store it in a variable called base_model.

```python

from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(include_top=True, weights='imagenet', input_tensor=Input(shape=(299,299,3)))

```
The final layer of the network is a fully connected layer that separates out the 1000 different object categories in the ImageNet database. So using the following code I remove this final layer and save the resultant network without the final layer in a new variable called model.

```python
base_model = InceptionV3(include_top=True, weights=weights, input_tensor=Input(shape=(299,299,3)))
model = Model(input=base_model.input, outputs=base_model.get_layer('avg_pool').output)
image_size = (299, 299)

```
This new model will no longer return a predicted image class, since the classification layer has been removed; however, the CNN now stored in model still provides us with a useful way to extract features from images. By passing each of the images through this model, we can convert each image from its 299x299x3 array of raw image pixels to a vector with 2048 entries. In practice, this dataset of 2048-dimensional points is referred to as InceptionV3 bottleneck features. These features along with its labels are stored locally using HDF5 file format. The full code for this is stored in the github repository: link to repo.

Training A Machine Learning Model: The features and labels extracted by means of the inception v3 model can now be used to train a machine learning classfier. We trained the following models and finetuned the parameters. For details see the jupyter book for each model.

1. Logistic
2. Random Forest
3. Support Vector Machines
4. XGBOOST


Testing the Models:
The fine tuned models using the training data was next evaluated using the test images. The precision, recall and f1 scores are 


#### Evaluation Metrics



**Logistic Regression**

|Room     |  precision_score	|recall_score|	f1_score|	support|
|---------|---------------------|------------|----------|----------|
|bathroom |		0.90|	0.90|	0.90|	20.0|
bedroom	|0.83	|0.75	|0.79	|20.0|
corridor|		1.00|	1.00|	1.00|	20.0|
dining_room|	0.82|	0.90|	0.86|	20.0|
kitchen |0.94	|0.75	|0.83	|20.0|
living room|0.71|	0.75|	0.73|	20.0|
pantry|		0.91|	1.00|	0.95|	20.0|
staircase|		0.95|	1.00|	0.98|	20.0|
**avg/total**|	0.88|	0.88|	0.88|	160.0|




**Random Forest**

|Room     |  precision_score	|recall_score|	f1_score|	support|
|---------|---------------------|------------|----------|----------|
|bathroom |	       	0.90|	0.95|	0.93|	20.0|
|bedroom|		0.80|	0.80|	0.80|	20.0|
|corridor	|	0.91|	1.00|	0.95|	20.0|
|dining_room	|	0.73|	0.95|	0.83|	20.0|
|kitchen	|	1.00	|0.70|	0.82|	20.0|
|living room|	0.81|	0.65|	0.72|	20.0|
|pantry	|	0.95|	1.00	|0.98	|20.0|
|staircase|		0.95|	0.95|	0.95|	20.0|
|**avg/total**	|	0.88	|0.88|	0.87|	160.0 |



**Support Vector**

|Room     |  precision_score	|recall_score|	f1_score|	support|
|---------|---------------------|------------|----------|----------|
bathroom |		0.89|	0.85|	0.87|	20.0|
bedroom	|	0.82|	0.90|	0.86|	20.0|
corridor|		1.00|	0.95|	0.97|	20.0|
dining_room	|	0.75|	0.90|	0.82|	20.0|
kitchen	|1.00|	0.65|	0.79|	20.0|
living |	0.75|	0.75|	0.75|	20.0|
pantry	|	0.91|	1.00	|0.95	|20.0|
staircase	|0.95|	1.00|	0.98|	20.0|
**avg/total**	|	0.88|	0.88|	0.87|	160.0|


**XGBoost**

|Room     |  precision_score	|recall_score|	f1_score|	support|
|---------|---------------------|------------|----------|----------|
bathroom |	0.89 |	0.80 |	0.84 |	20.0 |
bedroom	 |	0.76 |	0.80	 |0.78	 |20.0 |
corridor |	0.95 |	0.95 |	0.95 |	20.0 |
dining_room	 |0.73	 |0.95 |	0.83 |	20.0 |
kitchen	 |0.88 |	0.70 |	0.78 |	20.0 |
living room	 |0.72	 |0.65	 |0.68	 |20.0 |
pantry	 |0.95 |	1.00 |	0.98	 |20.0 |
staircase	 |	0.95 |	0.95 |	0.95 |	20.0 |
**avg/total**|	0.85|	0.85|	0.85|	160.0|


The Confusion Matrices for each of the models :

**Logistic Regression**

|Predicted	True|Bathroom|bedroom|corridor|dining room	|kitchen|livingroom|pantry|staircase|
|-----------|--------|-------|--------|-------------|--------|---------|-----|----------|								
Bathroom|	18|	1|	0|	0|	0|	0|	1|	0|
bedroom|1|	1	|15	|0	|1	|0	|3	|0	|0|
corridor|2	|0	|0	|20	|0	|0	|0|	0|	0|
dining room|3|	0	|0	|0|	18	|0	|1	|1	|0|
kitchen|	1|	0|	0|	2	|15	|2|0|	0|
living room|	0	|2	|0	|1	|1|	15	|0	|1|
pantry|	0	|0	|0	|0|	0	|0|	20|	0|
staircase|	0|	0|	0|	0|	0|	0|	0|20







