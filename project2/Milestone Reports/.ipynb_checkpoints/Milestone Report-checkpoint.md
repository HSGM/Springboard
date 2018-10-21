### Capstone Project 2 Milestone Report.

Build an image classifier that will help label images belonging to different rooms in a home.

Usefulness of project/prospective clients: Real Eastate Agents/ Vacation home Rental agencies
A real estate agent may visit several homes in aday and take many fotos of homes. These fotos then woudl need to be manually sorted by room type for display. This could be labor intensive. A model that automatically classifies and labels the images would be a time saver and enable easy retrieval of images when necessary.

**Source of Images:** Images downloaded from here. 
                   http://web.mit.edu/torralba/www/indoor.html 
                   
                  Images belonging to the following categories was  used.
                  1. Bathroom
                  2. Bedroom
                  3. Corridor
                  4. Dining Room
                  5. Kitchem
                  6. Living Room
                  7. Pantry
                  8. Staircase
                 
   A total of 150 images per category was used. 130 images were stored in a folder called train and was used for training the models. Each category of image was in a subfolder labelled by the category name. 
   20 images per category was stored in a folder called test/category and reserved for testing the model.
   
The images were visually inspected for mislabelling prior to using for training. 


1. Different pretrained neural networks ( vgg10, resnet, inceptionv3) were used to extract features and a logistic regression was fitted to each set of features thus obtained,
Each image had a total of 13,000 features. This required a long time for processing even on an AWS ec3 system. Also 150 -800 images were used for the different categories initially.
 
2.The project was rescaled to 150 images per category for the ease of processing and secondarily to work with a balanced number of images per category. FEatures were extracted from the one but the last layer ( bottleneck layer). Form this layer  each image had 2048 features. FEatures were extracted using only Inception V3 which seemed to perform the best.

3. The programs were run on a AWS EC2 machine.

4. The features extracted were stored as HDF5 files that were retrieved and fed into a machine learning algorithms.

The following Algorithms were tried out:

1. Logistic
2. Random Forest
3. Support Vector
4. XG boost. 

The programs were run in parallel to save time that the EC2 machine was run. Scikit learn 
was used to optimise the first three models. The hyper parameter space was optimised using a gridsearch and 10 fold cross validation.

The package XGBOOST was used to fit the XGBOOST Model. THe hyperparametes space was optimised 
using gridsearch and CV.

The optimised models were stores and evaluated for performance using the test images. 
Evaluation metrics used were Accuracy, Precision, Recall , F1-score and confusion matrix.

All models were comparable in performance with about 88% for accuracy.




   
   
  
   