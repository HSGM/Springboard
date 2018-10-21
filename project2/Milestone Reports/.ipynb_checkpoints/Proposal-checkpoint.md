### Project 2 Proposal:

**Objective:** Build an image classifier that will help label images belonging to different rooms in a home.

Usefulness of project/prospective clients: Real Eastate Agents/ Vacation home Rental agencies
A real estate agent may visit several homes in aday and take many fotos of homes. These fotos then woudl need to be manually sorted by room type for display. This could be labor intensive. A model that automatically classifies and labels the images would be a time saver and enable easy retrieval of images when necessary.

**Source of Images:** Images would be downloaded from here. 
                   http://web.mit.edu/torralba/www/indoor.html 
                   
                  Images belonging to the following categories will be used.
                  1. Bathroom
                  2. Bedroom
                  3. Corridor
                  4. Dining Room
                  5. Kitchem
                  6. Living Room
                  7. Pantry
                  8. Staircase
                 
       ( For reasons of resource limitations 150 images per category would be used for building the model)  
       

**Approach to building the classifier:**


1. Use Inception-V3 neural net with imagenet weights to extract features.
2. Feed these features into machine learning algorithms: 
   Logistic
   Random Forest
   Support Vector
   XGBoost
   
3. Evaluate models using Accuracy, Precision, Recall, F1-score using test images(  20 images per category)

**Output:**

1. The final optimised models will be saved and uploaded to a github repository along with 
 a powerpoint presentation, a report and jupyter notebooks.
       
 
 
                  