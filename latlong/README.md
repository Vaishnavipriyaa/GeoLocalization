Overview
This focuses on predicting the latitude and longitude of images, given their visual content. The model is built using the EfficientNet architecture, fine-tuned for regression tasks. The training data consists of images with known latitude and longitude values, and the goal is to predict these geographical coordinates for new images. We utilize a clean dataset and handle anomalies during the training process to improve model performance.

Approach
1. Model Architecture
Type: EfficientNet (a convolutional neural network based on EfficientNet B0)

Pre-training: The EfficientNet B0 model is pre-trained on ImageNet and fine-tuned on the specific task of latitude and longitude prediction. The model output is a single value (latitude or longitude), with a final regression layer to predict the coordinates.

2. Data Preprocessing
Image Transformation: We apply standard image transformations such as resizing, normalization, random rotation, and random horizontal flipping for data augmentation. The images are resized to 224x224 pixels and normalized using the mean and standard deviation of ImageNet images.

Normalization of Latitude and Longitude: Latitude and longitude values are normalized using min-max scaling based on the training dataset's minimum and maximum values, ensuring that the model is trained on a consistent scale.

3. Handling Anomalies
Anomalies in Data: The dataset contains a few anomalies (e.g., outlier latitudes and longitudes). These anomalies are identified and removed before training the model, ensuring that the model's learning process is not biased by erroneous data.

4. Model Training
Optimizer: Adam optimizer is used with a learning rate of 1e-3 and a learning rate scheduler that decays the learning rate by a factor of 0.5 every 5 epochs.

Loss Function: We use Mean Squared Error (MSE) as the loss function, which is standard for regression tasks. The goal is to minimize the error between the predicted and actual latitude and longitude values.

5. Early Stopping
Patience: To avoid overfitting, early stopping is implemented with a patience of 10 epochs. If the validation loss does not improve after 10 consecutive epochs, training will stop early to save time and prevent overfitting.

6. Model Evaluation
The performance of the model is evaluated based on the validation loss. The model with the best validation loss is saved and used for inference.

Files Included
Solution Code:

main.ipynb contains the entire pipeline, from data loading, model training, to evaluation and early stopping.

Solution CSV:

solution.csv contains the final predictions for latitude and longitude for test or validation images.

README.md:

This file describes the project, model, approach, and results.

Model Details
EfficientNet Lat-Long Model
The model consists of two branches:

Latitude Model: An EfficientNet-based model that predicts latitude.

Longitude Model: Another EfficientNet-based model that predicts longitude.

Both models share the same backbone but have different heads to predict latitude and longitude independently.

Key Components of the Model:
EfficientNet B0: A convolutional neural network known for its high efficiency and performance.

Head Layers: After the backbone, a series of fully connected layers (head and head1) are used to regress the output to a single scalar value (latitude or longitude).

Sigmoid Activation: A Sigmoid function is applied at the output layer to scale the output between 0 and 1 for both latitude and longitude.

Training and Evaluation
Data Preparation
Cleaning: Anomalies in latitude and longitude values are filtered out to improve model accuracy. The cleaned dataset is saved as train_cleaned.csv and val_cleaned.csv.

Data Augmentation: Images undergo random rotation and horizontal flipping to enhance the model’s ability to generalize.

Normalization: Latitude and longitude values are normalized based on the min-max scaling, calculated using the training dataset’s minimum and maximum values.

Training Loop
The model is trained for a maximum of 60 epochs, with the training process stopping early if there is no improvement in validation loss after 10 consecutive epochs (early stopping).

Loss Function: The training loss is calculated using Mean Squared Error (MSE) between the predicted and actual latitude and longitude values.

Optimizer: Adam optimizer is used with a learning rate of 1e-3, and a learning rate scheduler decays the learning rate by a factor of 0.5 every 5 epochs.

Best Model: The model with the lowest validation loss is saved. The best models for latitude and longitude predictions are stored as best_model_lat_anom.pt and best_model_long_anom.pt.

Validation
During validation, the model evaluates its performance on a held-out set of images (validation set) to compute the validation loss.

Early stopping is triggered if the validation loss doesn't improve for 10 consecutive epochs.

Model Weights:

The model weights are saved for latitude and longitude prediction as best_model_lat_anom.pt and best_model_long_anom.pt.

best_model_lat_anom:
(https://drive.google.com/file/d/1FVue2lWwPVw-sEb45hAl0XscJrUx6HJJ/view?usp=sharing)

best_model_long_anom:
https://drive.google.com/file/d/1WnQ-XiXKPSfze3QTOfkX4viC2HbdJ72L/view?usp=sharing
