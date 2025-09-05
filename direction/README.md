Overview
This aims to predict the directional angle of images (0 to 360 degrees) using a deep learning model. The model utilizes a Vision Transformer (ViT) architecture, fine-tuned on a dataset consisting of images and corresponding angle labels. The solution employs a cosine similarity-based loss function to optimize the model for angle prediction, achieving the task's objective effectively.

The dataset consists of labeled training images and corresponding angles, and the model is evaluated based on the Mean Angular Absolute Error (MAAE) metric.

Approach
1. Model Architecture
Type: Vision Transformer (ViT)

Pre-training: The model is fine-tuned using a pre-trained vit_large_patch16_224 model from the timm library.

Final Layer: The final layer consists of a simple fully connected (FC) layer with two output units representing the cosine and sine components of the angle.

2. Data Preprocessing
Resize: All images are resized to 224x224 pixels for consistency.

Augmentation: Random horizontal flipping and slight color jitter (brightness and contrast adjustments) are applied for data augmentation.

Normalization: Images are normalized with a mean and standard deviation of (0.5, 0.5, 0.5) for the RGB channels.

Angle Representation: The target angle is represented as a 2D vector using the cosine and sine of the angle in radians. This transformation allows the model to predict the angle direction more accurately using vector-based loss functions.

3. Training
Optimizer: AdamW with a learning rate of 2e-5 and weight decay of 1e-4 is used to optimize the model.

Learning Rate Scheduler: A cosine annealing scheduler is employed to adjust the learning rate throughout the training process, allowing the model to converge more efficiently.

Loss Function: The custom loss function is based on the cosine similarity between the predicted and ground truth vectors. The loss is minimized during training to ensure accurate angle predictions.

4. Evaluation Metric
Mean Angular Absolute Error (MAAE): This metric computes the mean of the smallest angular differences between predicted and true angles, considering that angles are cyclical (i.e., angles close to 0 and 360 degrees are similar).

5. Early Stopping
The model training includes early stopping with a patience of 5 epochs, meaning training stops if there is no improvement in validation MAAE for 5 consecutive epochs.

6. Final Model
The best model (with the lowest MAAE on the validation set) is saved and used for generating predictions on both the validation and test datasets.

Files Included
Solution Code:

main.ipynb contains the entire pipeline from data loading, training, evaluation, and final submission generation.

Model Weights:

Solution CSV:

solution.csv contains the final predictions for both validation and test images, with columns id and angle.

README.md:

This file describing the project, model, and approach.

Innovative Ideas
Angle Representation: Instead of predicting a single angle directly, we represent the angle as a 2D vector using the cosine and sine. This approach helps to predict angles more accurately by leveraging vector operations like cosine similarity and avoids issues like angle wrapping.

Vision Transformer: Using a pre-trained ViT model allows the network to leverage transfer learning from large-scale image datasets. The ViT architecture, known for its success in image classification tasks, is fine-tuned for this regression problem.

Model Evaluation
The model achieved a significant reduction in Mean Angular Absolute Error (MAAE) on the validation set, demonstrating its effectiveness in angle prediction. Early stopping and the use of a cosine-based loss function ensured that the model didn't overfit the data and generalized well to unseen images.

Submission
The final output is saved as solution.csv, which includes predictions for both the validation and test datasets. The submission includes the necessary files required for the competition.

The model weights are hosted on a cloud storage platform-Google Drive. Here is the link to the weights:

https://drive.google.com/file/d/1Ih4qc1lBmyxRgUaVMCTJlXHqMzP5T-M7/view?usp=sharing

