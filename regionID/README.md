Overview
This focuses on predicting the Region ID of satellite images using deep learning. The task is framed as a multi-class classification problem where each image must be assigned a Region ID (1–15). We use the Swin Transformer, a powerful vision transformer architecture, fine-tuned for this classification task.

Model Summary
Architecture: Swin Transformer swin_base_patch4_window7_224

Pretrained: Yes (on ImageNet)

Output: 15 classes (Region IDs from 1 to 15)

Loss Function: Label Smoothing Cross-Entropy

Optimizer: AdamW

Learning Rate Scheduler: Cosine Annealing

Inference Technique: Test Time Augmentation (TTA) using ttach

Data and Preprocessing
Dataset: Satellite image dataset with Region_ID labels for training and validation.

Data Cleaning: Images not found in the directory are dropped automatically.

Transforms: We use Albumentations for advanced augmentations:

Resize to 224×224

Horizontal and vertical flips

Random 90-degree rotations

Color jittering

Brightness and contrast adjustment

Normalization to standard ImageNet stats

Test Time Augmentation (TTA): Applied during inference with 8 standard geometric transforms using ttach.

Training Strategy
Training Time: Up to 25 epochs (with early stopping based on validation accuracy).

Early Stopping: Patience of 10 epochs to avoid overfitting.

Batch Size: 32

Loss Function: Label smoothing (ε=0.1) to prevent overconfidence and improve generalization.

Best Model: Saved as best_model.pth based on highest validation accuracy.

Evaluation
Metric: Classification Accuracy

Validation Accuracy: Tracked after every epoch and used to select the best-performing model.

Final Model: Used with TTA to generate predictions on both validation and test datasets.

Download the best model weights from the following link:

https://drive.google.com/file/d/1IPxw9RmKMJRXS499Pki9xRUBFFz_sHGz/view?usp=sharing
