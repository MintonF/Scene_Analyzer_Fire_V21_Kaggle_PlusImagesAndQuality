
This Python code implements a robust **deep learning workflow** for binary classification using **transfer learning** with the **MobileNetV2** architecture. The pipeline is structured into specific functions to cover the entire machine learning lifecycle, including dataset preparation, training, evaluation, and visualization. It also incorporates various metrics and methods to assess model performance and tuning flexibility.

The key processes included in the code are:
1. **Dataset Splitting**: Organizing a raw dataset into train, validation, and test subsets.
2. **Transfer Learning**: Training a customized **MobileNetV2** model with fine-tuned layers.
3. **Evaluation Metrics**: Computing diverse metrics such as **confusion matrix**, **ROC-AUC**, **PR-AUC**, and specificity.
4. **Visualization**: Graphically plotting the confusion matrix, accuracy/loss curves, and performance curves (ROC and Precision-Recall).

The modularity of the code allows for easy adjustments, reproducibility, and extensibility to similar projects. Below, each functionality is detailed.

---

### **Functionality Breakdown**

---

#### **1. Dataset Splitting: `split_dataset`**

Functionality:
- Takes directories containing raw datasets organized by class (`input_dirs`) and splits them into **Train**, **Validation**, and **Test** folders based on specified ratios (`train_ratio`, `val_ratio`, `test_ratio`).
- Ensures stratified and balanced splits.

Key Aspects:
- Validates the existence of input folders to prevent operation on non-existent directories.
- Uses `train_test_split` from `sklearn` for controlled random splitting.
- Copies files into organized subdirectories: `Train`, `Validation`, and `Test` under the specified `output_dir`.

Output:
- Organized dataset folder structure:
```
output_dir/
    Train/
      Class1/
      Class2/
    Validation/
      Class1/
      Class2/
    Test/
      Class1/
      Class2/
```

Example:
- A dataset of images for "Fire" vs. "Not Fire" classifications is split for training the model.

---

#### **2. Model Training: `train_mobilenet`**

Functionality:
- Fine-tunes MobileNetV2 for binary classification using transfer learning, while adding custom dense layers.
- Optimizes the model with **Adam optimizer** and **binary cross-entropy loss**, tracking performance on validation datasets.
- Incorporates advanced callbacks (`LearningRateScheduler`, `EarlyStopping`, `ReduceLROnPlateau`) to ensure efficient training.

Key Features:
- **Data Augmentation** for training images via `ImageDataGenerator`.
- **Class Weighting** to handle any class imbalances in the dataset.
- Custom layers include a **Dense(128)** layer with L2 regularization, **Dropout(0.6)** for generalization, and a **Dense(1)** sigmoid activation for binary classification.
- Freezes most layers in MobileNetV2 except the last 20 for fine-tuning.

Performance Outputs:
- Trained model is saved as `mobilenet_fire_model.keras` in the specified output directory.
- Evaluation metrics on `test_dir`, including:
  - Loss
  - Accuracy
  - Precision, Recall, F1-score, Specificity, ROC-AUC, PR-AUC.
- Final evaluation includes confusion matrix, plotted ROC curve, and Precision-Recall curve.

---

#### **3. Visualization: `visualize_training`**

Functionality:
- Extracts training results (accuracy/loss for both training and validation sets) and visualizes them across epochs.

Key Outputs:
- Two plots for:
  - **Accuracy** over epochs for training/validation sets.
  - **Loss** over epochs for training/validation sets.

Purpose:
To help diagnose overfitting, underfitting, or a well-balanced training process.

---

### **Key Components Explained**

---

#### **Model Architecture**

The backbone architecture is **MobileNetV2**, a lightweight and efficient CNN model pretrained on **ImageNet**. This implementation:
- Leverages pretrained layers for feature extraction to save training time.
- Adds fully connected layers on top of the base to perform binary classification.

The custom model structure:
1. **Base (Frozen)**: MobileNetV2 with weights frozen (except the last 20 layers).
2. **Global Pooling**: Converts feature maps from the convolutions into feature vectors.
3. **Dense Regularized Layers**: Fully connected layers with dropout and kernel regularization to reduce overfitting.
4. **Output Layer**: Final Dense(1) layer with sigmoid activation outputs the probability of the positive class.

---

#### **Advanced Callbacks**

1. **EarlyStopping**:
   - Monitors validation loss and halts training if no improvement is seen for 5 epochs.
   - Restores the best weights to prevent overfitting.

2. **ReduceLROnPlateau**:
   - Dynamically reduces the learning rate when the validation loss plateaus, allowing finer convergence.

3. **Cosine Decay Scheduler**:
   - Smoothly reduces the learning rate across epochs for stable optimization, preventing oscillations.

---

#### **Metrics Evaluation**

Post-training, the model is evaluated rigorously using:
- **Confusion Matrix**: Summarizes true positives, false positives, true negatives, and false negatives.
- **Precision**: Fraction of correctly identified positive samples.
- **Recall (Sensitivity)**: Fraction of actual positives correctly identified.
- **Specificity**: Ability to correctly classify negatives.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve for measuring classification separability.
- **PR-AUC**: Area under the Precision-Recall curve for imbalanced datasets.

Charts such as **ROC Curve** and **Precision-Recall Curve** are plotted to visualize model performance efficiently.

---

### **Workflow Summary**

1. **Dataset Setup:**
   - Organize raw data directories by class.
   - Execute `split_dataset` to divide the dataset into Train/Validation/Test sets.

2. **Model Training:**
   - Define paths to the Train/Validation/Test folders.
   - Execute `train_mobilenet` to train the MobileNetV2 model using transfer learning.
   - During training:
     - Check validation accuracy/loss to prevent overfitting.
     - Use class weights to balance imbalanced datasets.
     - Automatically adjust the learning rate.

3. **Model Evaluation:**
   - Evaluate the trained model using test data.
   - Output and plot relevant metrics (e.g., confusion matrix, ROC-AUC, PR-AUC, F1-score).

4. **Visualization:**
   - Use `visualize_training` to inspect training/validation accuracy and loss over epochs.

---

### **Advantages of the Code**

- **Readability and Modular Design**:
  - Functions are self-contained and serve clear purposes.
  - Easily reusable for new binary classification projects.

- **Extensive Evaluations**:
  - Outputs comprehensive metrics covering all aspects of model performance.
  - Includes advanced analysis (e.g., Specificity, PR-AUC).

- **Scalability**:
  - Highly customizable to other datasets and architectures due to modularity.
  - Multiple callbacks ensure scalability across various parameter configurations.

---

### **Areas for Improvement**

1. **Error Handling**:
   - Enhance error handling in `train_mobilenet` (e.g., ensure all directories exist before execution).

2. **Multi-class Compatibility**:
   - The code handles binary classification (`class_mode="binary"`). Generalizing this to multi-class problems (`class_mode="categorical"`) would improve versatility.

3. **Performance Logging**:
   - Extend the code to log metrics and hyperparameter configurations to a file for easier tracking over multiple runs.

---

### **Conclusion**

This code demonstrates a highly effective and flexible approach to transfer learning using MobileNetV2. From preprocessing and training to metric computation and visualization, the pipeline ensures reliability, reproducibility, and thorough performance evaluation. The structured implementation makes it suitable for real-world applications and easily extensible to other deep learning projects.
