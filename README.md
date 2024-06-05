# image-classification-using-cifar-10-dataset

1. Import Libraries
Essential libraries such as NumPy for numerical operations, Matplotlib for plotting, Keras for deep learning, and sklearn for data splitting are imported.

2. Load and Preprocess the CIFAR-10 Dataset
Loading Data: The CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes, is loaded using keras.datasets.cifar10.load_data().
Normalization: The pixel values of the images are normalized by converting the data type to float64 and dividing by 255.0 to scale the pixel values between 0 and 1.
One-Hot Encoding: The labels are one-hot encoded using keras.utils.to_categorical to convert the class vectors into binary class matrices.
Train-Validation Split: The training data is further split into training and validation sets using sklearn.model_selection.train_test_split with a test size of 40%.

3. Define the CNN Model
Model Initialization: A Sequential model is defined using keras.models.Sequential().
Input Layer: An input layer is added with the shape (32, 32, 3).
Convolutional and Pooling Layers: The first convolutional layer has 32 filters of size (3, 3), with 'same' padding and a stride of 1.
A max pooling layer with a pool size of (2, 2) and 'same' padding.
Another convolutional layer with 64 filters of size (3, 3), followed by another max pooling layer.
A final convolutional layer with 64 filters of size (3, 3) and 'valid' padding.
Flatten Layer: Converts the 2D matrix data to a 1D vector.
Dense Layers: A dense layer with 128 units and ReLU activation.
A dropout layer with a rate of 0.06 to prevent overfitting.
A final dense layer with 10 units (one for each class) and softmax activation for multi-class classification.

4. Compile the Model
The model is compiled using the Adam optimizer, categorical cross-entropy loss function, and accuracy as a metric.

6. Train the Model
The model is trained on the training data for 12 epochs with a batch size of 32. Validation is performed using the validation set, and the progress is stored in the history object.

8. Evaluate the Model
The model's performance is evaluated on the test set using model.evaluate(), and the loss and accuracy are printed.

10. Make Predictions
Predictions are made on the test set using model.predict().

12. Visualize Predictions
A subplot grid of 5x3 is created to display the first 15 test images. Each subplot shows the actual and predicted labels.

14. Plot Loss vs Accuracy
A line plot is created to visualize the training loss and accuracy over the epochs. This helps in understanding the model's learning progress.


Summary
This code effectively demonstrates how to build a basic CNN for image classification on the CIFAR-10 dataset, including data preprocessing, model definition, training, evaluation, and visualization of results. The model architecture consists of multiple convolutional layers followed by pooling layers, a flatten layer, and dense layers, which are typical components of a CNN for image classification tasks.
