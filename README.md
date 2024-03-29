# Low Dimensional Projection using Autoencoder and t-SNE - GitHub Repository

## Introduction
This repository contains the code for Task 2, which involves training an unsupervised learning neural network (autoencoder) on the Fashion-MNIST dataset to obtain a lower-dimensional representation of the images. Additionally, t-SNE from Scikit-Learn is applied to further reduce the dimensionality to 2, and the results are visualized in a single scatter plot.

## Task 2: Low Dimensional Projection (5pt)
### Code Summary
1. **Import Necessary Libraries:**
   - NumPy: For numerical operations.
   - Matplotlib: For data visualization.
   - scikit-learn: For t-SNE.
   - TensorFlow: For building the autoencoder.
   - Fashion-MNIST: From TensorFlow's datasets module.

2. **Load Fashion-MNIST Dataset:**
   - Grayscale images of various clothing items are loaded.

3. **Normalize the Data:**
   - Pixel values in the dataset are normalized to a range of [0, 1].

4. **Flatten the Images:**
   - Images are flattened from a 28x28 matrix to a 1D array of length 784 for both training and test datasets.

5. **Define Encoding Dimensions:**
   - An encoding dimension (32 in this code) is specified to control the size of the lower-dimensional representation learned by the autoencoder.

6. **Create Autoencoder Model:**
   - Autoencoder architecture is defined using TensorFlow's Keras API, consisting of an input layer, encoding layer with ReLU activation, and decoding layer with sigmoid activation.

7. **Compile and Train the Autoencoder:**
   - Autoencoder is compiled with the Adam optimizer and mean squared error (MSE) loss function. It is trained on the training data for 10 epochs with a batch size of 256.

8. **Obtain Lower-Dimensional Representations:**
   - A separate encoder model is created to extract the lower-dimensional representations from the input data. This encoder model is then applied to the test data to get encoded images.

9. **Apply t-SNE:**
   - scikit-learn's t-SNE implementation is used to reduce the dimensionality of the encoded images to 2 dimensions.

10. **Visualize the Results:**
    - Matplotlib is used to create a scatter plot of the t-SNE results. Each point represents an image's lower-dimensional representation, and colors are based on ground-truth labels to visualize the clustering of different fashion items.

### Code Location
The code for low-dimensional projection using autoencoder and t-SNE can be found in the file: `Autoencoders.py`

## Usage
1. Clone this repository.
2. Navigate to the root directory.
3. Run the Python script `Autoencoders.py`.

## Author
- Name: Ashiq Rahman Anwar Batcha
