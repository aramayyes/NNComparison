## Neural networks comparison

Two neural networks, one with a single hidden layer (first) and the other with convolutional layers (second) are compared for digits recognition.

#### Code

- The first neural network is taken from the ["Make Your Own Neural Network"](https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G) book. Thanks to Tariq Rashid. Code for that book is available here https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork.

- The second neural network is taken from this video tutorial: https://www.youtube.com/watch?v=cPS67_Ww91E&t=1769s. Thanks to Dmitry Korobchenko. Code is available here: https://drive.google.com/file/d/0BzviurZGjZkiQk5Kc0FHY0pPdzA/view. 

#### Neural Networks

- The first neural network contains just a single hidden layer. The gradient descent algorithm without any optimization is used for updating the network's weights.

- The second neural network contains convolutional and pooling layers. The gradient descent algorithm with Adam optimization is used for updating the network's weights. 

#### Projects

There are three projects in this repository. 

- `NNComparisonAndroidApp`  is an android application, which allows to write digits on the screen and test the neural network using those digits.

- `NNetworkWithSingleHiddenLayer` is a server application which represents the first neural network.

- `ConvolutionalNNetwork` is a server application which represents the second neural network.

In order to test one of the neural networks, run the corresponding server application first, update the `serverUrl` in [`DigitsPaintingActivity.kt`](/NNComparisonAndroidApp/app/src/main/java/info/ayyes/nncomparison/DigitsPaintingActivity.kt) and then run the android application.


In the image below, the first neural network is on the left side, and the second neural network is on the right side.

<p align="center">
  <img width="80%" alt="Networks comparison" src="https://www.dropbox.com/s/rztjzx55ns4pt55/NNComparison.gif?raw=1">
</p>