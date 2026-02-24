# CNN-C-implementation
Simple CNN implementation written in C. It was done for educational purposes and reaches roughly 92% accuracy on MNIST digits. As this is my first real C project there are a lot of flaws, both known and unknown probably.

## Architecture
For now the architecture consist of only one conv + maxpool stage before being fed into three dense layers. So far there are no plans on changing this.

## Data
In order to obtain the data with correct formatting, use mnist_preprocessing.py. It obtains MNIST through Keras and applies normalization before exporting. Make sure to add the resulting data files to cnn/resources or change the paths in main().

## Hyperparameters
The main configurable hyperparams are learning rate, num_filters, filter_size, batch_size, momentum. These are all controlled through defines for now, this to should be refactored.
