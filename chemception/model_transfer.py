import tensorflow as tf
from keras.layers import (Input, Conv2D, Concatenate, GlobalAveragePooling2D,
                                     Dense, MaxPooling2D, Dropout)
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


class Chemception():
    def __init__(self, dense_layers=1, neurons=512, dropout=0.3, config=None):
        
        if config:
            self.neurons = config['neurons']
            self.dense_layers = config['dense_layers']
            self.input_shape = (80,80,3)
            self.dropout = config['dropout']
        else:
            self.neurons = neurons
            self.dense_layers = dense_layers
            self.input_shape = (80,80,3)
            self.dropout = dropout

    def build(self):

        # Load the InceptionV3 model, excluding the top layers
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Add a fully-connected layer with 512 hidden units and ReLU activation
        for i in range(self.dense_layers):
            x = Dense(self.neurons, activation='relu')(x)
            x = Dropout(self.dropout)(x)

        # Add a final output layer with sigmoid activation for binary classification
        predictions = Dense(1, activation='sigmoid')(x)

        # Create the transfer learning model
        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the weights of all layers in the InceptionV3 model except the last two
        for layer in base_model.layers[:-2]:
            layer.trainable = False

        return model