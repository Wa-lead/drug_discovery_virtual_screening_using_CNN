import tensorflow as tf
from keras.layers import (Input, Conv2D, Concatenate, GlobalAveragePooling2D,
                                     Dense, MaxPooling2D, Dropout)
from keras.models import Model
from keras.optimizers import Adam

class Chemception():
    def __init__(self, N=64,
                 inceptionA_count=1, inceptionB_count=1, inceptionC_count=1, reductionA_count=1, reductionB_count=1,
                 input_shape=(48, 48, 4),
                 config=None
                 ):
        
        if config:
            self.N = config['N']
            self.inceptionA_count = config['inceptionA_count']
            self.inceptionB_count = config['inceptionB_count']
            self.inceptionC_count = config['inceptionC_count']
            self.reductionA_count = config['reductionA_count']
            self.reductionB_count = config['reductionB_count']
            self.input_shape = input_shape
        else:
            self.N = N
            self.inceptionA_count = inceptionA_count
            self.inceptionB_count = inceptionB_count
            self.inceptionC_count = inceptionC_count
            self.reductionA_count = reductionA_count
            self.reductionB_count = reductionB_count
            self.input_shape = input_shape

    def build(self):
        input_layer = Input(shape=self.input_shape)
        x = self.build_stem(input_layer)
        for _ in range(self.inceptionA_count):
            x = self.inceptionA(x)
        for _ in range(self.reductionA_count):
            x = self.reductionA(x)
        for _ in range(self.inceptionB_count):
            x = self.inceptionB(x)
        for _ in range(self.reductionB_count):
            x = self.reductionB(x)
        for _ in range(self.inceptionC_count):
            x = self.inceptionC(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.N*2, activation='sigmoid')(x)
        x = Dense(self.N, activation='sigmoid')(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(input_layer, x)
        return model


    def build_stem(self,x):
        x = Conv2D(0.25*self.N, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2D(0.25*self.N, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(self.N, (3, 3), padding='same', activation='relu')(x) # Changed filter size from 16 to self.N
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x) # Changed filter size from 20 to self.N
        x = Conv2D(2*self.N, (3, 3), padding='same', activation='relu')(x) # Added a convolutional layer
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        return x

    def inceptionA(self,x):
        x1 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        
        x2 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(self.N, (5, 5), padding='same', activation='relu')(x2)
        
        x3 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x3 = Conv2D(int(1.5*self.N), (3, 3), padding='same', activation='relu')(x3)
        x3 = Conv2D(2*self.N, (3, 3), padding='same', activation='relu')(x3)
        
        concatenated = Concatenate(axis=-1)([x1, x2, x3])
       
        return concatenated

    def inceptionB(self,x):
        x1 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(int(1.25*self.N), (1, 7), padding='same', activation='relu')(x2)
        x2 = Conv2D(int(1.5*self.N), (7, 1), padding='same', activation='relu')(x2)
        concatenated = Concatenate(axis=-1)([x1, x2])
        return concatenated

    def inceptionC(self,x):
        x1 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(int(1.25*self.N), (1, 3), padding='same', activation='relu')(x2)
        x2 = Conv2D(int(1.5*self.N), (3, 1), padding='same', activation='relu')(x2)
        concatenated = Concatenate(axis=-1)([x1, x2])
        return concatenated

    def reductionA(self,x):
        x1 = Conv2D(int(1.5*self.N), (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x2 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(self.N, (3, 3), padding='same', activation='relu')(x2)
        x2 = Conv2D(int(1.5*self.N), (3, 3), strides=(2, 2), padding='same', activation='relu')(x2)
        x3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        concatenated = Concatenate(axis=-1)([x1, x2, x3])
        return concatenated

    def reductionB(self,x):
        x1 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x1 = Conv2D(int(1.5*self.N), (3, 3), strides=(2, 2), padding='same', activation='relu')(x1)
        x2 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x2 = Conv2D(int(1.25*self.N), (3, 1), padding='same', activation='relu')(x2)
        x2 = Conv2D(int(1.5*self.N), (3, 1), padding='same', strides=(2,2) ,activation='relu')(x2)
        x4 = Conv2D(self.N, (1, 1), padding='same', activation='relu')(x)
        x4 = Conv2D(int(1.5*self.N), (3, 3), strides=(2, 2), padding='same', activation='relu')(x4)
        x3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        concatenated = Concatenate(axis=-1)([x1, x2, x3, x4])
        return concatenated
    

if __name__ == "__main__":
    model = Chemception()
    model = model.build()
    model.summary()