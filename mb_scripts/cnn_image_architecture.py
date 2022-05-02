def build_typical_cnn(input_shape,activation_conv="relu",dropout_rate=0.5,activation_dense="relu",optimizer="nadam",
                      padding="same",pool_size=2):

    """
    Typical CNN model is used for image classification
    :param input_shape: shape of the training instances: [height,width,channels]
    :param activation_conv: activation function for the Conv2D layers
    :param dropout_rate: specifies the fraction of neurons to be dropped
    :param activation_dense: activation function for the dense layers
    :param optimizer: optimizer for compiling the model
    :param padding: "same" or "valid"
    :param pool_size: specifies the pool_size for MaxPooling2D
    :return: returns a CNN model built based on the typical CNN architecture
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout

    typical_cnn = tf.keras.models.Sequential([
        Conv2D(filters=64,kernel_size=7,strides=1,padding=padding,input_shape=input_shape,activation=activation_conv),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(128,3,activation=activation_conv,padding=padding),
        Conv2D(128,3,activation=activation_conv,padding=padding),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(256,3,activation=activation_conv,padding=padding),
        Conv2D(256,3,activation=activation_conv,padding=padding),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(128,activation=activation_dense),
        Dropout(dropout_rate),
        Dense(64,activation=activation_dense),
        Dropout(0.5),
        Dense(10,activation="sigmoid")
    ],name="typical_cnn")

    typical_cnn.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics="accuracy")

    return typical_cnn

def build_lenet5(input_shape,optimizer="nadam",
                      padding="same"):

    """
    LeNet5 CNN model espescially used for digit recognition
    :param input_shape: shape of the training instances: [height,width,channels]
    :param activation_conv: activation function for the Conv2D layers
    :param dropout_rate: specifies the fraction of neurons to be dropped
    :param activation_dense: activation function for the dense layers
    :param optimizer: optimizer for compiling the model
    :param padding: "same" or "valid"
    :param pool_size: specifies the pool_size for MaxPooling2D
    :return: returns a CNN model built based on the LeNet5 architecture adapted from Yann LeCun's paper
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Flatten, Conv2D, AveragePooling2D,Dense,Activation

    tf.random.set_seed(42)

    lenet5 = tf.keras.models.Sequential([
        Conv2D(filters=6, kernel_size=5, input_shape=input_shape, strides=1, activation="tanh"),
        AveragePooling2D(pool_size=2, strides=2),
        Activation("tanh"),
        Conv2D(filters=16, kernel_size=5, strides=1, activation="tanh", padding=padding),
        AveragePooling2D(pool_size=2, strides=2),
        Activation("tanh"),
        Conv2D(filters=120, kernel_size=5, strides=1, activation="tanh", padding=padding),
        Flatten(),
        Dense(84, activation="tanh"),
        Dense(10, activation="sigmoid")
    ], name="lenet5")

    lenet5.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics="accuracy")

    return lenet5

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,GlobalAvgPool2D,MaxPooling2D,Flatten,Dense,Activation

class ResidualUnit(tf.keras.layers.Layer):

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            Conv2D(filters, 3, strides=strides, padding="same"),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, padding="same"),
            BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, 1, strides=strides, padding="same"),
                BatchNormalization()
            ]

    def call(self, X):
        Z = X
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = X
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

def build_resnet(input_shape,optimizer):
    resnet = tf.keras.models.Sequential()
    resnet.add(Conv2D(64,7,2,input_shape=input_shape,padding="same"))
    resnet.add(BatchNormalization())
    resnet.add(Activation("relu"))
    resnet.add(MaxPooling2D(pool_size=3,strides=2,padding="same"))
    prev_filters = 64
    for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
        strides = 1 if filters==prev_filters else 2
        resnet.add(ResidualUnit(filters,strides=strides))
        prev_filters = filters
    resnet.add(GlobalAvgPool2D())
    resnet.add(Flatten())
    resnet.add(Dense(10,activation="softmax"))

    resnet.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,
               metrics="accuracy")

    return resnet
