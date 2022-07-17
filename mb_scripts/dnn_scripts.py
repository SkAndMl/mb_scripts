import tensorflow as tf
from tensorflow.keras import layers
def build_dense(units=128, lower_thresh=16, activation="selu", kernel_initializer="lecun_normal",
                optimizer="adam", metrics="accuracy", dropout=False, do_rate=0.5,loss="binary_crossentropy",
                no_of_outputs=1):
    model = tf.keras.models.Sequential()
    if dropout == False:
        while units >= lower_thresh:
            model.add(layers.Dense(units=units, activation=activation, kernel_initializer=kernel_initializer))
            units = int(units / 2)
    else:
        while units >= lower_thresh * 2:
            model.add(layers.Dense(units=units, activation=activation, kernel_initializer=kernel_initializer))
            units = int(units / 2)
        model.add(layers.Dropout(do_rate))
        model.add(layers.Dense(units=int(units / 2), activation=activation, kernel_initializer=kernel_initializer))
        model.add(layers.Dropout(do_rate))
    model.add(layers.Dense(no_of_outputs, activation="sigmoid"))

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=metrics)

    return model


def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img
  

class Dense(tf.keras.layers.Layer):

    def __init__(self,units,activation,**kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
    
    def build(self,batch_input_shape):
        self.kernel = self.add_weight(name="kernel",
                                    shape=[batch_input_shape[-1],self.units],
                                    initializer="glorot_normal")
        self.bias = self.add_weight(name="bias",
                                    shape=[self.units],
                                    initializer="zeros")
        super().build(batch_input_shape)
    
    def call(self,X):
        return self.activation(X@self.kernel + self.bias)
    
    def compute_output_shape(self,batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1]+[self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,"units":self.units,
                "activation" : tf.keras.activations.serialize(self.activation)}
