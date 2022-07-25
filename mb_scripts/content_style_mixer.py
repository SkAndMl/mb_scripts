import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from tensorflow.keras.applications import vgg19
import sys


mpl.rcParams["figure.figsize"] = (12,12)
mpl.rcParams["axes.grid"] = False
# mpl.rcParams["figure.axis"] = "off"

def vgg_layers_model(layer_names):

    vgg = vgg19.VGG19(include_top=False,weights="imagenet")
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=[vgg.input],outputs=outputs)
    return model

def gram_matrix(tensor):

    gram_sum = tf.linalg.einsum("bijc,bijd->bcd",tensor,tensor)
    shape = tf.cast(tf.shape(tensor),tf.float32)
    divisor = shape[1]*shape[2]
    return gram_sum/divisor

class StyleContentExtractor(tf.keras.Model):

    def __init__(self,style_layers,content_layers):
        
        super(StyleContentExtractor, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.vgg = vgg_layers_model(style_layers+content_layers)
        self.vgg.trainable = False
        self.n_style_layers = len(style_layers)
    
    def call(self,inputs):
        
        inputs = inputs*255.
        inputs = vgg19.preprocess_input(inputs)
        
        outputs = self.vgg(inputs)
        style_outputs,content_outputs = outputs[:self.n_style_layers],outputs[self.n_style_layers:]

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        style_dict = {name:output for name,output in zip(self.style_layers,style_outputs)}
        content_dict = {name:output for name,output in zip(self.content_layers,content_outputs)}

        return {"style":style_dict,"content":content_dict}



class StyleContentMixer:

    

    def __init__(self,content_img,style_layers,content_layers,style_choice="kandinsky",
                style_weight=1e-2,content_weight=1e4):

        self.content_img = tf.cast(tf.Variable(content_img),tf.float32)
        self.style_img = self._get_style(style_choice)
        self.content_img = self._preprocess(self.content_img)
        self.style_img = self._preprocess(self.style_img)
        self.style_layers = style_layers
        self.content_layers = content_layers
        
        self.opt = tf.keras.optimizers.Adam(learning_rate=2e-2,beta_1=0.9,epsilon=1e-1)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self._get_extractor()
    

    def _get_style(self,style_choice):
        
        style_path = tf.keras.utils.get_file('kandinsky5.jpg',
                                            'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
        style_img = tf.io.read_file(style_path)
        style_img = tf.image.decode_image(style_img,channels=3)
        style_img = tf.image.convert_image_dtype(style_img,dtype=tf.float32)
        return style_img
    
    def _preprocess(self,img):

        img = img/255.
        max_dim = 512
        shape = tf.cast(img.shape[:-1],tf.float32)
        long_dim = max(shape)
        scale = max_dim/long_dim
        new_shape = tf.cast(shape*scale,tf.int32)

        new_img = tf.image.resize(img,new_shape)
        return new_img[tf.newaxis,...]
    
    def _get_extractor(self):
        self.extractor = StyleContentExtractor(self.style_layers,self.content_layers)
        self.content_targets = self.extractor(self.content_img)["content"]
        self.style_targets = self.extractor(self.style_img)["style"]

    def _clip_0_1(self,img):
        return tf.clip_by_value(img,clip_value_min=0.0,clip_value_max=1.0)

    
    def _style_content_loss(self,outputs):
        
        style_output = outputs["style"]
        content_output = outputs["content"]

        style_loss = tf.add_n(
            [tf.reduce_mean((style_output[name]-self.style_targets[name])**2) for name in style_output.keys()]
        )
        content_loss = tf.add_n(
            [tf.reduce_mean((content_output[name]-self.content_targets[name])**2) for name in content_output.keys()]
        )

        style_loss = style_loss*self.style_weight/len(self.style_layers)
        content_loss = content_loss*self.content_weight/len(self.content_layers)

        return style_loss + content_loss


    
    def _run_gradient(self,img):

        with tf.GradientTape() as tape:
            outputs = self.extractor(img)
            loss = self._style_content_loss(outputs=outputs)
        
        grad = tape.gradient(loss,img)
        self.opt.apply_gradients([(grad,img)])
        return self._clip_0_1(img)
    
    def mix(self,epochs=10,steps_per_epoch=100):

        new_img = tf.Variable(self.content_img)

        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                self._run_gradient(new_img)
        
        plt.figure(figsize=(12,12))
        plt.imshow(new_img[0])
        plt.show()
        
        return new_img

if __name__ == "__main__":

    content_img = plt.imread(sys.argv[1])
    print("\nReading in the image")

    style_layers = ["block1_conv1",
                    "block2_conv1",
                    "block3_conv1",
                    "block4_conv1",
                    "block5_conv1"]
    content_layers = ["block5_conv2"]

    n_epochs = int(input("\nNumber of epochs? "))
    n_steps_per_epoch = int(input("\nNumber of steps per epoch? "))

    savefig_name = input("\nMixed image's saving names? ")


    scm = StyleContentMixer(content_img, style_layers, content_layers)
    mixed_img = scm.mix(epochs=n_epochs,steps_per_epoch=n_steps_per_epoch)
    plt.figsave(f"savefig_name.jpg")