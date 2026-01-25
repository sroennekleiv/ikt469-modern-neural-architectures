import re
import torch

import tensorflow as tf

# Preprocessing module for data preparation (textual and image data)
class Preprocessor:
    def __init__(self, lower=True, size=(28, 28)):
        self.lower = lower
        self.size = size
    
    def preprocess_images(self, x, augment=False):
        # Normalize and resize images
        x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.0
        x = tf.image.resize(x[..., tf.newaxis], self.size)
        
        # Augment images during training
        if augment:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, max_delta=0.1)
            x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        return x
        