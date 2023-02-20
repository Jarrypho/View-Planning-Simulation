'''
Copyright (C) 2022  Jan-Philipp Kaiser

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
'''

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers

def conv_bn(x: tf.Tensor, filters: int, name:str) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    #x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def dense_bn(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    #x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
        x = conv_bn(inputs, filters=32, name=f"{name}_1")
        x = conv_bn(x, filters=64, name=f"{name}_2")
        x = conv_bn(x, filters=512, name=f"{name}_3")
        x = layers.GlobalMaxPool1D()(x)
        x = dense_bn(x, filters=256, name=f"{name}_1_1")
        x = dense_bn(x, filters=128, name=f"{name}_2_1")
        return layers.Dense(
            num_features * num_features, 
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
            activity_regularizer=OrthogonalRegularizer(num_features=num_features),
            name=f"{name}_final"
        )(x)

def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
        transformed_features = transformation_net(inputs, num_features, name=name)
        transformed_features = layers.Reshape((num_features, num_features))(transformed_features)
        return layers.Dot(axes=(2,1), name=f"{name}_mm")([inputs, transformed_features])


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)
        
    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2,2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        config = {"num_features": self.num_features, "l2reg": self.l2reg}
        return config

def get_pointnet_model_2(last_layer_size: int, num_points: int, dim: int) -> keras.Model:
    input_points = keras.Input(shape=(num_points, dim))
    transformed_inputs = transformation_block(
            input_points, num_features=dim, name="input_transformation_block"
        )
    features_32 = conv_bn(transformed_inputs, filters=32, name="features_32")
    features_64_1 = conv_bn(features_32, filters=64, name="features_64_1")
    features_64_2 = conv_bn(features_64_1, filters=64, name="features_64_2")
    transformed_features = transformation_block(features_64_2, num_features=64, name="transformed_features")
    features_256 = conv_bn(transformed_features, filters=256, name="features_256")
    features_512 = conv_bn(features_256, filters=512, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(features_512)
    #global_features = tf.tile(global_features, [1, num_points, 1])
    x = layers.Flatten()(global_features)
    x = layers.Dense(256, "relu", name="dense_1")(x)
    x = layers.Dense(128, "relu", name="dense_2")(x)
    outputs = layers.Dense(last_layer_size, name="output_layer")(x)
    return keras.Model(input_points, outputs)

# class Pointnet_Model(tf.keras.Model):
#     def __init__(self, last_layer_size, num_points, dim):
#         super().__init__()
#         self.last_layer_size = last_layer_size
#         self.num_points = num_points
#         self.dim = dim
#         self.max_pool = layers.MaxPool1D(pool_size=num_points, name="global_features")
#         self.dense_1 = layers.Dense(256, "relu", name="dense_1")
#         self.dense_2 = layers.Dense(128, "relu", name="dense_2")
#         self.dense_3 = layers.Dense(self.last_layer_size, name="output_layer")

#     def call(self, input_points):
#         #input_points = keras.Input(shape=(self.num_points,self.dim))
#         transformed_inputs = transformation_block(
#             input_points, num_features=self.dim, name="input_transformation_block"
#         )
#         features_32 = conv_bn(transformed_inputs, filters=32, name="features_32")
#         features_64_1 = conv_bn(features_32, filters=64, name="features_64_1")
#         features_64_2 = conv_bn(features_64_1, filters=64, name="features_64_2")
#         transformed_features = transformation_block(features_64_2, num_features=64, name="transformed_features")
#         features_256 = conv_bn(transformed_features, filters=256, name="features_256")
#         features_512 = conv_bn(features_256, filters=512, name="pre_maxpool_block")
#         global_features = self.max_pool(features_512)
#         global_features = tf.tile(global_features, [1, self.num_points, 1])
#         x = self.dense_1(global_features)
#         x = self.dense_2(x)
#         outputs = self.dense_3(x)
#         return outputs

    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape(input_shape[0:1] + (self.last_layer_size,))
