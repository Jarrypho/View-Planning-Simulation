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



class NN_Model_2(tf.keras.Model):
    def __init__(self, last_layer_size):
        super().__init__()
        self.last_layer_size = last_layer_size
        self.conv_1D_11 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu', name='Conv_1_1',
                                                 input_shape=(2048, 4))
        self.conv_1D_12 = tf.keras.layers.Conv1D(filters=256, kernel_size=1, activation=None, name='Conv_1_2')

        # Pointnet Layer 2
        self.conv_1D_21 = tf.keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', name='Conv_2_1')
        self.conv_1D_22 = tf.keras.layers.Conv1D(filters=512, kernel_size=1, activation=None, name='Conv_2_2')

        self.batch_normalization_1 = tf.keras.layers.BatchNormalization(momentum=0.0, name="ba1")
        self.batch_normalization_2 = tf.keras.layers.BatchNormalization(momentum=0.0, name="ba2")

        self.batch_normalization_3 = tf.keras.layers.BatchNormalization(momentum=0.0, name="ba3")

        self.batch_normalization_4 = tf.keras.layers.BatchNormalization(momentum=0.0, name="ba4")


        # Dense Layers
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu', name='MLP_1')
        self.dense_2 = tf.keras.layers.Dense(units=128, activation='relu', name='MLP_2')
        self.dense_3 = tf.keras.layers.Dense(units=last_layer_size, activation=None, name='MLP_3')

    def call(self, inputs):
        #inputs = tf.keras.Input(shape=(2048, 4), name='Input')
        x = self.conv_1D_11(inputs)
        x = self.batch_normalization_1(x)
        features = self.conv_1D_12(x)
        features = self.batch_normalization_2(features)
        features_global = tf.reduce_max(features, axis=1, keepdims=True, name='MaxPool_1')
        # features_global = tf.keras.layers.MaxPool1D(pool_size=n_points, name='MaxPool_1')(features)

        # Assembling features
        features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        # features = tf.keras.layers.concatenate([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2,name='concat')

        # Pointnet Layer 2
        x = self.conv_1D_21(features)
        x = self.batch_normalization_3(x)
        x = self.conv_1D_22(x)
        x = self.batch_normalization_4(x)
        # latent = tf.keras.layers.MaxPool1D(pool_size=n_points, name='MaxPool_2')(x)  # Latent Dim
        x = tf.reduce_max(x, axis=1, name='MaxPool_2')
        decoder_inputs = tf.keras.layers.Flatten()(x)

        # Decoder
        # decoder_inputs = tf.keras.Input(shape=(512,), name='Decoder_Input')
        x = self.dense_1(decoder_inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)

        # x = tf.keras.layers.Reshape([1, 3])(x)
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[0:1] + (self.last_layer_size,))
