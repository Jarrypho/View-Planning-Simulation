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

from pyparsing import Or
import tensorflow as tf
import numpy as np


# def orthogonal_regularizer(x, num_features, l2reg, eye):
#     x = tf.reshape(x, (-1, num_features, num_features))
#     xxt = tf.tensordot(x, x, axes=(2,2))
#     xxt = tf.reshape(xxt, (-1, num_features,num_features))
#     return tf.reduce_sum(l2reg * tf.square(xxt - eye))
@tf.keras.utils.register_keras_serializable(package='Custom', name='esgd')
class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
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

class Pointnet_Model(tf.keras.Model):
    def __init__(self, last_layer_size, num_features, num_points):
        super().__init__()
        self.last_layer_size = last_layer_size
        self.num_features = num_features
        self.num_points = num_points
        self.identiy = tf.eye(num_features)

        #Transformation block 1
        self.inp_trans_block_1_conv = tf.keras.layers.Conv1D(filters=32, kernel_size=1, padding="valid", name="input_transformation_block_1_conv", input_shape=(2048,self.num_features))
        self.inp_trans_block_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="input_transformation_block_1_batch_norm")
        self.inp_trans_block_1_relu = tf.keras.layers.Activation("relu", name="input_transformation_block_1_relu")
        self.inp_trans_block_2_conv = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding="valid", name="input_transformation_block_2_conv")
        self.inp_trans_block_2_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="input_transformation_block_2_batch_norm")
        self.inp_trans_block_2_relu = tf.keras.layers.Activation("relu", name="input_transformation_block_2_relu")
        self.inp_trans_block_3_conv = tf.keras.layers.Conv1D(filters=512, kernel_size=1, padding="valid", name="input_transformation_block_3_conv")
        self.inp_trans_block_3_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="input_transformation_block_3_batch_norm")
        self.inp_trans_block_3_relu = tf.keras.layers.Activation("relu", name="input_transformation_block_3_relu")
        self.inp_trans_block_1_1_dense = tf.keras.layers.Dense(256, name="input_transformation_block_1_1_dense")
        self.inp_trans_block_1_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="input_transformation_block_1_1_batch_norm")
        self.inp_trans_block_1_1_relu = tf.keras.layers.Activation("relu", name="input_transformation_block_1_1_relu")
        self.inp_trans_block_2_1_dense = tf.keras.layers.Dense(128, name="input_transformation_block_2_1_dense")
        self.inp_trans_block_2_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="input_transformation_block_2_1_batch_norm")
        self.inp_trans_block_2_1_relu = tf.keras.layers.Activation("relu", name="input_transformation_block_2_1_relu")
        self.tnet_custom_final = tf.keras.layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(np.eye(num_features).flatten()),
            #activity_regularizer=tf.keras.regularizers.L2(0.001),
            name="input_transformation_block_final"
        )
        self.tnet_dot_1 = tf.keras.layers.Dot(axes=(2,1), name="input_transformation_block_mm")
        #conv_bn_1
        self.features_32_1_conv = tf.keras.layers.Conv1D(filters=32, kernel_size=1, name="features_32_1_conv")
        self.features_32_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="features_32_1_batch_norm")
        self.features_32_1_relu = tf.keras.layers.Activation("relu", name="features_32_1_relu")
        #conv_bn_2
        self.features_32_2_conv = tf.keras.layers.Conv1D(filters=64, kernel_size=1, name="features_32_2_conv")
        self.features_32_2_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="features_32_2_batch_norm")
        self.features_32_2_relu = tf.keras.layers.Activation("relu", name="features_32_2_relu")
        #Transformation block 2
        self.trans_block_1_conv = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding="valid", name="transformed_features_1_conv")
        self.trans_block_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="transformed_features_1_batch_norm")
        self.trans_block_1_relu = tf.keras.layers.Activation("relu", name="transformed_features_1_relu")
        self.trans_block_2_conv = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding="valid", name="transformed_features_2_conv")
        self.trans_block_2_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="transformed_features_2_batch_norm")
        self.trans_block_2_relu = tf.keras.layers.Activation("relu", name="transformed_features_2_relu")
        self.trans_block_3_conv = tf.keras.layers.Conv1D(filters=512, kernel_size=1, padding="valid", name="transformed_features_3_conv")
        self.trans_block_3_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="transformed_features_3_batch_norm")
        self.trans_block_3_relu = tf.keras.layers.Activation("relu", name="transformed_features_3_relu")
        self.trans_block_1_1_dense = tf.keras.layers.Dense(256, name="transformed_features_1_1_dense")
        self.trans_block_1_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="transformed_features_1_1_batch_norm")
        self.trans_block_1_1_relu = tf.keras.layers.Activation("relu", name="transformed_features_1_1_relu")
        self.trans_block_2_1_dense = tf.keras.layers.Dense(128, name="transformed_features_2_1_dense")
        self.trans_block_2_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="transformed_features_2_1_batch_norm")
        self.trans_block_2_1_relu = tf.keras.layers.Activation("relu", name="transformed_features_2_1_relu")
        self.out_tnet_custom_final = tf.keras.layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(np.eye(num_features).flatten()),
            #activity_regularizer=tf.keras.regularizers.L2(0.001),
            name="transformed_features_final"
        )
        self.tnet_dot_2 = tf.keras.layers.Dot(axes=(2,1), name="transformed_features_mm")
        #conv_bn_3
        self.features_128_1_conv = tf.keras.layers.Conv1D(filters=256, kernel_size=1, padding="valid", name="features_128_conv")
        self.features_128_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="features_128_batch_norm")
        self.features_128_1_relu = tf.keras.layers.Activation("relu", name="features_128_relu")
        #conv_bn_4
        self.features_256_1_conv = tf.keras.layers.Conv1D(filters=512, kernel_size=1, padding="valid", name="features_256_conv")
        self.features_256_1_ba = tf.keras.layers.BatchNormalization(momentum=0.0, name="features_256_batch_norm")
        self.features_256_1_relu = tf.keras.layers.Activation("relu", name="features_256_relu")
        
        #custom net for regression at end
        self.dense_1 = tf.keras.layers.Dense(256, "relu", name="dense_1")
        self.dense_2 = tf.keras.layers.Dense(128, "relu", name="dense_2")
        self.pos = tf.keras.layers.Dense(3, name="pos_layer")
        self.dense_3 = tf.keras.layers.Dense(16, "relu", name="dense_3")
        self.dense_4 = tf.keras.layers.Dense(16, "relu", name="dense_4")
        self.dense_5 = tf.keras.layers.Dense(8, "relu", name="dense_5")
        self.rot = tf.keras.layers.Dense(2, name="rot_layer")
        

    def call(self, inputs):
        ###Transformation block 1
        #conv_bn_1
        x = self.inp_trans_block_1_conv(inputs)
        #x = self.inp_trans_block_1_ba(x)
        x = self.inp_trans_block_1_relu(x)
        #conv_bn_2
        x = self.inp_trans_block_2_conv(x)
        #x = self.inp_trans_block_2_ba(x)
        x = self.inp_trans_block_2_relu(x)
        #conv_bn_3
        x = self.inp_trans_block_3_conv(x)
        #x = self.inp_trans_block_3_ba(x)
        x = self.inp_trans_block_3_relu(x)
        #Max pool
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        #dense_bn_1_1
        x = self.inp_trans_block_1_1_dense(x)
        #x = self.inp_trans_block_1_1_ba(x)
        x = self.inp_trans_block_1_1_relu(x)
        #dense_bn_2_1
        x = self.inp_trans_block_2_1_dense(x)
        #x = self.inp_trans_block_2_1_ba(x)
        x = self.inp_trans_block_2_1_relu(x)
        #custom dense
        transformed_features = self.tnet_custom_final(x)
        transformed_features = tf.keras.layers.Reshape((self.num_features, self.num_features))(transformed_features)
        transformed_inputs = self.tnet_dot_1([inputs, transformed_features])
        ###conv_bn_1
        features_32_1 = self.features_32_1_conv(transformed_inputs)
        #features_32_1 = self.features_32_1_ba(features_32_1)
        features_32_1 = self.features_32_1_relu(features_32_1)
        #conv_bn_2
        features_32_2 = self.features_32_2_conv(features_32_1)
        #features_32_2 = self.features_32_2_ba(features_32_2)
        features_32_2 = self.features_32_2_relu(features_32_2)
        ###Transformation block 2
        #conv_bn_1
        x = self.trans_block_1_conv(inputs)
        #x = self.trans_block_1_ba(x)
        x = self.trans_block_1_relu(x)
        #conv_bn_2
        x = self.trans_block_2_conv(x)
        #x = self.trans_block_2_ba(x)
        x = self.trans_block_2_relu(x)
        #conv_bn_3
        x = self.trans_block_3_conv(x)
        #x = self.trans_block_3_ba(x)
        x = self.trans_block_3_relu(x)
        #Max pool
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        #dense_bn_1_1
        x = self.trans_block_1_1_dense(x)
        #x = self.trans_block_1_1_ba(x)
        x = self.trans_block_1_1_relu(x)
        #dense_bn_2_1
        x = self.trans_block_2_1_dense(x)
        #x = self.trans_block_2_1_ba(x)
        x = self.trans_block_2_1_relu(x)
        #custom dense
        transformed_features = self.out_tnet_custom_final(x)
        transformed_features = tf.keras.layers.Reshape((self.num_features, self.num_features))(transformed_features)
        transformed_inputs = self.tnet_dot_2([inputs, transformed_features])
        #conv_3
        features_128 = self.features_128_1_conv(transformed_inputs)
        #features_128 = self.features_128_1_ba(features_128)
        features_128 = self.features_128_1_relu(features_128)
        #conv_4
        features_256 = self.features_256_1_conv(features_128)
        #features_256 = self.features_256_1_ba(features_256)
        features_256 = self.features_256_1_relu(features_256)
        #Max_Pool
        global_features = tf.keras.layers.MaxPool1D(pool_size=self.num_points, name="global_features")(features_256)
        x = tf.keras.layers.Flatten()(global_features)
        x = self.dense_1(x)
        x = self.dense_2(x)
        pos = self.pos(x)
        x = self.dense_3(pos)
        x = self.dense_4(x)
        x = self.dense_5(x)
        rot = self.rot(x)
        output = tf.concat([pos, rot], axis=1)
        return output

    def compute_output_shape(self, input_shape):
            return tf.TensorShape(input_shape[0:1] + (self.last_layer_size,))
