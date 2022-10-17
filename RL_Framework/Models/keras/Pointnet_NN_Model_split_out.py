import tensorflow as tf



class NN_Model_split_out(tf.keras.Model):
    def __init__(self, last_layer_size):
        super().__init__()
        self.last_layer_size = last_layer_size
        self.conv_1D_11 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu', name='Conv_1_1',
                                                 input_shape=(2048, 4))
        self.conv_1D_12 = tf.keras.layers.Conv1D(filters=256, kernel_size=1, activation=None, name='Conv_1_2')

        # Pointnet Layer 2
        self.conv_1D_21 = tf.keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', name='Conv_2_1')
        self.conv_1D_22 = tf.keras.layers.Conv1D(filters=512, kernel_size=1, activation=None, name='Conv_2_2')


        # Dense Layers
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu', name='MLP_1')
        self.dense_2 = tf.keras.layers.Dense(units=128, activation='relu', name='MLP_2')
        self.dense_3 = tf.keras.layers.Dense(units=3, activation=None, name='MLP_3')
        self.dense_4 = tf.keras.layers.Dense(units=16, activation="relu", name="MLP_4")
        self.dense_5 = tf.keras.layers.Dense(units=16, activation="relu", name="MLP_5")
        self.dense_6 = tf.keras.layers.Dense(units=8, activation="relu", name="MLP_6")
        self.dense_7 = tf.keras.layers.Dense(units=2, activation=None, name='MLP_7')


    def call(self, inputs):
        #inputs = tf.keras.Input(shape=(2048, 4), name='Input')
        x = self.conv_1D_11(inputs)
        features = self.conv_1D_12(x)
        features_global = tf.reduce_max(features, axis=1, keepdims=True, name='MaxPool_1')
        # features_global = tf.keras.layers.MaxPool1D(pool_size=n_points, name='MaxPool_1')(features)

        # Assembling features
        features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        # features = tf.keras.layers.concatenate([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2,name='concat')

        # Pointnet Layer 2
        x = self.conv_1D_21(features)
        x = self.conv_1D_22(x)
        # latent = tf.keras.layers.MaxPool1D(pool_size=n_points, name='MaxPool_2')(x)  # Latent Dim
        x = tf.reduce_max(x, axis=1, name='MaxPool_2')
        decoder_inputs = tf.keras.layers.Flatten()(x)

        # Decoder
        # decoder_inputs = tf.keras.Input(shape=(512,), name='Decoder_Input')
        x = self.dense_1(decoder_inputs)
        x = self.dense_2(x)
        pos = self.dense_3(x)
        if self.last_layer_size > 2:
            x = self.dense_4(pos)
            x = self.dense_5(x)
            x = self.dense_6(x)
            rotation = self.dense_7(x)
            x = tf.concat([pos, rotation], axis=1)
            
        else:
            x = pos

        # x = tf.keras.layers.Reshape([1, 3])(x)
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[0:1] + (self.last_layer_size,))