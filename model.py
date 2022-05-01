from constants import DROPOUT_RATE, SEQUENCE_LENGTH, NUM_FEATURES, NUM_CLASSES, NUM_TRAINING
import tensorflow as tf


class ConvAE(tf.keras.Model):
    def __init__(self, name=None):
        super(ConvAE, self).__init__(name=name)
        self.batch1 = tf.keras.layers.BatchNormalization(momentum=0.9)

        depth_ae = 4
        l_ae, w_ae = 2, 1

        self.conv1 = tf.keras.layers.Conv2D(filters=depth_ae,
                                            kernel_size=[2, 2], padding="same",
                                            activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[l_ae, w_ae],
                                                  strides=[l_ae, w_ae])

        self.conv1_p = tf.keras.layers.Conv2D(filters=depth_ae,
                                              kernel_size=[2, 2],
                                              padding="same",
                                              activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[l_ae, w_ae],
                                                  strides=[l_ae, w_ae])

        # decoder
        self.conv2dt = tf.keras.layers.Conv2DTranspose(kernel_size=5,
                                                       filters=1,
                                                       strides=[l_ae, w_ae],
                                                       padding='same')
        self.conv2dt_p = tf.keras.layers.Conv2DTranspose(kernel_size=5,
                                                         filters=1,
                                                         strides=[l_ae, w_ae],
                                                         padding='same')

        self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE)

    def call(self, xs):
        input_ = tf.reshape(xs,
                            [-1, SEQUENCE_LENGTH, NUM_FEATURES, 1])  # [200, 14]
        input_ = self.batch1(input_)
        input_ = self.dropout(input_)

        conv1 = self.conv1(input_)
        h_t = self.pool1(conv1)

        conv1_p = self.conv1_p(input_)
        h_p = self.pool2(conv1_p)

        # decoder
        output_t = self.conv2dt(h_t)
        output_p = self.conv2dt_p(h_p)
        output = (output_t + output_p) / 2

        output = tf.reshape(output, [-1, SEQUENCE_LENGTH * NUM_FEATURES])

        return output, h_t, h_p


class CnnT(tf.keras.Model):
    def __init__(self, depth_1, depth_2, depth_3, depth_4, l_1, l_2, l_3, l_4,
                 w_1, w_2, w_3, w_4, name=None):
        super(CnnT, self).__init__(name=name)
        self.batch0 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv1 = tf.keras.layers.Conv2D(filters=depth_1,
                                            kernel_size=[3, 3], padding="same",
                                            activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[l_1, w_1],
                                                  strides=[l_1, w_1])
        self.batch1 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv2 = tf.keras.layers.Conv2D(filters=depth_2,
                                            kernel_size=[3, 3], padding="same",
                                            activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[l_2, w_2],
                                                  strides=[l_2, w_2])
        self.batch2 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv3 = tf.keras.layers.Conv2D(filters=depth_3,
                                            kernel_size=[2, 2], padding="same",
                                            activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=[l_3, w_3],
                                                  strides=[l_3, w_3])
        self.batch3 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv4 = tf.keras.layers.Conv2D(filters=depth_4,
                                            kernel_size=[2, 2], padding="same",
                                            activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=[l_4, w_4],
                                                  strides=[l_4, w_4])
        self.batch4 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.flatten = tf.keras.layers.Flatten()  # flatten the pool 2
        self.dropout1 = tf.keras.layers.Dropout(DROPOUT_RATE)
        self.dropout2 = tf.keras.layers.Dropout(DROPOUT_RATE)

        # """Add another FC layer"""
        self.fc1 = tf.keras.layers.Dense(units=300, activation='sigmoid')
        dim_hidden = 21
        self.fc2 = tf.keras.layers.Dense(units=dim_hidden,
                                         activation='sigmoid')
        # Attention layer: Comment out the two lines below if running `attention` ablation
        self.fc3 = tf.keras.layers.Dense(units=dim_hidden,
                                         activation='sigmoid')
        self.multiply = tf.keras.layers.Multiply()

        self.fc4 = tf.keras.layers.Dense(units=NUM_CLASSES)

    def call(self, h_t, xs):
        x_image_t = self.batch0(h_t)

        conv1 = self.conv1(x_image_t)
        pool1 = self.pool1(conv1)
        pool1 = self.batch1(pool1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        pool2 = self.batch2(pool2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        pool3 = self.batch3(pool3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        pool4 = self.batch4(pool4)

        fc1 = self.flatten(pool4)  # flatten the pool4

        # Add another FC layer
        fc1 = self.fc1(fc1)
        fc1 = self.dropout1(fc1)

        fc3 = self.fc2(fc1)
        fc3 = self.dropout2(fc3)

        # Attention layer: Comment out the two lines below if running `attention` ablation
        att = self.fc3(xs)
        fc3 = self.multiply([fc3, att])

        prediction_t = self.fc4(fc3)

        return prediction_t


class Cnn(tf.keras.Model):
    def __init__(self, depth_1, depth_2, depth_3, depth_4, l_1, l_2, l_3, l_4,
                 w_1, w_2, w_3, w_4, name=None):
        super(Cnn, self).__init__(name=name)
        self.batch0 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv1_p = tf.keras.layers.Conv2D(filters=depth_1,
                                              kernel_size=[3, 3],
                                              padding="same",
                                              activation='relu')
        self.pool1_p = tf.keras.layers.MaxPooling2D(pool_size=[l_1, w_1],
                                                    strides=[l_1, w_1])
        self.batch1 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv2_p = tf.keras.layers.Conv2D(filters=depth_2,
                                              kernel_size=[3, 3],
                                              padding="same",
                                              activation='relu')
        self.pool2_p = tf.keras.layers.MaxPooling2D(pool_size=[l_2, w_2],
                                                    strides=[l_2, w_2])
        self.batch2 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv3_p = tf.keras.layers.Conv2D(filters=depth_2,
                                              kernel_size=[2, 2],
                                              padding="same",
                                              activation='relu')
        self.pool3_p = tf.keras.layers.MaxPooling2D(pool_size=[l_3, w_3],
                                                    strides=[l_3, w_3])
        self.batch3 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv4_p = tf.keras.layers.Conv2D(filters=depth_4,
                                              kernel_size=[2, 2],
                                              padding="same",
                                              activation='relu')
        self.pool4_p = tf.keras.layers.MaxPooling2D(pool_size=[l_4, w_4],
                                                    strides=[l_4, w_4])
        self.batch4 = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE)

        dim_hidden_p = 200
        self.fc1 = tf.keras.layers.Dense(units=dim_hidden_p,
                                         activation='sigmoid')
        self.fc2 = tf.keras.layers.Dense(units=NUM_TRAINING)

    def call(self, h_p):
        x_image_p = self.batch0(h_p)

        conv1_p = self.conv1_p(x_image_p)
        pool1_p = self.pool1_p(conv1_p)
        pool1_p = self.batch1(pool1_p)

        conv2_p = self.conv2_p(pool1_p)
        pool2_p = self.pool2_p(conv2_p)
        pool2_p = self.batch2(pool2_p)

        conv3_p = self.conv3_p(pool2_p)
        pool3_p = self.pool3_p(conv3_p)
        pool3_p = self.batch3(pool3_p)

        conv4_p = self.conv4_p(pool3_p)
        pool4_p = self.pool4_p(conv4_p)
        pool4_p = self.batch4(pool4_p)

        fc1_p = self.flatten(pool4_p)  # flatten the pool 2

        fc3_p = self.fc1(fc1_p)
        fc3_p = self.dropout(fc3_p)

        prediction_p = self.fc2(fc3_p)

        return prediction_p


class Model(tf.keras.Model):
    def __init__(self, name=None):
        super(Model, self).__init__(name=name)
        self.conv_ae = ConvAE()

        depth_1, depth_2, depth_3, depth_4 = 16, 32, 64, 128
        l_1, l_2, l_3, l_4 = 2, 2, 2, 2
        w_1, w_2, w_3, w_4 = 2, 2, 2, 1

        """CNN code for task"""
        self.cnn_t = CnnT(depth_1, depth_2, depth_3, depth_4, l_1, l_2, l_3,
                          l_4, w_1, w_2, w_3, w_4)
        """CNN code for person"""
        self.cnn = Cnn(depth_1, depth_2, depth_3, depth_4, l_1, l_2, l_3, l_4,
                       w_1, w_2, w_3, w_4)

    def call(self, xs, training=None):
        output, h_t, h_p = self.conv_ae(xs, training=training)
        prediction_t = self.cnn_t(h_t, xs, training=training)
        prediction_p = self.cnn(h_p, training=training)
        return output, prediction_t, prediction_p
