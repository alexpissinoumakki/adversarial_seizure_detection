from enum import Enum
import numpy as np
import os
import pickle
from sklearn import preprocessing
import tensorflow as tf
import time
import tqdm


# Enum Class denoting the execution mode
class ExecutionMode(Enum):
    EVALUATION = 1
    TRAINING = 2
    FULL = 3


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] to
    # [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def extract(input_, n_fea, time_window, moving, n_classes):
    xx = input_[:, :n_fea]
    yy = input_[:, n_fea:n_fea + 1]
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)
    for i in range(number):
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
        if ave_y in range(n_classes + 1):
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    # add the last sample again, to make the sample number round
    data = np.vstack((data, data[-1]))
    return data


def compute_accuracy_t(prediction_t, v_ys):
    """This function only calculate the acc of CNN_task.
    """
    correct_prediction = tf.math.equal(tf.math.argmax(prediction_t, axis=1),
                                       tf.math.argmax(v_ys, axis=1))
    accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, tf.float64))
    return accuracy


def compute_accuracy_p(prediction_p, v_ys):
    """This function only calculates the acc of CNN_task.
    """
    correct_prediction = tf.math.equal(tf.math.argmax(prediction_p, axis=1),
                                       tf.math.argmax(v_ys, axis=1))
    accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, tf.float64))
    return accuracy


"""Models"""


class ConvAE(tf.keras.Model):
    def __init__(self, keep_prob, seg_length, no_fea, name=None):
        super(ConvAE, self).__init__(name=name)
        self.keep_prob = keep_prob
        self.seg_length = seg_length
        self.no_fea = no_fea
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

        self.dropout = tf.keras.layers.Dropout(1 - self.keep_prob)

    def call(self, xs):
        input_ = tf.reshape(xs,
                            [-1, self.seg_length, self.no_fea, 1])  # [200, 14]
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

        output = tf.reshape(output, [-1, self.seg_length * self.no_fea])

        return output, h_t, h_p


class CnnT(tf.keras.Model):
    def __init__(self, depth_1, depth_2, depth_3, depth_4, l_1, l_2, l_3, l_4,
                 w_1, w_2, w_3, w_4, keep_prob, n_class_t,
                 name=None):
        super(CnnT, self).__init__(name=name)
        self.keep_prob = keep_prob

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
        self.dropout1 = tf.keras.layers.Dropout(1 - self.keep_prob)
        self.dropout2 = tf.keras.layers.Dropout(1 - self.keep_prob)

        # """Add another FC layer"""
        self.fc1 = tf.keras.layers.Dense(units=300, activation='sigmoid')
        dim_hidden = 21
        self.fc2 = tf.keras.layers.Dense(units=dim_hidden,
                                         activation='sigmoid')
        # Attention layer: Comment out the two lines below if running `attention` ablation
        self.fc3 = tf.keras.layers.Dense(units=dim_hidden,
                                         activation='sigmoid')
        self.multiply = tf.keras.layers.Multiply()

        self.fc4 = tf.keras.layers.Dense(units=n_class_t)

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
                 w_1, w_2, w_3, w_4, keep_prob, n_class_p,
                 name=None):
        super(Cnn, self).__init__(name=name)
        self.keep_prob = keep_prob

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
        self.dropout = tf.keras.layers.Dropout(1 - self.keep_prob)

        dim_hidden_p = 200
        self.fc1 = tf.keras.layers.Dense(units=dim_hidden_p,
                                         activation='sigmoid')
        self.fc2 = tf.keras.layers.Dense(units=n_class_p)

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
    def __init__(self, keep_prob, seg_length, no_fea, n_class_t, n_class_p,
                 name=None):
        super(Model, self).__init__(name=name)
        self.keep_prob = keep_prob
        self.conv_ae = ConvAE(self.keep_prob, seg_length, no_fea)

        depth_1, depth_2, depth_3, depth_4 = 16, 32, 64, 128
        l_1, l_2, l_3, l_4 = 2, 2, 2, 2
        w_1, w_2, w_3, w_4 = 2, 2, 2, 1

        """CNN code for task"""
        self.cnn_t = CnnT(depth_1, depth_2, depth_3, depth_4, l_1, l_2, l_3,
                          l_4, w_1, w_2, w_3, w_4, self.keep_prob,
                          n_class_t)
        """CNN code for person"""
        self.cnn = Cnn(depth_1, depth_2, depth_3, depth_4, l_1, l_2, l_3, l_4,
                       w_1, w_2, w_3, w_4, self.keep_prob,
                       n_class_p)

    def call(self, xs, training=None):
        output, h_t, h_p = self.conv_ae(xs, training=training)
        prediction_t = self.cnn_t(h_t, xs, training=training)
        prediction_p = self.cnn(h_p, training=training)
        return output, prediction_t, prediction_p


def loss_object(class1, class2, ae, xs, output, prediction_t, ys_t,
                prediction_p, ys_p):
    """cost calculation"""
    l2_ae = 0.005 * sum(tf.nn.l2_loss(var) for var in ae.trainable_variables)
    l2_class = 0.005 * sum(
        tf.nn.l2_loss(var) for var in class1.trainable_variables)
    l2_class += 0.005 * sum(
        tf.nn.l2_loss(var) for var in class2.trainable_variables)

    cross_entropy_t = 5 * tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction_t, labels=tf.stop_gradient(ys_t)))
    cross_entropy_p = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction_p, labels=tf.stop_gradient(ys_p)))

    xs_ = tf.cast(xs, tf.float32)
    cost_ae = tf.reduce_mean(input_tensor=tf.pow(xs_ - output, 2)) + l2_ae

    cost = cost_ae + cross_entropy_t + cross_entropy_p + l2_class + l2_ae
    return cost, cross_entropy_t


def train_step(xs, ys_t, ys_p, model, optimizer, is_task):
    with tf.GradientTape() as tape:
        output, prediction_t, prediction_p = model(xs, training=True)
        cost, cross_entropy_t = loss_object(model.cnn_t, model.cnn,
                                            model.conv_ae, xs, output,
                                            prediction_t, ys_t, prediction_p,
                                            ys_p)
    if is_task:
        task_gradients = tape.gradient(cost, model.trainable_variables)
        optimizer.apply_gradients(
            zip(task_gradients, model.trainable_variables))

        return compute_accuracy_t(prediction_t, ys_t)
    else:
        t_gradients = tape.gradient(cross_entropy_t, model.trainable_variables)
        optimizer.apply_gradients(zip(t_gradients, model.trainable_variables))

        return compute_accuracy_p(prediction_p, ys_p)


def test_step(feature_test, label_test_t, model):
    _, prediction_t, _ = model(feature_test, training=False)

    return compute_accuracy_t(prediction_t, label_test_t)


def main(mode: ExecutionMode, data_dir: str, model_dir: str):
    # THU seizure data reading
    # the first 21 columns are features, the 22nd column is seizure/normal,
    # the 23rd column is person label.
    # in the task label, 0: normal, 1: seizure

    # data reading
    all_data = pickle.load(open(os.path.join(data_dir, "all_14sub.p"), "rb"),
                           encoding='iso-8859-1')
    print(type(all_data), all_data.shape, all_data[:, -1])

    n_classes = 2
    # the number of training subjects
    n_person_ = 13
    # we have overlapping now
    sample_per_subject = 250 * 500
    print(type(all_data), all_data.shape, all_data[:, -1])

    # data.shape[-1] - 1
    no_fea = 21
    # 255 for raw data, 96 for layer 23, 64 for layer 2, 32 for layer 2
    seg_length = 250

    scaler = preprocessing.MinMaxScaler()  # normalization
    f = scaler.fit_transform(all_data[:, :no_fea])  # scale to [0, 1]

    # only use the task ID
    all_data = np.hstack((f, all_data[:, no_fea:no_fea + 1]))

    """Make person label"""
    # the number of samples of each subject after reshape
    n_sample_ = int(2 * sample_per_subject / seg_length)
    ll = np.ones([n_sample_, 1]) * 0
    for hh in range(1, n_person_):
        ll_new = np.ones([n_sample_, 1]) * hh
        ll = np.vstack((ll, ll_new))
    print('the shape of made person label', ll.shape)

    ll_test = np.ones([n_sample_, 1]) * n_person_

    ss_train = time.process_time()

    lr = 0.00001  # use 0.0001 for parameter tuning
    keep = 0.8
    n_class_t = 2  # 0-3
    n_class_p = n_person_  # 0-8
    train_time, test_time = 0.0, 0.0
    models = []
    # Person Independent
    for P_ID in range(14):
        """Select train and test subject"""
        data_ = all_data[sample_per_subject * P_ID:sample_per_subject * (P_ID + 1)]

        lst = range(sample_per_subject * P_ID, sample_per_subject * (P_ID + 1))
        data = np.delete(all_data, lst, axis=0)
        # overlap
        train_data = extract(data, n_fea=no_fea, time_window=seg_length,
                             moving=seg_length / 2, n_classes=n_classes)
        test_data = extract(data_, n_fea=no_fea, time_window=seg_length,
                            moving=seg_length / 2, n_classes=n_classes)

        """Replace the original person data by the made data"""
        # here is - 2, because has two IDs
        no_fea_long = train_data.shape[-1] - 1
        train_data = np.hstack((train_data[:, :no_fea_long + 1], ll))
        test_data = np.hstack((test_data[:, :no_fea_long + 1], ll_test))
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

        feature_train = train_data[:, :no_fea_long]
        feature_test = test_data[:, :no_fea_long]
        lbl_train_t = train_data[:, no_fea_long:no_fea_long + 1]
        lbl_test_t = test_data[:, no_fea_long:no_fea_long + 1]
        lbl_train_p = train_data[:, no_fea_long + 1:no_fea_long + 2]

        lbl_test_t = one_hot(lbl_test_t)

        model = Model(keep, seg_length, no_fea, n_class_t, n_class_p)

        if mode == ExecutionMode.EVALUATION:
            t_acc = test_step(feature_test, lbl_test_t, model)
            print("test accuracy task: {:.3f}".format(t_acc))
        else:
            lbl_train_t = one_hot(lbl_train_t)
            lbl_train_p = one_hot(lbl_train_p)

            a = feature_train

            # batch split
            batch_size = int(feature_test.shape[0])
            train_fea = []
            n_group = int(feature_train.shape[0] / feature_test.shape[0])
            for i in range(n_group):
                f = a[0 + batch_size * i:batch_size + batch_size * i]
                train_fea.append(f)
            print(train_fea[0].shape)

            train_lbl_t = []
            for i in range(n_group):
                f = lbl_train_t[0 + batch_size * i:batch_size + batch_size * i, :]
                train_lbl_t.append(f)
            print(train_lbl_t[0].shape)

            train_lbl_p = []
            for i in range(n_group):
                f = lbl_train_p[0 + batch_size * i:batch_size + batch_size * i, :]
                train_lbl_p.append(f)
            print(train_lbl_p[0].shape)

            """Optimizers"""
            task_optimizer = tf.keras.optimizers.Adam(lr)
            t_optimizer = tf.keras.optimizers.Adam(lr)

            pbar = tqdm.tqdm(range(1, 251))
            best_acc = float('-inf')
            print(f'subject {P_ID}')
            for idx in pbar:  # 250 iterations
                task_acc = t_acc = 0.0
                start = time.time()
                for i in range(n_group):
                    xs, ys_t, ys_p = train_fea[i], train_lbl_t[i], train_lbl_p[i]
                    task_acc += train_step(xs, ys_t, ys_p, model, task_optimizer,
                                           is_task=True)
                    t_acc += train_step(xs, ys_t, ys_p, model, t_optimizer,
                                        is_task=False)

                p_out = ["train accuracy task: {:.3f}".format(task_acc / n_group),
                         "train accuracy person: {:.3f}".format(t_acc / n_group)]
                train_time += time.time() - start

                if mode == ExecutionMode.FULL and idx % 10 == 0:
                    start = time.time()
                    t_acc = test_step(feature_test, lbl_test_t, model)
                    if t_acc > best_acc:
                        best_acc = t_acc
                        print(f"best test accuracy: {best_acc}")
                        # Use `models/subject_{P_ID}_ablated_model` for the ablated model
                        model.save(f'models/subject_{P_ID}_model')
                    p_out.append("test accuracy task: {:.3f}".format(t_acc))
                    test_time += time.time() - start

                # print(f'\titeration {idx}:', ("; ".join(p_out)))
                pbar.set_description(f'\titeration {idx}: {"; ".join(p_out)}')
            models.append(model)

            print(f'total train time: {train_time}, total test time: {test_time}')


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    assert tf.executing_eagerly()

    main(mode=ExecutionMode.FULL, data_dir="data", model_dir="model")
