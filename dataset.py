from constants import NUM_CLASSES, NUM_FEATURES, SAMPLE_PER_SUBJECT, SEQUENCE_LENGTH, NUM_TRAINING, NUM_SUBJECTS
import numpy as np
import os
import pickle
import sklearn.preprocessing


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


def load_data(data_dir):
    """
    Loads the data from the given path.
    """

    # THU seizure data reading
    # the first 21 columns are features, the 22nd column is seizure/normal,
    # the 23rd column is person label.
    # in the task label, 0: normal, 1: seizure
    all_data = pickle.load(open(os.path.join(data_dir, "all_14sub.p"), "rb"),
                           encoding='iso-8859-1')

    scaler = sklearn.preprocessing.MinMaxScaler()  # normalization
    f = scaler.fit_transform(all_data[:, :NUM_FEATURES])  # scale to [0, 1]

    # only use the task ID
    all_data = np.hstack((f, all_data[:, NUM_FEATURES:NUM_FEATURES + 1]))

    """Make person label"""
    # the number of samples of each subject after reshape
    n_sample_ = int(2 * SAMPLE_PER_SUBJECT / SEQUENCE_LENGTH)
    ll = np.ones([n_sample_, 1]) * 0
    for hh in range(1, NUM_TRAINING):
        ll_new = np.ones([n_sample_, 1]) * hh
        ll = np.vstack((ll, ll_new))

    ll_test = np.ones([n_sample_, 1]) * NUM_TRAINING

    for p_id in range(NUM_SUBJECTS):
        """Select train and test subject"""
        data_ = all_data[SAMPLE_PER_SUBJECT * p_id:SAMPLE_PER_SUBJECT * (p_id + 1)]

        lst = range(SAMPLE_PER_SUBJECT * p_id, SAMPLE_PER_SUBJECT * (p_id + 1))
        data = np.delete(all_data, lst, axis=0)
        # overlap
        train_data = extract(data, n_fea=NUM_FEATURES, time_window=SEQUENCE_LENGTH,
                             moving=SEQUENCE_LENGTH / 2, n_classes=NUM_CLASSES)
        test_data = extract(data_, n_fea=NUM_FEATURES, time_window=SEQUENCE_LENGTH,
                            moving=SEQUENCE_LENGTH / 2, n_classes=NUM_CLASSES)

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
        lbl_train_t = one_hot(lbl_train_t)
        lbl_train_p = one_hot(lbl_train_p)

        yield p_id, (feature_train, lbl_train_t, lbl_train_p, feature_test, lbl_test_t)


def get_batches(feature_train, lbl_train_t, lbl_train_p, batch_size):
    a = feature_train

    # batch split
    batch_size = int(batch_size)
    train_fea = []
    n_group = int(feature_train.shape[0] / batch_size)
    for i in range(n_group):
        f = a[0 + batch_size * i:batch_size + batch_size * i]
        train_fea.append(f)

    train_lbl_t = []
    for i in range(n_group):
        f = lbl_train_t[0 + batch_size * i:batch_size + batch_size * i, :]
        train_lbl_t.append(f)

    train_lbl_p = []
    for i in range(n_group):
        f = lbl_train_p[0 + batch_size * i:batch_size + batch_size * i, :]
        train_lbl_p.append(f)

    return train_fea, train_lbl_t, train_lbl_p, n_group
