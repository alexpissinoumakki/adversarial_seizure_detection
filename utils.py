from constants import NUM_CLASSES
import tensorflow as tf


def compute_accuracy_t(prediction_t, v_ys):
    """This function only calculates the acc of CNN_task.
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


def compute_sensitivity_specificity(prediction_t, v_ys):
    conf = tf.math.confusion_matrix(tf.argmax(v_ys, axis=1),
                                    tf.argmax(prediction_t, axis=1), num_classes=NUM_CLASSES).numpy()
    tn, fp, fn, tp = conf.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def loss_object(class1, class2, ae, xs, output, prediction_t, ys_t,
                prediction_p, ys_p):
    """cost calculation"""
    l2_ae = 0.005 * sum(tf.nn.l2_loss(var) for var in ae.trainable_variables)
    l2_class = 0.005 * sum(tf.nn.l2_loss(var) for var in class1.trainable_variables)
    l2_class += 0.005 * sum(tf.nn.l2_loss(var) for var in class2.trainable_variables)

    cross_entropy_t = 10 * tf.reduce_mean(
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
    accuracy = compute_accuracy_t(prediction_t, label_test_t)
    sensitivity, specificity = compute_sensitivity_specificity(prediction_t, label_test_t)

    return accuracy, sensitivity, specificity
