from constants import LEARNING_RATE, NUM_ITERATIONS, DATA_DIR, ExecutionMode
from dataset import load_data, get_batches
from model import Model
import tensorflow as tf
import time
from utils import train_step, test_step


def main(mode: ExecutionMode):
    train_time, test_time = 0.0, 0.0
    for p_id, (feature_train, lbl_train_t, lbl_train_p, feature_test, lbl_test_t) in load_data(data_dir=DATA_DIR):
        model = Model()

        if mode == ExecutionMode.EVALUATION:
            t_acc = test_step(feature_test, lbl_test_t, model)
            print("test accuracy task: {:.3f}".format(t_acc))
        else:
            train_fea, train_lbl_t, train_lbl_p, num_batches = get_batches(feature_train, lbl_train_t, lbl_train_p,
                                                                           batch_size=feature_test.shape[0])

            """Optimizers"""
            task_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
            t_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

            best_acc = float('-inf')
            print(f'subject {p_id}')
            for idx in range(1, NUM_ITERATIONS + 1):
                task_acc = t_acc = 0.0
                start = time.time()
                for i in range(num_batches):
                    xs, ys_t, ys_p = train_fea[i], train_lbl_t[i], train_lbl_p[i]
                    task_acc += train_step(xs, ys_t, ys_p, model, task_optimizer, is_task=True)
                    t_acc += train_step(xs, ys_t, ys_p, model, t_optimizer, is_task=False)

                p_out = ["train accuracy task: {:.3f}".format(task_acc / num_batches),
                         "train accuracy person: {:.3f}".format(t_acc / num_batches)]
                train_time += time.time() - start

                if idx % 10 == 0:
                    start = time.time()
                    t_acc = test_step(feature_test, lbl_test_t, model)
                    if t_acc > best_acc:
                        best_acc = t_acc
                        print(f"best test accuracy: {best_acc}")
                        # Use `models/subject_{p_id}_ablated_model` for the ablated model
                        model.save(f'models/subject_{p_id}_model')
                    p_out.append("test accuracy task: {:.3f}".format(t_acc))
                    test_time += time.time() - start

                print(f'\titeration {idx}:', ("; ".join(p_out)))

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

    main(mode=ExecutionMode.TRAINING)
