from collections import defaultdict
from constants import LOG_DIR, OUTPUT_DIR, DATA_DIR, MODELS_DIR
from dataset import load_data
import json
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from utils import compute_sensitivity_specificity

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_accuracy(log_file: str):
    """
    Get test accuracy of a model
    :param log_file: log file of the model
    :return: accuracy map, key: subject, value: best test task accuracy
    """
    acc_map = defaultdict(float)
    with open(log_file, "r") as f:
        for line in f.readlines():
            if "subject" in line:
                subject = int(line.split("subject ")[1].split(" ")[0])
            if "test accuracy task" in line:
                acc_map[subject] = max(acc_map[subject], float(line.split("test accuracy task: ")[1].split(" ")[0]))
    return acc_map


def analyze_accuracy():
    # Get original paper accuracy
    paper_accuracy = get_accuracy(os.path.join(LOG_DIR, "paper-model.log"))
    # Get reproduced model accuracy
    reproduced_accuracy = get_accuracy(os.path.join(LOG_DIR, "main-model.log"))
    # Get reproduced ablated model accuracy
    reproduced_ablated_accuracy = get_accuracy(os.path.join(LOG_DIR, "ablated-model.log"))
    # Scatter plot
    plt.scatter(list(paper_accuracy.keys()), list(paper_accuracy.values()), label="Original paper")
    plt.scatter(list(reproduced_accuracy.keys()), list(reproduced_accuracy.values()), label="Reproduced model")
    plt.scatter(list(reproduced_ablated_accuracy.keys()), list(reproduced_ablated_accuracy.values()),
                label="Reproduced ablated model")
    plt.xlabel("Subject")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy of the original model in the paper and reproduced models")
    plt.legend(bbox_to_anchor=(0, -0.3), loc="lower left", borderaxespad=0)
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"), bbox_inches="tight", dpi=300)
    print("Saved accuracy figure to {}".format(os.path.join(OUTPUT_DIR, "accuracy.png")))


def confusion_matrix():
    confusion_matrices = [{}] * 2
    model_types = ("main", "ablated")
    num_samples = 0
    for p_id, (_, _, _, feature_test, lbl_test_t) in load_data(data_dir=DATA_DIR):
        print("===== Processing subject {} =====".format(p_id))
        main_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, f'subject_{p_id}_model'))
        ablated_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, f'subject_{p_id}_ablated_model'))
        for idx, model in enumerate([main_model, ablated_model]):
            _, prediction_t, _ = model(feature_test, training=False)
            print(prediction_t)
            prediction_labels = tf.argmax(prediction_t, axis=1)
            true_labels = tf.argmax(lbl_test_t, axis=1)
            print(prediction_labels)
            print(true_labels)
            sensitivity, specificity = compute_sensitivity_specificity(prediction_t, lbl_test_t)
            print(f"sensitivity: {sensitivity}, specificity: {specificity}")
            confusion_matrices[idx][p_id] = {}
            confusion_matrices[idx][p_id]["sensitivity"] = sensitivity
            confusion_matrices[idx][p_id]["specificity"] = specificity
            print()
        num_samples += 1

    return confusion_matrices[0], confusion_matrices[1]


def analyze_sensitivity_specificity():
    output = {"main": {"sensitivity": [], "specificity": []}, "ablated": {"sensitivity": [], "specificity": []}}
    for conf_matrix, model_type in zip(confusion_matrix(), ("main", "ablated")):
        for subject in sorted(conf_matrix.keys()):
            output[model_type]["sensitivity"].append(conf_matrix[subject]["sensitivity"])
            output[model_type]["specificity"].append(conf_matrix[subject]["specificity"])
    with open(os.path.join(OUTPUT_DIR, "sensitivity_specificity.json"), "w") as f:
        json.dump(output, f, indent=4)


def main():
    analyze_accuracy()
    analyze_sensitivity_specificity()


if __name__ == "__main__":
    main()
