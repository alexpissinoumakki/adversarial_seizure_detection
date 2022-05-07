from constants import LOG_DIR, OUTPUT_DIR, DATA_DIR, MODELS_DIR
from dataset import load_data
import json
from matplotlib import pyplot as plt
from mdtable import MDTable
import os
import tensorflow as tf
import re
from utils import compute_sensitivity_specificity

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_log_map(log_file: str):
    """
    Get test accuracy of a model
    :param log_file: log file of the model
    :return: accuracy map, key: subject, value: best test task accuracy
    """
    log_map = {"accuracy": [], "sensitivity": [], "specificity": []}
    subject_regex = re.compile(r".+subject (\d)")
    metrics_regex = re.compile(r".+test accuracy task: (.+); test sensitivity task: (.+); test specificity task: (.+)")
    with open(log_file, "r") as f:
        for line in f.readlines():
            # find subject
            subject_match = subject_regex.match(line)
            if subject_match:
                log_map["accuracy"].append(0.0)
                log_map["sensitivity"].append(0.0)
                log_map["specificity"].append(0.0)
            # find groups that match the regex in the line
            match = metrics_regex.search(line)
            if match:
                log_map["accuracy"][-1] = max(log_map["accuracy"][-1], float(match.group(1)))
                log_map["sensitivity"][-1] = max(log_map["sensitivity"][-1], float(match.group(2)))
                log_map["specificity"][-1] = max(log_map["specificity"][-1], float(match.group(3)))
    return log_map


def analyze_metrics():
    # Get original paper log map
    paper_log_map = get_log_map(os.path.join(LOG_DIR, "paper-model.log"))
    # Get reproduced model log map
    reproduced_log_map = get_log_map(os.path.join(LOG_DIR, "main-model.log"))
    # Get reproduced ablated model log map
    reproduced_ablated_log_map = get_log_map(os.path.join(LOG_DIR, "ablated-model.log"))
    for metric in ("accuracy", "sensitivity", "specificity"):
        if metric == "accuracy":
            # Scatter plot metric
            x = range(len(paper_log_map[metric]))
            plt.figure()
            plt.scatter(x, paper_log_map[metric], label="paper")
            plt.scatter(x, reproduced_log_map[metric], label="reproduced")
            plt.scatter(x, reproduced_ablated_log_map[metric], label="reproduced ablated")
            plt.xlabel("Subject")
            plt.ylabel(f"{metric.capitalize()}")
            plt.title(f"{metric.capitalize()} of the paper model vs the reproduced model vs"
                      " the reproduced ablated model")
            plt.legend(bbox_to_anchor=(0, -0.3), loc="lower left", borderaxespad=0)
            plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}.png"), bbox_inches="tight", dpi=300)
            print("Saved {} figure to {}".format(metric, os.path.join(OUTPUT_DIR, f"{metric}.png")))
        save_metrics({"paper": paper_log_map[metric], "reproduced": reproduced_log_map[metric],
                      "reproduced_ablated": reproduced_ablated_log_map[metric]}, metric, f"{metric}.csv")


def save_metrics(metrics_map: dict, metric_name: str, file_name: str):
    with open(os.path.join(OUTPUT_DIR, file_name), "w") as f:
        paper = metrics_map["paper"]
        main_ = metrics_map["reproduced"]
        ablated = metrics_map["reproduced_ablated"]
        f.write('paper,reproduced,reproduced ablated\n')
        f.writelines(["{},{},{}\n".format(p, m, a) for p, m, a in zip(paper, main_, ablated)])
        print("Saved {} metrics to {}".format(metric_name, os.path.join(OUTPUT_DIR, file_name)))
    table = MDTable(os.path.join(OUTPUT_DIR, file_name))
    markdown_table = table.get_table()
    print(f"\n{metric_name} metrics table:")
    print(markdown_table)


def confusion_matrix():
    confusion_matrices = [{}] * 2
    model_names = ("main", "ablated")
    metric_names = ("spec", "sens")
    for p_id, (_, _, _, feature_test, lbl_test_t) in load_data(data_dir=DATA_DIR):
        print("===== Processing subject {} =====".format(p_id))
        for idx in range(4):
            model_idx, metric_idx = idx // 2, idx % 2
            model_name, metric_name = model_names[model_idx], metric_names[metric_idx]
            model = tf.keras.models.load_model(os.path.join(MODELS_DIR, model_name,
                                                            f'subject_{p_id}_{metric_name}_{model_name}_model'))
            _, prediction_t, _ = model(feature_test, training=False)
            if idx == 0 or idx == 2:
                print(f"\n{model_name} model")
            sensitivity, specificity = compute_sensitivity_specificity(prediction_t, lbl_test_t)
            if p_id not in confusion_matrices[model_idx]:
                confusion_matrices[model_idx][p_id] = {}
            if metric_idx == 0:
                print(f"specificity: {specificity}")
                confusion_matrices[model_idx][p_id]["specificity"] = specificity
            else:
                confusion_matrices[model_idx][p_id]["sensitivity"] = sensitivity
                print(f"sensitivity: {sensitivity}")

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
    analyze_metrics()
    analyze_sensitivity_specificity()


if __name__ == "__main__":
    main()
