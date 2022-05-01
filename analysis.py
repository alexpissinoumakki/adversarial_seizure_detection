from collections import defaultdict
from constants import LOG_DIR, OUTPUT_DIR
from matplotlib import pyplot as plt
import os


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
    pass


def main():
    analyze_accuracy()


if __name__ == "__main__":
    main()
