from enum import Enum

# Data constants
DATA_DIR = "data"
NUM_TRAINING = 13
NUM_CLASSES = 2
NUM_FEATURES = 21
SEQUENCE_LENGTH = 250
SAMPLE_PER_SUBJECT = 250 * 500
NUM_SUBJECTS = 14

# Model-specific constants
NUM_ITERATIONS = 250
LEARNING_RATE = 0.00001  # use 0.0001 for parameter tuning
DROPOUT_RATE = 0.2

# Location
LOG_DIR = "logs"
OUTPUT_DIR = "figures"


# Enum Class denoting the execution mode
class ExecutionMode(Enum):
    EVALUATION = 1
    TRAINING = 2
