# fmt: off

# Create directory structure if it doesn't exist
import os
for dir_name in ["logs", "data", "models"]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Set up logging
import logging
logger = logging.getLogger("SearchLogger")
logger.setLevel(logging.INFO)
log_file_handler = logging.FileHandler("logs/lambda_search.log", "w")
log_file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_file_handler.setFormatter(formatter)
logger.addHandler(log_file_handler)

import json
import numpy as np
from SequentialLearning.SequentialLearningManager import SequentialLearningManager
from SequentialLearning.Tasks.MNISTClassificationTask import MNISTClassificationTask
from SequentialLearning.EWC_Methods.EWC_Methods import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on
logger.info(f"GPU: {tf.config.list_physical_devices('GPU')}")

# Classes to use for each task
# Each sub-list represents a new task, so 2 or more lists is sequential learning
# For example, when the task is MNIST, a valid structure would be:
# task_digit_classes = [
#     [0, 1, 2],
#     [4, 5, 6],
# ]
# meaning first train on classifying digits 0,1,2 and then classify digits 4,5,6
#
# Choose classes that are sensible for the selected task!
task_classes = [
    [0, 1, 2],
    [4, 5, 6],
]

# Training parameters
epochs = 10
batch_size = 32
ewc_method = EWC_Method.FISHER_MATRIX

# Initial binary search parameters
# Note tuple is ordered by (min_val, max_val)
# For each trial a random low and high param are selected from this range
SEARCH_LOW_PARAM_RANGE = (0, 0)
SEARCH_HIGH_PARAM_RANGE = (50, 100)

# Number of steps for each range
NUM_STEPS = 100

# Value which (high-low) must be smaller than to terminate
# Note: For a non-recursive binary search (e.g. one pass only)
# Simple set this to a value that will *always* be exceeded after one pass
# e.g. if the smallest possible initial range is 0-100 with num_steps = 100
# the largest range in the next pass would be 2 (e.g. 1 on either side of optimal)
# Choosing a search termination of any value greater than 2 would be sufficient!
SEARCH_TERMINATION_THRESHOLD = 10

MODEL_SAVE_PATH = "models/lambda_search_base"
data_file = "data/lambda_search_fisher_data.json"
with open(data_file, "x") as f:
    json.dump([], f)

task_head_layers = [
    [tf.keras.layers.Dense(len(labels))] for labels in task_classes
]

def measure_lambda_performance(experiment_results):
    """
    Currently experiment results are expected to be the validation dictionary
    from SequentialValidation callback. Essentially a dictionary of
    {
        "metric_name": List[List[float]]
    }
    Where metrics names are most likely ["loss", "base_loss"] and lists are 
    indexed by (epoch_index, task_index)

    Current measure of a parameter is given by summing base_loss from the 
    corresponding epoch forward, then summing these results
    -  A perfect sequential learning rule would have each task base_loss drop 
        immediately to a low value and stay there indefinitely, meaning
        our measure would be a minima
    - A terrible sequential learning rule would have all tasks stay at a large
        base_loss, a maxima of our measure
    - A typical sequential learning rule will see a task have low loss during 
        respective training epochs and high loss elsewhere
    - A good sequential learning rule will see a task have low loss during
        respective training epochs and low loss elsewhere

    This measure gives us a good way to evaluate if a value of lambda is
    good or bad - or at least hopefully order them!

    An augmentation to this rule is to discount the weight of a task by a factor
    for each additional task that has past since training - as long distance tasks
    are likely to perform worse (?). This could be implemented with a simple
    constant multiplied by epochs//TASK_EPOCHS but has been eschewed here
    """

    base_losses: np.ndarray = np.array(experiment_results["base_loss"])
    num_tasks = base_losses.shape[1]
    for i in range(1, num_tasks):
        base_losses[:num_tasks*epochs - i*epochs, num_tasks-i] = 0
    measure = np.sum(base_losses)
    return measure


def create_and_save_base_model():
    """
    Create and save base_model
    """
    # base model for sequential tasks
    # each model gets these layers as a base, then adds own head layers
    # i.e. these weights are *shared*
    model_input_shape = (28, 28, 1)
    model_inputs = model_layer = tf.keras.Input(shape=model_input_shape)
    model_layer = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_0")(model_layer)
    model_layer = tf.keras.layers.MaxPool2D((2, 2))(model_layer)
    model_layer = tf.keras.layers.BatchNormalization()(model_layer)
    model_layer = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_1")(model_layer)
    model_layer = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_2")(model_layer)
    model_layer = tf.keras.layers.BatchNormalization()(model_layer)
    model_layer = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_3")(model_layer)
    model_layer = tf.keras.layers.Flatten()(model_layer)
    model_layer = tf.keras.layers.Dense(32, activation="relu")(model_layer)
    model_layer = tf.keras.layers.Dense(32, activation="relu")(model_layer)
    model = tf.keras.Model(inputs=model_inputs, outputs=model_layer, name="base_model")
    print(f"BASE MODEL SUMMARY")
    model.summary()
    model.save(MODEL_SAVE_PATH)


def create_tasks(base_model: tf.keras.models.Model):
    """
    Create a new set of tasks given the task models
    """

    tasks = []
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Create, compile, and build all models
    for task_index in range(len(task_classes)):
        if task_head_layers is None:
            layers = []
        else:
            layers = task_head_layers[task_index]

        curr_model_layer = base_model.output
        for layer in layers:
            curr_model_layer = layer(curr_model_layer)

        curr_model = tf.keras.Model(
            inputs=base_model.inputs, outputs=curr_model_layer, name=f"task_{task_index+1}_model")

        tasks.append(MNISTClassificationTask(
            name=f"Task {task_index+1}",
            model=curr_model,
            model_base_loss=loss_fn,
            task_labels=task_classes[task_index],
            training_batches=0,
            validation_batches=0,
            batch_size=batch_size
        ))
    return tasks


def perform_experiment(current_bin_search_parameter: float):
    """
    Perform a single experiment, meaning train over a single task
    This corresponds to a single value of lambda
    """

    # Load all the models from storages to ensure we have consistent model behavior
    # over all lambda in this repetition
    base_model: tf.keras.models.Model = tf.keras.models.load_model(MODEL_SAVE_PATH)  # type: ignore

    # Create the manager
    manager = SequentialLearningManager(base_model, create_tasks(
        base_model), epochs, ewc_method, current_bin_search_parameter)
    # Train all tasks sequentially
    manager.train_all()
    validation_data = manager.get_validation_data()
    measure = measure_lambda_performance(validation_data)
    # Add the gridsearch to the repetition array
    with open(data_file, "r") as f:
        saved_data = json.load(f)
    saved_data[-1].append((current_bin_search_parameter, measure))
    with open(data_file, "w") as f:
        json.dump(saved_data, f)

    return measure


def perform_repetition():
    """
    With new models, start a gridsearch
    Start with low and high selected from the respective range, and
    choose a number of evenly spaced points between. Then, evaluate
    the measures at each point, and finally choose new high/low points

    Note that for an entire repetition (until the termination criteria of low/high)
    the same initial models are used (saved and loaded multiple times) to avoid variance
    """
    search_low_param = np.random.uniform(
        low=SEARCH_LOW_PARAM_RANGE[0],
        high=SEARCH_LOW_PARAM_RANGE[1]
    )
    search_high_param = np.random.uniform(
        low=SEARCH_HIGH_PARAM_RANGE[0],
        high=SEARCH_HIGH_PARAM_RANGE[1]
    )
    logger.info(f"INITIAL LOW/HIGH ({search_low_param}, {search_high_param})")

    logger.info(f"CREATING NEW MODELS")
    create_and_save_base_model()

    # Add a new array to the experiment file to store this repetition
    with open(data_file, "r") as f:
        saved_data = json.load(f)
    saved_data.append([])
    with open(data_file, "w") as f:
        json.dump(saved_data, f)

    while (search_high_param-search_low_param) > SEARCH_TERMINATION_THRESHOLD:
        target_parameters = np.linspace(search_low_param, search_high_param, NUM_STEPS)
        logger.info(f"{target_parameters=}")
        # Store all results of current loop
        current_gridsearch = []
        for current_param in target_parameters:
            current_measure = perform_experiment(current_param)
            # Very occasionally the model falls over - prevent logging this!
            if np.isnan(current_measure):
                logger.info(f"NAN ERROR ON {current_param}")
                break
            logger.info(f"PARAM {current_param}, MEASURE {current_measure}")
            current_gridsearch.append([current_param, current_measure])

        # Just to ensure the entire gridsearch didn't fall over...
        if len(current_gridsearch) == 0:
            logger.info(f"NO VALID MEASURES ON {target_parameters=}")
            break
        # Find the best measure of this gridsearch and get the indices on either side
        best_measure_index = np.argmin(current_gridsearch, axis=0)[1]  # type: ignore
        new_lower_index = max(best_measure_index-1, 0)
        new_higher_index = min(best_measure_index+1, len(current_gridsearch)-1)

        # Get the new high and low based on those indices
        search_low_param = current_gridsearch[new_lower_index][0]
        search_high_param = current_gridsearch[new_higher_index][0]
        logger.info(f"NEW PARAM RANGE ({search_low_param}, {search_high_param})")


i = 0
while True:
    logger.info("-*-"*50)
    logger.info(f"TRIAL {i}")
    logger.info("-*-"*50)
    i += 1
    perform_repetition()
