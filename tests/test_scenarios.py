import os
import pandas as pd
import tensorflow as tf
from scipy.stats import circmean
from tenning.losses import geodesic_distance
from tenning.generic_utils import build_json_model
from attitude_determination.wahba_solutions import TFWahbaSolutions
from attitude_determination.test_cases import TestCases
from src.utils import read_params
from src.utils import wahba_error
from src.utils import count_flops
from src.models import get_attitude_model


def evaluate(algorithm,
             test_case,
             num_samples,
             csv_file,
             performance=False,
             save_csv=False,
             **kwargs):

    test_cases = TestCases()

    ref_vectors, body_vectors, stddevs = test_cases(num_samples, test_case)

    solution = TFWahbaSolutions(algorithm)

    kwargs["stddevs"] = stddevs

    angle_errors = []
    losses = []
    for index in range(num_samples):

        attitude_result = solution(body_vectors[index], ref_vectors[index], **kwargs)

        pred_vectors = test_cases.apply_rotation([ref_vectors[index]], attitude_result)

        losses.append(wahba_error(body_vectors[index], pred_vectors[0], stddevs))

        angle_errors.append(geodesic_distance(test_cases.true_attitude, attitude_result))

    if performance:

        if algorithm in ('svd', 'q_method'):
            input_signatures = [tf.TensorSpec(body_vectors.shape[1:], tf.float32),
                                tf.TensorSpec(ref_vectors.shape[1:], tf.float32),
                                tf.TensorSpec([len(stddevs)], tf.float32)]
            function = solution.algorithm

        elif algorithm == 'nn':
            input_signatures = [tf.TensorSpec([1, 9, 1], tf.float32)]
            function = tf.function(kwargs["model"])

        else:
            input_signatures = [tf.TensorSpec(body_vectors.shape[1:], tf.float32),
                                tf.TensorSpec(ref_vectors.shape[1:], tf.float32),
                                tf.TensorSpec([len(stddevs)], tf.float32),
                                tf.TensorSpec([], tf.int32)]
            function = solution.algorithm

        print(f"FLOPS: {count_flops(function, input_signatures)}")

    mean_angle = circmean(angle_errors)
    mean_loss = tf.reduce_mean(losses).numpy()

    if kwargs.get('verbose', False):
        print(f"mean angle error (rad): {mean_angle}")
        print(f"mean loss: {mean_loss}")

    if save_csv:

        results = pd.DataFrame(columns=['test_case', 'algorithm', 'phi_error', 'loss'])

        if not os.path.exists(csv_file):
            results.to_csv(csv_file, index=False)

        data = [f"case{test_case}", algorithm, mean_angle, mean_loss]

        results.loc[0] = data

        results.to_csv(csv_file, mode='a', header=False, index=False)


def main(params):

    algorithms = params["algorithms"]

    for algorithm in algorithms:
        if algorithm == "nn":
            arch = get_attitude_model()
            json_model = os.path.join(params["model_dir"], "architecture.json")
            model = build_json_model(params["model_dir"], json_model, arch.get_custom_objs())

        iterations = params["iterations"]
        if iterations < 0:
            raise ValueError("The number of iterations must be greater or equal to zero")

        num_samples = params["samples"]
        if num_samples < 1:
            raise ValueError("The number of samples must be greater than zero")

        if algorithm == "nn":
            kwargs = {"num_iterations": iterations, "model": model}
        else:
            kwargs = {"num_iterations": iterations}

        for test_case in range(1, 13):

            evaluate(algorithm=algorithm,
                     test_case=test_case,
                     csv_file=params["csv_file"],
                     num_samples=num_samples,
                     save_csv=params["save_csv"],
                     performance=params["performance"],
                     **kwargs)
