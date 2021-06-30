import os
import numpy as np
import pandas as pd
import tensorflow as tf
from attitude_determination.test_cases import TestCases
from tenning.generic_utils import build_json_model
from tenning.rotation_utils import average_rotation_svd
from tenning.rotation_utils import angle_from_dcm
from src.models import get_attitude_model
from src.utils import read_params


def get_ang_diff(model, test_case, rounds):

    test_cases = TestCases()

    ref_vectors, body_vectors, _ = test_cases(1, test_case)

    body_vectors = tf.cast(body_vectors, "float32")
    ref_vectors = tf.cast(ref_vectors, "float32")

    # Generates equal weights for each sample (star tracking scenario)
    obs = body_vectors.shape[1]
    weights = tf.ones([1, obs, 1]) / obs

    # Builds the profile matrices for each sample
    profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
    profile_matrix = tf.reshape(profile_matrix, [1, -1, 1])

    pred_attitudes = []
    for _ in range(rounds):
        # Setting training=True enables the dropout layers
        pred = model(profile_matrix, training=True)

        pred_attitudes.append(pred)

    pred_attitudes = np.vstack(pred_attitudes)  # (rounds, 3, 3)
    pred_angles = angle_from_dcm(pred_attitudes)

    true_angle = angle_from_dcm(test_cases.true_attitude)

    angular_difference = np.abs(true_angle - pred_angles)

    return angular_difference


def main(params):

    arch = get_attitude_model()

    model_name = params["model_name"]

    for observation in params["observations"]:
        for dropout in params["dropout"]:

            model_dir = os.path.join(params["model_dir"], f"{model_name}_{observation}_{dropout}")
            json_model = os.path.join(model_dir, "architecture.json")
            # Builds the trained model given its weights and architecture
            dropout_model = build_json_model(model_dir, json_model, arch.get_custom_objs())

            results = pd.DataFrame(columns=['observations',
                                            'dropout',
                                            'case',
                                            'angular_difference'])

            if not os.path.exists(params["csv_file"]):
                results.to_csv(params["csv_file"], index=False)

            for case_number in range(1, 13):
                angular_difference = get_ang_diff(dropout_model, case_number, params["rounds"])

                results["observations"] = [observation] * params["rounds"]
                results["dropout"] = [dropout] * params["rounds"]
                results["case"] = [case_number] * params["rounds"]
                results["angular_difference"] = angular_difference

                results.to_csv(params["csv_file"], mode='a', header=False, index=False)
