import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tenning.generic_utils import build_json_model
from tenning.rotation_utils import average_rotation_svd
from tenning.rotation_utils import angle_from_dcm
from tenning.rotation_utils import dcm_from_quaternion
from tenning.losses import geodesic_distance
from tenning.losses import quaternion_distance
from src.models import BModel
from src.models import RotationContinuity
from src.models import SmoothRepresentation
from src.utils import read_params
from src.load_dataset import PointLoader


def ang_diff(original_pc, target_pc, true_rotation, model, input_format, parametrization):

    if input_format == "profile_matrix":
        original_pc = original_pc / tf.linalg.norm(original_pc, axis=-1, keepdims=True)
        target_pc = target_pc / tf.linalg.norm(target_pc, axis=-1, keepdims=True)

        batch_size = original_pc.shape[0]
        obs = original_pc.shape[1]
        weights = tf.ones([batch_size, obs, 1]) / obs

        # Vectorized form of equation 38 from [Shuster1981]
        input_data = tf.matmul(original_pc, target_pc * weights, transpose_a=True)
        input_data = tf.reshape(input_data, [batch_size, -1, 1])
    elif input_format == "concat":
        input_data = tf.concat([original_pc, target_pc], axis=-1)
    else:
        input_data = [original_pc, target_pc]

    predictions = model(input_data, training=False)

    if parametrization == 'quaternion':
        true_rotation = dcm_from_quaternion(true_rotation)
        predictions = dcm_from_quaternion(predictions)

    # true_angles = angle_from_dcm(true_rotation)
    # pred_angles = angle_from_dcm(predictions)

    # return true_angles - pred_angles
    return geodesic_distance(true_rotation, predictions)


def main(params):

    available_models = {'bmodel': BModel,
                        'rotation_continuity': RotationContinuity,
                        'smooth_representation': SmoothRepresentation}

    data_dir = params["data_dir"]
    min_angle = params["min_angle"]
    max_angle = params["max_angle"]
    num_rotations = params["num_rotations"]

    test_config = params["config"]

    for model_name, test_params in test_config.items():
        arch = available_models[model_name]

        for parametrization, train_points in test_params.items():
            for train_point, test_points in train_points.items():

                model = f"{model_name}_{parametrization}_{train_point}"
                model_dir = os.path.join(params["model_dir"], model)
                json_model = os.path.join(model_dir, "architecture.json")
                # Builds the trained model given its weights and architecture
                trained_model = build_json_model(model_dir, json_model, arch.get_custom_objs())

                for test_point in test_points:

                    results = pd.DataFrame(columns=['model',
                                                    'test_point',
                                                    'min_angle',
                                                    'max_angle',
                                                    'angular_difference'])

                    if not os.path.exists(params["csv_file"]):
                        results.to_csv(params["csv_file"], index=False)

                    points_loader = PointLoader(data_dir,
                                                num_points=test_point,
                                                val_ratio=1.0,
                                                test_ratio=0.0,
                                                batch_size=num_rotations,
                                                parametrization=parametrization,
                                                min_angle=min_angle,
                                                max_angle=max_angle)
                    data_iterator = points_loader.val_iterator()

                    if model_name == 'bmodel':
                        input_format = "profile_matrix"
                    elif model_name == 'smooth_representation':
                        input_format = "concat"
                    else:
                        input_format = "list"

                    diffs = []
                    for sample in data_iterator:
                        original = sample.get("original")
                        target = sample.get("target")
                        true_rotation = sample.get("true_rotation")

                        angular_difference = ang_diff(original,
                                                      target,
                                                      true_rotation,
                                                      trained_model,
                                                      input_format,
                                                      parametrization)

                        diffs = np.concatenate([diffs, angular_difference])

                    results["angular_difference"] = diffs
                    results["model"] = [model] * len(diffs)
                    results["test_point"] = [test_point] * len(diffs)
                    results["min_angle"] = [min_angle] * len(diffs)
                    results["max_angle"] = [max_angle] * len(diffs)

                    results.to_csv(params["csv_file"], mode='a', header=False, index=False)
