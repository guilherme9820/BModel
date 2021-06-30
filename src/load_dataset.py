import sys
import os
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
from tenning.data_utils import IteratorBuilder
from tenning.rotation_utils import gen_random_dcm
from tenning.rotation_utils import gen_boresight_vector
from tenning.rotation_utils import gen_rot_quaternion
from tenning.rotation_utils import rotate_vector


class AttitudeDataset(IteratorBuilder):

    def __init__(self,
                 num_samples=2**13,
                 num_observations=4,
                 add_noise=False,
                 std_range=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.obs = num_observations
        self.num_samples = num_samples
        self.add_noise = add_noise
        self.std_range = std_range or [1e-6, 0.01]

        self.gen_data()

    def gen_data(self):

        ref_vectors = gen_boresight_vector(self.num_samples, self.obs, 15)

        attitudes = gen_random_dcm(self.num_samples).numpy()

        body_vectors = attitudes @ np.transpose(ref_vectors, [0, 2, 1])
        body_vectors = np.transpose(body_vectors, [0, 2, 1])

        def gen_noise(stds):

            # Creates an empty array
            noise = np.zeros((0, 3))
            for std in stds:
                # Generates random values from a Gaussian distribution
                temp = np.random.normal(scale=std, size=(1, 3))
                noise = np.vstack([noise, temp])

            return noise

        # Selects standard deviation values from a given range
        stds = np.random.uniform(self.std_range[0], self.std_range[1], size=(self.num_samples, self.obs))

        if self.add_noise:

            noise = np.apply_along_axis(gen_noise, axis=1, arr=stds)

            body_vectors += noise

        stds = np.tile(stds[..., np.newaxis], [1, 1, 3])

        dataset = tf.concat([ref_vectors, body_vectors, stds, attitudes], axis=1)  # (samples, 13, 3)

        self.set_dataset(dataset.numpy())

    def yielder(self, data):

        ref_vectors = tf.cast(data[:, :self.obs, :], tf.float32)  # (batch, observations, 3)

        body_vectors = tf.cast(data[:, self.obs:(2*self.obs), :], tf.float32)  # (batch, observations, 3)

        stds = tf.cast(data[:, (2*self.obs):(3*self.obs), 0], tf.float32)  # (batch, observations)

        true_rotation = tf.cast(data[:, (3*self.obs):, :], tf.float32)  # (batch, 3, 3)

        return body_vectors, ref_vectors, stds, true_rotation

    def post_process(self, *args):

        return {'body_vectors': args[0],
                'ref_vectors': args[1],
                'stddevs': args[2],
                'true_rotation': args[3]}


class PointLoader(IteratorBuilder):

    def __init__(self,
                 dataset_path,
                 num_points=50,
                 num_rotations=10,
                 parametrization='dcm',
                 **kwargs):

        # It will load a single .pts file at a time
        kwargs.update(batch_size=1)
        super().__init__(**kwargs)

        self.num_points = num_points
        self.dataset_path = dataset_path
        self.parametrization = parametrization
        self.num_rotations = num_rotations

        point_files = np.array(os.listdir(dataset_path))

        self.set_dataset(point_files)

    def yielder(self, data):

        @tf.function
        def string_to_float(element):
            split_data = tf.strings.split(element, sep=" ")
            return tf.strings.to_number(split_data)

        pc_file = tf.strings.join([self.dataset_path, data[0]], "/")

        # Reads the whole text file
        point_clouds = tf.io.read_file(pc_file)

        # Removes trailing whitespaces and newlines
        point_clouds = tf.strings.strip(point_clouds)

        # Converts in a list of lists whose elements are
        # coordinates in string format
        point_clouds = tf.strings.split(point_clouds, sep="\n")

        # Converts the point clouds from string to float32
        point_clouds = tf.map_fn(string_to_float, point_clouds, dtype=tf.float32)

        # Selects 'num_points' random point clouds
        pc_indices = tf.range(tf.shape(point_clouds)[0])[:, tf.newaxis]
        pc_indices = tf.random.shuffle(pc_indices)[:self.num_points]

        point_clouds = tf.gather_nd(point_clouds, pc_indices)

        return point_clouds

    def post_process(self, *data):

        # (1, point_num, 3)
        original_points = tf.reshape(data[0], [1, self.num_points, 3])
        # (batch, point_num, 3)
        original_points = tf.tile(original_points, [self.num_rotations, 1, 1])

        if self.parametrization == 'dcm':
            # (batch, 3, 3)
            random_rotations = gen_random_dcm(self.num_rotations)
        else:
            # (batch, 4)
            random_rotations = gen_rot_quaternion(self.num_rotations)

        target_points = rotate_vector(random_rotations, original_points, self.parametrization)

        return {'original': original_points,
                'target': target_points,
                'true_rotation': random_rotations}


def get_handler(params):

    if params['handler'] == 'point_loader':
        data_handler = PointLoader(params["dataset_path"],
                                   num_points=params["num_points"],
                                   num_rotations=params["num_rotations"],
                                   parametrization=params["parametrization"],
                                   val_ratio=params["val_ratio"],
                                   test_ratio=params["test_ratio"])

    else:
        data_handler = AttitudeDataset(num_samples=params["num_samples"],
                                       num_observations=params["num_observations"],
                                       add_noise=params["add_noise"],
                                       std_range=params["std_range"],
                                       val_ratio=params["val_ratio"],
                                       test_ratio=params["test_ratio"],
                                       batch_size=params["batch_size"])

    return data_handler


# if __name__ == '__main__':

#     args = argparse.ArgumentParser()
#     args.add_argument("--config", default="params.yml")
#     parsed_args = args.parse_args()

#     handler = get_handlers(parsed_args.config)

#     print(handler.get_config())

#     path = "/home/gsantos/Documents/TensorflowProjects/BModel/datasets/shapenet/points"

#     handler = PointLoader(path, num_points=10, num_rotations=2, val_ratio=0.2, test_ratio=0.0, parametrization='quaternion')

#     iterator = handler.train_iterator()

#     print(next(iter(iterator)))
