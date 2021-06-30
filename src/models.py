from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
from tenning.base_model import BaseModel
from tenning.rotation_utils import rotation_matrix_from_ortho6d
from tenning.activations import Swish
from tenning.activations import PReLU
from tenning.linalg_utils import sym_matrix_from_array
import tensorflow as tf


class MCDAttitudeEstimator(BaseModel):

    def __init__(self,
                 batch_size=None,
                 mcd_rate=0.1,
                 initializer='he_normal',
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        self.batch_size = batch_size
        self.initializer = initializer
        self.mcd_rate = mcd_rate

    def build_model(self):

        input_tensor = Input(shape=[9, 1], batch_size=self.batch_size, name=f"{self.name}/profile_matrix")
        # ref_vectors = Input(shape=[self.observations, 3], batch_size=self.batch_size, name=f"{self.name}/ref_vectors")
        # body_vectors = Input(shape=[self.observations, 3], batch_size=self.batch_size, name=f"{self.name}/body_vectors")

        # input_tensor = tf.keras.layers.Concatenate()([ref_vectors, body_vectors])

        x = Conv1D(64, 9, padding='same', kernel_initializer=self.initializer, name=f"{self.name}/conv1")(input_tensor)
        x = Swish()(x)

        x = Dropout(self.mcd_rate)(x)
        x = Conv1D(128, 9, padding='same', kernel_initializer=self.initializer, name=f"{self.name}/conv2")(x)
        x = Swish()(x)

        x = Dropout(self.mcd_rate)(x)
        x = Conv1D(256, 9, kernel_initializer=self.initializer, name=f"{self.name}/conv3")(x)
        x = tf.squeeze(x, axis=1)
        x = Swish()(x)

        x = Dropout(self.mcd_rate)(x)
        x = Dense(512, kernel_initializer=self.initializer, name=f"{self.name}/dense1")(x)
        x = Swish()(x)

        x = Dropout(self.mcd_rate)(x)
        rotations = Dense(6, kernel_initializer=self.initializer, name=f"{self.name}/output")(x)
        rotations = rotation_matrix_from_ortho6d(rotations)  # (batch, 3, 3)

        return {"inputs": input_tensor, "outputs": rotations, "trainable": True}

    @tf.function
    def update_step(self, **data_dict):

        body_vectors = data_dict.get('body_vectors')
        ref_vectors = data_dict.get('ref_vectors')
        true_rotation = data_dict.get('true_rotation')
        # stddevs = data_dict.get('stddevs')

        # # Equation 97 from [Shuster1981]
        # sig_tot = 1. / tf.reduce_sum(1/stddevs**2, axis=1, keepdims=True)
        # # Equation 96 from [Shuster1981]
        # weights = sig_tot / stddevs**2
        # weights = weights[..., tf.newaxis]

        obs = body_vectors.shape[1]
        weights = tf.ones([self.batch_size, obs, 1]) / obs

        # Vectorized form of equation 38 from [Shuster1981]
        profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
        profile_matrix = tf.reshape(profile_matrix, [self.batch_size, -1, 1])

        with tf.GradientTape() as tape:

            rotations = self(profile_matrix, training=True)
            # rotations = self([ref_vectors, body_vectors], training=True)

            predictions = tf.transpose(tf.matmul(rotations, ref_vectors, transpose_b=True), [0, 2, 1])

            total_loss = self.loss(true_rotation, rotations)

        gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(body_vectors, predictions)
        self.metrics[1].update_state(true_rotation, rotations)

        results = [total_loss, self.metrics[0].result(), self.metrics[1].result()]

        return results

    @tf.function
    def predict_step(self, **data_dict):

        body_vectors = data_dict.get('body_vectors')
        ref_vectors = data_dict.get('ref_vectors')
        true_rotation = data_dict.get('true_rotation')
        # stddevs = data_dict.get('stddevs')

        # # Equation 97 from [Shuster1981]
        # sig_tot = 1. / tf.reduce_sum(1/stddevs**2, axis=1, keepdims=True)
        # # Equation 96 from [Shuster1981]
        # weights = sig_tot / stddevs**2
        # weights = weights[..., tf.newaxis]

        obs = body_vectors.shape[1]
        weights = tf.ones([self.batch_size, obs, 1]) / obs

        # Vectorized form of equation 38 from [Shuster1981]
        profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
        profile_matrix = tf.reshape(profile_matrix, [self.batch_size, -1, 1])

        rotations = self(profile_matrix, training=False)
        # rotations = self([ref_vectors, body_vectors], training=False)

        predictions = tf.transpose(tf.matmul(rotations, ref_vectors, transpose_b=True), [0, 2, 1])

        total_loss = self.loss(true_rotation, rotations)

        self.metrics[0].update_state(body_vectors, predictions)
        self.metrics[1].update_state(true_rotation, rotations)

        results = [total_loss, self.metrics[0].result(), self.metrics[1].result()]

        return results

    def get_config(self):

        config = super().get_config()

        config.update({"batch_size": self.batch_size,
                       "mcd_rate": self.mcd_rate,
                       "initializer": self.initializer,
                       "trainable": self.trainable,
                       "name": self.name})

        return config

    @staticmethod
    def get_custom_objs():
        return {'MCDAttitudeEstimator': MCDAttitudeEstimator,
                'Swish': Swish}


class RotationContinuity(BaseModel):
    """ This model is an adaptation of the PointNet used by [Zhou2019].
        The original implementation can be found in https://github.com/papagina/RotationContinuity.

    References:
        - [Zhou2019] Zhou, Yi, et al. "On the continuity of rotation representations in neural networks." 
                        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(self,
                 batch_size=64,
                 out_channel=6,
                 initializer='he_normal',
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        self.batch_size = batch_size
        self.initializer = initializer
        self.out_channel = out_channel

    def feature_extractor(self):

        # (batch, point_num, 3) -> (batch, point_num, 3)
        input_tensor = Input(shape=[None, 3], batch_size=self.batch_size, name=f"{self.name}/feat_extractor/point_cloud")

        # (batch, point_num, 3) -> (batch, point_num, 64)
        x = Conv1D(64, 1, kernel_initializer=self.initializer, name=f"{self.name}/feat_extractor/conv1")(input_tensor)
        x = LeakyReLU()(x)

        # (batch, point_num, 64) -> (batch, point_num, 128)
        x = Conv1D(128, 1, kernel_initializer=self.initializer, name=f"{self.name}/feat_extractor/conv2")(x)
        x = LeakyReLU()(x)

        # (batch, point_num, 128) -> (batch, point_num, 1024)
        x = Conv1D(1024, 1, kernel_initializer=self.initializer, name=f"{self.name}/feat_extractor/conv3")(x)
        # (batch, point_num, 1024) -> (batch, 1024)
        x = GlobalAveragePooling1D()(x)

        return Model(inputs=input_tensor, outputs=x, trainable=True)

    def build_model(self):

        original_point = Input(shape=[None, 3], batch_size=self.batch_size, name=f"{self.name}/original_point")
        target_point = Input(shape=[None, 3], batch_size=self.batch_size, name=f"{self.name}/target_point")

        feature_extractor = self.feature_extractor()

        feature_pt1 = feature_extractor(original_point)
        feature_pt2 = feature_extractor(target_point)

        # (batch, 2048)
        features = Concatenate()([feature_pt1, feature_pt2])

        # (batch, 2048) -> (batch, 512)
        x = Dense(512, kernel_initializer=self.initializer)(features)
        x = LeakyReLU()(x)

        # (batch, 512) -> (batch, self.out_channel)
        rotations = Dense(self.out_channel, kernel_initializer=self.initializer)(x)
        rotations = rotation_matrix_from_ortho6d(rotations)  # (batch, 3, 3)

        return {"inputs": [original_point, target_point], "outputs": rotations, "trainable": True}

    @tf.function
    def update_step(self, **data_dict):
        original_points = data_dict.get("original")
        target_points = data_dict.get("target")
        true_rotation = data_dict.get("true_rotation")

        with tf.GradientTape() as tape:

            rotations = self([original_points, target_points], training=True)

            total_loss = self.loss(true_rotation, rotations)

        gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(true_rotation, rotations)

        results = [total_loss, self.metrics[0].result()]

        return results

    @tf.function
    def predict_step(self, **data_dict):
        original_points = data_dict.get("original")
        target_points = data_dict.get("target")
        true_rotation = data_dict.get("true_rotation")

        rotations = self([original_points, target_points], training=False)

        total_loss = self.loss(true_rotation, rotations)

        self.metrics[0].update_state(true_rotation, rotations)

        results = [total_loss, self.metrics[0].result()]

        return results

    def get_config(self):

        config = super().get_config()

        config.update({"batch_size": self.batch_size,
                       "out_channel": self.out_channel,
                       "initializer": self.initializer,
                       "trainable": self.trainable,
                       "name": self.name})

        return config

    @staticmethod
    def get_custom_objs():
        return {'RotationContinuity': RotationContinuity}


class BModel(BaseModel):

    def __init__(self,
                 batch_size=None,
                 initializer='he_normal',
                 trainable=True,
                 parametrization="dcm",
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        self.batch_size = batch_size
        self.initializer = initializer
        self.parametrization = parametrization

    def build_model(self):

        input_tensor = Input(shape=[9, 1], batch_size=self.batch_size, name=f"{self.name}/profile_matrix")

        x = Conv1D(64, 1, kernel_initializer=self.initializer, name=f"{self.name}/conv1")(input_tensor)
        x = Swish()(x)

        x = Conv1D(128, 9, kernel_initializer=self.initializer, name=f"{self.name}/conv3")(x)
        x = tf.squeeze(x, axis=1)
        x = Swish()(x)

        x = Dense(256, kernel_initializer=self.initializer, name=f"{self.name}/dense1")(x)
        x = Swish()(x)

        if self.parametrization == "dcm":
            x = Dense(6, kernel_initializer=self.initializer, name=f"{self.name}/output")(x)
            rotations = rotation_matrix_from_ortho6d(x)  # (batch, 3, 3)

        else:
            x = Dense(10, kernel_initializer=self.initializer, name=f"{self.name}/output")(x)
            x = Lambda(sym_matrix_from_array, output_shape=[4, 4], name=f"{self.name}/symm_matrix")(x)

            _, x = tf.linalg.eigh(x)

            rotations = x[:, :, 0]

        return {"inputs": input_tensor, "outputs": rotations, "trainable": True}

    @tf.function
    def update_step(self, **data_dict):

        body_vectors = data_dict.get('original')
        ref_vectors = data_dict.get('target')
        true_rotation = data_dict.get('true_rotation')

        body_vectors /= tf.linalg.norm(body_vectors, axis=-1, keepdims=True)
        ref_vectors /= tf.linalg.norm(ref_vectors, axis=-1, keepdims=True)

        obs = body_vectors.shape[1]
        weights = tf.ones([self.batch_size, obs, 1]) / obs

        # Vectorized form of equation 38 from [Shuster1981]
        profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
        profile_matrix = tf.reshape(profile_matrix, [self.batch_size, -1, 1])

        with tf.GradientTape() as tape:

            rotations = self(profile_matrix, training=True)

            total_loss = self.loss(true_rotation, rotations)

        gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(true_rotation, rotations)

        results = [total_loss, self.metrics[0].result()]

        return results

    @tf.function
    def predict_step(self, **data_dict):

        body_vectors = data_dict.get('original')
        ref_vectors = data_dict.get('target')
        true_rotation = data_dict.get('true_rotation')

        body_vectors /= tf.linalg.norm(body_vectors, axis=-1, keepdims=True)
        ref_vectors /= tf.linalg.norm(ref_vectors, axis=-1, keepdims=True)

        obs = body_vectors.shape[1]
        weights = tf.ones([self.batch_size, obs, 1]) / obs

        # Vectorized form of equation 38 from [Shuster1981]
        profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
        profile_matrix = tf.reshape(profile_matrix, [self.batch_size, -1, 1])

        rotations = self(profile_matrix, training=False)

        total_loss = self.loss(true_rotation, rotations)

        self.metrics[0].update_state(true_rotation, rotations)

        results = [total_loss, self.metrics[0].result()]

        return results

    def get_config(self):

        config = super().get_config()

        config.update({"batch_size": self.batch_size,
                       "initializer": self.initializer,
                       "trainable": self.trainable,
                       "parametrization": self.parametrization,
                       "name": self.name})

        return config

    @staticmethod
    def get_custom_objs():
        return {'BModel': BModel, 'Swish': Swish}


class SmoothRepresentation(BaseModel):
    """ This model is an adaptation of the model presented by [Peretroukhin2020].
        The original implementation can be found in https://github.com/utiasSTARS/bingham-rotation-learning.

    References:
        - [Peretroukhin2020] Peretroukhin, Valentin, et al. "A smooth representation of belief over so (3) 
                             for deep rotation learning with uncertainty." arXiv preprint arXiv:2006.01031 (2020).
    """

    def __init__(self,
                 batch_size=32,
                 initializer='he_normal',
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        self.batch_size = batch_size
        self.initializer = initializer

    def build_model(self):
        input_tensor = Input(shape=[None, 6], batch_size=self.batch_size, name=f"{self.name}/feat_extractor/point_cloud")

        x = Conv1D(64, 1, kernel_initializer=self.initializer, name=f"{self.name}/feat_extractor/conv1")(input_tensor)
        x = PReLU()(x)

        x = Conv1D(128, 1, kernel_initializer=self.initializer, name=f"{self.name}/feat_extractor/conv2")(x)
        x = PReLU()(x)

        x = Conv1D(1024, 1, kernel_initializer=self.initializer, name=f"{self.name}/feat_extractor/conv3")(x)
        # (batch, point_num, 1024) -> (batch, 1024)
        x = GlobalAveragePooling1D()(x)

        x = Dense(256)(x)
        x = PReLU()(x)

        x = Dense(128)(x)
        x = PReLU()(x)

        x = Dense(10)(x)

        x = Lambda(sym_matrix_from_array, output_shape=[4, 4], name=f"{self.name}/symm_matrix")(x)

        _, x = tf.linalg.eigh(x)

        q_opt = x[:, :, 0]

        return {"inputs": input_tensor, "outputs": q_opt, "trainable": True}

    @tf.function
    def update_step(self, **data_dict):

        original = data_dict.get('original')
        target = data_dict.get('target')
        true_quaternion = data_dict.get('true_rotation')

        # (batch, point_num, 6)
        input_tensor = tf.concat([original, target], axis=-1)

        with tf.GradientTape() as tape:

            quaternions = self(input_tensor, training=True)

            total_loss = self.loss(true_quaternion, quaternions)

        gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(true_quaternion, quaternions)

        results = [total_loss, self.metrics[0].result()]

        return results

    @tf.function
    def predict_step(self, **data_dict):

        original = data_dict.get('original')
        target = data_dict.get('target')
        true_quaternion = data_dict.get('true_rotation')

        # (batch, point_num, 6)
        input_tensor = tf.concat([original, target], axis=-1)

        quaternions = self(input_tensor, training=False)

        total_loss = self.loss(true_quaternion, quaternions)

        self.metrics[0].update_state(true_quaternion, quaternions)

        results = [total_loss, self.metrics[0].result()]

        return results

    def get_config(self):

        config = super().get_config()

        config.update({"batch_size": self.batch_size,
                       "initializer": self.initializer,
                       "trainable": self.trainable,
                       "name": self.name})

        return config

    @staticmethod
    def get_custom_objs():
        return {'SmoothRepresentation': SmoothRepresentation,
                'PReLU': PReLU}


def get_attitude_model():
    return MCDAttitudeEstimator


def get_pose_model():
    return BModel
