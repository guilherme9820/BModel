import os
import argparse
import importlib
import tensorflow as tf
import tensorflow_addons as tfa
from tenning.generic_utils import maybe_load_weights
from tenning.generic_utils import save_json_model
from tenning.losses import GeodesicLoss
from tenning.losses import QuatChordalSquaredLoss
from tenning.metrics import GeodesicError
from tenning.metrics import QuaternionDistance
from tenning.metrics import WahbaError
from src.models import BModel
from src.models import MCDAttitudeEstimator
from src.models import RotationContinuity
from src.models import SmoothRepresentation
from src.load_dataset import get_handler
from src.utils import set_vram_growth
from src.utils import read_params
from src.utils import instantiate
import wandb
from wandb.keras import WandbCallback

# Ignore excessive warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

available_metrics = {'wahba_error': WahbaError,
                     'geodesic_error': GeodesicError,
                     'quaternion_error': QuaternionDistance}

available_losses = {'geodesic_loss': GeodesicLoss,
                    'quaternion_loss': QuatChordalSquaredLoss}

available_models = {'mcd_attitude_estimator': MCDAttitudeEstimator,
                    'bmodel': BModel,
                    'rotation_continuity': RotationContinuity,
                    'smooth_representation': SmoothRepresentation}


def train(params):

    saving_dir = os.path.join(params["model_dir"], params["model_name"])
    os.makedirs(saving_dir, exist_ok=True)

    ############## OPTIMIZER ##############
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay([500, 1500], [1e-4, 1e-5, 1e-6])
    lr = schedule(step)  # lr and wd can be a function or a tensor
    wd = 1e-4 * schedule(step)
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    #######################################

    train_params = params["train_params"]

    metrics = train_params["metrics"]

    if not isinstance(metrics, list):
        metrics = [metrics]

    metrics_fn = [available_metrics[metric]() for metric in metrics]
    loss_fn = available_losses[train_params["loss"]]()

    wandb.init(group=params["group_name"], project=params["project_name"])
    # Default values for hyper-parameters
    config = wandb.config  # Config is a variable that holds and saves hyperparameters and inputs
    config.update(params["dataset_params"])
    config.update(train_params)

    ############ CALLBACKS ###############
    checkpoint_template = os.path.join(saving_dir, "best_weights.h5")
    model_saver = tf.keras.callbacks.ModelCheckpoint(checkpoint_template,
                                                     monitor=train_params["monitor"],
                                                     mode='min',
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     verbose=0)
    wandb_callback = WandbCallback(monitor=train_params["monitor"],
                                   verbose=1,
                                   mode='min',
                                   log_weights=False,
                                   save_weights_only=False,
                                   save_model=False)
    ######################################

    trainable_model = instantiate(available_models[params["architecture"]], train_params["hyperparams"])
    trainable_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics_fn)

    if params["save_architecture"]:
        json_model = os.path.join(saving_dir, "architecture.json")
        save_json_model(trainable_model, json_model)

    if params["allow_resume"]:
        maybe_load_weights(saving_dir, trainable_model)

    if params["display_summary"]:
        trainable_model.summary()

    data_handler = get_handler(params["dataset_params"])
    train_iterator = data_handler.train_iterator()
    val_iterator = data_handler.val_iterator()

    return trainable_model.fit(train_iterator,
                               train_params["epochs"],
                               train_params["hyperparams"]["batch_size"],  # Only for progbar, but it does not work properly on Ubuntu for some weird reason
                               validation_iterator=val_iterator,
                               samples=data_handler.get_config()['train_info']['size'],
                               verbose=2,
                               callbacks=[wandb_callback, model_saver])


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="params.yml")
    args.add_argument("--test_name", type=str, default=None)
    parsed_args = args.parse_args()

    params = read_params(parsed_args.config)

    set_vram_growth()

    if parsed_args.test_name:
        test_routine = importlib.import_module(f"tests.{parsed_args.test_name}")
        test_routine.main(params[parsed_args.test_name])
    else:
        history = train(params)
