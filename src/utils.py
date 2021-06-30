from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
import numpy as np
import yaml
import inspect


def instantiate(clas, arguments):
    params = inspect.signature(clas).parameters

    valid_arguments = {arg: arguments[arg] for arg in params if arguments.get(arg)}

    return clas(**valid_arguments)


def count_flops(function, input_signatures):

    concrete_func = function.get_concrete_function(*input_signatures)

    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        return flops.total_float_ops


def wahba_error(y_true, y_pred, stddevs: list) -> tf.Tensor:
    """ Loss function proposed by Grace Wahba in [Wahba1965].

    Args:
        y_true: True body vectors.
        y_pred: Predicted body vectors.
        stddevs: Standard deviations for each measurement vector.

    Returns:
        Cost corresponding to the difference between the true and predicted body vectors.

    References:
        - [Wahba1965] Wahba, Grace. "A least squares estimate of satellite attitude." SIAM review 7.3 (1965): 409-409.
    """

    stddevs = tf.convert_to_tensor(stddevs)

    # Equation 97 from [Shuster1981]
    sig_tot = 1. / tf.reduce_sum(1/stddevs**2)
    # Equation 96 from [Shuster1981]
    weights = sig_tot / stddevs**2

    error = weights * tf.norm(y_true - y_pred, axis=1)

    return 0.5 * tf.reduce_mean(error)


def set_vram_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        print("Num GPUs Available: ", len(gpus))
        try:
            for gpu in gpus:
                # To prevent crash of CUDNN and lack of video memory
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("Could not find any GPU.")


def read_params(config_path):
    """ Reads a YAML file and converts it to a python object.

    Args:
        config_path: The YAML file path.

    Returns:
        A python dictionary containing all YAML values.
    """

    # define custom tag handler
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def multiply(loader, node):
        seq = loader.construct_sequence(node)
        return int(np.prod([float(i) for i in seq]))

    # register the tag handler
    yaml.add_constructor('!join', join)
    yaml.add_constructor('!multiply', multiply)

    with open(config_path) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config
