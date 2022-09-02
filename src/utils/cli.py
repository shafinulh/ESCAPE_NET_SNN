import argparse


def SNN_arg_parser():
    parser = argparse.ArgumentParser(description="SNN training")
    parser.add_argument("--gpu", default=True, type=bool, help="use gpu")
    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="seed for random number"
    )
    parser.add_argument("--dataset", default="RAT4", type=str, help="dataset name")
    parser.add_argument("--batch_size", default=32, type=int, help="minibatch size")
    parser.add_argument(
        "--architecture", default="ESCAPE_NET", type=str, help="network architecture"
    )
    parser.add_argument("-lr", default=1e-4, type=float, help="initial learning_rate")
    parser.add_argument(
        "--pretrained_ann", default="", type=str, help="pretrained ANN model"
    )
    parser.add_argument(
        "--pretrained_snn",
        default="",
        type=str,
        help="pretrained SNN for inference",
    )
    parser.add_argument(
        "--log",
        default=False,
        type=bool,
        help="to print the output on terminal or to log file",
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--timesteps", default=28, type=int, help="simulation timesteps"
    )
    parser.add_argument("--leak", default=1.0, type=float, help="membrane leak")
    parser.add_argument(
        "--default_threshold",
        default=1.0,
        type=float,
        help="intial threshold to train SNN from scratch",
    )
    parser.add_argument(
        "--activation",
        default="Linear",
        type=str,
        help="SNN activation function",
        choices=["Linear", "STDB"],
    )
    parser.add_argument(
        "--alpha", default=0.3, type=float, help="parameter alpha for STDB"
    )
    parser.add_argument(
        "--beta", default=0.01, type=float, help="parameter beta for STDB"
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        help="optimizer for SNN backpropagation",
        choices=["SGD", "Adam"],
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="weight decay parameter for the optimizer",
    )
    parser.add_argument(
        "--momentum",
        default=0.95,
        type=float,
        help="momentum parameter for the SGD optimizer",
    )
    parser.add_argument(
        "--amsgrad",
        default=True,
        type=bool,
        help="amsgrad parameter for Adam optimizer",
    )
    parser.add_argument(
        "--dropout", default=0.2, type=float, help="dropout percentage for conv layers"
    )
    parser.add_argument(
        "--kernel_size", default=8, type=int, help="filter size for the conv layers"
    )
    parser.add_argument(
        "--test_acc_every_batch",
        action="store_true",
        help="print acc of every batch during inference",
    )
    parser.add_argument(
        "--train_acc_batches",
        default=50,
        type=int,
        help="print training progress after this many batches",
    )
    parser.add_argument(
        "--devices", default="0", type=str, help="list of gpu device(s)"
    )
    parser.add_argument(
        "--dataset_path",
        default="",
        type=str,
        help="",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
        help="",
    )
    parser.add_argument("--save_name", default="", type=str, help="")
    parser.add_argument("-f")
    args = parser.parse_args()
    return args


def SNN_arg_parser():
    parser = argparse.ArgumentParser(description="SNN training")
    parser.add_argument("--gpu", default=True, type=bool, help="use gpu")
    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="seed for random number"
    )
    parser.add_argument("--dataset", default="RAT4", type=str, help="dataset name")
    parser.add_argument("--batch_size", default=32, type=int, help="minibatch size")
    parser.add_argument(
        "--architecture", default="ESCAPE_NET", type=str, help="network architecture"
    )
    parser.add_argument("-lr", default=1e-4, type=float, help="initial learning_rate")
    parser.add_argument(
        "--pretrained_ann", default="", type=str, help="pretrained ANN model"
    )
    parser.add_argument(
        "--pretrained_snn", default="", type=str, help="pretrained SNN for inference"
    )
    parser.add_argument(
        "--log",
        default=True,
        type=bool,
        help="to print the output on terminal or to log file",
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--timesteps", default=28, type=int, help="simulation timesteps"
    )
    parser.add_argument("--leak", default=1.0, type=float, help="membrane leak")
    parser.add_argument(
        "--scaling_factor",
        default=1.0,
        type=float,
        help="scaling factor for thresholds at reduced timesteps",
    )
    parser.add_argument(
        "--default_threshold",
        default=1.0,
        type=float,
        help="intial threshold to train SNN from scratch",
    )
    parser.add_argument(
        "--activation",
        default="Linear",
        type=str,
        help="SNN activation function",
        choices=["Linear", "STDB"],
    )
    parser.add_argument(
        "--alpha", default=0.3, type=float, help="parameter alpha for STDB"
    )
    parser.add_argument(
        "--beta", default=0.01, type=float, help="parameter beta for STDB"
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        help="optimizer for SNN backpropagation",
        choices=["SGD", "Adam"],
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="weight decay parameter for the optimizer",
    )
    parser.add_argument(
        "--momentum",
        default=0.95,
        type=float,
        help="momentum parameter for the SGD optimizer",
    )
    parser.add_argument(
        "--amsgrad",
        default=True,
        type=bool,
        help="amsgrad parameter for Adam optimizer",
    )
    parser.add_argument(
        "--dropout", default=0.2, type=float, help="dropout percentage for conv layers"
    )
    parser.add_argument(
        "--kernel_size", default=8, type=int, help="filter size for the conv layers"
    )
    parser.add_argument(
        "--test_acc_every_batch",
        action="store_true",
        help="print acc of every batch during inference",
    )
    parser.add_argument(
        "--train_acc_batches",
        default=50,
        type=int,
        help="print training progress after this many batches",
    )
    parser.add_argument(
        "--devices", default="0", type=str, help="list of gpu device(s)"
    )
    parser.add_argument(
        "--dataset_path",
        default="",
        type=str,
        help="",
    )
    parser.add_argument("--save_dir", default="", type=str, help="")
    parser.add_argument("--save_name", default="", type=str, help="")
    # parser.add_argument('-f')
    args = parser.parse_args()


def ANN_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train ANN to be later converted to SNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu", default=True, type=bool, help="use gpu")
    parser.add_argument(
        "--log",
        action="store_true",
        help="to print the output on terminal or to log file",
    )
    parser.add_argument(
        "-s", "--seed", default=120, type=int, help="seed for random number"
    )
    parser.add_argument(
        "--dataset",
        default="RAT4",
        type=str,
        help="dataset name",
        choices=["RAT4"],
    )
    parser.add_argument("--batch_size", default=64, type=int, help="minibatch size")
    parser.add_argument(
        "-a",
        "--architecture",
        default="IG2",
        type=str,
        help="network architecture",
        choices=["IG1", "IG2", "ESCAPE_NET"],
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        type=float,
        help="initial learning_rate",
    )
    parser.add_argument(
        "--pretrained_ann",
        default="",
        type=str,
        help="pretrained model to initialize ANN",
    )
    parser.add_argument(
        "--test_only", action="store_true", help="perform only inference"
    )
    parser.add_argument(
        "--epochs", default=150, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        help="optimizer for SNN backpropagation",
        choices=["SGD", "Adam"],
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-6,
        type=float,
        help="weight decay parameter for the optimizer",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum parameter for the SGD optimizer",
    )
    parser.add_argument(
        "--amsgrad",
        default=True,
        type=bool,
        help="amsgrad parameter for Adam optimizer",
    )
    parser.add_argument(
        "--dropout", default=0.2, type=float, help="dropout percentage for conv layers"
    )
    parser.add_argument(
        "--kernel_size", default=3, type=int, help="filter size for the conv layers"
    )
    parser.add_argument(
        "--devices", default="0", type=str, help="list of gpu device(s)"
    )
    parser.add_argument(
        "--dataset_path",
        default="",
        type=str,
        help="",
    )
    parser.add_argument("--save_dir", default="", tpye=str, help="")
    parser.add_argument("--save_name", default="", tpye=str, help="")
    # parser.add_argument('-f')
    args = parser.parse_args()
