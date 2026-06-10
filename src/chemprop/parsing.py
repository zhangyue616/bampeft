from argparse import ArgumentError, ArgumentParser, Namespace
import json
import math
import os
import torch
from chemprop.utils import makedirs
from chemprop.features import get_available_features_generators


def add_predict_args(parser: ArgumentParser):
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--test_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made',
                        default='../input/test.csv')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--preds_path', type=str,
                        help='Path to CSV file where predictions will be saved',
                        default='test_pred')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints (walks directory and ensembles all models that are found)',
                        default=None)
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')


def add_train_args(parser: ArgumentParser):
    add_predict_args(parser)

    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file',
                        default='data.csv')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression', 'multiclass'],
                        help='Type of dataset',
                        default='regression')
    parser.add_argument('--multiclass_num_classes', type=int, default=3,
                        help='Number of classes when using multiclass dataset')
    parser.add_argument('--save_dir', type=str, default='./ckpt',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--separate_val_path', type=str, default=None,
                        help='Path to separate val set')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*', default=None,
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str, default=None,
                        help='Path to separate test set')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*', default=None,
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined',
                                 'cluster_balanced'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs when training a model')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of folds to use for splitting')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for crossval splitting')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for crossval splitting')
    parser.add_argument('--crossval_index_dir', type=str,
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str,
                        help='Indices of files to use as train/val/test. Overrides --crossval_index_dir')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'],
                        help='Metric to use during evaluation')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual runs, not just the average')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--config_path', type=str,
                        help='Path to a .json file containing arguments')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature')

    parser.add_argument('--encoder_name', type=str, default='CMPNN', choices=['CMPNN', 'MPNN'],
                        help='Name of the encoder to use')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU', 'GELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test split')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--dump_path', default='dumped', type=str,
                        help='Dump path')
    parser.add_argument('--exp_name', default='test', type=str,
                        help='Experiment name')
    parser.add_argument('--exp_id', default='1', type=str,
                        help='Experiment ID')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='Rank used by LoRA/DoRA adapters')
    parser.add_argument('--lora_alpha', type=float, default=16,
                        help='Alpha scaling value used by LoRA/DoRA adapters')
    parser.add_argument('--lora_target_modules', type=str, nargs='*',
                        default=['W_i_atom', 'W_i_bond', 'W_h_0', 'W_h_1', 'W_o', 'lr'],
                        help='CMPNEncoder linear layer names wrapped by LoRA/DoRA adapters')
    parser.add_argument('--lora_ffn_policy', type=str, default='trainable', choices=['trainable'],
                        help='FFN training policy used with LoRA/DoRA adapters')
    parser.add_argument('--lora_bias_policy', type=str, default='none', choices=['none'],
                        help='Bias training policy used with LoRA/DoRA adapters')
    parser.add_argument('--use-dora', '--use_dora', dest='use_dora', action='store_true', default=False,
                        help='Use DoRA adapters instead of LoRA adapters')
    parser.add_argument('--adapter_bottleneck_dim', type=int, default=8,
                        help='Bottleneck dimension used by AdapterGNN-lite adapters')
    parser.add_argument('--adapter_scale_init', type=float, default=0.01,
                        help='Initial learnable residual scale used by AdapterGNN-lite adapters')
    parser.add_argument('--peft_method', type=str, default='auto',
                        choices=['auto', 'none', 'lora', 'dora', 'adapter'],
                        help='PEFT method selector. auto preserves legacy LoRA/DoRA behavior; '
                             'none skips adapter wrapping (M1/M2 vanilla path); '
                             'lora forces LoRA + Head; dora forces DoRA + Head; '
                             'adapter forces AdapterGNN-lite + Head.')
    parser.add_argument('--head_only', action='store_true', default=False,
                        help='Freeze CMPNN encoder and train only FFN/head. '
                             'Requires --peft_method none.')
    parser.add_argument('--ablation_method', type=str, default=None,
                        choices=['M', 'A1', 'A2', 'A3', 'A4-zero', 'A4-trained',
                                 'A5', 'A6', 'A7', 'A9', 'A10-lite', 'A10-full'],
                        help='Path A FG-DoRA ablation method selector')
    parser.add_argument('--use_fg_dora', action='store_true', default=False,
                        help='Enable FG-conditioned adapter gates')
    parser.add_argument('--fg_descriptor_config_path', type=str, default=None,
                        help='Path to the pre-registered FG descriptor JSON config')
    parser.add_argument('--fg_gated_layers', type=str, nargs='*', default=['W_i_atom', 'W_o'],
                        help='Atom-row CMPNEncoder layers gated by FG-DoRA')
    parser.add_argument('--fg_gate_hidden_dim', type=int, default=64,
                        help='Hidden dimension for FG magnitude gate MLPs')
    parser.add_argument('--fg_tau_init', type=float, default=0.1,
                        help='Initial per-layer FG gate tau scalar')
    parser.add_argument('--A5_shuffle_seed', type=int, default=None,
                        help='FG descriptor permutation seed for A5')
    parser.add_argument('--A9_size_only_flag', action='store_true', default=False,
                        help='Use size-only descriptor input for A9')
    parser.add_argument('--A10_rank', type=int, default=20,
                        help='LoRA rank for A10 parameter-budget matched control. '
                             'rank=20 gives +1.12% vs M, nearest integer-rank match '
                             'with conservative bias toward higher-rank LoRA.')


def update_checkpoint_args(args: Namespace):
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_path is not None:
        args.checkpoint_paths = [args.checkpoint_path]
        return

    if args.checkpoint_dir is not None:
        args.checkpoint_paths = []
        for root, _, files in os.walk(args.checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    args.checkpoint_paths.append(os.path.join(root, fname))
        if len(args.checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')
        args.ensemble_size = len(args.checkpoint_paths)
        return

    args.checkpoint_paths = None


def modify_predict_args(args: Namespace):
    assert args.test_path
    assert args.preds_path
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None
    update_checkpoint_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda
    makedirs(args.preds_path, isfile=True)


def parse_predict_args() -> Namespace:
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    modify_predict_args(args)
    return args


def configure_path_a_ablation_args(args: Namespace):
    method = getattr(args, 'ablation_method', None)
    if getattr(args, 'fg_descriptor_config_path', None) is None:
        from chemprop.features import default_fg_config_path
        args.fg_descriptor_config_path = default_fg_config_path()

    args.use_fg_concat = method == 'A3'
    args.fg_runtime_training = False

    if method is None:
        args.fg_num_features = 57
        return

    if method == 'A1':
        args.use_dora = False
        args.use_fg_dora = False
    elif method == 'A2':
        args.use_dora = True
        args.use_fg_dora = False
    elif method in {'M', 'A4-zero', 'A4-trained', 'A5', 'A6', 'A9'}:
        args.use_dora = True
        args.use_fg_dora = True
    elif method == 'A7':
        args.use_dora = False
        args.use_fg_dora = True
    elif method == 'A3':
        args.use_dora = False
        args.use_fg_dora = False
    elif method in {'A10-lite', 'A10-full'}:
        args.use_dora = False
        args.use_fg_dora = False
        if args.A10_rank is None:
            args.A10_rank = 20
        args.lora_rank = args.A10_rank

    if method == 'A5' and args.A5_shuffle_seed is None:
        args.A5_shuffle_seed = 42
    if method == 'A9':
        args.A9_size_only_flag = True

    args.fg_num_features = 3 if method == 'A9' else 57


def resolve_effective_method(args: Namespace) -> str:
    """Returns one of lora_head, dora_head, adapter_head, head_only, or full_ft."""
    peft_method = getattr(args, 'peft_method', 'auto')
    head_only = bool(getattr(args, 'head_only', False))
    if peft_method == 'auto':
        return 'dora_head' if args.use_dora else 'lora_head'
    if peft_method == 'none':
        return 'head_only' if head_only else 'full_ft'
    if peft_method == 'lora':
        return 'lora_head'
    if peft_method == 'dora':
        return 'dora_head'
    if peft_method == 'adapter':
        return 'adapter_head'
    raise ValueError(f'Unknown peft_method: {peft_method}')


def validate_peft_args(args: Namespace) -> None:
    peft_method = getattr(args, 'peft_method', 'auto')
    head_only = bool(getattr(args, 'head_only', False))
    if head_only and peft_method != 'none':
        raise ArgumentError(None, '--head_only requires --peft_method none')

    incompatible_ablation_methods = {'A1', 'A2', 'A3', 'A4', 'A4-zero', 'A4-trained'}
    if peft_method == 'none' and args.ablation_method in incompatible_ablation_methods:
        raise ArgumentError(
            None,
            f'--peft_method none is incompatible with --ablation_method {args.ablation_method}'
        )

    if peft_method == 'lora':
        args.use_dora = False
    elif peft_method == 'dora':
        args.use_dora = True
    elif peft_method == 'adapter':
        if getattr(args, 'use_dora', False):
            raise ArgumentError(None, '--peft_method adapter is incompatible with --use_dora')
        args.use_dora = False
        if getattr(args, 'adapter_bottleneck_dim', 0) <= 0:
            raise ArgumentError(None, '--adapter_bottleneck_dim must be positive')
        if not math.isfinite(float(getattr(args, 'adapter_scale_init', 0.01))):
            raise ArgumentError(None, '--adapter_scale_init must be finite')

    args.effective_method = resolve_effective_method(args)


def modify_train_args(args: Namespace):
    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    validate_peft_args(args)

    if hasattr(args, 'exp_name') and hasattr(args, 'exp_id') and args.exp_name and args.exp_id:
        if args.save_dir == './ckpt':
            args.save_dir = os.path.join(args.dump_path, args.exp_name, args.exp_id)

    makedirs(args.save_dir)

    assert args.data_path is not None
    assert args.dataset_type is not None

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'multiclass':
            args.metric = 'cross_entropy'
        else:
            args.metric = 'rmse'

    args.minimize_score = args.metric in ['rmse', 'mae', 'mse', 'cross_entropy']

    update_checkpoint_args(args)

    if args.features_only:
        assert args.features_generator or args.features_path

    args.use_input_features = args.features_generator or args.features_path

    if args.features_generator is not None and 'rdkit_2d_normalized' in args.features_generator:
        assert not args.features_scaling

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size

    configure_path_a_ablation_args(args)
    validate_peft_args(args)


def parse_train_args() -> Namespace:
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)
    return args
