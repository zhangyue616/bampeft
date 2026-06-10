from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model, build_pretrain_model
from chemprop.models.adapter import DEFAULT_ADAPTER_TARGET_LAYERS, freeze_adapter_train_only, wrap_adapter_layers
from chemprop.models.lora import DEFAULT_LORA_TARGET_LAYERS, freeze_base_params, freeze_encoder_train_head_only, \
    wrap_lora_layers
from chemprop.parsing import resolve_effective_method
from chemprop.features import FGFeatureExtractor, default_fg_config_path
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint
from chemprop.data import MoleculeDataset
from tqdm import tqdm, trange
from chemprop.models import ContrastiveLoss
from chemprop.torchlight import initialize_exp, snapshot
from torch.optim import Adam
import rdkit
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


FG_CONFIG_METHODS = {'M', 'A3', 'A4-zero', 'A4-trained', 'A5', 'A6', 'A7', 'A9'}


def _needs_fg_features(args: Namespace) -> bool:
    return (
        getattr(args, 'ablation_method', None) in FG_CONFIG_METHODS
        or bool(getattr(args, 'use_fg_dora', False))
        or bool(getattr(args, 'use_fg_concat', False))
    )


def _prepare_fg_extractor(args: Namespace, train_data: MoleculeDataset) -> None:
    if not _needs_fg_features(args):
        return
    descriptor_type = 'size' if getattr(args, 'ablation_method', None) == 'A9' else 'fg'
    config_path = getattr(args, 'fg_descriptor_config_path', None) or default_fg_config_path()
    extractor = FGFeatureExtractor(config_path, descriptor_type=descriptor_type)
    extractor.fit_normalization(train_data.smiles())
    if getattr(args, 'ablation_method', None) == 'A5':
        shuffle_seed = getattr(args, 'A5_shuffle_seed', None)
        if shuffle_seed is None:
            shuffle_seed = 42
            args.A5_shuffle_seed = shuffle_seed
        extractor.fit_shuffle(train_data.smiles(), shuffle_seed)
    if getattr(args, 'ablation_method', None) == 'A6':
        args.A6_lora_m_static = None
    args.fg_descriptor_config_path = config_path
    args.fg_feature_extractor = extractor
    args.fg_num_features = extractor.n_features
    args.fg_runtime_training = False


def _collect_a6_lora_m_static(args: Namespace, model: torch.nn.Module) -> dict:
    if getattr(args, 'ablation_method', None) != 'A6':
        return None
    encoder = model.encoder.encoder
    if not hasattr(encoder, 'has_a6_static_fold') or not encoder.has_a6_static_fold():
        return getattr(args, 'A6_lora_m_static', None)
    static = {}
    for layer_name in getattr(args, 'fg_gated_layers', []):
        value = encoder.get_a6_lora_m_static(layer_name)
        if value is None:
            continue
        static[layer_name] = value.detach().cpu().float().tolist()
    return static or None


def _build_fg_config(args: Namespace, model: torch.nn.Module) -> dict:
    if not _needs_fg_features(args):
        return None
    extractor = getattr(args, 'fg_feature_extractor', None)
    if extractor is None:
        raise ValueError('FG checkpoint metadata requested but args.fg_feature_extractor is missing')
    metadata = extractor.metadata()
    encoder = model.encoder.encoder
    tau_learned = {}
    if hasattr(encoder, 'fg_taus'):
        tau_learned = {
            layer: float(param.detach().cpu().item())
            for layer, param in encoder.fg_taus.items()
        }
    fg_config = {
        'use_fg_dora': bool(getattr(args, 'use_fg_dora', False)),
        'fg_descriptor_config_hash': metadata['fg_descriptor_config_hash'],
        'n_fg': metadata['n_fg'],
        'fg_normalization_mean': metadata['fg_normalization_mean'],
        'fg_normalization_std': metadata['fg_normalization_std'],
        'fg_gated_layers': list(getattr(args, 'fg_gated_layers', []) if getattr(args, 'use_fg_dora', False) else []),
        'gate_hidden_dim': getattr(args, 'fg_gate_hidden_dim', 64),
        'tau_init': getattr(args, 'fg_tau_init', 0.1),
        'tau_learned_per_layer': tau_learned,
        'ablation_method': getattr(args, 'ablation_method', None),
        'A5_shuffle_seed': getattr(args, 'A5_shuffle_seed', None) if getattr(args, 'ablation_method', None) == 'A5' else None,
        'A9_size_only_flag': bool(getattr(args, 'A9_size_only_flag', False)) if getattr(args, 'ablation_method', None) == 'A9' else False,
        'A10_rank': getattr(args, 'A10_rank', None) if getattr(args, 'ablation_method', None) in {'A10-lite', 'A10-full'} else None,
        'descriptor_type': metadata['descriptor_type'],
        'descriptor_config_path': metadata['descriptor_config_path'],
        'A6_lora_m_static': _collect_a6_lora_m_static(args, model),
    }
    return fg_config


def _fold_a6_lora_m_static(model: torch.nn.Module, args: Namespace, train_data: MoleculeDataset) -> None:
    if getattr(args, 'ablation_method', None) != 'A6':
        return
    extractor = getattr(args, 'fg_feature_extractor', None)
    if extractor is None:
        raise ValueError('A6 static fold requires args.fg_feature_extractor')
    encoder = model.encoder.encoder
    gated_layers = list(getattr(args, 'fg_gated_layers', []) or [])
    if not gated_layers:
        raise ValueError('A6 static fold requires non-empty fg_gated_layers')

    device = next(model.parameters()).device
    ratio_sums = {layer_name: None for layer_name in gated_layers}
    n_mols = 0

    model.eval()
    with torch.no_grad():
        for smiles in train_data.smiles():
            fg_row = extractor.compute_from_smiles(smiles)
            fg_features = torch.from_numpy(fg_row).float().unsqueeze(0).to(device)
            for layer_name in gated_layers:
                gate_out = encoder.fg_gates[layer_name](fg_features)
                tau = F.softplus(encoder.fg_taus[layer_name]) / 10.0
                ratio = torch.exp(tau * gate_out).squeeze(0)
                if ratio_sums[layer_name] is None:
                    ratio_sums[layer_name] = ratio.detach().clone()
                else:
                    ratio_sums[layer_name] += ratio.detach()
            n_mols += 1

        if n_mols == 0:
            raise ValueError('A6 static fold cannot use an empty training split')

        static = {}
        for layer_name in gated_layers:
            mean_ratio = ratio_sums[layer_name] / float(n_mols)
            layer = getattr(encoder, layer_name)
            if not getattr(layer, 'use_dora', False):
                raise ValueError(f'A6 static fold requires DoRA-wrapped layer {layer_name}')
            lora_m_static = layer.lora_m.detach() * mean_ratio
            encoder.set_a6_lora_m_static(layer_name, lora_m_static)
            static[layer_name] = lora_m_static.detach().cpu().float().tolist()

    args.A6_lora_m_static = static


def _apply_ablation_trainability(model: torch.nn.Module, args: Namespace) -> None:
    method = getattr(args, 'ablation_method', None)
    if method in {'A4-zero', 'A4-trained'}:
        for name, param in model.named_parameters():
            if 'fg_gates' in name or 'fg_taus' in name or name.startswith('ffn.') or '.ffn.' in name:
                param.requires_grad = True
            elif 'lora_' in name:
                param.requires_grad = False


def _log_gate_activity(model: torch.nn.Module, writer: SummaryWriter, epoch: int) -> None:
    if writer is None:
        return
    encoder = getattr(getattr(model, 'encoder', None), 'encoder', None)
    stats = getattr(encoder, '_last_gate_stats', {}) if encoder is not None else {}
    for layer, layer_stats in stats.items():
        for key, value in layer_stats.items():
            writer.add_scalar(f'gate/{layer}/{key}', value, epoch)


def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    info('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    info(f'Number of tasks = {args.num_tasks}')

    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0),
                                              seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0),
                                             seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     seed=args.seed, args=args, logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                for smiles in dataset.smiles():
                    writer.writerow([smiles])

            with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    writer.writerow(lines_by_smiles[smiles])

            split_indices = []
            for smiles in dataset.smiles():
                split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)

        with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    _prepare_fg_extractor(args, train_data)

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    test_smiles, test_targets = test_data.smiles(), test_data.targets()

    all_run_scores = []

    for run_idx in range(args.num_runs):
        info('\n' + '=' * 80)
        info(f'Starting Run {run_idx + 1}/{args.num_runs}')
        info('=' * 80 + '\n')

        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

        for model_idx in range(args.ensemble_size):
            info(f'Training Model {model_idx + 1}/{args.ensemble_size}')

            save_dir = os.path.join(args.save_dir, f'run_{run_idx}', f'model_{model_idx}')
            makedirs(save_dir)
            try:
                writer = SummaryWriter(log_dir=save_dir)
            except:
                writer = SummaryWriter(logdir=save_dir)

            if getattr(args, 'ablation_method', None) == 'A6':
                args.A6_lora_m_static = None

            if args.checkpoint_path is not None:
                debug(f'Loading model from {args.checkpoint_path}')
                model = build_model(args, encoder_name=args.encoder_name)
                state_dict = torch.load(args.checkpoint_path, map_location='cpu')
                missing_keys, unexpected_keys = model.encoder.load_state_dict(state_dict, strict=False)
                if unexpected_keys:
                    debug(f'Unexpected keys in checkpoint (ignored): {unexpected_keys[:5]}{"..." if len(unexpected_keys) > 5 else ""}')
                if missing_keys:
                    debug(f'Missing keys in checkpoint (using init): {missing_keys[:5]}{"..." if len(missing_keys) > 5 else ""}')
            else:
                debug(f'Building model {model_idx}')
                model = build_model(args, encoder_name=args.encoder_name)

            effective_method = resolve_effective_method(args)
            args.effective_method = effective_method
            lora_config = None
            adapter_config = None

            if effective_method in ('lora_head', 'dora_head'):
                use_dora = effective_method == 'dora_head'
                args.use_dora = use_dora
                if use_dora and not args.lora_target_modules:
                    raise ValueError("DoRA requires non-empty lora_target_modules")

                wrap_lora_layers(
                    model,
                    rank=args.lora_rank,
                    alpha=args.lora_alpha,
                    target_layer_names=args.lora_target_modules,
                    use_dora=use_dora
                )
                freeze_base_params(model)
                _apply_ablation_trainability(model, args)
                lora_config = {
                    'method': 'DoRA + Head' if use_dora else 'LoRA + Head',
                    'rank': args.lora_rank,
                    'alpha': args.lora_alpha,
                    'target_modules': list(args.lora_target_modules),
                    'ffn_policy': args.lora_ffn_policy,
                    'bias_policy': args.lora_bias_policy,
                    'use_dora': use_dora,
                    'ablation_method': getattr(args, 'ablation_method', None),
                }
            elif effective_method == 'adapter_head':
                args.use_dora = False
                adapter_targets = list(DEFAULT_ADAPTER_TARGET_LAYERS)
                wrap_adapter_layers(
                    model,
                    target_layer_names=adapter_targets,
                    bottleneck_dim=args.adapter_bottleneck_dim,
                    scale_init=args.adapter_scale_init,
                )
                freeze_adapter_train_only(model)
                adapter_config = {
                    'method': 'Adapter + Head',
                    'bottleneck_dim': args.adapter_bottleneck_dim,
                    'scale_init': args.adapter_scale_init,
                    'target_modules': adapter_targets,
                    'activation': 'ReLU',
                    'batch_norm': True,
                    'ffn_policy': 'trainable',
                    'bias_policy': 'adapter_bias_trainable_base_bias_frozen',
                }
            elif effective_method == 'head_only':
                freeze_encoder_train_head_only(model)
            elif effective_method == 'full_ft':
                pass
            else:
                raise ValueError(f'Unhandled effective_method: {effective_method}')

            debug(model)
            debug(f'Number of parameters = {param_count(model):,}')

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params

            info('PEFT Configuration:')
            if lora_config is not None:
                info(f'  Method: {lora_config["method"]}')
                info(f'  LoRA rank: {args.lora_rank}')
                info(f'  LoRA alpha: {args.lora_alpha}')
                info(f'  LoRA target_modules: {args.lora_target_modules}')
                info(f'  FFN policy: {args.lora_ffn_policy}')
                info(f'  bias policy: {args.lora_bias_policy} (all frozen)')
            elif adapter_config is not None:
                info(f'  Method: {adapter_config["method"]}')
                info(f'  Adapter bottleneck_dim: {args.adapter_bottleneck_dim}')
                info(f'  Adapter scale_init: {args.adapter_scale_init}')
                info(f'  Adapter target_modules: {adapter_config["target_modules"]}')
                info(f'  FFN policy: {adapter_config["ffn_policy"]}')
                info(f'  bias policy: {adapter_config["bias_policy"]}')
            else:
                info(f'  Method: {effective_method}')
                info('  LoRA/DoRA wrapping: disabled')
            info('  optimizer filter: requires_grad')
            info(f'  Total params: {total_params:,}')
            info(f'  Trainable params: {trainable_params:,}')
            info(f'  Frozen params: {frozen_params:,}')

            if args.cuda:
                debug('Moving model to cuda')
                model = model.cuda()

            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args,
                            lora_config=lora_config, adapter_config=adapter_config,
                            fg_config=_build_fg_config(args, model))

            optimizer = build_optimizer(model, args)
            scheduler = build_lr_scheduler(optimizer, args)

            if model_idx == 0 and run_idx == 0:
                info('\n' + '=' * 80)
                info('MODEL ARCHITECTURE SUMMARY')
                info('=' * 80)

                info(f'Total parameters: {total_params:,}')
                info(f'Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)')
                info(f'Frozen parameters: {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)')

                info('=' * 80 + '\n')

            best_score = float('inf') if args.minimize_score else -float('inf')
            best_epoch, n_iter = 0, 0

            patience = 0
            patience_limit = getattr(args, 'patience', 30)

            for epoch in range(args.epochs):
                info(f'Epoch {epoch + 1}/{args.epochs}')

                n_iter = train(
                    model=model,
                    data=train_data,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    logger=logger,
                    writer=writer
                )
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
                _log_gate_activity(model, writer, epoch)

                val_scores = evaluate(
                    model=model,
                    data=val_data,
                    num_tasks=args.num_tasks,
                    metric_func=metric_func,
                    batch_size=args.batch_size,
                    dataset_type=args.dataset_type,
                    scaler=scaler,
                    logger=logger
                )

                avg_val_score = np.nanmean(val_scores)
                writer.add_scalar(f'validation_{args.metric}', avg_val_score, n_iter)

                test_preds = predict(
                    model=model,
                    data=test_data,
                    batch_size=args.batch_size,
                    scaler=scaler
                )
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=args.num_tasks,
                    metric_func=metric_func,
                    dataset_type=args.dataset_type,
                    logger=logger
                )

                avg_test_score = np.nanmean(test_scores)

                if args.show_individual_scores:
                    for task_name, val_score in zip(args.task_names, val_scores):
                        debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args,
                                    lora_config=lora_config, adapter_config=adapter_config,
                                    fg_config=_build_fg_config(args, model))
                    patience = 0
                else:
                    patience += 1
                    debug(f'No improvement. Patience: {patience}/{patience_limit}')

                if patience >= patience_limit:
                    info(f'Early stopping at epoch {epoch}. Best epoch: {best_epoch}')
                    break

            # Evaluate on test set using model with best validation score
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

            if getattr(args, 'ablation_method', None) == 'A6':
                best_checkpoint_path = os.path.join(save_dir, 'model.pt')
                fold_model = load_checkpoint(best_checkpoint_path, cuda=args.cuda, logger=logger)
                _fold_a6_lora_m_static(fold_model, args, train_data)
                save_checkpoint(best_checkpoint_path, fold_model, scaler, features_scaler, args,
                                lora_config=lora_config, adapter_config=adapter_config,
                                fg_config=_build_fg_config(args, fold_model))

            original_log_level = None
            if logger:
                original_log_level = logger.level
                logger.setLevel(logging.INFO)

            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

            if logger and original_log_level:
                logger.setLevel(original_log_level)

            test_preds = predict(
                model=model,
                data=test_data,
                batch_size=args.batch_size,
                scaler=scaler
            )

            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)

            avg_test_score = np.nanmean(test_scores)
            writer.add_scalar(f'test_{args.metric}', avg_test_score, 0)

            if args.show_individual_scores:
                for task_name, test_score in zip(args.task_names, test_scores):
                    info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{args.metric}', test_score, n_iter)

        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        avg_ensemble_test_score = np.nanmean(ensemble_scores)
        info(f'Run {run_idx + 1} Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

        all_run_scores.append(avg_ensemble_test_score)

        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    if args.num_runs > 1:
        mean_score = np.mean(all_run_scores)
        std_score = np.std(all_run_scores, ddof=1)

        info('\n' + '=' * 80)
        info('FINAL RESULTS (Test Set Performance)')
        info('=' * 80)
        info(f'Individual run test scores: {[f"{s:.6f}" for s in all_run_scores]}')
        info(f'Mean ± Std: {mean_score:.6f} ± {std_score:.6f}')
        info('=' * 80 + '\n')

        return all_run_scores
    else:
        info('\n' + '=' * 80)
        info('FINAL RESULTS (Test Set Performance)')
        info('=' * 80)
        info(f'Final test {args.metric} = {all_run_scores[0]:.6f}')
        info('=' * 80 + '\n')
        return [all_run_scores[0]]


def pre_training(args: Namespace, logger: Logger = None) -> List[float]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    debug('Loading data')
    data = get_data(path=args.data_path, args=args, logger=logger)

    args.data_size = len(data)

    debug(f'Total size = {len(data)}')

    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model1 = build_pretrain_model(args, encoder_name='CMPNN')
            model2 = build_pretrain_model(args, encoder_name='CMPNN')

        debug(model1)
        debug(f'Number of M1 parameters = {param_count(model1):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model1 = model1.cuda()

        debug(model2)
        debug(f'Number of M2 parameters = {param_count(model2):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model2 = model2.cuda()

        logger, dump_folder = initialize_exp(Namespace(**args.__dict__))
        dump_folder = f'{dump_folder}-model'

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        args.device = device
        criterion = ContrastiveLoss(loss_computer='nce_softmax', temperature=args.temperature, args=args).cuda()
        optimizer = Adam([{"params": model1.parameters()}, {"params": model2.parameters()}], lr=3e-5)
        scheduler = ExponentialLR(optimizer, 0.99, -1)
        step_per_schedule = 500
        global_step = 0

        mol = MoleculeDataset(data)
        smiles, features = mol.smiles(), mol.features()

        loader = DataLoader(smiles,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=12,
                            drop_last=True)

        for epoch in range(args.epochs):
            model1.train()
            model2.train()

            for batch in tqdm(loader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
                emb1 = model1(batch, None)
                emb2 = model2(batch, None)

                loss = criterion(emb1, emb2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

                if global_step % 1000 == 0:
                    snapshot(model1.encoder, global_step, dump_folder, 'original')
                    snapshot(model2.encoder, global_step, dump_folder, 'augment')
                if global_step % step_per_schedule == 0:
                    scheduler.step()
            logger.info(f'[{epoch + 1}/{args.epochs}] train loss {loss.item():.4f}')
