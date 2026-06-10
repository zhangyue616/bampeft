import random
import sys
import logging
from chemprop.parsing import parse_train_args
from chemprop.train import run_training


def setup_logging(args):
    logger = logging.getLogger('CMPNN')
    logger.setLevel(logging.DEBUG if not args.quiet else logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if not args.quiet else logging.INFO)

    import os
    log_dir = f'./logs/{args.exp_name}/{args.exp_id}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'{log_dir}/training.log'

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s [CMPNN] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f'Logging to file: {log_file}')

    return logger


def print_experiment_info(args, logger):
    logger.info('=' * 80)
    logger.info('EXPERIMENT CONFIGURATION')
    logger.info('=' * 80)

    logger.info(f'Experiment Name: {args.exp_name}')
    logger.info(f'Experiment ID: {args.exp_id}')
    logger.info('Mode: CMPNN Baseline')

    logger.info('-' * 80)
    logger.info('Dataset Configuration:')
    logger.info(f'  Data Path: {args.data_path}')
    logger.info(f'  Dataset Type: {args.dataset_type}')
    logger.info(f'  Split Type: {args.split_type}')
    logger.info(f'  Split Sizes: {args.split_sizes}')
    logger.info(f'  Metric: {args.metric}')

    logger.info('-' * 80)
    logger.info('Training Configuration:')
    logger.info(f'  Epochs: {args.epochs}')
    logger.info(f'  Batch Size: {args.batch_size}')
    logger.info(f'  Num Runs: {args.num_runs}')
    logger.info(f'  Seed: {args.seed}')
    logger.info(f'  GPU: {args.gpu if args.gpu is not None else "CPU"}')

    logger.info('-' * 80)
    logger.info('Model Configuration:')
    logger.info(f'  Encoder: {args.encoder_name}')
    logger.info(f'  Hidden Size: {args.hidden_size}')
    logger.info(f'  Depth: {args.depth}')
    logger.info(f'  Dropout: {args.dropout}')
    logger.info(f'  FFN Hidden Size: {args.ffn_hidden_size}')
    logger.info(f'  FFN Num Layers: {args.ffn_num_layers}')

    logger.info('-' * 80)
    logger.info('Learning Rate Configuration:')
    logger.info(f'  Initial LR: {args.init_lr}')
    logger.info(f'  Max LR: {args.max_lr}')
    logger.info(f'  Final LR: {args.final_lr}')
    logger.info(f'  Warmup Epochs: {args.warmup_epochs}')

    if args.checkpoint_path:
        logger.info('-' * 80)
        logger.info('Checkpoint Configuration:')
        logger.info(f'  Checkpoint Path: {args.checkpoint_path}')

    logger.info('=' * 80)
    logger.info('')

def main():
    try:
        args = parse_train_args()

        if args.seed == -1:
            args.seed = random.randint(1, 10000)

        logger = setup_logging(args)

        if not args.quiet:
            print_experiment_info(args, logger)

        logger.info('Starting training...')
        logger.info('')

        run_training(args, logger)

        logger.info('')
        logger.info('=' * 80)
        logger.info('Training completed successfully!')
        logger.info('=' * 80)

    except KeyboardInterrupt:
        print('\n')
        print('=' * 80)
        print('Training interrupted by user')
        print('=' * 80)
        sys.exit(0)

    except Exception as e:
        print('\n')
        print('=' * 80)
        print('Training failed with error:')
        print(f'   {type(e).__name__}: {e}')
        print('=' * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
