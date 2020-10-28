import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='dev') # Dev, as dev.jsonl for Hatefulmemes
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=32)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--epochs', type=int, default=38)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='/kaggle/working') # For kaggle; prev. default: /snap/test
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # After competition make this standard / remove
    parser.add_argument("--reg", action='store_const', default=False, const=True) # Applies Multi-sample dropout, SWA, Other regularization 
    parser.add_argument("--rcac", action='store_const', default=False, const=True) # Uses rcac loss instead of CE
    parser.add_argument("--acc", type=int, default=1, help='Amount of acc steps for bigger batch size - make sure to adjust LR')
    parser.add_argument("--tr", type=str, default="bert-base-uncased", help="Transformer Model")
    parser.add_argument("--swa", action='store_const', default=False, const=True)
    parser.add_argument("--exp", type=str, default="experiment", help="Name of experiment for csv's")
    parser.add_argument("--midsave", type=int, default=-1, help='Save a MID model after x steps')
    parser.add_argument("--textb", action='store_const', default=False, const=True) # Concatenate obj (& attr) preds to text
    parser.add_argument("--tsv", action='store_const', default=False, const=True)
    parser.add_argument("--extract", action='store_const', default=False, const=True) # Extract feats on the go
    parser.add_argument("--num_features", type=int, default=100, help='How many features we have per img (e.g. 100, 80)')
    parser.add_argument("--num_pos", type=int, default=4, help='How many position feats - 4 or 6')
    parser.add_argument("--pad", action='store_const', default=False, const=True)
    parser.add_argument("--topk", type=int, default=-1, help='For testing only load topk feats from tsv')
    parser.add_argument("--wd", type=float, default=0.0, help='Weight Decay')
    parser.add_argument("--clip", type=float, default=5.0, help='Clip Grad Norm')
    parser.add_argument("--case", action='store_const', default=False, const=True)
    parser.add_argument("--contrib", action='store_const', default=False, const=True)
    

    # Variable feature amounts not yet supported

    # Ensemble-related
    parser.add_argument("--enspath", type=str, default="/kaggle/working", help="Path to folder with all csvs")

    # Model Loading - Note: PATHS must be put in here! 
    parser.add_argument('--model', type=str, default='X', help='Type of Model: X = LXMERT, U = UNITER, V = VisualBERT')
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')               
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERT0', dest='load_lxmert0', type=str, default=None,
                        help='Loads a second lxmert model ontop (e.g. to swap some weights).')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskHM", dest='task_hm', action='store_const', default=False, const=True)
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=4)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()