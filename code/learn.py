import os
import json
import argparse
import numpy as np

import torch
from torch import optim

from datasets import Dataset
from models import *
from regularizers import *
from optimizers import KBCOptimizer

datasets = ['DB15K', 'MKG-W']

parser = argparse.ArgumentParser(
    description="Multi-modal Knowledge Graph Completion"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

parser.add_argument(
    '--model', type=str, default='ComCon'
)

parser.add_argument(
    '--regularizer', type=str, default='NA',
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

parser.add_argument('-train', '--do_train', action='store_true')
parser.add_argument('-test', '--do_test', action='store_true')
parser.add_argument('-save', '--do_save', action='store_true')
parser.add_argument('-weight', '--do_ce_weight', action='store_true')
parser.add_argument('-path', '--save_path', type=str, default='../logs/')
parser.add_argument('-id', '--model_id', type=str, default='0')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='')

args = parser.parse_args()

if args.do_save:
    assert args.save_path
    save_suffix = args.model + '_' + args.regularizer + '_' + args.dataset + '_' + args.model_id

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    save_path = os.path.join(args.save_path, save_suffix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

data_path = "../data"
dataset = Dataset(data_path, args.dataset)


examples = torch.from_numpy(dataset.get_train().astype('int64'))

if args.do_ce_weight:
    ce_weight = torch.Tensor(dataset.get_weight()).cuda()
else:
    ce_weight = None

print(dataset.get_shape())

model, model_exp, model_imp = None, None, None
regularizer = None
exec('model = '+args.model+'(dataset.get_shape(), args.rank, args.init, args.dataset)')
exec('model_exp = '+args.model+'(dataset.get_shape(), args.rank, args.init, args.dataset)')
exec('model_imp = '+args.model+'(dataset.get_shape(), args.rank, args.init, args.dataset)')
exec('regularizer = '+args.regularizer+'(args.reg)')
regularizer = [regularizer, N3(args.reg)]

device = 'cuda'
model.to(device)
model_exp.to(device)
model_imp.to(device)
for reg in regularizer:
    reg.to(device)

# Adagrad 自适应梯度算法
# Adam 自适应矩估计
# 随机梯度下降

# torch.optim.Adam(list(emb_model.parameters()) + list(conf.parameters()) + list(dis.parameters()), lr=cfg.lrd)


optim_method = {
    'Adagrad': lambda: optim.Adagrad(list(model.parameters()) + list(model_exp.parameters()) + list(model_imp.parameters()), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(list(model.parameters()) + list(model_exp.parameters()) + list(model_imp.parameters()), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(list(model.parameters()) + list(model_exp.parameters()) + list(model_imp.parameters()), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, model_exp, model_imp, regularizer, optim_method, args.batch_size)


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = mrrs['rhs']
    h = hits['rhs']
    return {'MRR': m, 'hits@[1,3,10]': h}


cur_loss = 0

if args.checkpoint != '':
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))

if args.do_train:
    with open(os.path.join(save_path, 'train.log'), 'w') as log_file:
        max_score = 0
        for e in range(args.max_epochs):
            print("Epoch: {}".format(e+1))

            cur_loss = optimizer.epoch(examples, e=e, weight=ce_weight)

            if (e + 1) % args.valid == 0:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 10000, task=args.dataset))
                    for split in ['valid', 'test', 'train']
                ]

                print("\t TRAIN: ", train)
                print("\t VALID: ", valid)
                print("\t  TEST: ", test)
                # 控制数据写入
                MRR = test["MRR"]
                hist1310 = test["hits@[1,3,10]"]
                hist1 = hist1310[0].item()
                hist3 = hist1310[1].item()
                hist10 = hist1310[2].item()
                current_score = MRR*3 + hist1 + hist3 + hist10
                if max_score < current_score:
                    log_file.write("Epoch: {}\n".format(e+1))
                    log_file.write("\t TRAIN: {}\n".format(train))
                    log_file.write("\t VALID: {}\n".format(valid))
                    log_file.write("\t TEST: {}\n".format(test))
                    log_file.flush()
                    max_score = current_score

        test = avg_both(*dataset.eval(model, 'test', 50000,task=args.dataset))
        print("\t TEST : ", test)

if args.do_save:
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint'))
    embeddings = model.embeddings
    len_emb = len(embeddings)
    if len_emb == 2:
        np.save(os.path.join(save_path, 'entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
    elif len_emb == 3:
        np.save(os.path.join(save_path, 'head_entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'tail_entity_embedding.npy'), embeddings[2].weight.detach().cpu().numpy())
    else:
        print('SAVE ERROR!')

