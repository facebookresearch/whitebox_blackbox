from __future__ import division
import argparse
import json
import torch
import torch.nn as nn

from src.datasets.cifar import CIFAR10
from src.evaluator import Evaluator
from src.metrics import computeMetrics
from src.model import build_model
from src.net import extractFeatures
from src.trainer import Trainer
from src.utils import bool_flag, initialize_exp, init_distributed_mode
from src.utils import getTransform


def mast_topline(model, train_data_loader, valid_data_loader):
    model = model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    train_logits, train_lbl = extractFeatures(train_data_loader, model)
    valid_logits, valid_lbl = extractFeatures(valid_data_loader, model)

    train_losses = criterion(train_logits, train_lbl)
    valid_losses = criterion(valid_logits, valid_lbl)

    map_train, map_test, acc, precision_train, recall_train = computeMetrics(- train_losses, - valid_losses)

    return acc, precision_train, recall_train


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool_flag, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="bypass")
    parser.add_argument("--save_periodic", type=int, default=0)
    parser.add_argument("--exp_id", type=str, default="")
    parser.add_argument("--debug_train", type=bool_flag, default=False)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)


    # Data
    parser.add_argument("--name", type=str, required=True)

    # Learning algorithm
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.001,momentum=0.9,weight_decay=0.0001")
    parser.add_argument("--validation_metrics", type=str, default="")

    # Model
    parser.add_argument("--architecture", type=str)
    parser.add_argument("--non_linearity", type=str, choices=["relu", "tanh"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--num_fc", type=int, default=128)
    parser.add_argument("--maxpool_size", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=5)

        # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)


    return parser


def main(params):
    init_distributed_mode(params)

    logger = initialize_exp(params)

    torch.cuda.manual_seed_all(params.seed)
    transform = getTransform(0)

    root_data = '/private/home/asablayrolles/data/cifar-dejalight2'
    trainset = CIFAR10(root=root_data, name=params.name, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, num_workers=2)

    valid_set = CIFAR10(root=root_data, name='public_0', transform=transform)
    valid_data_loader = torch.utils.data.DataLoader(valid_set, batch_size=params.batch_size, shuffle=False, num_workers=2)

    model = build_model(params)
    if params.gpu:
        model = model.cuda()

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum)

    trainer = Trainer(model=model, params=params)
    evaluator = Evaluator(trainer, params)

    for epoch in range(params.epochs):
        trainer.update_learning_rate()
        for images, targets in trainloader:
            trainer.classif_step(images, targets)

        # evaluate classification accuracy
        scores = evaluator.run_all_evals(trainer, evals=['classif'], data_loader=valid_data_loader)

        for name, val in trainer.get_scores().items():
            scores[name] = val

        accuracy, precision_train, recall_train = mast_topline(model, trainloader, valid_data_loader)
        print(f"Guessing accuracy: {accuracy}")

        scores["mast_accuracy"] = accuracy
        scores["mast_precision_train"] = precision_train
        scores["mast_recall_train"] = recall_train
        
        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))

        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)

    print('Finished Training')


if __name__ == "__main__":
    parser = get_parser()

    params = parser.parse_args()
    print(params)

    main(params)
