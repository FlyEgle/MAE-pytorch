#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

## https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/precise_bn.py

import itertools
import torch
import torch.nn as nn
import logging
from typing import Iterable, Any
from torch.distributed import ReduceOp, all_reduce

logger = logging.getLogger(__name__)

BN_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


# pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because its
#  type `no_grad` is not callable.
@torch.no_grad()
def update_bn_stats(
    args: Any, model: nn.Module, data_loader: Iterable[Any], num_iters: int = 200  # pyre-ignore
) -> None:
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    Args:
        model (nn.Module): the model whose bn stats will be recomputed.
            Note that:
            1. This function will not alter the training mode of the given model.
               Users are responsible for setting the layers that needs
               precise-BN to training mode, prior to calling this function.
            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations.  This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers, or backup the states.
        data_loader (iterator): an iterator. Produce data as inputs to the model.
        num_iters (int): number of iterations to compute the stats.
    """
    bn_layers = get_bn_modules(model)

    if len(bn_layers) == 0:
        return

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]
    if args.rank == 0:
        a = [round(i.running_mean.cpu().numpy().max(), 4) for i in bn_layers]
        logger.info('bn mean max, %s', max(a))
        logger.info(a)
        a = [round(i.running_var.cpu().numpy().max(), 4) for i in bn_layers]
        logger.info('bn var max, %s', max(a))
        logger.info(a)
    for bn in bn_layers:
        # pyre-fixme[16]: `Module` has no attribute `momentum`.
        # bn.running_mean = torch.ones_like(bn.running_mean)
        # bn.running_var = torch.zeros_like(bn.running_var)
        bn.momentum = 1.0

    # Note that PyTorch's running_var means "running average of
    # bessel-corrected batch variance". (PyTorch's BN normalizes by biased
    # variance, but updates EMA by unbiased (bessel-corrected) variance).
    # So we estimate population variance by "simple average of bessel-corrected
    # batch variance". This is the same as in the BatchNorm paper, Sec 3.1.
    # This estimator converges to population variance as long as batch size
    # is not too small, and total #samples for PreciseBN is large enough.
    # Its convergence may be affected by small batch size.

    # Alternatively, one can estimate population variance by the sample variance
    # of all batches combined. However, this needs a way to know the batch size
    # of each batch in this function (otherwise we only have access to the
    # bessel-corrected batch variance given by pytorch), which is an extra
    # requirement.
    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    ind = -1
    for ind, inputs in enumerate(itertools.islice(data_loader, num_iters)):
        with torch.no_grad():
            model(inputs)

        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
            running_var[i] += (bn.running_var - running_var[i]) / (ind + 1)
            if torch.sum(torch.isnan(bn.running_mean)) > 0 or torch.sum(torch.isnan(bn.running_var)) > 0:
                raise RuntimeError(
                    "update_bn_stats ERROR(args.rank {}): Got NaN val".format(args.rank))
            if torch.sum(torch.isinf(bn.running_mean)) > 0 or torch.sum(torch.isinf(bn.running_var)) > 0:
                raise RuntimeError(
                    "update_bn_stats ERROR(args.rank {}): Got INf val".format(args.rank))
            if torch.sum(~torch.isfinite(bn.running_mean)) > 0 or torch.sum(~torch.isfinite(bn.running_var)) > 0:
                raise RuntimeError(
                    "update_bn_stats ERROR(args.rank {}): Got INf val".format(args.rank))

    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind)
    )

    for i, bn in enumerate(bn_layers):
        if args.distributed:
            all_reduce(running_mean[i], op=ReduceOp.SUM)
            all_reduce(running_var[i], op=ReduceOp.SUM)
            running_mean[i] = running_mean[i] / args.gpu_nums
            running_var[i] = running_var[i] / args.gpu_nums

        # Sets the precise bn stats.
        # pyre-fixme[16]: `Module` has no attribute `running_mean`.
        bn.running_mean = running_mean[i]
        # pyre-fixme[16]: `Module` has no attribute `running_var`.
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]

    if args.rank == 0:
        a = [round(i.cpu().numpy().max(), 4) for i in running_mean]
        logger.info('bn mean max, %s (%s)', max(a), a)
        a = [round(i.cpu().numpy().max(), 4) for i in running_var]
        logger.info('bn var max, %s (%s)', max(a), a)


def get_bn_modules(model):
    """
    Find all BatchNorm (BN) modules that are in training mode. See
    fvcore.precise_bn.BN_MODULE_TYPES for a list of all modules that are
    included in this search.
    Args:
        model (nn.Module): a model possibly containing BN modules.
    Returns:
        list[nn.Module]: all BN modules in the model.
    """
    # Finds all the bn layers.
    bn_layers = [
        m
        for m in model.modules()
        if m.training and isinstance(m, BN_MODULE_TYPES)
    ]
    return bn_layers
