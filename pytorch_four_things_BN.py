"""This file gives the code for a normalization layer which includes Batch
Normalization, Ghost Batch Normalization, and Group Normalization as special
cases. It also supports custom weight decay on the scale and shift variables
(beta and gamma) and using a weighted average of example and moving average
statistics during inference.
Example usage:
# Batch Normalization (batch size = 128)
x = normalization_layer(x, channels_per_group=1, examples_per_group=128)
# Ghost Batch Normalization (ghost batch size = 16)
x = normalization_layer(x, channels_per_group=1, examples_per_group=16)
# Group Normalization
x = normalization_layer(x, channel_groups=32, examples_per_group=1)
# Batch/Group Normalization Generalization
x = normalization_layer(x, channel_groups=32, examples_per_group=2)


Key activity : TensorFlow --> Pytorch

Pytorch code, migrated from tensorflow code by Cecilia Summers,
https://github.com/ceciliaresearch/four_things_batch_norm/blob/master/normalization.py

Migrated by
Seungjae Han, KAIST EE.

History
2020. 5. 8)
    - First migration, some methods and functions are newly created, changed by me.
    - Implemented only using part, others will raise NotImplementedError.
    - I'll keep updating the code.

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def normalization_layer(x, is_training=True, channels_per_group=0,
    channel_groups=0, weight_decay=0., moving_average_decay=0.99,
    examples_per_group=0, eps=1e-5, example_eval_weight=0.):  # scope='norm', 

    # Assumes this is for a convolutional layer in channels_first format.
    num_examples, channels, height, width = list(x.size())
    channel_groups, channels_per_group = get_num_channel_groups(
        channels, channels_per_group, channel_groups)
    beta_regularizer, gamma_regularizer = get_bn_regularizers()
    
    # beta, gamma Regularizer needed
    beta = Variable(torch.zeros([1, channels, 1, 1]).type(dtype), requires_grad=True)
    gamma = Variable(torch.ones([1, channels, 1, 1]).type(dtype), requires_grad=True)

    moving_x = Variable(torch.zeros([1, channels, 1, 1]).type(dtype), requires_grad=True)
    moving_x2 = Variable(torch.ones([1, channels, 1, 1]).type(dtype), requires_grad=True)

    # Comput normalization statistics with sufficient_statistics for
    # flexibility and efficiency.
    counts, channel_x, channel_x2, _ = sufficient_statistics(
        x, [2, 3], keep_dims=True)
    
    channel_x /= counts  # Average
    channel_x2 /= counts  # Avarage of square

    if is_training:
        # Add updates:
        ''' At tensorflow code, not used in pytorch.
        x_update = moving_average_decay * moving_x + (1. - moving_average_decay) * \
                   torch.sum(channel_x, axis=[0], keepdims=True)
        x2_update = moving_average_decay * moving_x2 + (1. - moving_average_decay) * \
                    torch.sum(channel_x2, axis=[0], keepdims=True)
        '''

        # Group by example group and channel group.
        examples_per_group = min(examples_per_group, num_examples)
        # Assume that num_examples is always divisible by examples_per_group.
        example_groups = num_examples // examples_per_group
        channel_x = torch.reshape(channel_x,
                    [example_groups, examples_per_group,
                     channel_groups, channels_per_group, 1, 1])
        channel_x2 = torch.reshape(channel_x2,
                    [example_groups, examples_per_group,
                     channel_groups, channels_per_group, 1, 1])

        group_mean = torch.mean(channel_x, axis=[1, 3], keepdims=True)  # SJ
        group_x2 = torch.mean(channel_x2, axis=[1, 3], keepdims=True)
        group_var = group_x2 - group_mean.pow(2)

        nc_mean = torch.reshape(
            group_mean.repeat([1, examples_per_group, 1, channels_per_group, 1, 1]),
            [-1, channels, 1, 1])
        nc_var = torch.reshape(
            group_var.repeat([1, examples_per_group, 1, channels_per_group, 1, 1]),
            [-1, channels, 1, 1])

        mult = gamma * torch.rsqrt(nc_var + eps)

        add = -nc_mean * mult + beta
        x = x * mult + add
    
    else:
        # is_training == False
        channel_x = torch.reshape(channel_x,
                    [num_examples, channel_groups, channels_per_group, 1, 1])
        channel_x2 = torch.reshape(channel_x2,
                     [num_examples, channel_groups, channels_per_group, 1, 1])
        group_x = torch.sum(channel_x, axis=[2], keepdims=True)
        group_x2 = torch.sum(channel_x2, axis=[2], keepdims=True)
        moving_x_group = torch.sum(torch.reshape(moving_x,
                         [1, channel_groups, channels_per_group, 1, 1]), axis=[2], keepdims=True)
        moving_x2_group = torch.sum(torch.reshape(moving_x2,
                         [1, channel_groups, channels_per_group, 1, 1]), axis=[2], keepdims=True)
        
        norm_x = (1. - example_eval_weight) * moving_x_group + \
                 (example_eval_weight * group_x)
        norm_x2 = (1. -example_eval_weight) * moving_x2_group + \
                  (example_eval_weight * group_x2)
        norm_var = norm_x2 - norm_x.pow(2)

        norm_x = torch.reshape(
                 norm_x.repeat([1, 1, channels_per_group, 1, 1]), [num_examples, channels, 1, 1])
        norm_var = torch.reshape(
                   norm_var.repeat([1, 1, channels_per_group, 1, 1]), [num_examples, channels, 1, 1])
        
        mult = gamma * torch.rsqrt(norm_var + eps)
        add = -norm_x * mult + beta
        x = x * mult + add
    return x
                         
def get_num_channel_groups(channels, channels_per_group=0, channel_groups=0):
    if channels_per_group > 0:
        channels_per_group = min(channels_per_group, channels)
    elif channel_groups > 0:
        channels_per_group = max(channels // channel_groups, 1)
    else:
        raise ValueError('Either channels_per_group or channel_groups must be '
                         'provided.')
    channel_groups = channels // channels_per_group
    return (channel_groups, channels_per_group)

def get_bn_regularizers(weight_decay=0.):
    l2_crit = nn.MSELoss()
    if weight_decay > 0:
        gamma_reg_baseline = 1.
        beta_regularizer = lambda tensor: weight_decay * l2_crit(tensor)
        gamma_regularizer = lambda tensor: weight_decay * l2_crit(
            tensor - gamma_reg_baseline)
        raise NotImplementedError("didn't think about reduction of nn.MSELoss")
    else:
        beta_regularizer = None
        gamma_regularizer = None
    return beta_regularizer, gamma_regularizer

def sufficient_statistics(x, axes, shift=None, keep_dims=False):
    """Calculate the sufficient statistics for the mean and variance of 'x'.

    These sufficient statistics are computed using the one pass algorithm on
    an input that's optionally shifted. See:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data

    Args:
    x: A `Tensor`.
    axes: Array of ints. Axes along which to compute mean and variance.
    shift: A `Tensor` containing the value by which to shift the data for
        numerical stability, or `None` if no shift is to be performed. A shift
        close to the true mean provides the most numerically stable results.
    keepdims: produce statistics with the same dimensionality as the input.
    name: Name used to scope the operations that compute the sufficient stats.
    Returns:
    Four `Tensor` objects of the same type as `x`:
    * the count (number of elements to average over).
    * the (possibly shifted) sum of the elements in the array.
    * the (possibly shifted) sum of squares of the elements in the array.
    * the shift by which the mean must be corrected or None if `shift` is None.
    """
    x_shape = list(x.size())
    if len(x_shape) is not None and all(
        x_shape[d] is not None for d in axes):
        counts = 1
        for d in axes:
            counts *= x_shape[d]
        counts = torch.tensor(counts, dtype=x.dtype)
    else:  # shape needs to be inferred at runtime.
        raise NotImplementedError("Sorry")
    
    if shift is not None:
        raise NotImplementedError("Sorry")
    else:  # no shift
        m_ss = x
        v_ss = x.pow(2)
    m_ss = torch.sum(m_ss, dim=axes, keepdims=keep_dims)
    v_ss = torch.sum(v_ss, dim=axes, keepdims=keep_dims)

    return counts, m_ss, v_ss, shift        
