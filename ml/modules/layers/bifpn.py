""" PyTorch EfficientDet model
Based on official Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
Paper: https://arxiv.org/abs/1911.09070
Hacked together by Ross Wightman
"""
import itertools
from collections import OrderedDict
from types import SimpleNamespace
from typing import List

import torch
from torch import nn

from ml.modules.layers.conv_bn_relu import conv_bn_relu, convt_bn_relu


class BiFpn(nn.Sequential):
    def __init__(self, top_config):
        """
        :param in_channels: number of input channels, int
        :param out_channels: number of output channels, int
        :param reduction_ratio: ratio of reduction: e.g.: 2 to halve the resolution, 0.5 to double, it must be a power of 2, it is used to calculate the stride, double
        :param kernel: the kernel size, double
        :param padding: the number of padding
        :param norm: type of norm, can be: [None, batchnorm, groupnorm]
        :param activation: type of activation, can be: [None, relu, leaky_relu]
        :param init_w: type of weight initialization, can be: [normal, kaiming]
        :param use_erf_conv: whether to factorize the resizing convolution into erf type conv
        :param conv_type: convolution type after the resizing method, None to not use any second conv, (for a nonconvolutional resizing layer withdifferent input and output sizes there always be used a second conv), can be: [None, conv, separable_conv, inception_conv]
        """
        super(BiFpn, self).__init__()

        feature_numbers = top_config.feature_numbers
        cell_number = top_config.cell_number
        backbone_depth = top_config.backbone_depth

        num_levels = len(feature_numbers)
        max_level = backbone_depth - 1
        min_level = backbone_depth - num_levels
        assert min_level >= 0

        node_config = bifpn_config(min_level, max_level, feature_numbers)
        fpn_layer_config = {'kernel_size': 3, 'padding': 1, 'output_padding': 1}

        for rep in range(cell_number):
            fpn_layer = BiFpnLayer(node_configs=node_config, num_levels=num_levels, **fpn_layer_config)
            self.add_module('bifpn_layer_' + str(rep), fpn_layer)

        self.add_module('encoder', BiFpnEncoder(node_configs=node_config, num_levels=num_levels, **fpn_layer_config))
        fusion_node_config = [{'name': 0, 'reduction': 8, 'out_channels': 128},
                              {'name': 1, 'reduction': 4, 'out_channels': 64},
                              {'name': 2, 'reduction': 2, 'out_channels': 16},
                              {'name': 3, 'reduction': 1, 'out_channels': 16},
                              {'name': 4, 'reduction': 1, 'inputs_offsets': [3],
                               'in_channels': top_config.feature_type.params.top_feature_number,
                               'out_channels': top_config.feature_type.params.top_feature_number}]
        self.add_module('feature_fusion', FuseNodes(node_configs=fusion_node_config,
                                                    target_node_config=fusion_node_config[-1]))


class DownsampleFeatureMap(nn.Sequential):
    def __init__(self, downsample_type, **kwargs):
        super(DownsampleFeatureMap, self).__init__()
        # make the resizing
        # downsampling types
        if downsample_type == 'conv' or downsample_type == 'separable_conv' or downsample_type == 'inception_conv':
            # conv_type is the same as downsample type: one of the above
            kwargs['conv_type'] = downsample_type
            self.add_module('downsample', conv_bn_relu(**kwargs))
        elif downsample_type == 'pool':
            assert not kwargs['use_erf_conv'], 'erf convolution can only be used with convolutional downsampling.'
            self.add_module('downsample', nn.MaxPool2d(kwargs['stride'] + 1,
                                                       kwargs['stride'],
                                                       padding=kwargs['padding']))
            self.add_module('downsample_conv', conv_bn_relu(**kwargs))
        else:
            raise ValueError(f'Wrong downsample_type: {downsample_type}')


class UpSampleFeatureMap(nn.Sequential):
    def __init__(self, upsample_type, **kwargs):
        super(UpSampleFeatureMap, self).__init__()
        # make the resizing
        # upsampling types
        if upsample_type == 'convt':
            self.add_module('upsample', convt_bn_relu(**kwargs))
        elif upsample_type == 'nearest':
            self.add_module('upsample', nn.UpsamplingNearest2d(kwargs['stride']))
        elif upsample_type == 'bilinear':
            self.add_module('upsample', nn.UpsamplingBilinear2d(kwargs['striee']))
        elif upsample_type == 'bilinear_additive':
            # from https://arxiv.org/abs/1707.05847
            raise NotImplementedError()
        else:
            raise ValueError(f'Wrong upsample_type: {upsample_type}')


class ResampleFeatureMap(nn.Sequential):
    def __init__(self, downsample_type, upsample_type, **kwargs):
        """
        Resize the feature map to the reduction ratio.

        :param downsample_type: how to downsample, it can be: [conv, separable_conv, inception_conv, pool],
        :param upsample_type: how to upsample, it can be: [convt, nearest, bilinear, bilinear_additive]
        :param kwargs: arguments for the up or downsampling

        :param in_channels: number of input channels, int
        :param out_channels: number of output channels, int
        :param reduction_ratio: ratio of reduction: e.g.: 2 to halve the resolution, 0.5 to double, it must be a power of 2, it is used to calculate the stride, double
        :param kernel: the kernel size, double
        :param padding: the number of padding
        :param norm: type of norm, can be: [None, batchnorm, groupnorm]
        :param activation: type of activation, can be: [None, relu, leaky_relu]
        :param init_w: type of weight initialization, can be: [normal, kaiming]
        :param use_erf_conv: whether to factorize the resizing convolution into erf type conv
        :param conv_type: convolution type after the resizing method, None to not use any second conv, (for a nonconvolutional resizing layer withdifferent input and output sizes there always be used a second conv), can be: [None, conv, separable_conv, inception_conv]
        """
        super(ResampleFeatureMap, self).__init__()
        reduction_ratio = kwargs['reduction_ratio']
        assert reduction_ratio % 2 == 0 or 1 / reduction_ratio % 2 == 0 or reduction_ratio == 1, \
            f'The reduction ratio should be 1 or a power of 2. but is: {reduction_ratio}'

        if reduction_ratio > 1:
            # downsample
            kwargs['stride'] = int(reduction_ratio)
            self.add_module('downsample', DownsampleFeatureMap(downsample_type, **kwargs))
        elif reduction_ratio < 1:
            # upsample
            kwargs['stride'] = int(1 // reduction_ratio)
            self.add_module('upsample', UpSampleFeatureMap(upsample_type, **kwargs))
        elif kwargs['in_channels'] != kwargs['out_channels']:
            # change the channel number to the required one
            self.add_module('channel_change', conv_bn_relu(**kwargs))
        else:
            # do nothing
            pass


class FuseNodes(nn.Module):
    def __init__(self, node_configs, target_node_config, fusion_method='sum', weight_method='scalar',
                 downsample_type='conv', upsample_type='convt',
                 kernel_size=3, padding=1, output_padding=1, norm='batchnorm',
                 activation='relu', init_w='normal', use_erf_conv=False, conv_type='conv', **kwargs):
        super(FuseNodes, self).__init__()
        self.node_configs = node_configs
        self.target_node_config = target_node_config
        self.fusion_method = fusion_method
        self.weight_method = weight_method

        # make sure that the args are compatible: for concat fusion there can`t be an weight method (it`s redundant)
        weight_methods = ['scalar', 'predicted']
        assert not (self.fusion_method == 'concat' and self.weight_method in weight_methods), \
            'For concat fusion there can`t be an weight method (it`s redundant)'

        # resize or change the channel number of the inputs according to the target_node_config
        # to be compatible with the target node
        self.resample = nn.ModuleDict()
        for input_node_number in target_node_config['inputs_offsets']:
            input_node_config = node_configs[input_node_number]
            assert input_node_config['name'] == input_node_number, \
                f'Something went wrong, the name of the input node and its index  must be the same as,' \
                f' but is {input_node_config["name"]} and {input_node_number}'

            self.resample[str(input_node_number)] = \
                ResampleFeatureMap(upsample_type=upsample_type,
                                   downsample_type=downsample_type,
                                   in_channels=input_node_config['out_channels'],
                                   out_channels=target_node_config['out_channels'],
                                   reduction_ratio=target_node_config['reduction'] / input_node_config['reduction'],
                                   kernel_size=kernel_size, padding=padding, output_padding=output_padding, norm=norm,
                                   activation=activation, init_w=init_w, use_erf_conv=use_erf_conv, conv_type=conv_type)

        # make the parameter for the input node weighting
        if weight_method == 'scalar':
            # scalar learnable weight for every input nodes
            self.edge_weights = nn.Parameter(torch.ones(len(target_node_config['inputs_offsets'])), requires_grad=True)
        elif weight_method == 'predicted':
            # pixelwise weight method predicted by the input node
            raise NotImplementedError('This function is yet to be implemented, if it will be at all.')
        else:
            self.edge_weights = None

    def forward(self, x):
        dtype = x[0].dtype
        nodes = []
        for input_node_number in self.target_node_config['inputs_offsets']:
            input_node = x[input_node_number]
            input_node = self.resample[str(input_node_number)](input_node)
            nodes.append(input_node)

        if self.fusion_method == 'sum':
            if self.weight_method == 'scalar':
                edge_weights = nn.functional.relu(self.edge_weights.type(dtype))
                weights_sum = torch.sum(edge_weights)
                x = torch.stack([(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001)
                                 for i in range(len(nodes))], dim=-1)
            elif self.weight_method == 'predicted':
                # pixelwise weight method predicted by the input node
                raise NotImplementedError('This function is yet to be implemented, if it will be at all.')
            else:
                x = torch.stack(nodes, dim=-1)
            x = torch.sum(x, dim=-1)
        elif self.fusion_method == 'concat':
            x = torch.stack(nodes, dim=1)
        else:
            raise ValueError('Wrong fusion_method {}'.format(self.fusion_method))

        return x


class BiFpnNode(nn.Sequential):
    def __init__(self, node_configs, target_level_node, **kwargs):
        super(BiFpnNode, self).__init__()
        # fuse the nodes
        self.add_module('fuse', FuseNodes(node_configs, target_level_node, **kwargs))
        # conv_bn_relu after the fusion
        self.add_module('conv_bn_relu', conv_bn_relu(in_channels=target_level_node['in_channels'],
                                                     out_channels=target_level_node['out_channels'],
                                                     **kwargs))


class BiFpnLayer(nn.Module):
    def __init__(self, node_configs, num_levels, **kwargs):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        self.node_configs = node_configs

        self.nodes = nn.ModuleDict()
        for node in node_configs[num_levels:]:
            self.nodes['bifpn_node_' + str(node['name'])] = BiFpnNode(node_configs, node, **kwargs)

    def forward(self, features):
        for node_name, node in self.nodes.items():
            features.append(node(features))
        return features[-self.num_levels:]


class BiFpnEncoder(nn.Module):
    def __init__(self, node_configs, num_levels, **kwargs):
        super(BiFpnEncoder, self).__init__()
        self.num_levels = num_levels
        self.node_configs = node_configs

        self.nodes = nn.ModuleDict()
        for node in node_configs[num_levels:-num_levels + 1]:
            self.nodes['encoder_' + str(node['name'])] = BiFpnNode(node_configs, node, **kwargs)

    def forward(self, features):
        for node_name, node in self.nodes.items():
            features.append(node(features))

        return features[-self.num_levels:]


def bifpn_config(min_level=0, max_level=3, feature_numbers=[16, 32, 64, 128], fuse_method='sum'):
    """BiFPN config with sum.
    Adapted from https://github.com/google/automl/blob/56815c9986ffd4b508fe1d68508e268d129715c1/efficientdet/keras/fpn_configs.py
    """
    # p.nodes = [
    #     {'reduction': base_reduction << 3, 'inputs_offsets': [3, 4]},
    #     {'reduction': base_reduction << 2, 'inputs_offsets': [2, 5]},
    #     {'reduction': base_reduction << 1, 'inputs_offsets': [1, 6]},
    #     {'reduction': base_reduction, 'inputs_offsets': [0, 7]},
    #     {'reduction': base_reduction << 1, 'inputs_offsets': [1, 7, 8]},
    #     {'reduction': base_reduction << 2, 'inputs_offsets': [2, 6, 9]},
    #     {'reduction': base_reduction << 3, 'inputs_offsets': [3, 5, 10]},
    #     {'reduction': base_reduction << 4, 'inputs_offsets': [4, 11]},
    # ]
    # p.weight_method = 'sum'
    #
    assert max_level - min_level + 1 == len(feature_numbers), \
        f'The feature node number referred with max-min difference ' \
        f'and the number of features should be the same but is: ' \
        f'{max_level - min_level + 1} and {len(feature_numbers)}'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    def get_in_channel(nodes, input_offsets):
        """ Get the input channel according to the fuse method and the inputs"""
        if fuse_method == 'sum':
            return nodes[input_offsets[0]]['out_channels']
        elif fuse_method == 'concat':
            return nodes[input_offsets[0]]['out_channels'] * len(input_offsets)
        else:
            raise ValueError(f'Wrong fuse_method for bifpn: {fuse_method}')

    # define nodes as the input nodes, and later append the bifpn nodes
    nodes = [{'name': i, 'reduction': 2 ** (min_level + i), 'out_channels': feature_number}
             for i, feature_number in enumerate(feature_numbers)]
    decoder_nodes = nodes.copy()
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        nodes.append({
            'name': 2 * max_level - i,
            'reduction': 2 ** i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'in_channels': get_in_channel(nodes, [level_last_id(i), level_last_id(i + 1)]),
            'out_channels': nodes[level_last_id(i)]['out_channels']
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        nodes.append({
            'name': i + max_level * 2,
            'reduction': 2 ** i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)],
            'in_channels': get_in_channel(nodes, level_all_ids(i) + [level_last_id(i - 1)]),
            'out_channels': nodes[level_last_id(i)]['out_channels']
        })
        node_ids[i].append(next(id_cnt))

    return nodes


def decoder_config(min_level=0, max_level=3, feature_numbers=[16, 32, 64, 128], fuse_method='sum'):
    """BiFPN config with sum.
    Adapted from https://github.com/google/automl/blob/56815c9986ffd4b508fe1d68508e268d129715c1/efficientdet/keras/fpn_configs.py
    """
    # p.nodes = [
    #     {'reduction': base_reduction << 3, 'inputs_offsets': [3, 4]},
    #     {'reduction': base_reduction << 2, 'inputs_offsets': [2, 5]},
    #     {'reduction': base_reduction << 1, 'inputs_offsets': [1, 6]},
    #     {'reduction': base_reduction, 'inputs_offsets': [0, 7]},
    #     {'reduction': base_reduction << 1, 'inputs_offsets': [1, 7, 8]},
    #     {'reduction': base_reduction << 2, 'inputs_offsets': [2, 6, 9]},
    #     {'reduction': base_reduction << 3, 'inputs_offsets': [3, 5, 10]},
    #     {'reduction': base_reduction << 4, 'inputs_offsets': [4, 11]},
    # ]
    # p.weight_method = 'sum'
    #
    assert max_level - min_level + 1 == len(feature_numbers), \
        f'The feature node number referred with max-min difference ' \
        f'and the number of features should be the same but is: ' \
        f'{max_level - min_level + 1} and {len(feature_numbers)}'

    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    def get_in_channel(nodes, input_offsets):
        """ Get the input channel according to the fuse method and the inputs"""
        if fuse_method == 'sum':
            return nodes[input_offsets[0]]['out_channels']
        elif fuse_method == 'concat':
            return nodes[input_offsets[0]]['out_channels'] * len(input_offsets)
        else:
            raise ValueError(f'Wrong fuse_method for bifpn: {fuse_method}')

    # define nodes as the input nodes, and later append the bifpn nodes
    nodes = [{'name': i, 'reduction': 2 ** (min_level + i), 'out_channels': feature_number}
             for i, feature_number in enumerate(feature_numbers)]

    for i in range(max_level - 1, - 1, -1):
        # decoder
        nodes.append({
            'name': 2 * max_level - i,
            'reduction': 2 ** i,
            'inputs_offsets': [level_last_id(i), level_last_id(i + 1)],
            'in_channels': get_in_channel(nodes, [level_last_id(i), level_last_id(i + 1)]),
            'out_channels': nodes[level_last_id(i)]['out_channels']
        })
        node_ids[i].append(next(id_cnt))

    return nodes


class Decoder(nn.Module):
    def __init__(self, feature_numbers, backbone_depth):
        super(Decoder, self).__init__()
        num_levels = len(feature_numbers)
        max_level = backbone_depth - 1
        min_level = backbone_depth - num_levels
        assert min_level >= 0

        decoder_node_configs = decoder_config(min_level, max_level, feature_numbers)
        fpn_layer_config = {'kernel_size': 3, 'padding': 1, 'output_padding': 1}

        self.nodes = nn.ModuleDict()
        for node in decoder_node_configs[num_levels:]:
            self.nodes['decoder_node_' + str(node['name'])] = BiFpnNode(decoder_node_configs, node, **fpn_layer_config)

    def forward(self, features):
        for node_name, node in self.nodes.items():
            features.append(node(features))
        return features[-self.num_levels:]
