from collections import OrderedDict
from encoder_decoder import CGRU_cell, CLSTM_cell
from models import TrajGRU
import torch.nn as nn


# for encoder/decoder
# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [128, 128, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [128, 128, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(128,128), input_channels=16, filter_size=5, num_features=128),
        CLSTM_cell(shape=(64,64), input_channels=128, filter_size=5, num_features=128),
        CLSTM_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=128)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [128, 128, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [128, 128, 4, 2, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [128, 256, 4, 2, 1], # 256*256
            'conv4_leaky_1': [256, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=128),
        CLSTM_cell(shape=(64,64), input_channels=128, filter_size=5, num_features=128),
        CLSTM_cell(shape=(128,128), input_channels=128, filter_size=5, num_features=128),
    ]
]

convlstm_encoder_params_1 = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(128,128), input_channels=16, filter_size=5, num_features=64),
        CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=64)
    ]
]

convlstm_decoder_params_1 = [
    [
        OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 256, 4, 2, 1], # 256*256
            'conv4_leaky_1': [256, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(64,64), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(128,128), input_channels=64, filter_size=5, num_features=64),
    ]
]

#########################################################################################################

convlstm_encoder_params_2 = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 2, 3]}),
        OrderedDict({'conv2_leaky_1': [32, 96, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(128,128), input_channels=8, filter_size=3, num_features=32), # H*W = 128*128
        CLSTM_cell(shape=(64,64), input_channels=96, filter_size=3, num_features=96),
        CLSTM_cell(shape=(32,32), input_channels=96, filter_size=3, num_features=96)
    ]
]

convlstm_decoder_params_2 = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 32, 4, 2, 1]}),
        OrderedDict({
            'deconv3_pad_leaky_1': [32, 4, 3, 2, 1], # 256*256
            'conv3_leaky_1': [4, 4, 3, 1, 1],
            'conv4': [4, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(32,32), input_channels=96, filter_size=3, num_features=96),
        CLSTM_cell(shape=(64,64), input_channels=96, filter_size=3, num_features=96),
        CLSTM_cell(shape=(128,128), input_channels=32, filter_size=3, num_features=32),
    ]
]

#########################################################################################################

convgru_encoder_params_1 = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [128, 128, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [128, 128, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(128,128), input_channels=16, filter_size=5, num_features=128),
        CGRU_cell(shape=(64,64), input_channels=128, filter_size=5, num_features=128),
        CGRU_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=128)
    ]
]

convgru_decoder_params_1 = [
    [
        OrderedDict({'deconv1_leaky_1': [128, 128, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [128, 128, 4, 2, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [128, 256, 4, 2, 1], # 256*256
            'conv4_leaky_1': [256, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(32,32), input_channels=128, filter_size=5, num_features=128),
        CGRU_cell(shape=(64,64), input_channels=128, filter_size=5, num_features=128),
        CGRU_cell(shape=(128,128), input_channels=128, filter_size=5, num_features=128),
    ]
]

#########################################################################################################

convgru_encoder_params_2 = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 2, 3]}),
        OrderedDict({'conv2_leaky_1': [32, 96, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(128,128), input_channels=8, filter_size=3, num_features=32), # H*W = 128*128
        CGRU_cell(shape=(64,64), input_channels=96, filter_size=3, num_features=96),
        CGRU_cell(shape=(32,32), input_channels=96, filter_size=3, num_features=96)
    ]
]

convgru_decoder_params_2 = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 32, 4, 2, 1]}),
        OrderedDict({
            'deconv3_pad_leaky_1': [32, 4, 3, 2, 1], # 256*256
            'conv3_leaky_1': [4, 4, 3, 1, 1],
            'conv4': [4, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(32,32), input_channels=96, filter_size=3, num_features=96),
        CGRU_cell(shape=(64,64), input_channels=96, filter_size=3, num_features=96),
        CGRU_cell(shape=(128,128), input_channels=32, filter_size=3, num_features=32),
    ]
]

#########################################################################################################

activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
TrajGRU_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 2, 3]}),
        OrderedDict({'conv2_leaky_1': [32, 96, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=32, b_h_w=(4, 128, 128), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation),

        TrajGRU(input_channel=96, num_filter=96, b_h_w=(4, 64, 64), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation),
        TrajGRU(input_channel=96, num_filter=96, b_h_w=(4, 32, 32), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation)
    ]
]

TrajGRU_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 32, 4, 2, 1]}),
        OrderedDict({
            'deconv3_pad_leaky_1': [32, 4, 3, 2, 1], # 256*256
            'conv3_leaky_1': [4, 4, 3, 1, 1],
            'conv4': [4, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=96, num_filter=96, b_h_w=(4, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation),

        TrajGRU(input_channel=96, num_filter=96, b_h_w=(4, 64, 64), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation),
        TrajGRU(input_channel=32, num_filter=32, b_h_w=(4, 128, 128), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation)
    ]
]