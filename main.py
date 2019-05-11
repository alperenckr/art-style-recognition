import layers
import numpy as np
import CNN

def epoch():
    pass
    #conv_forward(input: 224x224x3, filter_size: 11x11x96, stride: 4, padding: 2)output: 55x55x96
    #maxpool_forward(input: 55x55x96, filter_size: 3x3, stride: 2)output: 27x27x96
    #conv_forward(input: 27x27x96, filter_size: 5x5x256, stride: 1, padding: 2)output: 27x27x256
    #maxpool_forward(input: 27x27x256, filter_size: 3x3, stride: 2)output: 13x13x256
    #conv_forward(input: 13x13x256, filter_size: 3x3x384, stride: 1, padding: 1)output: 13x13x384
    #conv_forward(input: 13x13x384, filter_size: 3x3x384, stride: 1, padding: 1)output: 13x13x384
    #conv_forward(input: 13x13x384, filter_size: 3x3x256, stride: 1, padding: 1)output: 13x13x256
    #maxpool_forward(input: 13x13x256, filter_size: 3x3, stride: 2)output: 6x6x256
    #fc_forward(input: 1x1x9216)output: 1x1x4096
    #fc_forward(input: 1x1x4096)output: 1x1x1000


a = CNN.OurConvNet()