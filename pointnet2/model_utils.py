from ast import arg
from numpy import block
import torch
import torch.nn as nn
from typing import List, Tuple


class SharedMLP(nn.Sequential):

    def __init__(
                self,
                args: List[int],
                *,
                bn: bool = False,
                activation = nn.ReLU(inplace=True),
                preact: bool = False,
                first: bool = False,
                name: str = "",
                instance_norm: bool = False
    ):
        super().__init__()
        for i in range(len(args)-1):

            block = nn.Sequential(
                    nn.Conv2d(args[i], args[i+1], kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(args[i+1]),
                    nn.ReLU()
                )
            self.add_module(
                name + 'layer{}'.format(i),
                block
            )                  





            # self.add_module(
            #     name + 'layer{}'.format(i), 
            #     nn.Conv2d(args[i], args[i+1], kernel_size=(1, 1), stride=(1, 1), bias=False)
            #     )
            # self.add_module(
            #     name + 'layer{}'.format(i),
            #     nn.BatchNorm2d(args[i+1])
            # )
            # self.add_module(
            #     name + 'layer{}'.format(i),
            #     nn.ReLU()
            



# class Conv2d(nn.Sequential):
#     def __init__(
#                 self,
#                 input_channel,
#                 output_channel,
#                 kernel_size=(1, 1),
#                 bn,
#                 activation
            
#     ):
#         self.input_channel = input_channel
#         self.output_channel = output_channel
#         self.mlp