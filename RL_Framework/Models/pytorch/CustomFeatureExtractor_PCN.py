import gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomPCN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    : param features_dim: (int) Number of features extracted -> number of unit for last layer
    """
    def __init__(self, observation_space: gym.spaces.Box, in_channels:int=4, features_dim: int=1024):
        super(CustomPCN, self).__init__(observation_space, features_dim)
        self.in_channels=in_channels
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=128, kernel_size=1),    
            #nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            #nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Conv1d(in_channels=512, out_channels=features_dim, kernel_size=1)
        )    

    def forward(self, xyzh):
        B, N, _ = xyzh.shape

        #encoder
        feature = self.first_conv(xyzh.transpose(2, 1)) #(B, 256, N)
        feature_global = th.max(feature, dim=2, keepdim=True) #(B, 256, 1)
        feature_global = feature_global[0]
        feature = th.cat([feature_global.expand(-1, -1, N), feature], dim=1) #(B, 512, N)
        feature = self.second_conv(feature) #(B, 1024, N)
        feature_global = th.max(feature, dim=2, keepdim=False)[0] #(B, 1024)   
        return feature_global.contiguous()
        
