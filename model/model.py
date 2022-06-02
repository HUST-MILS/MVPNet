from statistics import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.lightning import lightning_model

class Encoder(nn.Module):
    """Encode a range view tensor and a BEV tensor to a vector"""

    def __init__(self,cfg):
        super().__init__(cfg)
        'self, input_nc=3, encode_dim=1024, lstm_hidden_size=1024, seq_len=SEQ_SIZE, num_lstm_layers=1, bidirectional=False'
        self.encode_dim = 1024
        self.lstm_hidden_size=1024

        self.num_lstm_layers=2
        self.channel1 = 1
        self.channel2 = 1
        self.feature1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel1,out_channels=16,kernel_size=3,stride=(2,4),padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=(2,4),padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=(2,4),padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=(2,4),padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=self.channel1,out_channels=256,kernel_size=3,stride=(2,4),padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            ## 2*2

        )
        self.feature2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channel1,out_channels=16,kernel_size=3,stride=(3,3),padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=(3,3),padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=(2,3),padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=self.channel1,out_channels=256,kernel_size=3,stride=(2,3),padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            ## 2*2

        )
        print(self.feature1.size())
        print(self.feature1.size(1))
        print(self.feature1.size(2))
        print(self.feature1.size(4))

        '''
        self.feature1 = self.feature1.view(self.feature1.size(0),print(self.feature1.size(1)),-1)
        self.feature2 = self.feature1.view(self.feature2.size(0),print(self.feature1.size(1)),-1)
        transpose(input,dim0,dim1)
        '''

        self.feature = torch.cat(self.feature1,self.feature2)
        self.fc = nn.Linear(self.feature,self.encode_dim)
        self.lstm = nn.LSTM(self.encode_dim,self.encode_dim,batch_first=True)


    def forward(self,x1,x2):
        """
        x1:range view [B,P,H,W]
        x2:BEV view [B,P,H,W]
        """