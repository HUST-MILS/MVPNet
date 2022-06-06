from base64 import decode
from statistics import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import yaml

from model.lightning import lightning_model

class Encoder(nn.Module):
    """Encode a range view tensor and a BEV tensor to a vector"""

    def __init__(self,cfg):
        super().__init__(cfg)
        'self, input_nc=3, encode_dim=1024, lstm_hidden_size=1024, seq_len=SEQ_SIZE, num_lstm_layers=1, bidirectional=False'
        self.encode_dim = 1024
        self.lstm_hidden_size=1024
        self.seq_len = 5
        self.num_lstm_layers=2
        self.num_directions = 1 # 单项or双向lstm

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

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=(2,3),padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            ## 2*2

        )
        
        self.fc = nn.Linear(self.feature,self.encode_dim)
        self.lstm = nn.LSTM(input_size=self.encode_dim,num_layers=self.num_lstm_layers,hidden_size=self.encode_dim,batch_first=True)

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
                self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return Variable(h), Variable(c)

    def forward(self,x1,x2):
        """ 
        x1:range view [B,P,H,W]
        x2:BEV view [B,P,H,W]
        """

        B = x1.size(0)
        x1 = x1.view(B*self.seq_len,1,64,2048)
        x2 = x2.view(B*self.seq_len,1,216,486)

        #[B*seq_len,256,2,2]
        x1 = self.feature1(x1)
        x2 = self.feature2(x2)

        #[B*seq_len,1024]
        x1 = x1.view(B*self.seq_len,-1)
        x2 = x2.view(B*self.seq_len,-1)

        #[B*seq_len,2048]
        x = torch.cat((x1,x2),1)

        #[B*seq_len,1024]
        x = self.fc(x)

        #[B,seq_len,1024]
        x = x.view(-1,self.seq_len,x.size(1))
        h0,c0 = self.init_hidden(x)
        output,(hn,cn) = self.lstm(x,(h0,c0))
        return hn

        '''
        print(self.feature1.size())
        print(self.feature1.size(1))
        print(self.feature1.size(2))
        print(self.feature1.size(4))

        
        self.feature1 = self.feature1.view(self.feature1.size(0),print(self.feature1.size(1)),-1)
        self.feature2 = self.feature1.view(self.feature2.size(0),print(self.feature1.size(1)),-1)
        transpose(input,dim0,dim1)
        

        self.feature = torch.cat(self.feature1,self.feature2)
        '''

class Decoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.encode_dim = 1024
        self.output_nc = 1
        self.project = nn.Sequential(
            nn.Linear(self.encode_dim, 1024*1*1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024,512,4),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512,256,4),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256,128,4),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64,4),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64,32,4),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32,16,4),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16,self.output_nc,4),
            nn.Sigmoid(),
        )
        def forward(self,x):
            x = self.project(x)
            x = x.view(-1,1024,1,1)
            decode = self.decoder(x)
            return decode

class Net():
    def __init__():
        super().__init__()
