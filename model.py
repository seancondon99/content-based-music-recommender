import torch
import torch.nn as nn
import torch.nn.functional as F


class Audio_CNN(nn.Module):
    '''
    The CNN model we will use to map song content to likeability score.
    More or less a PyTorch implementation of a similar architecture found
    here: https://benanne.github.io/2014/08/05/spotify-cnns.html
    '''
    def __init__(self):
        super().__init__()
        #input data size is (-1, 128, 599) = (batch, n_freq_bins, n_frames)
        self.batch_dim, self.freq_dim, self.time_dim = 0,1,2

        #1d convolutions of kernel size = 4 along time dimension, relu activation
        self.conv1 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 4)
        self.pool1 = nn.MaxPool1d(kernel_size = 4)

        #conv2 layers
        self.conv2 = nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size = 4)
        self.pool2 = nn.MaxPool1d(kernel_size = 2)

        #conv3 layers
        self.conv3 = nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 4)
        self.pool3 = nn.MaxPool1d(kernel_size = 2)

        #pool the time axis in three different ways, maxpool, average_pool, and L2_pool (global temporal pooling)
        self.temppool_max = nn.MaxPool1d(kernel_size = 35)
        self.temppool_avg = nn.AvgPool1d(kernel_size = 35)
        self.temppool_l2  = nn.LPPool1d(2, kernel_size = 35)

        #fully connected layers
        self.fc1 = nn.Linear(1536, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1)


    def forward(self, x):

        #apply conv1 and maxpool1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        #apply conv2 and maxpool2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        #apply conv3 and maxpool3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        #apply global temporal pooling
        x_max = self.temppool_max(x)
        x_avg = self.temppool_avg(x)
        x_l2  = self.temppool_l2(x)

        #flatten and concatenate all pooled layers
        x_max = torch.flatten(x_max, start_dim = -2, end_dim = -1)
        x_avg = torch.flatten(x_avg, start_dim=-2, end_dim=-1)
        x_l2  = torch.flatten(x_l2, start_dim=-2, end_dim=-1)
        x = torch.cat((x_max, x_avg, x_l2), dim=-1)

        #apply fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x




if __name__ == '__main__':

    trial_data = torch.rand(size = (10, 128, 599))
    print(trial_data.shape)

    model = Audio_CNN()
    trial_data_out = model(trial_data)
    print(trial_data_out.shape)