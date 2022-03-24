from torchvision import transforms
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 0, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        return x
    
class UNET(nn.Module):
    def __init__(
        self, in_c=3, out_c=8, features = [64, 128, 256, 512] 
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature
            
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.bottom = DoubleConv(features[-1], features[-1]*2)
        self.finalconv = nn.Conv2d(features[0], out_c, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottom(x)
        skip_connections = skip_connections[::-1]
        
        skipidx = 0
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            #print(x.shape)
            skip_connection = skip_connections[skipidx]
            if x.shape != skip_connection.shape:
                #x = TF.resize(x, size=skip_connection.shape[2:])
                transform = transforms.CenterCrop((x.shape[2], x.shape[3]))
                skip_connection = transform(skip_connection)
                # skip_connection = TF.resize(skip_connection, size=x.shape[2:])
            #print(skip_connection.shape, skip_connection.shape[2:])
            skipidx +=1
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            
        return self.finalconv(x)