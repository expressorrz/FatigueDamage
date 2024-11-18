import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dropout):
        super(deconv_block, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, dropout=0.1):
        super(AutoEncoder, self).__init__()
        # Encoder part
        self.encoder = nn.ModuleList([
            # in_channels, out_channels, kernel_size, stride, padding, dropout
            conv_block(1, 16, 10, 8, 1, dropout),
            conv_block(16, 32, 8, 4, 1, dropout),
            conv_block(32, 64, 6, 3, 1, dropout),
            conv_block(64, 128, 4, 2, 1, dropout),
            conv_block(128, 256, 4, 2, 1, dropout)
        ])

        self.encoder_flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(256 * 5 * 5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256 * 5 * 5),
            nn.BatchNorm1d(256 * 5 * 5),
            nn.ReLU(),
        )
        self.decoder_unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 5, 5))
        
        # Decoder part
        self.decoder = nn.ModuleList([
            # in_channels, out_channels, kernel_size, stride, padding, output_padding
            deconv_block(256, 128, 4, 2, 1, 0, dropout),
            deconv_block(128, 64, 4, 2, 1, 0, dropout),
            deconv_block(64, 32, 7, 3, 1, 1, dropout),
            deconv_block(32, 16, 9, 4, 1, 1, dropout),
            deconv_block(16, 1, 10, 8, 1, 0, dropout),
            nn.Sigmoid()
        ])

    def encoder_layer(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # print(f'{i} x shape: {x.shape}')

        x = self.encoder_flatten(x)
        x = self.encoder_lin(x)

        return x
    
    def decoder_layer(self, x):
        x = self.decoder_lin(x)
        x = self.decoder_unflatten(x)

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            # print(f'{i} x shape: {x.shape}')     

        return x
    
    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.decoder_layer(x)
        x = torch.sigmoid(x)
        return x
    

# Example usage
if __name__ == "__main__":
    model = AutoEncoder()
    input_image = torch.randn(64, 1, 2048, 2048)


    feature = model.encoder_layer(input_image)
    output_image = model.decoder_layer(feature)

    print('Input image shape:', input_image.shape)
    print('Feature shape:', feature.shape)
    print('Output image shape:', output_image.shape)