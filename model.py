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
    
class linear_block(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(linear_block, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder_dropout=0.0, decoder_dropout=0.0):
        super(AutoEncoder, self).__init__()
        # Encoder part
        self.encoder = nn.Sequential(
            conv_block(2, 16, 10, 4, 1, encoder_dropout),
            conv_block(16, 32, 8, 4, 1, encoder_dropout),
            conv_block(32, 64, 6, 2, 1, encoder_dropout),
            conv_block(64, 128, 4, 2, 1, encoder_dropout),
            conv_block(128, 256, 4, 2, 1, encoder_dropout),
            nn.Flatten(start_dim=1),
            linear_block(256 * 3 * 3, 64, encoder_dropout),
            nn.Linear(64, 8),
            nn.BatchNorm1d(8)
        )

        # Decoder part
        self.decoder = nn.Sequential(
            linear_block(8, 64, dropout=decoder_dropout),
            linear_block(64, 256 * 3 * 3, dropout=decoder_dropout),
            nn.Unflatten(dim=1, unflattened_size=(256, 3, 3)),
            deconv_block(256, 128, 4, 2, 1, 1, dropout=decoder_dropout),
            deconv_block(128, 64, 4, 2, 1, 0, dropout=decoder_dropout),
            deconv_block(64, 32, 6, 2, 1, 1, dropout=decoder_dropout),
            deconv_block(32, 16, 8, 4, 1, 1, dropout=decoder_dropout),
            nn.ConvTranspose2d(16, 2, 11, 4, 2, 1),
            nn.Sigmoid()  # Assuming the output needs to be normalized between 0 and 1
        )

    def encoder_layer(self, x):
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # print(f'Layer {i}: {x.shape}')        
        return x
    
    def decoder_layer(self, x):
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            # print(f'Layer {i}: {x.shape}')
        return x
    
    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.decoder_layer(x)
        return x

    

# Example usage
if __name__ == "__main__":
    model = AutoEncoder()
    input_image = torch.randn(64, 2, 512, 512)

    feature = model.encoder_layer(input_image)
    output_image = model.decoder_layer(feature)

    print('Input image shape:', input_image.shape)
    print('Feature shape:', feature.shape)
    print('Output image shape:', output_image.shape)