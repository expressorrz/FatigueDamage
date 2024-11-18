import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, dropout=0.1):
        super(AutoEncoder, self).__init__()
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=10, stride=8, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(16, 32, kernel_size=8, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=6, stride=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

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
            nn.ReLU()
        )

        self.decoder_unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 5, 5))

        # Decoder part
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=3, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(32, 16, kernel_size=9, stride=4, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose2d(16, 1, kernel_size=10, stride=8, padding=1, output_padding=0),
            nn.Sigmoid()  # Assuming the output needs to be normalized between 0 and 1
        )

        self.regression = nn.Linear(8, 1)

    def encoder_layer(self, x):
        # x0 = x
        # for i, layer in enumerate(self.encoder):
        #     x0 = layer(x0)
        #     print(f'Layer {i}: {x0.shape}')

        x = self.encoder(x)
        x = self.encoder_flatten(x)
        x = self.encoder_lin(x)
        
        return x
    
    def decoder_layer(self, x):
        x = self.decoder_lin(x)
        x = self.decoder_unflatten(x)
        # x0 = x
        # for i, layer in enumerate(self.decoder):
        #     x0 = layer(x0)
        #     print(f'Layer {i}: {x0.shape}')
        x = self.decoder(x)
        return x
    
    def regression_layer(self, x):
        x = self.encode_layer(x)
        x = self.regression(x)
        return x

    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.decoder_layer(x)
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