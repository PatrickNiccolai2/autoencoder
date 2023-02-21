from torch import nn

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_img_size = 28
        #encoder_layers = 5
        encoder_channels= [8, 16, 32, 10]
        encoder_layers = len(encoder_channels)
        encoder_kernel_size = [2,2,3,3]
        stride = 2
        padding = 0

        self.encoder = nn.Sequential()

        for i in range(encoder_layers):
            if i == 0:
                # First layer
                # 1 channel input
                self.encoder.add_module("conv_" + str(i), nn.Conv2d(1, encoder_channels[i], encoder_kernel_size[i], stride, padding))
                self.encoder.add_module("relu_" + str(i), nn.ReLU())
            elif i == encoder_layers-1:
                # Last layer, no relu, kernel = 2
                self.encoder.add_module("conv_" + str(i), nn.Conv2d(encoder_channels[i-1], encoder_channels[i], encoder_kernel_size[i], stride, padding))
                #self.encoder.add_module("pool_" + str(i), nn.MaxPool2d(2, stride, padding))
            else:
                self.encoder.add_module("conv_" + str(i), nn.Conv2d(encoder_channels[i-1], encoder_channels[i], encoder_kernel_size[i], stride, padding))
                self.encoder.add_module("relu_" + str(i), nn.ReLU())

        decoder_layers = encoder_layers
        decoder_channels = encoder_channels[::-1]
        decoder_kernel_size = encoder_kernel_size[::-1]

        self.decoder = nn.Sequential()

        for i in range(decoder_layers):
            if i == decoder_layers-1:
                # Last layer
                self.decoder.add_module("deconv_" + str(i), nn.ConvTranspose2d(decoder_channels[i], 1, decoder_kernel_size[i], stride, padding))
                self.decoder.add_module("sig_" + str(i), nn.Sigmoid())
            elif i == 0:
                # First layer, do unpooling ?
                # kernel = 2
                #self.decoder.add_module("unpool_" + str(i), nn.MaxUnpool2d(2, stride, padding))
                self.decoder.add_module("deconv_" + str(i), nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], decoder_kernel_size[i], stride, padding))
                self.decoder.add_module("relu_" + str(i), nn.ReLU())
            else:
                self.decoder.add_module("deconv_" + str(i), nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], decoder_kernel_size[i], stride, padding))
                self.decoder.add_module("relu_" + str(i), nn.ReLU())

        #print(self.encoder)
        #print(self.decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        #print("encoded = " + str(encoded))
        decoded = self.decoder(encoded)
        return decoded