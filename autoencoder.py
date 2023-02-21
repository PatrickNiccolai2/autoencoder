from torch import nn

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_img_size = 28
        #encoder_layers = 5
        encoder_hidden_units = [512, 256, 128, 64, 10]
        encoder_layers = len(encoder_hidden_units)

        self.encoder = nn.Sequential()

        for i in range(encoder_layers):
            if i == 0:
                # First layer
                self.encoder.add_module("fc_" + str(i), nn.Linear(input_img_size * input_img_size, encoder_hidden_units[i]))
                self.encoder.add_module("relu_" + str(i), nn.ReLU())
            elif i == encoder_layers-1:
                # Last layer, no rele
                self.encoder.add_module("fc_" + str(i), nn.Linear(encoder_hidden_units[i-1], encoder_hidden_units[i]))
            else:
                self.encoder.add_module("fc_" + str(i), nn.Linear(encoder_hidden_units[i-1], encoder_hidden_units[i]))
                self.encoder.add_module("relu_" + str(i), nn.ReLU())

        decoder_layers = encoder_layers
        decoder_hidden_units = encoder_hidden_units[::-1]

        self.decoder = nn.Sequential()

        for i in range(decoder_layers):
            if i == decoder_layers-1:
                # Last layer
                self.decoder.add_module("fc_" + str(i), nn.Linear(decoder_hidden_units[i], input_img_size * input_img_size))
                self.decoder.add_module("sig_" + str(i), nn.Sigmoid())
            else:
                self.decoder.add_module("fc_" + str(i), nn.Linear(decoder_hidden_units[i], decoder_hidden_units[i+1]))
                self.decoder.add_module("relu_" + str(i), nn.ReLU())

        #print(self.encoder)
        #print(self.decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        #print("encoded = " + str(encoded))
        decoded = self.decoder(encoded)
        return decoded