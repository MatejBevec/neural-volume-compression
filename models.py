import torch
from torch import nn
from torch.nn import functional as F


def save_model(model, pth):
    torch.save(model.state_dict(), pth)

def load_model(model, pth):
    model.load_state_dict(torch.load(pth))
    model.eval()



# BASIC CONVOLUTIONAL VAE

class ConvVAE(nn.Module):

    def __init__(self, in_shape,
                        latent_dim=64,
                        hidden_channels=[16, 64, 128, 256],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        log_shapes=False):
        
        super().__init__()

        self.latent_dim = latent_dim 
        self.hidden_channels = hidden_channels
        self.ch, self.h, self.w = in_shape

        # ENCODER

        in_ch = self.ch # we only use mono
        encoder_blocks = []
        for ch in hidden_channels:
            encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels=ch,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding),
                    nn.BatchNorm2d(ch),
                    nn.LeakyReLU()
                )
            )
            in_ch = ch

        self.encoder = nn.Sequential(*encoder_blocks) # convolutional encoder
        # wait do we always end up with 2x2 or do we need to compute this?

        probe = self.encoder(torch.rand((1, self.ch, self.h, self.w)))
        _, _, self.ac_h, self.ac_w = probe.shape
        after_conv_size = hidden_channels[-1] * self.ac_h * self.ac_w
        print(probe.shape)
        self.fc_mu = nn.Linear(after_conv_size, latent_dim) # FC -> mean vector
        self.fc_var = nn.Linear(after_conv_size, latent_dim) # FC -> variance vector


        # DECODER

        self.fc_decoder = nn.Linear(latent_dim, after_conv_size) # input to decoder

        decoder_blocks = []
        in_ch = hidden_channels[-1]
        dec_channels = list(reversed(hidden_channels))
        print(dec_channels)
        dec_channels.append(hidden_channels[0]) # yes or no
        for ch in dec_channels[1:]:
            decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch,
                                       ch,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=1),
                    nn.BatchNorm2d(ch),
                    nn.LeakyReLU())
            )
            in_ch = ch

        self.decoder = nn.Sequential(*decoder_blocks) # (de)convolutional decoder

        self.final_layer = nn.Sequential(
                                nn.Conv2d(in_ch, out_channels=self.ch,
                                      kernel_size=kernel_size, padding=padding),
                                nn.Tanh(),
        )

        print(self)


    def encode(self, x):
        
        print("in: ", x.shape)
        res = self.encoder(x)
        print("after conv: ", res.shape)
        res = torch.flatten(res, start_dim=1)
        print("flattened: ", res.shape)
        mu = self.fc_mu(res)
        var = self.fc_var(res)

        print("mu, var: ", mu.shape, var.shape)

        return mu, var

    def reparametrize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        res = self.fc_decoder(z)
        print("after fc_decoder: ", res.shape)
        # again, idk about the 2x2
        res = res.view(-1, self.hidden_channels[-1], self.ac_h, self.ac_w)
        print("unflattened", res.shape)
        res = self.decoder(res)
        print("after conv ", res.shape)
        y = self.final_layer(res)
        print("out: ", y.shape)
        
        return y


    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparametrize(mu, var)
        y = self.decode(z)
        
        return y, mu, var
    


class DenseVAE(nn.Module):

    def __init__(self, in_shape,
                        latent_dim=64,
                        hidden_dims=[256]):
        
        super().__init__()

        self.latent_dim = latent_dim
        self.ch, self.h, self.w = in_shape
        self.input_dim = self.ch * self.h * self.w 
        self.hidden_dims = hidden_dims

        encoder_layers = []
        in_dim = self.input_dim
        for out_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.LeakyReLU())
            in_dim = out_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        decoder_layers = []
        in_dim = self.latent_dim
        for out_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.LeakyReLU())
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, self.input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        print(self)


    def encode(self, x):
        res = torch.flatten(x, start_dim=1)
        res = self.encoder(res)
        mu = self.fc_mu(res)
        var = self.fc_var(res)

        return mu, var

    def reparametrize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def decode(self, z):
        y = torch.sigmoid(self.decoder(z))

        return y


    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparametrize(mu, var)
        res = self.decode(z)
        y = res.view(-1, self.ch, self.h, self.w)
        return y, mu, var
    

    def save_model(self, pth):
        torch.save({
            "state_dict": self.state_dict(),
            "in_shape": (self.ch, self.h, self.w),
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims
        }, pth)

    @staticmethod
    def load_model(pth):
        params = torch.load(pth)
        model = DenseVAE(params["in_shape"], params["latent_dim"], params["hidden_dims"])
        model.load_state_dict(params["state_dict"])
        model.eval()
        return model
    
