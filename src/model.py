import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Transformer
import numpy as np
from MMT.modules.transformer import TransformerEncoder
from torch.utils.tensorboard import SummaryWriter
import os
from torch.distributions import LowRankMultivariateNormal
from torch.optim import Adam
import matplotlib.pyplot as plt

X_A_SHAPE = (128,128)
"""Processed spectrogram shape: ``[freq_bins, time_bins]``"""
X_A_DIM = np.prod(X_A_SHAPE)
"""Processed spectrogram dimension: ``freq_bins * time_bins``"""

X_V_SHAPE = (240,320)
"""Processed video shape: ``[freq_bins, time_bins]``"""
X_V_DIM = np.prod(X_V_SHAPE)
"""Processed video dimension: ``freq_bins * time_bins``"""

X_L_SHAPE = (1,1)
X_L_DIM = np.prod(X_L_SHAPE)

class MultiTransModelContext_Reduced(nn.Module):
    def __init__(self, hyp_params, writer=None):
        """
        Construct a MulT model.
        """
        super(MultiTransModelContext_Reduced, self).__init__()
        self.hyp_params = hyp_params

        self.save_dir = hyp_params.save_dir
        self.writer = writer

        self.z_dim = hyp_params.z_dim
        self.d_v = hyp_params.z_dim
        self.d_a = hyp_params.z_dim
        self.lr = hyp_params.lr

        self.orig_d_v = hyp_params.orig_d_v
        self.orig_d_a = hyp_params.orig_d_a
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        combined_dim =  self.d_a + self.d_v

        self.partial_mode = self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_a   # assuming d_l == d_a == d_v
        else:
            combined_dim = (self.d_a + self.d_v)
        
        self._vis_network()
        self._aud_network()
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        self.epoch = 0
        self.loss = {'train':{}, 'test':{}}
        self.device = "cuda"
        if self.save_dir != '' and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.trans_v = self.get_transformer(layers=self.layers)
        # 2. Crossmodal Attentions
        # if self.lonly:
        #     self.trans_l_with_a = self.get_transformer(self_type='la')
        #     self.trans_l_with_v = self.get_transformer(self_type='lv')
        if self.aonly:
            self.trans_a_with_v = self.get_transformer(self_type='av')
        if self.vonly:
            self.trans_v_with_a = self.get_transformer(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_transformer(self_type='a_mem', layers=self.layers)
        self.trans_v_mem = self.get_transformer(self_type='v_mem', layers=self.layers)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        

    def _vis_network(self):
        """Define all the visual network layers."""
        self.v_conv1 = nn.Conv2d(1, 4, 3,1,padding=1)
        self.v_conv2 = nn.Conv2d(4, 4, 3,2,padding=1)
        self.v_conv3 = nn.Conv2d(4, 8, 3,1,padding=1)
        self.v_conv4 = nn.Conv2d(8, 8, 3,2,padding=1)
        self.v_conv5 = nn.Conv2d(8, 16,3,1,padding=1)
        self.v_conv6 = nn.Conv2d(16,16,3,2,padding=1)
        self.v_conv7 = nn.Conv2d(16,24,3,1,padding=1)
        self.v_conv8 = nn.Conv2d(24,24,3,2,padding=1)
        self.v_conv9 = nn.Conv2d(24,32,3,1,padding=1)

        self.v_bn1 = nn.BatchNorm2d(1)
        self.v_bn2 = nn.BatchNorm2d(4)
        self.v_bn3 = nn.BatchNorm2d(4)
        self.v_bn4 = nn.BatchNorm2d(8)
        self.v_bn5 = nn.BatchNorm2d(8)
        self.v_bn6 = nn.BatchNorm2d(16)
        self.v_bn7 = nn.BatchNorm2d(16)
        self.v_bn8 = nn.BatchNorm2d(24)
        self.v_bn9 = nn.BatchNorm2d(24)
        self.v_fc1 = nn.Linear(9600,1200)

        self.v_fc21 = nn.Linear(1200,300)
        self.v_fc22 = nn.Linear(1200,300)
        self.v_fc23 = nn.Linear(1200,300)

        #separate decoders for each modality
        self.v_fc3 = nn.Linear(300,1200)
        self.v_fc4 = nn.Linear(1200,9600)

        self.v_convt1 = nn.ConvTranspose2d(32,24,3,1,padding=1) #collapse across filters
        self.v_convt2 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1) # changing size of output image
        self.v_convt3 = nn.ConvTranspose2d(24,16,3,1,padding=1)
        self.v_convt4 = nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1)
        self.v_convt5 = nn.ConvTranspose2d(16,8,3,1,padding=1)
        self.v_convt6 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1)
        self.v_convt7 = nn.ConvTranspose2d(8,4,3,1,padding=1)
        self.v_convt8 = nn.ConvTranspose2d(4,4,3,2,padding=1,output_padding=1)
        self.v_convt9 = nn.ConvTranspose2d(4,1,3,1,padding=1)
        
        
        self.v_bn10 = nn.BatchNorm2d(32)
        self.v_bn11 = nn.BatchNorm2d(24)
        self.v_bn12 = nn.BatchNorm2d(24)
        self.v_bn13 = nn.BatchNorm2d(16)
        self.v_bn14 = nn.BatchNorm2d(16)        
        self.v_bn15 = nn.BatchNorm2d(8)
        self.v_bn16 = nn.BatchNorm2d(8)
        self.v_bn17 = nn.BatchNorm2d(4)
        self.v_bn18 = nn.BatchNorm2d(4)

    def _aud_network(self):
        """Define all the audio network layers."""
        self.a_conv1 = nn.Conv2d(1, 8, 3,1,padding=1)
        self.a_conv2 = nn.Conv2d(8, 8, 3,2,padding=1)
        self.a_conv3 = nn.Conv2d(8, 16,3,1,padding=1)
        self.a_conv4 = nn.Conv2d(16,16,3,2,padding=1)
        self.a_conv5 = nn.Conv2d(16,24,3,1,padding=1)
        self.a_conv6 = nn.Conv2d(24,24,3,2,padding=1)
        self.a_conv7 = nn.Conv2d(24,32,3,1,padding=1)

        self.a_bn1 = nn.BatchNorm2d(1)
        self.a_bn2 = nn.BatchNorm2d(8)
        self.a_bn3 = nn.BatchNorm2d(8)
        self.a_bn4 = nn.BatchNorm2d(16)
        self.a_bn5 = nn.BatchNorm2d(16)
        self.a_bn6 = nn.BatchNorm2d(24)
        self.a_bn7 = nn.BatchNorm2d(24)

        self.a_fc1 = nn.Linear(8192,1024)

        self.a_fc21 = nn.Linear(1024,300)
        self.a_fc22 = nn.Linear(1024,300)
        self.a_fc23 = nn.Linear(1024,300)

        #separate decoders for each modality
        self.a_fc3 = nn.Linear(300,1024)
        self.a_fc4 = nn.Linear(1024,8192)

        self.a_convt1 = nn.ConvTranspose2d(32,24,3,1,padding=1) #collapse across filters
        self.a_convt2 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1) # changing size of output image
        self.a_convt3 = nn.ConvTranspose2d(24,16,3,1,padding=1)
        self.a_convt4 = nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1)
        self.a_convt5 = nn.ConvTranspose2d(16,8,3,1,padding=1)
        self.a_convt6 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1)
        self.a_convt7 = nn.ConvTranspose2d(8,1,3,1,padding=1)
        
        self.a_bn8 = nn.BatchNorm2d(32)
        self.a_bn9 = nn.BatchNorm2d(24)
        self.a_bn10 = nn.BatchNorm2d(24)
        self.a_bn11 = nn.BatchNorm2d(16)
        self.a_bn12 = nn.BatchNorm2d(16)        
        self.a_bn13 = nn.BatchNorm2d(8)
        self.a_bn14 = nn.BatchNorm2d(8)

    def _get_layers(self):
        """Return a dictionary mapping names to network layers."""
        return {'a_fc1':self.a_fc1, 'a_fc21':self.a_fc21, 'a_fc22':self.a_fc22, 'a_fc23':self.a_fc23, 
                'a_fc3':self.a_fc3, 'a_fc4':self.a_fc4, 'a_bn1':self.a_bn1,
                'a_bn2':self.a_bn2, 'a_bn3':self.a_bn3, 'a_bn4':self.a_bn4, 'a_bn5':self.a_bn5,
                'a_bn6':self.a_bn6, 'a_bn7':self.a_bn7, 'a_bn8':self.a_bn8, 'a_bn9':self.a_bn9,
                'a_bn10':self.a_bn10, 'a_bn11':self.a_bn11, 'a_bn12':self.a_bn12,
                'a_bn13':self.a_bn13, 'a_bn14':self.a_bn14, 
                'a_conv1':self.a_conv1, 'a_conv2':self.a_conv2, 'a_conv3':self.a_conv3,
                'a_conv4':self.a_conv4, 'a_conv5':self.a_conv5, 'a_conv6':self.a_conv6,
                'a_conv7':self.a_conv7,  'a_convt1':self.a_convt1, 'a_convt2':self.a_convt2,
                'a_convt3':self.a_convt3, 'a_convt4':self.a_convt4,
                'a_convt5':self.a_convt5, 'a_convt6':self.a_convt6,'a_convt7':self.a_convt7,
                
                'v_fc1':self.v_fc1, 'v_fc21':self.v_fc21, 'v_fc22':self.v_fc22, 'v_fc23':self.v_fc23, 
                'v_fc3':self.v_fc3, 'v_fc4':self.v_fc4, 'v_bn1':self.v_bn1,
                'v_bn2':self.v_bn2, 'v_bn3':self.v_bn3, 'v_bn4':self.v_bn4, 'v_bn5':self.v_bn5,
                'v_bn6':self.v_bn6, 'v_bn7':self.v_bn7, 'v_bn8':self.v_bn8, 'v_bn9':self.v_bn9,
                'v_bn10':self.v_bn10, 'v_bn11':self.v_bn11, 'v_bn12':self.v_bn12,
                'v_bn13':self.v_bn13, 'v_bn14':self.v_bn14, 'v_bn15':self.v_bn15,
                'v_bn16':self.v_bn16, 'v_bn17':self.v_bn17, 'v_bn18':self.v_bn18,
                'v_conv1':self.v_conv1, 'v_conv2':self.v_conv2, 'v_conv3':self.v_conv3,
                'v_conv4':self.v_conv4, 'v_conv5':self.v_conv5, 'v_conv6':self.v_conv6,
                'v_conv7':self.v_conv7, 'v_conv8':self.v_conv8, 'v_conv9':self.v_conv9,
                'v_convt1':self.v_convt1, 'v_convt2':self.v_convt2,
                'v_convt3':self.v_convt3, 'v_convt4':self.v_convt4,
                'v_convt5':self.v_convt5, 'v_convt6':self.v_convt6,
                'v_convt7':self.v_convt7, 'v_convt8':self.v_convt8, 'v_convt9':self.v_convt9}
    
    def get_transformer(self, self_type='v', layers=-1):

        # if self_type in ['l', 'al', 'vl']:
        #     embed_dim, attn_dropout = self.d_l, self.attn_dropout
        if self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        # elif self_type == 'l_mem':
        #     embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    def encode_a(self, x):
        """
        Compute :math:`q(z|x)`.

        .. math:: q(z|x) = \mathcal{N}(\mu, \Sigma)
        .. math:: \Sigma = u u^{T} + \mathtt{diag}(d)

        where :math:`\mu`, :math:`u`, and :math:`d` are deterministic functions
        of `x` and :math:`\Sigma` denotes a covariance matrix.

        Parameters
        ----------
        x : torch.Tensor
            The input images, with shape: ``[batch_size, height=128,
            width=128]``

        Returns
        -------
        mu : torch.Tensor
            Posterior mean, with shape ``[batch_size, self.z_dim]``
        u : torch.Tensor
            Posterior covariance factor, as defined above. Shape:
            ``[batch_size, self.z_dim]``
        d : torch.Tensor
            Posterior diagonal factor, as defined above. Shape:
            ``[batch_size, self.z_dim]``
        """
        x = x.unsqueeze(1)
        x = F.relu(self.a_conv1(self.a_bn1(x)))
        x = F.relu(self.a_conv2(self.a_bn2(x)))
        x = F.relu(self.a_conv3(self.a_bn3(x)))
        x = F.relu(self.a_conv4(self.a_bn4(x)))
        x = F.relu(self.a_conv5(self.a_bn5(x)))
        x = F.relu(self.a_conv6(self.a_bn6(x)))
        x = F.relu(self.a_conv7(self.a_bn7(x)))
        x = x.view(-1, 8192)
        x = F.relu(self.a_fc1(x))

        mu = self.a_fc21(x)
        u = self.a_fc22(x).unsqueeze(-1)
        d = torch.exp(self.a_fc23(x))

        return mu, u, d
    
    def encode_v(self, x):
        """
        Compute :math:`q(z|x)`.

        .. math:: q(z|x) = \mathcal{N}(\mu, \Sigma)
        .. math:: \Sigma = u u^{T} + \mathtt{diag}(d)

        where :math:`\mu`, :math:`u`, and :math:`d` are deterministic functions
        of `x` and :math:`\Sigma` denotes a covariance matrix.

        Parameters
        ----------
        x : torch.Tensor
            The input images, with shape: ``[batch_size, height=128,
            width=128]``

        Returns
        -------
        mu : torch.Tensor
            Posterior mean, with shape ``[batch_size, self.z_dim]``
        u : torch.Tensor
            Posterior covariance factor, as defined above. Shape:
            ``[batch_size, self.z_dim]``
        d : torch.Tensor
            Posterior diagonal factor, as defined above. Shape:
            ``[batch_size, self.z_dim]``
        """
        x = x.unsqueeze(1)
        x = F.relu(self.v_conv1(self.v_bn1(x)))
        x = F.relu(self.v_conv2(self.v_bn2(x)))
        x = F.relu(self.v_conv3(self.v_bn3(x)))
        x = F.relu(self.v_conv4(self.v_bn4(x)))
        x = F.relu(self.v_conv5(self.v_bn5(x)))
        x = F.relu(self.v_conv6(self.v_bn6(x)))
        x = F.relu(self.v_conv7(self.v_bn7(x)))
        x = F.relu(self.v_conv8(self.v_bn8(x)))
        x = F.relu(self.v_conv9(self.v_bn9(x)))
        x = x.view(-1, 9600)
        x = F.relu(self.v_fc1(x))

        mu = self.v_fc21(x)
        u = self.v_fc22(x).unsqueeze(-1)
        d = torch.exp(self.v_fc23(x))

        return mu, u, d
    
    def decode_a(self, z):
        """
        Compute :math:`p(x|z)`.

        .. math:: p(x|z) = \mathcal{N}(\mu, \Lambda)

        .. math:: \Lambda = \mathtt{model\_precision} \cdot I

        where :math:`\mu` is a deterministic function of `z`, :math:`\Lambda` is
        a precision matrix, and :math:`I` is the identity matrix.

        Parameters
        ----------
        z : torch.Tensor
            Batch of latent samples with shape ``[batch_size, self.z_dim]``

        Returns
        -------
        x : torch.Tensor
            Batch of means mu, described above. Shape: ``[batch_size,
            X_DIM=128*128]``
        """
       
        z = F.relu(self.a_fc3(z))
        z = F.relu(self.a_fc4(z))
        z = z.view(-1,32,16,16)

        z = F.relu(self.a_convt1(self.a_bn8(z)))
        z = F.relu(self.a_convt2(self.a_bn9(z)))
        z = F.relu(self.a_convt3(self.a_bn10(z)))
        z = F.relu(self.a_convt4(self.a_bn11(z)))
        z = F.relu(self.a_convt5(self.a_bn12(z)))
        z = F.relu(self.a_convt6(self.a_bn13(z)))
        z = self.a_convt7(self.a_bn14(z))
        return z.view(-1, X_A_DIM)
    
    def decode_v(self, z):
        """
        Compute :math:`p(x|z)`.

        .. math:: p(x|z) = \mathcal{N}(\mu, \Lambda)

        .. math:: \Lambda = \mathtt{model\_precision} \cdot I

        where :math:`\mu` is a deterministic function of `z`, :math:`\Lambda` is
        a precision matrix, and :math:`I` is the identity matrix.

        Parameters
        ----------
        z : torch.Tensor
            Batch of latent samples with shape ``[batch_size, self.z_dim]``

        Returns
        -------
        x : torch.Tensor
            Batch of means mu, described above. Shape: ``[batch_size,
            X_DIM=128*128]``
        """
        z = F.relu(self.v_fc3(z))
        z = F.relu(self.v_fc4(z))
        z = z.view(-1,32,20,15)

        z = F.relu(self.v_convt1(self.v_bn10(z)))
        z = F.relu(self.v_convt2(self.v_bn11(z)))
        z = F.relu(self.v_convt3(self.v_bn12(z)))
        z = F.relu(self.v_convt4(self.v_bn13(z)))
        z = F.relu(self.v_convt5(self.v_bn14(z)))
        z = F.relu(self.v_convt6(self.v_bn15(z)))
        z = F.relu(self.v_convt7(self.v_bn16(z)))
        z = F.relu(self.v_convt8(self.v_bn17(z)))
        z = self.v_convt9(self.v_bn18(z))
        return z.view(-1, X_V_DIM)
    
    def forward(self, xa, xa_ns, xv, xv_ns, return_latent_rec=False):
        """
        Send `x` round trip and compute a loss.

        In more detail: Given `x`, compute :math:`q(z|x)` and sample:
        :math:`\hat{z} \sim q(z|x)` . Then compute :math:`\log p(x|\hat{z})`,
        the log-likelihood of `x`, the input, given :math:`\hat{z}`, the latent
        sample. We will also need the likelihood of :math:`\hat{z}` under the
        model's prior: :math:`p(\hat{z})`, and the entropy of the latent
        conditional distribution, :math:`\mathbb{H}[q(z|x)]` . ELBO can then be
        estimated as:

        .. math:: \\frac{1}{N} \sum_{i=1}^N \mathbb{E}_{\hat{z} \sim q(z|x_i)}
            \log p(x_i,\hat{z}) + \mathbb{H}[q(z|x_i)]

        where :math:`N` denotes the number of samples from the data distribution
        and the expectation is estimated using a single latent sample,
        :math:`\hat{z}`. In practice, the outer expectation is estimated using
        minibatches.

        Parameters
        ----------
        x : torch.Tensor
            A batch of samples from the data distribution (spectrograms).
            Shape: ``[batch_size, height=128, width=128]``
        return_latent_rec : bool, optional
            Whether to return latent means and reconstructions. Defaults to
            ``False``.

        Returns
        -------
        loss : torch.Tensor
            Negative ELBO times the batch size. Shape: ``[]````
        latent : numpy.ndarray, if `return_latent_rec`
            Latent means. Shape: ``[batch_size, self.z_dim]``
        reconstructions : numpy.ndarray, if `return_latent_rec`
            Reconstructed means. Shape: ``[batch_size, height=128, width=128]``
        """
        # h_v = self.trans_v(zv) 
        # h_v = torch.cat((h_v[0],h_v[1]),0) #reshape (BCxD)
        
        mu_a, u_a, d_a = self.encode_a(xa)
        latent_dist_a = LowRankMultivariateNormal(mu_a, u_a, d_a)
        za = latent_dist_a.rsample() #(24 x 64)
        za = torch.stack(torch.split(za, int(za.shape[0]/2))) #reshape to be (Batch x Context x D) (2x12x64)

        mu_v, u_v, d_v = self.encode_v(xv)
        latent_dist_v = LowRankMultivariateNormal(mu_v, u_v, d_v)
        zv = latent_dist_v.rsample() #(24 x 64)
        zv = torch.stack(torch.split(zv, int(zv.shape[0]/2))) #reshape to be (Batch x Context x D) (2x12x64)
        
        # print('zv', zv.shape)
        if self.aonly:
            # (V) --> A
            h_a_with_vs = self.trans_a_with_v(za, zv, zv)
            h_as = self.trans_a_mem(h_a_with_vs)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = torch.cat((h_as[0],h_as[1]),0)

        if self.vonly:
            # (A) --> V
            h_v_with_as = self.trans_v_with_a(zv, za, za)
            h_vs = self.trans_v_mem(h_v_with_as)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = torch.cat((h_vs[0],h_vs[1]),0)
        
        if self.partial_mode == 2:
            last_hs = torch.cat([last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        # last_hs_proj = torch.cat((last_hs_proj[0],last_hs_proj[1]),0)

        rec_a =  self.decode_a(last_hs_proj[:,:300]) #reconstruct original audio image
        rec_v =  self.decode_v(last_hs_proj[:,300:]) #reconstruct original video image        

        loss_a= torch.mean((xa_ns.view(xa_ns.shape[0],-1) - rec_a) ** 2)
        loss_v= torch.mean((xv_ns.view(xv_ns.shape[0],-1) - rec_v) ** 2)

        total_loss = loss_a + loss_v

        # loss= nn.MSELoss()
        # output = loss(x_rec, x_ns.view(x_ns.shape[0],-1))




        if return_latent_rec:
            return total_loss, za.detach().cpu().numpy(), zv.detach().cpu().numpy(), \
                rec_a.view(-1, X_A_SHAPE[0], X_A_SHAPE[1]).detach().cpu().numpy(), \
                rec_v.view(-1, X_V_SHAPE[0], X_V_SHAPE[1]).detach().cpu().numpy()
        return total_loss


    def train_epoch(self, loader):
        """
        Train the model for a single epoch.

        Parameters
        ----------
        train_loader : torch.utils.data.Dataloader
            ava.models.vae_dataset.SyllableDataset Dataloader for training set

        Returns
        -------
        elbo : float
            A biased estimate of the ELBO, estimated using samples from
            `train_loader`.
        """
        self.train()
        train_loss = 0.0

        for i_batch, data in enumerate(loader):
            sample_ind, batchX_text, batchX_audio, batchX_vision = data
            self.optimizer.zero_grad()
            batchX_vision=batchX_vision[0]
            batchX_audio=batchX_audio[0]

            audio=batchX_audio[0:-1,:].to(self.device)
            audio_ns=batchX_audio[1:,:].to(self.device)  
            vision=batchX_vision[0:-1,:].to(self.device)
            vision_ns=batchX_vision[1:,:].to(self.device)
          
            loss = self.forward(audio, audio_ns, vision, vision_ns)

            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(loader.dataset)

        if self.writer is not None:
            self.writer.add_scalar("Train/batch", train_loss, self.epoch)
        print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
                train_loss))
        self.epoch += 1
        return train_loss


    def test_epoch(self, loader, test=False):
        """
        Test the model on a held-out test set, return an ELBO estimate.

        Parameters
        ----------
        test_loader : torch.utils.data.Dataloader
            ava.models.vae_dataset.SyllableDataset Dataloader for test set

        Returns
        -------
        elbo : float
            An unbiased estimate of the ELBO, estimated using samples from
            `test_loader`.
        """
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i_batch, data in enumerate(loader):
                sample_ind, batchX_text, batchX_audio, batchX_vision = data
                batchX_vision=batchX_vision[0]
                batchX_audio=batchX_audio[0]

                audio=batchX_audio[0:-1,:].to(self.device)
                audio_ns=batchX_audio[1:,:].to(self.device)  
                vision=batchX_vision[0:-1,:].to(self.device)
                vision_ns=batchX_vision[1:,:].to(self.device)
                loss = self.forward(vision, vision_ns)
                test_loss += loss.item()
        test_loss /= len(loader.dataset)
        if self.writer is not None:
            self.writer.add_scalar("Test/batch", test_loss, self.epoch)
        print('Test loss: {:.4f}'.format(test_loss))
        return test_loss


    def train_loop(self, train_loader, test_loader, hyp_params, epochs=100,
                   test_freq=2, save_freq=10, vis_freq=1):
        """
        Train the model for multiple epochs, testing and saving along the way.

        Parameters
        ----------
        loaders : dictionary
            Dictionary mapping the keys ``'test'`` and ``'train'`` to respective
            torch.utils.data.Dataloader objects.
        epochs : int, optional
            Number of (possibly additional) epochs to train the model for.
            Defaults to ``100``.
        test_freq : int, optional
            Testing is performed every `test_freq` epochs. Defaults to ``2``.
        save_freq : int, optional
            The model is saved every `save_freq` epochs. Defaults to ``10``.
        vis_freq : int, optional
            Syllable reconstructions are plotted every `vis_freq` epochs.
            Defaults to ``1``.
        """
        print("="*40)
        print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
        print("Training set:", len(train_loader.dataset))
        print("Test set:", len(train_loader.dataset))
        # print("Test set:", len(test_loader.dataset))
        print("="*40)
        # For some number of epochs...
        for epoch in range(self.epoch, self.epoch+epochs):
            # Run through the training data and record a loss.

            loss = self.train_epoch(train_loader)
            self.loss['train'][epoch] = loss
            # Run through the test data and record a loss.
            # if (test_freq is not None) and (epoch % test_freq == 0):
            #     #loss = self.test_epoch(train_loader)
            #     loss = self.test_epoch(test_loader)
            #     self.loss['test'][epoch] = loss
            # Save the model.
            if (save_freq is not None) and (epoch % save_freq == 0) and \
                    (epoch > 0):
                filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
                self.save_state(filename)
            # Plot reconstructions.
            if (vis_freq is not None) and (epoch % vis_freq == 0):
                self.visualize(train_loader)
                # self.visualize(test_loader)

    def load_state(self, filename, save_dir=''):
        """
        Load all the model parameters from the given ``.tar`` file.

        The ``.tar`` file should be written by `self.save_state`.

        Parameters
        ----------
        filename : str
            File containing a model state.

        Note
        ----
        - `self.lr`, `self.save_dir`, and `self.z_dim` are not loaded.
        """
        
        checkpoint = torch.load(filename, map_location=self.device)
        
        layers = self._get_layers()
        for layer_name in layers:
            layer = layers[layer_name]
            layer.load_state_dict(checkpoint[layer_name])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.loss = checkpoint['loss']
        self.epoch = checkpoint['epoch']

        param_filename= os.path.join(save_dir, 'parameters.npy')
        self.hyp_params = np.load(param_filename, allow_pickle=True).tolist()
        self.z_dim = self.hyp_params.z_dim

    def save_state(self, filename):
        """Save all the model parameters to the given file."""
        layers = self._get_layers()
        

        state = {}
        for layer_name in layers:
            state[layer_name] = layers[layer_name].state_dict()
        state['optimizer_state'] = self.optimizer.state_dict()
        state['loss'] = self.loss
        state['z_dim'] = self.z_dim
        state['epoch'] = self.epoch
        state['lr'] = self.lr
        state['save_dir'] = self.save_dir
        state['hyp_params'] = self.hyp_params
        filename = os.path.join(self.save_dir, filename)
        param_filename = os.path.join(self.save_dir, 'parameters')
        torch.save(state, filename)
        np.save(param_filename,self.hyp_params)

    def check_latents(self, x):

        with torch.no_grad():
            mu, u, d = self.encode(x)
            latent_dist = LowRankMultivariateNormal(mu, u, d)
            z = latent_dist.rsample() #(24 x 64) (BCxD)

            z_trans = torch.stack(torch.split(z, int(z.shape[0]/2))) #reshape to be (Batch x Context x D) (2x12x64)

            h_v = self.trans_v(z_trans)
            h_v = torch.cat((h_v[0],h_v[1]),0) #reshape (BCxD)
            # z = torch.cat((z[0], z[1]),0)
        return(z.detach().cpu().numpy(), h_v.detach().cpu().numpy())
        


    def visualize(self, loader, num_specs=5, gap=(2,6), \
        a_save_filename='auditory_reconstruction.pdf', v_save_filename='visual_reconstruction.pdf'):
        """
        Plot spectrograms and their reconstructions.

        Spectrograms are chosen at random from the Dataloader Dataset.

        Parameters
        ----------
        loader : torch.utils.data.Dataloader
            Spectrogram Dataloader
        num_specs : int, optional
            Number of spectrogram pairs to plot. Defaults to ``5``.
        gap : int or tuple of two ints, optional
            The vertical and horizontal gap between images, in pixels. Defaults
            to ``(2,6)``.
        save_filename : str, optional
            Where to save the plot, relative to `self.save_dir`. Defaults to
            ``'temp.pdf'``.

        Returns
        -------
        specs : numpy.ndarray
            Spectrograms from `loader`.
        rec_specs : numpy.ndarray
            Corresponding spectrogram reconstructions.
        """
        # Collect random indices.
        assert num_specs <= len(loader.dataset) and num_specs >= 1
        indices = int(np.random.choice(np.arange(len(loader.dataset)),
            size=1,replace=False))
        # print(len(loader.dataset))
        # Retrieve spectrograms from the loader.
        sample_ind, batchX_text, batchX_audio, batchX_vision = loader.dataset[indices]
        
        audio=batchX_audio[0:-1,:].to(self.device)
        audio_ns=batchX_audio[1:,:].to(self.device)  
        vision=batchX_vision[0:-1,:].to(self.device)
        vision_ns=batchX_vision[1:,:].to(self.device)

        # Get reconstructions.
        with torch.no_grad():
            _,_,_, aud_rec_specs, vis_rec_specs = self.forward(audio, audio_ns, vision, vision_ns, return_latent_rec=True)
        aud_specs = audio_ns.detach().cpu().numpy()
        aud_all_specs = np.stack([aud_rec_specs[0:num_specs], aud_specs[0:num_specs]])

        vis_specs = vision_ns.detach().cpu().numpy()
        vis_all_specs = np.stack([vis_rec_specs[0:num_specs], vis_specs[0:num_specs]])

        # Plot.
        a_save_filename = os.path.join(self.save_dir, a_save_filename)
        v_save_filename = os.path.join(self.save_dir, v_save_filename)
        self.grid_plot(aud_all_specs, gap=gap, filename=a_save_filename)
        self.grid_plot(vis_all_specs, gap=gap, filename=v_save_filename)
        return aud_specs, aud_rec_specs, vis_specs, vis_rec_specs

    def grid_plot(self, specs, gap=3, vmin=0.0, vmax=1.0, ax=None, save_and_close=True, \
        filename='temp.pdf'):
            """
            Parameters
            ----------
            specs : numpy.ndarray
                Spectrograms
            gap : int or tuple of two ints, optional
                The vertical and horizontal gap between images, in pixels. Defaults to
                `3`.
            vmin : float, optional
                Passed to matplotlib.pyplot.imshow. Defaults to `0.0`.
            vmax : float, optional
                Passed to matplotlib.pyplot.imshow. Defaults to `1.0`.
            ax : matplotlib.pyplot.axis, optional
                Axis to plot figure. Defaults to matplotlib.pyplot.gca().
            save_and_close : bool, optional
                Whether to save and close after plotting. Defaults to True.
            filename : str, optional
                Save the image here.
            """
            if type(gap) == type(4):
                gap = (gap,gap)
            try:
                a, b, c, d = specs.shape
            except:
                print("Invalid shape:", specs.shape, "Should have 4 dimensions.")
                quit()
            dx, dy = d+gap[1], c+gap[0]
            height = a*c + (a-1)*gap[0]
            width = b*d + (b-1)*gap[1]
            img = np.zeros((height, width))
            for j in range(a):
                for i in range(b):
                    img[j*dy:j*dy+c,i*dx:i*dx+d] = specs[-j-1,i]
            for i in range(1,b):
                img[:,i*dx-gap[1]:i*dx] = np.nan
            for j in range(1,a):
                img[j*dy-gap[0]:j*dy,:] = np.nan
            if ax is None:
                ax = plt.gca()
            ax.imshow(img, aspect='equal', interpolation='none',
                vmin=vmin, vmax=vmax)
            ax.axis('off')
            if save_and_close:
                plt.tight_layout()
                plt.savefig(filename)
                plt.close('all')

    def reconstruction_plot(self, loaders, a_save_filename='auditory_reconstruction.pdf', \
                            v_save_filename='visual_reconstruction.pdf'):
        """
        Train the model for multiple epochs, testing and saving along the way.

        Parameters
        ----------
        loaders : dictionary
            Dictionary mapping the keys ``'test'`` and ``'train'`` to respective
            torch.utils.data.Dataloader objects.
        filename :  str, optional
            Save the image here.
        """
        self.visualize(loaders)
