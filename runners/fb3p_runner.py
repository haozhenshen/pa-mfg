import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from models.fbsde_net import Network
from losses import mse
from sample_path import get_path
from utils import *


class Runner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def get_models(self):
        layer_dim = self.config.model.layer_dim
        k = self.config.model.k
        t1 = self.config.model.t1
        t2 = self.config.model.t2
        nt = self.config.model.nt
        models = {key: [] for key in ['n_models','y_models', 'a_models', 'lam_models', 'u_models', 
                                    'v_models', 'h_models', 'phi_models', 'z_n_models', 'z_y_models', 
                                    'z_a_models', 'z_lam_models','z_u_models', 'z_v_models', 'z_h_models',
                                    'z_phi_models']}
        
        for _ in range(k):
            n_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device)
            y_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device)
            a_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device)
            lam_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device)
            u_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device)
            v_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device)
            h_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device)
            phi_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device)
            z_n_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device) for i in range(0, nt-1)]
            z_y_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device) for i in range(0, t2-1)]
            z_a_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device) for i in range(0, t2-1)]
            z_lam_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device) for i in range(0, t2-1)]
            z_u_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device) for i in range(0, t1-1)]
            z_v_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device) for i in range(0, t1-1)]
            z_h_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device) for i in range(0, t1-1)]
            z_phi_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(self.config.device) for i in range(0, t1-1)]

            models["n_models"].append(n_net)
            models["y_models"].append(y_net)
            models["a_models"].append(a_net)
            models["lam_models"].append(lam_net)
            models["u_models"].append(u_net)
            models["v_models"].append(v_net)
            models["h_models"].append(h_net)
            models["phi_models"].append(phi_net)
            models["z_n_models"].append(z_n_nets)
            models["z_y_models"].append(z_y_nets)
            models["z_a_models"].append(z_a_nets)
            models["z_lam_models"].append(z_lam_nets)
            models["z_u_models"].append(z_u_nets)
            models["z_v_models"].append(z_v_nets)
            models["z_h_models"].append(z_h_nets)
            models["z_phi_models"].append(z_phi_nets)
        return models

    def train(self):
        models = self.get_models()
        params = get_foward_net_params(models)
        #Set up optimizer
        optimizer = self.get_optimizer(params)

        if self.args.resume_training:
            pass

        #Generate an initial batch
        dB = SampleBMIncr(self.config.model)
        init_x = sample_mu(self.config.model)

        loss_lst = []

        pbar = trange(0, self.config.training.n_epochs, desc="BSDE_training", unit="Epoch")
        for k in pbar:
            sloss=0
            if k % self.config.training.plot_freq == 0:
                plot_path(k, models, get_path(dB, init_x, models, self.config.model), self.args, self.config.model)
            for _ in range(0, self.config.training.optim_steps):
                optimizer.zero_grad()
                loss = mse(dB, init_x, self.config.model.delta, models, self.config.model)
                loss.backward()
                optimizer.step()
                nloss = loss.to('cpu').detach().numpy()
                sloss += nloss
                #print('OptimStep: '+ str(l+1))
                #print('forward_loss: ' + str(nloss))
                pbar.set_postfix({'nloss': nloss})

            avgloss = sloss/(self.config.training.optim_steps * self.config.model.k)
            loss_lst.append(avgloss)
            pbar.set_postfix({'avg_loss': avgloss})
            if k % 20 == 0:
                save_model(models, k, self.config.model.delta)
        save_model(models, self.config.training.n_epochs, self.config.model.delta)