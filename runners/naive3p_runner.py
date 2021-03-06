import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from models.fbsde_net import Network
from losses import *
from sample_path.naive3p_getpath import naive3p_getpath
from utils import *


class Naive3p_Runner():
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
        models = {key: [] for key in ['n_models','y_models', 'a_models', 'u_models', 
                                    'v_models', 'h_models', 'z_n_models', 'z_y_models', 
                                    'z_a_models','z_u_models', 'z_v_models', 'z_h_models']}

        if not self.args.test:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        for _ in range(k):
            n_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device)
            y_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device)
            a_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device)
    
            u_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device)
            v_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device)
            h_net = Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device)
            z_n_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device) for i in range(0, nt-1)]
            z_y_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device) for i in range(0, t2-1)]
            z_a_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device) for i in range(0, t2-1)]      
            z_u_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device) for i in range(0, t1-1)]
            z_v_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device) for i in range(0, t1-1)]
            z_h_nets = [Network(input_dims=[1], fc1_dims=layer_dim, fc2_dims=layer_dim,
                        n_outputs=1).to(device) for i in range(0, t1-1)]
            
            models["n_models"].append(n_net)
            models["y_models"].append(y_net)
            models["a_models"].append(a_net)
     
            models["u_models"].append(u_net)
            models["v_models"].append(v_net)
            models["h_models"].append(h_net)
    
            models["z_n_models"].append(z_n_nets)
            models["z_y_models"].append(z_y_nets)
            models["z_a_models"].append(z_a_nets)
    
            models["z_u_models"].append(z_u_nets)
            models["z_v_models"].append(z_v_nets)
            models["z_h_models"].append(z_h_nets)

        return models

    def train(self):
        models = self.get_models()
        params = get_foward_net_params(models)
        #Set up optimizer
        optimizer = self.get_optimizer(params)
        current_batch = 0
        if int(self.args.resume_training) > 0:
            try:
                load_model(models, self.args.resume_training, self.config.model.delta, self.args, self.config.model)
                current_batch = int(self.args.resume_training)
                print(f'Pretrained model found, loaded from epoch {self.args.resume_training} with delta = {self.config.model.delta}.')
            except:
                print("No pretrained model found. Training from scratch.")

        #Generate an initial batch
        if current_batch != self.config.training.n_epochs:
            loss_lst = []

            pbar = trange(current_batch, self.config.training.n_epochs, desc="BSDE_training", unit="Epoch")
            for k in pbar:
                dB = SampleBMIncr(self.args, self.config.model)
                init_x = sample_mu(self.args, self.config.model)
                sloss=0
                if k % self.config.training.plot_freq == 0:
                    plot_path(k, models, naive3p_getpath, self.args, self.config.model)
                for _ in range(0, self.config.training.optim_steps):
                    optimizer.zero_grad()
                    loss = naive3p_mse(dB, init_x, self.config.model.delta, models, self.config.model)
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
                if k % 50 == 0 and k != 0:
                    save_model(models, k, self.config.model.delta, self.args, self.config.model)
            save_model(models, self.config.training.n_epochs, self.config.model.delta, self.args, self.config.model)

    def test(self):
        models = self.get_models()
        #Set up optimizer
        loaded = False
        try:
            load_model(models, self.config.training.n_epochs, self.config.model.delta, self.args, self.config.model)
            loaded = True
            print(f'Pretrained model found, loaded from epoch {self.config.training.n_epochs} with delta = {self.config.model.delta}.')
            print(f'Creating process plots.')
        except:
            print("Testing failed. No pretrained model found.")


        if loaded:
            plots = {key: [] for key in ['x_plot','n_plot', 'y_plot','a_plot',
                                        'u_plot','v_plot','h_plot',
                                        'g_rental_plot','cum_g_rental_plot','gam_plot',
                                        'cum_gam_plot',
                                        'cum_gam_vol_plot', 's_plot', 
                                        'expan_rate_plot', 'cum_expan_rate_plot']}

            for count in range(self.args.num_test_batch):
   
                dB = SampleBMIncr(self.args, self.config.model)
                init_x =  sample_mu(self.args, self.config.model)
                paths = naive3p_getpath(dB, init_x, models, self.config.model)
                for key in plots.keys():
                    for i in range(self.config.model.k):
                        if count == 0:
                            plots[key].append(paths[key[:-5] + '_path'][i].to("cpu").detach().numpy())
                        else:
                            plots[key][i] = np.concatenate((plots[key][i], paths[key[:-5] + '_path'][i].to("cpu").detach().numpy()), 0)
            

            for key in plots.keys():
                plt.clf()
                double_plot(plots[key][0], plots[key][1], self.args.log + f'/plots/{key}.png', self.args, self.config.model, title=f'{key} Paths')
                plt.clf()
                double_plot(plots[key][0], plots[key][1], self.args.log + f'/plots/{key}_CI.png', 
                            self.args, self.config.model, ci=True, title=f'{key} Paths 95% CI')
                plt.clf()