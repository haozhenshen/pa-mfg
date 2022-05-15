import numpy as np
import torch as torch
import matplotlib.pyplot as plt
import seaborn as sns


def sample_mu(args, config):
    v = config.v
    m = config.m
    n = config.n
    k = config.k
    if not args.test:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return [(v[i] + m[i] * torch.randn(n[i])).view(-1,1).to(device) for i in range(k)]


def SampleBMIncr(args, config):
    # Returns Matrix of Dimension Npaths x Nsteps With Sample Increments of of BM
    # Here an increment is of the form dBt
    n = config.n
    nt = config.nt - 1
    t = config.t
    k = config.k
    if not args.test:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return [torch.FloatTensor(np.sqrt(t / nt) * np.random.randn(n[i], nt)).to(device) for i in range(k)]


def get_foward_net_params(models):
    params = []
    for key in models.keys():
        for nets in models[key]:
            if type(nets) == list:
                for net in nets:
                    params += list(net.parameters())
            else:
                params += list(nets.parameters())
    return params

 
def F(x, delta):
    out = ((x+delta)**2)/(4*delta)
    out[x < -delta] = 0
    out[x > delta] = x[x > delta]
    return out   
            
    
def F_prime(x, delta):
    return torch.clamp((x+delta)/(2*delta),min=0,max=1)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# def plot_path(batch, models, get_path, args, config):
#     n = config.n
#     k = config.k
#     t = config.t
#     nt = config.nt
#     dB = SampleBMIncr(args, config)
#     init_x =  sample_mu(args, config)
#     paths = get_path(dB, init_x, models, config)

#     for i in range(k):
#         number_of_paths = 64
#         idx_list = np.random.choice(n[i], number_of_paths, replace = False)
#         if config.name == "fb3p":
#             plots = {key: [] for key in ['x_plot','n_plot', 'y_plot','a_plot',
#                                         'lam_plot','u_plot','v_plot','h_plot','phi_plot',
#                                         'g_rental_plot','cum_g_rental_plot','gam_cur_plot',
#                                         'gam_prev_plot','cum_gam_cur_plot','cum_gam_prev_plot', 
#                                         'cum_gam_cur_vol_plot', 'cum_gam_prev_vol_plot', 's_cur_plot', 
#                                         's_prev_plot', 'expan_rate_plot', 'cum_expan_rate_plot', 'theta_plot']}
#         elif config.name == "naive3p":
#             plots = {key: [] for key in ['x_plot','n_plot', 'y_plot','a_plot',
#                                         'u_plot','v_plot','h_plot',
#                                         'g_rental_plot','cum_g_rental_plot','gam_plot',
#                                         'cum_gam_plot',
#                                         'cum_gam_vol_plot', 's_plot', 
#                                         'expan_rate_plot', 'cum_expan_rate_plot']}
#         for key in plots.keys():
#             plots[key] = paths[key[:-5] + '_path'][i].detach().numpy()[idx_list]
        
#         t_nt = np.array([i for i in range(0, nt)]) * t/(nt-1)
        
#         cmap = get_cmap(len(plots))
#         color = 0
#         for key in plots.keys():
#             mod_key = 'z_' + key[:-5] + '_models'
#             if mod_key in models.keys():
#                 time_steps = len(models[mod_key][0]) + 1
#                 t_t = np.array([i for i in range(0, time_steps)]) * t/(nt-1)
#                 for s in range(number_of_paths):
#                         plt.plot(t_t, plots[key][s], color=cmap(color), alpha=0.5)
#             else:
#                 for s in range(number_of_paths):
#                         plt.plot(t_nt, plots[key][s], color=cmap(color), alpha=0.5)
#             color += 1
#             plt.title(key.strip('_plot') + f' Batch_{batch}')
#             plt.savefig(args.log + f'/paths/' + key[:-5] + f'_pop_{i + 1}_batch_{batch}.png')
#             plt.close() 
        
      
        
        
# def create_gif_for_pop_k(k, args, config):
#     if config.name == "fb3p":
#         filenames = {key: [] for key in ['x', 'n','y', 'a', 'u','v','h','phi',
#                                     'g_rental','cum_g_rental','gam_cur',
#                                     'gam_prev','cum_gam_cur','cum_gam_prev', 
#                                     'cum_gam_cur_vol', 'cum_gam_prev_vol', 's_cur', 
#                                     's_prev', 'expan_rate', 'cum_expan_rate', 'theta']}
#     elif config.name == "naive3p":
#         filenames = {key: [] for key in ['x', 'n','y', 'a', 'u','v','h',
#                                     'g_rental','cum_g_rental','gam',
#                                     'cum_gam',
#                                     'cum_gam_vol' , 's', 
#                                     'expan_rate', 'cum_expan_rate']}
    
#     for i in range(0, config.training.n_epochs, config.training.plot_freq):
#         # create file name and append it to a list
        
#         for key in filenames.keys():
#             filenames[key].append(args.log + 'paths/' + key + f'_pop_{k}_batch_{i}.png')
        

#     for key in filenames.keys():
#         with imageio.get_writer(args.log + 'paths/' + key + f'_pop_{k}.gif', mode='I', duration=0.75) as writer:
#             for filename in filenames[key]:
#                 image = imageio.imread(filename)
#                 writer.append_data(image)


def CI(paths):
    return np.percentile(paths,5, axis=0), np.percentile(paths, 95, axis=0)

def double_plot(path0, path1, loc, args, config, ci=False, price=False, xlab='Time', ylab=None, title=None):

    num_path = 64
    idx_list = np.random.choice(config.n[0] * args.num_test_batch, num_path, replace = False)
    
    t = path0.shape[1]
    t_nt = np.array([i for i in range(0, t)]) * config.t/(config.nt-1)
    
    if price:
        for s in range(num_path):
            if s==0:
                plt.plot(t_nt,path0[idx_list][s], color="blue", alpha=0.5, label = "Current Price")      
                plt.plot(t_nt,path1[idx_list][s], color="red", alpha=0.5, label ="Previous Price")
            else:
                plt.plot(t_nt,path0[idx_list][s], color="blue", alpha=0.5)
                plt.plot(t_nt,path1[idx_list][s], color="red", alpha=0.5)
        plt.xlabel(xlab)
        plt.title(title)
        plt.legend()
        plt.savefig(loc, dpi=600)
    elif ci:
        for s in range(num_path):
            if s==0:
                plt.plot(t_nt,path0[idx_list][s], color="blue", alpha=0.05)      
                plt.plot(t_nt,path1[idx_list][s], color="red", alpha=0.05)
            else:
                plt.plot(t_nt,path0[idx_list][s], color="blue", alpha=0.05)
                plt.plot(t_nt,path1[idx_list][s], color="red", alpha=0.05)
        plt.fill_between(t_nt, CI(path0)[0], CI(path0)[1], color='b', alpha=0.5, label = "Population 0")
        plt.fill_between(t_nt, CI(path1)[0], CI(path1)[1], color='r', alpha=0.5, label ="Population 1")
        plt.xlabel(xlab)
        plt.title(title)
        plt.legend()
        plt.savefig(loc, dpi=600)
    else:
        for s in range(num_path):
            if s==0:
                plt.plot(t_nt,path0[idx_list][s], color="blue", alpha=0.5, label = "Population 0")      
                plt.plot(t_nt,path1[idx_list][s], color="red", alpha=0.5, label ="Population 1")
            else:
                plt.plot(t_nt,path0[idx_list][s], color="blue", alpha=0.5)
                plt.plot(t_nt,path1[idx_list][s], color="red", alpha=0.5)
        plt.xlabel(xlab)
        plt.title(title)
        plt.legend()
        plt.savefig(loc, dpi=600)


def plot_histogram(path, time_stamps, loc, config, title=None):
    for k in range(config.k):
        for i in range(0,time_stamps):
            ax = sns.distplot(path[k][:,26*i], label = 'step ' + str(i*26))
            ax.legend()
        plt.title(title + f' (Population {k})')
        plt.savefig(loc, dpi=600)
        plt.clf()


def save_model(models, batch, delta, args, config):
    for i in range(config.k):
        for key in models.keys():
            if type(models[key][i]) == list:
                for j in range(len(models[key][i])):
                    torch.save(models[key][i][j].state_dict(), args.log + '/saved_models/'+ key + f'_pop{i+1}_time{j}_batch_{batch}_delta_{delta}.pt')
            else:
                torch.save(models[key][i].state_dict(), args.log + '/saved_models/'+ key + f'_pop{i+1}_batch_{batch}_delta_{delta}.pt')
                
def load_model(models, batch, delta, args, config):
    for i in range(config.k):
        for key in models.keys():
            if type(models[key][i]) == list:
                for j in range(len(models[key][i])):
                    models[key][i][j].load_state_dict(torch.load(args.log + '/saved_models/' + key + f'_pop{i+1}_time{j}_batch_{batch}_delta_{delta}.pt'))
            else:
                models[key][i].load_state_dict(torch.load(args.log + '/saved_models/' + key + f'_pop{i+1}_batch_{batch}_delta_{delta}.pt'))