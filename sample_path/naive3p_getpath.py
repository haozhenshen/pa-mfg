import torch as torch
from drifts.drift_naive3p import drift_naive3p
from utils import *
from drifts import *


def naive3p_getpath(dB, init_x, models, config):
    n = config.n
    k = config.k
    h = config.h
    t1 = config.t1
    t2 = config.t2
    nt = config.nt
    dt = config.t/(nt - 1)
    beta = config.beta
    sigma = config.sigma

    # helper for update paths
    #######################################################
    def update_path(t, paths, proc, x, t1=t1, t2=t2, nt=nt):
        if t < t1:
            paths['n_path'][i][:,j] = proc['n'][i].squeeze()
            paths['y_path'][i][:,j] = proc['y'][i].squeeze()
            paths['a_path'][i][:,j] = proc['a'][i].squeeze()
 
            paths['u_path'][i][:,j] = proc['u'][i].squeeze()
            paths['v_path'][i][:,j] = proc['v'][i].squeeze()
            paths['h_path'][i][:,j] = proc['h'][i].squeeze()

        elif t1 <= t < t2:
            paths['n_path'][i][:,j] = proc['n'][i].squeeze()
            paths['y_path'][i][:,j] = proc['y'][i].squeeze()
            paths['a_path'][i][:,j] = proc['a'][i].squeeze()

        elif t2 <= t < nt:
            paths['n_path'][i][:,j] = proc['n'][i].squeeze()
        paths['x_path'][i][:,j] = x[i].squeeze()
        paths['g_rental_path'][i][:,j] = proc['g_rental'][i].squeeze()
        paths['cum_g_rental_path'][i][:,j] = proc['cum_g_rental'][i].squeeze()
        paths['gam_path'][i][:,j] = proc['gam'][i].squeeze()
        paths['cum_gam_path'][i][:,j] = proc['cum_gam'][i].squeeze()
        paths['cum_gam_vol_path'][i][:,j] = proc['cum_gam_vol'][i].squeeze()
        paths['s_path'][i][:,j] =  proc['s'][i].squeeze()
        paths['expan_rate_path'][i][:,j] = proc['expan_rate'][i].squeeze()
        paths['cum_expan_rate_path'][i][:,j] = proc['cum_expan_rate'][i].squeeze()
        
    #######################################################
    
    paths = {key: [] for key in ['x_path','n_path', 'y_path','a_path',
                                 'u_path','v_path','h_path',
                                 'g_rental_path','cum_g_rental_path','gam_path',
                                 'cum_gam_path', 
                                 'cum_gam_vol_path', 's_path', 
                                 '_path', 'expan_rate_path', 'cum_expan_rate_path']}
    
    for key in paths.keys():
        mod_key = 'z_' + key[:-5] + '_models'
        if mod_key in models.keys():        
            for i in range(k):
                paths[key].append(torch.ones(n[i],len(models[mod_key][0]) + 1))
        else:
            for i in range(k):
                paths[key].append(torch.ones(n[i], nt))
        

    
    x = init_x
    
    proc = {key: [] for key in ['n','y', 'a', 'u','v','h',
                                'g_rental','cum_g_rental','gam',
                                'cum_gam','cum_gam_vol', 's',
                                'expan_rate', 'cum_expan_rate']}
    for key in proc.keys():
        mod_key = key + '_models'
        if mod_key in models.keys():        
            for i in range(k):
                proc[key].append(torch.sigmoid(models[mod_key][i](x[i])))
        else:
            for i in range(k):
                proc[key].append(torch.zeros(x[i].shape, device=x[0].device))

                
    # Initialize paths variables at time zero
    #########################################################################
    for i in range(k):   
        s, g, gam = drift_naive3p(i, 1, proc, config)

        paths['x_path'][i][:,0] = x[i].squeeze()
        paths['n_path'][i][:,0] = proc['n'][i].squeeze()
        paths['y_path'][i][:,0] = proc['y'][i].squeeze()
        paths['a_path'][i][:,0] = proc['a'][i].squeeze()
        paths['u_path'][i][:,0] = proc['u'][i].squeeze()
        paths['v_path'][i][:,0] = proc['v'][i].squeeze()
        paths['h_path'][i][:,0] = proc['h'][i].squeeze()

        
        paths['g_rental_path'][i][:,0] = (g-h[i]).squeeze()
        paths['cum_g_rental_path'][i][:,0] = ((g-h[i])*dt).squeeze()
        
        paths['gam_path'][i][:,0] = gam.squeeze()
 
        
        paths['cum_gam_path'][i][:,0] = (gam*dt).squeeze()
        
        paths['cum_gam_vol_path'][i][:,0] = (torch.abs(gam)*dt).squeeze()
        
        paths['s_path'][i][:,0] = s.squeeze()
        
        paths['expan_rate_path'][i][:,0] = (((proc['u'][i] + proc['v'][i] + proc['h'][i])*(t1) + 
                                             (proc['y'][i] + proc['a'][i])*(t2-t1) + proc['n'][i]*(nt-t2)) * dt/beta[i]).squeeze()
        paths['cum_expan_rate_path'][i][:,0] = proc['cum_expan_rate'][i].squeeze()
        
    ###############################################################################################
    # Initializing stochatic processes.
    period1_z_proc = ['z_u', 'z_v','z_h']
    period2_z_proc = ['z_y','z_a']
    period3_z_proc = ['z_n']
    z_proc = {key: [] for key in period1_z_proc + period2_z_proc + period3_z_proc} 
    
    
    for j in range(1, t1):
        for i in range(k):

            for key in period1_z_proc + period2_z_proc + period3_z_proc:
                z_proc[key] = models[key + '_models'][i][j-1](x[i])
            
            s, g, gam = drift_naive3p(i, j, proc, config)
            proc['s'][i] = s
            proc['gam'][i] = gam

            
            proc['expan_rate'][i] = ((proc['u'][i] + proc['v'][i] + proc['h'][i])*(t1 - (j-1)) + 
                                     (proc['y'][i] + proc['a'][i])*(t2-t1) + proc['n'][i]*(nt-t2)) * dt/beta[i]
            
            x[i] = x[i] + (g + gam  + proc['cum_expan_rate'][i]) * dt + sigma[i]*dB[i][:,j-1].view(-1,1)

            proc['cum_expan_rate'][i] = proc['cum_expan_rate'][i] + proc['expan_rate'][i] * dt
            
            for key in period1_z_proc + period2_z_proc + period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i] + z_proc[key]*(1-proc[key[2:]][i])*dB[i][:,j-1].view(-1,1)
                
            proc['g_rental'][i] = g - h[i]
            proc['cum_g_rental'][i] = proc['cum_g_rental'][i] + proc['g_rental'][i]*dt
            
            proc['cum_gam'][i] = proc['cum_gam'][i] + gam*dt
            
            proc['cum_gam_vol'][i] = proc['cum_gam_vol'][i] + torch.abs(gam)*dt    
  
         
                
            for key in period1_z_proc + period2_z_proc + period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i].clamp(0,1)
                
            # updatae path variables according to process     
            update_path(j, paths, proc, x)
            

    for j in range(t1, t2):
        for i in range(k):
            for key in period2_z_proc + period3_z_proc:
                z_proc[key] = models[key + '_models'][i][j-1](x[i])
            

            s, g, gam = drift_naive3p(i, j, proc, config)
            proc['s'][i] = s

            proc['gam'][i] = gam
            
            proc['expan_rate'][i] = ((proc['y'][i] + proc['a'][i])*(t2-(j-1)) + proc['n'][i]*(nt-t2)) *dt/beta[i]
            
            x[i] = x[i] + (g + gam  + proc['cum_expan_rate'][i]) * dt + sigma[i]*dB[i][:,j-1].view(-1,1)
            
            proc['cum_expan_rate'][i] = proc['cum_expan_rate'][i] + proc['expan_rate'][i] * dt
            
            for key in period2_z_proc + period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i] + z_proc[key]*(1-proc[key[2:]][i])*dB[i][:,j-1].view(-1,1)
                
            proc['g_rental'][i] = g - h[i]
            proc['cum_g_rental'][i] = proc['cum_g_rental'][i] + proc['g_rental'][i]*dt
            
            proc['cum_gam'][i] = proc['cum_gam'][i] + gam*dt
  
            
            proc['cum_gam_vol'][i] = proc['cum_gam_vol'][i] + torch.abs(gam)*dt           
                
            for key in period2_z_proc + period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i].clamp(0,1)
        
            # updatae path variables according to process     
            update_path(j, paths, proc, x)
            

    for j in range(t2, nt):
        for i in range(k):
            for key in period3_z_proc:
                z_proc[key] = models[key + '_models'][i][j-1](x[i])

            s, g, gam = drift_naive3p(i, j, proc, config)
            proc['s'][i] = s
 
            proc['gam'][i] = gam
            
            proc['expan_rate'][i] = (proc['n'][i]*(nt-(j-1)))*dt/beta[i]
            
            x[i] = x[i] + (g + gam  + proc['cum_expan_rate'][i]) * dt + sigma[i]*dB[i][:,j-1].view(-1,1)
            proc['cum_expan_rate'][i] = proc['cum_expan_rate'][i] + proc['expan_rate'][i] * dt
            
            for key in period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i] + z_proc[key]*(1-proc[key[2:]][i])*dB[i][:,j-1].view(-1,1)
            
            proc['g_rental'][i] = g - h[i]
            proc['cum_g_rental'][i] = proc['cum_g_rental'][i] + proc['g_rental'][i]*dt
            
            proc['cum_gam'][i] = proc['cum_gam'][i] + gam*dt

            
            proc['cum_gam_vol'][i] = proc['cum_gam_vol'][i] + torch.abs(gam)*dt         
                
            for key in period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i].clamp(0,1)
            
            # updatae path variables according to process     
            update_path(j, paths, proc, x)
            
    return paths