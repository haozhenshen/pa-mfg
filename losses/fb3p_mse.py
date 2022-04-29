import torch as torch
import torch.nn.functional as F
from utils import *

def mse(dB, init_x, delta, models, config):
    k = config.k
    t1 = config.t1
    t2 = config.t2
    nt = config.nt
    dt = config.t/(nt - 1)
    r = config.r
    beta = config.beta
    sigma = config.sigma


    x = init_x.copy()
    proc = {key: [] for key in ['n','y', 'a', 'lam', 'u', 
                                  'v', 'h', 'phi']}
    for key in proc.keys():
        for i in range(k):
            proc[key].append(torch.sigmoid(models[key + '_models'][i](x[i])))

    
    c = [torch.zeros(proc['y'][i].shape, device=proc['y'][0].device) for i in range(k)]
    theta = [torch.zeros(proc['y'][i].shape, device=proc['y'][0].device) for i in range(k)]
    
    period1_z_proc = ['z_u', 'z_v','z_h','z_phi']
    period2_z_proc = ['z_y','z_a','z_lam']
    period3_z_proc = ['z_n']
    z_proc = {key: [] for key in period1_z_proc + period2_z_proc + period3_z_proc} 
    
    for j in range(1, t1):
        for i in range(k):
            for key in period1_z_proc + period2_z_proc + period3_z_proc:
                z_proc[key] = models[key + '_models'][i][j-1](x[i])

            s_cur, s_prev, g, gam_cur, gam_prev = drift(i, j, proc, config)
            
            a = ((proc['u'][i] + proc['v'][i] + proc['h'][i] - proc['phi'][i])*(t1 - (j-1)) + 
                 (proc['y'][i] + proc['a'][i])*(t2-t1) + proc['n'][i]*(nt-t2)) * dt/beta[i]
            
            x[i] = x[i] + (g + gam_cur + gam_prev + c[i]) * dt + sigma[i]*dB[i][:,j-1].view(-1,1)

            c[i] = c[i] + a * dt
            
            for key in period1_z_proc + period2_z_proc + period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i] + z_proc[key]*(1-proc[key[2:]][i])*dB[i][:,j-1].view(-1,1)
            for key in period1_z_proc + period2_z_proc + period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i].clamp(0,1)
                
     
    x_t1 = x.copy()
    y_t1 = proc['y'].copy()
    a_t1 = proc['a'].copy()
    lam_t1 = proc['lam'].copy()
    for j in range(t1, t2):
        for i in range(k):
            for key in period2_z_proc + period3_z_proc:
                z_proc[key] = models[key + '_models'][i][j-1](x[i])
            

            s_cur, s_prev, g, gam_cur, gam_prev = drift(i, j, proc, config)
            
            a = ((proc['y'][i] + proc['a'][i])*(t2-(j-1)) + proc['n'][i]*(nt-t2)) *dt/beta[i]
            
            x[i] = x[i] + (g + gam_cur + gam_prev + c[i]) * dt + sigma[i]*dB[i][:,j-1].view(-1,1)
            
            c[i] = c[i] + a * dt
            theta[i] = theta[i] + gam_prev*dt
            
            for key in period2_z_proc + period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i] + z_proc[key]*(1-proc[key[2:]][i])*dB[i][:,j-1].view(-1,1)
            for key in period2_z_proc + period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i].clamp(0,1)
                
    x_t2 = x.copy()
    theta_t2 = theta.copy()
    n_t2 = proc['n'].copy()
    for j in range(t2, nt):
        for i in range(k):
            for key in period3_z_proc:
                z_proc[key] = models[key + '_models'][i][j-1](x[i])

            s_cur, s_prev, g, gam_cur, gam_prev = drift(i, j, proc, config)
            a = (proc['n'][i]*(nt-(j-1)))*dt/beta[i]
            
            x[i] = x[i] + (g + gam_cur + gam_prev + c[i]) * dt + sigma[i]*dB[i][:,j-1].view(-1,1)
            c[i] = c[i] + a * dt
            theta[i] = theta[i] + gam_prev*dt
            
            for key in period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i] + z_proc[key]*(1-proc[key[2:]][i])*dB[i][:,j-1].view(-1,1)
            for key in period3_z_proc:
                proc[key[2:]][i] = proc[key[2:]][i].clamp(0,1)
      
    x_t3 = x.copy()
    
    
    loss = 0
    for i in range(k):
        n_loss = torch.mean((proc['n'][i]- F_prime(r - x_t3[i] + x_t2[i] - F(x_t2[i]-x_t1[i] + F(x_t1[i] - r,delta) - r, delta) 
                                                   + F(F(x_t1[i]-r,delta) + theta_t2[i] -r,delta) , delta))**2)
        
        y_loss = torch.mean((proc['y'][i]- F_prime(r - x_t2[i] + x_t1[i] - F(x_t1[i]-r,delta), delta))**2)

        a_loss = torch.mean((proc['a'][i]- n_t2[i] * F_prime(x_t2[i] - x_t1[i] + F(x_t1[i]-r,delta) -r, delta))**2)
        
        lam_loss = torch.mean((proc['lam'][i]- n_t2[i] * F_prime(F(x_t1[i]-r,delta) + theta_t2[i] - r, delta))**2)
        
        u_loss = torch.mean((proc['u'][i]- F_prime(r - x_t1[i], delta))**2)
        
        v_loss =  torch.mean((proc['v'][i]- y_t1[i] * F_prime(x_t1[i] - r, delta))**2)
        
        h_loss = torch.mean((proc['h'][i]- a_t1[i] * F_prime(x_t1[i] - r, delta))**2)
        
        phi_loss = torch.mean((proc['phi'][i]- lam_t1[i] * F_prime(x_t1[i] - r, delta))**2)
        
        loss += (1/8) * (n_loss + y_loss + a_loss + lam_loss + u_loss + v_loss + h_loss + phi_loss)
        
    return loss