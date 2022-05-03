import numpy as np
import torch as torch
import numpy as np

def drift_naive3p(pop, t, proc, config):
    k = np.array(config.k)
    t1 = config.t1
    t2 = config.t2
    nt = config.nt
    h = config.h
    pi = np.array(config.pi)
    gamma = np.array(config.gamma)

    zeta = np.array(config.zeta)
    eta = [(pi[i]/gamma[i]) / np.sum(pi/gamma) for i in range(k)]


    if t < t1:
        sum_ubar = 0
        sum_vbar = 0
        sum_hbar = 0
        for i in range(k):
            sum_ubar += eta[i] * torch.mean(proc['u'][i])
            sum_vbar += eta[i] * torch.mean(proc['v'][i])
            sum_hbar += eta[i] * torch.mean(proc['h'][i])

        s = sum_ubar + sum_vbar + sum_hbar 
 
        g = h[pop] + (proc['u'][pop] + proc['v'][pop] + proc['h'][pop])/zeta[pop]
        gam = (1/gamma[pop]) * (proc['u'][pop] + proc['v'][pop] + proc['h'][pop] - s)

        
    elif t1 <= t < t2:
        sum_ybar = 0
        sum_abar = 0
        for i in range(k):
            sum_ybar += eta[i] * torch.mean(proc['y'][i])
            sum_abar += eta[i] * torch.mean(proc['a'][i])
        s = sum_ybar + sum_abar
        
     

        g = h[pop] + (proc['y'][pop] + proc['a'][pop])/zeta[pop]
        gam = (1/gamma[pop]) * (proc['y'][pop] + proc['a'][pop] - s)
 

    elif t2 <= t < nt:
        sum_nbar = 0
        for i in range(k):
            sum_nbar += eta[i] * torch.mean(proc['n'][i])
        s = sum_nbar 
        
     
        g = h[pop] + proc['n'][pop]/zeta[pop]
        gam = (1/gamma[pop]) * (proc['n'][pop] - s)
  
        
    return s, g, gam