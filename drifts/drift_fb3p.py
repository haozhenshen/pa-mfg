import numpy as np
import torch as torch
import numpy as np

def drift_fb3p(pop, t, proc, config):
    k = np.array(config.k)
    t1 = config.t1
    t2 = config.t2
    nt = config.nt
    h = config.h
    pi = np.array(config.pi)
    gamma_cur = np.array(config.gamma_cur)
    gamma_prev = np.array(config.gamma_prev)
    zeta = np.array(config.zeta)
    eta_cur = [(pi[i]/gamma_cur[i]) / np.sum(pi/gamma_cur) for i in range(k)]
    eta_prev = [(pi[i]/gamma_prev[i]) / np.sum(pi/gamma_prev) for i in range(k)]

    if t < t1:
        sum_ubar_cur = 0
        sum_vbar_cur = 0
        sum_hbar_cur = 0
        sum_phibar_cur = 0
        for i in range(k):
            sum_ubar_cur += eta_cur[i] * torch.mean(proc['u'][i])
            sum_vbar_cur += eta_cur[i] * torch.mean(proc['v'][i])
            sum_hbar_cur += eta_cur[i] * torch.mean(proc['h'][i])
            sum_phibar_cur += eta_cur[i] * torch.mean(proc['phi'][i])
        s_cur = sum_ubar_cur + sum_vbar_cur + sum_hbar_cur - sum_phibar_cur
        s_prev = torch.zeros(s_cur.shape, device=s_cur.device)
        g = h[pop] + (proc['u'][pop] + proc['v'][pop] + proc['h'][pop] - proc['phi'][pop])/zeta[pop]
        gam_cur = (1/gamma_cur[pop]) * (proc['u'][pop] + proc['v'][pop] + proc['h'][pop] - proc['phi'][pop] - s_cur)
        gam_prev = torch.zeros(gam_cur.shape, device=gam_cur.device)
        
    elif t1 <= t < t2:
        sum_ybar_cur = 0
        sum_abar_cur = 0
        for i in range(k):
            sum_ybar_cur += eta_cur[i] * torch.mean(proc['y'][i])
            sum_abar_cur += eta_cur[i] * torch.mean(proc['a'][i])
        s_cur = sum_ybar_cur + sum_abar_cur
        
        sum_ybar_prev = 0
        sum_abar_prev = 0
        sum_lambar_prev = 0
        for i in range(k):
            sum_ybar_prev += eta_prev[i] * torch.mean(proc['y'][i])
            sum_abar_prev += eta_prev[i] * torch.mean(proc['a'][i])
            sum_lambar_prev += eta_prev[i] * torch.mean(proc['lam'][i])
        
        s_prev = sum_ybar_prev + sum_abar_prev - sum_lambar_prev

        g = h[pop] + (proc['y'][pop] + proc['a'][pop])/zeta[pop]
        gam_cur = (1/gamma_cur[pop]) * (proc['y'][pop] + proc['a'][pop] - s_cur)
        gam_prev = (1/gamma_prev[pop]) * (proc['y'][pop] + proc['a'][pop] - proc['lam'][pop] - s_prev)

    elif t2 <= t < nt:
        sum_nbar_cur = 0
        for i in range(k):
            sum_nbar_cur += eta_cur[i] * torch.mean(proc['n'][i])
        s_cur = sum_nbar_cur 
        
        sum_nbar_prev = 0
        for i in range(k):
            sum_nbar_prev += eta_prev[i] * torch.mean(proc['n'][i])
        s_prev = sum_nbar_prev
        
        g = h[pop] + proc['n'][pop]/zeta[pop]
        gam_cur = (1/gamma_cur[pop]) * (proc['n'][pop] - s_cur)
        gam_prev = (1/gamma_prev[pop]) * (proc['n'][pop] - s_prev)
        
    return s_cur, s_prev, g, gam_cur, gam_prev