#code for centralized version of scp 
#init all solutions: 
import numpy as np

def initAllSolutions(po, pf, h, K):
    po = np.squeeze(po)
    pf = np.squeeze(pf)
    N = po.shape[1]
    t = np.arange(0, K*h, h)
    p = np.zeros((po.shape[0], len(t), N))
    for i in range(N):
        diff = pf[:,i] - po[:,i]
        for k in range(len(t)):
            p[:,k,i] = po[:,i] + t[k]*diff/((K-1)*h)
    return p
