#%%
import numpy as np
import matplotlib.pyplot as plt
from src import edc_gen,tvec
#%%
_dB = lambda x: 10*np.log10(x+np.spacing(1))
def herm(A):
    return A.conj().T
def svd(A):
    U,S,V=np.linalg.svd(A)
    return U,S,herm(V)


def time2idx(t,val):
    return abs(t-val).argmin()

def pseudo_inv(A):
    U,S,Vh = np.linalg.svd(A,full_matrices=False)
    return (U*(1/S))@Vh
#%%

fs = 44100
NFFT = 2**(20)
t = tvec(NFFT,fs)

t_trunc=7
t_mask = slice(time2idx(t,t_trunc))
t_div = slice(None,None,256)

TR_values = [2,9]
TR_amplitudes = [.7,.3]
EDC = edc_gen(TR_values,TR_amplitudes,.00,t)[:-1,:].sum(axis=0)[t_mask][t_div]
t = t[t_mask][t_div]

dt = t[1]-t[0]
print(f'dt: {dt*1E3:.3f} ms, T={t[-1]:.3f}, N={len(t):E}')
print(f'{(t.size**2)*(np.finfo(float).bits/8)/(2**20):.2f} MiB')

# %%
N = len(EDC)
L = int(N/3)

y_idx = np.indices((N-L-1, L)).sum(axis=0)
Y = EDC[y_idx]
U,s,V = svd(Y)

max_precision =1E-2
S_norm = s/s[0]
M = abs(S_norm-max_precision).argmin()
print(f'Number of exponentials: {M}')

V_trunc=V[:,:M]
V1 = V_trunc[:-1,:]
V2=V_trunc[1:,:]
z = np.linalg.eigvals(herm(V2)@V1)
ai = (np.log(z)/dt).real # Re{si} = alpha <- fator de amortecimento
RT = np.log(1E-6)/ai
print(f'TR: {RT}')

# %%
