#%%
import numpy as np
import matplotlib.pyplot as plt
from src import edc_synth,tvec,MatrixPencil,DataPreparer
_dB = lambda x: 10*np.log10(x + np.finfo(float).eps)
#%%
from tqdm import tqdm
fs = 44100
NFFT = 2**(20)

#%%
# Notas gerais:
# - Levar em conta que as amplitudes devem 
#   ser sorteadas em escala log
# - Avaliar o termo ruído


#### --- Hyperparameters 
# MatrixPencil
p = -5
pencil_param = 1/2

# EDC preprocessing
trunc_s = 5
step_s = 10E-3
#### --- Hyper

t = tvec(NFFT,fs)
n_exponentials= 2
TR_values = [3.28036303, 7.54766463]
TR_amplitudes=[0.4, 1E-2]
TR_amplitudes /= np.mean(TR_amplitudes)

exp_true = edc_synth(TR_values,TR_amplitudes,t,60)
EDC_true = exp_true[:,:].sum(axis=0)
data = (DataPreparer(EDC_true,t,fs,
                    verbose=False )
                    .truncate_time(max_time=trunc_s,
                                   step=step_s))
matpen = (
    MatrixPencil(p=p,
                 pencil_param=pencil_param,
                 verbose=True)
    .fit(data.X,data.dt))

EDC_estm = edc_synth(matpen.RT,matpen.amplitudes,t,0)
EDC_estm_full = EDC_estm[:-1,:].sum(axis=0)

plt.subplot(211)
plt.plot(t,_dB(EDC_true),color='black',alpha=.5,label='EDC')
plt.plot(t,_dB(EDC_estm_full),linestyle='dotted',label='EDC reconstruida',color='black')
plt.ylim(-80,2)
plt.title(f'')
plt.grid()
plt.legend()
plt.ylabel('EDC (dB)')
# plt.xlabel('Tempo (s)')
plt.xlim(0,max(TR_values)*(1.1))
plt.subplot(212)
plt.plot(t,_dB(EDC_true),color='black',alpha=.5,label='EDC')
plt.plot(t,_dB(exp_true[0]),linestyle='dashed',color='blue',label='Verdadeiro')
plt.plot(t,_dB(exp_true[1]),linestyle='dashed',color='blue')
plt.plot(t,_dB(EDC_estm[0]),linestyle='dotted',color='red',label='Estimado')
plt.plot(t,_dB(EDC_estm[1]),linestyle='dotted',color='red')

plt.ylim(-80,2)
plt.title(f'')
plt.grid()
plt.legend()
plt.ylabel('EDC (dB)')
plt.xlabel('Tempo (s)')
plt.xlim(0,max(TR_values)*(1.1))
# %%
