#%%
import numpy as np
import matplotlib.pyplot as plt
from src import edc_gen,tvec,MatrixPencil,DataPreparer
_dB = lambda x: np.log10(x + np.finfo(float).eps)
#%%
from tqdm import tqdm
fs = 44100
NFFT = 2**(20)
t = tvec(NFFT,fs)
max_tries = 50

p_test = [-4,-5,-6,-7]
err_p = np.zeros(len(p_test))
err_p_std = np.zeros(len(p_test))
misses_p = np.zeros([len(p_test),4])
for i_p,p in enumerate(p_test):
    err = np.zeros(max_tries)
    M_true = np.zeros(max_tries)
    M_estm = np.zeros(max_tries)
    for i in tqdm(range(max_tries),desc='trying',disable=False):
        n_exponentials = np.random.randint(1,5)
        TR_values = np.sort(abs(np.random.rand(n_exponentials)*12))
        TR_amplitudes = np.random.rand(n_exponentials)
        TR_amplitudes /= np.sum(TR_amplitudes)
        EDC_true = edc_gen(TR_values,TR_amplitudes,.00,t)[:-1,:].sum(axis=0)
        data = (DataPreparer(EDC_true,t,fs,
                            verbose=False
                            ).truncate_time(max_time=8,
                                            step=10E-3))
        pencil= (MatrixPencil(p=p, 
                            pencil_param=1/3,
                            verbose=False
                            ).fit(data.X,data.dt))
        EDC_estm = edc_gen(pencil.RT,pencil.amplitudes,.00,t)[:-1,:].sum(axis=0)

        err[i] = ((EDC_true-EDC_estm)/(EDC_true+1E-10)).mean()
        M_true[i] = n_exponentials
        M_estm[i] = len(pencil.RT)
    err_p[i_p] = err.mean()
    err_p_std[i_p] = err.std()
    misses = np.zeros(4)
    for idx,n in enumerate(range(1,5)):
        n_id = np.where(M_true==n)[0]
        misses[idx] = abs(M_true[n_id]-M_estm[n_id]).mean()
    misses_p[i_p] = misses

#%%

for i_p in range(4):
    plt.plot(np.arange(1,5),misses_p[i_p],label=f'p={p_test[i_p]}')
plt.xticks(np.arange(1,5,dtype=int))
plt.yticks(np.arange(1,5,dtype=int))
plt.xlabel('M verdadeiro')
plt.ylabel('|$M_{True} - M_{est}|$ (média)')
plt.grid()
plt.legend()

# %%

plt.errorbar(p_test,err_p,yerr=err_p_std,label=f'p={p_test[i_p]}')
# plt.xticks(np.arange(1,5,dtype=int))
# plt.yticks(np.arange(1,5,dtype=int))
plt.xlim(min(p_test)-1,max(p_test)+1)
plt.xlabel('p')
plt.ylabel('mean relative error')
plt.grid()
plt.legend()

# %%
