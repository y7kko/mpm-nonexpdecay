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
max_tries = 20


err = np.zeros(max_tries)
M_true = np.zeros(max_tries)
M_estm = np.zeros(max_tries)

E_true = []
E_estm = []
for i in tqdm(range(max_tries),desc='trying',disable=False):
    n_exponentials = np.random.randint(1,5)
    TR_values = np.sort(abs(np.random.rand(n_exponentials)**(.5)*12))
    TR_amplitudes = np.random.rand(n_exponentials)
    TR_amplitudes /= np.sum(TR_amplitudes)
    EDC_true = edc_gen(TR_values,TR_amplitudes,.00,t)[:-1,:].sum(axis=0)
    data = (DataPreparer(EDC_true,t,fs,
                        verbose=False
                        ).truncate_time(max_time=8,
                                        step=10E-3))
    pencil= (MatrixPencil(p=-6, 
                        pencil_param=1/3,
                        verbose=False
                        ).fit(data.X,data.dt))
    EDC_estm = edc_gen(pencil.RT,pencil.amplitudes,.00,t)[:-1,:].sum(axis=0)

    err[i] = ((EDC_true-EDC_estm)/(EDC_true+1E-10)).mean()
    M_true[i] = n_exponentials
    M_estm[i] = len(pencil.RT)
    E_true.append(EDC_true)
    E_estm.append(EDC_estm)
#%%
plt.plot(_dB(EDC_true))
#%%
for Et,Ee in zip(E_true,E_estm):
    l1 = plt.plot(_dB(Et),alpha=.5)
    plt.plot(_dB(Ee),color=l1[0].get_color(),linestyle='dashed')
plt.grid()

#%%
n_exponentials = np.random.randint(1,5)
TR_values = np.sort(abs(np.random.rand(n_exponentials)**(.5)*12))
TR_amplitudes = np.random.rand(n_exponentials)
TR_amplitudes /= np.sum(TR_amplitudes)
EDC_true = edc_gen(TR_values,TR_amplitudes,.00,t)[:-1,:].sum(axis=0)
data = (DataPreparer(EDC_true,t,fs,
                    verbose=False
                    ).truncate_time(max_time=8,
                                    step=10E-3))
plt.plot(data.t,_dB(data.X))
plt.title(f'')
