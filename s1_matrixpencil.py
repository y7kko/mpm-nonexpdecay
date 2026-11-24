#%%
import numpy as np
import matplotlib.pyplot as plt
from src import edc_gen,tvec,MatrixPencil,DataPreparer

#%%

fs = 44100
NFFT = 2**(20)
t = tvec(NFFT,fs)
TR_values = np.sort([2,9])
TR_amplitudes = np.asarray([.7,.3])
EDC = edc_gen(TR_values,TR_amplitudes,.00,t)[:-1,:].sum(axis=0)
# %%
data = (DataPreparer(EDC,t,fs,
                     verbose=True
                     ).truncate_time(max_time=7,
                                     step=10E-3))
pencil= (MatrixPencil(p=-2, 
                      pencil_param=1/3,
                      verbose=True
                      ).fit(data.X,data.dt))
#%%
abs(np.sort(pencil.RT)-TR_values)/TR_values
