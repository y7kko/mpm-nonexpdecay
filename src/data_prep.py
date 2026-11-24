import numpy as np

class DataPreparer():
    def __init__(self,X,t,fs,verbose=False):
        self.X = X.copy()
        self.t = t.copy()
        self.fs = fs
        self._update_dt()
        self.verbose=verbose

    def truncate_time(self,max_time=None,step=None,step_in_samples=False):
        step = self._parse_step(step,step_in_samples)
        if max_time is None:
            t_mask = slice(None)
        else:
            t_mask = slice(self._s2i(self.t,max_time))

        t_div = slice(None,None,step)
        self.X = self.X[t_mask][t_div]
        self.t = self.t[t_mask][t_div]
        self._update_dt()

        self._log(f'dt: {self.dt*1E3:.3f} ms, T={self.t[-1]:.3f}, N={len(self.t):E}')
        self._log(f'{(self.t.size**2)*(np.finfo(float).bits/8)/(2**20):.2f} MiB')
        return self

    def _s2i(self,t,val):
        return abs(t-val).argmin()

    def _parse_step(self,step,step_in_samples):
        if step is None:
            return step
        elif step_in_samples:
            return int(step)
        return int(self.fs*step)

    def _update_dt(self,):
        self.dt = abs(self.t[1]-self.t[0])

    def _log(self,msg):
        if self.verbose:
            print(msg)
