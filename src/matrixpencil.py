import numpy as np


class MatrixPencil():
    def __init__(self,p=2,pencil_param=1/3,verbose=False):
        self.precision = self._M_precision_parser(p)
        self.pencil_param = pencil_param
        self.verbose = verbose

    def fit(self,X,dt):
        dt = self._parse_dt(dt)
        self.dt=dt
        X = X.squeeze()
        self.X= X
        N = len(X)
        L = self._pencil_param_parser(self.pencil_param,N)

        Y = self._generate_kernel(X,N,L)
        U,s,V = self._svd(Y)
        M = self._estimate_M(s)

        self._log(f'Number of exponentials: {M}')
        V_trunc = V[:,:M]
        V1 = V_trunc[:-1,:]
        V2 = V_trunc[1:,:]
        self.z = np.linalg.eigvals(self._hermitian(V2) @ V1).real
        self.RT = self._poles_to_RT(self.z,dt)
        self._estimate_amplitudes(X,self.z,N)
        self._log(f'TR: {self.RT}')
        return self


    def _estimate_amplitudes(self,X,z,N):
        B = np.vander(z, N, increasing=True).T
        X = np.asarray(X).reshape(N,-1)   
        self.amplitudes = np.linalg.lstsq(B, X, rcond=None)[0].squeeze()

    def compute_residues(self):
        raise NotImplementedError('Não impelemntei... (Sarkar, Pereira) eq 25')

    def _parse_dt(self,dt):
        if np.isscalar(dt):
            return dt
        else:
            return dt[0]-dt[1]

    def _poles_to_RT(self,z,dt):
        ai = (np.log(z)/dt).real # Re{si} = alpha <- fator de amortecimento
        RT = np.log(1E-6)/ai
        return RT

    def _estimate_M(self,s,):
        S_norm = s/s[0]
        return abs(S_norm-self.precision).argmin()

    def _log(self,msg):
        if self.verbose:
            print(msg)

    def _generate_kernel(self,y,N,L):
        y_idx = np.indices((N-L-1, L)).sum(axis=0)
        Y = y[y_idx]
        return Y

    def _svd(self,A):
        U,S,V=np.linalg.svd(A)
        return U,S,self._hermitian(V)

    def _hermitian(self,A):
        if np.iscomplexobj(A):
            return A.conj().T
        return A.T

    def _pencil_param_parser(self,L,N):
        if L<1:
            return int(L*N)
        elif L>=1:
            return int(L)
        else:
            raise ValueError('Property `pencil_param` is invalid')

    def _M_precision_parser(self,p):
        if p==0:
            raise ValueError('`p` cannot be zero')
        elif p>0:
            return p
        else:
            return 10**(p)
