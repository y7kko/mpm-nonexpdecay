import numpy as np

def edc_gen(t_vals, a_vals, noise_level, time_axis, compensate_uli=True) -> np.ndarray:
    t_vals = np.asarray(t_vals)
    a_vals = np.asarray(a_vals)
    value_dim = len(t_vals.shape) - 1

    # get decay rate: decay energy should have decreased by 60 db after T seconds
    zero_a = (a_vals == 0)
    tau_vals = np.log(1e6) / t_vals
    tau_vals[zero_a] = 0

    # calculate decaying exponential terms
    time_vals = - np.tile(time_axis, (*t_vals.shape, 1)) * np.expand_dims(tau_vals, -1)
    exponentials = np.exp(time_vals)

    # account for limited upper limit of integration, see: Xiang, N., Goggans, P. M., Jasa, T. & Kleiner, M.
    # "Evaluation of decay times in coupled spaces: Reliability analysis of Bayeisan decay time estimation."
    # J Acoust Soc Am 117, 3707–3715 (2005).
    if compensate_uli:
        exp_offset = np.expand_dims(exponentials[..., -1], -1)
    else:
        exp_offset = 0

    # calculate final exponential terms
    exponentials = (exponentials - exp_offset) * np.expand_dims(a_vals, -1)

    # zero exponentials where T=A=0 (they are NaN now because div by 0, and NaN*0=NaN in python)
    exponentials[zero_a, :] = 0

    # calculate noise term
    noise = noise_level * np.linspace(len(time_axis), 1, len(time_axis))
    noise = np.expand_dims(noise, value_dim)

    edc_model = np.concatenate((exponentials, noise), value_dim)
    return edc_model

def tvec(data,fs):
    """Gera um vetor de frequências para um dado no tempo

    Args:
        data (ndarray or int): 
            - Caso seja um escalar, é gerado um vetor
              com len(time_vector) = data.
            - Caso seja um vetor, é gerado um vetor com
                len(time_vector) = len(data)
        fs (float): Frequência de amostragem do sinal

    Returns:
        ndarray: Um vetor temporal correspondente aos dados de entrada
    """
    # print('att')
    if np.isscalar(data):
        n_spl = data
    else:
        n_spl = len(data)
    max_time = (n_spl - 1)/fs
    time_vector = np.linspace(0,max_time,n_spl)
    return time_vector

