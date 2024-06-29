import numpy as np
from math import pi


def calc_RT_ASA(w, n, k, d):
    """
    Calculate R and T for the thick substrate (neglecting interference). Above and below the sample a media with
    n = 1 is assumed.
    Formulae are taken from (Tsu, 1999) and n - ik notation is assumed (k > 0 still denotes absorption).

    Parameters
    ----------
    w :
        Wavelength.
    n :
        Refractive index of the substrate material.
    k :
        Extinction coefficient of the substrate material.
    d :
        Substrate thickness.

    Any unit of length could be used, but it should be the same for wavelength and substrate thickness.

    Returns
    -------
    tuple : R and T values
    """
    n0 = 1
    n1 = n - 1j * k
    delta1 = 2 * pi / w * n1 * d
    r01 = (n0 - n1) / (n0 + n1)
    g01 = r01.real
    h01 = r01.imag
    a1 = -delta1.imag
    R01 = g01 ** 2 + h01 ** 2
    e2 = np.exp(-2 * a1)
    e4 = e2 ** 2
    denom = 1 - R01 ** 2 * e4
    out_R = R01 * (1 + e4 * (1 - 2 * (g01 ** 2 - h01 ** 2))) / denom
    out_T = (1 + R01 ** 2 - 2 * (g01 ** 2 - h01 ** 2)) * e2 / denom
    return out_R, out_T


def calc_RT_AFSA(w, n_f, k_f, d_f, n_s, k_s, d_s):
    """
    Calculate R and T for a thin film on a thick substrate. Interference is taken into account for the film only.
    Above and below the sample a media with n = 1 is assumed.
    Formulae are taken from (Tsu, 1999) and n - ik notation is assumed (k > 0 still denotes absorption).

    Parameters
    ----------
    w :
        Wavelength.
    n_f :
        Refractive index of the film material.
    k_f :
        Extinction coefficient of the film material.
    d_f :
        Thickness of the film.
    n_s :
        Refractive index of the substrate material.
    k_s :
        Extinction coefficient of the substrate material.
    d_s :
        Thickness of the substrate.

    Returns
    -------
    tuple : R and T values
    """
    n0 = 1
    n1 = n_f - 1j * k_f
    n2 = n_s - 1j * k_s
    n3 = 1

    d1 = d_f
    d2 = d_s

    delta1 = 2 * pi / w * n1 * d1
    gamma1, a1 = delta1.real, -delta1.imag
    ea1 = np.exp(-a1)
    ea1_2 = ea1**2
    ea1_4 = ea1_2**2

    delta2 = 2 * pi / w * n2 * d2
    gamma2, a2 = delta2.real, -delta2.imag
    ea2 = np.exp(-a2)
    ea2_2 = ea2**2
    ea2_4 = ea2_2**2

    r01 = (n0 - n1) / (n0 + n1)
    g01, h01 = r01.real, r01.imag
    R01 = g01**2 + h01**2

    r12 = (n1 - n2) / (n1 + n2)
    g12, h12 = r12.real, r12.imag
    R12 = g12**2 + h12**2

    r23 = (n2 - n3) / (n2 + n3)
    g23, h23 = r23.real, r23.imag
    R23 = g23**2 + h23**2

    A = 2 * (g01 * g12 + h01 * h12)
    B = 2 * (g01 * h12 - h01 * g12)
    C = 2 * (g01 * g12 - h01 * h12)
    D = 2 * (g01 * h12 + h01 * g12)

    denom = 1 + ea1_2 * (ea1_2 * R01 * R12 + C * np.cos(2 * gamma1) + D * np.sin(2 * gamma1))
    Denom = 1 - ea2_4 * R23 * (R12 + ea1_2 * (ea1_2 * R01 + A * np.cos(2 * gamma1) - B * np.sin(2 * gamma1))) / denom

    out_R = R23 * ea1_4 * ea2_4
    out_R *= (1 - g01)**2 + h01**2
    out_R *= (1 - g12)**2 + h12**2
    out_R *= (1 + g01)**2 + h01**2
    out_R *= (1 + g12)**2 + h12**2
    out_R /= Denom * denom**2
    out_R += (R01 + ea1_2 * (ea1_2 * R12 + A * np.cos(2 * gamma1) + B * np.sin(2 * gamma1))) / denom

    out_T = n3 / n0 * ea1_2 * ea2_2 / (Denom * denom)
    out_T *= (1 + g01)**2 + h01**2
    out_T *= (1 + g12)**2 + h12**2
    out_T *= (1 + g23)**2 + h23**2

    return out_R, out_T


if __name__ == '__main__':
    import time
    t0 = time.time()
    R, T = 0, 0
    N = 50
    for i in range(N**2):
        R, T = calc_RT_AFSA(1000e-9, np.array([3.5, 3.6, 3.7]), np.array([1.5, 1.4, 1.3]), 104.7e-9, 1.5, 1e-4, 0.7e-3)
    t1 = time.time()
    print(f'R = {R}, T = {T}, calculated in {(t1 - t0) / N**2 * 1e6:.1f} us')
