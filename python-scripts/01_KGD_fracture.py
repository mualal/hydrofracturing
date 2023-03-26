import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
from scipy.special import beta, betainc


def fcn_g_del(
        kh,
        ch
):
    ikh = (kh < 0) | (abs(np.imag(kh)) > 0)
    kh[ikh] = 0
    if ikh.any() == True:
        print('Предупруждение: отрицательное или комплексное число kh в fcn_g_del')

    kh = kh + np.spacing(1.0)
    kh[kh > 1] = 1

    ich = (ch < 0) | (abs(np.imag(ch)) > 0)
    ch[ich] = 0
    if ich.any() == True:
        print('Предупруждение: отрицательное или комплексное число ch в fcn_g_del')

    betam = 2**(1/3) * 3**(5/6)
    betamt = 4/15**(1/4) / ((2)**(1/2) - 1)**(1/4)

    b0 = 3 * betamt**4 / 4 / betam**3

    f = lambda kh, ch, c1: (1 - kh**3 - 3/2 * ch * (1 - kh**2) + 3 * ch**2 * (1 - kh) - \
                            3 * ch**3 * 2 * np.arctanh((1-kh) / (2*ch+1+kh))) / (3 * c1)

    fkmt = lambda kh, ch, c1: (1 / (4*ch)*(1-kh**4)-1/(5*ch**2)*(1-kh**5)+1/(6*ch**3)*(1-kh**6))/c1

    c1 = lambda delt: 4*(1-2*delt)/(delt*(1-delt))*np.tan(np.pi*delt)
    c2 = lambda delt: 16*(1-3*delt)/(3*delt*(2-3*delt))*np.tan(3*np.pi/2*delt)

    ich = (ch > 1e3)
    delt = betam**3 / 3 * f(kh, b0*ch, betam**3/3) * (1 + b0 * ch)
    delt[ich]=betam**3 / 3 * fkmt(kh[ich], b0*ch[ich], betam**3/3) * (1 + b0 * ch[ich])

    delt[delt <= 0] = 1e-6
    delt[delt >= 1/3] = 1/3 - 1e-6
    
    bh = c2(delt) / c1(delt)
    
    xh = f(kh, ch*bh, c1(delt))
    xh[ich] = fkmt(kh[ich], ch[ich]*bh[ich], c1(delt[ich]))

    return xh


def fcn_delta_p(
        kh,
        ch,
        p
):

    ikh = (kh < 0) | (abs(np.imag(kh)) > 0)
    kh[ikh] = 0
    if ikh.any() == True:
        print('Предупруждение: отрицательное или комплексное число kh в fcn_delta_p')

    kh = kh + np.spacing(1.0)
    kh[kh > 1] = 1

    ich = (ch < 0) | (abs(np.imag(ch)) > 0)
    ch[ich] = 0

    betam = 2**(1/3) * 3**(5/6)
    betamt = 4/15**(1/4) / ((2)**(1/2) - 1)**(1/4)

    b0 = 3 * betamt**4 / 4 / betam**3

    f = lambda kh, ch, c1: (1 - kh**3 - 3/2 * ch * (1 - kh**2) + 3 * ch**2 * (1 - kh) - \
                            3 * ch**3 * 2 * np.arctanh((1-kh) / (2*ch+1+kh))) / (3 * c1)

    fkmt = lambda kh, ch, c1: (1 / (4*ch)*(1-kh**4)-1/(5*ch**2)*(1-kh**5)+1/(6*ch**3)*(1-kh**6))/c1

    ich = (ch > 1e3)
    delt = betam**3 / 3 * f(kh, b0*ch, betam**3/3) * (1 + b0 * ch)
    delt[ich]=betam**3 / 3 * fkmt(kh[ich], b0*ch[ich], betam**3/3) * (1 + b0 * ch[ich])

    delt_p = (1 - p + p * f(kh, ch*b0, betam**3/3) * (betam**3 + betamt**4 * ch)) * delt
    delt_p[ich]=(1 - p + p * fkmt(kh[ich], ch[ich]*b0, betam**3/3) * (betam**3 + betamt**4 * ch[ich])) * delt[ich]

    return delt_p


def fcn_lam_kgd(
        kh,
        ch,
        alpha
):

    lam_k = 0.5
    lam_m = 0.588
    lam_mt = 0.520
    
    
    lam_0 = 0.5
    delt = (1 + fcn_delta_p(kh, ch, 0)) / 2
    b0 = lambda x, p1, p2: beta(p1, p2) * (1 - betainc(p1, p2, x))
    fcn_b2 = 2**(1+delt) * (b0(1/2, lam_0+1, 1+delt)) + ch * alpha**(3/2) * beta(alpha,3/2)
    eta0 = 1 - ch * alpha**(3/2) * beta(alpha, 3/2) / fcn_b2

    p_k = kh**3
    peta = eta0
    lam = lam_m * (1-p_k) * peta + lam_mt * (1-p_k) * (1-peta) + lam_k * p_k

    return lam


def fcn_b_kgd(
        kh,
        ch,
        alpha
):
    p = 0.0
    delt = (1 + fcn_delta_p(kh, ch, p)) / 2
    lam = fcn_lam_kgd(kh, ch, alpha)
    
    b0 = lambda x, p1, p2: beta(p1, p2) * (1 - betainc(p1, p2, x))
    b = 2**(1+delt) * (b0(1/2, lam+1, 1+delt)) + ch * alpha**(3/2)*beta(alpha,3/2)

    return b


def fcn_m_kgd(
        rho,
        s
):
    m = s / (s**2 - rho**2)
    return m


def fcn_press_calc(
        delt,
        lam,
        rho
):
    w = (1-rho)**delt * (1+rho)**lam / 2**lam
    
    ds=rho[1]-rho[0]
    
    [rrho, s] = np.meshgrid(rho, rho)
    
    m = fcn_m_kgd(rrho, s+ds/2) - fcn_m_kgd(rrho, s-ds/2)
    
    p = np.matmul(m.T, w / (2 * np.pi))

    return p


def kgd_hf_approximation(
    tau,
    k_m
) -> tuple:
    if k_m < 1e-30:
        k_m = 1e-30
    th = tau * 2**6 * k_m**(-12)
    q_h = k_m**(-4) / 2

    # начальное приближение
    alpha = 2/3
    kh_0 = 0 * th + 1/2
    ch_0 = 0 * th + 1/2

    for alpha_index in range(1,4):
        res = 1
        itt_max = 100
        itt = 0
        tol = 1e-5

        while itt < itt_max and res > tol:
            if itt == itt_max-1 and alpha_index == 3:
                print('Нет сходимости')
                print(res)
            itt += 1
            kh = kh_0
            ch = ch_0

            itt_k = 0
            res_k = 1
            while itt_k < itt_max and res_k > tol:
                itt_k += 1
                fg = fcn_g_del(kh, ch)
                fg_k = -fg / (1 - kh + np.spacing(1.0))
                
                f1 = kh**6 - alpha**(1/2) / th**(1/2) * ch**3 * fg
                f1_k = 6 * kh**5 - alpha**(1/2) / th**(1/2) * ch**3 * fg_k

                kh = 0.0 * kh + 1.0 * (kh - f1 / f1_k)
                kh[kh < 0] = 1e-5
                kh[kh > 1] = 1 - 1e-5
                
                res_k = max(abs(f1 / f1_k))
            
            itt_c = 0
            res_c = 1
            while itt_c < itt_max and res_c > tol:
                itt_c += 1
                ch_test = ch
                ch = th**(1/6) * kh**(2/3) / alpha**(1/2) / q_h**(1/3) * (fcn_b_kgd(kh, ch, alpha))**(1/3)
                res_c = max(abs(ch - ch_test))
            
            res = max(((kh - kh_0)**2 + (ch - ch_0)**2)**(1/2))
            kh_0 = kh
            ch_0 = ch
        
        sh = fcn_g_del(kh, ch)
        print(sh)

        lh = ch**4 * sh**2 / (kh**10)

        alpha = 0*lh
        alpha[1:] = (np.log(lh[1:]) - np.log(lh[:-1])) / (np.log(th[1:]) - np.log(th[:-1]))
        alpha[0] = alpha[1]
    
    p = 0.0
    delt = (1 + fcn_delta_p(kh, ch, p)) / 2

    eff = 1 - ch * alpha**(3/2) * beta(alpha, 3/2) / fcn_b_kgd(kh, ch, alpha)

    lam = fcn_lam_kgd(kh, ch, alpha)
    wha = ch**2 * sh / kh**6 / 2**lam
    
    gamma = lh / (2**4) * k_m**10
    om = wha / (2**2) * k_m**6

    return om, gamma, eff, delt, lam, alpha


def get_kgd_solution(
        e_p,
        mu_p,
        k_p,
        c_p,
        q_0,
        t,
        n
):
    t_mmt = mu_p * q_0**3 / (e_p * c_p**6)
    tau = t / t_mmt
    k_m = (k_p**4 / (mu_p * q_0 * e_p**3))**(1/4)

    tau_arr = np.array([tau/4, tau/2, tau])
    omega, gamma, eff, delt, lam, _ =  kgd_hf_approximation(tau_arr, k_m)

    lst = (mu_p * q_0**5 / (e_p * c_p**8))**(1/2)
    eps = c_p**2 / q_0

    xi0 = np.linspace(0, 1, n+1)
    xi = (xi0[:-1] + xi0[1:]) / 2

    l = gamma[-1] * lst
    w = omega[-1] * eps * lst * (1-xi)**delt[-1] * (1+xi)**lam[-1]
    p = eps * e_p * 2**lam[-1] * omega[-1] / gamma[-1] * fcn_press_calc(delt[-1], lam[-1], xi)
    eta = eff[-1]

    return l, w, p, xi, eta


def kgd_vert_sol(
        e_p,
        mu_p,
        k_p,
        c_p,
        q_0,
        t,
        n
):
    xi0 = np.linspace(0, 1, n+1)
    xi = (xi0[:-1] + xi0[1:]) / 2

    l = np.zeros(4)
    eta = np.zeros(4)
    w = np.zeros((n, 4))
    p = np.zeros((n, 4))

    # M vertex
    l[0] = 0.6159 * (q_0**3 * e_p * t**4 / mu_p)**(1/6)
    eta[0] = 1
    w[:, 0] = 1.1265 * (mu_p * q_0**3 * t**2 / e_p)**(1/6) * (1+xi)**0.588 * (1-xi)**(2/3)
    p[:, 0] = 2.7495 * (mu_p * e_p**2 / t)**(1/3) * fcn_press_calc(2/3, 0.588, xi)

    # Mt vertex
    l[1] = 0.3183 * q_0 * t**(1/2) / c_p
    eta[1] = 0
    w[:, 1] = 0.8165 * (mu_p * q_0**3 * t / e_p / c_p**2)**(1/4) * (1+xi)**0.520 * (1-xi)**(5/8)
    p[:, 1] = 3.6783 * (c_p**2 * mu_p * e_p**3 / t / q_0)**(1/4) * fcn_press_calc(5/8, 0.520, xi)
    
    # K vertex
    l[2] = 0.9324 * (e_p * q_0 * t / k_p)**(2/3)
    eta[2] = 1
    w[:, 2] = 0.6828 * (k_p**2 * q_0 * t / e_p**2)**(1/3) * (1-xi**2)**(1/2)
    p[:, 2] = 0.1831 * (k_p**4 / e_p / q_0 / t)**(1/3) * np.ones(n)
    
    # Kt vertex
    l[3] = 0.3183 * q_0 * t**(1/2) / c_p
    eta[3] = 0
    w[:, 3] = 0.3989 * (k_p**4 * q_0**2 * t / e_p**4 / c_p**2)**(1/4) * (1-xi**2)**(1/2)
    p[:, 3] = 0.3183 * (k_p**4 * c_p**2 / q_0**2 / t)**(1/4) * np.ones(n)

    return l, w, p, xi, eta


def main():
    e_p = 10 * 1e9  # Па
    mu_p = 0.1  # Па*с
    k_p = 1 * 1e6  # Па*м^(1/2)
    c_p = 1e-6  # м/с^(1/2)
    t = 1e3  # с
    q_0 = 0.001  # м^3/с
    h = 1  # м
    n = 1000

    l, w, p, xi, eta = get_kgd_solution(e_p, mu_p, k_p, c_p, q_0/h, t, n)
    l_v, w_v, p_v, xi_v, eta_v = kgd_vert_sol(e_p, mu_p, k_p, c_p, q_0/h, t, n)


    vertex_index = 0
    col = 'b'
    match vertex_index:
        case 0:
            col = 'b'
        case 1:
            col = 'g'
        case 2:
            col = 'r'
        case 3:
            col = 'm'
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 7))
    #fig.tight_layout(h_pad=2)
    #ax[1].fmt_ydata = (lambda x: '$%1.1fM' % (x*1e-6))
    ax[0].plot(l * xi, w, 'k')
    ax[0].plot(l_v[vertex_index] * xi_v, w_v[:, vertex_index], '--', color=col)
    ax[0].set_xlabel('Координата x, м')
    ax[0].set_ylabel('Раскрытие w, м')
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth=0.8)
    ax[0].grid(which='minor', linestyle='--', linewidth=0.3)
    ax[1].plot(l * xi, p, 'k')
    ax[1].plot(l_v[vertex_index] * xi_v, p_v[:, vertex_index], '--', color=col)
    #ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax[1].set_xlabel('Координата x, м')
    ax[1].set_ylabel('Давление p, Па')
    ax[1].minorticks_on()
    ax[1].grid(which='major', linestyle='-', linewidth=0.8)
    ax[1].grid(which='minor', linestyle='--', linewidth=0.3)
    plt.show()


if __name__ == '__main__':
    main()
