import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick
from scipy.special import beta, betainc, ellipk, ellipe


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


def fcn_lam_radial(
        kh,
        ch,
        alpha
):

    lam_k = 0.5
    lam_m = 0.487
    lam_mt = 0.397
    
    
    lam_0 = 0.5
    delt = (1 + fcn_delta_p(kh, ch, 0)) / 2
    b0 = lambda x, p1, p2: beta(p1, p2) * (1 - betainc(p1, p2, x))
    fcn_b2 = 2**(1+delt) * (-b0(1/2, lam_0+1, 2+delt) + b0(1/2, lam_0+2, 1+delt)) + \
        ch * alpha**(3/2) * beta(2*alpha,3/2)
    eta0 = 1 - ch * alpha**(3/2) * beta(2*alpha, 3/2) / fcn_b2

    p_k = kh**4
    peta = eta0
    lam = lam_m * (1-p_k) * peta + lam_mt * (1-p_k) * (1-peta) + lam_k * p_k

    return lam


def fcn_b_radial(
        kh,
        ch,
        alpha
):
    p = 0.0
    delt = (1 + fcn_delta_p(kh, ch, p)) / 2
    lam = fcn_lam_radial(kh, ch, alpha)
    
    b0 = lambda x, p1, p2: beta(p1, p2) * (1 - betainc(p1, p2, x))
    b = 2**(1+delt) * (-b0(1/2, lam+1, 2+delt)+b0(1/2, lam+2, 1+delt)) + ch * alpha**(3/2)*beta(2*alpha,3/2)

    return b


def fcn_m_radial(
        rho,
        s
):
    m = 0 * s

    ind_1 = (rho > s)
    ind_2 = (rho < s)

    s_1 = s[ind_1]
    rho_1 = rho[ind_1]
    s_2 = s[ind_2]
    rho_2 = rho[ind_2]
    k1, e1 = ellipk(s_1**2 / rho_1**2), ellipe(s_1**2 / rho_1**2)
    e2 = ellipe(rho_2**2 / s_2**2)

    m[ind_1] = 1 / rho_1 * k1 + rho_1 / (s_1**2 - rho_1**2) * e1

    m[ind_2] = s_2 / (s_2**2 - rho_2**2) * e2

    return m


def fcn_press_calc(
        delt,
        lam,
        rho
):
    w = (1-rho)**delt * (1+rho)**lam / 2**lam
    
    ds=rho[1]-rho[0]
    
    [rrho, s] = np.meshgrid(rho, rho)
    
    m = fcn_m_radial(rrho, s+ds/2) - fcn_m_radial(rrho, s-ds/2)
    
    p = np.matmul(m.T, w / (2 * np.pi))

    return p


def radial_hf_approximation(
    tau,
    phi
) -> tuple:
    if phi < 1e-30:
        phi = 1e-30
    th = tau * (2**6 * phi**(3/2))
    q_h = 8 / np.pi * phi

    # начальное приближение
    alpha = 4/9
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
                ch = th**(3/10) * kh**(4/5) / alpha**(1/2) / q_h**(1/5) * (fcn_b_radial(kh, ch, alpha))**(1/5)
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

    eff = 1 - ch * alpha**(3/2) * beta(2*alpha, 3/2) / fcn_b_radial(kh, ch, alpha)

    lam = fcn_lam_radial(kh, ch, alpha)
    wha = ch**2 * sh / kh**6 / 2**lam
    
    gamma = lh / (2**4 * phi)
    om = wha / (2**2 * phi**(1/2))

    return om, gamma, eff, delt, lam


def get_radial_solution(
        e_p,
        mu_p,
        k_p,
        c_p,
        q_0,
        t,
        n
):
    t_mk = (mu_p**5 * e_p**(13) * q_0**3 / k_p**(18))**(1/2)
    tau = t / t_mk
    phi = mu_p**3 * e_p**(11) * c_p**4 * q_0 / k_p**(14)

    tau_arr = np.array([tau/4, tau/2, tau])
    omega, gamma, eff, delt, lam =  radial_hf_approximation(tau_arr, phi)

    lst = (q_0**3 * e_p * t_mk**4 / mu_p)**(1/9)
    eps = (mu_p / (e_p * t_mk))**(1/3)

    rho0 = np.linspace(0, 1, n+1)
    rho = (rho0[:-1] + rho0[1:]) / 2

    r = gamma[-1] * lst
    w = omega[-1] * eps * lst * (1-rho)**delt[-1] * (1+rho)**lam[-1]
    p = eps * e_p * 2**lam[-1] * omega[-1] / gamma[-1] * fcn_press_calc(delt[-1], lam[-1], rho)
    eta = eff[-1]

    return r, w, p, rho, eta, tau, phi


def radial_vert_sol(
        e_p,
        mu_p,
        k_p,
        c_p,
        q_0,
        t,
        n
):
    rho0 = np.linspace(0, 1, n+1)
    rho = (rho0[:-1] + rho0[1:]) / 2

    r = np.zeros(4)
    eta = np.zeros(4)
    w = np.zeros((n, 4))
    p = np.zeros((n, 4))

    # M vertex
    r[0] = 0.6944 * (q_0**3 * e_p * t**4 / mu_p)**(1/9)
    eta[0] = 1
    w[:, 0] = 1.1901 * (mu_p**2 * q_0**3 * t / e_p**2)**(1/9) * (1+rho)**0.487 * (1-rho)**(2/3)
    p[:, 0] = 2.4019 * (mu_p * e_p**2 / t)**(1/3) * fcn_press_calc(2/3, 0.487, rho)

    # Mt vertex
    r[1] = 0.4502 * (q_0**2 * t / c_p**2)**(1/4)
    eta[1] = 0
    w[:, 1] = 1.0574 * (mu_p**4 * q_0**6 * t / e_p**4 / c_p**2)**(1/16) * (1+rho)**0.397 * (1-rho)**(5/8)
    p[:, 1] = 3.0931*(c_p**6 * mu_p**4 * e_p**(12) / t**3 / q_0**2)**(1/16) * fcn_press_calc(5/8, 0.397, rho)
    
    # K vertex
    r[2] = 0.8546 * (e_p * q_0 * t / k_p)**(2/5)
    eta[2] = 1
    w[:, 2] = 0.6537 * (k_p**4 * q_0 * t / e_p**4)**(1/5) * (1-rho**2)**(1/2)
    p[:, 2] = 0.3004 * (k_p**6 / e_p / q_0 / t)**(1/5) * np.ones(n)
    
    # Kt vertex
    r[3] = 0.4502 * (q_0**2 * t / c_p**2)**(1/4)
    eta[3] = 0
    w[:, 3] = 0.4744 * (k_p**8 * q_0**2 * t / e_p**8 / c_p**2)**(1/8) * (1-rho**2)**(1/2)
    p[:, 3] = 0.4139 * (k_p**8 * c_p**2 / q_0**2 / t)**(1/8) * np.ones(n)

    return r, w, p, rho, eta


def plot_radial_parametric_space(
        tau,
        phi
):
    tau_min = -10
    tau_max = 20
    phi_min = -30
    phi_max = 20

    t_mk0 = 4.54e-2
    t_mk1 = 2.59e6

    t_mmt0 = 7.41e-6
    t_mmt1 = 7.20e2

    t_kkt0 = 5.96e-8
    t_kkt1 = 4.81e2

    t_mtkt0 = 4.18
    t_mtkt1 = 2.01e11

    tau_mt = (t_mmt1**(14/9) * t_mtkt0**2)**(9/32)
    phi_mt = (tau_mt / t_mtkt0)**2
    tau_kt = (t_kkt1**(6/5) * t_mtkt1**2)**(5/16)
    phi_kt = (tau_kt / t_mtkt1)**2

    plt.plot(np.log10(np.array([t_mk0, t_mk0, t_mmt0 / (10**phi_max)**(9/14)], dtype='float')),
             np.log10(np.array([10**phi_min, (t_mk0 / t_mmt0)**(-14/9), 10**phi_max], dtype='float')), 'b-')
    plt.plot(np.log10([t_mmt1 / (10**phi_max)**(9/14), tau_mt, t_mtkt0*10**(phi_max/2)], dtype='float'),
             np.log10(np.array([10**phi_max, phi_mt, 10**phi_max], dtype='float')), 'g-')
    plt.plot(np.log10(np.array([t_mk1, t_mk1, t_kkt0 / (10**phi_min)**(5/6)], dtype='float')),
             np.log10([10**phi_min, (t_kkt0 / t_mk1)**(6/5), 10**phi_min]), 'r-')
    plt.plot(np.log10(np.array([10**tau_max, tau_kt, 10**tau_max], dtype='float')),
             np.log10([(t_kkt1 / 10**(tau_max))**(6/5), phi_kt, (10**tau_max / t_mtkt1)**2]), 'm-')
    
    if tau < 10**tau_min:
        tau = 10**tau_min
    if tau > 10**tau_max:
        tau =  10**tau_max
    if phi < 10**phi_min:
        phi =  10**phi_min
    if phi > 10**phi_max:
        phi = 10**phi_max
    plt.plot(np.log10(tau), np.log10(phi), 'ko')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.8)
    plt.grid(which='minor', linestyle='--', linewidth=0.3)

    plt.show()


def main(plot_parametric_space=True):
    e_p = 1e10  # Па
    mu_p = 0.1  # Па*с
    k_p = 1e6  # Па*м^(1/2)
    c_p = 1e-6  # м/с^(1/2)
    t = 1e3  # с
    q_0 = 0.001  # м^3/с
    h = 1  # м
    n = 1000

    r, w, p, rho, eta, tau, phi = get_radial_solution(e_p, mu_p, k_p, c_p, q_0, t, n)
    r_v, w_v, p_v, rho_v, eta_v = radial_vert_sol(e_p, mu_p, k_p, c_p, q_0, t, n)

    if plot_parametric_space:
        plot_radial_parametric_space(tau, phi)

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
    ax[0].plot(r * rho, w, 'k')
    ax[0].plot(r_v[vertex_index] * rho_v, w_v[:, vertex_index], '--', color=col)
    ax[0].set_xlabel('Координата x, м')
    ax[0].set_ylabel('Раскрытие w, м')
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth=0.8)
    ax[0].grid(which='minor', linestyle='--', linewidth=0.3)
    ax[1].plot(r * rho, p, 'k')
    ax[1].plot(r_v[vertex_index] * rho_v, p_v[:, vertex_index], '--', color=col)
    #ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax[1].set_xlabel('Координата x, м')
    ax[1].set_ylabel('Давление p, Па')
    ax[1].minorticks_on()
    ax[1].grid(which='major', linestyle='-', linewidth=0.8)
    ax[1].grid(which='minor', linestyle='--', linewidth=0.3)
    plt.show()


if __name__ == '__main__':
    main()
