from dataset import *

c = 2.9979e10          # Speed of light in cm/s
G = 6.67408e-8         # Gravitational constant in cm^3 g^-1 s^-2
Msun = 1.989e33        # Solar mass in grams
rsol = G * Msun / c**2 # Schwarzschild radius of the sun in cm
Density = Msun / rsol**3
time = rsol / c        # Time unit in seconds
Psol = Msun * c**2 / rsol**3
rhosat = 2.66e14
Psat = 2.5 * 1.602e33
pi = np.pi
four_pi = 4 * pi

msol = Msun
rhosol = Density

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to calculate initial central density
def rho0f(ii, rho1, rho2, imax):
    return rho1 + ((ii - 1) / (imax - 1))**3 * (rho2 - rho1)

pmin = 1e2
rmin = 0.001
rmax = 100

# Low-density parameters
rhoL1 = 2.62789e12 / Density
rhoL2 = 3.78358e11 / Density
rhoL3 = 2.44034e7 / Density
rhoL4 = 0.0

gammaL1 = 1.35692
gammaL2 = 0.62223
gammaL3 = 1.28733
gammaL4 = 1.58425

KL1 = 3.99874e-8 * (Msun / rsol**3)**(gammaL1 - 1)
KL2 = 5.32697e1 * (Msun / rsol**3)**(gammaL2 - 1)
KL3 = 1.06186e-6 * (Msun / rsol**3)**(gammaL3 - 1)
KL4 = 6.80110e-9 * (Msun / rsol**3)**(gammaL4 - 1)

@numba.njit
def PofRho_numba(rho, argsz):
    (rhoL3, rhoL2, rhoL1, rho0, rho1, rho2,
     KL4, KL3, KL2, KL1, K1, K2, K3,
     gammaL4, gammaL3, gammaL2, gammaL1, gamma1, gamma2, gamma3) = argsz
    if rho < rhoL3:
        return KL4 * rho**gammaL4
    elif rhoL3 <= rho < rhoL2:
        return KL3 * rho**gammaL3
    elif rhoL2 <= rho < rhoL1:
        return KL2 * rho**gammaL2
    elif rhoL1 <= rho < rho0:
        return KL1 * rho**gammaL1
    elif rho0 <= rho < rho1:
        return K1 * rho**gamma1
    elif rho1 <= rho < rho2:
        return K2 * rho**gamma2
    else:
        return K3 * rho**gamma3

@numba.njit
def epsOfP_numba(p, args2z):
    (rhoL3, rhoL2, rhoL1, rho0, rho1, rho2,
     KL4, KL3, KL2, KL1, K1, K2, K3,
     gammaL4, gammaL3, gammaL2, gammaL1, gamma1, gamma2, gamma3,
     pL3, pL2, pL1, p0, p1, p2,
     alphaL4, alphaL3, alphaL2, alphaL1, alpha1, alpha2, alpha3) = args2z
    if p < pL3:
        return (1 + alphaL4) * (p / KL4)**(1 / gammaL4) + p / (gammaL4 - 1)
    elif pL3 <= p < pL2:
        return (1 + alphaL3) * (p / KL3)**(1 / gammaL3) + p / (gammaL3 - 1)
    elif pL2 <= p < pL1:
        return (1 + alphaL2) * (p / KL2)**(1 / gammaL2) + p / (gammaL2 - 1)
    elif pL1 <= p < p0:
        return (1 + alphaL1) * (p / KL1)**(1 / gammaL1) + p / (gammaL1 - 1)
    elif p0 <= p < p1:
        return (1 + alpha1) * (p / K1)**(1 / gamma1) + p / (gamma1 - 1)
    elif p1 <= p < p2:
        return (1 + alpha2) * (p / K2)**(1 / gamma2) + p / (gamma2 - 1)
    else:
        return (1 + alpha3) * (p / K3)**(1 / gamma3) + p / (gamma3 - 1)

@numba.njit
def epsOfRho_numba(rho, args2z):
    (rhoL3, rhoL2, rhoL1, rho0, rho1, rho2,
     KL4, KL3, KL2, KL1, K1, K2, K3,
     gammaL4, gammaL3, gammaL2, gammaL1, gamma1, gamma2, gamma3,
     pL3, pL2, pL1, p0, p1, p2,
     alphaL4, alphaL3, alphaL2, alphaL1, alpha1, alpha2, alpha3) = args2z
    if rho < rhoL3:
        return (1 + alphaL4) * rho + KL4 / (gammaL4 - 1) * rho**gammaL4
    elif rhoL3 <= rho < rhoL2:
        return (1 + alphaL3) * rho + KL3 / (gammaL3 - 1) * rho**gammaL3
    elif rhoL2 <= rho < rhoL1:
        return (1 + alphaL2) * rho + KL2 / (gammaL2 - 1) * rho**gammaL2
    elif rhoL1 <= rho < rho0:
        return (1 + alphaL1) * rho + KL1 / (gammaL1 - 1) * rho**gammaL1
    elif rho0 <= rho < rho1:
        return (1 + alpha1) * rho + K1 / (gamma1 - 1) * rho**gamma1
    elif rho1 <= rho < rho2:
        return (1 + alpha2) * rho + K2 / (gamma2 - 1) * rho**gamma2
    else:
        return (1 + alpha3) * rho + K3 / (gamma3 - 1) * rho**gamma3

@numba.njit
def tov_numba(r, y, pmin, args2z):
    mS, pS = y
    if pS <= pmin or r == 0:
        return np.array([0.0, 0.0])
    eps = epsOfP_numba(pS, args2z)
    r2 = r * r
    m_over_r = mS / r
    one_minus_2m_over_r = 1 - 2 * m_over_r
    if one_minus_2m_over_r <= 0:
        return np.array([0.0, 0.0])
    denom = r * one_minus_2m_over_r
    dm_dr = four_pi * eps * r2
    dp_dr = - (pS + eps) * (mS + four_pi * pS * r2) / denom
    return np.array([dm_dr, dp_dr])




def epsOfP(p, p1, gamma1, gamma2, gamma3):

        rho1 = 10**14.7 / rhosol
        rho2 = 10**15.0 / rhosol

        K1 = p1 / rho1**gamma1
        K2 = K1 * rho1**(gamma1 - gamma2)
        K3 = K2 * rho2**(gamma2 - gamma3)

        # Initialize alpha parameters
        epsL4 = 0.0
        alphaL4 = 0.0
        epsL3 = (1 + alphaL4) * rhoL3 + KL4 / (gammaL4 - 1) * rhoL3**gammaL4
        alphaL3 = epsL3 / rhoL3 - 1 - KL3 / (gammaL3 - 1) * rhoL3**(gammaL3 - 1)
        epsL2 = (1 + alphaL3) * rhoL2 + KL3 / (gammaL3 - 1) * rhoL2**gammaL3
        alphaL2 = epsL2 / rhoL2 - 1 - KL2 / (gammaL2 - 1) * rhoL2**(gammaL2 - 1)
        epsL1 = (1 + alphaL2) * rhoL1 + KL2 / (gammaL2 - 1) * rhoL1**gammaL2
        alphaL1 = epsL1 / rhoL1 - 1 - KL1 / (gammaL1 - 1) * rhoL1**(gammaL1 - 1)
        rho0 = (KL1 / K1)**(1 / (gamma1 - gammaL1))
        eps0 = (1 + alphaL1) * rho0 + KL1 / (gammaL1 - 1) * rho0**gammaL1
        alpha1 = eps0 / rho0 - 1 - K1 / (gamma1 - 1) * rho0**(gamma1 - 1)
        eps1 = (1 + alpha1) * rho1 + K1 / (gamma1 - 1) * rho1**gamma1
        alpha2 = eps1 / rho1 - 1 - K2 / (gamma2 - 1) * rho1**(gamma2 - 1)
        eps2 = (1 + alpha2) * rho2 + K2 / (gamma2 - 1) * rho2**gamma2
        alpha3 = eps2 / rho2 - 1 - K3 / (gamma3 - 1) * rho2**(gamma3 - 1)

        pL3 = KL3 * rhoL3**gammaL3
        pL2 = KL2 * rhoL2**gammaL2
        pL1 = KL1 * rhoL1**gammaL1
        p0 = KL1 * rho0**gammaL1
        p2 = K2 * rho2**gamma2



        if p < pL3:
            return (1 + alphaL4) * (p / KL4)**(1 / gammaL4) + p / (gammaL4 - 1)
        elif pL3 <= p < pL2:
            return (1 + alphaL3) * (p / KL3)**(1 / gammaL3) + p / (gammaL3 - 1)
        elif pL2 <= p < pL1:
            return (1 + alphaL2) * (p / KL2)**(1 / gammaL2) + p / (gammaL2 - 1)
        elif pL1 <= p < p0:
            return (1 + alphaL1) * (p / KL1)**(1 / gammaL1) + p / (gammaL1 - 1)
        elif p0 <= p < p1:
            return (1 + alpha1) * (p / K1)**(1 / gamma1) + p / (gamma1 - 1)
        elif p1 <= p < p2:
            return (1 + alpha2) * (p / K2)**(1 / gamma2) + p / (gamma2 - 1)
        else:
            return (1 + alpha3) * (p / K3)**(1 / gamma3) + p / (gamma3 - 1)

        
def plot_histograms(data, original_values, bins=30, labels=None, title=None):
    plt.figure(figsize=(14, 5))
    plt.rcParams.update({'font.size': 14})

    num_features = data.shape[1]
    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        feature_data = data[:, i]
        plt.axvline(original_values[i], color='red', linestyle='dashed', linewidth=1.5, label='Original Value')

        # Plot histogram for current feature
        plt.hist(feature_data, bins=bins, alpha=0.7, label=labels[i] if labels and i < len(labels) else f"Feature {i+1}")
        plt.title(labels[i] if labels and i < len(labels) else f"Feature {i+1}")
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend()

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_histograms_NOV(data, bins=30, labels=None, title=None):
    plt.figure(figsize=(14, 5))
    plt.rcParams.update({'font.size': 14})

    num_features = data.shape[1]
    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        feature_data = data[:, i]

        plt.hist(feature_data, bins=bins, alpha=0.7, label=labels[i] if labels and i < len(labels) else f"Feature {i+1}")
        plt.title(labels[i] if labels and i < len(labels) else f"Feature {i+1}")
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend()

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def remove_negative_rows(arr):
    mask = np.all(arr >= 0, axis=1)
    filtered_array = arr[mask]
    return filtered_array

def mode(data):
    (sorted_data, idx, counts) = np.unique(data, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    return data[index]

def add_noise_to_dataframe(df, row_index=None, col_name=None, noise_level=0.05):
    rows = df.index if row_index is None else row_index
    columns = df.columns if col_name is None else col_name

    if isinstance(rows, int):
        rows = [rows]

    if isinstance(columns, str):
        columns = [columns]

    for row in rows:
        for col in columns:
            noise_factor = 1 + np.random.uniform(-noise_level, noise_level)
            df.at[row, col] *= noise_factor

    return df

def add_noise_and_recalculate_compactness(df, col_name, noise_level=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
    for col in col_name:
        noise_factor = 1 + np.random.uniform(-noise_level, noise_level, size=len(df))
        df[col] = df[col] * noise_factor

    df['cc'] = 1.47 * df['massa'] / df['raggio']

    return df
def mr_pred_from_eos_pred(p_values,
                          median_eps_values,  # median E(p)
                          lower_16th,         # lower 16% bound for E(p)
                          upper_84th,         # upper 84% bound for E(p)
                          lower_5th,          # lower 5% bound for E(p)
                          upper_95th):        # upper 95% bound for E(p)
    
    p_values          = np.array(p_values)
    median_eps_values = np.array(median_eps_values)
    lower_16th        = np.array(lower_16th)
    upper_84th        = np.array(upper_84th)
    lower_5th         = np.array(lower_5th)
    upper_95th        = np.array(upper_95th)

    # Build interpolators for each EoS
    epsilon_median_of_p  = interp1d(p_values, median_eps_values,  kind='linear', fill_value='extrapolate')
    epsilon_lower_68_of_p= interp1d(p_values, lower_16th,         kind='linear', fill_value='extrapolate')
    epsilon_upper_68_of_p= interp1d(p_values, upper_84th,         kind='linear', fill_value='extrapolate')
    epsilon_lower_90_of_p= interp1d(p_values, lower_5th,          kind='linear', fill_value='extrapolate')
    epsilon_upper_90_of_p= interp1d(p_values, upper_95th,         kind='linear', fill_value='extrapolate')

    # TOV equations
    def tov_equations(r, y, epsilon_of_p):

        P = y[0]
        m = y[1]
        if P <= 0 or r == 0:
            return [0.0, 0.0]
        eps = epsilon_of_p(P)
        dPdr = - (G * (eps + P / c**2) * (m + 4.0 * pi * r**3 * P / c**2)) / \
                (r * (r - 2.0 * G * m / c**2))
        dmdr = 4.0 * pi * eps * r**2
        return [dPdr, dmdr]

    # Event function to stop integration when pressure reaches zero
    def stop_integration(r, y, epsilon_of_p):
        return y[0]  # y[0] = P
    stop_integration.terminal = True
    stop_integration.direction = -1

    def solve_tov(Pc, epsilon_of_p):
        r0 = 1e-4  # small starting radius
        eps_c = epsilon_of_p(Pc)
        # approximate mass inside the tiny sphere
        m0 = (4.0/3.0) * pi * eps_c * r0**3
        y0 = [Pc, m0]
        r_span = [r0, 2e7]  # integrate up to some large radius (in cm)

        sol = solve_ivp(
            fun=tov_equations,
            t_span=r_span,
            y0=y0,
            args=(epsilon_of_p,),
            events=stop_integration,
            max_step=1000,
            dense_output=True
        )

        # Final results
        r = sol.t
        P = sol.y[0]
        m = sol.y[1]

        R = r[-1]
        M = m[-1]

        return M, R, r, P, m

    Pc_values = np.logspace(33.7, 37.0, 50)

    # Dictionaries to store M-R data for each EoS
    MR_data = {
        'median':   {'M': [], 'R': []},
        'lower_68': {'M': [], 'R': []},
        'upper_68': {'M': [], 'R': []},
        'lower_90': {'M': [], 'R': []},
        'upper_90': {'M': [], 'R': []}
    }

    for Pc in Pc_values:
        # Median
        try:
            M, R, _, _, _ = solve_tov(Pc, epsilon_median_of_p)
            MR_data['median']['M'].append(M / Msun)
            MR_data['median']['R'].append(R / 1e5)  # convert cm -> km
        except Exception as e:
            print(f"Median EoS: Error at Pc={Pc:.2e}: {e}")

        # Lower 68%
        try:
            M, R, _, _, _ = solve_tov(Pc, epsilon_lower_68_of_p)
            MR_data['lower_68']['M'].append(M / Msun)
            MR_data['lower_68']['R'].append(R / 1e5)
        except Exception as e:
            print(f"Lower 68% EoS: Error at Pc={Pc:.2e}: {e}")

        # Upper 68%
        try:
            M, R, _, _, _ = solve_tov(Pc, epsilon_upper_68_of_p)
            MR_data['upper_68']['M'].append(M / Msun)
            MR_data['upper_68']['R'].append(R / 1e5)
        except Exception as e:
            print(f"Upper 68% EoS: Error at Pc={Pc:.2e}: {e}")

        # Lower 90%
        try:
            M, R, _, _, _ = solve_tov(Pc, epsilon_lower_90_of_p)
            MR_data['lower_90']['M'].append(M / Msun)
            MR_data['lower_90']['R'].append(R / 1e5)
        except Exception as e:
            print(f"Lower 90% EoS: Error at Pc={Pc:.2e}: {e}")

        # Upper 90%
        try:
            M, R, _, _, _ = solve_tov(Pc, epsilon_upper_90_of_p)
            MR_data['upper_90']['M'].append(M / Msun)
            MR_data['upper_90']['R'].append(R / 1e5)
        except Exception as e:
            print(f"Upper 90% EoS: Error at Pc={Pc:.2e}: {e}")

    # Convert all lists to numpy arrays
    for key in MR_data:
        MR_data[key]['M'] = np.array(MR_data[key]['M'])
        MR_data[key]['R'] = np.array(MR_data[key]['R'])

    # ----------------------------
    # PLOTTING WITH NESTED BANDS
    # ----------------------------

    def fill_confidence_region(R_lower, M_lower, R_upper, M_upper,
                               color, alpha, label, zorder=1):
        idx_l = np.argsort(R_lower)
        Rl_sorted = R_lower[idx_l]
        Ml_sorted = M_lower[idx_l]

        idx_u = np.argsort(R_upper)
        Ru_sorted = R_upper[idx_u]
        Mu_sorted = M_upper[idx_u]
        
        R_polygon = np.concatenate([Rl_sorted, Ru_sorted[::-1]])
        M_polygon = np.concatenate([Ml_sorted, Mu_sorted[::-1]])

        plt.fill(R_polygon, M_polygon, color=color, alpha=alpha,
                 label=label, zorder=zorder)

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})

    fill_confidence_region(
        MR_data['lower_90']['R'], MR_data['lower_90']['M'],
        MR_data['upper_90']['R'], MR_data['upper_90']['M'],
        color='green', alpha=0.3,
        label='90% Confidence Interval', zorder=1
    )

    fill_confidence_region(
        MR_data['lower_68']['R'], MR_data['lower_68']['M'],
        MR_data['upper_68']['R'], MR_data['upper_68']['M'],
        color='blue', alpha=0.5,
        label='68% Confidence Interval', zorder=2
    )

    plt.plot(
        MR_data['median']['R'], MR_data['median']['M'],
        '-k', label='Median EoS', zorder=3
    )
    plt.plot(
        MR_data['lower_68']['R'], MR_data['lower_68']['M'],
        '-.k', zorder=3
    )
    plt.plot(
        MR_data['upper_68']['R'], MR_data['upper_68']['M'],
        '-.k', zorder=3
    )
    plt.plot(
        MR_data['lower_90']['R'], MR_data['lower_90']['M'],
        '--k', zorder=3
    )
    plt.plot(
        MR_data['upper_90']['R'], MR_data['upper_90']['M'],
        '--k', zorder=3
    )
    plt.xlabel('R (km)')
    plt.ylabel(r'M (M$_\odot$)')
    plt.title('Mass-Radius Relation with Nested Confidence Intervals')
    plt.grid(True)
    plt.legend()
    plt.show()

    return MR_data


def df_column_switch(df, column1, column2):
    temp = df[column1].copy()
    df[column1] = df[column2]
    df[column2] = temp
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df.columns = i

    return df

def log(x):
    return np.log(x)

def cgspressure(x):
    c = 2.9979e10
    G = 6.67408e-8
    Msun = 1.989e33
    return x*c**8/(G**3*Msun*Msun)

def safe_normalize(data, mean, std):
    std_zero_mask = std < 1e-8
    std_safe = torch.where(std_zero_mask, torch.ones_like(std), std)
    normalized_data = (data - mean) / std_safe
    normalized_data = torch.where(std_zero_mask, torch.ones_like(data), normalized_data)
    return normalized_data
def denormalize(tensor, mean, std):
    return tensor * std + mean

def eos_plot(eosParams):
    (label, p1CGS, gamma1, gamma2, gamma3) = eosParams
    p_values = np.logspace(30, 36, 500)  # Range of p values
    p1 = p1CGS/Psol
    # Compute energy densities for each pressure value
    eps_values = [epsOfP(p/Psol, p1, gamma1, gamma2, gamma3)*rhosol for p in p_values]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    plt.plot(p_values, eps_values, label="Energy Density vs Pressure", color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Pressure (p)")
    plt.ylabel("Energy Density (ε)")
    plt.title("Energy Density as a Function of Pressure")
    plt.legend()
    plt.xlim(1e30,1e36)
    plt.ylim(1e11,3e15)
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    
def process_eos(eosParams, imax, rho1l, rho2l):
    (label, p1CGS, gamma1, gamma2, gamma3) = eosParams

    p1 = p1CGS / (rhosol * c**2)
    rho1 = 10**14.7 / rhosol
    rho2 = 10**15.0 / rhosol

    K1 = p1 / rho1**gamma1
    K2 = K1 * rho1**(gamma1 - gamma2)
    K3 = K2 * rho2**(gamma2 - gamma3)

    # Initialize alpha parameters
    epsL4 = 0.0
    alphaL4 = 0.0
    epsL3 = (1 + alphaL4) * rhoL3 + KL4 / (gammaL4 - 1) * rhoL3**gammaL4
    alphaL3 = epsL3 / rhoL3 - 1 - KL3 / (gammaL3 - 1) * rhoL3**(gammaL3 - 1)
    epsL2 = (1 + alphaL3) * rhoL2 + KL3 / (gammaL3 - 1) * rhoL2**gammaL3
    alphaL2 = epsL2 / rhoL2 - 1 - KL2 / (gammaL2 - 1) * rhoL2**(gammaL2 - 1)
    epsL1 = (1 + alphaL2) * rhoL1 + KL2 / (gammaL2 - 1) * rhoL1**gammaL2
    alphaL1 = epsL1 / rhoL1 - 1 - KL1 / (gammaL1 - 1) * rhoL1**(gammaL1 - 1)
    rho0 = (KL1 / K1)**(1 / (gamma1 - gammaL1))
    eps0 = (1 + alphaL1) * rho0 + KL1 / (gammaL1 - 1) * rho0**gammaL1
    alpha1 = eps0 / rho0 - 1 - K1 / (gamma1 - 1) * rho0**(gamma1 - 1)
    eps1 = (1 + alpha1) * rho1 + K1 / (gamma1 - 1) * rho1**gamma1
    alpha2 = eps1 / rho1 - 1 - K2 / (gamma2 - 1) * rho1**(gamma2 - 1)
    eps2 = (1 + alpha2) * rho2 + K2 / (gamma2 - 1) * rho2**gamma2
    alpha3 = eps2 / rho2 - 1 - K3 / (gamma3 - 1) * rho2**(gamma3 - 1)

    pL3 = KL3 * rhoL3**gammaL3
    pL2 = KL2 * rhoL2**gammaL2
    pL1 = KL1 * rhoL1**gammaL1
    p0 = KL1 * rho0**gammaL1
    p2 = K2 * rho2**gamma2

    global argsz
    global args2z

    argsz = (rhoL3, rhoL2, rhoL1, rho0, rho1, rho2, KL4, KL3, KL2, KL1,
            K1, K2, K3, gammaL4, gammaL3, gammaL2, gammaL1,
            gamma1, gamma2, gamma3)

    args2z = (rhoL3, rhoL2, rhoL1, rho0, rho1, rho2, KL4, KL3, KL2, KL1,
             K1, K2, K3, gammaL4, gammaL3, gammaL2, gammaL1,
             gamma1, gamma2, gamma3, pL3, pL2, pL1, p0, p1, p2,
             alphaL4, alphaL3, alphaL2, alphaL1, alpha1, alpha2, alpha3)

    # Define rhs(p)
    def energydensity(p):
        if np.isscalar(p):
            return epsOfP_numba(p, args2z)
        else:
            return np.array([epsOfP_numba(ps, args2z) for ps in p])

    # Define epss(rho)
    def pressure(rho):
        if np.isscalar(rho):  # Check if rho is a single value
            return PofRho_numba(rho, argsz)
        else:
            return np.array([PofRho_numba(rhos, argsz) for rhos in rho])

    MRpoli = []
    for i in range(1, imax + 1):
        rho0 = rho0f(i, rho1l, rho2l, imax)
        p0 = PofRho_numba(rho0 / rhosol, argsz)
        M = np.zeros(imax+1)
        R = np.zeros(imax+1)
        L = np.zeros(imax+1)
        C = np.zeros(imax+1)


        def ToV(r,y): 
            P,m = y[0],y[1]

            eps=energydensity(P)
            dPdr = -(eps + P)*(m + 4.0*pi*r**3*P)/(r*(r - 2.0*m))
            dmdr = 4*np.pi*eps*(r**2)
            return [dPdr,dmdr]

        def tidal_y(r,yy,funcs):
            eden,p,m,dedp=funcs
            y=yy
            Q = 4*pi*((1 - 2*m(r)/r)**(-1)) *(5*eden(r) + 9*p(r) + dedp(r) *(eden(r) + p(r))) - 6*((1 - 2*m(r)/r)**(-1))/r**2 - (2*((1 - 2*m(r)/r)**(-1))*(m(r)+ 4*pi*p(r)*r**3)/r**2)**2
            dydr=(r**-1)*(-y**2 - y*((1 - 2*m(r)/r)**(-1))*(1 + 4*pi*r**2*(p(r)-eden(r))) - (r**2 )*Q)

            return dydr

        def solve_ToV(rho0):

            rmin = 0.001
            rspan = (1e-3,1e5)
            P0 = pressure(rho0) #BC on pressure
            eps0 = epsOfRho_numba(rho0,args2z)
            m0 = 4.0/3.0*np.pi*rmin**3*eps0

            y0 = [P0, m0]
            #y = odeint(ToV, y0, r)
            soln = solve_ivp(ToV, rspan, y0, method='RK45',max_step=0.1)
            r = soln.t
            p = soln.y[0]
            M = soln.y[1]

            eden = energydensity(p)

            mass = M[-1]
            radius = r[-1]*rsol/1e5

            return [mass,radius,(r,eden, p, M)]

        def solve_tidal(rho0):
            M, R, funcs  = solve_ToV(rho0)
            r, eden, p, m = funcs

            edenint = UnivariateSpline(r, eden, k=1,s=0)
            pressint = UnivariateSpline(r, p, k=1,s=0)
            massint = UnivariateSpline(r, m, k=1,s=0)
            dedr = edenint.derivative()
            dpdr = pressint.derivative()

            def dedpint(r):
                return dedr(r) / dpdr(r)

            rspan = (min(r),max(r))
            y0=[2.0]

            # solution = solve_ivp(tidal_y, rspan, y0, args=([edenint, pressint, massint,dedpint],), max_step=1)
            solution = solve_ivp(tidal_y, rspan, y0,method='LSODA', args=([edenint, pressint, massint, dedpint],))

            Y=solution.y[0]
            y=Y[-1]
            C = compactness = M*rsol /(10**5 *R)

            k2 = 8 / 5 * C ** 5 * (1 - 2 * C) ** 2 * (2 + 2 * C * (y - 1) - y) * (
                  2 * C * (6 - 3 * y + 3 * C * (5 * y - 8)) + 4 * C ** 3 * (
                    13 - 11 * y + C * (3 * y - 2) + 2 * C ** 2 * (1 + y)) + 3 * (1 - 2 * C) ** 2 * (2 - y + 2 * C * (y - 1)) * (
                    np.log(1 - 2 * C))) ** (-1)

            Lns = 2*k2*C**(-5) /3;

            return [M,R,Lns,C]
        M, R, Lns, C = solve_tidal(rho0/rhosol)
        MRpoli.append([p1CGS, gamma1, gamma2, gamma3, M, R, rho0, np.log(Lns), C])



    # Filter out invalid entries
    mr = [entry for entry in MRpoli if all(x >= 0 for x in entry)]
    return mr


def encode_eos(model, eos_tensor, rho_0):
    model.train()
    with torch.no_grad():
        eos_tensor2 = torch.cat([torch.tensor(eos_tensor), torch.tensor([np.log(rho_0)])])
        norm_tensor = safe_normalize(eos_tensor2, input_mean, input_std)
        reconstructed_features = model.encoder(norm_tensor.to(device))
        denorm_reconstructed = denormalize(reconstructed_features.to(device), target_mean.to(device), target_std.to(device))
    return denorm_reconstructed

def decode_latent_features(model, latent_tensor):
    model.train()
    with torch.no_grad():
        reconstructed_features = model.decoder(latent_tensor.to(device))
        denorm_reconstructed_features = denormalize(reconstructed_features.to(device), input_mean.to(device), input_std.to(device))
    return denorm_reconstructed_features


def gammas_from_mr(model, mrmr,mc):
    length = len(mrmr)
    tens = torch.zeros(length, 4, dtype=torch.float64)

    for i in range(length):
        tens[i] = torch.tensor(mrmr[i,:])

    latent_features_samples = []
    decoded_outputs_samples = []

    for i in range(length):
        for _ in range(mc):
            trial_tensor_normalized = safe_normalize(tens[i], target_mean, target_std)
            decoded_output = decode_latent_features(model,trial_tensor_normalized)
            decoded_outputs_samples.append(decoded_output.cpu().numpy()[0,:])

    return decoded_outputs_samples

def mr_prediction(model, p1, gamma1, gamma2, gamma3, rho1l, rho2l, switch, arbitrary_string=""):
    gammass = [np.log(p1), gamma1, gamma2, gamma3]
    masse = []
    raggi = []
    tidals = []
    compattezze = []

    MRtab = process_eos(["label", np.exp(gammass[0]), gammass[1], gammass[2], gammass[3]], 50, rho1l, rho2l)
    # eos_plot(["label", np.exp(gammass[0]), gammass[1], gammass[2], gammass[3]])
    masseO = [row[4] for row in MRtab]
    raggiO = [row[5] for row in MRtab]
    tidalsO = [row[7] for row in MRtab]
    tidalsO = np.exp(tidalsO)
    compattezzeO = [row[8] for row in MRtab]

    mc=0

    for epsilon in np.logspace(np.log10(rho1l), np.log10(rho2l), 100):
        if mc==1:
            for _ in np.arange(1,100,10):
                encoded = encode_eos(model, gammass, epsilon)
                masse.append(encoded.cpu().numpy()[0,0])
                raggi.append(encoded.cpu().numpy()[0,1])
                tidals.append(np.exp(encoded.cpu().numpy()[0,2]))
                compattezze.append(encoded.cpu().numpy()[0,3])
        else:
            encoded = encode_eos(model, gammass, epsilon)
            masse.append(encoded.cpu().numpy()[0,0])
            raggi.append(encoded.cpu().numpy()[0,1])
            tidals.append(np.exp(encoded.cpu().numpy()[0,2]))
            compattezze.append(encoded.cpu().numpy()[0,3])

    fig, axs = plt.subplots(2, 1, figsize=(7, 10)) 
    plt.rcParams.update({'font.size': 14})
    #(ax1, ax2), (ax3, ax4) = axs
    (ax1), (ax2) = axs
    ax1.scatter(raggi, masse, c='red', label='Original', alpha=1, s=10)
    ax1.plot(raggiO, masseO, c='black', label='Original', alpha=1)
    ax1.set_xlabel(r'R (km)')
    ax1.set_ylabel(r'M ($M_\odot$)')
    ax1.grid(True)

    ax2.scatter(compattezze, np.log10(tidals), c='red', label='Original', alpha=1, s=10)
    ax2.plot(compattezzeO, np.log10(tidalsO), c='black', label='Original', alpha=1)
    ax2.set_xlabel("C")
    ax2.set_ylabel(r'$\Lambda$')
    ax2.grid(True)

    
    # Plot 1: Mass-Radius
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(raggi, masse, c='red', label='Predicted', alpha=1, s=10)
    ax1.plot(raggiO, masseO, c='black', label='Original', alpha=1)
    ax1.set_xlabel(r'R (km)')
    ax1.set_ylabel(r'M ($M_\odot$)')
    ax1.grid(True)
    ax1.legend()

    if switch == 1:
        pdf_filename1 = f"MR_Mass_Radius_p1_{p1}_gamma1_{gamma1}_gamma2_{gamma2}_gamma3_{gamma3}_{arbitrary_string}.pdf"
        plt.savefig(pdf_filename1, format='pdf')
        print(f"Mass-Radius Figure saved as {pdf_filename1}")
    plt.close(fig1)

    # Plot 2: Compactness-Tidal Lambda
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(compattezze, np.log10(tidals), c='red', label='Predicted', alpha=1, s=10)
    ax2.plot(compattezzeO, np.log10(tidalsO), c='black', label='Original', alpha=1)
    ax2.set_xlabel("C")
    ax2.set_ylabel(r'$\log_{10}(\Lambda)$')
    ax2.grid(True)
    ax2.legend()

    if switch == 1:
        pdf_filename2 = f"MR_Compactness_Tidal_p1_{p1}_gamma1_{gamma1}_gamma2_{gamma2}_gamma3_{gamma3}_{arbitrary_string}.pdf"
        plt.savefig(pdf_filename2, format='pdf')
        print(f"Compactness-Tidal Figure saved as {pdf_filename2}")
    plt.close(fig2)
    
def plot_eos_prediction(xx, p1_c, g1_c, g2_c, g3_c, switch, arbitrary_string=""): 
    p_log_values = np.linspace(25, 36, 200)
    p_values = 10 ** p_log_values

    if p1_c != 0 and g1_c != 0 and g2_c != 0 and g3_c != 0:
        eps_values_real = [
            epsOfP(p / Psol, p1_c / Psol, g1_c, g2_c, g3_c) * rhosol
            for p in p_values
        ]

    median_eps_values = []
    lower_16th = []
    upper_84th = []
    lower_5th = []
    upper_95th = []

    for p in p_values:
        eps_values = []
        for i in range(len(xx)):
            try:
                p1 = np.exp(xx[i][0]) / Psol
                gamma1 = xx[i][1]
                gamma2 = xx[i][2]
                gamma3 = xx[i][3]
                eps = epsOfP(p / Psol, p1, gamma1, gamma2, gamma3) * Density
                
                # We add do add this check for few problematic points in gw170817 data that
                # statistically can give an overlflow
                if not np.isnan(eps) and np.isfinite(eps) and eps > 0:
                    eps_values.append(eps)
            except (OverflowError, ValueError, RuntimeWarning):
                # Skip problematic points
                continue

        if eps_values: 
            log_eps_values = np.log10(eps_values)
            median_log_eps = np.median(log_eps_values)
            lower_16th_log_eps = np.percentile(log_eps_values, 16)
            upper_84th_log_eps = np.percentile(log_eps_values, 84)
            lower_5th_log_eps = np.percentile(log_eps_values, 5)
            upper_95th_log_eps = np.percentile(log_eps_values, 95)

            median_eps_values.append(10 ** median_log_eps)
            lower_16th.append(10 ** lower_16th_log_eps)
            upper_84th.append(10 ** upper_84th_log_eps)
            lower_5th.append(10 ** lower_5th_log_eps)
            upper_95th.append(10 ** upper_95th_log_eps)
        else:
            median_eps_values.append(np.nan)
            lower_16th.append(np.nan)
            upper_84th.append(np.nan)
            lower_5th.append(np.nan)
            upper_95th.append(np.nan)

    p_values = np.array(p_values)
    median_eps_values = np.array(median_eps_values)
    lower_16th = np.array(lower_16th)
    upper_84th = np.array(upper_84th)
    lower_5th = np.array(lower_5th)
    upper_95th = np.array(upper_95th)

    valid_indices = ~np.isnan(median_eps_values)
    p_values = p_values[valid_indices]
    median_eps_values = median_eps_values[valid_indices]
    lower_16th = lower_16th[valid_indices]
    upper_84th = upper_84th[valid_indices]
    lower_5th = lower_5th[valid_indices]
    upper_95th = upper_95th[valid_indices]

    if p1_c != 0 and g1_c != 0 and g2_c != 0 and g3_c != 0:
        eps_values_real = np.array(eps_values_real)

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    plt.loglog(p_values, median_eps_values, label='Median ε(p)', color='black')
    if p1_c != 0 and g1_c != 0 and g2_c != 0 and g3_c != 0:
        plt.loglog(p_values, eps_values_real, label='Real EoS', color='red')

    plt.fill_between(
        p_values,
        lower_16th,
        upper_84th,
        color='blue',
        alpha=0.5,
        label='68% Confidence Interval'
    )

    plt.fill_between(
        p_values,
        lower_5th,
        upper_95th,
        color='green',
        alpha=0.3,
        label='90% Confidence Interval'
    )

    plt.xlabel(r'p (dyne/cm$^2$)')
    plt.ylabel(r'$\epsilon$ (g/cm$^3$)')
    plt.grid(True, which="both", ls="--")
    plt.xlim(1e31, 1e36)
    plt.ylim(4e12, 3e15)
    plt.legend()
    if switch == 1:
        pdf_filename = f"EoS_Prediction_p1_{p1_c}_g1_{g1_c}_g2_{g2_c}_g3_{g3_c}_{arbitrary_string}.pdf"
        plt.savefig(pdf_filename, format='pdf')
        plt.close()
        print(f"Figure saved as {pdf_filename}")
    
    plt.show()
    return p_values, median_eps_values, lower_16th, upper_84th, lower_5th, upper_95th


def plot_all_eos_prediction(xx):
    p_log_values = np.linspace(30, 36, 100)  
    p_values = 10 ** p_log_values  

    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(xx))) 

    for i in range(len(xx)):
        p1 = np.exp(xx[i][0])/Psol
        gamma1 = xx[i][1]
        gamma2 = xx[i][2]
        gamma3 = xx[i][3]

        eps_values = []
        for p in p_values:
            eps = epsOfP(p / Psol, p1, gamma1, gamma2, gamma3) * Density
            eps_values.append(eps)

        plt.plot(p_values, eps_values, label=f'EoS {i+1}', color=colors[i])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e31, 1e36)
    plt.ylim(4e12, 3e15)
    plt.xlabel(r'p (dyne/cm$^2$)')
    plt.ylabel(r'$\epsilon$ (g/cm$^3$)')
    plt.title("EoS Curves")
    plt.grid(True, which="both", ls="--")
    # Save the plot if needed
    # plt.savefig('eos_curves.png')

    plt.show()
