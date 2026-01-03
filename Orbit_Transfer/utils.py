
# =====================================================================
# Description
# =====================================================================
"""
Simulate a satellite launched from Earth's position into a heliocentric trajectory,
    while feeling gravitational acceleration from:
      - the Sun (primary)
      - Mars (additional perturbation)

    This is a direct translation of the provided MATLAB code, including its numerical scheme:
      - fixed timestep integration
      - position update: x += v*dt + 0.5*a*dt^2
      - acceleration recomputed from gravity
      - velocity update: v += a*dt
"""


# =====================================================================
# Import Libraries
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from matplotlib.animation import FuncAnimation, PillowWriter


# =====================================================================
# Global Constants
# =====================================================================
G           = 6.674e-20         # usiversal gravitational constant (in km^3/kg.s^2)
AU          = 1.495978707e8     # conversion of astronomical unit to km
mass_sun    = 1.989e30          # mass of Sun (in kg)
mass_mars   = 0.95e28           # mass of Mars (in kg)
mu_sun      = G * mass_sun      # mu of Sun (in km^3/s^2)
mu_mars     = G * mass_mars     # mu of Mars (in km^3/s^2)
Ve          = 29.5              # Tangential velocity of Earth wrt Sun (km/s)


# =====================================================================
# Solar Conics
# =====================================================================
def solar_conic():
    """
    Earth and Mars Orbital Parameters
    """
    # Earth and Mars Orbital Parameters
    rp = np.array([147.1, 206.6]) * 1e6  # perihelion distance (km)
    ra = np.array([152.1, 249.2]) * 1e6  # aphelion distance (km)

    e = (ra - rp) / (rp + ra)            # eccentricity
    a = (rp + ra) / 2.0                  # semi-major axis (km)
    b = a * np.sqrt(1.0 - e**2)          # semi-minor axis (km)
    l = a * (1.0 - e**2)                 # semi-latus rectum (km)

    # orbital period in days
    T = np.sqrt((4.0 * np.pi**2) * a**3 / mu_sun) / (24.0 * 3600.0)

    return e, a, b, l, T


# =====================================================================
# Solar Initialise
# =====================================================================
def solar_initialize(epoch, dt, true_anomaly):
    """
    Purpose
    -------
    1) Get orbital parameters (e, a, b, l, T) from solar_conic().
    2) Advance initial true anomalies (theta) from t=0 to 'epoch' using a
       simple areal-velocity based update rule.
    3) Generate one full orbital period worth of sampled positions for:
         - Earth (Pe): shape (2, Te/dt)
         - Mars  (Pm): shape (2, Tm/dt)
       Positions are stored in AU (Astronomical Units), not km.

    Parameters
    ----------
    epoch : float
        Time to advance the initial planet positions "before" generating full-period ephemerides.
        Units is "days".
    dt : float
        Sampling step.
        Unit is "days".
    true_anomaly : numpy.ndarray | shape : (2,) 
        True Anomaly -> Initial position of Earth and Mars.
        Assuming Prihelion for both Earth and Mars is at theta=0.
        example -> theta_initial = np.array([0.0, np.pi/3.0]).
        Unit in "radians".

    Returns
    -------
    Pe : np.ndarray, shape (2, Ne)
        Earth positions (AU): [xe; ye]
    Pm : np.ndarray, shape (2, Nm)
        Mars positions (AU): [xm; ym]
    T : np.ndarray or sequence
        Orbital periods returned by solar_conic(); this code uses T[0] (Earth),
        T[1] (Mars) just like MATLAB uses T(1), T(2).
    """

    # Obtain orbital parameters of planets from helper function
    e, a, b, l, T = solar_conic()

    # ---------------------------------------------------------------------
    # Oribt Equation in Polar Coordinates (for Earth and Mars)
    # ---------------------------------------------------------------------
    theta   = true_anomaly.copy()    # unit in radians
    r       = l / (1.0 + e*np.cos(theta)) # radial distance (km)

    # ---------------------------------------------------------------------
    # Advance "theta" and "r" up to the given epoch
    # ---------------------------------------------------------------------
    if epoch >= 1:
        t_vals = np.arange(1.0, epoch + 1e-12, dt)  # small epsilon for endpoint
        for _ in t_vals:
            d_theta = (2*np.pi) * a * b * dt / (T * (r**2))
            theta   = (theta + d_theta) % (2*np.pi) # Reset logic: if theta > 2*pi, compute modulo
            r       = l / (1.0 + e * np.cos(theta)) # Update r based on new theta
            
    # ---------------------------------------------------------------------
    # Generate one full orbit worth of Earth samples
    # ---------------------------------------------------------------------
    Te = T[0]
    theta_e = theta[0]
    re = r[0]
    ae = a[0]
    be = b[0]
    ee = e[0]
    le = l[0]

    # Store position in Cartesian coordinates
    Ne      = int(Te / dt) # number of samples    
    xe      = np.zeros(Ne) # initialize x-coord storage array
    ye      = np.zeros(Ne) # initialize y-coord storage array
    xe[0]   = (re * np.cos(theta_e)) / AU # initial x-coord (AU)
    ye[0]   = (re * np.sin(theta_e)) / AU # initial y-coord (AU)

    # Compute x and y coordinates for one complete orbit
    for idx in range(1, Ne):
        d_theta = (2.0*np.pi) * ae * be * dt / (Te * (re**2))
        theta_e = (theta_e + d_theta) % (2*np.pi) # Reset logic: if theta > 2*pi, compute modulo
        re      = le / (1.0 + ee * np.cos(theta_e))
        xe[idx] = (re * np.cos(theta_e)) / AU
        ye[idx] = (re * np.sin(theta_e)) / AU

    # ---------------------------------------------------------------------
    # Generate one full orbit worth of Mars samples
    # ---------------------------------------------------------------------
    Tm = T[1]
    theta_m = theta[1]
    rm = r[1]
    am = a[1]
    bm = b[1]
    em = e[1]
    lm = l[1]

    # Store position in Cartesian coordinates
    Nm = int(Tm / dt)
    xm = np.zeros(Nm)
    ym = np.zeros(Nm)
    xm[0] = (rm * np.cos(theta_m)) / AU # initial x-coord (AU)
    ym[0] = (rm * np.sin(theta_m)) / AU # initial y-coord (AU)

    # Compute x and y coordinates for one complete orbit
    for idx in range(1, Nm):
        d_theta = (2.0 * np.pi) * am * bm * dt / (Tm * (rm**2))
        theta_m = (theta_m + d_theta) % (2*np.pi) # Reset logic: if theta > 2*pi, compute modulo
        rm      = lm / (1.0 + em * np.cos(theta_m))
        xm[idx] = (rm * np.cos(theta_m)) / AU
        ym[idx] = (rm * np.sin(theta_m)) / AU

    # ---------------------------------------------------------------------
    # Store positions in 2xN arrays
    # ---------------------------------------------------------------------
    Pe = np.vstack([xe, ye])
    Pm = np.vstack([xm, ym])

    return Pe, Pm, T


# =====================================================================
# Launch Satellite
# =====================================================================
def launch_satellite(epoch, dt, C3, duration, true_anomaly):
    """
    Parameters
    ----------
    epoch : any
        Time to advance the initial planet positions "before" generating full-period ephemerides.
        Units is "days".
    dt : float
        Time-step in "days".
    C3 : float
        Extra tangential speed added to Earth's heliocentric speed (km/s).
    duration : float
        Total simulation duration in "days".

    Returns
    -------
    Ps : np.ndarray shape (2, N)
        Satellite trajectory in AU: [X; Y]
    Pe : np.ndarray
        Earth position history returned by solar_initialise()
    Pm : np.ndarray
        Mars position history returned by solar_initialise()
    T : np.ndarray or sequence
        Time-related array returned by solar_initialise()
        (This code uses T[1] which corresponds to MATLAB T(2)).
    """

    # -------------------------------------------------------------------------
    # Obtain initial conditions of Earth and Mars
    # -------------------------------------------------------------------------
    Pe, Pm, T = solar_initialize(epoch, dt, true_anomaly)

    # -------------------------------------------------------------------------
    # Initial parameters of satellite wrt Sun
    # -------------------------------------------------------------------------
    # (assuming satellite is on Earth)
    x = Pe[0, 0] * AU           # in km
    y = Pe[1, 0] * AU           # in km
    R = np.sqrt(x**2 + y**2)    # in km

    # Tangential velocity of satellite wrt Sun (km/s)
    V = Ve + C3**0.5 # in km/s

    # Angle of position vector wrt Sun
    theta = np.arctan2(y, x)    # unit in radians in range [-pi, pi]

    # Acceleration due to Sun gravity: a = -mu / r^2 (direction radially inward)
    acc = -mu_sun / R**2        # in km/s^2
    axs = acc * np.cos(theta)   # in km/s^2
    ays = acc * np.sin(theta)   # in km/s^2

    # -------------------------------------------------------------------------
    # Initial parameters of satellite wrt Mars
    # -------------------------------------------------------------------------
    # Mars position history for a (roughly) year-long trajectory
    xm = Pm[0, :] * AU # in km
    ym = Pm[1, :] * AU # in km

    # Initial parameters of satellite wrt Mars (perturbing acceleration)
    xsm = x - xm[0]
    ysm = y - ym[0]
    phi = np.arctan2(ysm, xsm)
    Rsm = np.sqrt(xsm**2 + ysm**2) # distance of satellite from Mars

    # Acceleration due to Mars gravity
    asm = -mu_mars / Rsm**2 # direction radially inward
    axm = asm * np.cos(phi)
    aym = asm * np.sin(phi)

    # -------------------------------------------------------------------------
    # Net acceleration on satellite due to Sun and Mars
    # -------------------------------------------------------------------------
    # Net initial acceleration on satellite
    ax = axs + axm
    ay = ays + aym

    # Initial heliocentric velocity components (km/s)
    # Note: velocity is considered only wrt the Sun.
    Vx = -V * np.sin(theta)
    Vy =  V * np.cos(theta)

    # -------------------------------------------------------------------------
    # Integration loop
    # -------------------------------------------------------------------------
    dt_days = dt # days
    dt      = dt * 24 * 3600.0  # seconds
    n_steps = int(duration / dt_days)

    X = np.zeros(n_steps)
    Y = np.zeros(n_steps)

    # The "Mars period" in samples:
    j = 0  # index for Mars position lookup
    mars_period_samples = int(T[1] / dt_days)

    # iterate over "duration"
    for i in range(n_steps):
        
        # Velocity update (dt in seconds)
        Vx = Vx + ax*dt
        Vy = Vy + ay*dt

        # Satellite position update wrt Sun (kinematic update)
        x = x + Vx*dt + 0.5*ax*(dt**2)
        y = y + Vy*dt + 0.5*ay*(dt**2)

        # Recompute Sun gravity acceleration at new location
        R       = np.sqrt(x**2 + y**2)
        acc     = -mu_sun / R**2
        theta   = np.arctan2(y, x)
        axs     = acc * np.cos(theta)
        ays     = acc * np.sin(theta)

        # -------------------------------------------------------------
        # Mars perturbation at this step
        # wraps j so it stays within one Mars period worth of samples.
        # -------------------------------------------------------------
        j += 1
        if j > mars_period_samples:
            j = 0

        xsm = x - xm[j]
        ysm = y - ym[j]
        phi = np.arctan2(ysm, xsm)
        Rsm = np.sqrt(xsm**2 + ysm**2)

        asm = -mu_mars / Rsm**2
        axm = asm * np.cos(phi)
        aym = asm * np.sin(phi)

        # Net acceleration
        ax = axs + axm
        ay = ays + aym

        # save trajectory
        X[i]    = x / AU # in AU
        Y[i]    = y / AU # in AU

    # satellite position history
    P_sat = np.vstack([X, Y])

    return P_sat, Pe, Pm, T


# =====================================================================
# Visualise Satellite
# =====================================================================
def visualise_satellite(epoch, dt, C3, duration, speed, true_anomaly):

    P_sat, Pe, Pm, T = launch_satellite(epoch, dt, C3, duration, true_anomaly)

    xe, ye = Pe[0, :], Pe[1, :]
    xm, ym = Pm[0, :], Pm[1, :]
    x_sat, y_sat = P_sat[0, :], P_sat[1, :]

    Te, Tm = T[0], T[1]

    # counters (MATLAB is 1-based; we use 0-based)
    inx_e = 0  # Earth
    inx_m = 0  # Mars
    inx_sat = 0  # Satellite

    n_e     = int(Te / dt)
    n_m     = int(Tm / dt)
    n_sat   = int(duration / dt)

    plt.ion()
    fig, ax = plt.subplots()

    while inx_sat < n_sat:
        ax.clear()

        # Sun, Earth, Mars, Satellite
        ax.plot(0, 0, 'yo') # Sun
        ax.plot(xe[inx_e], ye[inx_e], 'go') # Earth
        ax.plot(xm[inx_m], ym[inx_m], 'ro') # Mars
        ax.plot(x_sat[inx_sat], y_sat[inx_sat], '*') # Satellite
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Solar System')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.grid(True)
        # ax.axis("off")

        clear_output(wait=True)
        display(fig)

        inx_e   += speed
        inx_m   += speed
        inx_sat += speed

        if inx_e >= n_e:
            inx_e = 0
        if inx_m >= n_m:
            inx_m = 0

    # plt.ioff()
    # plt.show()
    return 


# =====================================================================
# Save as GIF
# =====================================================================
def save_satellite_gif(
        epoch, dt, C3, duration, speed, true_anomaly, 
        gif_path="orbit.gif", fps=30):
    
    Ps, Pe, Pm, T = launch_satellite(epoch, dt, C3, duration, true_anomaly)

    xe, ye = Pe[0, :], Pe[1, :]
    xm, ym = Pm[0, :], Pm[1, :]
    xs, ys = Ps[0, :], Ps[1, :]

    Te, Tm  = T[0], T[1]
    n_sat   = int(duration / dt)
    n_e     = int(Te / dt)
    n_m     = int(Tm / dt)

    # indices we will render as frames (k increments by "speed")
    k_vals = np.arange(0, n_sat, speed, dtype=int)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("Solar System")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    # Create artists once, then update their data each frame
    sun,    = ax.plot([0], [0], "yo")
    earth,  = ax.plot([xe[0]], [ye[0]], "go")
    mars,   = ax.plot([xm[0]], [ym[0]], "ro")
    sat,    = ax.plot([xs[0]], [ys[0]], "*")

    def update(frame_idx):
        k = k_vals[frame_idx]
        i = (k % n_e)
        j = (k % n_m)

        earth.set_data([xe[i]], [ye[i]])
        mars.set_data([xm[j]], [ym[j]])
        sat.set_data([xs[k]], [ys[k]])
        return earth, mars, sat, sun

    anim = FuncAnimation(
        fig,
        update,
        frames=len(k_vals),
        interval=int(1000 / fps),
        blit=True
    )

    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return gif_path

# Example:
# save_satellite_gif(epoch=10, dt=0.01, C3=3.5, duration=500, speed=100, gif_path="earth_to_mars.gif", fps=30)
