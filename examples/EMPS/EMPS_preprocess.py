import matplotlib
matplotlib.use("TkAgg")
import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join("..", ".."))
#from sklearn.preprocessing import StandardScaler


def scale_pos(q_unsc):
    """
    Scale the position of the EMPS dataset to the range [-1, 1].
    """
    q_sc = (q_unsc - 0.125)/0.125
    return q_sc

def unscale_pos(q_sc):
    """
    Unscale the position of the EMPS dataset to the original range.
    """
    q_unsc = q_sc*0.125 + 0.125
    return q_unsc


if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    #DATASET = "DATA_EMPS"
    DATASET = "DATA_EMPS_PULSES"

    SCALE_POS = True # If True, scale the dataset
    DECIMATE = 5  # Decimate data by a factor DECIMATE
    DIFF_FILTER = False  # If True, use a derivative filter for velocity. Otherwise, np.diff

    # In[Load dataset]

    emps_data = sp.io.loadmat(os.path.join("data", DATASET + ".mat"))
    q_ref = emps_data['qg'].astype(np.float32)
    q_meas = emps_data['qm'].astype(np.float32)
    u_in = emps_data['vir'].astype(np.float32)
    time_exp = emps_data['t'].astype(np.float32)
    if DATASET == 'DATA_EMPS_PULSES':
        force_dist = emps_data['pulses_N']
    else:
        force_dist = np.zeros(time_exp.shape)

    force_tot = u_in * emps_data['gtau'] + force_dist
    u_tot = u_in + force_dist/emps_data['gtau']

    # In[Scale Position]
    if SCALE_POS:
        q_ref = scale_pos(q_ref)
        q_meas = scale_pos(q_meas)

    # In[Compute velocity and acceleration]
    ts_orig = np.mean(np.diff(time_exp.ravel())) #time_exp[1] - time_exp[0]
    if DIFF_FILTER:
        # Design a differentiator filter to estimate unmeasured velocities from noisy, measured positions
        fs = 1 / ts_orig       # Sample rate, Hz
        cutoff = 10.0    # Desired cutoff frequency, Hz
        trans_width = 100  # Width of transition from pass band to stop band, Hz
        n_taps = 32      # Size of the FIR filter.
        taps = scipy.signal.remez(n_taps, [0, cutoff, cutoff + trans_width, 0.5 * fs], [2 * np.pi * 2 * np.pi * 10 * 1.5, 0], Hz=fs, type='differentiator')
        v_est = np.convolve(q_meas.ravel(), taps, 'same')
        v_est[0:n_taps] = v_est[n_taps + 1]
        v_est[-n_taps:] = v_est[-n_taps - 1]
        v_est = v_est.reshape(-1, 1)  # signal.lfilter(taps, 1, y_meas[:,0])*2*np.pi
    else: # Perform a simple numerical differentiation using np.diff
        v_est = np.concatenate((np.array([0]), np.diff(q_meas[:, 0])))
        v_est = v_est.reshape(-1, 1)/ts_orig

        a_est = np.concatenate((np.array([0]), np.diff(v_est[:, 0])))
        a_est = a_est.reshape(-1, 1)/ts_orig

    # In[Resample data]
    if DECIMATE > 1:
        q_ref = q_ref[0:-1:DECIMATE]
        q_meas = q_meas[0:-1:DECIMATE]
        v_est = v_est[0:-1:DECIMATE]
        a_est = a_est[0:-1:DECIMATE]
        u_in = u_in[0:-1:DECIMATE]
        time_exp = time_exp[0:-1:DECIMATE]
        force_dist = force_dist[0:-1:DECIMATE]
        force_tot = force_tot[0:-1:DECIMATE]
        u_tot = u_tot[0:-1:DECIMATE]

    ts = np.mean(np.diff(time_exp.ravel()))
    v_int = q_meas[0, :] + np.cumsum(v_est)*ts

    # In[plots]
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 10.5))
    ax[0].plot(time_exp, q_meas, 'k', label='$q_{\mathrm{meas}}$')
    ax[0].plot(time_exp, v_int, 'r', label='$q_{\mathrm{int}}$')

    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_ylabel("Position (m)")

    ax[1].plot(time_exp, v_est,  'k',  label='$v_{\mathrm{est}}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_ylabel("Velocity (m/s)")

    ax[2].plot(time_exp, a_est,  'k',  label='$v_{\mathrm{est}}$')
    ax[2].legend(loc='upper right')
    ax[2].grid(True)
    ax[2].set_ylabel("Acceleration (m/s^2)")

    ax[3].plot(time_exp, u_in, 'k', label='$u_{in}$')
    ax[3].plot(time_exp, u_in, 'k', label='$u_{tot}$')
    ax[3].legend(loc='upper right')
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Voltage (V)")
    ax[3].grid(True)
    ax[3].set_xlabel("Time (s)")

    ax[4].plot(time_exp, force_dist, 'k', label='$F_{d}$')
    ax[4].plot(time_exp, force_tot, 'r', label='$F_{t}$')
    ax[4].legend(loc='upper right')
    ax[4].set_xlabel("Time (s)")
    ax[4].set_ylabel("Force (N)")
    ax[4].grid(True)
    ax[4].set_xlabel("Time (s)")

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 7.5))
    ax = [ax]
    idx_stop = np.abs(v_est) < 0.2
    idx_move = np.abs(v_est) >= 0.2
    idx_1t = (time_exp > ts) & (time_exp < 100.0)
    idx_vp = v_est >= 0

    ax[0].plot(force_tot[idx_1t & idx_vp], a_est[idx_1t & idx_vp], 'k')
    ax[0].plot(force_tot[idx_1t & ~idx_vp], a_est[idx_1t & ~idx_vp], 'r')
    ax[0].plot(force_tot[1], a_est[1], 's')

    ax[0].grid(True)
    ax[0].set_xlabel("Input (V)")
    ax[0].set_ylabel("Acceleration (m/s^2)")

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 7.5))
    ax = [ax]
    ax[0].plot(a_est, u_in)
    ax[0].grid(True)
    ax[0].set_xlabel("Acceleration (m/s^2)")
    ax[0].set_ylabel("Input (V)")

    # In[save preprocessed data]
    arr_data = np.hstack((time_exp, q_ref, q_meas, v_est, u_in, u_tot))
    df_data = pd.DataFrame(arr_data, columns=["time_exp", "q_ref", "q_meas", "v_est", "u_in", "u_tot"])
    df_data.to_csv(os.path.join("data", DATASET + "_SC.csv"))
