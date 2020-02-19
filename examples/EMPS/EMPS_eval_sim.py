import matplotlib
matplotlib.use('TKAgg')
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join("..", ".."))
from torchid.ssmodels_ct import MechanicalStateSpaceSystem
from torchid.ss_simulator_ct import RK4Simulator, ExplicitRKSimulator, ForwardEulerSimulator
from EMPS_preprocess import unscale_pos
from common import metrics

if __name__ == '__main__':

    #matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    matplotlib.rc('text', usetex=True)

    plot_input = True

    model_name = 'model_SS_64step_RK'
#    model_name = 'model_SS_soft'

    #dataset = 'id'
    dataset = 'test'

    if dataset == 'id':
        dataset_filename = 'DATA_EMPS_SC.csv' # used for identification
    elif dataset == 'test':
        dataset_filename = 'DATA_EMPS_PULSES_SC.csv' # used for test

    # Load dataset
    df_data = pd.read_csv(os.path.join("data", dataset_filename))
    time_exp = np.array(df_data[["time_exp"]]).astype(np.float32)
    q_ref = np.array(df_data[["q_ref"]]).astype(np.float32)
    q_meas = np.array(df_data[["q_meas"]]).astype(np.float32)
    v_est = np.array(df_data[["v_est"]]).astype(np.float32)
    u_in = np.array(df_data[["u_in"]]).astype(np.float32)
    x_est = np.zeros((q_ref.shape[0], 2), dtype=np.float32)
    x_est[:, 0] = np.copy(q_meas[:, 0])
    x_est[:, 1] = np.copy(v_est[:, 0])

    ts = np.mean(np.diff(time_exp.ravel()))

    # In[Build validation data]
    t_val_start = 0
    t_val_end = time_exp[-1]
    idx_val_start = int(t_val_start//ts)
    idx_val_end = int(t_val_end//ts)

    u_in_val = u_in[idx_val_start:idx_val_end]
    q_meas_val = q_meas[idx_val_start:idx_val_end]
    v_est_val = v_est[idx_val_start:idx_val_end]
    x_est_val = x_est[idx_val_start:idx_val_end, :]
    time_val = time_exp[idx_val_start:idx_val_end]

    # In[Setup neural model structure]
    ss_model = MechanicalStateSpaceSystem(n_feat=64, init_small=True, typical_ts=ts)
    nn_solution = RK4Simulator(ss_model, ts=ts)
    model_filename = f"{model_name}.pkl"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # In[Evaluate the model in open-loop simulation against validation data]
    x_0 = np.array([q_meas_val[0], 0.0]).astype(np.float32)
    with torch.no_grad():
        x_sim_torch = nn_solution(torch.tensor(x_0), torch.tensor(u_in_val))
        loss = torch.mean(torch.abs(x_sim_torch - torch.tensor(x_est_val)))
    x_sim = np.array(x_sim_torch)

    # In[Plot results]

    # if plot_input:
    #     fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 6.5))
    # else:
    #     fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 4.5))
    #
    # idx_plot_start = 0
    # idx_plot_end = time_val.size
    #
    # ax[0].plot(time_val[idx_plot_start:idx_plot_end], unscale_pos(q_meas_val[idx_plot_start:idx_plot_end,0]), 'k',  label='$v_C$')
    # ax[0].plot(time_val[idx_plot_start:idx_plot_end], unscale_pos(x_sim[idx_plot_start:idx_plot_end, 0]),'r--', label='$\hat{v}^{\mathrm{sim}}_C$')
    # ax[0].legend(loc='upper right')
    # ax[0].set_xlabel("Time (s)")
    # ax[0].set_ylabel("Position (m)")
    # ax[0].set_ylim([0, 0.3])
    # ax[0].grid(True)
    #
    # ax[1].plot(time_val[idx_plot_start:idx_plot_end], v_est_val[idx_plot_start:idx_plot_end,0], 'k',  label='$v_C$')
    # ax[1].plot(time_val[idx_plot_start:idx_plot_end], x_sim[idx_plot_start:idx_plot_end, 1],'r--', label='$\hat{v}^{\mathrm{sim}}_C$')
    # ax[1].legend(loc='upper right')
    # ax[1].set_xlabel("Time (s)")
    # ax[1].set_ylabel("Speed (m/s)")
    # ax[1].set_ylim([-1.5, 1.5])
    # ax[1].grid(True)
    #
    # if plot_input:
    #     ax[2].plot(time_val[idx_plot_start:idx_plot_end], u_in_val[idx_plot_start:idx_plot_end,0], 'k',  label='$u_{in}$')
    #     ax[2].legend(loc='upper right')
    #     ax[2].set_xlabel("Time (s)")
    #     ax[2].set_ylabel("Voltage (V)")
    #     ax[2].set_ylim([-5, 5])
    #     ax[2].grid(True)
    #
    # fig_name = f"RLC_SS_{dataset}_{model_filename}.pdf"
    # fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    if plot_input:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6.5))
    else:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 2.5))
        ax = [ax]

    idx_plot_start = 0 #1000
    idx_plot_end = time_val.size

    ax[0].plot(time_val[idx_plot_start:idx_plot_end], unscale_pos(q_meas_val[idx_plot_start:idx_plot_end,0]), 'k',  label='$p$')
    ax[0].plot(time_val[idx_plot_start:idx_plot_end], unscale_pos(x_sim[idx_plot_start:idx_plot_end, 0]),'r--', label='${p}^{\mathrm{sim}}$')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (m)")
    ax[0].set_ylim([0, 0.3])
    ax[0].grid(True)
    ax[0].set_xlim([0, 21])

    if plot_input:
        ax[1].plot(time_val[idx_plot_start:idx_plot_end], u_in_val[idx_plot_start:idx_plot_end,0], 'k',  label='$\\tau$')
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Force (N)")
        ax[1].set_ylim([-10, 10])
        ax[1].grid(True)

    fig_name = f"EMPS_SS_{dataset}_{model_name}.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # In[compute metrics]
    R_sq_idx = metrics.r_squared(x_est_val, x_sim)
    rmse_idx = metrics.error_rmse(x_est_val, x_sim)
    fit_idx = metrics.fit_index(x_est_val, x_sim)

    print(f"R-squared metrics: {R_sq_idx}")
    print(f"RMSE-squared metrics: {rmse_idx}")
    print(f"fit index {fit_idx}")

