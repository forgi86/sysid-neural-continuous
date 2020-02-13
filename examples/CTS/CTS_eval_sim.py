import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join("..", ".."))
from torchid.ssmodels_ct import CascadedTanksOverflowNeuralStateSpaceModel
from torchid.ss_simulator_ct import ExplicitRKSimulator, ForwardEulerSimulator
from common import metrics

if __name__ == '__main__':

    plot_input = True

    dataset_type = 'val'

    #model_name = 'model_custom_SS_128step'
    #hidden_name = 'hidden_custom_SS_128step'

    model_name = 'model_SS_custom_hidden_integration'
    hidden_name = 'hidden_SS_custom_hidden_integration'

    # Load dataset
    df_data = pd.read_csv(os.path.join("data", "dataBenchmark.csv"))
    if dataset_type == 'id':
        u = np.array(df_data[['uEst']]).astype(np.float32)
        y = np.array(df_data[['yEst']]).astype(np.float32)
    else:
        u = np.array(df_data[['uVal']]).astype(np.float32)
        y = np.array(df_data[['yVal']]).astype(np.float32)

    ts = df_data['Ts'][0].astype(np.float32)
    time_exp = np.arange(y.size).astype(np.float32) * ts


    # Build validation data
    t_val_start = 0
    t_val_end = time_exp[-1]
    idx_val_start = int(t_val_start//ts)
    idx_val_end = int(t_val_end//ts)

    y_meas_val = y[idx_val_start:idx_val_end]
    u_val = u[idx_val_start:idx_val_end]
    time_val = time_exp[idx_val_start:idx_val_end]

    # Setup neural model structure
    ss_model = CascadedTanksOverflowNeuralStateSpaceModel(n_feat=100)
    nn_solution = ForwardEulerSimulator(ss_model, ts=ts) #ForwardEulerSimulator(ss_model, ts=ts)
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_name + ".pkl")))
    x_hidden_fit = torch.load(os.path.join("models", hidden_name + ".pkl"))

    # Evaluate the model in open-loop simulation against validation data
    x_0 = x_hidden_fit[0, :].detach().numpy() # initial state had to be estimated, according to the dataset description
    #x_0 = np.array([u_val[0], 0.0]).astype(np.float32)
    with torch.no_grad():
        x_sim_val_torch = nn_solution(torch.tensor(x_0[None, :]), torch.tensor(u_val[:, None, :]))
        x_sim_val_torch = x_sim_val_torch.squeeze(1)

    x_sim_val = np.array(x_sim_val_torch)
    y_sim_val = x_sim_val[:, [1]]


    # In[Plot results]
    if plot_input:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 5.5))
    else:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(5.7, 2.5))
        ax = [ax]

    idx_plot_start = 0
    idx_plot_end = time_val.size

    ax[0].plot(time_val[idx_plot_start:idx_plot_end], y_meas_val[idx_plot_start:idx_plot_end, 0], 'k', label='$y$')
    ax[0].plot(time_val[idx_plot_start:idx_plot_end], y_sim_val[idx_plot_start:idx_plot_end, 0], 'r--', label='$\hat{y}^{\mathrm{sim}}$')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].set_ylim([1.0, 12.0])
    ax[0].set_xlim([0, 1024*4 + 500])
    ax[0].grid(True)

    if plot_input:
        ax[1].plot(time_val[idx_plot_start:idx_plot_end], u_val[idx_plot_start:idx_plot_end, 0], 'k', label='$u$')
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Voltage (V)")
        #ax[1].set_ylim([-5, 5])
        ax[1].grid(True)

    # Plot all
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig_name = f"CTS_{dataset_type}_{model_name}.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # R-squared metrics
    R_sq = metrics.r_squared(y_sim_val, y_meas_val)
    rmse_sim = metrics.error_rmse(y_sim_val, y_meas_val)
    fit_index = metrics.fit_index(y_sim_val, y_meas_val)

    print(f"R-squared metrics: {R_sq}")
    print(f"RMSE-squared metrics: {rmse_sim}")
    print(f"fit index {fit_index}")
