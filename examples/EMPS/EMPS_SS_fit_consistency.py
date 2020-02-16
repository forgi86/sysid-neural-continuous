import matplotlib
matplotlib.use("TkAgg")
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join("..", ".."))
from torchid.ssmodels_ct import MechanicalStateSpaceSystem
from torchid.ss_simulator_ct import ExplicitRKSimulator, ForwardEulerSimulator
from torch.utils.tensorboard import SummaryWriter  # requires tensorboard

def scale_pos(q_unsc):
    q_sc = (q_unsc - 0.125)/0.125
    return q_sc


def unscale_pos(q_sc):
    q_unsc = q_sc*0.125 + 0.125
    return q_unsc


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 40000  # gradient-based optimization steps
    batch_size = 32 # batch size
    t_fit = 2e-3  # fitting on t_fit ms of data
    alpha = 1e3  # fit/consistency trade-off constant
    lr = 1e-5  # learning rate
    test_freq = 100  # print message every test_freq iterations

    # Load dataset
    df_data = pd.read_csv(os.path.join("data", "DATA_EMPS_SC.csv"))
    time_exp = np.array(df_data[["time_exp"]]).astype(np.float32)
    q_ref = np.array(df_data[["q_ref"]]).astype(np.float32)
    q_meas = np.array(df_data[["q_meas"]]).astype(np.float32)
    v_est = np.array(df_data[["v_est"]]).astype(np.float32)
    u_in = np.array(df_data[["u_in"]]).astype(np.float32)
    ts = np.mean(np.diff(time_exp.ravel())) #time_exp[1] - time_exp[0]

    x_est = np.zeros((q_ref.shape[0], 2), dtype=np.float32)
    x_est[:, 0] = np.copy(q_meas[:, 0])
    x_est[:, 1] = np.copy(v_est[:, 0])

    # Hidden velocity variable
    x_hidden_fit = torch.tensor(x_est, dtype=torch.float32, requires_grad=True)  # hidden state is an optimization variable

    y_fit = q_meas
    u_fit = u_in
    time_fit = time_exp

    # y and u to torch
    y_torch_fit = torch.tensor(y_fit)
    u_torch_fit = torch.tensor(u_fit)

    # Setup neural model structure
    ss_model = MechanicalStateSpaceSystem(n_feat=64, init_small=True, typical_ts=ts)
    nn_solution = ForwardEulerSimulator(ss_model, ts=ts)

    # Setup optimizer
    params_net = list(nn_solution.ss_model.parameters())
    params_hidden = [x_hidden_fit]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': lr},
    ], lr=lr*10)

    LOSS = []
    writer = SummaryWriter("logs")
    start_time = time.time()
    # Training loop

    #scripted_nn_solution = torch.jit.script(nn_solution)
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Compute fit loss
        err_fit = x_hidden_fit[:, [0]] - y_torch_fit
        loss_fit = 1000*torch.mean(err_fit**2)

        # Compute consistency loss

        DX = ts/2 * ( ss_model(x_hidden_fit[1:, :], u_torch_fit[1:, :]) + ss_model(x_hidden_fit[0:-1, :],  u_torch_fit[0:-1, :]) )
        err_consistency = x_hidden_fit[1:, :] - x_hidden_fit[0:-1, :] - DX
        err_consistency_scaled = err_consistency
        loss_consistency = 1000*torch.mean(err_consistency_scaled**2)

        # Compute trade-off loss
        loss = loss_fit + alpha*loss_consistency

        # Statistics
        LOSS.append(loss.item())
        writer.add_scalar("loss", loss, itr)
        writer.add_scalar("loss_consistency", loss_consistency, itr)
        writer.add_scalar("loss_fit", loss_fit, itr)
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Tradeoff Loss {loss:.4f}   Consistency Loss {alpha*loss_consistency:.7f}   Fit Loss {loss_fit:.7f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 182 seconds

    if not os.path.exists("models"):
        os.makedirs("models")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filename = f"model_SS_consistency.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))


    # Plot figures
    if not os.path.exists("fig"):
        os.makedirs("fig")

    # In[Plot loss]

    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig_name = f"EMPS_SS_loss_consistency.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # In[Plot hidden]

    x_hidden_fit_np = x_hidden_fit.detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_est[:, 0], 'b', label='Measured')
    ax[0].plot(x_hidden_fit_np[:, 0], 'r', label='Hidden')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(x_est[:, 1], 'k', label='Estimated')
    ax[1].plot(x_hidden_fit_np[:, 1], 'r', label='Hidden')
    ax[1].legend()
    ax[1].grid(True)

    # In[Simulate]
    y_val = np.copy(y_fit)
    u_val = np.copy(u_fit)

    x0_val = np.array(x_est[0, :])
    x0_val[1] = 0.0
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(u_val)

    with torch.no_grad():
        x_sim_torch_val = nn_solution(x0_torch_val[None, :], u_torch_val[:, None, :])
        x_sim_torch_val = x_sim_torch_val.squeeze(1)
        x_sim_val = x_sim_torch_val.detach().numpy()
        q_sim = x_sim_val[:, 0]
        v_sim = x_sim_val[:, 1]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7.5))
    #ax[0].plot(time_exp, q_ref,  'k',  label='$q_{\mathrm{ref}}$')
    ax[0].plot(time_exp, q_meas, 'k', label='$q_{\mathrm{meas}}$')
    ax[0].plot(time_exp, q_sim, 'r', label='$q_{\mathrm{sim}}$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_ylabel("Position (m)")

    ax[1].plot(time_exp, v_est,  'k--',  label='$v_{\mathrm{est}}$')
    ax[1].plot(time_exp, v_sim,  'r',  label='$v_{\mathrm{sim}}$')
    ax[1].grid(True)
    ax[1].set_ylabel("Velocity (m/s)")

    ax[2].plot(time_exp, u_in, 'k', label='$u_{in}$')
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Voltage (V)")
    ax[2].grid(True)
    ax[2].set_xlabel("Time (s)")


