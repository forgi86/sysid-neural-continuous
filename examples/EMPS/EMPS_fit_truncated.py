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
from torchid.ss_simulator_ct import RK4Simulator, ExplicitRKSimulator, ForwardEulerSimulator
from torch.utils.tensorboard import SummaryWriter  # requires tensorboard

# Truncated simulation error minimization method
if __name__ == '__main__':

    # In[Set seed for reproducibility]
    np.random.seed(0)
    torch.manual_seed(0)

    # In[Overall parameters]
    num_iter = 10000  # gradient-based optimization steps
    seq_len = 64  # subsequence length m
    batch_size = 32 # batch size
    t_fit = 2e-3  # fitting on t_fit ms of data
    alpha = 1  # regularization weight
    lr = 1e-4  # learning rate
    test_freq = 100  # print message every test_freq iterations

    # In[Load dataset]
    df_data = pd.read_csv(os.path.join("data", "DATA_EMPS_SC.csv"))
    time_exp = np.array(df_data[["time_exp"]]).astype(np.float32)
    q_ref = np.array(df_data[["q_ref"]]).astype(np.float32)
    q_meas = np.array(df_data[["q_meas"]]).astype(np.float32)
    v_est = np.array(df_data[["v_est"]]).astype(np.float32)
    u_in = np.array(df_data[["u_in"]]).astype(np.float32)
    ts = np.mean(np.diff(time_exp.ravel()))  #time_exp[1] - time_exp[0]

    # In[Init hidden state]
    x_est = np.zeros((q_ref.shape[0], 2), dtype=np.float32)
    x_est[:, 0] = np.copy(q_meas[:, 0])
    x_est[:, 1] = np.copy(v_est[:, 0])
    x_hidden_fit = torch.tensor(x_est, dtype=torch.float32, requires_grad=True) # hidden state is an optimization variable

    # In[Fit variables]
    y_fit = q_meas
    u_fit = u_in
    time_fit = time_exp

    # In[Setup neural model structure]
    ss_model = MechanicalStateSpaceSystem(n_feat=64, init_small=True, typical_ts=ts)
    nn_solution = RK4Simulator(ss_model, ts=ts)

    # In[Setup optimizer]
    params_net = list(nn_solution.ss_model.parameters())
    params_hidden = [x_hidden_fit]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': lr},
    ], lr=lr*10)


    # In[Batch extraction funtion]
    def get_batch(batch_size, seq_len):

        # Select batch indexes
        num_train_samples = u_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch samples indices
        batch_idx = batch_idx.T  # transpose indexes to obtain batches with structure (m, q, n_x)

        # Extract batch data
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_x_hidden = x_hidden_fit[[batch_idx]]
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_y = torch.tensor(y_fit[batch_idx])

        return batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden

    # In[Scale loss with respect to the initial one]
    with torch.no_grad():
        batch_t, batch_x0_hidden, batch_u, batch_x, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_sim = nn_solution(batch_x0_hidden, batch_u)
        traced_nn_solution = torch.jit.trace(nn_solution, (batch_x0_hidden, batch_u))
        err_init = batch_x_sim - batch_x
        scale_error = torch.sqrt(torch.mean(err_init**2, dim=(0, 1)))

    # In[Training loop]
    LOSS = []
    writer = SummaryWriter("logs")
    start_time = time.time()
    # Training loop
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_sim = traced_nn_solution(batch_x0_hidden, batch_u) # 52 seconds RK | 13 FE
        #batch_x_sim = nn_solution(batch_x0_hidden, batch_u) # 70 seconds RK | 13 FE
        #batch_x_sim = scripted_nn_solution(batch_x0_hidden, batch_u) # 71 seconds RK | 13 FE

        # Compute fit loss
        err_fit = batch_x_sim[:, :, [0]] - batch_y
        err_fit_scaled = err_fit/scale_error[0]
        loss_fit = torch.mean(err_fit_scaled**2)

        # Compute consistency loss
        err_consistency = batch_x_sim - batch_x_hidden
        err_consistency_scaled = err_consistency/scale_error
        loss_consistency = torch.mean(err_consistency_scaled**2)

        # Compute trade-off loss
        loss = loss_fit + alpha*loss_consistency

        # Statistics
        LOSS.append(loss.item())
        writer.add_scalar("loss", loss, itr)
        writer.add_scalar("loss_consistency", loss_consistency, itr)
        writer.add_scalar("loss_fit", loss_fit, itr)
        if itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Tradeoff Loss {loss:.4f}   Consistency Loss {loss_consistency:.4f}   Fit Loss {loss_fit:.4f}')

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 598 seconds

    if not os.path.exists("models"):
        os.makedirs("models")

    # In[Save model]
    if not os.path.exists("models"):
        os.makedirs("models")

    model_filename = f"model_SS_{seq_len}step_RK.pkl"
    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))

    # In[Plot loss]
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig_name = f"EMPS_SS_loss_{seq_len}step.pdf"
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


    # In[Plot simulation]

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
