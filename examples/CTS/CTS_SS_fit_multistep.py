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
import torchid
from torchid.ssmodels_ct import CascadedTanksOverflowNeuralStateSpaceModel
from torchid.ss_simulator_ct import ForwardEulerSimulator, ExplicitRKSimulator
from torch.utils.tensorboard import SummaryWriter # requires tensorboard
from datetime import datetime
import shutil

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 100000  # gradient-based optimization steps
    seq_len = 128  # subsequence length m
    batch_size = 64  # batch size
    alpha = 1.0  # regularization weight
    lr = 1e-4  # learning rate
    test_freq = 100  # print message every test_freq iterations
    val_freq = 100

    # Load dataset
    df_data = pd.read_csv(os.path.join("data", "dataBenchmark.csv"))
    u_id = np.array(df_data[['uEst']]).astype(np.float32)
    y_id = np.array(df_data[['yEst']]).astype(np.float32)
    ts = df_data['Ts'][0].astype(np.float32)
    time_exp = np.arange(y_id.size).astype(np.float32)*ts

    x_est = np.zeros((time_exp.shape[0], 2), dtype=np.float32)
    x_est[:, 1] = np.copy(y_id[:, 0])

    # Hidden state variable
    x_hidden_fit = torch.tensor(x_est, dtype=torch.float32, requires_grad=True)  # hidden state is an optimization variable

    # Fit variables
    y_fit = y_id
    u_fit = u_id
    time_fit = time_exp

    # Setup neural model structure
    ss_model = CascadedTanksOverflowNeuralStateSpaceModel(n_feat=100)
    nn_solution = ForwardEulerSimulator(ss_model, ts=ts)

    # Setup optimizer
    params_net = list(nn_solution.ss_model.parameters())
    params_hidden = [x_hidden_fit]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': lr},
    ], lr=lr*10)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, min_lr=1e-6, verbose=True)

    # Batch extraction funtion
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

    # Scale loss with respect to the initial one
    with torch.no_grad():
        batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_sim = nn_solution(batch_x0_hidden, batch_u)
        batch_y_sim = batch_x_sim[:, :, [1]]
        traced_nn_solution = torch.jit.trace(nn_solution, (batch_x0_hidden, batch_u))
        err_init = batch_y_sim - batch_y
        scale_error = torch.sqrt(torch.mean(err_init**2, dim=(0, 1)))

    LOSS = []
    LOSS_CONSISTENCY = []
    LOSS_FIT = []
    LOSS_SIM = []

    # Log tensorboard data with generating scripts
    time_str = datetime.now().strftime("%Y_%m%d_%H%M")
    log_dir = os.path.join("logs", time_str)
    os.makedirs(log_dir)
    log_dir_src = os.path.join(log_dir, "src")
    os.makedirs(log_dir_src)
    shutil.copy(__file__, log_dir_src)
    shutil.copytree(torchid.__path__[0], os.path.join(log_dir_src, "torchid"))
    log_dir_tb = os.path.join(log_dir, "tb")
    writer = SummaryWriter(log_dir_tb)
    start_time = time.time()
    # Training loop

    #scripted_nn_solution = torch.jit.script(nn_solution)
    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Simulate
        batch_t, batch_x0_hidden, batch_u, batch_y, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_sim = traced_nn_solution(batch_x0_hidden, batch_u) # 52 seconds RK | 13 FE

        # Compute fit loss
        batch_y_sim = batch_x_sim[:, :, [1]]
        err_fit = batch_y_sim - batch_y
        err_fit_scaled = err_fit/scale_error[0]
        loss_fit_sim = torch.mean(err_fit_scaled**2)
        loss_fit = loss_fit_sim

        # Compute consistency loss
        err_consistency = batch_x_sim - batch_x_hidden
        err_consistency_scaled = err_consistency/scale_error
        loss_consistency = torch.mean(err_consistency_scaled**2)

        # Compute trade-off loss
        if itr > 20000:
            loss = loss_fit + alpha*loss_consistency
        else:
            loss = loss_fit

        # Statistics
        writer.add_scalars("opt_losses", {
                           "total_loss": loss,
                           "loss_fit": loss_fit,
                           "loss_consistency": alpha*loss_consistency}, itr)

        writer.add_scalars("learning_rates", {
                           "log_lr_net": np.log10(optimizer.param_groups[0]['lr']),
                           "log_lr_hidden": np.log10(optimizer.param_groups[1]['lr'])}, itr )

        LOSS.append(loss.item())
        LOSS_CONSISTENCY.append(alpha*loss_consistency.item())
        LOSS_FIT.append(loss_fit.item())

        if itr % test_freq == 0:
            with torch.no_grad():
                x0_torch_fit = x_hidden_fit[0, :]
                x_sim_torch_fit = nn_solution(x0_torch_fit[None, :], torch.tensor(u_fit)[:, None, :])
                x_sim_torch_fit = x_sim_torch_fit.squeeze(1)
                y_sim_torch_fit = x_sim_torch_fit[:, [1]]
                err_sim_torch_fit = y_sim_torch_fit - torch.tensor(y_fit)
                loss_sim = torch.sqrt(torch.mean(err_sim_torch_fit**2))
                LOSS_SIM.append(loss_sim.item())
                writer.add_scalar("loss/loss_train_sim", loss_sim, itr)
                scheduler.step(loss_sim)
                print(f'Iter {itr} | Tradeoff Loss {loss:.4f}   Consistency Loss {alpha*loss_consistency:.4f}   Fit Loss {loss_fit:.4f}   Simulation Loss {loss_sim:.4f}')



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

    model_filename =  f"model_custom_SS_{seq_len}step.pkl"
    hidden_filename = f"hidden_custom_SS_{seq_len}step.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))
    torch.save(x_hidden_fit, os.path.join("models", hidden_filename))

    # Plot figures
    if not os.path.exists("fig"):
        os.makedirs("fig")

    # In[Plot loss]
    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, label='tot')
    ax.plot(LOSS_FIT, label='fit')
    ax.plot(LOSS_CONSISTENCY, label='consistency')
    ax.grid(True)
    plt.legend()
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig_name = f"CTS_SS_loss_{seq_len}step_noise.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # In[Hidden variable plot]
    x_hidden_fit_np = x_hidden_fit.detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    #ax[0].plot(x_est[:, 0], 'b', label='Measured')
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

    #x0_val = np.array(x_est[0, :])
    #x0_val[1] = 0.0
    x0_val = x_hidden_fit[0, :].detach().numpy() # initial state had to be estimated, according to the dataset description
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(u_val)

    with torch.no_grad():
        x_sim_torch_val = nn_solution(x0_torch_val[None, :], u_torch_val[:, None, :])
        x_sim_torch_val = x_sim_torch_val.squeeze(1)
        x_sim_val = x_sim_torch_val.detach().numpy()
        y_sim = x_sim_val[:, 1]



    # In[Plot simulation]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7.5))
    # ax[0].plot(time_exp, q_ref,  'k',  label='$q_{\mathrm{ref}}$')
    ax[0].plot(time_exp, y_id, 'k', label='$q_{\mathrm{meas}}$')
    ax[0].plot(time_exp, y_sim, 'r', label='$q_{\mathrm{sim}}$')
    ax[0].plot(time_exp, x_hidden_fit_np[:, 1], 'b', label='${x_{2}}^m$') # may be excluded from the plot
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_ylabel("Voltage (V)")

    ax[1].plot(time_exp, x_hidden_fit_np[:, 0], 'b', label='${x_{1}}^m$')
    ax[1].set_ylabel("$x_1$ (-)")
    ax[1].grid()

    ax[2].plot(time_exp, u_id, 'k', label='$u_{in}$')
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Voltage (V)")
    ax[2].grid(True)
    ax[2].set_xlabel("Time (s)")
