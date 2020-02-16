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
from torchid.ssmodels_ct import CascadedTanksOverflowNeuralStateSpaceModel
from torchid.ss_simulator_ct import ExplicitRKSimulator, ForwardEulerSimulator


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    num_iter = 100000  # gradient-based optimization steps
    t_fit = 2e-3  # fitting on t_fit ms of data
    alpha = 90000  # regularization weight
    lr = 1e-5  # learning rate
    test_freq = 100  # print message every test_freq iterations
    add_noise = True

    # Load dataset
    df_data = pd.read_csv(os.path.join("data", "dataBenchmark.csv"))
    u_id = np.array(df_data[['uEst']]).astype(np.float32)
    y_id = np.array(df_data[['yEst']]).astype(np.float32)
    ts_meas = df_data['Ts'][0].astype(np.float32)
    time_exp = np.arange(y_id.size).astype(np.float32) * ts_meas

    # Build initial state estimate
    x_est = np.zeros((time_exp.shape[0], 2), dtype=np.float32)
    x_est[:, 1] = np.copy(y_id[:, 0])

    # Create torch tensors
    x_hidden_fit_torch = torch.tensor(x_est, dtype=torch.float32, requires_grad=True)  # hidden state is an optimization variable
    y_fit_torch = torch.tensor(y_id, dtype=torch.float32)
    u_fit_torch = torch.tensor(u_id, dtype=torch.float32)

    # Build neural state-space model
    ts_integ = ts_meas  # fictitious sampling time, better for numerical reasons
    ss_model = CascadedTanksOverflowNeuralStateSpaceModel(n_feat=100)#, activation='tanh')
    nn_solution = ForwardEulerSimulator(ss_model, ts=ts_integ)
    #model_filename = f"model_SS_{64}step_noise.pkl"
    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

    # Setup optimizer
    params_net = list(ss_model.parameters())
    params_hidden = [x_hidden_fit_torch]
    optimizer = optim.Adam([
        {'params': params_net,    'lr': lr},
        {'params': params_hidden, 'lr': 100*lr},
    ], lr=lr)

    # Scale loss with respect to the initial one
    scale_error = torch.tensor([1.0]).float()

    LOSS = []
    LOSS_FIT = []
    LOSS_CONSISTENCY = []
    start_time = time.time()
    # Training loop

    for itr in range(0, num_iter):

        optimizer.zero_grad()

        # Compute fit loss
        y_hidden_fit = x_hidden_fit_torch[:, [1]]
        err_fit = y_hidden_fit - y_fit_torch
        err_fit_scaled = 1.0*err_fit/scale_error
        loss_fit = torch.mean(err_fit_scaled**2)

        # Compute consistency loss
        DX = ts_integ/2 * (ss_model(x_hidden_fit_torch[1:, :], u_fit_torch[1:, :]) +
                           ss_model(x_hidden_fit_torch[0:-1, :], u_fit_torch[0:-1, :]) ) # midpoint integration

        err_consistency = x_hidden_fit_torch[1:, :] - x_hidden_fit_torch[0:-1, :] - DX
        err_consistency_scaled = err_consistency/scale_error
        loss_consistency = torch.mean(err_consistency_scaled**2)

        hidden_norm = torch.mean(x_hidden_fit_torch[:, [0]]**2)
        loss_hidden_norm = (1.0 - hidden_norm)**2

        # Compute trade-off loss
        loss = loss_fit + alpha*loss_consistency #+ loss_hidden_norm

        # Statistics
        LOSS.append(loss.item())
        LOSS_FIT.append(loss_fit.item())
        LOSS_CONSISTENCY.append(loss_consistency.item())
        if itr % test_freq == 0:
            print(f'Iter {itr} | Tradeoff Loss {loss:.6f}   Consistency Loss {(alpha*loss_consistency):.6f}   Fit Loss {loss_fit:.6f} Loss Size Norm {loss_hidden_norm:.6f}')

        # Optimize
        loss.backward()
        optimizer.step()

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    model_filename =  f"model_SS_custom_hidden_integration.pkl"
    hidden_filename = f"hidden_SS_custom_hidden_integration.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))
    torch.save(x_hidden_fit_torch, os.path.join("models", hidden_filename))

    torch.save(ss_model.state_dict(), os.path.join("models", model_filename))

    if not os.path.exists("fig"):
        os.makedirs("fig")

    # Plot figures
    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS, 'k', label='ALL')
    ax.plot(LOSS_CONSISTENCY, 'r', label='CONSISTENCY')
    ax.plot(LOSS_FIT, 'b', label='FIT')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    fig_name = f"CTS_SS_loss_hidden_integration_noise.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    # Simulate
    y_val = np.copy(y_id)
    u_val = np.copy(u_id)

    #  initial state had to be estimated, according to the dataset description
    x0_val = x_hidden_fit_torch[0, :].detach().numpy()
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(u_val)

    with torch.no_grad():
        x_sim_torch_val = nn_solution(x0_torch_val[None, :], u_torch_val[:, None, :])
        x_sim_torch_val = x_sim_torch_val.squeeze(1)
        x_sim_val = x_sim_torch_val.detach().numpy()
        y_sim = x_sim_val[:, 1]

    # In[Simulation plot]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 7.5))
    #ax[0].plot(time_exp, q_ref,  'k',  label='$q_{\mathrm{ref}}$')
    ax[0].plot(time_exp, y_id, 'k', label='$q_{\mathrm{meas}}$')
    ax[0].plot(time_exp, y_sim, 'r', label='$q_{\mathrm{sim}}$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_ylabel("Voltage (V)")

    ax[1].plot(time_exp, u_id, 'k', label='$u_{in}$')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Voltage (V)")
    ax[1].grid(True)
    ax[1].set_xlabel("Time (s)")


    # Hidden variable plot
    x_hidden_fit_np = x_hidden_fit_torch.detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_est[:, 1], 'b', label='Measured')
    ax[0].plot(x_hidden_fit_np[:, 1], 'r', label='Hidden')
    ax[0].legend()
    ax[0].grid(True)

    #ax[1].plot(x_est[:, 1], 'k', label='Estimated')
    ax[1].plot(x_hidden_fit_np[:, 1], 'r', label='Hidden')
    ax[1].legend()
    ax[1].grid(True)
