clear;
clc;
close all;

%% Read data table 

data_path = fullfile("data", "dataBenchmark.csv");
data_table = readtable(data_path);


u_in =  data_table.uEst;
y_meas = data_table.yEst;
ts = data_table.Ts(1);
time_exp =  (0:(length(u_in)-1))*ts;
time_exp = time_exp(:);

%% Identification data
data_id = iddata(y_meas,u_in,ts);
%% Fit model 
opt = n4sidOptions;
[model_id, x0_id] = n4sid(data_id, 2);%, 2)
%% Simulate model
opt = simOptions('InitialCondition',x0_id);
y_sim = sim(model_id, data_id, opt);
y_sim = y_sim.OutputData;

%% Plot data %%

figure()
plot(time_exp, y_meas, 'k');
hold on;
plot(time_exp, y_sim, 'b');
legend('True', 'Model');

%% Metrics
SSE = sum((y_meas - y_sim(:,1)).^2);
y_mean = mean(y_meas);
SST = sum((y_meas - y_mean).^2);
R_sq = 1 - SSE/SST;

RMSE = sqrt(mean((y_meas - y_mean).^2));

fprintf("Subspace fitting performance\n");
fprintf("R-squred:%.3f\n", R_sq)
fprintf("RMSE:%.3f\n", RMSE)



%% Read data table val

u_in =  data_table.uVal;
y_meas = data_table.yVal;
ts = data_table.Ts(1);
time_exp =  (0:(length(u_in)-1))*ts;
time_exp = time_exp(:);

%% Identification data
ts = mean(diff(time_exp));
data_val = iddata(y_meas,u_in,ts);

%% Simulate model
opt = simOptions('InitialCondition',x0_id);
y_sim = sim(model_id, data_val, x0_id);
y_sim = y_sim.OutputData;

%% Plot data %%

figure()
plot(time_exp, y_meas, 'k');
hold on;
plot(time_exp, y_sim(:,1), 'b');
legend('True', 'Model');

%% Metrics
SSE = sum((y_meas - y_sim(:,1)).^2);
y_mean = mean(y_meas);
SST = sum((y_meas - y_mean).^2);
R_sq = 1 - SSE/SST;

RMSE = sqrt(mean((y_meas - y_mean).^2));

fprintf("Subspace fitting performance\n");
fprintf("R-squred:%.3f\n", R_sq)
fprintf("RMSE:%.3f\n", RMSE)
