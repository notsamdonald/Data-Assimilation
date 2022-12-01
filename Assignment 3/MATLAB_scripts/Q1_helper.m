function [cost,grad] = Q1_helper(x0)
%Q1_HELPER Summary of this function goes here
%   Detailed explanation goes here

% Setting seed for reproducibility
rng(381996)

% Importing Lorenz96 model
m = otp.lorenz63.presets.Canonical;

% Model configurations
n_states = 3;
n_obs_states = 2;

% Inital conditions
x0_true = [-10.0375; -4.3845; 34.6514];
B0 = [12.4294, 12.4323, -0.2139; 12.4323, 16.0837, -0.0499; -0.2139, -0.0499, 14.7634];

% Time steps
dt = 0.1; % time step unit
h = dt/200;
T_max = 0.1;
T = 0:dt:T_max;

% Array initalization
X_true = zeros(3,(T_max/dt) + 1);
% iterate through timesteps
x = x0_true;
X_true(:, 1) = x;
for i = 1:(length(T)-1)
    [~, x] = ode45(m.RHS.F, [T(i), T(i+1)], x);
    x = x(end, :);
    X_true(:, i+1) = x;
end


% Array initalization
X = zeros(3,(T_max/dt) + 1);
% iterate through timesteps
x = x0;
X(:, 1) = x;
for i = 1:(length(T)-1)
    [~, x] = ode45(m.RHS.F, [T(i), T(i+1)], x);
    x = x(end, :);
    X(:, i+1) = x;
end

% Generating observations

n_obs = 1;
% Apply observation operator and add random noise to reference trajectory
% Generating observation operator (observing first two states)

% Not 100% on this size (0 x3)
H = [1, 0, 0;
    0, 1, 0;
    0, 0, 0];

% Applying observation vector to reference trajectory
% H * Reference -> Observations, Clone Observatations + noise -> ensemble 
X_obvs = H * X_true;
X_obvs_ens = repmat(X_obvs, 1,1, n_obs);

% Generating and applying observation noise (for each ensemble member)
obvs_sigma = 0.025;
obvs_noise = normrnd(0,obvs_sigma,size(X_obvs_ens));  % (note normrnd(mu,sigma))
obvs_noise(3,:,:) = 0;
X_obvs_ens = X_obvs_ens + obvs_noise;


X_rk4 = zeros(3,(T_max/dt) + 1);
X_rk4_steps = zeros(3, 4, dt/h, T_max/dt);
x = x0;
X_rk4(:, 1) = x;

for i = 1:(length(T)-1)
    [~, x_test, x_k] = rk4(x, T(i), T(i+1), h);
    x = x_test(:, end);
    X_rk4(:, i+1) = x;
    X_rk4_steps(:, :, :, i) = x_k;
end

H_jacobian = [1, 0, 0;
              0, 1, 0;
              0, 0, 0];

R = obvs_sigma.^2 * eye(3);
lambda_init = H_jacobian' * inv(R) * (H* X_rk4(:,end) -  X_obvs_ens(:,end,1));

xb=x0_true; % Not 100% on this
cost = psi_cost(X, X_obvs_ens, xb, B0, R, H);
grad = FourD_Var(X_rk4_steps,X_rk4,X_obvs_ens,lambda_init,R,B0,h,x0,xb,H_jacobian,H);

end

