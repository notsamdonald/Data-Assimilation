function [cost,grad] = var4(x0, xb, X_obvs_ens, T_max, dt)
%VAR4 Summary of this function goes here
%   Detailed explanation goes here

% Time steps
%dt = 0.1; % time step unit
h = dt/200;
%T_max = 0.5;
T = 0:dt:T_max;

% Run the forward model and save RK4 steps
X_rk4 = zeros(3,round(T_max/dt) + 1);
X_rk4_steps = zeros(3, 4, round(dt/h), round(T_max/dt));
x = x0;
X_rk4(:, 1) = x;

for i = 1:(length(T)-1)
    [~, x_test, x_k] = rk4(x, T(i), T(i+1), h);
    x = x_test(:, end);
    X_rk4(:, i+1) = x;
    X_rk4_steps(:, :, :, i) = x_k;
end

% Adjoint model
obvs_sigma = 0.25;
R = obvs_sigma.^2 * eye(3); % Also can be passed in
H = [1, 0, 0;
    0, 1, 0;
    0, 0, 0]; % = H Jacob


% Misc constants not passed in
B0 = [12.4294, 12.4323, -0.2139; 12.4323, 16.0837, -0.0499; -0.2139, -0.0499, 14.7634];

lambda_init = H' * inv(R) * (H* X_rk4(:,end) -  X_obvs_ens(:,end,1));
cost = psi_cost(X_rk4, X_obvs_ens, xb, B0, R, H);
grad = FourD_Var(X_rk4_steps,X_rk4,X_obvs_ens,lambda_init,R,B0,h,x0,xb,H,H);
end

