function [cost,grad] = Q1_helper(x0, X_obvs_ens)
%Q1_HELPER Helper function to modified 4DVar runs to validate grad calcs
% via the adjoint

% This is effectively packaging the RK4 forward and adjoint backward calcs
% together

% Inital conditions [TODO - should be passed in]
x0_true = [-10.0375; -4.3845; 34.6514];
B0 = [12.4294, 12.4323, -0.2139; 12.4323, 16.0837, -0.0499; -0.2139, -0.0499, 14.7634];
H_jacobian = [1, 0, 0;
              0, 1, 0;
              0, 0, 0];
R = obvs_sigma.^2 * eye(3);

% Time steps
dt = 0.1; % time step unit
h = dt/200;
T_max = 0.1;
T = 0:dt:T_max;

% Running RK4 forward model
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

% Init lambda
lambda_init = H_jacobian' * inv(R) * (H* X_rk4(:,end) -  X_obvs_ens(:,end,1));

% Calculating cost and gradient
xb=x0_true;
cost = psi_cost(X, X_obvs_ens, xb, B0, R, H);
grad = FourD_Var(X_rk4_steps,X_rk4,X_obvs_ens,lambda_init,R,B0,h,x0,xb,H_jacobian,H);

end

