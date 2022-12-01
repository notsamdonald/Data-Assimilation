function [lambda_i] =  discrete_adj_rk4(lambda_k, rk_steps, h)
% TODO - pass in dxdy instead (actually dxdt)

sigma = 10;
beta = 8/3;
rho = 28;

fx = @(x) [-sigma, sigma, 0;
    (rho-x(3)), -1, -x(1);
    x(2), x(1), -beta]';

lambda = zeros(3,size(rk_steps,3)+ 1);
lambda(:,end) = lambda_k;

for k = size(rk_steps,3):-1:1
    u_4 = h*fx(rk_steps(:,4,k)) * (1/6) * lambda(:,k+1);
    u_3 = h*fx(rk_steps(:,3,k)) * ((1/3) * lambda(:,k+1) + u_4);
    u_2 = h*fx(rk_steps(:,2,k)) * ((1/3) * lambda(:,k+1) + (1/2) * u_3);
    u_1 = h*fx(rk_steps(:,1,k)) * ((1/6) * lambda(:,k+1) + (1/2) * u_2);

    lambda(:,k) = lambda(:,k+1) + u_1 + u_2 + u_3 + u_4;
end

% Return final lambda value
lambda_i = lambda(:,1);

