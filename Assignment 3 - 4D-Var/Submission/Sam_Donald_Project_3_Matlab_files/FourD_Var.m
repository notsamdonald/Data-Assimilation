function [psi_grad] = FourD_Var(X_rk4_steps,X_rk4,X_obvs_ens,lambda_init,R,B,h,x0,xb,H_jacobian,H)
% 4D-Var cost function helper function
% TODO - create anon function (lots passed in at the moment)


lambda_minus = lambda_init;
for i = size(X_rk4_steps,4):-1:1
    lambda_plus = discrete_adj_rk4(lambda_minus, X_rk4_steps(:,:,:,end), h);
    lambda_minus = lambda_plus +  H_jacobian' * inv(R) * (H* X_rk4(:,i) -  X_obvs_ens(:,i,1));

end
psi_grad = inv(B)*(x0-xb)+lambda_minus;
end

