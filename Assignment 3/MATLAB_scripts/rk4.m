function [t,x, x_k] =  rk4(x0, t0, tf, h)
% TODO - pass in dxdy instead (actually dxdt)
t = t0:h:tf;
x = zeros(3,length(t));
x_k = zeros(3, 4, length(x)-1);
x(:, 1)= x0 ;

A = [0,0,0,0;
    0.5,0,0,0;
    0,0.5,0,0;
    0,0,1,0];

b = [1/6;1/3;1/3;1/6];

c = [0;0.5;0.5;0];

sigma = 10;
beta = 8/3;
rho = 28;

dxdt = @(x) [sigma*(x(2)-x(1));
                x(1)*(rho-x(3))-x(2);
                x(1)*x(2)-beta*x(3)];



for i = 1:(length(x)-1)
    k_1 = x(:,i);
    k_2 = x(:,i) + h * 0.5 * dxdt(k_1);
    k_3 = x(:, i) + h * 0.5 * dxdt(k_2);
    k_4 = x(:, i) + h * dxdt(k_3);

    x(:, i+1) = x(:, i) + h*(1/6)*(dxdt(k_1)+2*dxdt(k_2)+ ...
                                    2*dxdt(k_3)+dxdt(k_4));

    x_k(:,:,i) = [k_1, k_2, k_3, k_4];
end
end