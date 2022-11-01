function tmp = arenstorfOrbit_Hesstr_vec_f_py( ~, y, u, k )

	mu = 0.012277471;
	mu_hat = 1 - mu;

tmp(1,1) = - k(2)*(u(4)*(1/((mu_hat - y(1))^2 + y(2)^2)^(3/2) - ...
    (3*y(2)^2)/((mu_hat - y(1))^2 + y(2)^2)^(5/2) - (3*mu_hat*(2*mu + ...
    2*y(1)))/(2*((mu + y(1))^2 + y(2)^2)^(5/2)) + ...
    (15*mu_hat*y(2)^2*(2*mu + 2*y(1)))/(2*((mu + y(1))^2 + ...
    y(2)^2)^(7/2))) + u(3)*((3*y(2)*(mu_hat - y(1)))/((mu_hat - ...
    y(1))^2 + y(2)^2)^(5/2) - (3*mu_hat*y(2))/((mu + y(1))^2 + ...
    y(2)^2)^(5/2) + (15*mu_hat*y(2)*(2*mu + 2*y(1))*(mu + ...
    y(1)))/(2*((mu + y(1))^2 + y(2)^2)^(7/2)))) - ...
    k(1)*(u(4)*((3*y(2)*(2*mu_hat - 2*y(1)))/(2*((mu_hat - y(1))^2 + ...
    y(2)^2)^(5/2)) - (3*mu_hat*y(2))/((mu + y(1))^2 + y(2)^2)^(5/2) + ...
    (15*mu_hat*y(2)*(2*mu + 2*y(1))^2)/(4*((mu + y(1))^2 + ...
    y(2)^2)^(7/2))) - u(3)*((3*mu_hat*(2*mu + 2*y(1)))/((mu + ...
    y(1))^2 + y(2)^2)^(5/2) - 1/((mu_hat - y(1))^2 + y(2)^2)^(3/2) + ...
    (3*mu_hat*(mu + y(1)))/((mu + y(1))^2 + y(2)^2)^(5/2) + ...
    (3*(2*mu_hat - 2*y(1))*(mu_hat - y(1)))/(2*((mu_hat - y(1))^2 + ...
    y(2)^2)^(5/2)) - (15*mu_hat*(2*mu + 2*y(1))^2*(mu + y(1)))/(4*((mu + ...
    y(1))^2 + y(2)^2)^(7/2))));

tmp(2,1) = k(1)*(u(4)*((3*y(2)*(2*mu + 2*y(1)))/(2*((mu + y(1))^2 + ...
    y(2)^2)^(5/2)) - (3*mu*y(2))/((mu_hat - y(1))^2 + y(2)^2)^(5/2) + ...
    (15*mu*y(2)*(2*mu_hat - 2*y(1))^2)/(4*((mu_hat - y(1))^2 + ...
    y(2)^2)^(7/2))) + u(3)*((3*mu*(mu_hat - y(1)))/((mu_hat - ...
    y(1))^2 + y(2)^2)^(5/2) - 1/((mu + y(1))^2 + y(2)^2)^(3/2) + ...
    (3*mu*(2*mu_hat - 2*y(1)))/((mu_hat - y(1))^2 + y(2)^2)^(5/2) + ...
    (3*(2*mu + 2*y(1))*(mu + y(1)))/(2*((mu + y(1))^2 + y(2)^2)^(5/2)) - ...
    (15*mu*(2*mu_hat - 2*y(1))^2*(mu_hat - y(1)))/(4*((mu_hat - ...
    y(1))^2 + y(2)^2)^(7/2)))) - k(2)*(u(4)*(1/((mu + y(1))^2 + ...
    y(2)^2)^(3/2) - (3*y(2)^2)/((mu + y(1))^2 + y(2)^2)^(5/2) - ...
    (3*mu*(2*mu_hat - 2*y(1)))/(2*((mu_hat - y(1))^2 + ...
    y(2)^2)^(5/2)) + (15*mu*y(2)^2*(2*mu_hat - 2*y(1)))/(2*((mu_hat - ...
    y(1))^2 + y(2)^2)^(7/2))) - u(3)*((3*y(2)*(mu + y(1)))/((mu + ...
    y(1))^2 + y(2)^2)^(5/2) - (3*mu*y(2))/((mu_hat - y(1))^2 + ...
    y(2)^2)^(5/2) + (15*mu*y(2)*(2*mu_hat - 2*y(1))*(mu_hat - ...
    y(1)))/(2*((mu_hat - y(1))^2 + y(2)^2)^(7/2))));

return;
