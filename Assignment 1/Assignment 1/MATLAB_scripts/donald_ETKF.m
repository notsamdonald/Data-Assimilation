function [X_ens_array] = donald_ETKF(m, X_ens_a,X_obvs, X_obvs_ens, X_ref, H, time_steps, n_states, n_ens, covar_localization, covar_inflation, L, alpha, plot)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here


% Used for plotting
title_info = "ETKF";
if covar_localization
    title_info = title_info + ", with Covar Localization ";
end
if covar_inflation
    title_info = title_info + "and Inflation";
end


X_ens_array = zeros(n_states, n_ens, length(time_steps));
X_ens_array(:,:,1) = X_ens_a;

X_ens_b = X_ens_a;

C_ens = 1/(sqrt(n_ens-1));
R = 0.025.^2 * eye(20);% Define this above (observation errors)
E_i = mvnrnd(zeros(20 ,1).', eye(20)*0.025^2, n_ens).';

for t_id = 1:(length(time_steps)-1)
    
    x_mean_b = mean(X_ens_b, 2);
    z_mean_b = mean(H*X_ens_b, 2);

    X_dot_b = C_ens*(X_ens_b - x_mean_b);
    Z_dot_b = C_ens*(H*X_ens_b - z_mean_b);  % (3.27)

    % Get observations for current timestep
    Y = X_obvs(:, t_id); % Rename X_obvs_ens -> Y_obvs_ens
    
    % Calculate Transform matrix
    [U, D] = eig(Z_dot_b.' * inv(R) * Z_dot_b);  %(3.61) - inv is frowned upon
    T = real(U) * (eye(n_ens) + real(D))^(-0.5) * real(U).';  % (3.62)

    % Caclualte Kalman gain
    K = X_dot_b * inv(eye(n_ens) + Z_dot_b.' * inv(R) * Z_dot_b) * Z_dot_b.' * inv(R);  %(3.51a)  - inv is frowned upon
    K = X_dot_b * T * T.' * Z_dot_b.' * inv(R);  % <- check where this is from (3.51a) %Aparently I can use T to calculate this? More than current?

    % Update ensemble mean
    x_mean_a = x_mean_b + K * (Y - z_mean_b);  %(3.51a)

    if covar_localization
          
        % Covarance localization
        A = repmat(x_mean_a, [1,n_states]);
        distance = A - A.';
        L = 5;
        rho = arrayfun(@gaspari_cohn, distance/L);
        [u_r, d_r] = eig(rho);
        X_dot_b = rho * X_dot_b;
    end
    
    if covar_inflation
        X_dot_b = X_dot_b * alpha^2; % I think for ETKF it shouldnt be squared, since this is on X_dot and not the covar
    end

    % Update the ensemble of scaled deviations
    X_dot_a = X_dot_b * T;  % (3.62)

    % Recreate the ensemble
    X_ens_a = x_mean_a + C_ens.^-1 * X_dot_a;

    % Move forward in time via model
    for i = 1:n_ens
        [~, x] = ode45(m.RHS.F, [time_steps(t_id) time_steps(t_id+1)], X_ens_a(:, i));
        X_ens_b(:, i) = x(end, :).';  % New prior
        X_ens_array(:,i,t_id+1) = X_ens_b(:, i);  % Save enemble of priors
    end
end
disp('ETKF run complete')

if plot
    clf
    figure;
    rank_histogram_plot(X_obvs_ens,X_ens_array, time_steps, title_info); 

    figure;
    rmse_plot(X_ref, X_ens_array, time_steps, title_info);
end