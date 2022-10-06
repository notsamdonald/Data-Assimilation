function [X_ens_array] = donald_EnKF(m, X_ens_a, X_obvs_ens,X_ref, H, time_steps, R, n_states, n_ens, covar_localization, covar_inflation, L, plot)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Initalizing array of ensemble members over time
X_ens_array = zeros(n_states, n_ens, length(time_steps));
X_ens_array(:,:,1) = X_ens_a;


for t_id = 1:(length(time_steps)-1)
    
    % Emperical ensemble covariance
    B = cov(X_ens_a.');

    if covar_localization
        % Covarance localization
        X_ens_a_avg = mean(X_ens_a, 2); % Unclear if this is correct
        A = repmat(X_ens_a_avg, [1,n_states]);
        distance = A - A.';
        %L = 5;
        rho = arrayfun(@gaspari_cohn, distance/L);
        B = rho.*B;  %Think this is broken? As currently the RMSE is worse once the covariance localization is applied
    end

    if covar_inflation
    % To add covar inflation
    end

    % Kalman gain
    K_t = B * H.' * inv(H * B * H.' + R);  
    
    % Get observations from prior trajectory with observation errors added
    Y_t = squeeze(X_obvs_ens(:,t_id,:)); 

    % Caclulate innovation factor and update
    X_ens_a = X_ens_a + K_t * (Y_t - H * X_ens_a);    
    
    % Move forward in time via model
    for i = 1:n_ens
        [~, x] = ode45(m.RHS.F, [time_steps(t_id) time_steps(t_id+1)], X_ens_a(:, i));
        X_ens_a(:, i) = x(end, :).';
        X_ens_array(:,i,t_id+1) = X_ens_a(:, i);
    end
end
disp('EnKF run complete')

if plot
    clf
    figure;
    rank_histogram_plot(X_obvs_ens,X_ens_array, time_steps);
    
    figure;
    rmse_plot(X_ref, X_ens_array, time_steps);


end