function [X_ens_array_b] = donald_EnKF(m, X_ens_b, X_obvs_ens,X_ref, H, time_steps, R, n_states, n_ens, covar_localization, covar_inflation, L, alpha, plot)
% EnKF with option covariance localization/inflation
% Able to generate RMSE/Rank histograms based on resultant analysis
% ensemble

% Used for plotting
title_info = "EnKF";
if covar_localization
    title_info = title_info + ", with Covar Localization ";
end
if covar_inflation
    title_info = title_info + "and Inflation";
end

% Initalizing array of ensemble members over time
X_ens_array_b = zeros(n_states, n_ens, length(time_steps));
X_ens_array_a = zeros(n_states, n_ens, length(time_steps));

for t_id = 1:(length(time_steps))
    
    % Emperical ensemble covariance
    B = cov(X_ens_b.');

    if covar_localization
        
        % Taking the mean of the ensemble
        X_ens_b_avg = mean(X_ens_b, 2);
        A = repmat(X_ens_b_avg, [1,n_states]);

        % Absolute difference of the mean of each state
        distance = abs(A - A.');

        % Scaling by L, and applying gaspari_cohn
        rho = arrayfun(@gaspari_cohn, distance/L);

        % Localizing ensemble covariance
        B = rho.*B;
    end

    if covar_inflation
        B = B * alpha^2;
    end

    % Kalman gain
    K_t = B * H.' * inv(H * B * H.' + R);  
    
    % Get observations from prior trajectory with observation errors added
    Y_t = squeeze(X_obvs_ens(:,t_id,:)); 

    % Caclulate innovation factor and update
    X_ens_a = X_ens_b + K_t * (Y_t - H * X_ens_b);    
    
    % Updating trackers
    X_ens_array_b(:,:,t_id) = X_ens_b;
    X_ens_array_a(:,:,t_id) = X_ens_a;

    % Move forward for all but final timestep
    if t_id < length(time_steps)
        % Move forward in time via model to get new prior from posterior
        for i = 1:n_ens
            [~, x] = ode45(m.RHS.F, [time_steps(t_id) time_steps(t_id+1)], X_ens_a(:, i));
            X_ens_b(:, i) = x(end, :).'; % Creating new prior
        end
    end

end

disp('EnKF run complete')

% Generating plots
if plot
    clf
    figure;
    rank_histogram_plot(X_obvs_ens,X_ens_array_b, time_steps, title_info);
   
    figure;
    rmse_plot(X_ref, X_ens_array_a, X_ens_array_b, time_steps, title_info);
end