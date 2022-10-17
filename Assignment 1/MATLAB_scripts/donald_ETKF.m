function [X_ens_array_b] = donald_ETKF(m, X_ens_b, X_obvs, X_obvs_ens, X_ref, H, time_steps, n_states, n_ens, covar_localization, covar_inflation, plot)
% ETKF
% Able to generate RMSE/Rank histograms based on resultant analysis
% ensemble


% Used for plotting
title_info = "ETKF";
if covar_localization
    title_info = title_info + ", with Covar Localization ";
end
if covar_inflation
    title_info = title_info + "and Inflation";
end


% Initalizing array of ensemble members over time
X_ens_array_b = zeros(n_states, n_ens, length(time_steps));
X_ens_array_a = zeros(n_states, n_ens, length(time_steps));

% Ensemble calculations
C_ens = 1/(sqrt(n_ens-1));

% Observation errors, [TODO - pass in this value]
R = 0.025.^2 * eye(20); 

for t_id = 1:length(time_steps)

    x_mean_b = mean(X_ens_b, 2);
    z_mean_b = mean(H*X_ens_b, 2);

    X_dot_b = C_ens*(X_ens_b - x_mean_b);
    Z_dot_b = C_ens*(H*X_ens_b - z_mean_b);  % (3.27)

    % Get observations for current timestep
    Y = X_obvs(:, t_id); % Rename X_obvs_ens -> Y_obvs_ens
    
    % Calculate Transform matrix
    [U, D] = eig(Z_dot_b.' * inv(R) * Z_dot_b);  %(3.61)
    T = real(U) * (eye(n_ens) + real(D))^(-0.5) * real(U).';  % (3.62)

    % Caclualte Kalman gain
    K = X_dot_b * T * T.' * Z_dot_b.' * inv(R);

    % Update ensemble mean
    x_mean_a = x_mean_b + K * (Y - z_mean_b);  %(3.51a)

    % Update the ensemble of scaled deviations
    X_dot_a = X_dot_b * T;  % (3.62)

    % Recreate the ensemble
    X_ens_a = x_mean_a + C_ens.^-1 * X_dot_a;

    % Updating trackers
    X_ens_array_b(:,:,t_id) = X_ens_b;
    X_ens_array_a(:,:,t_id) = X_ens_a;

    % Move forward for all but final timestep
    if t_id < length(time_steps)
        for i = 1:n_ens
            [~, x] = ode45(m.RHS.F, [time_steps(t_id) time_steps(t_id+1)], X_ens_a(:, i));
            X_ens_b(:, i) = x(end, :).';  % New prior
        end
    end
end
disp('ETKF run complete')

% Generating plots
if plot
    clf
    figure;
    rank_histogram_plot(X_obvs_ens,X_ens_array_b, time_steps, title_info); 

    figure;
    rmse_plot(X_ref, X_ens_array_a, X_ens_array_b, time_steps, title_info);
end