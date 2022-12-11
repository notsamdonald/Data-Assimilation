function [X_ens_array_b] = donald_LETKF(m, X_ens_b,X_obvs, X_obvs_ens, X_ref, H, time_steps, n_states, n_ens, covar_inflation, L, alpha, plot)
% LETKF with option covariance inflation
% Able to generate RMSE/Rank histograms based on resultant analysis
% ensemble

% Used for plotting
title_info = "LETKF";
if covar_inflation
    title_info = title_info + " with Inflation";
end

subset_size = 10 % Number of states to group together

% Initalizing array of ensemble members over time
X_ens_array_b = zeros(n_states, n_ens, length(time_steps));
X_ens_array_a = zeros(n_states, n_ens, length(time_steps));

% Ensemble calculations
C_ens = 1/(sqrt(n_ens-1));

% Observation errors, [TODO - pass in this value]
full_R = 0.025.^2 * eye(20);

% Messy [TODO initalize this]
X_ens_a = X_ens_b;
X_ens_a = X_ens_a * 0;

for t_id = 1:(length(time_steps))
    
    % Calculating x and z metrics over full ensemble
    full_x_mean_b = mean(X_ens_b, 2);
    full_z_mean_b = mean(H*X_ens_b, 2);

    full_X_dot_b = C_ens*(X_ens_b - full_x_mean_b);
    full_Z_dot_b = C_ens*(H*X_ens_b - full_z_mean_b);  % (3.27)

    % Get observations for current timestep
    full_Y = X_obvs(:, t_id); % Rename X_obvs_ens -> Y_obvs_ens
    
    for subset_id = 1:(n_states/subset_size)

        % Calculating index for subsets
        subset_id_start = 1+(subset_id-1)*subset_size;
        subset_id_end = subset_id_start + subset_size - 1;

        z_subset_id_start  = 1+(subset_id-1)*(subset_size/2);
        z_subset_id_end  = z_subset_id_start + (subset_size/2) - 1;

        L = subset_id_start:subset_id_end;
        L_z = z_subset_id_start:z_subset_id_end;

        % Creating subsets of the ensemble
        x_mean_b = full_x_mean_b(L, :);
        z_mean_b = full_z_mean_b(L_z, :);
        X_dot_b = full_X_dot_b(L, :);
        Z_dot_b = full_Z_dot_b(L_z, :);
        Y = full_Y(L_z, :);
        R = full_R(L_z,L_z);

        % Calculate Transform matrix of subset
        [U, D] = eig(Z_dot_b.' * inv(R) * Z_dot_b);  %(3.61) - inv is frowned upon
        T = real(U) * (eye(n_ens) + real(D))^(-0.5) * real(U).';  % (3.62)
    
        % Caclualte Kalman gain of subset
        K = X_dot_b * inv(eye(n_ens) + Z_dot_b.' * inv(R) * Z_dot_b) * Z_dot_b.' * inv(R);  %(3.51a)  - inv is frowned upon
        K = X_dot_b * T * T.' * Z_dot_b.' * inv(R);  % <- check where this is from (3.51a) %Aparently I can use T to calculate this? More than current?
    
        %LETKF weight calcs
        w = Z_dot_b.' * inv(Z_dot_b *  Z_dot_b.' + R) * (Y - z_mean_b);

        % Update ensemble mean
        x_mean_a = x_mean_b + K * (Y - z_mean_b);  %(3.51a)
        
        % Covar inflation
        if covar_inflation
            X_dot_b = X_dot_b * alpha;
        end
    
        % Update the ensemble of scaled deviations
        X_dot_a = X_dot_b * (repmat(w, [1,n_ens]).'+ T);  % (3.62)
    
        % Recreate the ensemble
        % Update the subset of the full ensemble
        X_ens_a(L,:) = x_mean_a + C_ens.^-1 * X_dot_a;
    end

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