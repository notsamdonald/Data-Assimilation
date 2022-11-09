function [X_prediction] = donald_ENKF_63(m, X_ens_b, X_obvs_ens, X_ref, H, time_steps, n_states, n_ens, plot)
% EnKF particle with model nudging
% Able to generate Mean error + Rank histograms based on resultant analysis

% Redo using the method of weight updates in the ntoes

% Used for plotting
title_info = "EnKF";

% Model constants (TODO - pass these in via function)
obvs_sigma = 0.025;

% Particle weights
w = (ones(1,n_ens)/n_ens);

% Initalizing array of ensemble members over time
X_ens_array_a = zeros(n_states, n_ens, length(time_steps));
X_prediction = zeros(n_states, length(time_steps));

for t_id = 1:(length(time_steps))

    % Get observations from prior trajectory with observation errors added
    Y_t = X_obvs_ens(:,t_id,:); 
    obvs_size = size(Y_t);
    Y_t = reshape(Y_t, [obvs_size(1), obvs_size(3)]);    

    % Kalman section
    % Emperical ensemble covariance
    B = cov(X_ens_b.');
    R = obvs_sigma.^2 * eye(obvs_size(1));  % TODO move this out
    Q = obvs_sigma.^2 * eye(3)./10; % Model error

    % Kalman gain
    K_t = B * H.' * inv(H * B * H.' + R);  

     % Caclulate innovation factor and update
    A = (eye(3) - (K_t*H));
    Q_enkf = A * Q * A.' + K_t * R * K_t.';
    x_q_enkf = X_ens_b + K_t * (Y_t - H * X_ens_b);

    % Analysis update
    for k = 1:n_ens

        x_mean_enkf = A * X_ens_b(:,k) + K_t * Y_t(:,k);

        alpha = exp(-0.5 * (mean(Y_t - H * x_q_enkf(:,k),2)).' * inv(R) * (mean(Y_t - H * x_q_enkf(:,k),2)) ...
        - 0.5 * (x_q_enkf(:,k) - X_ens_b(:,k)).' * inv(Q) * (x_q_enkf(:,k) - X_ens_b(:,k)) ...
        + 0.5 * (x_q_enkf(:,k) - x_mean_enkf).' * inv(Q_enkf) * (x_q_enkf(:,k) - x_mean_enkf));

        w(k) = w(k) * alpha;
        alpha;
    end

   
    % Rescale weights
    w = w./sum(w);

    % Calculate number of effective particles
    n_eff = 1/sum(w.^2);

    % Apply threshold value
    w = max(1e-16, w);
    w = w./sum(w);

    % Resample if effective particles is below threshold
    resample_threshold = 250;
    if n_eff < resample_threshold
        %disp("Reampling!")
        X_ens_old = X_ens_b;
        edges = min([0 cumsum(w)],1); 
        edges(end) = 1;
        for i = 1:n_ens
            [~, id] = histc(rand,edges);
            % Update new ensemble based on sample from old
            X_ens_b(:,i) = X_ens_old(:,id);
        end

        % Reset weights to equal
        w = (ones(1,n_ens)/n_ens);
        
    end

    % Compute estimated state
    xhk = zeros(n_states,1);
    for i = 1:n_ens;
       xhk = xhk + w(i)*X_ens_b(:,i);
    end
    X_prediction(:,t_id) = xhk;

    % Update tracker
    X_ens_array_a(:,:,t_id) = X_ens_b;


    % Move forward for all but final timestep
    if t_id < length(time_steps)
        % Move forward in time via model to get new prior from posterior
        for i = 1:n_ens
            [~, x] = ode45(m.RHS.F, [time_steps(t_id) time_steps(t_id+1)], X_ens_b(:, i));
            X_ens_b(:, i) = x(end, :).';
        end
        % Adding noise such that it is stochastic
        X_ens_b = X_ens_b + normrnd(0,0.0005,size(X_ens_b));
    end
end

disp('EnKF run complete')

% Generating plots
if plot
    clf
    %figure;
    rank_histogram_plot(X_obvs_ens,X_ens_array_a, time_steps, title_info);
   
    figure;
    mean_error_plot(X_ref, X_prediction, time_steps, title_info);
end