function [X_prediction] = donald_nudge_63(m, X_ens_b, X_obvs_ens, X_ref, H, time_steps, n_states, n_ens, plot)
% EnKF with option covariance localization/inflation
% Able to generate RMSE/Rank histograms based on resultant analysis
% ensemble

% Redo using the method of weight updates in the ntoes

% Used for plotting
title_info = "Nudge";

obvs_sigma = 0.025;
Q = obvs_sigma.^2 * eye(3); % Model error (placeholder)

% Particle weights (pass threse through instead)
w = (ones(1,n_ens)/n_ens)

% Initalizing array of ensemble members over time
X_ens_array_b = zeros(n_states, n_ens, length(time_steps));
X_ens_array_a = zeros(n_states, n_ens, length(time_steps));
X_prediction = zeros(n_states, length(time_steps))


% Compute estimated state
xhk = zeros(n_states,1);
for i = 1:n_ens;
   xhk = xhk + w(i)*X_ens_b(:,i);
end
X_prediction(:,1) = xhk;


for t_id = 1:(length(time_steps)-1)
 
    time_steps(t_id)

    % Get observations from prior trajectory with observation errors added
    Y_t = X_obvs_ens(:,t_id,:); 
    obvs_size = size(Y_t);
    Y_t = reshape(Y_t, [obvs_size(1), obvs_size(3)]);
    R = obvs_sigma.^2 * eye(obvs_size(1));  % TODO move this out

    % 1. Forcast
    
    % Move forward for all but final timestep
    X_ens_b_previous = X_ens_b;
    if t_id < length(time_steps)

        % Move forward in time via model to get new prior from posterior
        for i = 1:n_ens
            [~, x] = ode45(m.RHS.F, [time_steps(t_id) time_steps(t_id+1)], X_ens_b(:, i));

            % Without model error for now (Just to get this working)
            X_ens_b(:, i) = x(end, :).';% + innovation(:,i);

        end
        % Adding noise such that it is stochastic
        X_ens_b = X_ens_b + normrnd(0,0.0001,size(X_ens_b));
    end

    % 1.1 Nudge updates
    L =  0.0001 * H.';
    %Y_future = squeeze(X_obvs_ens(:,t_id+1,:)); 

    Y_t = X_obvs_ens(:,t_id+1,:); 
    obvs_size = size(Y_t);
    Y_future = reshape(Y_t, [obvs_size(1), obvs_size(3)]);


    innovation = L * (Y_future - H * X_ens_b_previous);
    model_noise = normrnd(0,obvs_sigma, size(X_ens_b));  % (note normrnd(mu,sigma))
    X_hat = X_ens_b + innovation + model_noise;
    

    % 1.2 Update weights of particles based on nudged forcast
    Qinv = inv(Q);
    for k = 1:n_ens
        % Caclulate difference between observation and particles
        A = X_hat(:,k) - X_ens_b(:,k);
        alpha = exp(-0.5*(A.'*Qinv*A) + 0.5*model_noise(:,k).'*Qinv*model_noise(:,k));
        w(k) = w(k) * alpha;
    end
    % 1.2a Rescale particles
    w = w./sum(w);

    % 1.3 Update position of particles based on nudged forcast
    X_ens_b = X_hat;


    % 2. Analysis
    % 2.1 Particle weight update

    % Get "Current" observation
    % Get observations from prior trajectory with observation errors added
    Y_t = X_obvs_ens(:,t_id+1,:); 
    obvs_size = size(Y_t);
    Y_t = reshape(Y_t, [obvs_size(1), obvs_size(3)]);


    for k = 1:n_ens
        % Caclulate difference between observation and particles
        d =  norm((Y_t(:,k) - H * X_ens_b(:,k)),2);    
        alpha = exp(-d.^2/(2*obvs_sigma.^2));
        w(k) = w(k) * alpha;
    end

    % 2.2 Rescale particles
    w = w./sum(w);

    % Calculate number of effective particles
    n_eff = 1/sum(w.^2);

    % Apply threshold value
    w = max(1e-40, w);
    w = w./sum(w);

    % Resample if effective particles is below threshold
    % TODO (tune this value)
    resample_threshold = 250;
    if n_eff < resample_threshold

        disp("Resampling!")

        X_ens_old = X_ens_b;
        edges = min([0 cumsum(w)],1); % protect against accumulated round-off
        edges(end) = 1;                 % get the upper edge exact
        for i = 1:n_ens
            [~, id] = histc(rand,edges);
            % Update new ensemble based on sample from old
            X_ens_b(:,i) = X_ens_old(:,id);
        end

        % Reset weights to equal
        w = (ones(1,n_ens)/n_ens);

    end

    if t_id < length(time_steps)
        % Compute estimated state
        xhk = zeros(n_states,1);
        for i = 1:n_ens;
           xhk = xhk + w(i)*X_ens_b(:,i);
        end
        X_prediction(:,t_id+1) = xhk;
    end

    mean(abs(X_prediction(:, t_id)-X_ref(:, t_id)),1)
end

disp('EnKF run complete')

% Generating plots
if plot
    clf
    %figure;
    %rank_histogram_plot(X_obvs_ens,X_ens_array_b, time_steps, title_info);
   
    figure;
    rmse_plot(X_ref, X_ens_array_a, X_ens_array_b,X_prediction, time_steps, title_info);
end