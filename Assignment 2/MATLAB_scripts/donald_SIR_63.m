function [X_ens_array_b] = donald_SIR_63(m, X_ens_b, X_obvs_ens, X_ref, H, time_steps, n_states, n_ens, plot)
% EnKF with option covariance localization/inflation
% Able to generate RMSE/Rank histograms based on resultant analysis
% ensemble

% Used for plotting
title_info = "SIR";

obvs_sigma = 0.025;

% Particle weights (pass threse through instead)
w = (ones(1,n_ens)/n_ens)

% Initalizing array of ensemble members over time
X_ens_array_b = zeros(n_states, n_ens, length(time_steps));
X_ens_array_a = zeros(n_states, n_ens, length(time_steps));
X_prediction = zeros(n_states, length(time_steps))

for t_id = 1:(length(time_steps))
    
    %% Compute estimated state
    xhk = zeros(n_states,1);
    for i = 1:n_ens;
       xhk = xhk + w(i)*X_ens_b(:,i);
    end
    X_prediction(:,t_id) = xhk;


    % Get observations from prior trajectory with observation errors added
    Y_t = squeeze(X_obvs_ens(:,t_id,:)); 

    % Caclulate difference between observation and particles
    d =  (Y_t - H * X_ens_b);    

    % Calculate P(yk|xk)
    prob_placeholder = mvnpdf(d.',0, obvs_sigma*eye(2)).';
    
    % Rescale weights
    w = prob_placeholder .* w;
    w = w./sum(w);
    
    % Calculate number of effective particles
    n_eff = 1/sum(w.^2)

    % Resample if effective particles is below threshold
    % TODO (tune this value)
    resample_threshold = 250;
    if n_eff < resample_threshold

        disp("Reampling!")

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


    % Updating trackers
    %X_ens_array_b(:,:,t_id) = X_ens_b;
    %X_ens_array_a(:,:,t_id) = X_ens_a;

    % Move forward for all but final timestep
    if t_id < length(time_steps)
        % Move forward in time via model to get new prior from posterior
        for i = 1:n_ens
            [~, x] = ode45(m.RHS.F, [time_steps(t_id) time_steps(t_id+1)], X_ens_b(:, i));
            X_ens_b(:, i) = x(end, :).';

        end
        % Adding noise such that it is stochastic
        X_ens_b = X_ens_b + normrnd(0,0.0025,size(X_ens_b));
    end

    


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