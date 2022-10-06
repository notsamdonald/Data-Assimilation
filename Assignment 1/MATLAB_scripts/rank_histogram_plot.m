function rank_histogram_plot(X_obvs_ens,X_ens_array, time_steps)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here


% Rank histogram:

state_id = 1;
bins = zeros(1, 99); % Not dynamic, related to the downsampling below by 2x (200 -> 100)

for i = 1:(length(time_steps))
    state_obs = sort(squeeze(X_obvs_ens(state_id,i,:))); % TODO fix the time dimension (should be final)
    state_forcast = sort(squeeze(X_ens_array(state_id,:,i)));
    [N, ~] = histcounts(state_obs, state_forcast(1:2:end));
    bins = bins + N;


end

bar(bins/sum(bins))
xlabel('Bin number') 
ylabel('Relative frequency')
title("EnKF Rank Histogram (State 1)")
grid on


end