function rank_histogram_plot(X_obvs_ens,X_ens_array, time_steps, title_info)
% Generating a Rank Histogram for the analysis ensemble

state_id = 1;  % State to plot
n_ens = size(X_obvs_ens, 3);
downsample = 10;  % Used to create fewer bins than number of particles

% Bin initalizations
bin_size = sort(squeeze(X_ens_array(state_id,:,1)));
bin_size = size(bin_size(downsample:downsample:end));
bins = zeros(1, bin_size(2)-1);

for i = 1:(length(time_steps))
    state_obs = sort(squeeze(X_obvs_ens(state_id,i,:)));
    state_forcast = sort(squeeze(X_ens_array(state_id,:,i)));
    [N, ~] = histcounts(state_obs, state_forcast(downsample:downsample:end));
    bins = bins + N;

end

bar(bins/sum(bins))
yline(downsample/n_ens,'Color','red','LineWidth',3)
xlabel('Bin number') 
ylabel('Relative frequency')
title("Rank Histogram (State 1)" + newline + title_info)
grid on


end