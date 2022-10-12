function rmse_plot(X_ref, X_ens_array, time_steps, title_info)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

rmse_enkf = rmse(reshape(X_ref, [40, 1, 101]), X_ens_array); % TODO make this dynamic
rmse_avg_enkf = squeeze(mean(rmse_enkf, 2));

plot(time_steps, rmse_avg_enkf)
xlabel('time') 
ylabel(['$\overline {RMSE}$'],'interpreter','latex')
title("RMSE over time" + newline + title_info)
grid on

end