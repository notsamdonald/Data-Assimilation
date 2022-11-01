function rmse_plot(X_ref, X_ens_array_a,X_ens_array_b,X_particle, time_steps, title_info)
% Generating an RMSE plot for both background and analysis ensemble

% RMSE calculations
%rmse_a = rmse(X_ref, squeeze(mean(X_ens_array_a, 2)));
%rmse_b = rmse(X_ref, squeeze(mean(X_ens_array_b, 2)));

rmse_particle = rmse(X_ref, X_particle)
set(gca, 'YScale', 'log')

hold on
%semilogy(time_steps, rmse_a, 'LineWidth',1.5,'DisplayName','Analysis')
%semilogy(time_steps, rmse_b, 'LineWidth',1.5,'DisplayName','Background')
semilogy(time_steps, rmse_particle, 'LineWidth',1.5,'DisplayName','Background')

hold off

xlabel('time') 
ylabel('RMSE')
title("RMSE over time" + newline + title_info)
grid on
legend

end