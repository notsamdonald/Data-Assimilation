function mean_error_plot(X_ref, prediction, time_steps, title_info)
% Generating an mean error plot for of analysis ensemble and reference

spinup=1;

mean_error = mean(abs(prediction(:, spinup:end)-X_ref(:, spinup:end)),1);
set(gca, 'YScale', 'log')

hold on
semilogy(time_steps(spinup:end), mean_error, 'LineWidth',1.5)

hold off

xlabel('time') 
ylabel('Mean Error')
title("Mean Error over time" + newline + title_info)
grid on

end