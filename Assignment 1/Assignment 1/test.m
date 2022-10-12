clear;
close all;

% Reference - README. 

m = otp.lorenz63.presets.Canonical;

% Not on solution manifold
[~, y] = ode45(m.RHS.F, [0 10], m.Y0);
y0 = y(end, :).';

disp(m);

disp(m.Y0);

disp(m.Parameters);

disp(m.RHS);

[~, y] = ode45(m.RHS.F, [0 40], y0);
plot3(y(:, 1), y(:, 2), y(:, 3));
xlabel('x');
ylabel('y');
zlabel('z');