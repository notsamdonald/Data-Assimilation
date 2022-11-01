function J = jacobian(~, y)

y1 = y(1);
y2 = y(2);

J = [ ...
    77.27 * (1 - y2 - 1.675e-5 * y1), 77.27 * (1 - y1), 0; ...
    -y2 / 77.27, -(1 + y1) / 77.27, 1 / 77.27; ...
    0.161, 0, -0.161];

end
