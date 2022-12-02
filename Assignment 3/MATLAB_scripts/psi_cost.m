function [cost] = psi_cost(X, X_obvs, xb, B0, R, H)
%PSI Summary of this function goes here
%   Detailed explanation goes here

t2 = zeros(1);
for i = 1:1:size(X,2)
    t2 = t2 + (H*X(:,i)-X_obvs(:,i))' * inv(R) * (H*X(:,i)-X_obvs(:,i));
end
cost = 0.5 * (X(:,1)-xb)' * inv(B0) * (X(:,1)-xb) + 0.5 * t2;
%cost = t2;
end

