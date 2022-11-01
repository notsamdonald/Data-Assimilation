function rho = gaspari_cohn(z)
    % applying gaspari cohn to a singluar scalar value

    if (0 <= z) && (z <= 1)
        rho = -1/4*z^5 + 1/2*z^4 + 5/8*z^3 - 5/3*z^2 + 1;
    elseif (1 < z) && (z <= 2)
        rho = 1/12*z^5 - 1/2*z^4 + 5/8*z^3 + 5/3*z^2 - 5*z + 4 - 2/(3*z);
    else
        rho = 0;
    end

end