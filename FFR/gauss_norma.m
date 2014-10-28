function norma = gauss_norma(x, sigma2, trunc)
    [n,d] = size(x);
    if trunc
        norma = 1./((2*pi*sigma2)^(d/2)*prod(normcdf((1-x)/sqrt(sigma2))-normcdf(-x/sqrt(sigma2)),2)*n);
    else
        norma = 1./((2*pi*sigma2)^(d/2)*ones(n,1)*n);
    end
end