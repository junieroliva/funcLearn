function phix = evalbasis(x, inds)

[n,d] = size(x,1);
phix = ones(n,size(inds,1));
for i=1:d
    imax = max(inds(:,i))-1;
    phi_m = ( (0:imax)*pi );
    phi_c = [ 1, sqrt(2)*ones(imax,1) ];
    phi_d = bsxfun(@times,cos(bsxfun(@times,x(:,i),phi_m)),phi_c);
    phix = phix.*phi_d(:,inds(:,i));
end

end