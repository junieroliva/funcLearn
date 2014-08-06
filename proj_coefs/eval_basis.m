function [ Phi ] = eval_basis( X, inds, basis )
%EVAL_BASIS Summary of this function goes here
%   Detailed explanation goes here
[n, d] = size(X);
if strcmp(basis,'trig')
    phi_k = @(x,k) (k==1)+(k>1)*(mod(k,2)==0)*sqrt(2)*cos(2*pi*(k/2)*x)+(k>1)*(mod(k,2)~=0)*sqrt(2)*sin(2*pi*((k-1)/2)*x);
else
    phi_k = @(x,k) (k==1)+(k>1)*sqrt(2)*cos((k-1)*pi*x);
end
Phi = ones(n,size(inds,1));
for di=1:d
    m = max(inds(:,di));
    Phid = ones(n,m);
    for k=1:m
        Phid(:,k) = phi_k(X(:,di),k);
    end
    Phi = Phi.*Phid(:,inds(:,di));
end

end

