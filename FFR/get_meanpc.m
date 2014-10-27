function [pc, phix] = get_meanpc(X, T)
d = size(X,2);
rfreqs = zeros(d,T,d);
for i=1:d
    rfreqs(i,:,i) = (0:T-1)*pi;
end

ni = size(X,1);
pcd = cos(mtimesx(X,rfreqs));
pcd(:,2:end,:) = sqrt(2)*pcd(:,2:end,:);    
phix = nan(ni,T^d);
for m=1:ni
    oprod = squeeze(pcd(m,:,1));
    for k=2:d
        oprod = oprod(:)*squeeze(pcd(m,:,k));
    end
    phix(m,:) = oprod(:);
end

pc = mean(phix);
end