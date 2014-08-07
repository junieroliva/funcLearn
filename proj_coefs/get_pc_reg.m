function [ beta, B, inds ] = get_pc_reg( Y, X, varargin )
%GET_PC_REG Get projection coefficients for regressing a function
%   Detailed explanation goes here
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
[n, d] = size(X);
B = get_opt(opts,'B',[]);
if isempty(B)
    Phi = get_opt(opts,'Phi',[]);
    if isempty(Phi)
        % get indices for basis functions
        M = get_opt(opts,'M',floor(n^(1/d)));
        if d==1
            inds = (1:M)';
        elseif d==2
            [i1,i2] = ind2sub([M M], (1:M^2)');
            inds = [i1 i2];
            inds = inds(sum(inds.^2,2)<=M^2);
        else
            [i1,i2,i3] = ind2sub([M M M], (1:M^3)');
            inds = [i1 i2 i3];
            inds = inds(sum(inds.^2,2)<=M^2);
        end
        % get basis values at X
        basis = get_opt(opts,'basis','trig');
        Phi = eval_basis(X,inds,basis);
    else
        inds = get_opt(opts,'inds',[]);
    end
    % cross-validate number of basis funcs on holdout set?
    do_cv = get_opt(opts,'do_cv',false);
    if ~do_cv
        B = (Phi'*Phi)\(Phi');
    else
        if isempty(inds)
            inds = (1:size(Phi,2))';
        end
        trn_perc = get_opt(opts,'trn_perc',.9);
        trn_set = false(n,1);
        trn_set(randperm(n,ceil(n*trn_perc))) = true;
        norm2s = sum(inds.^2,2);
        lnorm2s = find([norm2s(1:end-1)~=norm2s(2:end); true]);
        hol_MSE = nan(length(lnorm2s),1);
        for lni=1:length(lnorm2s)
            ii = lnorm2s(lni);
            beta = (Phi(trn_set,1:ii)'*Phi(trn_set,1:ii))\(Phi(trn_set,1:ii)'*Y(trn_set));
            hol_MSE(lni) = sum((Y(~trn_set)-Phi(~trn_set,1:ii)*beta).^2);
        end
        [~,lni] = min(hol_MSE);
        ii = lnorm2s(lni);
        B = (Phi(:,1:ii)'*Phi(:,1:ii))\(Phi(:,1:ii)');
        inds = inds(1:ii,:);
    end
end
beta = B*Y;
end

