function [ PC, B, inds ] = get_multi_pc_reg( Y, X, varargin )
%GET_MULTI_PC_REG Get the projection coefficients of multiple functions
%   Detailed explanation goes here
if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
inds = [];
[n, p, N] = size(Y);
d = size(X,2);

B = get_opt(opts,'B',[]);
if isempty(B)
    Phi = get_opt(opts,'Phi',[]);
    if isempty(Phi)
        % get indices for basis functions
        M = get_opt(opts,'M',floor(n^(1/d)));
        if d==1
            inds = (1:M)';
        elseif d==2
            [i1,i2] = sub2inds([M M], (1:M^2)');
            inds = [i1 i2];
            inds = inds(sum(inds.^2,2)<=M^2);
        else
            [i1,i2,i3] = sub2inds([M M M], (1:M^3)');
            inds = [i1 i2 i3];
            inds = inds(sum(inds.^2,2)<=M^2);
        end
        % get basis values at X
        basis = get_opt(opts,'basis','trig');
        Phi = eval_basis(X,inds,basis);
    end
    % cross-validate number of basis funcs on holdout set?
    do_cv = get_opt(opts,'do_cv',false);
    if ~do_cv
        B = (Phi'*Phi)\(Phi');
    else
        opts.Phi = Phi;
        opts.inds = inds;
        N_rot = get_opt(opts,'N_rot',5);
        irprm = randperm(N,min(N,N_rot));
        m_rots = nan(length(irprm),min(N_rot,p));
        for i=1:length(irprm)
            grprm = randperm(p,min(p,N_rot));
            for j=1:length(grprm)
                [~,B_cv] = get_pc_reg(Y(:,grprm(j),irprm(i)),X,opts);
                m_rots(i,j) = size(B_cv,1);
            end
        end
        m = floor(mean(m_rots(:)));
        Phi = Phi(:,1:m);
        B = (Phi'*Phi)\(Phi');
        if isempty(inds)
            inds = (1:m)';
        else
            inds = inds(1:m,:);
        end
    end
end

m = size(B,1);
PC = reshape(B*reshape(Y,n,p*N),m*p,N)';

end

