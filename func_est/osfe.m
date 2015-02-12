function [pcs, inds, PM] = osfe(x, y, varargin)
[N,~,p] = size(y);
y = permute(y,[2 3 1]);
d = size(x,2);

if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

N_rot = get_opt(opts, 'N_rot', min(N,20));
p_rot = get_opt(opts, 'p_rot', min(p,5));

if ~iscell(x) 
    inds = get_opt(opts, 'inds');
    if isempty(inds)
        cv_norms = nan(N_rot, p_rot);
        irprm = randperm(N,N_rot);
        jrprm = randperm(p,p_rot);
        for ii=1:N_rot
            for jj=1:p_rot
                [~,~,inds] = cv_os(x,squeeze(y(:,jrprm(jj),irprm(ii))),opts);
                cv_norms(ii,jj) = max(sqrt(sum(inds.^2,2)));
            end
        end

        max_norm = mean(cv_norms(:));
        inds = outerprodinds(0:max_norm,d,max_norm);
    end
    phix = eval_basis(x,inds);
    PM = (phix'*phix) \ phix';

    if exist('mtimesx', 'file')
        pcs = mtimesx(PM,y);
    else % TODO: implement

    end
    pcs = reshape(pcs,[],N)';
else
    [cN,p] = size(x);
    pcs = cell(cN,p);
    for i=1:cN
        for j=1:p
            pcs{i,j} = osfe(x{i,j}, y{i,j}, opts);
        end
    end
    pcs = cell2mat(pcs);
end



end
