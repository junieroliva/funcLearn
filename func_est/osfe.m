function [pcs, inds, PM, yhat] = osfe(x, y, varargin)
% TODO: write this.

if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

ensure_rep = get_opt(opts, 'ensure_rep', true);
concat = get_opt(opts, 'concat', true);

if ~iscell(x) 
    [N,n,p] = size(y);
    d = size(x,2);
    
    N_rot = get_opt(opts, 'N_rot', min(N,20));
    p_rot = get_opt(opts, 'p_rot', min(p,5));
    
    y = permute(y,[2 3 1]);
    inds = get_opt(opts, 'inds');
    if isempty(inds)
        inds_norm = get_opt(opts, 'inds_norm');
        if isempty(inds_norm)
            cv_norms = nan(N_rot, p_rot);
            irprm = randperm(N,N_rot);
            jrprm = randperm(p,p_rot);
            for ii=1:N_rot
                for jj=1:p_rot
                    [~,~,inds] = cv_os(x,squeeze(y(:,jrprm(jj),irprm(ii))),opts);
                    cv_norms(ii,jj) = max(sqrt(sum(inds.^2,2)));
                end
            end
            inds_norm = mean(cv_norms(:));
        end
          
        inds = outerprodinds(0:inds_norm,d,inds_norm);
        inds = sortrows( [ sum(inds.^2,2), inds] );
        inds = inds(:, 2:end);
        
        if ensure_rep && size(inds,1)<=4 
            % the functions are probably undersampled if we have so few
            % basis functions
            inds_norm = (n-1)^(1/d);
            inds = outerprodinds(0:inds_norm,d,inds_norm);
            inds = sortrows( [ sum(inds.^2,2), inds] );
            inds = inds(:, 2:end);
        end
    end
    phix = eval_basis(x,inds);
    PM = (phix'*phix) \ phix';

    if exist('mtimesx', 'file')
        pcs = mtimesx(PM,y);
        if nargout>=4
            yhat = mtimesx(phix,pcs);
            yhat = permute(yhat,[3 1 2]);
        end
    else % TODO: implement
        error('Need mtimesx package');
    end
    
    if concat
        pcs = reshape(pcs,[],N)';
    end
    
else
    [cN,p] = size(x);
    N_rot = get_opt(opts, 'N_rot', min(cN,20));
    p_rot = get_opt(opts, 'p_rot', min(p,5));
    
    inds = get_opt(opts, 'inds');
    if isempty(inds)
        inds_norm = get_opt(opts, 'inds_norm');
        if isempty(inds_norm) 
            % TODO: consider making an option for seperate indices based on
            %       each input function
            cv_norms = nan(N_rot, p_rot);
            irprm = randperm(cN,N_rot);
            jrprm = randperm(p,p_rot);
            for ii=1:N_rot
                for jj=1:p_rot
                    [~,inds] = osfe(x{irprm(ii),jrprm(jj)}, y{irprm(ii),jrprm(jj)}, opts);
                    cv_norms(ii,jj) = max(sqrt(sum(inds.^2,2)));
                end
            end
            opts.inds_norm = mean(cv_norms(:));
        end
    end
    
    pcs = cell(cN,p);
    inds = cell(cN,p);
    for i=1:cN
        for j=1:p
            [pcs{i,j}, inds{i,j}] = osfe(x{i,j}, y{i,j}, opts);
        end
    end
    if concat
        if cN>1
            % TODO: implement
        else
            pcs = cell2mat(pcs);
        end
    end
end



end
