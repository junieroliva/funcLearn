function [x, objs] = fista_active( Y, K, funcs, opts )
p = size(K,2);

% set options
maxIter = opts.maxIter;
verbose = opts.verbose;
intercept = opts.intercept;
opts.verbose = false;
x_curr = get_opt(opts,'x_0',zeros(p,1));

% initialize variables
objs = nan(maxIter+1,1);
obj_curr = funcs.obj(Y,K,x_curr,opts);
objs(1) = obj_curr;
working_set = active_mask(x_curr);

if verbose
    fprintf('# Beginning Active Regression with %i co-variates \n', p);
end

% main loop
for k = 1:maxIter
    if verbose
        if verbose
            funcs.opt_msg(x_curr,opts);
        end
    end
    
    % optimize over the current working set
    if sum(working_set)>0
        opts.x_0 = x_curr([working_set; true]);
        x_working = fista(Y, K(:,working_set), funcs, opts);
        x_curr([working_set; true]) = x_working;
    end
    obj_curr = funcs.obj(Y,K,x_curr,opts);
    objs(k+1) = obj_curr;
    
    if verbose
        fprintf('\t{%d}\t[obj_curr: %f]\n', t_k, obj_curr);
    end
    
    % get valid zeros according to stationarity
    valid_zero = valid_zeros(x_curr);
    nworkgroups = sum(reshape(working_set,opts.g_size,[]),1)==0;
    invalid_zeros = nworkgroups & ~valid_zero; %in ~working set & shouldn't be zero
    any_invalid = any(invalid_zeros);
    
    % if stationarity holds, we are done
    if ~any_invalid
        break;
    end
    % else add violating groups to working set
    if verbose
        fprintf('**********************************************************\n');
        fprintf('************** Adding %i groups **************************\n', sum(invalid_zeros));
    end
    invalid_zeros = repmat(invalid_zeros,opts.g_size,1);
    invalid_zeros = invalid_zeros(:);

    working_set = working_set | invalid_zeros(:);
end
objs = objs(1:k+1);
x = x_curr;

% helper functions
    function vz = valid_zeros(x)
        vz = funcs.prox(funcs.grad_g(Y,K,x,opts),1,opts);
        if intercept
            vz = sqrt(sum(reshape(vz(1:end-1),opts.g_size,[]).^2,1))==0;
        else
            vz = sqrt(sum(reshape(vz,opts.g_size,[]).^2,1))==0;
        end
    end
    function nzg = nzero_groups(x)
        if intercept
            nzg = sum(reshape(x(1:end-1),opts.g_size,[]).^2,1)>0;
        else
            nzg = sum(reshape(x,opts.g_size,[]).^2,1)>0;
        end
    end
    function am = active_mask(x)
        am = nzero_groups(x);
        am = repmat(am,opts.g_size,1);
        am = am(:);
    end
end