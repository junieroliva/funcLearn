function [ funcs ] = make_glasso_funcs()
%MAKE_GLASSO_FUNCS Summary of this function goes here
%   Detailed explanation goes here
funcs.obj = @obj;
funcs.g = @g;
funcs.grad_g = @grad_g;
funcs.prox = @prox;
funcs.opt_msg = @opt_msg;

% group lasso functions
    function opt_msg(iter, x_curr, opts)
        fprintf('__________________________________________________________\n');
        fprintf('Iteration: %i\n', iter);
        if opts.lambda1>0
            if opts.intercept
                fprintf('[non-zero coefficients: %i] ', sum(sum(x_curr(1:end-1,:)~=0)) );
            else
                fprintf('[non-zero coefficients: %i] ', sum(sum(x_curr~=0)) );
            end
        end
        if opts.lambda2>0
            if opts.intercept
                fprintf('[start non-zero groups: %i]', sum(sum(reshape(x_curr(1:end-1,:),opts.g_size,[]).^2,1)>0));
            else
                fprintf('[start non-zero groups: %i]', sum(sum(reshape(x_curr,opts.g_size,[]).^2,1)>0));
            end
        end
        fprintf('\n');
    end
    function o = obj(Y, K, x, opts)
        o = g(Y,K,x,opts);
        if opts.intercept
            x = x(1:end-1,:);
        end
        if opts.lambda1>0
            o = o + opts.lambda1*sum(abs(x(:)));
        end
        if opts.lambda2>0
            o = o + opts.lambda2*sum(sqrt(sum(reshape(x,opts.g_size,[]).^2,1)));
        end
    end
    function o = g(Y,K,y,opts)
        if opts.intercept
            o = Y-K*y(1:end-1)-y(end);
            y = y(1:end-1,:);
        else
            o = Y-K*y;
        end
        o = o(:)'*o(:)/2;
        if opts.lambdae>0 && ~isempty(y)
            o = o + opts.lambdae*(y(:)'*y(:))/2;
        end
    end
    function grad = grad_g(Y,K,y,opts)
        if opts.intercept
            resid = K*y(1:end-1)+y(end)-Y;
            grad = K'*resid;
            grad = [grad; sum(resid)];
            grad(1:end-1,:) = grad(1:end-1,:)+opts.lambdae*y(1:end-1,:);
        else
            resid = K*y-Y;
            grad = K'*resid;
            grad = grad+opts.lambdae*y;
        end
    end
    function y_s = prox(y,t,opts)
        if opts.intercept
            y_s = y(1:end-1,:);
        else
            y_s = y;
        end
        if opts.lambda1>0
            rho = t*opts.lambda1;
            y_s = sign(y_s).*max(0,abs(y_s)-rho);
        end
        if opts.lambda2>0
            y_p = size(y_s,1);
            y_s = reshape(y_s,opts.g_size,[]);
            rho = t*opts.lambda2;
            scale = repmat(max(0,1-rho./sqrt(sum(y_s.^2,1))),opts.g_size,1);
            y_s = reshape(scale(:).*y_s(:),y_p,[]);
        end
        if opts.intercept
            if ~isempty(y_s)
                y_s(end+1,:) = y(end,:);
            else
                y_s = y;
            end
        end
    end

end

