function [ funcs ] = make_group_lasso_funcs(params)
%MAKE_GLASSO_FUNCS Summary of this function goes here
%   Detailed explanation goes here
funcs.obj = @obj;
funcs.g = @g;
funcs.grad_g = @grad_g;
funcs.prox = @prox;
% Parameters for functions
Y = params.Y;
K = params.K;
intercept = get_opt(params,'intercept',false);
lambda1 = get_opt(params,'lambda1',0);
lambda2 = get_opt(params,'lambda2',0);
lambdae = get_opt(params,'lambdae',0);
ginds = get_opt(params,'ginds');
gsize = nan;
grep = [];
gmult = [];
if isempty(ginds)
    gsize = get_opt(params,'gsize',1);
else
    grep = zeros(ginds(end),1);
    grep(ginds) = 1;
    grep = cumsum(grep);
    grep(ginds) = grep(ginds-1);
    grep = grep+1;
    gmult = get_opt(params,'gmult',ones(ginds(end),1));
end
% Group lasso functions
function o = obj(x)
    o = g(x);
    if intercept
        x = x(1:end-1,:);
    end
    if ~isempty(x)
        if lambda1>0
            o = o + lambda1*sum(abs(x(:)));
        end
        if lambda2>0
            if isnan(gsize)
                r = cumsum(x(:).^2);
                o = o + lambda2.*gmult'*sqrt(r(ginds)-[0; r(ginds(1:end-1))]);
            else
                o = o + lambda2*sum(sqrt(sum(reshape(x,gsize,[]).^2,1)));
            end
        end
    end
end
function o = g(y)
    if intercept
        if size(y,1)>1
            o = bsxfun(@minus,Y-K*y(1:end-1,:),y(end,:));
        else
            o = bsxfun(@minus,Y,y(end,:));
        end
        y = y(1:end-1,:);
    else
        o = Y-K*y;
    end
    o = o(:)'*o(:)/2;
    if lambdae>0 && ~isempty(y)
        o = o + lambdae*(y(:)'*y(:))/2;
    end
end
function grad = grad_g(y)
    if intercept
        if size(y,1)>1
            resid = bsxfun(@plus,K*y(1:end-1,:),y(end,:))-Y;
            grad = K'*resid;
            grad = [grad; sum(resid)];
            grad(1:end-1,:) = grad(1:end-1,:)+lambdae*y(1:end-1,:);
        else
            resid = bsxfun(@plus,-Y,y(end,:));
            grad = sum(resid);
        end
    else
        resid = K*y-Y;
        grad = K'*resid;
        grad = grad+lambdae*y;
    end
end
function y_s = prox(y,t)
    if intercept
        y_s = y(1:end-1,:);
    else
        y_s = y;
    end
    if ~isempty(y_s)
        if lambda1>0
            rho = t*lambda1;
            y_s = sign(y_s).*max(0,abs(y_s)-rho);
        end
        if lambda2>0
            y_p = size(y_s,1);
            y_s = reshape(y_s,gsize,[]);
            if isnan(gsize)
                rho = t*lambda2.*gmult;
                r = cumsum(y_s(:).^2);
                r = sqrt(r(ginds)-[0; r(ginds(1:end-1))]);
                r = max(0,1-rho./r);
                scale = r(grep);
            else
                rho = t*lambda2;
                y_s = reshape(y_s,gsize,[]);
                scale = repmat(max(0,1-rho./sqrt(sum(y_s.^2,1))),gsize,1);
            end
            y_s = reshape(scale(:).*y_s(:),y_p,[]);
        end
    end
    if intercept
        if ~isempty(y_s)
            y_s(end+1,:) = y(end,:);
        else
            y_s = y;
        end
    end
end

end

