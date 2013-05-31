function [ output_args ] = evaluateAttributes( X, Y, thetas, trainParams )
    allPred = ones(size(Y));
    for i = 1:length(thetas)
        W = stack2param(thetas, trainParams.decodeInfo);
        pred = exp(W{1}*X); % k by n matrix with all calcs needed
        allPred = pred .* bsxfun(@rdivide,pred,sum(pred));
    end

end

