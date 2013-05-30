function [ guessedCategories ] = softmaxPredict( mappedImages, thetaSoftmaxUnseen, trainParamsUnseen )

Wu = stack2param(thetaSoftmaxUnseen, trainParamsUnseen.decodeInfo);
pred = exp(Wu{1}*mappedImages); % k by n matrix with all calcs needed
pred = bsxfun(@rdivide,pred,sum(pred));
[~, gind] = max(pred);
guessedCategories = gind;

end
