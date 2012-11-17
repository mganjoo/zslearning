function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta.

EPSILON = 1e-4;
basisMat = EPSILON * eye(size(theta, 1));
thetaPlus = bsxfun(@plus, theta, basisMat);
thetaMinus = bsxfun(@minus, theta, basisMat);
numgrad = (0.5 / EPSILON * (cellfun(J, num2cell(thetaPlus, 1)) - cellfun(J, num2cell(thetaMinus, 1))))';

end
