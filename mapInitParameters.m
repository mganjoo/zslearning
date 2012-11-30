function theta = mapInitParameters(trainParams)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(trainParams.outputSize+trainParams.inputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
W = rand(trainParams.outputSize, trainParams.inputSize) * 2 * r - r;
b = zeros(trainParams.outputSize, 1);

% Flatten
theta = [W(:) ; b(:)];

end

