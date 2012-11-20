% Prepare the training data

X = load('image_data/trainX.mat');
Y = load('image_data/trainY.mat');

disp('Creating training set');
makeFeatureBatches(X.trainXCs, Y.trainY, 5, { 'cat', 'truck' });

disp('Creating mini-training set');
makeFeatureBatches(X.trainXCs, Y.trainY, 5, { 'cat', 'truck' }, 'mini_batch', 4);

clear X Y;