combinations = [ 4 6 ; 1 2 ; 2 5 ; 5 9 ; 4 10 ];
load image_data/images/cifar10/meta.mat;
basedir = 'gauss_cifar10_acl';

set(gcf,'paperunits','centimeters')
set(gcf,'papersize',[30,22]) % Desired outer dimensionsof figure
set(gcf,'paperposition',[0,0,30,22]) % Place plot on figure

numComparisons = size(combinations, 1);
labels = cell(1, numComparisons);
accuracies = zeros(1, numComparisons);
for i = 1:size(combinations, 1)
    zeroList = label_names(combinations(i,:));
    zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];  
    labels{i} = sprintf('%s-%s', label_names{combinations(i,1)}, label_names{combinations(i,2)});
    load([ basedir '/out_' zeroStr '.mat'], 'unseenAccuracies');
    accuracies(i) = unseenAccuracies(end);
end

bar(1:numComparisons, accuracies, 0.4);
set(gca,'XTick', 1:numComparisons, 'XTickLabel', labels);