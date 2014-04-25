combinations = [ 2 7 ; 5 10 ; 7 10 ; 8 9 ; 8 10 ];
load image_data/images/cifar10/meta.mat;
basedir = 'gauss_cifar10_acl';

numObs = 80;
t = [ 1:(numObs/10):80 80];
colors = { 'r', 'g', 'b', 'k', 'm' };
markers = { '+', 'o', 'x', 'd', 's' };
figure;

set(gcf,'paperunits','centimeters')
set(gcf,'papersize',[24,20]) % Desired outer dimensionsof figure
set(gcf,'paperposition',[0,0,24,20]) % Place plot on figure

hold on;
labels = cell(1, 5);
for i = 1:size(combinations, 1)
    zeroList = label_names(combinations(i,:));
    zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];  
    labels{i} = sprintf('%s-%s', label_names{combinations(i,1)}, label_names{combinations(i,2)});
    load([ basedir '/out_' zeroStr '.mat'], 'unseenAccuracies');
    unseenAccuracies = fliplr(unseenAccuracies);
    plot(0:0.1:1, unseenAccuracies(t), 'Color', colors{i}, 'Marker', markers{i}, 'LineWidth', 2);
end

h_legend = legend(labels);

for i = 1:size(combinations, 1)
    zeroList = label_names(combinations(i,:));
    zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];    
    load([ basedir '/out_' zeroStr '.mat'], 'seenAccuracies');
    seenAccuracies = fliplr(seenAccuracies);
    plot(0:0.1:1, seenAccuracies(t), 'Color', colors{i}, 'Marker', markers{i}, 'LineWidth', 2);
end

h_xl = xlabel('Fraction of points classified as unseen');
h_yl = ylabel('Accuracy');

set(h_legend, 'FontSize', 16);
set(h_xl, 'FontSize', 24);
set(h_yl, 'FontSize', 24);

hold off;

% filename_base = 'figures/accuracy_5';
% print('-dpdf', sprintf('%s.pdf', filename_base));
% print('-deps', sprintf('%s.eps', filename_base));
