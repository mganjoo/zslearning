types = {'pdf', 'loop'};
titles = {'Using Gaussian PDF model', 'Using LoOP model'};
wordsets = { 'huang', 'turian.100', 'tasa' };

auc = cell(length(types), 1);
for tt = 1:length(types)
    type = char(types(tt));
    auc{tt} = zeros(1, length(wordsets));
    dataset = 'cifar10';
    zeroCategories = [ 4, 10 ];
    thresholds = 0:0.1:1;

    i = 1;
    accuracies = zeros(length(wordsets), length(thresholds));
    fprintf('Type: %s\n', type);
    for wordset = wordsets
        load(['word_data/' char(wordset) '/' dataset '/wordTable.mat']);
        zeroList = label_names(zeroCategories);
        zeroStr = [sprintf('%s_',zeroList{1:end-1}),zeroList{end}];    
        outputPath = sprintf('gauss_%s_%s_%s', dataset, char(wordset), zeroStr);
        load(sprintf('%s/out_%s.mat', outputPath, zeroStr));
        acctype = [ type 'Accuracies' ];
        accuracies(i, :) = eval(acctype);
        auc{tt}(i) = trapz(accuracies(i,:));
        i = i + 1;
    end
%     figure;
%     colors = hsv(12);
%     markers = { '+', 'o', 'x', 'd', 's' };
%     hold on;
%     for i = 1:size(accuracies, 1)
%         plot(thresholds, accuracies(i, :), 'Color', colors(i*2,:), 'Marker', markers{i});
%     end
%     h_legend = legend(wordsets);
%     h_xl = xlabel('Unseen probability threshold as percentile');
%     h_yl = ylabel('Unseen classification accuracy');
%     title(char(titles{tt}));
% %     for i = 1:size(seenAccuracies, 1)
% %         plot(thresholds, seenAccuracies(i, :), 'Color', colors(i,:));
% %     end
%     hold off
%     
%     set(h_legend, 'FontSize', 14);
%     set(h_xl, 'FontSize', 11);
%     set(h_yl, 'FontSize', 11);
% 
%     set(gcf,'paperunits','centimeters')
%     set(gcf,'papersize',[10,10]) % Desired outer dimensionsof figure
%     set(gcf,'paperposition',[0,0,10,10]) % Place plot on figure
% 
%     filename_base = sprintf('figures/wordset_%s', type);
%     print('-dpdf', sprintf('%s.pdf', filename_base));
%     print('-deps', sprintf('%s.eps', filename_base));
end

for i = 1:length(types)
    disp(types{i});
    disp(auc{i});
end