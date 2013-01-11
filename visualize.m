function [] = visualize(mappedX, Y, mappedWordTable, label_names)

gscatter(mappedX(:,1), mappedX(:,2), label_names(Y), [], [], 7);
hold on;
scatter(mappedWordTable(:,1), mappedWordTable(:,2), 100, 'o', 'filled');
for i = 1:length(label_names)
    text(mappedWordTable(i,1),mappedWordTable(i,2),label_names{i},'BackgroundColor',[.7 .9 .7]);        
end
hold off;

end