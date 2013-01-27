function [] = visualize(mappedX, Y, mappedWordTable, label_names)

gscatter(mappedX(:,1), mappedX(:,2), label_names(Y), [], '+o*.xsd^v>', 7);
hold on;
scatter(mappedWordTable(:,1), mappedWordTable(:,2), 200, 'd', 'k', 'filled');
for i = 1:length(label_names)
    text(mappedWordTable(i,1),mappedWordTable(i,2),label_names{i},'BackgroundColor',[.7 .9 .7]);        
end
axis off;
hold off;

set(gcf,'paperunits','centimeters')
set(gcf,'papersize',[30,25]) % Desired outer dimensionsof figure
set(gcf,'paperposition',[0,0,30,25]) % Place plot on figure

print -dpdf vis.pdf;
print -deps vis.eps;
print -dpng vis.png;

end