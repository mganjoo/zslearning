% Visualize t-SNE mapped word vectors.

function [] = visualize(mappedX, Y, mappedWordTable, label_names, images, images_to_show)

gscatter(mappedX(:,1), mappedX(:,2), label_names(Y), [], '+o*.xsd^v>', 7);
hold on;

% Add actual images
for j = 1:length(images_to_show)
    i = images_to_show(j);
    I = displayColorNetwork2(double(images(:, i)));
    I = imresize(I, 1.5);
    I = flipdim(I, 1);
    I = flipdim(I, 2);
    x = mappedX(i, 1);
    y = mappedX(i, 2);
    w = size(I, 1) / 2;
    I(I < 0) = 0;
    I(I > 1) = 1;
    image([x-w/2,x+w/x], [y-w/2,y+w/2], I);
end

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
