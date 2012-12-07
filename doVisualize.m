numImages = size(X, 2);
gscatter(t(1:numImages,1), t(1:numImages,2), label_names(Y), [], [], 7);
hold on;
scatter(t(numImages+(1:10),1), t(numImages+(1:10),2), 100, 'o', 'filled');
for i = 1:10
    text(t(numImages+i,1),t(numImages+i,2),label_names{i});        
end
hold off;
    