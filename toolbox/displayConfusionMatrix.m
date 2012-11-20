function [] = displayConfusionMatrix( confusion, categoryLabels )

numCategories = length(categoryLabels);
disp('Confusion matrix:');
str = sprintf('%-7s', '...');
for i = 1:numCategories
    categ = char(categoryLabels(i));
    if length(categ) > 4
        categ = categ(1:4);
    end
    str = sprintf('%s%-7s', str, categ);
end
str = sprintf('%s\n', str);
disp(str);
for i = 1:numCategories
    categ = char(categoryLabels(i));
    if length(categ) > 4
        categ = categ(1:4);
    end
    str = sprintf('%-7s', categ);
    for j = 1:numCategories
        str = sprintf('%s%-7d', str, confusion(i, j));
    end
    str = sprintf('%s\n', str);
    disp(str);
end

end

