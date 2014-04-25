addpath toolbox;
addpath toolbox/pwmetric;

mX = mappedX(1:10000, :);
mWordTable = mappedX(10001:10010, :);

n_close = 2;
n_far = 2;
indices_to_show = zeros(1, 10 * (n_close + n_far));

dists = slmetric_pw(mWordTable', mX', 'eucdist');
j = 0;
for i = 1:10
    [~, idxs] = sort(dists(i, :));
    indices_to_show(j+1:j+n_close) = idxs(1:n_close);
    j = j + n_close;
    indices_to_show(j+1:j+n_far) = idxs(5000+(1:n_far));
    j = j + n_far;
end

visualize(mX, testY, mWordTable, label_names, data, indices_to_show);