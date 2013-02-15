load gauss_cifar10_huang_cat_truck/out_cat_truck.mat;

set(gcf,'paperunits','centimeters')
set(gcf,'papersize',[30,20]) % Desired outer dimensionsof figure
set(gcf,'paperposition',[0,0,30,20]) % Place plot on figure

thresholds = 0:0.1:1;
subplot(2,2,1);
hold on;
plot(thresholds, pdfUnseenAccuracies, 'r-+', 'LineWidth', 2);
plot(thresholds, pdfSeenAccuracies, 'r-+', 'LineWidth', 2);
h_title = title('(a) Gaussian model');
h_xl = xlabel('Fraction of points classified as unseen');
h_yl = ylabel('Accuracy');
set(h_title, 'FontSize', 24);
set(h_xl, 'FontSize', 24);
set(h_yl, 'FontSize', 24);
subplot(2,2,2);
hold on;
plot(thresholds, loopUnseenAccuracies, 'b-o', 'LineWidth', 2);
plot(thresholds, loopSeenAccuracies, 'b-o', 'LineWidth', 2);
h_title = title('(b) LoOP model');
h_xl = xlabel('Outlier probability threshold');
h_yl = ylabel('Accuracy');
set(h_title, 'FontSize', 24);
set(h_xl, 'FontSize', 24);
set(h_yl, 'FontSize', 24);
subplot(2,2,3.5);
hold on;
h_title = title('(c) Comparison');
plot(thresholds, pdfAccuracies, 'r-', 'LineWidth', 2);
plot(thresholds, loopAccuracies, 'b-', 'LineWidth', 2);
h_legend = legend({'Gaussian', 'LoOP'});
set(h_title, 'FontSize', 24);
h_xl = xlabel('Fraction of points classified as unseen/outlier probability threshold');
h_yl = ylabel('Accuracy');
set(h_legend, 'FontSize', 16);
set(h_xl, 'FontSize', 24);
set(h_yl, 'FontSize', 24);
hold off;