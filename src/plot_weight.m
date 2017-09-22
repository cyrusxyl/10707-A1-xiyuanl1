% Cyrus Liu
% the Robotics Institute, Carnegie Mellon University

N = size(W{1},1);

ax = ceil(sqrt(N));
weights = reshape(W{1},N,28,28);
for i = 1:N
    subplot(ax,ax,i);
    w = squeeze(weights(i,:,:));
    imagesc(w);
    colormap gray
    axis image
    axis off
end