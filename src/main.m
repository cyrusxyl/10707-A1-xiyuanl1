%% script setup
show_digit = 0;

%% loading data
if ~exist('data_loaded')
[train_data, train_labels] = ImportData('digitstrain.txt');
[valid_data, valid_labels] = ImportData('digitsvalid.txt');
[test_data, test_labels] = ImportData('digitstest.txt');
end
data_loaded = 1;
%% display some digits
if show_digit
    for i = randi([1 3000],1,100)
        digit = reshape(train_data(i,:), 28, 28)';
        imshow(digit);
        drawnow
    end
end

%% initialize network
layers = [784, 100, 10];
epoch = 30;
eta = 0.1;
momentum = 0.0;
batch_size = 1;

activ_fun = @(x) 1 ./ (1 + exp(-x));
%relu @(x) max(0,x);
%tanh @tanh;
%sigmoid @(x) 1 ./ (1 + exp(-x));
dactiv_fun = @(x) x.*(1-x);
%relu @(x) x>=0;
%tanh @(x) 1-tanh(x).^2;
%sigmoid @(x) x.*(1-x);

[W, b] = InitializeNetwork(layers);

train_acc = zeros(epoch+1,1);
train_loss = zeros(epoch+1,1)+inf;
valid_acc = zeros(epoch+1,1);
valid_loss = zeros(epoch+1,1)+inf;
current_best = inf;
%% train
create_zero_cell = @(x) zeros(size(x));
delta_W = cellfun(create_zero_cell, W, 'UniformOutput', false);
delta_b = cellfun(create_zero_cell, b, 'UniformOutput', false);

for j = 1:epoch
    [n_W, n_b, n_delta_W, n_delta_b] = ...
        Train(W, b, delta_W, delta_b, ...
        train_data, train_labels, ...
        eta, momentum, batch_size, ...
        activ_fun, dactiv_fun);
    
    [train_acc(j+1), train_loss(j+1)] = ComputeAccuracyAndLoss(...
        n_W, n_b, train_data, train_labels, activ_fun);
    [valid_acc(j+1), valid_loss(j+1)] = ComputeAccuracyAndLoss(...
        n_W, n_b, valid_data, valid_labels, activ_fun);
    
    if valid_loss(j+1)>current_best
        eta = max(1e-4, eta * 0.5);
    else
        current_best = valid_loss(j+1);
    end
    
    W = n_W;
    b = n_b;
    delta_W = n_delta_W;
    delta_b = n_delta_b;

    fprintf('Epoch %d - accuracy: %.5f, %.5f \t loss: %.5f, %.5f \n', j, train_acc(j+1), valid_acc(j+1), train_loss(j+1), valid_loss(j+1))
end
%% plot performance stats
% cross-entropy error
figure(1)
plot(0:epoch,train_loss)
hold on
plot(0:epoch,valid_loss)
title('cross-entropy error')
legend('train error','valid error')

% classification error
figure(2)
plot(0:epoch,1-train_acc)
hold on
plot(0:epoch,1-valid_acc)
title('classification error')
legend('train error','valid error')

%% test accuracy
[test_acc,test_loss] = ComputeAccuracyAndLoss(W, b, test_data, test_labels, activ_fun)
