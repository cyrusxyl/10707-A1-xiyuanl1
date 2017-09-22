%% loading data
[train_data, train_labels] = ImportData('digitstrain.txt');
[valid_data, valid_labels] = ImportData('digitsvalid.txt');
[test_data, test_labels] = ImportData('digitstest.txt');

%% display some digits
for i = randi([1 3000],1,100)
    digit = reshape(train_data(i,:), 28, 28)';
    imshow(digit);
    drawnow
end

%% initialize network
layers = [784, 100, 10];
epoch = 200;
eta = 0.1;

[W, b] = InitializeNetwork(layers);

train_acc = zeros(epoch+1,1);
train_loss = zeros(epoch+1,1)+inf;
valid_acc = zeros(epoch+1,1);
valid_loss = zeros(epoch+1,1)+inf;
current_best = inf;
%% train
for j = 1:epoch
    [n_W, n_b] = Train(W, b, train_data, train_labels, eta);
    
    [train_acc(j+1), train_loss(j+1)] = ComputeAccuracyAndLoss(n_W, n_b, train_data, train_labels);
    [valid_acc(j+1), valid_loss(j+1)] = ComputeAccuracyAndLoss(n_W, n_b, valid_data, valid_labels);
    
    if valid_loss(j+1)>current_best
        eta = eta * 0.5
    else
        current_best = valid_loss(j+1)
        W = n_W;
        b = n_b;
    end
    fprintf('Epoch %d - accuracy: %.5f, %.5f \t loss: %.5f, %.5f \n', j, train_acc(j+1), valid_acc(j+1), train_loss(j+1), valid_loss(j+1))
end
%% plot performance stats
    figure(1)
    plot(0:epoch,train_loss)
    hold on
    plot(0:epoch,valid_loss)
    title('Network Loss')
    legend('train loss','valid loss')

%% test accuracy
[test_acc,test_loss] = ComputeAccuracyAndLoss(W, b, test_data, test_labels)
