function [W, b, delta_W, delta_b] = Train(W, b, delta_W, delta_b, train_data, train_label, learning_rate, momentum)
% [W, b] = Train(W, b, train_data, train_label, learning_rate) trains the network
% for one epoch on the input training data 'train_data' and 'train_label'. This
% function should returned the updated network parameters 'W' and 'b' after
% performing backprop on every data sample.


% This loop template simply prints the loop status in a non-verbose way.
% Feel free to use it or discard it
N = size(train_data,1);

L = length(W)+1;
order = randperm(N,N);
batch_grad_W = cell(1,L-1);
batch_grad_b = cell(1,L-1);

for l = 1:L-1
    batch_grad_W{l} = zeros(size(W{l}));
    batch_grad_b{l} = zeros(size(b{l}));
end

for i = 1:N
    X = train_data(order(i),:);
    Y = train_label(order(i),:);
    
    [output, act_h, act_a] = Forward(W, b, X);
    [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a);
    
    [W, b, delta_W, delta_b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate, delta_W, delta_b, momentum);
end

end
