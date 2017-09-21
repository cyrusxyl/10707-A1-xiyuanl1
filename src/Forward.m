function [output, act_h, act_a] = Forward(W, b, X)
% [OUT, act_h, act_a] = Forward(W, b, X) performs forward propogation on the
% input data 'X' uisng the network defined by weights and biases 'W' and 'b'
% (as generated by InitializeNetwork(..)).
%
% This function should return the final softmax output layer activations in OUT,
% as well as the hidden layer post activations in 'act_h', and the hidden layer
% pre activations in 'act_a'.
sigmoid = @(x) 1 ./ (1 + exp(-x));

L = length(W)+1;

act_a = cell(1,L);
act_h = cell(1,L);

act_h{1,1} = X';

for i = 2:L-1
    act_a{1,i} = W{1,i-1} * act_h{1,i-1} + b{1,i-1};
    act_h{1,i} = sigmoid(act_a{1,i});
end

act_a{1,L} = W{1,L-1} * act_h{1,L-1} + b{1,L-1};
act_h{1,L} = softmax(act_a{1,L});

output = act_h{1,L};
end