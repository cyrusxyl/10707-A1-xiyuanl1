function [W, b] = InitializeNetwork(layers)
% InitializeNetwork([INPUT, HIDDEN, OUTPUT]) initializes the weights and biases
% for a fully connected neural network with input data size INPUT, output data
% size OUTPUT, and HIDDEN number of hidden units.
% It should return the cell arrays 'W' and 'b' which contain the randomly
% initialized weights and biases for this neural network.

% number of layers
L = length(layers);

W = cell(1,L-1);
b = cell(1,L-1);

for i = 1:L-1
    n_in = layers(i);
    n_out = layers(i+1);
    
    bound = sqrt(6/(n_in+n_out));
    W{1,i} = 2*bound.*rand(n_out, n_in) - bound;
    b{1,i} = zeros(n_out, 1);
end


end
