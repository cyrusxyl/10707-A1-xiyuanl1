function [data, labels] = ImportData(filename)
%IMPORTDATA Summary of this function goes here
%   Detailed explanation goes here
    set = textread(filename,'','delimiter',',');
    num_entries = size(set,1);
    data = set(:, 1:end-1);
    class_number = set(:, end)+1;
    
    labels = zeros(num_entries, 10);
    labels(sub2ind(size(labels), 1:num_entries, class_number')) = 1;
end

