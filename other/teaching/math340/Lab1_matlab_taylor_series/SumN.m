function [output] = SumN(input)
    % The names "input" and "output" are chosen to make it clear how this
    % function works. The names don't matter as long as they are consistent.
    N=input;
    sum=0;
    for i=1:N
        % In each iteration we will add the new number to our sum
        sum=sum+i;
    end
    output=sum;
end