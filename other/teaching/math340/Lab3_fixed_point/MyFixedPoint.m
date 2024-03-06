function [xs] = MyFixedPoint(g, x1, tol)
    xs = [x1];
    err = Inf;
    while err > tol
        xs = [xs g(xs(end))];
        err = abs(xs(end) - xs(end-1));
    end
end