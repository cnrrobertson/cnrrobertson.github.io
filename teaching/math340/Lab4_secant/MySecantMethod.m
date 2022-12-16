function [xs] = MySecantMethod(f, x1, x2, tol, Nmax)
    g = @(x1,x2) x2 - f(x2) * ((x2 - x1) / (f(x2) - f(x1)));
    xs = [x1,x2];
    err = Inf;
    while err > tol
        xs = [xs g(xs(end-1), xs(end))];
        err = abs(f(xs(end)));
        if length(xs) > Nmax
            return;
        end
    end
end