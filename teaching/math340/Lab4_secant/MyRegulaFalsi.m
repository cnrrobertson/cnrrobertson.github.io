function [xs] = MyRegulaFalsi(f, a, b, tol)
    N = 0;
    xs = [];
    c = b - f(b) * ((b - a) / (f(b) - f(a)));
    while abs(f(c)) > tol
        c = b - f(b) * ((b - a) / (f(b) - f(a)));
        xs(N+1) = c;
        if f(c) == 0
            return
        elseif f(a)*f(c) < 0
            b = c;
        else
            a = c;
        end
        N = N + 1;
    end
end