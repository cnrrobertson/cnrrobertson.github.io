function [c, N] = MyBisectionMethod(f, a, b, tol)
    N = 0;
    while (b-a)/2 > tol
        c = (a + b) / 2;
        if f(c) == 0
            return
        elseif f(a)*f(c) < 0
            b = c;
        else
            a = c;
        end
        N = N + 1;
        disp(['At step ' num2str(N) ' approximation is ' num2str((a+b)/2)])
    end
end