function y = sf_legendre(x, n)
//SF_LEGENDRE Associated Legendre function.
//   P = SF_LEGENDRE(X, N) computes the associated Legendre functions 
//   of degree N and order M = 0, 1, ..., N, evaluated for each element
//   of X.  N must be a scalar integer and X must contain real values
//   between -1 <= X <= 1.  
//
//   If X is a vector, P is an L-by-(N+1) matrix, where L = length(X).
//   The P(i, M+1) entry corresponds to the associated Legendre function 
//   of degree N and order M evaluated at X(i). 
//
//   Unnormalized associated Legendre functions are defined by
// 
//       P(N,M;X) = (-1)^M * (1-X^2)^(M/2) * (d/dX)^M ( P(N,X) ),
//
//   where P(N,X) is the Legendre polynomial of degree N. Note that
//   the first column of P is the Legendre polynomial evaluated at X 
//   (the M == 0 case).
//
//   SF_LEGENDRE(X, N) computes the Schmidt semi-normalized
//   associated Legendre functions SP(N,M;X). These functions are 
//   related to the unnormalized associated Legendre functions 
//   P(N,M;X) by:
//               
//   SP(N,M;X) = P(N,X), M = 0
//             = (-1)^M * sqrt(2*(N-M)!/(N+M)!) * P(N,M;X), M > 0
//
//   Acknowledgment:
//
//   This program is based on a Fortran program by Robert L. Parker,
//   Scripps Institution of Oceanography, Institute for Geophysics and 
//   Planetary Physics, UCSD. February 1993.
//
//   Reference:
//     [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical
//         Functions", Dover Publications, 1965, Ch. 8.
//     [2] J. A. Jacobs, "Geomagnetism", Academic Press, 1987, Ch.4.
//
//   Note on Algorithm:
//
//   LEGENDRE uses a three-term backward recursion relationship in M.
//   This recursion is on a version of the Schmidt semi-normalized 
//   Associated Legendre functions SPc(n,m;x), which are complex 
//   spherical harmonics. These functions are related to the standard 
//   Abramowitz & Stegun functions P(n,m;x) by
//
//       P(n,m;x) = sqrt((n+m)!/(n-m)!) * SPc(n,m;x)
//   
//   They are related to the Schmidt form given previously by:
//
//       SP(n,m;x) = SPc(n,0;x), m = 0
//                 = (-1)^m * sqrt(2) * SPc(n,m;x), m > 0

if size(n,'*') > 1 | type(n)<>1 | n < 0 | n <> round(n)
    error('N must be a positive scalar integer')
end

if type(x)<>1 | max(abs(x)) > 1
    error('X must be real and in the range (-1,1)')
end

// Convert x to a single row vector
x  = x(:)'
nx = length(x)

// The n = 0 case
if n == 0
    y = ones(nx, 1)
    return
end

rootn = sqrt(0:2*n)
s     = sqrt(1-x.^2)
P     = zeros(n+3, nx)

// Calculate TWOCOT, separating out the x = -1,+1 cases first
twocot = x

// Evaluate x = +/-1 first to avoid error messages for division by zero
k = find(x==-1)
twocot(k) = %inf

k = find(x==1)
twocot(k) = -%inf

k = find(s)
twocot(k) = -2*x(k)./s(k)

// Find values of x,s for which there will be underflow

sn  = (-s).^n
tol = sqrt(2.225073858507201e-308) // realmin
ind = find(s>0 & abs(sn)<=tol)
if ind <> []
    // Approx solution of x*ln(x) = y 
    v = 9.2-log(tol)*ones(size(ind,1),size(ind,2))./(n*s(ind))
    w = ones(size(v,1),size(v,2))./log(v)
    m1 = 1+n*s(ind).*v.*w.*(1.0058+ w.*(3.819 - w*12.173))
    m1 = min(n, floor(m1))

    deff('[z]=rem(x,y)','n=fix(x./y), z=x-n.*y')

    // Column-by-column recursion
    for k = 1:length(m1)
        mm1 = m1(k)
        col = ind(k)
        P(mm1:n+1,col) = zeros(1,length(mm1:n+1))'

        // Start recursion with proper sign
        tstart = %eps
        P(mm1,col) = sign(rem(mm1,2)-0.5)*tstart
        if x(col) < 0
            P(mm1,col) = sign(rem(n+1,2)-0.5)*tstart
        end
       // Recur from m1 to m = 0, accumulating normalizing factor.
        sumsq = tol
        for m = mm1-2:-1:0
            P(m+1,col) = ((m+1)*twocot(col)*P(m+2,col)- ...
                  rootn(n+m+3)*rootn(n-m)*P(m+3,col))/ ...
                  (rootn(n+m+2)*rootn(n-m+1))
            sumsq = P(m+1,col)^2 + sumsq
        end
        scale = 1/sqrt(2*sumsq - P(1,col)^2)
        P(1:mm1+1,col) = scale*P(1:mm1+1,col)
    end     // FOR loop
end // small sine IF loop

// Find the values of x,s for which there is no underflow, and for
// which twocot is not infinite (x<>1).

// nind = find(x<>1 & abs(sn)>=tol)
// GCG, 24-Jan-2005
nind = find(abs(x)<>1 & abs(sn)>=tol)
if nind <> []

   // Produce normalization constant for the m = n function
    d = 2:2:2*n
    c = prod(1-ones(size(d,1),size(d,2))./d)

    // Use sn = (-s).^n (written above) to write the m = n function
    P(n+1,nind) = sqrt(c)*sn(nind)
    P(n,nind) = P(n+1,nind).*twocot(nind)*n./rootn($)

    // Recur downwards to m = 0
    for m = n-2:-1:0
        P(m+1,nind) = (P(m+2,nind).*twocot(nind)*(m+1) ...
            -P(m+3,nind)*rootn(n+m+3)*rootn(n-m))/ ...
            (rootn(n+m+2)*rootn(n-m+1))
    end
end // IF loop

y = P(1:n+1,:)

// Polar argument   (x = +-1)
s0 = find(s == 0)
y(1,s0) = x(s0).^n

// Calculate the standard Schmidt semi-normalized functions.
// For m = 1,...,n, multiply by (-1)^m*sqrt(2)
row1   = y(1,:)
y      = sqrt(2)*y
y(1,:) = row1;    // restore first row
const1 = 1
for r = 2:n+1
    const1 = -const1
    y(r,:) = const1*y(r,:)
end
y = y.'
endfunction

