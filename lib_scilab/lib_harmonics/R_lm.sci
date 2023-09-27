function Rlm = R_lm(l, theta, phi);
//   function Rlm = Rlm(l, theta, phi);
//   Spherical harmonics C_LM in the Racah normalization and
//   Condon & Shortley phase.
//   Returns C_LM(THETA,PHI) as a column vector
//   for M=-L:L, i.e. C_LM(THETA,PHI) = Rlm(M+L+1).
//   THETA and PHI are polar angles (radians).
//
//   adapted from P.E.S. Wormer, August, 2002.
//   TSelim March 27th, 2020

// Get phaseless, semi-normalized, associated Legendre functions:
sp     = sf_legendre(cos(theta), l);  // m=0,..,l
sp     = sp';
spud   = flipdim(sp,1);       // m=l,..,0
mpos   = [ 1: l]';
mneg   = [-l:-1]';
pow    = (-1).^mpos;
expneg = exp(%i.*mneg*phi)*sqrt(0.5);
exppos = (exp(%i.*mpos*phi).*pow)*sqrt(0.5);
Rlm    = [spud(1:l).*expneg; sp(1); sp(2:l+1).*exppos];

endfunction

