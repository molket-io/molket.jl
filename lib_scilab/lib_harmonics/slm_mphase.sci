function slm=S_lmm(l,theta,phi)
//S_LM Real spherical (Tesseral) harmonics in the Racah normalization.
//   SLM=S_LM(L,THETA,PHI) returns S_LM(THETA,PHI) where
//   S_LM(THETA,PHI) =
//  [(-1)^M Y_LM(THETA,PHI)+Y_L(-M)(THETA,PHI)]/sqrt(2)     for M > 0
//          Y_L0(THETA,0)                                   for M = 0
//  [-i (-1)^M Y_LM(THETA,PHI)+iY_L(-M)(THETA,PHI)]/sqrt(2) for M<0
//   Y_LM's are Racah normalized spherical harmonics.
//   S_LM(THETA,PHI) is returned as a column vector for M=-L:L, i.e.
//   S_LM(THETA,PHI) = SLM(L+1+M).
//   THETA and PHI are polar and azimuthal angle (radians), resp.
//
// P.E.S. Wormer, September 17, 2001.
// TSelim July 2nd, 2020: adapt the function for scilab
// GCG & TSelim 16-sept-2021: only the real part of s_lm
// include the (-1)^(m) in the good position


dum=R_lm(l,theta,phi);

for m=1:l,
    Slm(m+l+1)=real(dum(m+l+1) + (-1)^m.*dum(-m+l+1) )/sqrt(2);
end

Slm(l+1)=real(dum(l+1));

for m=-l:-1,
  Slm(m+l+1)=real( -%i*dum(-m+l+1) + (-1)^(-m).*%i.*dum(m+l+1) )/sqrt(2);
end

slm=Slm;

endfunction

