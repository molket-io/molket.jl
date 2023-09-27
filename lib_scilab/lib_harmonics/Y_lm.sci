function ylm = y_lm(l,theta,phi)
//Y_LM Spherical harmonics
//Y_LM(theta,phi) is returned
//   for M=-L:L as a matrix Y where:
// Y(L+M+1,I) = Y_LM(THETA(I),PHI)
// TSelim Feb. 2020

if nargin<3, phi=[]; end
P=sf_legendre(cos(theta), l)*sqrt( (l+0.5)/(4*%pi));
P=P';
fac=(-1).^(0:l)';
P(1,:)=P(1,:)*sqrt(2);
ylm=[flipdim(P(2:(l+1),:),1)
      diag(fac)*P];
if ~isempty(phi)
   ylm=diag( exp(%i*[-l:l]*phi) )*ylm;
end
endfunction

