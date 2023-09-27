

exec('/home/tselim/sci/lib5/lib_harmonics/s_lm.sci');
exec('/home/tselim/sci/lib5/lib_harmonics/R_lm.sci');
exec('/home/tselim/sci/lib5/lib_harmonics/Y_lm.sci');

exec('heco2_nu2_pes_eval_pmax4_Qcanonical_Qmax2_LRdamped_opt.sci');
// reading tesseral harmonics with the good phase
exec('/home/tselim/sci/lib5/lib_harmonics/slm_mphase.sci');

l = 2
m = 2
theta = 90*%pi/180
phi   = 90*%pi/180

Y =  y_lm(l,theta,phi)
norm =(-1)^m * sqrt(2*factorial(l-m)/factorial(l+m))
norm2 = sqrt((2*l+1)/(4*%pi))
ass_legendre = sf_legendre(cos(theta), 2)/norm

Y =  y_lm(l,theta,phi)
C = R_lm(l, theta, phi)


exec('/vol/thchem/gsci_lib6/v1.5.0/lib_poly/laguerre.sci')
 Ln = p_laguerre(1, 1)
 horner(Ln,1)
