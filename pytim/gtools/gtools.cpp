#include <vcl/vectorclass.h>
#include <vcl/vector3d.h>
#include <iostream>
#include <cmath>
#include <ctime>

#ifdef __cplusplus
extern "C" {
#endif

double sphere(int idx0, int idx1, int idx2, int idx3, 
              const double* points, const double* weights, double* center) {
	Vec3d r1,r2,r3,r4;
	r1.load_partial(3, points+3*idx0);
	r2.load_partial(3, points+3*idx1);
	r3.load_partial(3, points+3*idx2);
	r4.load_partial(3, points+3*idx3);
	
	Vec3d R1(weights[idx0]);
	Vec3d R2end(weights[idx1], weights[idx2], weights[idx3]);  
	
	Vec3d M1,M2,M3;
	M1 = r1 - r2;
	M2 = r1 - r3;
	M3 = r1 - r4;

    Vec3d d = R1 - R2end;
   
    double rr1 = dot_product(r1, r1);
    
    Vec3d sp1(rr1);
    Vec3d sp2(dot_product(r2, r2), dot_product(r3, r3), dot_product(r4, r4));
    
    Vec3d s = 0.5*(sp1 - sp2 - R1*R1 + R2end*R2end);
    
    Vec4d p1 = permute4d<1,2,0,-1>(M1);
    Vec4d p2 = permute4d<2,0,1,-1>(M2);
    Vec4d p3 = permute4d<0,1,2,-1>(M3);
    
    double dt = horizontal_add(p1 * p2 * p3);
    
    p1 = permute4d<2,0,1,-2>(M1);
    p2 = permute4d<1,2,0,-2>(M2);
    
    dt -= horizontal_add(p1 * p2 * p3);
    
    Vec3d det(dt);
    
    Vec3d iM1(-M2[2]*M3[1] + M2[1]*M3[2], M1[2]*M3[1] - M1[1]*M3[2], -M1[2]*M2[1] + M1[1]*M2[2]);
    iM1 /= det; 
    
    Vec3d iM2(M2[2]*M3[0] - M2[0]*M3[2], -M1[2]*M3[0] + M1[0]*M3[2], M1[2]*M2[0] - M1[0]*M2[2]);
    iM2 /= det; 
    
    Vec3d iM3(-M2[1]*M3[0] + M2[0]*M3[1], M1[1]*M3[0] - M1[0]*M3[1], -M1[1]*M2[0] + M1[0]*M2[1]);
    iM3 /= det; 
    
    Vec3d u(dot_product(iM1,d), dot_product(iM2,d), dot_product(iM3,d));
    Vec3d r0(dot_product(iM1,s), dot_product(iM2,s), dot_product(iM3,s));
    
    Vec3d v = r1 - r0;
    
    double a = 1.0 - dot_product(u,u);
    double b = 2*(R1[0] - dot_product	(u,v));
    double c = (R1[0]*R1[0] - dot_product(v,v));
    double ds2 = b*b-4*a*c;
    
    if(ds2 < 0)
		return 1e300;
    
    double ds = sqrt(ds2);
    
    double Rp  = (-b+ds)/(2*a);
	double Rm  = (-b-ds)/(2*a);
	double Rs;
	
	if(Rp < 0 && Rm < 0)
		return 1e300;
		
	if(Rp < 0)
		Rs = Rm;
	else {
		if (Rm < 0)
			Rs = Rp;
		else
			Rs = Rm < Rp ? Rm : Rp;
	}
    
    Vec3d Rsv(Rs);
    Vec3d cntr = r0 - Rsv*u;
    cntr.store_partial(3, center);
    return Rs; 
}
#ifdef __cplusplus
}
#endif

