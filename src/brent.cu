/* -------------------------------------------------------------------------
   Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
   Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

   This program includes Numerical Recipes (NR) based routines whose
   copyright is held by the NR authors. If NR routines are included,
   you are required to comply with the licensing set forth there.

   Part of the program also relies on an an ANSI C library for multi-stream
   random number generation from the related Prentice-Hall textbook
   Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
   for more information please contact leemis@math.wm.edu

   Additionally, this program uses some NVIDIA routines whose copyright is held
   by NVIDIA end user license agreement (EULA).

   For the original parts of this code, the following license applies:

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------
 */

#include "brent.cuh"
#include "nrutil.h"
#define ITMAX 500
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SHFT(a,b,c,d) (a)=(b); (b)=(c); (c)=(d);
__host__ float brent(float ax, float bx, float cx, float tol, float *xmin, float (*f)(float))
{
        float a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
        float e = 0.0;

        a = (ax < cx ? ax : cx);
        b = (ax > cx ? ax : cx);
        x = w = v = bx;
        fw=fv=fx=(*f)(x);
        for (int iter=1; iter<=ITMAX; iter++) {
                xm=0.5*(a+b);
                tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
                if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
                        *xmin=x;
                        return fx;
                }
                if (fabs(e) > tol1) {
                        r=(x-w)*(fx-fv);
                        q=(x-v)*(fx-fw);
                        p=(x-v)*q-(x-w)*r;
                        q=2.0*(q-r);

                        if (q > 0.0) {
                                p = -p;
                        }

                        q=fabs(q);
                        etemp=e;
                        e=d;

                        if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
                                d=CGOLD*(e=(x >= xm ? a-x : b-x));
                        else {
                                d=p/q;
                                u=x+d;
                                if (u-a < tol2 || b-u < tol2) {
                                        d=SIGN(tol1,xm-x);
                                }
                        }
                } else {
                        d=CGOLD*(e=(x >= xm ? a-x : b-x));
                }
                u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
                fu=(*f)(u);
                if (fu <= fx) {
                        if (u >= x) {
                                a=x;
                        }else{
                                b=x;
                        }
                        SHFT(v,w,x,u)
                        SHFT(fv,fw,fx,fu)
                } else {
                        if (u < x) {
                                a=u;
                        } else{
                                b=u;
                        }

                        if (fu <= fw || w == x) {
                                v=w;
                                w=u;
                                fv=fw;
                                fw=fu;
                        } else
                        if (fu <= fv || v == x || v == w) {
                                v=u;
                                fv=fu;
                        }
                }
        }
        printf("Too many iterations in brent\n");
        *xmin=x;
        return fx;

}
#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SHFT
