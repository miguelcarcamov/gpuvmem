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

#include "mnbrak.cuh"
#include "nrutil.h"
#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define SHFT(a,b,c,d) (a)=(b); (b)=(c); (c)=(d);

__host__ void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb, float *fc, float (*func)(float))
{
        float ulim,u,r,q,fu,dum;
        *fa = (*func)(*ax);
        *fb = (*func)(*bx);
        if(*fb > *fa) {
                SHFT(dum,*ax,*bx,dum)
                SHFT(dum,*fb,*fa,dum)
        }
        *cx=(*bx)+GOLD*(*bx-*ax);
        *fc=(*func)(*cx);
        while (*fb > *fc) {
                r=(*bx-*ax)*(*fb-*fc);
                q=(*bx-*cx)*(*fb-*fa);
                u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/
                   (2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
                ulim=(*bx)+GLIMIT*(*cx-*bx);
                if ((*bx-u)*(u-*cx) > 0.0) {
                        fu=(*func)(u);
                        if (fu < *fc) {
                                *ax=(*bx);
                                *bx=u;
                                *fa=(*fb);
                                *fb=fu;
                                return;
                        } else
                        if (fu > *fb) {
                                *cx=u;
                                *fc=fu;
                                return;
                        }

                        u=(*cx)+GOLD*(*cx-*bx);
                        fu=(*func)(u);
                } else
                if ((*cx-u)*(u-ulim) > 0.0) {
                        fu=(*func)(u);
                        if (fu < *fc) {
                                SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx))
                                SHFT(*fb,*fc,fu,(*func)(u))
                        }
                } else if ((u-ulim)*(ulim-*cx) >= 0.0) {
                        u=ulim;
                        fu=(*func)(u);
                } else {
                        u=(*cx)+GOLD*(*cx-*bx);
                        fu=(*func)(u);
                }
                SHFT(*ax,*bx,*cx,u)
                SHFT(*fa,*fb,*fc,fu)
        }

}
#undef GOLD
#undef GLIMIT
#undef TINY
#undef SHFT
