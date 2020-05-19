/* -------------------------------------------------------------
 * Name            : rvgs.h (header file for the library rvgs.c)
 * Author          : Steve Park & Dave Geyer
 * Language        : ANSI C
 * Latest Revision : 11-03-96
 * --------------------------------------------------------------
 */

#if !defined( _RVGS_ )
#define _RVGS_

long Bernoulli(double p);
long Binomial(long n, double p);
long Equilikely(long a, long b);
long Geometric(double p);
long Pascal(long n, double p);
long Poisson(double m);

double Uniform(double a, double b);
double Exponential(double m);
double Erlang(long n, double b);
double Normal(double m, double s);
double Lognormal(double a, double b);
double Chisquare(long n);
double Student(long n);

#endif
