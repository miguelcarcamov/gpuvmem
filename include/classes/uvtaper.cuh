#ifndef UVTAPER_CUH
#define UVTAPER_CUH

class UVTaper
{
public:
__host__ UVTaper(){
      this->amplitude = 1.0f;
      this->u_0 = 0.0;
      this->v_0 = 0.0;
      this->bmaj = 0.0f;
      this->bmin = 0.0f;
      this->bpa = 0.0f;
};

__host__ UVTaper(float size){
      this->amplitude = 1.0f;
      this->u_0 = 0.0;
      this->v_0 = 0.0;
      this->bmaj = size;
      this->bmin = size;
      this->bpa = 0.0f;
};

__host__ UVTaper(float bmaj, float bmin){
      this->amplitude = 1.0f;
      this->u_0 = 0.0;
      this->v_0 = 0.0;
      this->bmaj = bmaj;
      this->bmin = bmin;
      this->bpa = 0.0f;
};

__host__ UVTaper(float bmaj, float bmin, float bpa){
      this->amplitude = 1.0f;
      this->u_0 = 0.0;
      this->v_0 = 0.0;
      this->bmaj = bmaj;
      this->bmin = bmin;
      this->bpa = bpa;
};

__host__ float getBMaj(){
      return this->bmaj;
};

__host__ float getBMin(){
      return this->bmin;
};

__host__ float getBPA(){
      return this->bpa;
};

__host__ void setAmplitude(float amplitude){
      this->amplitude = amplitude;
};

__host__ void setBMaj(float bmaj){
      this->bmaj = bmaj;
};

__host__ void setBMin(float bmin){
      this->bmin = bmin;
};

__host__ void setBPA(float bpa){
      this->bpa = bpa;
};

__host__ void setCenter(double u_0, double v_0){
      this->u_0 = u_0;
      this->v_0 = v_0;
};

__host__ void setGaussianParameters(float bmaj, float bmin, float bpa){
      this->bmaj = bmaj;
      this->bmin = bmin;
      this->bpa = bpa;
};

__host__ float ellipticGaussianValue(double u, double v, double u_0, double v_0)
{
      double x = u - u_0;
      double y = v - v_0;

      float value = this->amplitude*exp((x*x/(2*this->bmaj*this->bmaj) - this->bpa*x*y/(this->bmaj*this->bmin) - y*y/(2*this->bmin*this->bmin)));
      return value;
};

__host__ float getValue(double u, double v)
{
      double x = u - this->u_0;
      double y = v - this->v_0;

      float value = this->amplitude*exp((x*x/(2*this->bmaj*this->bmaj) - this->bpa*x*y/(this->bmaj*this->bmin) - y*y/(2*this->bmin*this->bmin)));
      return value;
};

private:
  float bmaj;
  float bmin;
  float bpa;
  float amplitude;
  double u_0;
  double v_0;
};
#endif //UVTAPER_CUH
