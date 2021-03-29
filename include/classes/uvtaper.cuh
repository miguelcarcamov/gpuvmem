#ifndef UVTAPER_CUH
#define UVTAPER_CUH

class UVTaper
{
public:
__host__ UVTaper(){
      this->amplitude = 1.0f;
      this->u_0 = 0.0;
      this->v_0 = 0.0;
      this->sigma_maj = 0.0f;
      this->sigma_min = 0.0f;
      this->bpa = 0.0f;
};

__host__ UVTaper(float size){
      this->amplitude = 1.0f;
      this->u_0 = 0.0;
      this->v_0 = 0.0;
      this->sigma_maj = size;
      this->sigma_min = size;
      this->bpa = 0.0f;
};

__host__ UVTaper(float sigma_maj, float sigma_min){
      this->amplitude = 1.0f;
      this->u_0 = 0.0;
      this->v_0 = 0.0;
      this->sigma_maj = sigma_maj;
      this->sigma_min = sigma_min;
      this->bpa = 0.0f;
};

__host__ UVTaper(float sigma_maj, float sigma_min, float bpa){
      this->amplitude = 1.0f;
      this->u_0 = 0.0;
      this->v_0 = 0.0;
      this->sigma_maj = sigma_maj;
      this->sigma_min = sigma_min;
      this->bpa = bpa;
};

__host__ float getSigma_maj(){
      return this->sigma_maj;
};

__host__ float getSigma_min(){
      return this->sigma_min;
};

__host__ float getBPA(){
      return this->bpa;
};

__host__ void setAmplitude(float amplitude){
      this->amplitude = amplitude;
};

__host__ void setSigma_maj(float sigma_maj){
      this->sigma_maj = sigma_maj;
};

__host__ void setSigma_min(float sigma_min){
      this->sigma_min = sigma_min;
};

__host__ void setFWHM(float fwhm){
      float sigma_val = fwhm/(2.0f*sqrtf(2.0f*logf(2.0f)));
      this->sigma_maj = sigma_val;
      this->sigma_min = sigma_val;
};

__host__ void setFWHM(float fwhm_maj, float fwhm_min){
      this->sigma_maj = fwhm_maj/(2.0f*sqrtf(2.0f*logf(2.0f)));
      this->sigma_min = fwhm_min/(2.0f*sqrtf(2.0f*logf(2.0f)));
};

__host__ void setFWHM_arcsec(float fwhm_sky_arcsec){
      float fwhm_sky_rad  = (fwhm_sky_arcsec/3600.0f)*PI/180.0;
      float fwhm_lambda = 1.0f/fwhm_sky_rad;
      float sigma_val = fwhm_lambda/(2.0f*sqrtf(2.0f*logf(2.0f)));
      this->sigma_maj = sigma_val;
      this->sigma_min = sigma_val;
};

__host__ void setFWHM_arcsec(float fwhm_maj_sky_arcsec, float fwhm_min_sky_arcsec){
      float fwhm_maj_sky_rad  = (fwhm_maj_sky_arcsec/3600.0f)*PI/180.0;
      float fwhm_maj_lambda = 1.0f/fwhm_maj_sky_rad;

      float fwhm_min_sky_rad  = (fwhm_min_sky_arcsec/3600.0f)*PI/180.0;
      float fwhm_min_lambda = 1.0f/fwhm_min_sky_rad;

      this->sigma_maj = fwhm_maj_lambda/(2.0f*sqrtf(2.0f*logf(2.0f)));
      this->sigma_min = fwhm_min_lambda/(2.0f*sqrtf(2.0f*logf(2.0f)));
};


__host__ void setBPA(float bpa){
      this->bpa = bpa;
};

__host__ void setCenter(double u_0, double v_0){
      this->u_0 = u_0;
      this->v_0 = v_0;
};

__host__ void setGaussianParameters(float sigma_maj, float sigma_min, float bpa){
      this->sigma_maj = sigma_maj;
      this->sigma_min = sigma_min;
      this->bpa = bpa;
};

__host__ float getValue(double u, double v)
{
      double x = u - this->u_0;
      double y = v - this->v_0;

      float cos_bpa = cosf(this->bpa);
      float sin_bpa = sinf(this->bpa);
      float sin_bpa_2 = sinf(2.0f*this->bpa);
      float a = (cos_bpa*cos_bpa)/(2.0f*this->sigma_maj*this->sigma_maj) + (sin_bpa*sin_bpa)/(2.0f*this->sigma_min*this->sigma_min);
      float b = sin_bpa_2/(2.0f*this->sigma_maj*this->sigma_maj) - sin_bpa_2/(2.0f*this->sigma_min*this->sigma_min);
      float c = (sin_bpa*sin_bpa)/(2.0f*this->sigma_maj*this->sigma_maj) + (cos_bpa*cos_bpa)/(2.0f*this->sigma_min*this->sigma_min);
      float value = this->amplitude*exp(-a*x*x - b*x*y - c*y*y);
      return value;
};

private:
  float sigma_maj;
  float sigma_min;
  float bpa;
  float amplitude;
  double u_0;
  double v_0;
};
#endif //UVTAPER_CUH
