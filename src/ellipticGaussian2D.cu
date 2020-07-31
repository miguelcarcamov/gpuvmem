#include "ellipticGaussian2D.cuh"

float EllipticalGaussian2D::run(float deltau, float deltav){
  return 1.0;
};

namespace {
CKernel* CreateCKernel()
{
        return new EllipticalGaussian2D;
}
const int CKERNELID = 0;
const bool RegisteredCKernel = Singleton<CKernelFactory>::Instance().RegisterCKernel(CKERNELID, CreateCKernel);
};
