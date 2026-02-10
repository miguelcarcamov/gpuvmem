#ifndef REGULARIZERS_HOST_CUH
#define REGULARIZERS_HOST_CUH

__host__ float L1Norm(float* I,
                      float* ds,
                      float penalization_factor,
                      float epsilon,
                      int mod,
                      int order,
                      int index,
                      int iter);

__host__ void DL1Norm(float* I,
                      float* dgi,
                      float penalization_factor,
                      float epsilon,
                      int mod,
                      int order,
                      int index,
                      int iter);

__host__ float GL1NormK(float* I,
                        float* prior,
                        float* ds,
                        float penalization_factor,
                        float epsilon_a,
                        float epsilon_b,
                        int mod,
                        int order,
                        int index,
                        int iter);

__host__ void DGL1Norm(float* I,
                       float* prior,
                       float* dgi,
                       float penalization_factor,
                       float epsilon_a,
                       float epsilon_b,
                       int mod,
                       int order,
                       int index,
                       int iter);

__host__ float SEntropy(float* I,
                        float* ds,
                        float prior_value,
                        float eta,
                        float penalization_factor,
                        int mod,
                        int order,
                        int index,
                        int iter);

__host__ void DEntropy(float* I,
                       float* dgi,
                       float prior_value,
                       float eta,
                       float penalization_factor,
                       int mod,
                       int order,
                       int index,
                       int iter);

__host__ float SGEntropy(float* I,
                         float* ds,
                         float* prior,
                         float eta,
                         float penalization_factor,
                         int mod,
                         int order,
                         int index,
                         int iter);

__host__ void DGEntropy(float* I,
                        float* dgi,
                        float* prior,
                        float eta,
                        float penalization_factor,
                        int mod,
                        int order,
                        int index,
                        int iter);

__host__ float laplacian(float* I,
                         float* ds,
                         float penalization_factor,
                         int mod,
                         int order,
                         int imageIndex,
                         int iter);

__host__ void DLaplacian(float* I,
                         float* dgi,
                         float penalization_factor,
                         float mod,
                         float order,
                         float index,
                         int iter);

__host__ float quadraticP(float* I,
                          float* ds,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter);

__host__ void DQuadraticP(float* I,
                          float* dgi,
                          float penalization_factor,
                          int mod,
                          int order,
                          int index,
                          int iter);

__host__ float l2ConstantPrior(float* I,
                               float* ds,
                               float prior_value,
                               float penalization_factor,
                               int mod,
                               int order,
                               int index,
                               int iter);

__host__ void DL2ConstantPrior(float* I,
                               float* dgi,
                               float prior_value,
                               float penalization_factor,
                               int mod,
                               int order,
                               int index,
                               int iter);

__host__ float isotropicTV(float* I,
                           float* ds,
                           float epsilon,
                           float penalization_factor,
                           int mod,
                           int order,
                           int index,
                           int iter);

__host__ void DIsotropicTV(float* I,
                           float* dgi,
                           float epsilon,
                           float penalization_factor,
                           int mod,
                           int order,
                           int index,
                           int iter);

__host__ float anisotropicTV(float* I,
                             float* ds,
                             float epsilon,
                             float penalization_factor,
                             int mod,
                             int order,
                             int index,
                             int iter);

__host__ void DAnisotropicTV(float* I,
                             float* dgi,
                             float epsilon,
                             float penalization_factor,
                             int mod,
                             int order,
                             int index,
                             int iter);

__host__ float TotalSquaredVariation(float* I,
                                     float* ds,
                                     float penalization_factor,
                                     int mod,
                                     int order,
                                     int index,
                                     int iter);

__host__ void DTSVariation(float* I,
                           float* dgi,
                           float penalization_factor,
                           int mod,
                           int order,
                           int index,
                           int iter);

#endif  // REGULARIZERS_HOST_CUH
