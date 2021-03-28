#include "uniformweightingscheme.cuh"

UniformWeightingScheme::UniformWeightingScheme() : WeightingScheme(){};
UniformWeightingScheme::UniformWeightingScheme(int threads) : WeightingScheme(threads){};
UniformWeightingScheme::UniformWeightingScheme(int threads, UVTaper * uvtaper) : WeightingScheme(threads, uvtaper){};

void UniformWeightingScheme::apply(std::vector<MSDataset>& d)
{

        std::cout << "Running Uniform weighting scheme with " << this->threads << " threads" << std::endl;
        float w;
        double3 uvw;
        std::vector<float> g_weights(M*N);
        std::vector<int2> xy_pos;
        double grid_pos_x, grid_pos_y;
        int x,y;
        for(int j=0; j < d.size(); j++) {
                for(int f=0; f < d[j].data.nfields; f++) {
                        for(int i=0; i < d[j].data.total_frequencies; i++) {
                                for(int s=0; s < d[j].data.nstokes; s++) {
                                        d[j].fields[f].backup_visibilities[i][s].weight.resize(d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        xy_pos.resize(d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);

                                        #pragma omp parallel for schedule(static, 1) num_threads(this->threads) shared(g_weights) private(x, y, grid_pos_x, grid_pos_y, uvw, w)
                                        for (int z = 0; z < d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]; z++)
                                        {
                                                // First we save the original weights
                                                uvw = d[j].fields[f].visibilities[i][s].uvw[z];
                                                w = d[j].fields[f].visibilities[i][s].weight[z];
                                                if(!this->modify_weights)
                                                  d[j].fields[f].backup_visibilities[i][s].weight[z] = w;

                                                uvw.x = metres_to_lambda(uvw.x, d[j].fields[f].nu[i]);
                                                uvw.y = metres_to_lambda(uvw.y, d[j].fields[f].nu[i]);
                                                uvw.z = metres_to_lambda(uvw.z, d[j].fields[f].nu[i]);

                                                //Apply hermitian symmetry (it will be applied afterwards)
                                                if (uvw.x < 0.0) {
                                                        uvw.x *= -1.0;
                                                        uvw.y *= -1.0;
                                                }

                                                grid_pos_x = uvw.x / fabs(deltau);
                                                grid_pos_y = uvw.y / fabs(deltav);
                                                x = round(grid_pos_x) + N / 2;
                                                y = round(grid_pos_y) + M / 2;

                                                if(x >= 0 && y >= 0 && x < N && y < M){
                                                  xy_pos[z].x = x;
                                                  xy_pos[z].y = y;
                                                  // And we grid the weights
                                                  #pragma omp critical
                                                  {
                                                          g_weights[N * y + x] += w;

                                                  }
                                                }else{
                                                  xy_pos[z].x = -1;
                                                  xy_pos[z].y = -1;
                                                }


                                        }

                                        #pragma omp parallel for schedule(static, 1) num_threads(this->threads) private(x, y, uvw)
                                        for (int z = 0; z < d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]; z++)
                                        {
                                          uvw = d[j].fields[f].visibilities[i][s].uvw[z];

                                          uvw.x = metres_to_lambda(uvw.x, d[j].fields[f].nu[i]);
                                          uvw.y = metres_to_lambda(uvw.y, d[j].fields[f].nu[i]);
                                          uvw.z = metres_to_lambda(uvw.z, d[j].fields[f].nu[i]);

                                          //Apply hermitian symmetry (it will be applied afterwards)
                                          if (uvw.x < 0.0) {
                                                  uvw.x *= -1.0;
                                                  uvw.y *= -1.0;
                                          }

                                          x = xy_pos[z].x;
                                          y = xy_pos[z].y;

                                          if(x >= 0 && y >= 0 && x < N && y < M)
                                            d[j].fields[f].visibilities[i][s].weight[z] /= g_weights[N*y + x];
                                          else
                                            d[j].fields[f].visibilities[i][s].weight[z] = 0.0f;

                                          if(NULL != this->uvtaper)
                                            d[j].fields[f].visibilities[i][s].weight[z] *= this->uvtaper->getValue(uvw.x, uvw.y);

                                          if(this->modify_weights)
                                            d[j].fields[f].backup_visibilities[i][s].weight[z] = d[j].fields[f].visibilities[i][s].weight[z];
                                        }

                                        std::fill_n(g_weights.begin(), M*N, 0.0f);


                                }

                        }

                }

        }

};

namespace {
WeightingScheme* CreateWeightingScheme()
{
        return new UniformWeightingScheme;
}
const std::string name = "Uniform";
const bool RegisteredUniformWeighting = registerCreationFunction<WeightingScheme, std::string>(name, CreateWeightingScheme);
};
