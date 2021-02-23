#include "briggsweightingscheme.cuh"

BriggsWeightingScheme::BriggsWeightingScheme(){
};

float BriggsWeightingScheme::getRobustParam(){
        return this->robust_param;
};

void BriggsWeightingScheme::setRobustParam(float robust_param){
        if(robust_param >= -2.0 && robust_param <= 2.0) {
                this->robust_param = robust_param;
        }else{
                printf("Error. Robust parameter must have values between -2.0 and 2.0\n");
                exit(-1);
        }

};

void BriggsWeightingScheme::configure(void *params){
        float robust_param = *(float *)params;
        this->setRobustParam(robust_param);
        printf("Using robust %1.1f for Briggs weighting\n", this->getRobustParam());
};

void BriggsWeightingScheme::apply(std::vector<MSDataset>& d)
{

        float w;
        double3 uvw;
        std::vector<float> g_weights(M*N);
        std::vector<int2> xy_pos;
        double grid_pos_x, grid_pos_y;
        int x,y;

        float f_squared;
        float sum_original_weights = 0.0f;
        float sum_gridded_weights_squared = 0.0f;
        float average_weights;

        // Get the sum of the original weights
        for(int j=0; j < d.size(); j++) {
                for(int f=0; f < d[j].data.nfields; f++) {
                        for(int i=0; i < d[j].data.total_frequencies; i++) {
                                for(int s=0; s < d[j].data.nstokes; s++) {
                                        sum_original_weights += std::accumulate(d[j].fields[f].visibilities[i][s].weight.begin(), d[j].fields[f].visibilities[i][s].weight.end(), 0.0f);
                                }
                        }
                }
        }

        // Grid the weights and get the sum of the gridded weights squared
        for(int j=0; j < d.size(); j++) {
                for(int f=0; f < d[j].data.nfields; f++) {
                        for(int i=0; i < d[j].data.total_frequencies; i++) {
                                for(int s=0; s < d[j].data.nstokes; s++) {
                                        d[j].fields[f].backup_visibilities[i][s].weight.resize(d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        xy_pos.resize(d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        #pragma omp parallel for schedule(static, 1) num_threads(gridding) shared(g_weights) private(x, y, grid_pos_x, grid_pos_y, uvw, w)
                                        for (int z = 0; z < d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]; z++)
                                        {
                                                // First we save the original weights
                                                uvw = d[j].fields[f].visibilities[i][s].uvw[z];
                                                w = d[j].fields[f].visibilities[i][s].weight[z];
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
                                                x = round(grid_pos_x + N / 2);
                                                y = round(grid_pos_y + M / 2);

                                                // And we grid the weights
                                                if(x >= 0 && y >= 0 && x < M && y < M)
                                                {
                                                          #pragma omp critical
                                                          {

                                                                  g_weights[N * y + x] += w;

                                                          }

                                                }
                                        }

                                        for(int m=0; m<M; m++) {
                                                for(int n=N/2; n<N; n++) {
                                                        sum_gridded_weights_squared += g_weights[N * m + n] * g_weights[N * m + n];
                                                }
                                        }

                                        std::fill_n(g_weights.begin(), M*N, 0.0f);

                                }
                        }
                }
        }

        average_weights = sum_gridded_weights_squared / sum_original_weights;
        f_squared = (5.0f * powf(10.0f, -this->getRobustParam())) * (5.0f * powf(10.0f, -this->getRobustParam())) / average_weights;
        for(int j=0; j < d.size(); j++) {
                for(int f=0; f < d[j].data.nfields; f++) {
                        for(int i=0; i < d[j].data.total_frequencies; i++) {
                                for(int s=0; s < d[j].data.nstokes; s++) {
                                        #pragma omp parallel for schedule(static, 1) num_threads(gridding) shared(g_weights, xy_pos) private(x, y, grid_pos_x, grid_pos_y, uvw, w)
                                        for (int z = 0; z < d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]; z++)
                                        {
                                                // First we save the original weights
                                                uvw = d[j].fields[f].visibilities[i][s].uvw[z];
                                                w = d[j].fields[f].visibilities[i][s].weight[z];

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
                                                x = round(grid_pos_x + N / 2);
                                                y = round(grid_pos_y + M / 2);

                                                if(x >= 0 && y >= 0 && x < M && y < M)
                                                {
                                                  // And we grid the weights
                                                  #pragma omp critical
                                                  {
                                                          g_weights[N * y + x] += w;
                                                          xy_pos[z].x = x;
                                                          xy_pos[z].y = y;

                                                  }
                                                }else{
                                                          xy_pos[z].x = -1;
                                                          xy_pos[z].y = -1;
                                                }

                                        }

                                        #pragma omp parallel for schedule(static, 1) num_threads(gridding) shared(g_weights, xy_pos, f_squared) private(x, y)
                                        for (int z = 0; z < d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]; z++)
                                        {
                                                x = xy_pos[z].x;
                                                y = xy_pos[z].y;

                                                if(x >= 0 && y >= 0 && x < M && y < M)
                                                  d[j].fields[f].visibilities[i][s].weight[z] /= (1.0 + g_weights[N*y + x] * f_squared);
                                                else{
                                                  printf("Caigo aquí\n");
                                                  d[j].fields[f].visibilities[i][s].weight[z] = 0.0f;
                                                }
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
        return new BriggsWeightingScheme;
}

const std::string name = "Briggs";
const bool RegisteredBriggsWeighting = registerCreationFunction<WeightingScheme, std::string>(name, CreateWeightingScheme);
};
