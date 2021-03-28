#include "naturalweightingscheme.cuh"

NaturalWeightingScheme::NaturalWeightingScheme() : WeightingScheme(){};
NaturalWeightingScheme::NaturalWeightingScheme(int threads) : WeightingScheme(threads){};
NaturalWeightingScheme::NaturalWeightingScheme(int threads, UVTaper * uvtaper) : WeightingScheme(threads, uvtaper){};

void NaturalWeightingScheme::apply(std::vector<MSDataset>& d){
        std::cout << "Running Natural weighting scheme with " << this->threads << " threads" << std::endl;

        float w;
        double3 uvw;
        for(int j=0; j < d.size(); j++) {
                for(int f=0; f < d[j].data.nfields; f++) {
                        for(int i=0; i < d[j].data.total_frequencies; i++) {
                                for(int s=0; s < d[j].data.nstokes; s++) {
                                        d[j].fields[f].backup_visibilities[i][s].weight.resize(d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        #pragma omp parallel for schedule(static, 1) num_threads(this->threads) private(uvw, w)
                                        for (int z = 0; z < d[j].fields[f].numVisibilitiesPerFreqPerStoke[i][s]; z++)
                                        {
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

                                                if(NULL != this->uvtaper)
                                                        d[j].fields[f].visibilities[i][s].weight[z] *= this->uvtaper->getValue(uvw.x, uvw.y);

                                                if(this->modify_weights)
                                                        d[j].fields[f].backup_visibilities[i][s].weight[z] = d[j].fields[f].visibilities[i][s].weight[z];
                                                else
                                                        d[j].fields[f].backup_visibilities[i][s].weight[z] = w;
                                        }
                                }
                        }
                }
        }
};

namespace {
WeightingScheme* CreateWeightingScheme()
{
        return new NaturalWeightingScheme;
}

const std::string name = "Natural";
const bool RegisteredNaturalWeighting = registerCreationFunction<WeightingScheme, std::string>(name, CreateWeightingScheme);
};
