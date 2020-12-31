#ifndef IOMS_CUH
#define IOMS_CUH
#include "framework.cuh"
#include "functions.cuh"

class IoMS : public Io
{
public:
IoMS();
IoMS(std::string input_name, bool apply_noise_input, bool apply_noise_output, bool w_projection, float random_prob_input, float random_prob_output, int gridding);
void IoreadMS(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding);
void IocopyMS(char const *infile, char const *outfile);
void IowriteMS(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag);
private:
bool apply_noise_input;
bool apply_noise_output;
bool w_projection;
float random_prob_input;
float random_prob_output;
int gridding;
};

#endif
