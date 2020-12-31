#include "ioms.cuh"
IoMS::IoMS() : Io()
{
        this->apply_noise_input = false;
        this->apply_noise_output = false;
        this->w_projection = false;
        this->random_prob_input = 1.0;
        this->random_prob_output = 1.0;
        this->gridding = 0;
};

IoMS::IoMS(std::string input_name, bool apply_noise_input, bool apply_noise_output, bool w_projection, float random_prob_input, float random_prob_output, int gridding) : Io(input_name)
{
        this->apply_noise_input = apply_noise_input;
        this->apply_noise_output = apply_noise_output;
        this->w_projection = w_projection;
        this->random_prob_input = random_prob_input;
        this->random_prob_output = random_prob_output;
        this->gridding = gridding;
};

void IoMS::IoreadMS(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding)
{
        readMS(MS_name, antennas, fields, data, noise, W_projection, random_prob, gridding);
};

void IoMS::IocopyMS(char const *infile, char const *outfile)
{
        MScopy(infile, outfile);
};

void IoMS::IowriteMS(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag)
{
        writeMS(outfile, out_col, fields, data, random_probability, sim, noise, W_projection, verbose_flag);
};

namespace {
Io* CreateIoMS()
{
        return new IoMS;
}
const std::string IoMSId = "IoMS";
const bool RegisteredIoMS = registerCreationFunction<Io, std::string>(IoMSId, CreateIoMS);
};
