#ifndef IOMS_CUH
#define IOMS_CUH
#include "framework.cuh"
#include "functions.cuh"

class IoMS : public Io
{
public:
IoMS();
IoMS(std::string input, std::string output, std::string path);
IoMS(std::string input, std::string output, std::string path, std::string datacolumn_input, std::string datacolumn_output);
IoMS(std::string input, std::string output, std::string path, std::string datacolumn_input, std::string datacolumn_output, float random_probability, int gridding, bool apply_noise_input, bool apply_noise_output, bool W_projection, bool store_model_vis_input);
float getRandomProbability() override;
int getGridding() override;
bool getApplyNoiseInput() override;
bool getApplyNoiseOutput() override;
bool getWProjection() override;
bool getStoreModelVisInput() override;
std::string getDataColumnInput() override;
std::string getDataColumnOutput() override;
void setRandomProbability(float random_probability) override;
void setGridding(int gridding) override;
void setApplyNoiseInput(bool apply_noise_input) override;
void setApplyNoiseOutput(bool apply_noise_output) override;
void setNoise(bool input, bool output) override;
void setWProjection(bool wprojection) override;
void setStoreModelVisInput(bool store_model_vis_input) override;
void setDataColumnInput(std::string datacolumn_input) override;
void setDataColumnOutput(std::string datacolumn_output) override;
void setDataColumns(std::string datacolumn_input, std::string datacolumn_output) override;
void read(std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data) override;
void read(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_probability, int gridding) override;
void read(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data) override;
void readSpecificColumn(std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data) override;
void readSpecificColumn(std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, std::string data_column) override;
void readSpecificColumn(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_probability, int gridding) override;
void readSpecificColumn(char const *MS_name, std::string data_column, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_probability, int gridding) override;
void readSpecificColumn(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data) override;
void readSpecificColumn(char const *MS_name, std::string data_column, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data) override;
void copy(char const *infile, char const *outfile) override;
void copy() override;
void write(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool store_model_vis_input, bool noise, bool W_projection) override;
void write(char const *out_col, std::vector<Field>& fields, MSData data) override;
void write(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data) override;
void write(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, bool store_model_vis_input) override;
void write(char const *out_col, std::vector<Field>& fields, MSData data, bool store_model) override;
void writeSpecificColumn(char const *outfile, std::vector<Field>& fields, MSData data, float random_probability, bool store_model_vis_input, bool noise, bool W_projection) override;
void writeSpecificColumn(std::vector<Field>& fields, MSData data) override;
void writeSpecificColumn(char const *outfile, std::vector<Field>& fields, MSData data) override;
void writeSpecificColumn(char const *outfile, std::vector<Field>& fields, MSData data, bool store_model_vis_input) override;
void writeModelVisibilities(char const *outfile, std::vector<Field>& fields, MSData data) override;
void writeModelVisibilities(std::vector<Field>& fields, MSData data) override;
void writeResidualsAndModel(std::vector<Field>& fields, MSData data) override;
void writeResidualsAndModel(std::string input, std::string output, std::vector<Field>& fields, MSData data) override;

protected:
  float random_probability;
  int gridding;
  bool apply_noise_input;
  bool apply_noise_output;
  bool W_projection;
  bool store_model_vis_input;
  std::string datacolumn_input;
  std::string datacolumn_output;
};

#endif
