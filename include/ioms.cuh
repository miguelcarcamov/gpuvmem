#ifndef IOMS_CUH
#define IOMS_CUH
#include "framework.cuh"
#include "functions.cuh"

class IoMS : public Io {
 public:
  IoMS();
  IoMS(std::string input, std::string output, std::string path);
  IoMS(std::string input,
       std::string output,
       std::string path,
       std::string datacolumn_input,
       std::string datacolumn_output);
  IoMS(std::string input,
       std::string output,
       std::string path,
       std::string datacolumn_input,
       std::string datacolumn_output,
       float random_probability,
       int gridding,
       bool apply_noise_input,
       bool apply_noise_output,
       bool W_projection,
       bool store_model_vis_input);
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
  void setDataColumns(std::string datacolumn_input,
                      std::string datacolumn_output) override;
  void copy(char const* infile, char const* outfile) override;
  void copy() override;

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
