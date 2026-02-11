#include "ioms.cuh"

IoMS::IoMS() : Io() {
  this->random_probability = 1.0f;
  this->gridding = 0;
  this->apply_noise_input = false;
  this->apply_noise_output = false;
  this->W_projection = false;
  this->store_model_vis_input = false;
  this->datacolumn_input = "CORRECTED_DATA";
  this->datacolumn_output = "DATA";
};

IoMS::IoMS(std::string input, std::string output, std::string path)
    : Io(input, output, path) {
  this->random_probability = 1.0f;
  this->gridding = 0;
  this->apply_noise_input = false;
  this->apply_noise_output = false;
  this->W_projection = false;
  this->store_model_vis_input = false;
  this->datacolumn_input = "CORRECTED_DATA";
  this->datacolumn_output = "DATA";
};

IoMS::IoMS(std::string input,
           std::string output,
           std::string path,
           std::string datacolumn_input,
           std::string datacolumn_output)
    : Io(input, output, path) {
  this->random_probability = 1.0f;
  this->gridding = 0;
  this->apply_noise_input = false;
  this->apply_noise_output = false;
  this->W_projection = false;
  this->store_model_vis_input = false;
  this->datacolumn_input = datacolumn_input;
  this->datacolumn_output = datacolumn_output;
};

IoMS::IoMS(std::string input,
           std::string output,
           std::string path,
           std::string datacolumn_input,
           std::string datacolumn_output,
           float random_probability,
           int gridding,
           bool apply_noise_input,
           bool apply_noise_output,
           bool W_projection,
           bool store_model_vis_input)
    : Io(input, output, path) {
  this->random_probability = random_probability;
  this->gridding = gridding;
  this->apply_noise_input = apply_noise_input;
  this->apply_noise_output = apply_noise_output;
  this->W_projection = W_projection;
  this->store_model_vis_input = store_model_vis_input;
  this->datacolumn_input = datacolumn_input;
  this->datacolumn_output = datacolumn_output;
};

float IoMS::getRandomProbability() {
  return this->random_probability;
};

int IoMS::getGridding() {
  return this->gridding;
};

bool IoMS::getApplyNoiseInput() {
  return this->apply_noise_input;
};

bool IoMS::getApplyNoiseOutput() {
  return this->apply_noise_output;
};

bool IoMS::getWProjection() {
  return this->W_projection;
};

bool IoMS::getStoreModelVisInput() {
  return this->store_model_vis_input;
};

std::string IoMS::getDataColumnInput() {
  return this->datacolumn_input;
};

std::string IoMS::getDataColumnOutput() {
  return this->datacolumn_output;
};

void IoMS::setRandomProbability(float random_probability) {
  this->random_probability = random_probability;
};

void IoMS::setGridding(int gridding) {
  this->gridding = gridding;
};

void IoMS::setApplyNoiseInput(bool apply_noise_input) {
  this->apply_noise_input = apply_noise_input;
};

void IoMS::setApplyNoiseOutput(bool apply_noise_output) {
  this->apply_noise_output = apply_noise_output;
};

void IoMS::setNoise(bool input, bool output) {
  this->apply_noise_input = input;
  this->apply_noise_output = output;
};

void IoMS::setWProjection(bool wprojection) {
  this->W_projection = wprojection;
};

void IoMS::setStoreModelVisInput(bool store_model_vis_input) {
  this->store_model_vis_input = store_model_vis_input;
};

void IoMS::setDataColumnInput(std::string datacolumn_input) {
  this->datacolumn_input = datacolumn_input;
};
void IoMS::setDataColumnOutput(std::string datacolumn_output) {
  this->datacolumn_output = datacolumn_output;
};

void IoMS::setDataColumns(std::string datacolumn_input,
                          std::string datacolumn_output) {
  this->datacolumn_input = datacolumn_input;
  this->datacolumn_output = datacolumn_output;
};

void IoMS::copy(char const* infile, char const* outfile) {
  MScopy(infile, outfile);
};

void IoMS::copy() {
  MScopy(this->input.c_str(), this->output.c_str());
};

namespace {
Io* CreateIoMS() {
  return new IoMS;
}
const std::string IoMSId = "IoMS";
const bool RegisteredIoMS =
    registerCreationFunction<Io, std::string>(IoMSId, CreateIoMS);
};  // namespace
