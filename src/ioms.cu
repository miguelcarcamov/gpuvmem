#include "ioms.cuh"

IoMS::IoMS() : Io()
{
        this->random_probability = 1.0f;
        this->gridding = 0;
        this->apply_noise_input = false;
        this->apply_noise_output = false;
        this->W_projection = false;
        this->store_model_vis_input = false;
        this->datacolumn_input = "CORRECTED_DATA";
        this->datacolumn_output= "DATA";
};

IoMS::IoMS(std::string input, std::string output, std::string path) : Io(input, output, path)
{
        this->random_probability = 1.0f;
        this->gridding = 0;
        this->apply_noise_input = false;
        this->apply_noise_output = false;
        this->W_projection = false;
        this->store_model_vis_input = false;
        this->datacolumn_input = "CORRECTED_DATA";
        this->datacolumn_output= "DATA";
};

IoMS::IoMS(std::string input, std::string output, std::string path, std::string datacolumn_input, std::string datacolumn_output) : Io(input, output, path)
{
        this->random_probability = 1.0f;
        this->gridding = 0;
        this->apply_noise_input = false;
        this->apply_noise_output = false;
        this->W_projection = false;
        this->store_model_vis_input = false;
        this->datacolumn_input = datacolumn_input;
        this->datacolumn_output= datacolumn_output;
};


IoMS::IoMS(std::string input, std::string output, std::string path, std::string datacolumn_input, std::string datacolumn_output, float random_probability, int gridding, bool apply_noise_input, bool apply_noise_output, bool W_projection, bool store_model_vis_input) : Io(input, output, path)
{
        this->random_probability = random_probability;
        this->gridding = gridding;
        this->apply_noise_input = apply_noise_input;
        this->apply_noise_output = apply_noise_output;
        this->W_projection = W_projection;
        this->store_model_vis_input = store_model_vis_input;
        this->datacolumn_input = datacolumn_input;
        this->datacolumn_output= datacolumn_output;
};

float IoMS::getRandomProbability(){
        return this->random_probability;
};

int IoMS::getGridding(){
        return this->gridding;
};

bool IoMS::getApplyNoiseInput(){
        return this->apply_noise_input;
};

bool IoMS::getApplyNoiseOutput(){
        return this->apply_noise_output;
};

bool IoMS::getWProjection(){
        return this->W_projection;
};

bool IoMS::getStoreModelVisInput(){
        return this->store_model_vis_input;
};

std::string IoMS::getDataColumnInput(){
        return this->datacolumn_input;
};

std::string IoMS::getDataColumnOutput(){
        return this->datacolumn_output;
};

void IoMS::setRandomProbability(float random_probability){
        this->random_probability = random_probability;
};

void IoMS::setGridding(int gridding){
        this->gridding = gridding;
};

void IoMS::setApplyNoiseInput(bool apply_noise_input){
        this->apply_noise_input = apply_noise_input;
};

void IoMS::setApplyNoiseOutput(bool apply_noise_output){
        this->apply_noise_output = apply_noise_output;
};

void IoMS::setNoise(bool input, bool output){
        this->apply_noise_input = input;
        this->apply_noise_output = output;
};

void IoMS::setWProjection(bool wprojection){
        this->W_projection = wprojection;
};

void IoMS::setStoreModelVisInput(bool store_model_vis_input){
        this->store_model_vis_input = store_model_vis_input;
};

void IoMS::setDataColumnInput(std::string datacolumn_input){
        this->datacolumn_input = datacolumn_input;

};
void IoMS::setDataColumnOutput(std::string datacolumn_output){
        this->datacolumn_output= datacolumn_output;
};

void IoMS::setDataColumns(std::string datacolumn_input, std::string datacolumn_output){
        this->datacolumn_input = datacolumn_input;
        this->datacolumn_output= datacolumn_output;
};

void IoMS::read(std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data)
{
        readMS(this->input.c_str(), antennas, fields, data, this->apply_noise_input, this->W_projection, this->random_probability, this->gridding);
};

void IoMS::read(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_probability, int gridding)
{
        readMS(MS_name, antennas, fields, data, noise, W_projection, random_probability, gridding);
};

void IoMS::read(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data)
{
        readMS(MS_name, antennas, fields, data, this->apply_noise_input, this->W_projection, this->random_probability, this->gridding);
};

void IoMS::readSpecificColumn(std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data)
{
        readMS(this->input.c_str(), this->datacolumn_input, antennas, fields, data, this->apply_noise_input, this->W_projection, this->random_probability, this->gridding);
};

void IoMS::readSpecificColumn(std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, std::string data_column)
{
        readMS(this->input.c_str(), data_column, antennas, fields, data, this->apply_noise_input, this->W_projection, this->random_probability, this->gridding);
};

void IoMS::readSpecificColumn(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_probability, int gridding)
{
        readMS(MS_name, this->datacolumn_input, antennas, fields, data, noise, W_projection, random_probability, gridding);
};

void IoMS::readSpecificColumn(char const *MS_name, std::string data_column, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_probability, int gridding)
{
        readMS(MS_name, data_column, antennas, fields, data, noise, W_projection, random_probability, gridding);
};

void IoMS::readSpecificColumn(char const *MS_name, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data)
{
        readMS(MS_name, this->datacolumn_input, antennas, fields, data, this->apply_noise_input, this->W_projection, this->random_probability, this->gridding);
};

void IoMS::readSpecificColumn(char const *MS_name, std::string data_column, std::vector<MSAntenna>& antennas, std::vector<Field>& fields, MSData *data)
{
        readMS(MS_name, data_column, antennas, fields, data, this->apply_noise_input, this->W_projection, this->random_probability, this->gridding);
};

void IoMS::copy(char const *infile, char const *outfile)
{
        MScopy(infile, outfile);
};

void IoMS::copy()
{
        MScopy(this->input.c_str(), this->output.c_str());
};

void IoMS::write(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool store_model_vis_input, bool noise, bool W_projection)
{
        writeMS(outfile, out_col, fields, data, random_probability, store_model_vis_input, noise, W_projection);
};

void IoMS::write(char const *out_col, std::vector<Field>& fields, MSData data)
{
        writeMS(this->output.c_str(), out_col, fields, data, this->random_probability, this->store_model_vis_input, this->apply_noise_output, this->W_projection);
};

void IoMS::write(char const *out_col, std::vector<Field>& fields, MSData data, bool store_model)
{
        writeMS(this->output.c_str(), out_col, fields, data, this->random_probability, store_model, this->apply_noise_output, this->W_projection);
};

void IoMS::write(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data)
{
        writeMS(outfile, out_col, fields, data, this->random_probability, this->store_model_vis_input, this->apply_noise_output, this->W_projection);
};

void IoMS::write(char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, bool store_model_vis_input)
{
        writeMS(outfile, out_col, fields, data, this->random_probability, store_model_vis_input, this->apply_noise_output, this->W_projection);
};

void IoMS::writeSpecificColumn(char const *outfile, std::vector<Field>& fields, MSData data, float random_probability, bool store_model_vis_input, bool noise, bool W_projection)
{
        writeMS(outfile, this->datacolumn_output.c_str(), fields, data, random_probability, store_model_vis_input, noise, W_projection);
};

void IoMS::writeSpecificColumn(std::vector<Field>& fields, MSData data)
{
        writeMS(this->output.c_str(), this->datacolumn_output.c_str(), fields, data, this->random_probability, this->store_model_vis_input, this->apply_noise_output, this->W_projection);
};

void IoMS::writeSpecificColumn(char const *outfile, std::vector<Field>& fields, MSData data)
{
        writeMS(outfile, this->datacolumn_output.c_str(), fields, data, this->random_probability, this->store_model_vis_input, this->apply_noise_output, this->W_projection);
};

void IoMS::writeSpecificColumn(char const *outfile, std::vector<Field>& fields, MSData data, bool store_model_vis_input)
{
        writeMS(outfile, this->datacolumn_output.c_str(), fields, data, this->random_probability, store_model_vis_input, this->apply_noise_output, this->W_projection);
};

void IoMS::writeModelVisibilities(char const *outfile, std::vector<Field>& fields, MSData data)
{
        writeMS(outfile, "MODEL_DATA", fields, data, this->random_probability, true, this->apply_noise_output, this->W_projection);
};

void IoMS::writeModelVisibilities(std::vector<Field>& fields, MSData data)
{
        writeMS(this->output.c_str(), "MODEL_DATA", fields, data, this->random_probability, true, this->apply_noise_output, this->W_projection);
};

void IoMS::writeResidualsAndModel(std::vector<Field>& fields, MSData data)
{
    if(this->store_model_vis_input){
      write("DATA", fields, data, false);
      writeModelVisibilities(this->input.c_str(), fields, data);
    }else{    
      write("DATA", fields, data, false);
      writeModelVisibilities(fields, data);
    }
};

void IoMS::writeResidualsAndModel(std::string input, std::string output, std::vector<Field>& fields, MSData data)
{
    if(this->store_model_vis_input){
      write(output.c_str(), "DATA", fields, data, false);
      writeModelVisibilities(input.c_str(), fields, data);
    }else{
      write(output.c_str(), "DATA", fields, data, false);
      writeModelVisibilities(output.c_str(), fields, data);
    }
};

namespace {
Io* CreateIoMS()
{
        return new IoMS;
}
const std::string IoMSId = "IoMS";
const bool RegisteredIoMS = registerCreationFunction<Io, std::string>(IoMSId, CreateIoMS);
};
