#include "ioms.cuh"

extern long M,N;
extern float fg_scale;

IoMS::IoMS()
{
        this->original_FITS_name = "";
        this->fits_path = "";
};

IoMS::IoMS(char *original_FITS_name, char *fits_path)
{
        this->original_FITS_name = original_FITS_name;
        this->fits_path = fits_path;
};

headerValues IoMS::IoreadCanvas(char *canvas_name)
{
        this->original_FITS_name = canvas_name;
        return readFITSHeader(canvas_name);
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

void IoMS::IoPrintImage(float *I, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU)
{
        OCopyFITS(I, this->original_FITS_name, path, name_image, units, iteration, index, fg_scale, M, N, isInGPU);
}

void IoMS::IoPrintImage(float *I, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU)
{
        OCopyFITS(I, this->original_FITS_name, this->fits_path, name_image, units, iteration, index, fg_scale, M, N, isInGPU);
}

void IoMS::IoPrintImageIteration(float *I, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N, bool isInGPU)
{
        size_t needed;
        char *full_name;

        needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "%s_%d.fits", name_image, iteration);

        OCopyFITS(I, this->original_FITS_name, this->fits_path, full_name, units, iteration, index, fg_scale, M, N, isInGPU);
        free(full_name);
}

void IoMS::IoPrintOptImageIteration(float *I, char *name_image, char *units, int index, int iteration, bool isInGPU)
{
        size_t needed;
        char *full_name;

        needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "%s_%d.fits", name_image, iteration);

        std::string unit_str(units);
        if(unit_str == "JY/PIXEL")
                OCopyFITS(I, this->original_FITS_name, this->fits_path, full_name, units, iteration, index, fg_scale, M, N, isInGPU);
        else
                OCopyFITS(I, this->original_FITS_name, this->fits_path, full_name, units, iteration, index, 1.0, M, N, isInGPU);
        free(full_name);
}

void IoMS::IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option, bool isInGPU)
{
        OCopyFITSCufftComplex(I, this->original_FITS_name, this->fits_path, out_image, iteration, fg_scale, M, N, option, isInGPU);
}

void IoMS::IocloseCanvas(fitsfile *canvas)
{
        closeFITS(canvas);
};

namespace {
Io* CreateIoMS()
{
        return new IoMS;
}
const std::string IoMSId = "IoMS";
const bool RegisteredIoMS = registerCreationFunction<Io, std::string>(IoMSId, CreateIoMS);
};
