#include "ioms.cuh"

extern long M,N;
extern float fg_scale;
extern char* mempath;
extern fitsfile *mod_in;
extern int iter;

canvasVariables IoMS::IoreadCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag)
{
        return readCanvas(canvas_name, canvas, b_noise_aux, status_canvas, verbose_flag);
};

void IoMS::IoreadMS(char const *MS_name, std::vector<Field>& fields, MSData *data, bool noise, bool W_projection, float random_prob, int gridding)
{
        readMS(MS_name, fields, data, noise, W_projection, random_prob, gridding);
};

void IoMS::IocopyMS(char const *infile, char const *outfile)
{
        MScopy(infile, outfile);
};

void IoMS::IowriteMS(char const *infile, char const *outfile, char const *out_col, std::vector<Field>& fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag)
{
        writeMS(infile, outfile, out_col, fields, data, random_probability, sim, noise, W_projection, verbose_flag);
};

void IoMS::IoPrintImage(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N)
{
        OFITS(I, canvas, path, name_image, units, iteration, index, fg_scale, M, N);
}

void IoMS::IoPrintImageIteration(float *I, fitsfile *canvas, char *path, char const *name_image, char *units, int iteration, int index, float fg_scale, long M, long N)
{
        size_t needed;
        char *full_name;

        needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "%s_%d.fits", name_image, iteration);

        OFITS(I, canvas, path, full_name, units, iteration, index, fg_scale, M, N);
        free(full_name);
}

void IoMS::IoPrintMEMImageIteration(float *I, char *name_image, char *units, int index)
{
        size_t needed;
        char *full_name;

        needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iter) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "%s_%d.fits", name_image, iter);

        OFITS(I, mod_in, mempath, full_name, units, iter, index, fg_scale, M, N);
        free(full_name);
}

void IoMS::IoPrintcuFFTComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option)
{
        fitsOutputCufftComplex(I, canvas, out_image, mempath, iteration, fg_scale, M, N, option);
}

void IoMS::IocloseCanvas(fitsfile *canvas)
{
        closeCanvas(canvas);
};

namespace {
Io* CreateIoMS()
{
        return new IoMS;
}
const int IoMSId = 0;
const bool RegisteredIoMS = Singleton<IoFactory>::Instance().RegisterIo(IoMSId, CreateIoMS);
};
