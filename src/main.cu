/* -------------------------------------------------------------------------
   Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
   Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

   This program includes Numerical Recipes (NR) based routines whose
   copyright is held by the NR authors. If NR routines are included,
   you are required to comply with the licensing set forth there.

   Part of the program also relies on an an ANSI C library for multi-stream
   random number generation from the related Prentice-Hall textbook
   Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
   for more information please contact leemis@math.wm.edu

   For the original parts of this code, the following license applies:

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------
 */

#include "functions.cuh"
#include "directioncosines.cuh"

long M, N, numVisibilities;
int iter=0;

cufftHandle plan1GPU;

cufftComplex *device_V, *device_Inu;

float2 *device_dphi, *device_2I;
float *device_mask, *device_dS, *device_dS_alpha, *device_chi2, *device_dchi2, *device_S, *device_S_alpha, beam_noise, beam_bpa, nu_0, *device_noise_image, *device_weight_image;
float b_noise_aux, noise_cut, MINPIX, minpix, lambda, ftol, random_probability, beam_bmaj, beam_bmin;
float noise_jypix, fg_scale, final_chi2, final_H, antenna_diameter, pb_factor, pb_cutoff, alpha_start, eta, epsilon, *host_noise_image;

dim3 threadsPerBlockNN;
dim3 numBlocksNN;

int threadsVectorReduceNN, blocksVectorReduceNN, crpix1, crpix2, nopositivity = 0, verbose_flag = 0, clip_flag = 0, apply_noise = 0, print_images = 0, checkpoint = 0, adaptive = 0, use_mask=0, spec_idx=0, gridding, it_maximum, status_mod_in;
int num_gpus, multigpu, firstgpu, selected, t_telescope, reg_term, valid_pixels;
int2 *pixels;
char *output, *mempath, *out_image;

double ra, dec, DELTAX, DELTAY, deltau, deltav;

freqData data;

fitsfile *mod_in;
fitsfile *mod_in_alpha;

Field *fields;

VariablesPerField *vars_per_field;

varsPerGPU *vars_gpu;

char *checkp;

double2 *host_M_k;
double2 *host_Q_k;

float2 *temp;
double2 *M_k_out, *Q_k_out;

FILE *outfile;
FILE *outfile_its;
FILE *shutdown_file;

int position_in_file;

int accepted_afterburndown;

int real_iterations;

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
  #ifdef _WIN32
        return (bool)(pProp->tccDriver ? true : false);
  #else
        return (bool)(pProp->major >= 2);
  #endif
}

inline bool IsAppBuiltAs64()
{
  #if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
        return 1;
  #else
        return 0;
  #endif
}

__host__ int main(int argc, char **argv) {
        clock_t t;
        double start, end;
        ////CHECK FOR AVAILABLE GPUs
        cudaGetDeviceCount(&num_gpus);

        printf("gpuvmem Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus, Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl\n");
        printf("This program comes with ABSOLUTELY NO WARRANTY; for details use option -w\n");
        printf("This is free software, and you are welcome to redistribute it under certain conditions; use option -c for details.\n\n\n");


        if(num_gpus < 1) {
                printf("No CUDA capable devices were detected\n");
                return 1;
        }

        if (!IsAppBuiltAs64()) {
                printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target. Test is being waived.\n", argv[0]);
                exit(EXIT_SUCCESS);
        }


        Vars variables = getOptions(argc, argv);
        char *msinput = variables.input;
        char *msoutput = variables.output;
        char *inputdat = variables.inputdat;
        char *modinput = variables.modin;
        char *alpha_name = variables.alpha_name;
        alpha_start = variables.alpha_value;
        out_image = variables.output_image;
        selected = variables.select;
        mempath = variables.path;
        it_maximum = variables.it_max;
        int total_visibilities = 0;
        b_noise_aux = variables.noise;
        lambda = variables.lambda;
        minpix = variables.minpix;
        noise_cut = variables.noise_cut;
        random_probability = variables.randoms;
        reg_term = variables.reg_term;
        nu_0 = variables.nu_0;
        eta = variables.eta;
        epsilon = variables.epsilon;
        gridding = variables.gridding;

        multigpu = 0;
        firstgpu = -1;
        //checkp = "checkpoint/";
        size_t checkpoint_many;
        checkpoint_many = snprintf(NULL, 0, "%scheckpoint/", mempath) + 1;
        checkp = (char*)malloc(checkpoint_many*sizeof(char));


        struct stat st = {0};

        if(stat(mempath, &st) == -1) mkdir(mempath,0700);

        //CONCAT MEMPATH WITH CHECKPOINT

        snprintf(checkp, checkpoint_many*sizeof(char), "%scheckpoint/", mempath, 0);

        struct stat st2 = {0};

        if(stat(checkp, &st2) == -1) mkdir(checkp,0700);

        if(verbose_flag) {
                printf("Number of host CPUs:\t%d\n", omp_get_num_procs());
                printf("Number of CUDA devices:\t%d\n", num_gpus);


                for(int i = 0; i < num_gpus; i++) {
                        cudaDeviceProp dprop;
                        cudaGetDeviceProperties(&dprop, i);

                        printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, dprop.name, (IsGPUCapableP2P(&dprop) ? "IS " : "NOT"));

                        //printf("   %d: %s\n", i, dprop.name);
                }
                printf("---------------------------\n");
        }

        if(selected > num_gpus || selected < 0) {
                printf("ERROR. THE SELECTED GPU DOESN'T EXIST\n");
                exit(-1);
        }

        readInputDat(inputdat);
        init_beam(t_telescope);
        if(verbose_flag) {
                printf("Counting data for memory allocation\n");
        }

        canvasVariables canvas_vars = readCanvas(modinput, mod_in, b_noise_aux, status_mod_in, verbose_flag);

        M = canvas_vars.M;
        N = canvas_vars.N;
        DELTAX = canvas_vars.DELTAX;
        DELTAY = canvas_vars.DELTAY;
        ra = canvas_vars.ra;
        dec = canvas_vars.dec;
        crpix1 = canvas_vars.crpix1;
        crpix2 = canvas_vars.crpix2;
        beam_bmaj = canvas_vars.beam_bmaj;
        beam_bmin = canvas_vars.beam_bmin;
        beam_bpa = (canvas_vars.beam_bpa + 90.0) * RPDEG_D;
        beam_noise = canvas_vars.beam_noise;

        data = countVisibilities(msinput, fields);

        vars_per_field = (VariablesPerField*)malloc(data.nfields*sizeof(VariablesPerField));

        if(verbose_flag) {
                printf("Number of fields = %d\n", data.nfields);
                printf("Number of frequencies = %d\n", data.total_frequencies);
        }

        if(strcmp(variables.multigpu, "NULL")!=0) {
                //Counts number of gpus to use
                char *pt;
                pt = strtok(variables.multigpu,",");

                while(pt!=NULL) {
                        if(multigpu==0) {
                                firstgpu = atoi(pt);
                        }
                        multigpu++;
                        pt = strtok (NULL, ",");
                }
        }else{
                multigpu = 0;
        }

        if(multigpu < 0 || multigpu > num_gpus) {
                printf("ERROR. NUMBER OF GPUS CANNOT BE NEGATIVE OR GREATER THAN THE NUMBER OF GPUS\n");
                exit(-1);
        }else{
                if(multigpu == 0) {
                        num_gpus = 1;
                }else{
                        if(data.total_frequencies == 1) {
                                printf("ONLY ONE FREQUENCY. CHANGING NUMBER OF GPUS TO 1\n");
                                num_gpus = 1;
                        }else{
                                num_gpus = multigpu;
                                omp_set_num_threads(num_gpus);
                        }
                }
        }

        //printf("number of FINAL host CPUs:\t%d\n", omp_get_num_procs());
        if(verbose_flag) {
                printf("Number of CUDA devices and threads: \t%d\n", num_gpus);
        }

        //Check peer access if there is more than 1 GPU
        if(num_gpus > 1) {
                for(int i=firstgpu + 1; i< firstgpu + num_gpus; i++) {
                        cudaDeviceProp dprop0, dpropX;
                        cudaGetDeviceProperties(&dprop0, firstgpu);
                        cudaGetDeviceProperties(&dpropX, i);
                        int canAccessPeer0_x, canAccessPeerx_0;
                        cudaDeviceCanAccessPeer(&canAccessPeer0_x, firstgpu, i);
                        cudaDeviceCanAccessPeer(&canAccessPeerx_0, i, firstgpu);
                        if(verbose_flag) {
                                printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", dprop0.name, firstgpu, dpropX.name, i, canAccessPeer0_x ? "Yes" : "No");
                                printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", dpropX.name, i, dprop0.name, firstgpu, canAccessPeerx_0 ? "Yes" : "No");
                        }
                        if(canAccessPeer0_x == 0 || canAccessPeerx_0 == 0) {
                                printf("Two or more SM 2.0 class GPUs are required for %s to run.\n", argv[0]);
                                printf("Support for UVA requires a GPU with SM 2.0 capabilities.\n");
                                printf("Peer to Peer access is not available between GPU%d <-> GPU%d, waiving test.\n", 0, i);
                                exit(EXIT_SUCCESS);
                        }else{
                                cudaSetDevice(firstgpu);
                                if(verbose_flag) {
                                        printf("Granting access from %d to %d...\n",firstgpu, i);
                                }
                                cudaDeviceEnablePeerAccess(i,0);
                                cudaSetDevice(i);
                                if(verbose_flag) {
                                        printf("Granting access from %d to %d...\n", i, firstgpu);
                                }
                                cudaDeviceEnablePeerAccess(firstgpu,0);
                                if(verbose_flag) {
                                        printf("Checking GPU %d and GPU %d for UVA capabilities...\n", firstgpu, i);
                                }
                                const bool has_uva = (dprop0.unifiedAddressing && dpropX.unifiedAddressing);
                                if(verbose_flag) {
                                        printf("> %s (GPU%d) supports UVA: %s\n", dprop0.name, firstgpu, (dprop0.unifiedAddressing ? "Yes" : "No"));
                                        printf("> %s (GPU%d) supports UVA: %s\n", dpropX.name, i, (dpropX.unifiedAddressing ? "Yes" : "No"));
                                }
                                if (has_uva) {
                                        if(verbose_flag) {
                                                printf("Both GPUs can support UVA, enabling...\n");
                                        }
                                }
                                else{
                                        printf("At least one of the two GPUs does NOT support UVA, waiving test.\n");
                                        exit(EXIT_SUCCESS);
                                }
                        }
                }
                vars_gpu = (varsPerGPU*)malloc(num_gpus*sizeof(varsPerGPU));
        }

        for(int f=0; f<data.nfields; f++) {
                fields[f].visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
                fields[f].gridded_visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
                fields[f].device_visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
        }

        //ALLOCATE MEMORY AND GET TOTAL NUMBER OF VISIBILITIES
        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i < data.total_frequencies; i++) {
                        fields[f].visibilities[i].stokes = (int*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(int));
                        fields[f].visibilities[i].uvw = (double3*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(double3));
                        fields[f].visibilities[i].weight = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                        fields[f].visibilities[i].Vo = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
                        if(gridding) {
                                fields[f].gridded_visibilities[i].uvw = (double3*)malloc(M*N*sizeof(double3));
                                fields[f].gridded_visibilities[i].weight = (float*)malloc(M*N*sizeof(float));
                                fields[f].gridded_visibilities[i].Vo = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));

                                memset(fields[f].gridded_visibilities[i].uvw, 0, M*N*sizeof(double3));
                                memset(fields[f].gridded_visibilities[i].weight, 0, M*N*sizeof(float));
                                memset(fields[f].gridded_visibilities[i].Vo, 0, M*N*sizeof(cufftComplex));
                        }else{
                                fields[f].visibilities[i].Vm = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
                        }
                        fields[f].visibilities[i].Vm = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
                }
        }



        if(verbose_flag) {
                printf("Reading visibilities and FITS input files...\n");
        }

        if(apply_noise && random_probability < 1.0) {
                readMCNoiseSubsampledMS(msinput, fields, data, random_probability);
        }else if(random_probability < 1.0) {
                readSubsampledMS(msinput, fields, data, random_probability);
        }else if(apply_noise) {
                readMSMCNoise(msinput, fields, data);
        }else{
                readMS(msinput, fields, data);
        }

        if(verbose_flag) {
                printf("MS File Successfully Read\n");
                if(beam_noise == -1) {
                        printf("Beam noise wasn't provided by the user... Calculating...\n");
                }
                printf("Calculating weights sum\n");
        }

        float deltax = RPDEG_D*DELTAX; //radians
        float deltay = RPDEG_D*DELTAY; //radians
        deltau = 1.0 / (M * deltax);
        deltav = 1.0 / (N * deltay);
        if(gridding) {
                omp_set_num_threads(gridding);
                do_gridding(fields, &data, deltau, deltav, M, N, variables.robust_param);
                omp_set_num_threads(num_gpus);
        }

        float analytical_noise = calculateNoise(fields, data, &total_visibilities, variables.blockSizeV, gridding);

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].uvw, sizeof(double3)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vr, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                cudaSetDevice((i%num_gpus) + firstgpu);
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].uvw, sizeof(double3)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vr, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                        }
                }
        }


        if(num_gpus == 1) {
                cudaSetDevice(selected);
                gpuErrchk(cudaMalloc((void**)&device_dchi2, sizeof(float)*M*N));
                gpuErrchk(cudaMemset(device_dchi2, 0, sizeof(float)*M*N));

                gpuErrchk(cudaMalloc(&device_chi2, sizeof(float)*data.max_number_visibilities_in_channel));
                gpuErrchk(cudaMemset(device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel));

                for(int f=0; f<data.nfields; f++) {
                        gpuErrchk(cudaMalloc((void**)&vars_per_field[f].atten_image, sizeof(cufftComplex)*M*N));
                        gpuErrchk(cudaMemset(vars_per_field[f].atten_image, 0, sizeof(cufftComplex)*M*N));
                        for(int i=0; i < data.total_frequencies; i++) {

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].uvw, fields[f].visibilities[i].uvw, sizeof(double3)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].weight, fields[f].visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vr, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vm, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));

                        }
                }
        }else{

                for(int g=0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_dchi2, sizeof(float)*M*N));
                        gpuErrchk(cudaMemset(vars_gpu[g].device_dchi2, 0, sizeof(float)*M*N));

                        gpuErrchk(cudaMalloc(&vars_gpu[g].device_chi2, sizeof(float)*data.max_number_visibilities_in_channel));
                        gpuErrchk(cudaMemset(vars_gpu[g].device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel));
                }

                for(int f=0; f<data.nfields; f++) {
                        cudaSetDevice(firstgpu);
                        gpuErrchk(cudaMalloc((void**)&vars_per_field[f].atten_image, sizeof(cufftComplex)*M*N));
                        gpuErrchk(cudaMemset(vars_per_field[f].atten_image, 0, sizeof(cufftComplex)*M*N));
                        for(int i=0; i < data.total_frequencies; i++) {
                                cudaSetDevice((i%num_gpus) + firstgpu);

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].uvw, fields[f].visibilities[i].uvw, sizeof(double3)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].weight, fields[f].visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vr, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vm, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                        }
                }
        }

        //Declaring block size and number of blocks for Image
        dim3 threads(variables.blockSizeX, variables.blockSizeY);
        dim3 blocks(M/threads.x, N/threads.y);
        threadsPerBlockNN = threads;
        numBlocksNN = blocks;


        noise_jypix = analytical_noise / (PI_D * beam_bmaj * beam_bmin / (4 * log(2) ));

        if(lambda == 0.0) {
                MINPIX = 0.0;
        }else{
                MINPIX = minpix;
        }

        /////////////////////////////////////////////////////CALCULATE DIRECTION COSINES/////////////////////////////////////////////////
        double raimage = ra * RPDEG_D;
        double decimage = dec * RPDEG_D;

        if(verbose_flag) {
                printf("FITS: Ra: %.16e (rad), dec: %.16e (rad)\n", raimage, decimage);
                printf("FITS: Center pix: (%lf,%lf)\n", crpix1-1, crpix2-1);
        }

        double lobs, mobs, lphs, mphs;
        double dcosines_l_pix_ref, dcosines_m_pix_ref, dcosines_l_pix_phs, dcosines_m_pix_phs;

        for(int f=0; f<data.nfields; f++) {

                direccos(fields[f].ref_ra, fields[f].ref_dec, raimage, decimage, &lobs,  &mobs);
                direccos(fields[f].phs_ra, fields[f].phs_dec, raimage, decimage, &lphs,  &mphs);

                dcosines_l_pix_ref = lobs/ -deltax; // Radians to pixels
                dcosines_m_pix_ref = mobs/fabs(deltay); // Radians to pixels

                dcosines_l_pix_phs = lphs/ -deltax; // Radians to pixels
                dcosines_m_pix_phs = mphs/fabs(deltay); // Radians to pixels

                if(verbose_flag)
                {
                        printf("Ref: l (pix): %e, m (pix): %e\n", dcosines_l_pix_ref, dcosines_m_pix_ref);
                        printf("Phase: l (pix): %e, m (pix): %e\n", dcosines_l_pix_phs, dcosines_m_pix_phs);

                }


                fields[f].ref_xobs = (crpix1 - 1.0f) + dcosines_l_pix_ref;// + 6.0f;
                fields[f].ref_yobs = (crpix2 - 1.0f) + dcosines_m_pix_ref;// - 7.0f;

                fields[f].phs_xobs = (crpix1 - 1.0f) + dcosines_l_pix_phs;// + 5.0f;
                fields[f].phs_yobs = (crpix2 - 1.0f) + dcosines_m_pix_phs;// - 7.0f;


                if(verbose_flag) {
                        printf("Ref: Field %d - Ra: %.16e (rad), dec: %.16e (rad), x0: %f (pix), y0: %f (pix)\n", f, fields[f].ref_ra, fields[f].ref_dec,
                               fields[f].ref_xobs, fields[f].ref_yobs);
                        printf("Phase: Field %d - Ra: %.16e (rad), dec: %.16e (rad), x0: %f (pix), y0: %f (pix)\n", f, fields[f].phs_ra, fields[f].phs_dec,
                               fields[f].phs_xobs, fields[f].phs_yobs);
                }

                if(fields[f].ref_xobs < 0 || fields[f].ref_xobs >= M || fields[f].ref_xobs < 0 || fields[f].ref_yobs >= N) {
                        printf("Pointing reference center (%f,%f) is outside the range of the image\n", fields[f].ref_xobs, fields[f].ref_yobs);
                        goToError();
                }

                if(fields[f].phs_xobs < 0 || fields[f].phs_xobs >= M || fields[f].phs_xobs < 0 || fields[f].phs_yobs >= N) {
                        printf("Pointing phase center (%f,%f) is outside the range of the image\n", fields[f].phs_xobs, fields[f].phs_yobs);
                        goToError();
                }
        }
        ////////////////////////////////////////////////////////MAKE STARTING IMAGE////////////////////////////////////////////////////////
        float2 *host_2I = (float2*)malloc(M*N*sizeof(float2));

        float *input_Inu_0;
        float *host_mask;
        float *input_alpha;


        size_t needed_I_nu_0;
        size_t needed_alpha;

        open_read_fits<float>(&input_Inu_0, modinput, M*N, TFLOAT);
        open_read_fits<float>(&input_alpha, alpha_name, M*N, TFLOAT);

        if(use_mask)
                host_mask = (float*)malloc(M*N*sizeof(float));

        for(int i=0; i<M; i++) {
                for(int j=0; j<N; j++) {
                        host_2I[N*i+j].x = input_Inu_0[N*i+j]; // I_nu
                        host_2I[N*i+j].y = input_alpha[N*i+j];
                        if(use_mask)
                                host_mask[N*i+j] = input_Inu_0[N*i+j];
                }
        }

        // IN CASE OF CHECKPOINT ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        double *host_M_k_alpha;
        double *host_M_k_I_nu_0;

        double *host_Q_k_alpha;
        double *host_Q_k_I_nu_0;

        char *I_nu_0_M_k;
        char *alpha_M_k;

        char *I_nu_0_Q_k;
        char *alpha_Q_k;

        if(checkpoint) {
                host_M_k = (double2*)malloc(M*N*sizeof(double2));
                host_Q_k = (double2*)malloc(M*N*sizeof(double2));
                ///////////////////////////////////READ TOTAL SUM //////////////////////////////////////////////////////////////////////////////////
                needed_alpha = snprintf(NULL, 0, "%salpha_%d.fits", checkp, 0) + 1;
                needed_I_nu_0 = snprintf(NULL, 0, "%sI_nu_0_%d.fits", checkp, 0) + 1;

                alpha_M_k = (char*)malloc(needed_alpha*sizeof(char));
                snprintf(alpha_M_k, needed_alpha*sizeof(char), "%salpha_%d.fits", checkp, 0);

                I_nu_0_M_k = (char*)malloc(needed_I_nu_0*sizeof(char));
                snprintf(I_nu_0_M_k, needed_I_nu_0*sizeof(char), "%sI_nu_0_%d.fits", checkp, 0);

                open_read_fits<double>(&host_M_k_alpha, alpha_M_k, M*N, TDOUBLE);
                open_read_fits<double>(&host_M_k_I_nu_0, I_nu_0_M_k, M*N, TDOUBLE);

                /////////////////////////////////READ TOTAL SUM ^2 /////////////////////////////////////////////////////////////////////////////////
                alpha_Q_k = (char*)malloc(needed_alpha*sizeof(char));
                snprintf(alpha_Q_k, needed_alpha*sizeof(char), "%salpha_%d.fits", checkp, 1);
                open_read_fits<double>(&host_Q_k_alpha, alpha_Q_k, M*N, TDOUBLE);


                I_nu_0_Q_k = (char*)malloc(needed_I_nu_0*sizeof(char));
                snprintf(I_nu_0_Q_k, needed_I_nu_0*sizeof(char), "%sI_nu_0_%d.fits", checkp, 1);
                open_read_fits<double>(&host_Q_k_I_nu_0, I_nu_0_Q_k, M*N, TDOUBLE);


                /////////////////////////////////////////////////////PASSING TO DOUBLE2 ARRAY ///////////////////////////////////////////

                for(int i=0; i<M; i++) {
                        for(int j=0; j<N; j++) {
                                host_M_k[N*i+j].x = host_M_k_I_nu_0[N*i+j]; // I_nu
                                host_M_k[N*i+j].y = host_M_k_alpha[N*i+j];

                                host_Q_k[N*i+j].x = host_Q_k_I_nu_0[N*i+j]; // I_nu
                                host_Q_k[N*i+j].y = host_Q_k_alpha[N*i+j];
                        }
                }


        }

        ////////////////////////////////////////////////CUDA MEMORY ALLOCATION FOR DEVICE///////////////////////////////////////////////////
        if(num_gpus == 1) {
                cudaSetDevice(selected);
                gpuErrchk(cudaMalloc((void**)&device_Inu, sizeof(cufftComplex)*M*N));
                gpuErrchk(cudaMalloc((void**)&device_V, sizeof(cufftComplex)*M*N));
        }else{
                for(int g = 0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_Inu, sizeof(cufftComplex)*M*N));
                        gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_V, sizeof(cufftComplex)*M*N));

                }
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        gpuErrchk(cudaMalloc((void**)&device_2I, sizeof(float2)*M*N));
        gpuErrchk(cudaMemset(device_2I, 0, sizeof(float2)*M*N));
        gpuErrchk(cudaMemcpy(device_2I, host_2I, M*N*sizeof(float2), cudaMemcpyHostToDevice));


        if(use_mask) {
                gpuErrchk(cudaMalloc((void**)&device_mask, sizeof(float)*M*N));
                gpuErrchk(cudaMemset(device_mask, 0, sizeof(float)*M*N));
                gpuErrchk(cudaMemcpy(device_mask, host_mask, sizeof(float)*M*N, cudaMemcpyHostToDevice));
        }

        gpuErrchk(cudaMalloc((void**)&device_noise_image, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_noise_image, 0, sizeof(float)*M*N));

        gpuErrchk(cudaMalloc((void**)&device_weight_image, sizeof(cufftComplex)*M*N));
        gpuErrchk(cudaMemset(device_weight_image, 0, sizeof(cufftComplex)*M*N));

        gpuErrchk(cudaMalloc((void**)&device_dphi, sizeof(float2)*M*N));
        gpuErrchk(cudaMemset(device_dphi, 0, sizeof(float2)*M*N));


        gpuErrchk(cudaMalloc((void**)&device_dS, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_dS, 0, sizeof(float)*M*N));

        gpuErrchk(cudaMalloc((void**)&device_dS_alpha, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_dS_alpha, 0, sizeof(float)*M*N));

        gpuErrchk(cudaMalloc((void**)&device_S, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_S, 0, sizeof(float)*M*N));

        gpuErrchk(cudaMalloc((void**)&device_S_alpha, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_S_alpha, 0, sizeof(float)*M*N));



        if(num_gpus == 1) {
                cudaSetDevice(selected);
                gpuErrchk(cudaMemset(device_Inu, 0, sizeof(cufftComplex)*M*N));
                gpuErrchk(cudaMemset(device_V, 0, sizeof(cufftComplex)*M*N));
        }else{
                for(int g = 0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        gpuErrchk(cudaMemset(vars_gpu[g].device_V, 0, sizeof(cufftComplex)*M*N));
                        gpuErrchk(cudaMemset(vars_gpu[g].device_Inu, 0, sizeof(cufftComplex)*M*N));
                }
        }




        if(num_gpus == 1) {
                cudaSetDevice(selected);
                if ((cufftPlan2d(&plan1GPU, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
                        printf("cufft plan error\n");
                        return -1;
                }
        }else{
                for(int g = 0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        if ((cufftPlan2d(&vars_gpu[g].plan, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
                                printf("cufft plan error\n");
                                return -1;
                        }
                }
        }

        //Time is taken from first kernel
        t = clock();
        start = omp_get_wtime();
        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f < data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                hermitianSymmetry<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].uvw, fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].freq, fields[f].numVisibilitiesPerFreq[i]);
                                gpuErrchk(cudaDeviceSynchronize());
                        }
                }
        }else{
                for(int f = 0; f < data.nfields; f++) {
      #pragma omp parallel for schedule(static,1)
                        for (int i = 0; i < data.total_frequencies; i++)
                        {
                                unsigned int j = omp_get_thread_num();
                                //unsigned int num_cpu_threads = omp_get_num_threads();
                                // set and check the CUDA device for this CPU thread
                                int gpu_id = -1;
                                cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                cudaGetDevice(&gpu_id);
                                hermitianSymmetry<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].uvw, fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].freq, fields[f].numVisibilitiesPerFreq[i]);
                                gpuErrchk(cudaDeviceSynchronize());
                        }

                }
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                if(fields[f].numVisibilitiesPerFreq[i] > 0) {
                                        total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].atten_image, antenna_diameter, pb_factor, pb_cutoff, fields[f].visibilities[i].freq, fields[f].ref_xobs, fields[f].ref_yobs, DELTAX, DELTAY, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        #pragma omp parallel for schedule(static,1)
                        for (int i = 0; i < data.total_frequencies; i++)
                        {
                                unsigned int j = omp_get_thread_num();
                                //unsigned int num_cpu_threads = omp_get_num_threads();
                                // set and check the CUDA device for this CPU thread
                                int gpu_id = -1;
                                cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                cudaGetDevice(&gpu_id);
                                if(fields[f].numVisibilitiesPerFreq[i] > 0) {
                                        #pragma omp critical
                                        {
                                                total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].atten_image, antenna_diameter, pb_factor, pb_cutoff, fields[f].visibilities[i].freq, fields[f].ref_xobs, fields[f].ref_yobs, DELTAX, DELTAY, N);
                                                gpuErrchk(cudaDeviceSynchronize());
                                        }
                                }
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                if(num_gpus == 1) {
                        cudaSetDevice(selected);
                        mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].atten_image, fields[f].valid_frequencies, N);
                        gpuErrchk(cudaDeviceSynchronize());
                }else{
                        cudaSetDevice(firstgpu);
                        mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].atten_image, fields[f].valid_frequencies, N);
                        gpuErrchk(cudaDeviceSynchronize());
                }
                if(print_images)
                        fitsOutputFloat(vars_per_field[f].atten_image, mod_in, mempath, f, M, N, 0);
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        for(int f=0; f<data.nfields; f++) {
                weight_image<<<numBlocksNN, threadsPerBlockNN>>>(device_weight_image, vars_per_field[f].atten_image, noise_jypix, N);
                gpuErrchk(cudaDeviceSynchronize());
        }
        noise_image<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_weight_image, noise_jypix, N);
        gpuErrchk(cudaDeviceSynchronize());
        if(print_images)
                fitsOutputFloat(device_noise_image, mod_in, mempath, 0, M, N, 1);


        host_noise_image = (float*)malloc(M*N*sizeof(float));
        pixels = (int2*)malloc(1*sizeof(int2));
        //pixels[N*i+j] = N*i+j;
        gpuErrchk(cudaMemcpy(host_noise_image, device_noise_image, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        float noise_min = *std::min_element(host_noise_image,host_noise_image+(M*N));

        fg_scale = noise_min;
        noise_cut *= noise_min;

        valid_pixels = 0;
        for(int i=0; i<M; i++) {
                for(int j=0; j<N; j++) {
                        if(host_noise_image[N*i+j] < noise_cut) {
                                pixels = (int2*)realloc(pixels, (valid_pixels+1)*sizeof(int2));
                                pixels[valid_pixels].x = i;
                                pixels[valid_pixels].y = j;
                                valid_pixels++;
                        }
                }
        }


        if(verbose_flag) {
                printf("fg_scale = %e\n", fg_scale);
                printf("noise_jypix = %e\n", noise_jypix);
        }

        cudaFree(device_weight_image);
        for(int f=0; f<data.nfields; f++) {
                cudaFree(vars_per_field[f].atten_image);
        }



        //////////////////////////////////////////////////////Fletcher-Reeves Polak-Ribiere Minimization////////////////////////////////////////////////////////////////
        printf("\n\nStarting MCMC\n\n");
        float fret = 0.0;

        float2 theta_init;
        //theta_init.x = MINPIX * fg_scale + fg_scale;
        float analytical_noise_jypix = noise_jypix;
        float gpuvmem_factor = sqrtf(PI * (1.0/3.0) * beam_bmaj * (1.0/3.0) * beam_bmin / (4 * log(2) ));
        float analytical_noise_gpuvmem = analytical_noise_jypix * gpuvmem_factor;
        theta_init.x = analytical_noise_gpuvmem;
        float peak_I_nu_0 = *std::max_element(input_Inu_0,input_Inu_0+(M*N));
        float peak_alpha = *std::max_element(input_alpha,input_alpha+(M*N));
        //theta_init.y = peak_alpha / 1000;
        float nu_2 = 0.0f;
        float nu_1 = 0.0f;

        for (int i = 0; i < data.total_frequencies/2; i++) {
                nu_1 += fields[0].visibilities[i].freq;
        }

        nu_1 /= data.total_frequencies/2;

        int freq_count;
        for (int i = data.total_frequencies/2; i < data.total_frequencies; i++) {
                nu_2 += fields[0].visibilities[i].freq;
        }

        nu_2 /= (data.total_frequencies - data.total_frequencies/2);

        //theta_init.y = sqrt(2) * (analytical_noise_jypix/peak_I_nu_0) / sqrt(logf(nu_2/nu_1));
        printf("I_nu_0 Noise : %e\n", theta_init.x);
        printf("nu_2: %e, nu_1: %e\n", nu_2, nu_1);
        //printf("Alpha Noise : %f\n", theta_init.y);



        float2 *theta_device;
        float2 *theta_host = (float2*)malloc((M*N)*sizeof(float2));
        for(int i=0; i<M; i++) {
                for(int j=0; j<N; j++) {
                        theta_host[N*i+j].x = theta_init.x;
                        if(input_Inu_0[N*i+j]>= 5*analytical_noise_jypix)
                                theta_host[N*i+j].y = sqrt(2) * (analytical_noise_gpuvmem/input_Inu_0[N*i+j]) / logf(nu_2/nu_1);
                        else
                                theta_host[N*i+j].y = sqrt(2) * (analytical_noise_gpuvmem/peak_I_nu_0) / logf(nu_2/nu_1);
                }
        }

        free(input_alpha);
        free(input_Inu_0);
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        gpuErrchk(cudaMalloc((void**)&theta_device, sizeof(float2)*M*N));
        gpuErrchk(cudaMemset(theta_device, 0, sizeof(float2)*M*N));
        gpuErrchk(cudaMemcpy(theta_device, theta_host, sizeof(float2)*M*N, cudaMemcpyHostToDevice));

        printf("Iterations: %d\n", it_maximum);
        printf("Burn in steps: %d\n", variables.burndown_steps);
        printf("Accepted: %d\n", variables.current_k);

        MetropolisHasting(device_2I, theta_device, it_maximum, &variables.burndown_steps, &variables.current_k);

        t = clock() - t;
        end = omp_get_wtime();
        printf("MCMC ended successfully\n\n");

        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        double wall_time = end-start;
        printf("Total CPU time: %lf\n", time_taken);
        printf("Wall time: %lf\n\n\n", wall_time);

        if(strcmp(variables.ofile,"NULL") != 0) {
                FILE *outfile = fopen(variables.ofile, "w");
                if (outfile == NULL)
                {
                        printf("Error opening output file!\n");
                        goToError();
                }

                fprintf(outfile, "Iterations: %d\n", iter);
                fprintf(outfile, "chi2: %f\n", final_chi2);
                fprintf(outfile, "0.5*chi2: %f\n", 0.5*final_chi2);
                fprintf(outfile, "Total visibilities: %d\n", total_visibilities);
                fprintf(outfile, "Reduced-chi2 (Num visibilities): %f\n", (0.5*final_chi2)/total_visibilities);
                //fprintf(outfile, "Reduced-chi2 (Weights sum): %f\n", (0.5*final_chi2)/sum_weights);
                fprintf(outfile, "S: %f\n", final_H);
                if(reg_term != 1) {
                        fprintf(outfile, "Normalized S: %f\n", final_H/(M*N));
                }else{
                        fprintf(outfile, "Normalized S: %f\n", final_H/(M*M*N*N));
                }
                fprintf(outfile, "lambda*S: %f\n", lambda*final_H);
                fprintf(outfile, "Wall time: %lf", wall_time);
                fclose(outfile);
        }
        //Pass residuals to host
        /*printf("Saving final image to disk\n");
           float2toImage(device_2I, mod_in, out_image, mempath, iter, M, N, 0);
           //Saving residuals to disk
           residualsToHost(fields, data, num_gpus, firstgpu);
           printf("Saving residuals to MS...\n");
           writeMS(msinput,msoutput,fields, data, random_probability, verbose_flag);
           printf("Residuals saved.\n");*/

        //Free device and host memory
        printf("Free device and host memory\n");
        cufftDestroy(plan1GPU);
        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i<data.total_frequencies; i++) {
                        if(num_gpus > 1) {
                                cudaSetDevice((i%num_gpus) + firstgpu);
                        }
                        cudaFree(fields[f].device_visibilities[i].uvw);
                        cudaFree(fields[f].device_visibilities[i].weight);

                        cudaFree(fields[f].device_visibilities[i].Vr);
                        cudaFree(fields[f].device_visibilities[i].Vo);

                }
        }

        if(num_gpus>1) {
                for(int g = 0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        cufftDestroy(vars_gpu[g].plan);
                }
        }

        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i<data.total_frequencies; i++) {
                        if(fields[f].numVisibilitiesPerFreq[i] != 0) {
                                free(fields[f].visibilities[i].uvw);
                                free(fields[f].visibilities[i].weight);
                                free(fields[f].visibilities[i].Vo);
                                free(fields[f].visibilities[i].Vm);
                        }
                }
        }

        cudaFree(device_2I);
        if(num_gpus == 1) {
                cudaFree(device_V);
                cudaFree(device_Inu);
        }else{
                for(int g = 0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        cudaFree(vars_gpu[g].device_V);
                        cudaFree(vars_gpu[g].device_Inu);
                }
        }
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        cudaFree(device_noise_image);

        cudaFree(device_dphi);
        cudaFree(device_dS);
        cudaFree(device_dS_alpha);

        cudaFree(device_chi2);
        cudaFree(device_S);
        cudaFree(device_S_alpha);

        //Disabling UVA
        if(num_gpus > 1) {
                for(int i=firstgpu+1; i<num_gpus+firstgpu; i++) {
                        cudaSetDevice(firstgpu);
                        cudaDeviceDisablePeerAccess(i);
                        cudaSetDevice(i);
                        cudaDeviceDisablePeerAccess(firstgpu);
                }

                for(int i=0; i<num_gpus; i++ ) {
                        cudaSetDevice((i%num_gpus) + firstgpu);
                        cudaDeviceReset();
                }
        }
        free(host_2I);
        free(msinput);
        free(msoutput);
        free(modinput);
        free(host_noise_image);

        closeCanvas(mod_in);

        if (status_mod_in) {
                fits_report_error(stderr, status_mod_in);
                goToError();
        }

        return 0;
}
