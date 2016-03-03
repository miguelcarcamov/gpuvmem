#include "functions.cuh"


extern long M;
extern long N;
extern int numVisibilities;
extern int iterations;
extern int iter;


extern cufftHandle plan1GPU;
extern cufftComplex *device_I;
extern cufftComplex *device_V;
extern cufftComplex *device_noise_image;
extern cufftComplex *device_fg_image;
extern cufftComplex *device_image;

extern float *device_dphi;


extern float *device_chi2;
extern float *device_H;
extern float *device_dchi2_total;
extern float *device_dH;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int threadsVectorNN;
extern int blocksVectorNN;

extern float difmap_noise;
extern float fg_scale;

extern float global_time;

extern float global_xobs;
extern float global_yobs;

extern float DELTAX;
extern float DELTAY;
extern float deltau;
extern float deltav;

extern float noise_cut;
extern float MINPIX;
extern float minpix_factor;
extern float lambda;
extern float ftol;
extern float random_probability;


extern float beam_noise;
extern float beam_bmaj;
extern float beam_bmin;
extern float ra;
extern float dec;
extern float obsra;
extern float obsdec;
extern float crpix1;
extern float crpix2;
extern freqData data;
extern VPF *device_vars;
extern Vis *visibilities;
extern Vis *device_visibilities;

extern int num_gpus;

extern fitsfile *mod_in;
extern int status_mod_in;

__host__ void goToError()
{
  for(int i=1; i<num_gpus; i++){
        cudaSetDevice(0);
        cudaDeviceDisablePeerAccess(i);
        cudaSetDevice(i%num_gpus);
        cudaDeviceDisablePeerAccess(0);
  }

  for(int i=0; i<num_gpus; i++ ){
        cudaSetDevice(i%num_gpus);
        cudaDeviceReset();
  }

  printf("An error has ocurred, exiting\n");
  exit(0);

}

__host__ freqData getFreqs(char * file)
{
   freqData freqsAndVisibilities;
   sqlite3 *db;
   sqlite3_stmt *stmt;
   char *err_msg = 0;

   int rc = sqlite3_open(file, &db);

   if (rc != SQLITE_OK) {
    printf("Cannot open database: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    goToError();
  }else{
    printf("Database connection okay!\n");
  }

  char *sql = "SELECT n_internal_frequencies as nfreq FROM header";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
  sqlite3_step(stmt);
  freqsAndVisibilities.n_internal_frequencies = sqlite3_column_int(stmt, 0);

  freqsAndVisibilities.channels = (int*)malloc(freqsAndVisibilities.n_internal_frequencies*sizeof(int));

  sql = "SELECT COUNT(*) as channels FROM channels WHERE internal_freq_id = ?";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
  if (rc != SQLITE_OK ) {
    printf("Cannot open database: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    goToError();
  }

  for(int i=0; i< freqsAndVisibilities.n_internal_frequencies; i++){
      sqlite3_bind_int(stmt, 1, i);
      sqlite3_step(stmt);
      freqsAndVisibilities.channels[i] = sqlite3_column_int(stmt, 0);
      sqlite3_reset(stmt);
  }

  int total_frequencies = 0;
  for(int i=0; i <freqsAndVisibilities.n_internal_frequencies; i++){
    for(int j=0; j < freqsAndVisibilities.channels[i]; j++){
      total_frequencies++;
    }
  }

  freqsAndVisibilities.total_frequencies = total_frequencies;
  freqsAndVisibilities.numVisibilitiesPerFreq = (long*)malloc(freqsAndVisibilities.total_frequencies*sizeof(long));


  sql = "SELECT COUNT(*) AS visibilitiesPerFreq FROM(SELECT samples.u as u, samples.v as v, samples.w as w, visibilities.stokes as stokes, samples.id_field as id_field, visibilities.Re as Re, visibilities.Im as Im, weights.weight as We, id_antenna1, id_antenna2, channels.internal_freq_id as  internal_frequency_id , visibilities.channel as channel, frequency FROM visibilities,samples,weights,channels WHERE visibilities.flag=0 and samples.flag_row=0 and visibilities.id_sample = samples.id and weights.id_sample=samples.id and weights.stokes=visibilities.stokes and channels.internal_freq_id=samples.internal_freq_id and visibilities.channel=channels.channel) WHERE internal_frequency_id = ? AND channel = ?";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
  if (rc != SQLITE_OK ) {
    printf("Cannot open database: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    goToError();
  }
  int h=0;
  for(int i=0; i <freqsAndVisibilities.n_internal_frequencies; i++){
    for(int j=0; j < freqsAndVisibilities.channels[i]; j++){
      sqlite3_bind_int(stmt, 1, i);
      sqlite3_bind_int(stmt, 2, j);
      sqlite3_step(stmt);
      freqsAndVisibilities.numVisibilitiesPerFreq[h] = (long)sqlite3_column_int(stmt, 0);
      sqlite3_reset(stmt);
      sqlite3_clear_bindings(stmt);
      h++;
    }
  }

  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return freqsAndVisibilities;
}

__host__ long NearestPowerOf2(long n)
{
  if (!n) return n;  //(0 == 2^0)

  int x = 1;
  while(x < n)
  {
      x <<= 1;
  }
  return x;
}


bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}


__host__ void readInputDat(char *file)
{
  FILE *fp;
  char item[50];
  float status;
  if((fp = fopen(file, "r")) == NULL){
    printf("ERROR. The file path wasn't provided by the user.\n");
    goToError();
  }else{
    while(true){
      int ret = fscanf(fp, "%s %e", item, &status);

      if(ret==EOF){
        break;
      }else{
        if (strcmp(item,"lambda_entropy")==0) {
          lambda = status;
        }else if (strcmp(item,"noise_cut")==0){
          noise_cut = status;
        }else if(strcmp(item,"minpix_factor")==0){
          minpix_factor = status;
        } else if(strcmp(item,"ftol")==0){
          ftol = status;
        } else if(strcmp(item,"random_probability")==0){
          random_probability = status;
        } else{
          break;
        }
      }
    }
  }
}
__host__ void readMS(char *file, char *file2, char *file3, Vis *visibilities) {
  ///////////////////////////////////////////////////FITS READING///////////////////////////////////////////////////////////
  status_mod_in = 0;
  fits_open_file(&mod_in, file2, 0, &status_mod_in);
  if (status_mod_in) {
    fits_report_error(stderr, status_mod_in); /* print error message */
    goToError();
  }


  fits_read_key(mod_in, TFLOAT, "CDELT1", &DELTAX, NULL, &status_mod_in);
  fits_read_key(mod_in, TFLOAT, "CDELT2", &DELTAY, NULL, &status_mod_in);
  fits_read_key(mod_in, TFLOAT, "CRVAL1", &ra, NULL, &status_mod_in);
  fits_read_key(mod_in, TFLOAT, "CRVAL2", &dec, NULL, &status_mod_in);
  fits_read_key(mod_in, TFLOAT, "CRPIX1", &crpix1, NULL, &status_mod_in);
  fits_read_key(mod_in, TFLOAT, "CRPIX2", &crpix2, NULL, &status_mod_in);
  fits_read_key(mod_in, TLONG, "NAXIS1", &M, NULL, &status_mod_in);
  fits_read_key(mod_in, TLONG, "NAXIS2", &N, NULL, &status_mod_in);
  if (status_mod_in) {
    fits_report_error(stderr, status_mod_in); /* print error message */
    goToError();
  }


  fitsfile *fpointer2;
  int status2 = 0;
  fits_open_file(&fpointer2, file3, 0, &status2);
  if (status2) {
    fits_report_error(stderr, status2); /* print error message */
    goToError();
  }
  fits_read_key(fpointer2, TFLOAT, "NOISE", &beam_noise, NULL, &status2);
  fits_read_key(fpointer2, TFLOAT, "BMAJ", &beam_bmaj, NULL, &status2);
  fits_read_key(fpointer2, TFLOAT, "BMIN", &beam_bmin, NULL, &status2);
  if (status2) {
    fits_report_error(stderr, status2); /* print error message */
    goToError();
  }
  fits_close_file(fpointer2, &status2);
  if (status2) {
    fits_report_error(stderr, status2); /* print error message */
    goToError();
  }

  printf("FITS Files READ\n");

  ///////////////////////////////////////////////////MS SQLITE READING/////////////////////////////////////////////////////////
  sqlite3 *db;
  sqlite3_stmt *stmt;
  char *error = 0;
  int k = 0;
  int h = 0;


  int rc = sqlite3_open(file, &db);
  if (rc != SQLITE_OK) {
   printf("Cannot open database: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
  }else{
   printf("Database connection okay again!\n");
 }

  char *sql;
  sql = "SELECT id, stokes, u, v, Re, Im, We FROM(SELECT samples.id as id, samples.u as u, samples.v as v, samples.w as w, visibilities.stokes as stokes, samples.id_field as id_field, visibilities.Re as Re, visibilities.Im as Im, weights.weight as We, id_antenna1, id_antenna2, channels.internal_freq_id as  internal_frequency_id , visibilities.channel as channel, frequency FROM visibilities,samples,weights,channels WHERE visibilities.flag=0 and samples.flag_row=0 and visibilities.id_sample = samples.id and weights.id_sample=samples.id and weights.stokes=visibilities.stokes and channels.internal_freq_id=samples.internal_freq_id and visibilities.channel=channels.channel) WHERE internal_frequency_id = ? AND channel = ?";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK ) {
    printf("Cannot execute SELECT: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    goToError();
  }

  if(random_probability!=0.0){
    float u;
    SelectStream(0);
    PutSeed(1);
    for(int i=0; i < data.n_internal_frequencies; i++){
      for(int j=0; j < data.channels[i]; j++){
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_int(stmt, 2, j);
        while (1) {
          int s = sqlite3_step(stmt);
          if (s == SQLITE_ROW) {
            u = Random();
            if(u<1-random_probability){
              visibilities[k].id[h] = sqlite3_column_int(stmt, 0);
              visibilities[k].stokes[h] = sqlite3_column_int(stmt, 1);
              visibilities[k].u[h] = sqlite3_column_double(stmt, 2);
              visibilities[k].v[h] = sqlite3_column_double(stmt, 3);
              visibilities[k].Vo[h].x = sqlite3_column_double(stmt, 4);
              visibilities[k].Vo[h].y = sqlite3_column_double(stmt, 5);
              visibilities[k].weight[h] = sqlite3_column_double(stmt, 6);
              h++;
            }
          }else if(s == SQLITE_DONE) {
            break;
          }else{
            printf("Database queries failed ERROR: %d.\n", s);
            goToError();
          }
        }
        data.numVisibilitiesPerFreq[k] = (h+1);
        realloc(visibilities[k].id, (h+1)*sizeof(int));
        realloc(visibilities[k].stokes, (h+1)*sizeof(int));
        realloc(visibilities[k].u, (h+1)*sizeof(float));
        realloc(visibilities[k].v, (h+1)*sizeof(float));
        realloc(visibilities[k].Vo, (h+1)*sizeof(cufftComplex));
        realloc(visibilities[k].weight, (h+1)*sizeof(float));
        h=0;
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        k++;
      }
    }
  }else{
    for(int i=0; i < data.n_internal_frequencies; i++){
      for(int j=0; j < data.channels[i]; j++){
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_int(stmt, 2, j);
        while (1) {
          int s = sqlite3_step(stmt);
          if (s == SQLITE_ROW) {
            visibilities[k].id[h] = sqlite3_column_int(stmt, 0);
            visibilities[k].stokes[h] = sqlite3_column_int(stmt, 1);
            visibilities[k].u[h] = sqlite3_column_double(stmt, 2);
            visibilities[k].v[h] = sqlite3_column_double(stmt, 3);
            visibilities[k].Vo[h].x = sqlite3_column_double(stmt, 4);
            visibilities[k].Vo[h].y = sqlite3_column_double(stmt, 5);
            visibilities[k].weight[h] = sqlite3_column_double(stmt, 6);
            h++;
          }else if(s == SQLITE_DONE) {
            break;
          }else{
            printf("Database queries failed ERROR: %d.\n", s);
            goToError();
          }
        }
        h=0;
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        k++;
      }
    }
  }
  printf("Visibilities read!\n");



  sql = "SELECT ra, dec FROM fields";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK ) {
    printf("SQL error: %s\n", error);
    sqlite3_free(error);
    sqlite3_close(db);
    goToError();
  }
  sqlite3_step(stmt);
  obsra = sqlite3_column_double(stmt, 0);
  obsdec = sqlite3_column_double(stmt, 1);
  printf("Center read!\n");

  sql = "SELECT frequency as freq_vector FROM channels WHERE internal_freq_id = ? AND channel = ?";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
  if (rc != SQLITE_OK ) {
    printf("SQL error: %s\n", error);
    sqlite3_free(error);
    sqlite3_close(db);
    goToError();
  }
  h=0;
  for(int i = 0; i < data.n_internal_frequencies; i++){
    for(int j = 0; j < data.channels[i]; j++){
      sqlite3_bind_int(stmt, 1, i);
      sqlite3_bind_int(stmt, 2, j);
      while (1) {
        int s = sqlite3_step(stmt);
        if (s == SQLITE_ROW) {
          visibilities[h].freq = sqlite3_column_double(stmt, 0);
          h++;
        }else if(s == SQLITE_DONE) {
          break;
        }else{
          printf("Database queries failed ERROR %d at %d and %d.\n",s, i, j);
          goToError();
        }
      }
      sqlite3_reset(stmt);
      sqlite3_clear_bindings(stmt);
    }
  }
  printf("Frequencies read!\n");

  sqlite3_close(db);
}

__host__ void writeMS(char *file, Vis *visibilities) {
  sqlite3 *db;
  sqlite3_stmt *stmt;
  int rc;

  rc = sqlite3_open(file, &db);
  if (rc != SQLITE_OK) {
   printf("Cannot open output database: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
  }else{
   printf("Output Database connection okay!\n");
 }

 //char *sql = "UPDATE SET Re = ?, Im = ? WHERE u = ? AND v = ? AND Re = ? AND Im = ? AND We = ?";

 //char *sql = "UPDATE visibilities SET re = ?, im = ? WHERE flag=0 AND id_sample= ? AND stokes = ?";
 char *sql = "DELETE FROM visibilities WHERE flag = 0";
 //rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
 rc = sqlite3_exec(db, sql, NULL, NULL, 0);
 if (rc != SQLITE_OK ) {
   printf("Cannot execute DELETE: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
 }else{
   printf("Observed visibilities deleted from output file!\n");
 }
 sqlite3_exec(db, "PRAGMA synchronous = OFF", NULL, NULL, 0);
 sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", NULL, NULL, 0);

 int l = 0;
 sql = "INSERT INTO visibilities (id_sample, stokes, channel, re, im, flag) VALUES (?, ?, ?, ?, ?, 0)";
 rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
 if (rc != SQLITE_OK ) {
   printf("Cannot execute INSERT: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
 }else{
   printf("Saving residuals to file. Please wait...\n");
 }

 for(int i=0; i<data.n_internal_frequencies; i++){
   for(int j=0; j<data.channels[i]; j++){
     for(int k=0; k<data.numVisibilitiesPerFreq[l]; k++){
       sqlite3_bind_int(stmt, 1, visibilities[l].id[k]);
       sqlite3_bind_int(stmt, 2, visibilities[l].stokes[k]);
       sqlite3_bind_int(stmt, 3, j);
       sqlite3_bind_double(stmt, 4, visibilities[l].Vr[k].x);
       sqlite3_bind_double(stmt, 5, visibilities[l].Vr[k].y);
       int s = sqlite3_step(stmt);
       sqlite3_clear_bindings(stmt);
       sqlite3_reset(stmt);
       printf("\rIF[%d/%d], channel[%d/%d]: %d/%d ====> %d %%", i+1, data.n_internal_frequencies, j+1, data.channels[i], k+1, data.numVisibilitiesPerFreq[l], int((k*100.0)/data.numVisibilitiesPerFreq[l])+1);
       fflush(stdout);
     }
     printf("\n");
     l++;
   }
 }
 sqlite3_finalize(stmt);
 sqlite3_close(db);

}


__host__ void print_help() {
	printf("Example: executable_name options [ arguments ...]\n");
	printf("    -h  --help       Shows this\n");
	printf(	"    -i  --input      The name of the input file of visibilities(SQLite)\n");
  printf(	"    -o  --output     The name of the output file of visibilities(SQLite)\n");
  printf("    -d  --inputdat   The name of the input file of parameters\n");
  printf("    -m  --modin      mod_in_0 FITS file location\n");
  printf("    -b  --beam       beam_0 FITS file location\n");
  printf("    -g  --multigpu   Option for multigpu ON (Default OFF)\n");
}

__host__ Vars getOptions(int argc, char **argv) {
	Vars variables;
	variables.input = (char*) malloc(2000*sizeof(char));
  variables.output = (char*) malloc(2000*sizeof(char));
  variables.inputdat = (char*) malloc(2000*sizeof(char));
  variables.beam = (char*) malloc(2000*sizeof(char));
  variables.modin = (char*) malloc(2000*sizeof(char));
  variables.multigpu = 0;

	long next_op;
	const char* const short_op = "hi:o:d:m:b:g:";

	const struct option long_op[] = { {"help", 0, NULL, 'h' }, {"input", 1, NULL, 'i' }, {"output", 1, NULL, 'o'}, {"inputdat", 1, NULL, 'd'}, {"modin", 1, NULL, 'm' }, {"beam", 1, NULL, 'b' }, {"multigpu", 1, NULL, 'g'}, { NULL, 0, NULL, 0 } };

	if (argc == 1) {
		printf(
				"ERROR. THE PROGRAM HAS BEEN EXECUTED WITHOUT THE NEEDED PARAMETERS OR OPTIONS\n");
		print_help();
		exit(EXIT_SUCCESS);
	}

	while (1) {
		next_op = getopt_long(argc, argv, short_op, long_op, NULL);
		if (next_op == -1) {
			break;
		}

		switch (next_op) {
		case 'h':
			print_help();
			exit(EXIT_SUCCESS);
		case 'i':
			strcpy(variables.input, optarg);
			break;
    case 'o':
  		strcpy(variables.output, optarg);
  		break;
    case 'd':
      strcpy(variables.inputdat, optarg);
      break;
    case 'm':
    	strcpy(variables.modin, optarg);
    	break;
    case 'b':
    	strcpy(variables.beam, optarg);
    	break;
    case 'g':
      variables.multigpu = atoi(optarg);
      break;
		case '?':
			print_help();
			exit(1);
		case -1:
			break;
		default:
			abort();
		}
	}

	return variables;
}


__host__ void toFitsDouble(cufftComplex *I, int iteration, long M, long N, int option)
{
	fitsfile *fpointer;
	int status = 0;
	long fpixel = 1;
	long elements = M*N;
	char name[60]="";
	long naxes[2]={M,N};
	long naxis = 2;
  char *unit = "JY/PIXEL";
  switch(option){
    case 2:
      sprintf(name, "!out/atten_%d.fits", iteration);
      break;
    case 3:
      sprintf(name, "!out/total_atten_0.fits", iteration);
      break;
    case 4:
      sprintf(name, "!out/noise_0.fits", iteration);
      break;
    case -1:
      break;
    default:
      printf("Invalid case to FITS\n");
      goToError();
  }


	fits_create_file(&fpointer, name, &status);
  fits_copy_header(mod_in, fpointer, &status);
  if(option==0){
    fits_update_key(fpointer, TSTRING, "BUNIT", unit, "Unit of measurement", &status);
  }
  cufftComplex *host_IFITS;
  host_IFITS = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));
  gpuErrchk(cudaMemcpy2D(host_IFITS, sizeof(cufftComplex), I, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToHost));

	float* image2D;
	image2D = (float*) malloc(M*N*sizeof(float));

  int x = N-1;
  int y = M-1;
  for(int i=M-1; i >=0; i--){
		for(int j=0; j < N; j++){
      if(option == 0){
			  image2D[N*x+y] = host_IFITS[N*i+j].x * fg_scale;
      }else{
        image2D[N*x+y] = host_IFITS[N*i+j].x;
      }
      x--;
		}
    y--;
    x=N-1;
	}

	fits_write_img(fpointer, TFLOAT, fpixel, elements, image2D, &status);
	fits_close_file(fpointer, &status);
	fits_report_error(stderr, status);

  free(host_IFITS);
	free(image2D);
}


__host__ void toFitsFloat(cufftComplex *I, int iteration, long M, long N, int option)
{
	fitsfile *fpointer;
	int status = 0;
	long fpixel = 1;
	long elements = M*N;
	char name[60]="";
	long naxes[2]={M,N};
	long naxis = 2;
  char *unit = "JY/PIXEL";
  switch(option){
    case 0:
      sprintf(name, "!out/mod_out.fits", iteration);
      break;
    case 1:
      sprintf(name, "!out/MEM_%d.fits", iteration);
      break;
    case 2:
      sprintf(name, "!out/MEM_V_%d.fits", iteration);
      break;
    case 3:
      sprintf(name, "!out/MEM_VB_%d.fits", iteration);
      break;
    case -1:
      break;
    default:
      printf("Invalid case to FITS\n");
      goToError();
  }


	fits_create_file(&fpointer, name, &status);
  fits_copy_header(mod_in, fpointer, &status);
  if(option==0){
    fits_update_key(fpointer, TSTRING, "BUNIT", unit, "Unit of measurement", &status);
  }
  cufftComplex *host_IFITS;
  host_IFITS = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));
  gpuErrchk(cudaMemcpy2D(host_IFITS, sizeof(cufftComplex), I, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToHost));

	float* image2D;
	image2D = (float*) malloc(M*N*sizeof(float));

  int x = N-1;
  //int y = M-1;
  int y=0;
  for(int i=M-1; i >=0; i--){
		for(int j=0; j < N; j++){
      if(option == 0){
			  image2D[N*x+y] = host_IFITS[N*i+j].x * fg_scale;
      }else if (option == 2 || option == 3){
        image2D[N*x+y] = sqrt(host_IFITS[N*i+j].x * host_IFITS[N*i+j].x + host_IFITS[N*i+j].y * host_IFITS[N*i+j].y);
        //image2D[N*x+y] = host_IFITS[N*i+j].y;
      }else{
        image2D[N*x+y] = host_IFITS[N*i+j].x;
      }
      x--;
		}
    y++;
    x=N-1;
	}

	fits_write_img(fpointer, TFLOAT, fpixel, elements, image2D, &status);
	fits_close_file(fpointer, &status);
	fits_report_error(stderr, status);

  free(host_IFITS);
	free(image2D);
}


template <bool nIsPow2>
__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0.f;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}





__host__ float deviceReduce(float *in, long N) {
  float *device_out;
  gpuErrchk(cudaMalloc(&device_out, sizeof(float)*1024));
  gpuErrchk(cudaMemset(device_out, 0, sizeof(float)*1024));

  int threads = 512;
  int blocks = min((int(NearestPowerOf2(N)) + threads - 1) / threads, 1024);
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  bool isPower2 = isPow2(N);
  if(isPower2){
    deviceReduceKernel<true><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }else{
    deviceReduceKernel<false><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }

  float *h_odata = (float *) malloc(blocks*sizeof(float));
  float sum = 0.0;

  gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(float),cudaMemcpyDeviceToHost));
  for (int i=0; i<blocks; i++)
  {
    sum += h_odata[i];
  }
  cudaFree(device_out);
  free(h_odata);
	return sum;
}

__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities){
      if(Ux[i] < 0.0){
        Ux[i] *= -1.0;
        Vx[i] *= -1.0;
        Vo[i].y *= -1.0;
      }
      Ux[i] = (Ux[i] * freq) / LIGHTSPEED;
      Vx[i] = (Vx[i] * freq) / LIGHTSPEED;
  }
}
__global__ void attenuation(cufftComplex *attenMatrix, float frec, long N, float xobs, float yobs, float DELTAX, float DELTAY)
{

		int i = threadIdx.x + blockDim.x * blockIdx.x;
		int j = threadIdx.y + blockDim.y * blockIdx.y;


    float x = (i - (int)xobs) * DELTAX * RPDEG;
    float y = (j - (int)yobs) * DELTAY * RPDEG;

    float arc = sqrt(x*x+y*y);
    float c = 4.0*log(2.0);
    //printf("frec:%f\n", frec);
    float a = (FWHM*BEAM_FREQ/(frec*1e-9));
    float r = arc/a;
    float atten = exp(-c*r*r);
    attenMatrix[N*i+j].x = atten;
    attenMatrix[N*i+j].y = 0;
}

__global__ void total_attenuation(cufftComplex *total_atten, cufftComplex *attenperFreq, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  total_atten[N*i+j].x += attenperFreq[N*i+j].x;
  total_atten[N*i+j].y = 0;
}

__global__ void noise_image(cufftComplex *total_atten, cufftComplex *noise_image, float difmap_noise, long N, int nfreq)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  float weight = 0.0;
  float noiseval = 0.0;
  float atten = total_atten[N*i+j].x;
  atten = atten/nfreq;
  weight = (atten / difmap_noise) * (atten / difmap_noise);
  noiseval = sqrt(1.0/weight);
  noise_image[N*i+j].x = noiseval;
  noise_image[N*i+j].y = 0;
}

__global__ void apply_beam(cufftComplex *image, cufftComplex *fg_image, long N, float xobs, float yobs, float fg_scale, float frec, float DELTAX, float DELTAY)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;


    float dx = DELTAX * 60.0;
    float dy = DELTAY * 60.0;
    float x = (i - xobs) * dx;
    float y = (j - yobs) * dy;
    float arc = RPARCM*sqrt(x*x+y*y);
    float c = 4.0*log(2.0);
    float a = (FWHM*BEAM_FREQ/(frec*1e-9));
    float r = arc/a;
    float atten = exp(-c*r*r);

    image[N*i+j].x = fg_image[N*i+j].x * fg_scale * atten;
    image[N*i+j].y = 0.f;
}

__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs)
{

		int i = threadIdx.x + blockDim.x * blockIdx.x;
		int j = threadIdx.y + blockDim.y * blockIdx.y;

    float u,v;
    float du = -2.0 * (xphs/M);
    float dv = -2.0 * (yphs/N);

    if(i < M/2){
      u = du * i;
    }else{
      u = du * (i-M);
    }

    if(j < N/2){
      v = dv * j;
    }else{
      v = dv * (j-N);
    }

    float phase = u+v;
    float c, s;
    #if (__CUDA_ARCH__ >= 300 )
      sincospif(phase, &s, &c);
    #else
      c = cospif(phase);
      s = sinpif(phase);
    #endif
    float  re = data[N*i+j].x;
    float im = data[N*i+j].y;
    data[N*i+j].x = re * c - im * s;
    data[N*i+j].y = re * s + im * c;
}



__global__ void residual(cufftComplex *Vr, cufftComplex *Vo, cufftComplex *V, float *Ux, float *Vx, float deltau, float deltav, long numVisibilities, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int i1, i2, j1, j2;
  float du, dv;
  float v11, v12, v21, v22;
  float Zreal;
  float Zimag;
  if (i < numVisibilities){
    float u = Ux[i]/deltau;
    float v = Vx[i]/deltav;

    if(u < 0.0){
      u = N + u;
    }

    if(v < 0.0){
      v = N + v;
    }

    i1 = u;
    i2 = (i1+1)%N;
    du = u - i1;
    j1 = v;
    j2 = (j1+1)%N;
    dv = v - j1;

      /* Bilinear interpolation: real part */
    v11 = V[N*i1 + j1].x; /* [i1, j1] */
    v12 = V[N*i1 + j2].x; /* [i1, j2] */
    v21 = V[N*i2 + j1].x; /* [i2, j1] */
    v22 = V[N*i2 + j2].x; /* [i2, j2] */
    Zreal = (1-du)*(1-dv)*v11 + (1-du)*dv*v12 + du*(1-dv)*v21 + du*dv*v22;
    /* Bilinear interpolation: imaginary part */
    v11 = V[N*i1 + j1].y; /* [i1, j1] */
    v12 = V[N*i1 + j2].y; /* [i1, j2] */
    v21 = V[N*i2 + j1].y; /* [i2, j1] */
    v22 = V[N*i2 + j2].y; /* [i2, j2] */
    Zimag = (1-du)*(1-dv)*v11 + (1-du)*dv*v12 + du*(1-dv)*v21 + du*dv*v22;

    Vr[i].x =  Zreal - Vo[i].x;
    Vr[i].y =  Zimag - Vo[i].y;

  }

}



__global__ void clip(cufftComplex *I, float MINPIX, long N)
{
	int k = threadIdx.x + blockDim.x * blockIdx.x;
	int l = threadIdx.y + blockDim.y * blockIdx.y;

	if(I[N*k+l].x <= MINPIX){
		I[N*k+l].x = MINPIX;
	}
  I[N*k+l].y = 0;

  //printf("%f\n", I[N*k+l].x);


}

__global__ void clipWNoise(cufftComplex *fg_image, cufftComplex *noise, cufftComplex *I, long N, float noise_cut, float MINPIX)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;


  if(noise[N*i+j].x > noise_cut){
    I[N*i+j].x = MINPIX;
  }

  fg_image[N*i+j].x = I[N*i+j].x;
  //printf("%f\n", fg_image[N*i+j].x);
  fg_image[N*i+j].y = 0;
}


__global__ void getGandDGG(float *gg, float *dgg, float *xi, float *g, long N)
{
  int k = threadIdx.x + blockDim.x * blockIdx.x;
	int l = threadIdx.y + blockDim.y * blockIdx.y;

  gg[N*k+l] = g[N*k+l] * g[N*k+l];
  dgg[N*k+l] = (xi[N*k+l] + g[N*k+l]) * xi[N*k+l];
}

__global__ void newP(cufftComplex *p, float *xi, float xmin, float MINPIX, long N)
{
	int k = threadIdx.x + blockDim.x * blockIdx.x;
	int l = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N*k+l] *= xmin;
  if(p[N*k+l].x + xi[N*k+l] > MINPIX){
    p[N*k+l].x += xi[N*k+l];
  }else{
    p[N*k+l].x = MINPIX;
    xi[N*k+l] = 0.0;
  }
  p[N*k+l].y = 0.0;
}

__global__ void evaluateXt(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, float MINPIX, long N)
{
	int k = threadIdx.x + blockDim.x * blockIdx.x;
	int l = threadIdx.y + blockDim.y * blockIdx.y;

  if(pcom[N*k+l].x + x * xicom[N*k+l] > MINPIX){
      xt[N*k+l].x = pcom[N*k+l].x + x * xicom[N*k+l];
  }else{
      xt[N*k+l].x = MINPIX;
  }
  xt[N*k+l].y = 0.0;
}


__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, long numVisibilities){
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numVisibilities){
		chi2[i] =  w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
	}

}

__global__ void HVector(float *H, cufftComplex *noise, cufftComplex *I, long N, float noise_cut, float MINPIX)
{
	int k = threadIdx.x + blockDim.x * blockIdx.x;
	int l = threadIdx.y + blockDim.y * blockIdx.y;

  float entropy = 0.0;
  if(noise[N*k+l].x <= noise_cut){
    entropy = I[N*k+l].x * log(I[N*k+l].x / MINPIX);
  }

  H[N*k+l] = entropy;
}
__global__ void searchDirection(float *g, float *xi, float *h, long N)
{
  int k = threadIdx.x + blockDim.x * blockIdx.x;
	int l = threadIdx.y + blockDim.y * blockIdx.y;

  g[N*k+l] = -xi[N*k+l];
  xi[N*k+l] = h[N*k+l] = g[N*k+l];
}

__global__ void newXi(float *g, float *xi, float *h, float gam, long N)
{
  int k = threadIdx.x + blockDim.x * blockIdx.x;
	int l = threadIdx.y + blockDim.y * blockIdx.y;

  g[N*k+l] = -xi[N*k+l];
  xi[N*k+l] = h[N*k+l] = g[N*k+l] + gam * h[N*k+l];
}

__global__ void restartDPhi(float *dphi, float *dChi2, float *dH, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  dphi[N*i+j] = 0.0;
  dChi2[N*i+j] = 0.0;
  dH[N*i+j] = 0.0;

}

__global__ void DH(float *dH, cufftComplex *I, cufftComplex *noise, float noise_cut, float lambda, float MINPIX, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j].x <= noise_cut){
    dH[N*i+j] = lambda * (log(I[N*i+j].x / MINPIX) + 1.0);
  }
}

__global__ void DChi2(cufftComplex *noise, cufftComplex *atten, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY)
{

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

  float x = (i - (int)xobs) * DELTAX * RPDEG;
  float y = (j - (int)yobs) * DELTAY * RPDEG;

	float Ukv;
	float Vkv;

	float cosk;
	float sink;

  float dchi2 = 0.0;
  if(noise[N*i+j].x <= noise_cut){
  	for(int v=0; v<numVisibilities; v++){
      Ukv = x * U[v];
  		Vkv = y * V[v];
      #if (__CUDA_ARCH__ >= 300 )
        sincospif(2.0*(Ukv+Vkv), &sink, &cosk);
      #else
        cosk = cospif(2.0*(Ukv+Vkv));
        sink = sinpif(2.0*(Ukv+Vkv));
      #endif
      dchi2 += w[v]*((Vr[v].x * cosk) - (Vr[v].y * sink));
  	}

  dchi2 *= fg_scale * atten[N*i+j].x;
  dChi2[N*i+j] = dchi2;
  }
}

__global__ void DChi2_total(float *dchi2_total, float *dchi2, long N)
{

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

  dchi2_total[N*i+j] += dchi2[N*i+j];
}

__global__ void DPhi(float *dphi, float *dchi2, float *dH, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

  dphi[N*i+j] = dchi2[N*i+j] + dH[N*i+j];
}

__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

  x[N*i+j] = xc[N*i+j].x - lambda*gc[N*i+j];
}

__global__ void projection(float *px, float *x, float MINPIX, long N){

  int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;


  if(INFINITY < x[N*i+j]){
    px[N*i+j] = INFINITY;
  }else{
    px[N*i+j] = x[N*i+j];
  }

  if(MINPIX > px[N*i+j]){
    px[N*i+j] = MINPIX;
  }else{
    px[N*i+j] = px[N*i+j];
  }
}

__global__ void normVectorCalculation(float *normVector, float *gc, long N){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

  normVector[N*i+j] = gc[N*i+j] * gc[N*i+j];
}

__global__ void copyImage(cufftComplex *p, float *device_xt, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

  p[N*i+j].x = device_xt[N*i+j];
}

__host__ float chiCuadrado(cufftComplex *I)
{
  printf("**************Calculating phi - Iteration %d **************\n", iter);
  float resultPhi = 0.0;
  float resultchi2  = 0.0;
  float resultH  = 0.0;
  cudaEvent_t start, stop;
	float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  clip<<<numBlocksNN, threadsPerBlockNN>>>(I, MINPIX, N);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  //printf("CUDA clipping time = %f ms\n",time);
  global_time = global_time + time;

  //ACA SE HACE UNA ASIGNACION DE FG_IMAGE = P
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  clipWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_fg_image, device_noise_image, I, N, noise_cut, MINPIX);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  //printf("CUDA clipping noise time = %f ms\n",time);
  global_time = global_time + time;

  if(iter>=1 && MINPIX!=0.0){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    HVector<<<numBlocksNN, threadsPerBlockNN>>>(device_H, device_noise_image, device_fg_image, N, noise_cut, MINPIX);
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    //printf("CUDA HVector time = %f ms\n",time);
    global_time = global_time + time;
  }

  if(num_gpus == 1){
    for(int i=0; i<data.total_frequencies;i++){

      cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
    	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(device_image, device_fg_image, N, global_xobs, global_yobs, fg_scale, visibilities[i].freq, DELTAX, DELTAY);
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA applybeam execution time = %f ms\n",time);
      global_time = global_time + time;


    	//FFT

    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
    	if ((cufftExecC2C(plan1GPU, (cufftComplex*)device_image, (cufftComplex*)device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
    		printf("CUFFT exec error\n");
    		goToError();
    	}
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA FFT execution time = %f ms\n",time);
      global_time = global_time + time;



      //PHASE_ROTATE VISIBILITIES
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
      phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_V, M, N, global_xobs, global_yobs);
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA phase_rotate time = %f ms\n",time);
      global_time = global_time + time;

      //RESIDUAL CALCULATION

    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
      residual<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_visibilities[i].Vr, device_visibilities[i].Vo, device_V, device_visibilities[i].u, device_visibilities[i].v, deltau, deltav, data.numVisibilitiesPerFreq[i], N);
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA residual time = %f ms\n",time);
      global_time = global_time + time;

    	//CALCULATING chi2 VECTOR AND H VECTOR

    	////chi 2 VECTOR
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
    	chi2Vector<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_vars[i].chi2, device_visibilities[i].Vr, device_visibilities[i].weight, data.numVisibilitiesPerFreq[i]);
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA chi2Vector time = %f ms\n",time);
      global_time = global_time + time;

    	//REDUCTIONS
    	//chi2
    	resultchi2  += deviceReduce(device_vars[i].chi2, data.numVisibilitiesPerFreq[i]);
      //printf("resultchi2: %f\n", resultchi2);
    }
  }else{
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < data.total_frequencies; i++)
		{
      float result = 0.0;
      unsigned int j = omp_get_thread_num();
			//unsigned int num_cpu_threads = omp_get_num_threads();
			// set and check the CUDA device for this CPU thread
			int gpu_id = -1;
			cudaSetDevice(i % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
			cudaGetDevice(&gpu_id);
			//printf("CPU thread %d takes frequency %d and uses CUDA device %d\n", j, i, gpu_id);

      cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
    	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(device_vars[i].device_image, device_fg_image, N, global_xobs, global_yobs, fg_scale, visibilities[i].freq, DELTAX, DELTAY);
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA applybeam execution time = %f ms\n",time);
      global_time = global_time + time;


    	//FFT

    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
    	if ((cufftExecC2C(device_vars[i].plan, (cufftComplex*)device_vars[i].device_image, (cufftComplex*)device_vars[i].device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
    		printf("CUFFT exec error\n");
    		//return -1 ;
    		goToError();
    	}
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA FFT execution time = %f ms\n",time);
      global_time = global_time + time;



      //PHASE_ROTATE VISIBILITIES
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
      phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_vars[i].device_V, M, N, global_xobs, global_yobs);
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA phase_rotate time = %f ms\n",time);
      global_time = global_time + time;

      //RESIDUAL CALCULATION

    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
      residual<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_visibilities[i].Vr, device_visibilities[i].Vo, device_V, device_visibilities[i].u, device_visibilities[i].v, deltau, deltav, data.numVisibilitiesPerFreq[i], N);
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA residual time = %f ms\n",time);
      global_time = global_time + time;

    	//CALCULATING chi2 VECTOR AND H VECTOR

    	////chi 2 VECTOR
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	cudaEventRecord(start, 0);
    	chi2Vector<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_vars[i].chi2, device_visibilities[i].Vr, device_visibilities[i].weight, data.numVisibilitiesPerFreq[i]);
    	gpuErrchk(cudaDeviceSynchronize());
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	//printf("CUDA chi2Vector time = %f ms\n",time);
      global_time = global_time + time;

      result = deviceReduce(device_vars[i].chi2, data.numVisibilitiesPerFreq[i]);
    	//REDUCTIONS
    	//chi2
      #pragma omp critical
      {
        resultchi2  += result;
      }

    }
  }
    cudaSetDevice(0);
    resultH  = deviceReduce(device_H, M*N);
    resultPhi = (0.5 * resultchi2) + (lambda * resultH);
    printf("chi2 value = %.5f\n", resultchi2);
    printf("H value = %.5f\n", resultH);
    printf("(1/2) * chi2 value = %.5f\n", 0.5*resultchi2);
    printf("lambda * H value = %.5f\n", lambda*resultH);
    printf("Phi value = %.5f\n\n", resultPhi);

  	return resultPhi;
}



__host__ void dchiCuadrado(cufftComplex *I, float *dxi2)
{
	printf("**************Calculating dphi - Iteration %d *************\n", iter);
	cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  clip<<<numBlocksNN, threadsPerBlockNN>>>(I, MINPIX, N);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  //printf("CUDA clipping time = %f ms\n",time);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  restartDPhi<<<numBlocksNN, threadsPerBlockNN>>>(device_dphi, device_dchi2_total, device_H, N);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  toFitsFloat(I, iter, M, N, 1);
  //toFitsFloat(device_V, iter, M, N, 2);

  if(iter >= 1 && MINPIX!=0.0){

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    DH<<<numBlocksNN, threadsPerBlockNN>>>(device_dH, I, device_noise_image, noise_cut, lambda, MINPIX, N);
    gpuErrchk(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
  }

  if(num_gpus == 1){
    for(int i=0; i<data.total_frequencies;i++){
        cudaEventCreate(&start);
      	cudaEventCreate(&stop);
      	cudaEventRecord(start, 0);
        DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_vars[i].atten, device_vars[i].dchi2, device_visibilities[i].Vr, device_visibilities[i].u, device_visibilities[i].v, device_visibilities[i].weight, N, data.numVisibilitiesPerFreq[i], fg_scale, noise_cut, global_xobs, global_yobs, DELTAX, DELTAY);
      	gpuErrchk(cudaDeviceSynchronize());
      	cudaEventRecord(stop, 0);
      	cudaEventSynchronize(stop);
      	cudaEventElapsedTime(&time, start, stop);

        cudaEventCreate(&start);
      	cudaEventCreate(&stop);
      	cudaEventRecord(start, 0);
        DChi2_total<<<numBlocksNN, threadsPerBlockNN>>>(device_dchi2_total, device_vars[i].dchi2, N);
      	gpuErrchk(cudaDeviceSynchronize());
      	cudaEventRecord(stop, 0);
      	cudaEventSynchronize(stop);
      	cudaEventElapsedTime(&time, start, stop);
    }
  }else{
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < data.total_frequencies; i++)
    {
      unsigned int j = omp_get_thread_num();
      //unsigned int num_cpu_threads = omp_get_num_threads();
      // set and check the CUDA device for this CPU thread
      int gpu_id = -1;
      cudaSetDevice(i % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
      cudaGetDevice(&gpu_id);
      //printf("CPU thread %d takes frequency %d and uses CUDA device %d\n", j, i, gpu_id);
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
      DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_vars[i].atten, device_vars[i].dchi2, device_visibilities[i].Vr, device_visibilities[i].u, device_visibilities[i].v, device_visibilities[i].weight, N, data.numVisibilitiesPerFreq[i], fg_scale, noise_cut, global_xobs, global_yobs, DELTAX, DELTAY);
      gpuErrchk(cudaDeviceSynchronize());
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);

      #pragma omp critical
      {
        cudaEventCreate(&stop);
        cudaEventCreate(&start);
        cudaEventRecord(start, 0);
        DChi2_total<<<numBlocksNN, threadsPerBlockNN>>>(device_dchi2_total, device_vars[i].dchi2, N);
        gpuErrchk(cudaDeviceSynchronize());
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
      }

    }
  }

  cudaSetDevice(0);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  DPhi<<<numBlocksNN, threadsPerBlockNN>>>(device_dphi, device_dchi2_total, device_dH, N);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);


  //dxi2 = device_dphi;
  gpuErrchk(cudaMemcpy2D(dxi2, sizeof(float), device_dphi, sizeof(float), sizeof(float), M*N, cudaMemcpyDeviceToDevice));

}
