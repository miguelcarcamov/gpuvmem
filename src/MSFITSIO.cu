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

   Additionally, this program uses some NVIDIA routines whose copyright is held
   by NVIDIA end user license agreement (EULA).

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

#include "MSFITSIO.cuh"

__host__ MSData countVisibilities(char const* MS_name, Field *&fields, int gridding)
{
        MSData freqsAndVisibilities;
        std::string dir(MS_name);

        casacore::Vector<double> pointing_ref;
        casacore::Vector<double> pointing_phs;
        casacore::Table main_tab(dir);
        casacore::Table field_tab(main_tab.keywordSet().asTable("FIELD"));
        casacore::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casacore::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));
        freqsAndVisibilities.nfields = field_tab.nrow();
        casacore::ROTableRow field_row(field_tab, casacore::stringToVector("REFERENCE_DIR,PHASE_DIR"));

        casacore::ROScalarColumn<casacore::Int> n_corr(polarization_tab,"NUM_CORR");
        freqsAndVisibilities.nstokes=n_corr(0);

        freqsAndVisibilities.corr_type = (int*)malloc(freqsAndVisibilities.nstokes*sizeof(int));

        casacore::ROArrayColumn<casacore::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casacore::Vector<int> polarizations;
        polarizations=correlation_col(0);

        for(int i=0; i<freqsAndVisibilities.nstokes; i++) {
                freqsAndVisibilities.corr_type[i] = polarizations[i];
        }

        fields = (Field*)malloc(freqsAndVisibilities.nfields*sizeof(Field));
        for(int f=0; f<freqsAndVisibilities.nfields; f++) {
                const casacore::TableRecord &values = field_row.get(f);
                pointing_ref = values.asArrayDouble("REFERENCE_DIR");
                pointing_phs = values.asArrayDouble("PHASE_DIR");

                fields[f].ref_ra = pointing_ref[0];
                fields[f].ref_dec = pointing_ref[1];

                fields[f].phs_ra = pointing_phs[0];
                fields[f].phs_dec = pointing_phs[1];
        }


        freqsAndVisibilities.nsamples = main_tab.nrow();
        if (freqsAndVisibilities.nsamples == 0) {
                printf("ERROR : nsamples is zero... exiting....\n");
                exit(-1);
        }

        casacore::ROArrayColumn<casacore::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ"); //NUMBER OF SPW
        freqsAndVisibilities.n_internal_frequencies = spectral_window_tab.nrow();

        freqsAndVisibilities.channels = (int*)malloc(freqsAndVisibilities.n_internal_frequencies*sizeof(int));
        casacore::ROScalarColumn<casacore::Int> n_chan_freq(spectral_window_tab,"NUM_CHAN");
        for(int i = 0; i < freqsAndVisibilities.n_internal_frequencies; i++) {
                freqsAndVisibilities.channels[i] = n_chan_freq(i);
        }

        // We consider all chans .. The data will be processed this way later.

        int total_frequencies = 0;
        for(int i=0; i <freqsAndVisibilities.n_internal_frequencies; i++) {
                for(int j=0; j < freqsAndVisibilities.channels[i]; j++) {
                        total_frequencies++;
                }
        }

        freqsAndVisibilities.total_frequencies = total_frequencies;

        for(int f=0; f < freqsAndVisibilities.nfields; f++) {
                fields[f].numVisibilitiesPerFreqPerStoke = (long**)malloc(freqsAndVisibilities.total_frequencies*sizeof(long*));
                fields[f].numVisibilitiesPerFreq = (long*)malloc(freqsAndVisibilities.total_frequencies*sizeof(long));
                memset(fields[f].numVisibilitiesPerFreq, 0, freqsAndVisibilities.total_frequencies*sizeof(long));
                if(gridding) {
                        fields[f].backup_numVisibilitiesPerFreqPerStoke = (long **) malloc(freqsAndVisibilities.total_frequencies * sizeof(long *));
                        fields[f].backup_numVisibilitiesPerFreq = (long*)malloc(freqsAndVisibilities.total_frequencies*sizeof(long));
                        memset(fields[f].backup_numVisibilitiesPerFreq, 0, freqsAndVisibilities.total_frequencies*sizeof(long));
                }

                for(int i = 0; i < freqsAndVisibilities.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreqPerStoke[i] = (long*)malloc(freqsAndVisibilities.nstokes*sizeof(long));
                        memset(fields[f].numVisibilitiesPerFreqPerStoke[i], 0, freqsAndVisibilities.nstokes*sizeof(long));
                        if(gridding) {
                                fields[f].backup_numVisibilitiesPerFreqPerStoke[i] = (long *) malloc(
                                        freqsAndVisibilities.nstokes * sizeof(long));
                                memset(fields[f].backup_numVisibilitiesPerFreqPerStoke[i], 0,
                                       freqsAndVisibilities.nstokes * sizeof(long));
                        }
                }
        }

        casacore::Vector<float> weights;
        casacore::Matrix<bool> flagCol;

        int counter;
        std::string query;

        // Iteration through all fields

        for(int f=0; f<freqsAndVisibilities.nfields; f++) {
                counter = 0;
                for(int i=0; i < freqsAndVisibilities.n_internal_frequencies; i++) {

                        flagCol.resize(freqsAndVisibilities.nstokes, freqsAndVisibilities.channels[i]);
                        // Query for data with forced IF and FIELD
                        query = "select WEIGHT,FLAG from "+ dir +" where DATA_DESC_ID="+std::to_string(i)+" and FIELD_ID="+std::to_string(f)+" and !FLAG_ROW";

                        casacore::Table query_tab = casacore::tableCommand(query.c_str());

                        casacore::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casacore::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                weights=weight_col(k);
                                flagCol = flag_data_col(k);
                                for(int j=0; j < freqsAndVisibilities.channels[i]; j++) {
                                        for (int sto=0; sto<freqsAndVisibilities.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        fields[f].numVisibilitiesPerFreqPerStoke[counter+j][sto]++;
                                                        fields[f].numVisibilitiesPerFreq[counter+j]++;
                                                }
                                        }
                                }
                        }
                        counter+=freqsAndVisibilities.channels[i];
                }
        }


        int local_max = 0;
        int max = 0;
        for(int f=0; f < freqsAndVisibilities.nfields; f++) {
                for(int i=0; i<freqsAndVisibilities.total_frequencies; i++) {
                        local_max = *std::max_element(fields[f].numVisibilitiesPerFreqPerStoke[i],fields[f].numVisibilitiesPerFreqPerStoke[i]+freqsAndVisibilities.nstokes);
                        if(local_max > max) {
                                max = local_max;
                        }
                }
        }

        freqsAndVisibilities.max_number_visibilities_in_channel_and_stokes = max;

        return freqsAndVisibilities;
}


__host__ canvasVariables readCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag)
{
        status_canvas = 0;
        int status_noise = 0;

        canvasVariables c_vars;

        fits_open_file(&canvas, canvas_name, 0, &status_canvas);
        if (status_canvas) {
                fits_report_error(stderr, status_canvas); /* print error message */
                exit(0);
        }

        fits_read_key(canvas, TDOUBLE, "CDELT1", &c_vars.DELTAX, NULL, &status_canvas);
        fits_read_key(canvas, TDOUBLE, "CDELT2", &c_vars.DELTAY, NULL, &status_canvas);
        fits_read_key(canvas, TDOUBLE, "CRVAL1", &c_vars.ra, NULL, &status_canvas);
        fits_read_key(canvas, TDOUBLE, "CRVAL2", &c_vars.dec, NULL, &status_canvas);
        fits_read_key(canvas, TDOUBLE, "CRPIX1", &c_vars.crpix1, NULL, &status_canvas);
        fits_read_key(canvas, TDOUBLE, "CRPIX2", &c_vars.crpix2, NULL, &status_canvas);
        fits_read_key(canvas, TLONG, "NAXIS1", &c_vars.M, NULL, &status_canvas);
        fits_read_key(canvas, TLONG, "NAXIS2", &c_vars.N, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "BMAJ", &c_vars.beam_bmaj, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "BMIN", &c_vars.beam_bmin, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "NOISE", &c_vars.beam_noise, NULL, &status_noise);


        if (status_canvas) {
                fits_report_error(stderr, status_canvas); /* print error message */
                exit(0);
        }

        if(status_noise) {
                c_vars.beam_noise = b_noise_aux;
        }

        c_vars.beam_bmaj = c_vars.beam_bmaj/ fabs(c_vars.DELTAX);
        c_vars.beam_bmin = c_vars.beam_bmin/ c_vars.DELTAY;
        c_vars.DELTAX = fabs(c_vars.DELTAX);
        c_vars.DELTAY *= -1.0;

        if(verbose_flag) {
                printf("FITS Files READ\n");
        }

        return c_vars;
}

__host__ void readFITSImageValues(char *imageName, fitsfile *file, float *&values, int status, long M, long N)
{

        int anynull;
        long fpixel = 1;
        float null = 0.;
        long elementsImage = M*N;

        values = (float*)malloc(M*N*sizeof(float));
        fits_open_file(&file, imageName, 0, &status);
        fits_read_img(file, TFLOAT, fpixel, elementsImage, &null, values, &anynull, &status);

}

__host__ cufftComplex addNoiseToVis(cufftComplex vis, float weights){
        cufftComplex noise_vis;

        float real_n = Normal(0,1);
        float imag_n = Normal(0,1);

        noise_vis.x = vis.x + real_n * (1/sqrt(weights));
        noise_vis.y = vis.y + imag_n * (1/sqrt(weights));

        return noise_vis;
}

__host__ void readMS(char const *MS_name, Field *fields, MSData data, bool noise, bool W_projection, float random_prob)
{

        char *error = 0;
        int g = 0, h = 0;
        long c;

        std::string dir(MS_name);
        casacore::Table main_tab(dir);

        std::string data_column;

        if (main_tab.tableDesc().isColumn("CORRECTED"))
                data_column="CORRECTED";
        else
                data_column="DATA";


        casacore::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));

        casacore::ROArrayColumn<casacore::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casacore::Vector<float> weights;
        casacore::Vector<double> uvw;
        casacore::Matrix<casacore::Complex> dataCol;
        casacore::Matrix<bool> flagCol;

        for(int f=0; f < data.nfields; f++)
                for(int i = 0; i < data.total_frequencies; i++)
                        memset(fields[f].numVisibilitiesPerFreqPerStoke[i], 0, data.nstokes*sizeof(long));

        float u;
        SelectStream(0);
        PutSeed(-1);
        std::string query;


        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {

                        dataCol.resize(data.nstokes, data.channels[i]);
                        flagCol.resize(data.nstokes, data.channels[i]);

                        query = "select UVW,WEIGHT,"+data_column+",FLAG from "+dir+" where DATA_DESC_ID="+std::to_string(i)+" and FIELD_ID="+std::to_string(f)+" and !FLAG_ROW";
                        if(W_projection && random_prob < 1.0)
                        {
                                query += " and RAND()<%f ORDERBY ASC UVW[2]";
                        }else if(W_projection) {
                                query += " ORDERBY ASC UVW[2]";
                        }else if(random_prob < 1.0) {
                                query += " and RAND()< "+std::to_string(random_prob);
                        }

                        casacore::Table query_tab = casacore::tableCommand(query.c_str());

                        casacore::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casacore::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casacore::ROArrayColumn<casacore::Complex> data_col(query_tab, data_column);
                        casacore::ROArrayColumn<bool> flag_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                weights = weight_col(k);
                                flagCol = flag_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto < data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false) {
                                                        c = fields[f].numVisibilitiesPerFreqPerStoke[g+j][sto];
                                                        fields[f].visibilities[g+j][sto].uvw[c].x = uvw[0];
                                                        fields[f].visibilities[g+j][sto].uvw[c].y = uvw[1];
                                                        fields[f].visibilities[g+j][sto].uvw[c].z = uvw[2];

                                                        fields[f].visibilities[g+j][sto].Vo[c].x = dataCol(sto,j).real();
                                                        fields[f].visibilities[g+j][sto].Vo[c].y = dataCol(sto,j).imag();

                                                        if(noise)
                                                                fields[f].visibilities[g+j][sto].Vo[c] = addNoiseToVis(fields[f].visibilities[g+j][sto].Vo[c], weights[sto]);


                                                        fields[f].visibilities[g+j][sto].weight[c] = weights[sto];
                                                        fields[f].numVisibilitiesPerFreqPerStoke[g+j][sto]++;
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                }
        }


        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i<data.total_frequencies; i++) {
                        for (int sto=0; sto<data.nstokes; sto++) {
                                fields[f].numVisibilitiesPerFreq[i] += fields[f].numVisibilitiesPerFreqPerStoke[i][sto];
                        }
                }
        }


        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casacore::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].nu[h] = chan_freq_vector[j];
                                h++;
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                fields[f].valid_frequencies = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        for(int j = 0; j < data.channels[i]; j++) {
                                if(fields[f].numVisibilitiesPerFreq[h] > 0) {
                                        fields[f].valid_frequencies++;
                                }
                                h++;
                        }
                }
        }


}

__host__ void MScopy(char const *in_dir, char const *in_dir_dest)
{
        string dir_origin = in_dir;
        string dir_dest = in_dir_dest;

        casacore::Table tab_src(dir_origin);
        tab_src.deepCopy(dir_dest,casacore::Table::New);
}



__host__ void residualsToHost(Field *fields, MSData data, int num_gpus, int firstgpu)
{

        if(num_gpus == 1) {
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                for(int s=0; s<data.nstokes; s++) {
                                        gpuErrchk(cudaMemcpy(fields[f].visibilities[i][s].Vm, fields[f].device_visibilities[i][s].Vm,
                                                             sizeof(cufftComplex) * fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                                                             cudaMemcpyDeviceToHost));
                                        gpuErrchk(cudaMemcpy(fields[f].visibilities[i][s].weight,
                                                             fields[f].device_visibilities[i][s].weight,
                                                             sizeof(float) * fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                                                             cudaMemcpyDeviceToHost));
                                }
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                cudaSetDevice((i%num_gpus) + firstgpu);
                                for(int s=0; s<data.nstokes; s++) {
                                        gpuErrchk(cudaMemcpy(fields[f].visibilities[i][s].Vm, fields[f].device_visibilities[i][s].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreqPerStoke[i][s], cudaMemcpyDeviceToHost));
                                        gpuErrchk(cudaMemcpy(fields[f].visibilities[i][s].weight, fields[f].device_visibilities[i][s].weight, sizeof(float)*fields[f].numVisibilitiesPerFreqPerStoke[i][s], cudaMemcpyDeviceToHost));
                                }
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i<data.total_frequencies; i++) {
                        for(int s=0; s<data.nstokes; s++) {
                                for (int j = 0; j < fields[f].numVisibilitiesPerFreqPerStoke[i][s]; j++) {
                                        if (fields[f].visibilities[i][s].uvw[j].x < 0)
                                                fields[f].visibilities[i][s].Vm[j].y *= -1;
                                }
                        }
                }
        }

}

__host__ void writeMS(char const *outfile, char const *out_col, Field *fields, MSData data, float random_probability, bool sim, bool noise, bool W_projection, int verbose_flag)
{
        std::string dir = outfile;
        casacore::Table main_tab(dir,casacore::Table::Update);
        std::string column_name(out_col);
        std::string query;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                printf("Column %s already exists... skipping creation...\n", out_col);
        }else{
                printf("Adding %s to the main table...\n", out_col);
                main_tab.addColumn(casacore::ArrayColumnDesc <casacore::Complex>(column_name,"created by gpuvmem"));
                query = "UPDATE "+dir+" SET "+column_name+"=DATA";
                //query = "COPY COLUMN DATA TO MODEL";
                printf("Duplicating DATA column into %s ...\n", column_name.c_str());
                casacore::tableCommand(query.c_str());
                main_tab.flush();

        }


        for(int f=0; f < data.nfields; f++)
                for(int i = 0; i < data.total_frequencies; i++)
                        memset(fields[f].numVisibilitiesPerFreqPerStoke[i], 0, data.nstokes*sizeof(long));



        int g = 0;
        long c;
        cufftComplex vis;
        casacore::Vector<float> weights;
        casacore::Matrix<casacore::Complex> dataCol;
        casacore::Matrix<bool> flagCol;

        float real_n, imag_n;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {

                        dataCol.resize(data.nstokes, data.channels[i]);
                        flagCol.resize(data.nstokes, data.channels[i]);

                        query = "select WEIGHT,"+column_name+",FLAG from "+dir+" where DATA_DESC_ID="+std::to_string(i)+" and FIELD_ID="+std::to_string(f)+" and !FLAG_ROW";

                        if(W_projection)
                                query += " ORDERBY ASC UVW[2]";

                        casacore::Table query_tab = casacore::tableCommand(query.c_str());

                        casacore::ArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casacore::ArrayColumn<casacore::Complex> data_col(query_tab, column_name);
                        casacore::ArrayColumn<bool> flag_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                weights = weight_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto < data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        c = fields[f].numVisibilitiesPerFreqPerStoke[g+j][sto];

                                                        if(sim && noise)
                                                                vis = addNoiseToVis(fields[f].visibilities[g+j][sto].Vm[c], weights[sto]);
                                                        else if(sim)
                                                                vis = fields[f].visibilities[g+j][sto].Vm[c];
                                                        else{
                                                                vis.x = fields[f].visibilities[g+j][sto].Vo[c].x - fields[f].visibilities[g+j][sto].Vm[c].x;
                                                                vis.y = fields[f].visibilities[g+j][sto].Vo[c].y - fields[f].visibilities[g+j][sto].Vm[c].y;
                                                        }

                                                        dataCol(sto,j) = casacore::Complex(vis.x, vis.y);
                                                        weights[sto] = fields[f].visibilities[g+j][sto].weight[c];
                                                        fields[f].numVisibilitiesPerFreqPerStoke[g+j][sto]++;
                                                }
                                        }
                                }
                                data_col.put(k,dataCol);
                                weight_col.put(k, weights);
                        }

                        query_tab.flush();

                        string sub_query = "select from "+dir+" where DATA_DESC_ID="+std::to_string(i)+" and FIELD_ID="+std::to_string(f)+" and !FLAG_ROW";
                        if(W_projection)
                                sub_query += " ORDERBY ASC UVW[2]";

                        query = "update ["+sub_query+"], $1 tq set "+ column_name +"[!FLAG]=tq."+column_name+"[!tq.FLAG], WEIGHT=tq.WEIGHT";

                        casacore::TaQLResult result1 = casacore::tableCommand(query.c_str(), query_tab);

                        g+=data.channels[i];
                }
        }

        main_tab.flush();


}

__host__ void fitsOutputCufftComplex(cufftComplex *I, fitsfile *canvas, char *out_image, char *mempath, int iteration, float fg_scale, long M, long N, int option)
{
        fitsfile *fpointer;
        int status = 0;
        long fpixel = 1;
        long elements = M*N;
        size_t needed;
        char *name;
        long naxes[2]={M,N};
        long naxis = 2;
        char *unit = "JY/PIXEL";

        switch(option) {
        case 0:
                needed = snprintf(NULL, 0, "!%s", out_image) + 1;
                name = (char*)malloc(needed*sizeof(char));
                snprintf(name, needed*sizeof(char), "!%s", out_image);
                break;
        case 1:
                needed = snprintf(NULL, 0, "!%sMEM_%d.fits", mempath, iteration) + 1;
                name = (char*)malloc(needed*sizeof(char));
                snprintf(name, needed*sizeof(char), "!%sMEM_%d.fits", mempath, iteration);
                break;
        case -1:
                break;
        default:
                printf("Invalid case to FITS\n");
                exit(-1);
        }

        fits_create_file(&fpointer, name, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }
        fits_copy_header(canvas, fpointer, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }

        fits_update_key(fpointer, TSTRING, "BUNIT", unit, "Unit of measurement", &status);
        fits_update_key(fpointer, TINT, "NITER", &iteration, "Number of iteration in gpuvmem software", &status);


        cufftComplex *host_IFITS;
        host_IFITS = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));
        float *image2D = (float*) malloc(M*N*sizeof(float));
        gpuErrchk(cudaMemcpy2D(host_IFITS, sizeof(cufftComplex), I, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToHost));


        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        /*Absolute*/
                        image2D[N*i+j] = sqrt(host_IFITS[N*i+j].x * host_IFITS[N*i+j].x + host_IFITS[N*i+j].y * host_IFITS[N*i+j].y)* fg_scale;
                        /*Real part*/
                        //image2D[N*i+j] = host_IFITS[N*i+j].y;
                        /*Imaginary part*/
                        //image2D[N*i+j] = host_IFITS[N*i+j].y;
                }
        }

        fits_write_img(fpointer, TFLOAT, fpixel, elements, image2D, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }
        fits_close_file(fpointer, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }

        free(host_IFITS);
        free(image2D);
        free(name);
}

__host__ void OFITS(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N)
{
        fitsfile *fpointer;
        int status = 0;
        long fpixel = 1;
        long elements = M*N;
        size_t needed;
        long naxes[2]={M,N};
        long naxis = 2;
        char *full_name;

        needed = snprintf(NULL, 0, "!%s%s", path, name_image) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "!%s%s", path, name_image);

        fits_create_file(&fpointer, full_name, &status);
        if(status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }

        fits_copy_header(canvas, fpointer, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }

        fits_update_key(fpointer, TSTRING, "BUNIT", units, "Unit of measurement", &status);
        fits_update_key(fpointer, TINT, "NITER", &iteration, "Number of iteration in gpuvmem software", &status);

        float *host_IFITS = (float*)malloc(M*N*sizeof(float));

        //unsigned int offset = M*N*index*sizeof(float);
        int offset = M*N*index;
        gpuErrchk(cudaMemcpy(host_IFITS, &I[offset], sizeof(float)*M*N, cudaMemcpyDeviceToHost));

        for(int i=0; i<M; i++) {
                for(int j=0; j<N; j++) {
                        host_IFITS[N*i+j] *= fg_scale;
                }
        }

        fits_write_img(fpointer, TFLOAT, fpixel, elements, host_IFITS, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }
        fits_close_file(fpointer, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }

        free(host_IFITS);
}

__host__ void float2toImage(float *I, fitsfile *canvas, char *out_image, char*mempath, int iteration, float fg_scale, long M, long N, int option)
{
        fitsfile *fpointerI_nu_0, *fpointeralpha, *fpointer;
        int statusI_nu_0 = 0, statusalpha = 0;
        long fpixel = 1;
        long elements = M*N;
        char *Inu_0_name;
        char *alphaname;
        size_t needed_I_nu_0;
        size_t needed_alpha;
        long naxes[2]={M,N};
        long naxis = 2;
        char *alphaunit = "";
        char *I_unit = "JY/PIXEL";

        float *host_2Iout = (float*)malloc(M*N*sizeof(float)*2);

        gpuErrchk(cudaMemcpy(host_2Iout, I, sizeof(float)*M*N*2, cudaMemcpyDeviceToHost));

        float *host_alpha = (float*)malloc(M*N*sizeof(float));
        float *host_I_nu_0 = (float*)malloc(M*N*sizeof(float));

        switch(option) {
        case 0:
                needed_alpha = snprintf(NULL, 0, "!%s_alpha.fits", out_image) + 1;
                alphaname = (char*)malloc(needed_alpha*sizeof(char));
                snprintf(alphaname, needed_alpha*sizeof(char), "!%s_alpha.fits", out_image);
                break;
        case 1:
                needed_alpha = snprintf(NULL, 0, "!%salpha_%d.fits", mempath, iteration) + 1;
                alphaname = (char*)malloc(needed_alpha*sizeof(char));
                snprintf(alphaname, needed_alpha*sizeof(char), "!%salpha_%d.fits", mempath, iteration);
                break;
        case 2:
                needed_alpha = snprintf(NULL, 0, "!%salpha_error.fits", out_image) + 1;
                alphaname = (char*)malloc(needed_alpha*sizeof(char));
                snprintf(alphaname, needed_alpha*sizeof(char), "!%salpha_error.fits", out_image);
                break;
        case -1:
                break;
        default:
                printf("Invalid case to FITS\n");
                exit(-1);
        }

        switch(option) {
        case 0:
                needed_I_nu_0 = snprintf(NULL, 0, "!%s_I_nu_0.fits", out_image) + 1;
                Inu_0_name = (char*)malloc(needed_I_nu_0*sizeof(char));
                snprintf(Inu_0_name, needed_I_nu_0*sizeof(char), "!%s_I_nu_0.fits", out_image);
                break;
        case 1:
                needed_I_nu_0 = snprintf(NULL, 0, "!%sI_nu_0_%d.fits", mempath, iteration) + 1;
                Inu_0_name = (char*)malloc(needed_I_nu_0*sizeof(char));
                snprintf(Inu_0_name, needed_I_nu_0*sizeof(char), "!%sI_nu_0_%d.fits", mempath, iteration);
                break;
        case 2:
                needed_I_nu_0 = snprintf(NULL, 0, "!%s_I_nu_0_error.fits", out_image) + 1;
                Inu_0_name = (char*)malloc(needed_I_nu_0*sizeof(char));
                snprintf(Inu_0_name, needed_I_nu_0*sizeof(char), "!%s_I_nu_0_error.fits", out_image);
        case -1:
                break;
        default:
                printf("Invalid case to FITS\n");
                exit(-1);
        }


        fits_create_file(&fpointerI_nu_0, Inu_0_name, &statusI_nu_0);
        fits_create_file(&fpointeralpha, alphaname, &statusalpha);

        if (statusI_nu_0 || statusalpha) {
                fits_report_error(stderr, statusI_nu_0);
                fits_report_error(stderr, statusalpha);
                exit(-1); /* print error message */
        }

        fits_copy_header(canvas, fpointerI_nu_0, &statusI_nu_0);
        fits_copy_header(canvas, fpointeralpha, &statusalpha);

        if (statusI_nu_0 || statusalpha) {
                fits_report_error(stderr, statusI_nu_0);
                fits_report_error(stderr, statusalpha);
                exit(-1); /* print error message */
        }

        fits_update_key(fpointerI_nu_0, TSTRING, "BUNIT", I_unit, "Unit of measurement", &statusI_nu_0);
        fits_update_key(fpointeralpha, TSTRING, "BUNIT", alphaunit, "Unit of measurement", &statusalpha);


        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        host_I_nu_0[N*i+j] = host_2Iout[N*i+j];
                        host_alpha[N*i+j] = host_2Iout[N*M+N*i+j];
                }
        }

        fits_write_img(fpointerI_nu_0, TFLOAT, fpixel, elements, host_I_nu_0, &statusI_nu_0);
        fits_write_img(fpointeralpha, TFLOAT, fpixel, elements, host_alpha, &statusalpha);

        if (statusI_nu_0 || statusalpha) {
                fits_report_error(stderr, statusI_nu_0);
                fits_report_error(stderr, statusalpha);
                exit(-1); /* print error message */
        }
        fits_close_file(fpointerI_nu_0, &statusI_nu_0);
        fits_close_file(fpointeralpha, &statusalpha);

        if (statusI_nu_0 || statusalpha) {
                fits_report_error(stderr, statusI_nu_0);
                fits_report_error(stderr, statusalpha);
                exit(-1); /* print error message */
        }

        free(host_I_nu_0);
        free(host_alpha);

        free(host_2Iout);

        free(alphaname);
        free(Inu_0_name);


}

__host__ void float3toImage(float3 *I, fitsfile *canvas, char *out_image, char*mempath, int iteration, long M, long N, int option)
{
        fitsfile *fpointerT, *fpointertau, *fpointerbeta, *fpointer;
        int statusT = 0, statustau = 0, statusbeta = 0;
        long fpixel = 1;
        long elements = M*N;
        char *Tname;
        char *tauname;
        char *betaname;
        size_t needed_T;
        size_t needed_tau;
        size_t needed_beta;
        long naxes[2]={M,N};
        long naxis = 2;
        char *Tunit = "K";
        char *tauunit = "";
        char *betaunit = "";

        float3 *host_3Iout = (float3*)malloc(M*N*sizeof(float3));

        gpuErrchk(cudaMemcpy2D(host_3Iout, sizeof(float3), I, sizeof(float3), sizeof(float3), M*N, cudaMemcpyDeviceToHost));

        float *host_T = (float*)malloc(M*N*sizeof(float));
        float *host_tau = (float*)malloc(M*N*sizeof(float));
        float *host_beta = (float*)malloc(M*N*sizeof(float));

        switch(option) {
        case 0:
                needed_T = snprintf(NULL, 0, "!%s_T.fits", out_image) + 1;
                Tname = (char*)malloc(needed_T*sizeof(char));
                snprintf(Tname, needed_T*sizeof(char), "!%s_T.fits", out_image);
                break;
        case 1:
                needed_T = snprintf(NULL, 0, "!%sT_%d.fits", mempath, iteration) + 1;
                Tname = (char*)malloc(needed_T*sizeof(char));
                snprintf(Tname, needed_T*sizeof(char), "!%sT_%d.fits", mempath, iteration);
                break;
        case -1:
                break;
        default:
                printf("Invalid case to FITS\n");
                exit(-1);
        }

        switch(option) {
        case 0:
                needed_tau = snprintf(NULL, 0, "!%s_tau_0.fits", out_image) + 1;
                tauname = (char*)malloc(needed_tau*sizeof(char));
                snprintf(tauname, needed_tau*sizeof(char), "!%s_tau_0.fits", out_image);
                break;
        case 1:
                needed_tau = snprintf(NULL, 0, "!%stau_0_%d.fits", mempath, iteration) + 1;
                tauname = (char*)malloc(needed_tau*sizeof(char));
                snprintf(tauname, needed_tau*sizeof(char), "!%stau_0_%d.fits", mempath, iteration);
                break;
        case -1:
                break;
        default:
                printf("Invalid case to FITS\n");
                exit(-1);
        }

        switch(option) {
        case 0:
                needed_beta = snprintf(NULL, 0, "!%s_beta.fits", out_image) + 1;
                betaname = (char*)malloc(needed_beta*sizeof(char));
                snprintf(betaname, needed_beta*sizeof(char), "!%s_beta.fits", out_image);
                break;
        case 1:
                needed_beta = snprintf(NULL, 0, "!%sbeta_%d.fits", mempath, iteration) + 1;
                betaname = (char*)malloc(needed_beta*sizeof(char));
                snprintf(betaname, needed_beta*sizeof(char), "!%sbeta_%d.fits", mempath, iteration);
                break;
        case -1:
                break;
        default:
                printf("Invalid case to FITS\n");
                exit(-1);
        }

        fits_create_file(&fpointerT, Tname, &statusT);
        fits_create_file(&fpointertau, tauname, &statustau);
        fits_create_file(&fpointerbeta, betaname, &statusbeta);

        if (statusT || statustau || statusbeta) {
                fits_report_error(stderr, statusT);
                fits_report_error(stderr, statustau);
                fits_report_error(stderr, statusbeta);
                exit(-1);
        }

        fits_copy_header(canvas, fpointerT, &statusT);
        fits_copy_header(canvas, fpointertau, &statustau);
        fits_copy_header(canvas, fpointerbeta, &statusbeta);

        if (statusT || statustau || statusbeta) {
                fits_report_error(stderr, statusT);
                fits_report_error(stderr, statustau);
                fits_report_error(stderr, statusbeta);
                exit(-1);
        }

        fits_update_key(fpointerT, TSTRING, "BUNIT", Tunit, "Unit of measurement", &statusT);
        fits_update_key(fpointertau, TSTRING, "BUNIT", tauunit, "Unit of measurement", &statustau);
        fits_update_key(fpointerbeta, TSTRING, "BUNIT", betaunit, "Unit of measurement", &statusbeta);

        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        host_T[N*i+j] = host_3Iout[N*i+j].x;
                        host_tau[N*i+j] = host_3Iout[N*i+j].y;
                        host_beta[N*i+j] = host_3Iout[N*i+j].z;
                }
        }

        fits_write_img(fpointerT, TFLOAT, fpixel, elements, host_T, &statusT);
        fits_write_img(fpointertau, TFLOAT, fpixel, elements, host_tau, &statustau);
        fits_write_img(fpointerbeta, TFLOAT, fpixel, elements, host_beta, &statusbeta);
        if (statusT || statustau || statusbeta) {
                fits_report_error(stderr, statusT);
                fits_report_error(stderr, statustau);
                fits_report_error(stderr, statusbeta);
                exit(-1);
        }
        fits_close_file(fpointerT, &statusT);
        fits_close_file(fpointertau, &statustau);
        fits_close_file(fpointerbeta, &statusbeta);
        if (statusT || statustau || statusbeta) {
                fits_report_error(stderr, statusT);
                fits_report_error(stderr, statustau);
                fits_report_error(stderr, statusbeta);
                exit(-1);
        }

        free(host_T);
        free(host_tau);
        free(host_beta);
        free(host_3Iout);

        free(betaname);
        free(tauname);
        free(Tname);

}

__host__ void closeCanvas(fitsfile *canvas)
{
        int status = 0;
        fits_close_file(canvas, &status);
        if(status) {
                fits_report_error(stderr, status);
                exit(-1);
        }
}
