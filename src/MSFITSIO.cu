#include "MSFITSIO.cuh"

__host__ freqData countVisibilities(char * MS_name, Field *&fields)
{
        freqData freqsAndVisibilities;
        string dir = MS_name;
        string query;
        casa::Vector<double> pointing_ref;
        casa::Vector<double> pointing_phs;
        casa::Table main_tab(dir);
        casa::Table field_tab(main_tab.keywordSet().asTable("FIELD"));
        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));
        freqsAndVisibilities.nfields = field_tab.nrow();
        casa::ROTableRow field_row(field_tab, casa::stringToVector("REFERENCE_DIR,PHASE_DIR"));

        fields = (Field*)malloc(freqsAndVisibilities.nfields*sizeof(Field));
        for(int f=0; f<freqsAndVisibilities.nfields; f++) {
                const casa::TableRecord &values = field_row.get(f);
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

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ"); //NUMBER OF SPW
        freqsAndVisibilities.n_internal_frequencies = spectral_window_tab.nrow();

        freqsAndVisibilities.channels = (int*)malloc(freqsAndVisibilities.n_internal_frequencies*sizeof(int));
        casa::ROScalarColumn<casa::Int> n_chan_freq(spectral_window_tab,"NUM_CHAN");
        for(int i = 0; i < freqsAndVisibilities.n_internal_frequencies; i++) {
                freqsAndVisibilities.channels[i] = n_chan_freq(i);
        }

        int total_frequencies = 0;
        for(int i=0; i <freqsAndVisibilities.n_internal_frequencies; i++) {
                for(int j=0; j < freqsAndVisibilities.channels[i]; j++) {
                        total_frequencies++;
                }
        }

        freqsAndVisibilities.total_frequencies = total_frequencies;
        for(int f=0; f < freqsAndVisibilities.nfields; f++) {
                fields[f].numVisibilitiesPerFreq = (long*)malloc(freqsAndVisibilities.total_frequencies*sizeof(long));
                for(int i = 0; i < freqsAndVisibilities.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        casa::ROScalarColumn<casa::Int> n_corr(polarization_tab,"NUM_CORR");
        freqsAndVisibilities.nstokes=n_corr(0);

        casa::ROScalarColumn<int> data_desc_id_col(main_tab,"DATA_DESC_ID");
        casa::ROScalarColumn<bool> flag_col(main_tab,"FLAG_ROW");
        casa::ROArrayColumn<float> weight_col(main_tab,"WEIGHT");
        casa::ROScalarColumn<int> field_id_col(main_tab,"FIELD_ID");
        casa::ROArrayColumn<bool> flag_data_col(main_tab,"FLAG");

        casa::Vector<float> weights;
        casa::Matrix<casa::Bool> flagCol;

        bool flag;
        int spw, field, counter;

        for(int f=0; f<freqsAndVisibilities.nfields; f++) {
                counter = 0;
                for(int i=0; i < freqsAndVisibilities.n_internal_frequencies; i++) {
                        for(int j=0; j < freqsAndVisibilities.channels[i]; j++) {
                                for (int k=0; k < freqsAndVisibilities.nsamples; k++) {
                                        flag=flag_col(k);
                                        field=field_id_col(k);
                                        spw=data_desc_id_col(k);
                                        flagCol=flag_data_col(k);
                                        weights=weight_col(k);
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto<freqsAndVisibilities.nstokes; sto++) {
                                                        if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                                fields[f].numVisibilitiesPerFreq[counter]++;
                                                        }
                                                }
                                        }else continue;
                                }
                                counter++;
                        }
                }
        }

        int local_max = 0;
        int max = 0;
        for(int f=0; f < freqsAndVisibilities.nfields; f++) {
                local_max = *std::max_element(fields[f].numVisibilitiesPerFreq,fields[f].numVisibilitiesPerFreq+total_frequencies);
                if(local_max > max) {
                        max = local_max;
                }
        }
        freqsAndVisibilities.max_number_visibilities_in_channel = max;

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
        fits_read_key(canvas, TDOUBLE, "BMAJ", &c_vars.beam_bmaj, NULL, &status_canvas);
        fits_read_key(canvas, TDOUBLE, "BMIN", &c_vars.beam_bmin, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "BPA", &c_vars.beam_bpa, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "NOISE", &c_vars.beam_noise, NULL, &status_noise);

        if (status_canvas) {
                fits_report_error(stderr, status_canvas); /* print error message */
                exit(0);
        }

        if(status_noise) {
                c_vars.beam_noise = b_noise_aux;
        }

        c_vars.beam_bmaj /= fabs(c_vars.DELTAX);
        c_vars.beam_bmin /= fabs(c_vars.DELTAY);
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

__host__ void readMSMCNoise(char *MS_name, Field *fields, freqData data)
{

        char *error = 0;
        int g = 0, h = 0;
        long c;
        char *query;
        string dir = MS_name;
        casa::Table main_tab(dir);
        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));

        casa::ROArrayColumn<casa::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casa::Vector<int> polarizations;
        polarizations=correlation_col(0);

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casa::Vector<float> weights;
        casa::Vector<double> uvw;
        casa::Matrix<casa::Complex> dataCol;
        casa::Matrix<casa::Bool> flagCol;

        bool flag;
        size_t needed;

        for(int f=0; f < data.nfields; f++) {
                for(int i = 0; i < data.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        float u;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<casa::Complex> data_col(query_tab,"DATA");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");
                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_data_col(k);
                                weights = weight_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto<data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        c = fields[f].numVisibilitiesPerFreq[g+j];
                                                        fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                        fields[f].visibilities[g+j].uvw[c].x = uvw[0];
                                                        fields[f].visibilities[g+j].uvw[c].y = uvw[1];
                                                        fields[f].visibilities[g+j].uvw[c].z = uvw[2];
                                                        u = Normal(0.0, 1.0);
                                                        fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real() + u * (1/sqrt(weights[sto]));
                                                        u = Normal(0.0, 1.0);
                                                        fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag() + u * (1/sqrt(weights[sto]));
                                                        fields[f].visibilities[g+j].weight[c] = weights[sto];
                                                        fields[f].numVisibilitiesPerFreq[g+j]++;
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                        free(query);
                }
        }


        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casa::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].visibilities[h].freq = chan_freq_vector[j];
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

__host__ void readSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability)
{
        char *error = 0;
        int g = 0, h = 0;
        long c;
        char *query;
        string dir = MS_name;
        casa::Table main_tab(dir);
        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));

        casa::ROArrayColumn<casa::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casa::Vector<int> polarizations;
        polarizations=correlation_col(0);

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casa::Vector<float> weights;
        casa::Vector<double> uvw;
        casa::Matrix<casa::Complex> dataCol;
        casa::Matrix<casa::Bool> flagCol;

        bool flag;
        size_t needed;

        for(int f=0; f < data.nfields; f++) {
                for(int i = 0; i < data.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        float u;
        SelectStream(0);
        PutSeed(-1);
        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<casa::Complex> data_col(query_tab,"DATA");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_data_col(k);
                                weights = weight_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto < data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        u = Random();
                                                        if(u<random_probability) {
                                                                c = fields[f].numVisibilitiesPerFreq[g+j];
                                                                fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                                fields[f].visibilities[g+j].uvw[c].x = uvw[0];
                                                                fields[f].visibilities[g+j].uvw[c].y = uvw[1];
                                                                fields[f].visibilities[g+j].uvw[c].z = uvw[2];
                                                                fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real();
                                                                fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag();
                                                                fields[f].visibilities[g+j].weight[c] = weights[sto];
                                                                fields[f].numVisibilitiesPerFreq[g+j]++;
                                                        }else{
                                                                c = fields[f].numVisibilitiesPerFreq[g+j];
                                                                fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                                fields[f].visibilities[g+j].uvw[c].x = uvw[0];
                                                                fields[f].visibilities[g+j].uvw[c].y = uvw[1];
                                                                fields[f].visibilities[g+j].uvw[c].z = uvw[2];
                                                                fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real();
                                                                fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag();
                                                                fields[f].visibilities[g+j].weight[c] = 0.0;
                                                                fields[f].numVisibilitiesPerFreq[g+j]++;
                                                        }
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                        free(query);
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casa::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].visibilities[h].freq = chan_freq_vector[j];
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

__host__ void readMCNoiseSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability)
{
        char *error = 0;
        int g = 0, h = 0;
        long c;
        char *query;
        string dir = MS_name;
        casa::Table main_tab(dir);

        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));

        casa::ROArrayColumn<casa::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casa::Vector<int> polarizations;
        polarizations=correlation_col(0);

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casa::Vector<float> weights;
        casa::Vector<double> uvw;
        casa::Matrix<casa::Complex> dataCol;
        casa::Matrix<casa::Bool> flagCol;

        bool flag;
        size_t needed;

        for(int f=0; f < data.nfields; f++) {
                for(int i = 0; i < data.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        float u;
        float nu;
        SelectStream(0);
        PutSeed(-1);
        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<casa::Complex> data_col(query_tab,"DATA");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_data_col(k);
                                weights = weight_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto < data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        u = Random();
                                                        if(u<random_probability) {
                                                                c = fields[f].numVisibilitiesPerFreq[g+j];
                                                                fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                                fields[f].visibilities[g+j].uvw[c].x = uvw[0];
                                                                fields[f].visibilities[g+j].uvw[c].y = uvw[1];
                                                                fields[f].visibilities[g+j].uvw[c].z = uvw[2];
                                                                nu = Normal(0.0, 1.0);
                                                                fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real() + u * (1/sqrt(weights[sto]));
                                                                nu = Normal(0.0, 1.0);
                                                                fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag() + u * (1/sqrt(weights[sto]));
                                                                fields[f].visibilities[g+j].weight[c] = weights[sto];
                                                                fields[f].numVisibilitiesPerFreq[g+j]++;
                                                        }else{
                                                                c = fields[f].numVisibilitiesPerFreq[g+j];
                                                                fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                                fields[f].visibilities[g+j].uvw[c].x = uvw[0];
                                                                fields[f].visibilities[g+j].uvw[c].y = uvw[1];
                                                                fields[f].visibilities[g+j].uvw[c].z = uvw[2];

                                                                fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real();
                                                                fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag();
                                                                fields[f].visibilities[g+j].weight[c] = 0.0;
                                                                fields[f].numVisibilitiesPerFreq[g+j]++;
                                                        }
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                        free(query);
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casa::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].visibilities[h].freq = chan_freq_vector[j];
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


__host__ void readMS(char *MS_name, Field *fields, freqData data)
{

        char *error = 0;
        int g = 0, h = 0;
        long c;
        char *query;
        string dir = MS_name;
        casa::Table main_tab(dir);

        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));

        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));
        casa::ROArrayColumn<casa::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casa::Vector<int> polarizations;
        polarizations=correlation_col(0);

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casa::Vector<float> weights;
        casa::Vector<double> uvw;
        casa::Matrix<casa::Complex> dataCol;
        casa::Matrix<casa::Bool> flagCol;
        bool flag;
        size_t needed;

        for(int f=0; f < data.nfields; f++) {
                for(int i = 0; i < data.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<casa::Complex> data_col(query_tab,"DATA");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_data_col(k);
                                weights = weight_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto < data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        c = fields[f].numVisibilitiesPerFreq[g+j];
                                                        fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                        fields[f].visibilities[g+j].uvw[c].x = uvw[0];
                                                        fields[f].visibilities[g+j].uvw[c].y = uvw[1];
                                                        fields[f].visibilities[g+j].uvw[c].z = uvw[2];
                                                        fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real();
                                                        fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag();
                                                        fields[f].visibilities[g+j].weight[c] = weights[sto];
                                                        fields[f].numVisibilitiesPerFreq[g+j]++;
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                        free(query);
                }
        }


        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casa::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].visibilities[h].freq = chan_freq_vector[j];
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

__host__ void MScopy(char const *in_dir, char const *in_dir_dest, int verbose_flag)
{
        string dir_origin = in_dir;
        string dir_dest = in_dir_dest;

        casa::Table tab_src(dir_origin);
        tab_src.deepCopy(dir_dest,casa::Table::New);
        if (verbose_flag) {
                printf("Copied\n");
        }

}



__host__ void residualsToHost(Field *fields, freqData data, int num_gpus, int firstgpu)
{
        printf("Saving residuals to host memory\n");
        if(num_gpus == 1) {
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                gpuErrchk(cudaMemcpy(fields[f].visibilities[i].Vm, fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyDeviceToHost));
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                cudaSetDevice((i%num_gpus) + firstgpu);
                                gpuErrchk(cudaMemcpy(fields[f].visibilities[i].Vm, fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyDeviceToHost));
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i<data.total_frequencies; i++) {
                        for(int j=0; j<fields[f].numVisibilitiesPerFreq[i]; j++) {
                                if(fields[f].visibilities[i].uvw[j].x < 0) {
                                        fields[f].visibilities[i].Vm[j].y *= -1;
                                }
                        }
                }
        }

}

__host__ void writeMS(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                printf("Column %s already exists... skipping creation...\n", out_col);
        }else{
                printf("Adding %s to the main table...\n", out_col);
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                printf("Duplicating DATA column into %s\n", out_col);
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vo[h].x - fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vo[h].y - fields[f].visibilities[g].Vm[h].y);
                                                                weights[sto] = fields[f].visibilities[g].weight[h];
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}


__host__ void writeMSSIM(char *infile, char *outfile, Field *fields, freqData data, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                printf("Column %s already exists... skipping creation...\n", out_col);
        }else{
                printf("Adding %s to the main table...\n", out_col);
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                printf("Duplicating DATA column into %s\n", out_col);
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vm[h].y);
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}

__host__ void writeMSSIMMC(char *infile, char *outfile, Field *fields, freqData data, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                printf("Column %s already exists... skipping creation...\n", out_col);
        }else{
                printf("Adding %s to the main table...\n", out_col);
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                printf("Duplicating DATA column into %s\n", out_col);
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        float real_n, imag_n;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                real_n = Normal(0.0, 1.0);
                                                                imag_n = Normal(0.0, 1.0);
                                                                dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x + real_n * (1/sqrt(weights[sto])), fields[f].visibilities[g].Vm[h].y + imag_n * (1/sqrt(weights[sto])));
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}

__host__ void writeMSSIMSubsampled(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                printf("Column %s already exists... skipping creation...\n", out_col);
        }else{
                printf("Adding %s to the main table...\n", out_col);
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                printf("Duplicating DATA column into %s\n", out_col);
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        float u;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                u = Random();
                                                                if(u<random_probability) {
                                                                        dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vm[h].y);
                                                                }else{
                                                                        dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vm[h].y);
                                                                        weights[sto] = 0.0;
                                                                }
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}


__host__ void writeMSSIMSubsampledMC(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                printf("Column %s already exists... skipping creation...\n", out_col);
        }else{
                printf("Adding %s to the main table...\n", out_col);
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                printf("Duplicating DATA column into %s\n", out_col);
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        float real_n, imag_n;
        float u;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                u = Random();
                                                                if(u<random_probability) {
                                                                        real_n = Normal(0.0, 1.0);
                                                                        imag_n = Normal(0.0, 1.0);
                                                                        dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x + real_n * (1/sqrt(weights[sto])), fields[f].visibilities[g].Vm[h].y + imag_n * (1/sqrt(weights[sto])));
                                                                }else{
                                                                        dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vm[h].y);
                                                                        weights[sto] = 0.0;
                                                                }
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
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
        gpuErrchk(cudaMemcpy2D(host_IFITS, sizeof(cufftComplex), I, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToHost));

        float* image2D;
        image2D = (float*) malloc(M*N*sizeof(float));

        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        image2D[N*i+j] = host_IFITS[N*i+j].x * fg_scale;
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

__host__ void fitsOutputFloat(float *I, fitsfile *canvas, char *mempath, int iteration, long M, long N, int option)
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
                needed = snprintf(NULL, 0, "!%satten_%d.fits", mempath, iteration) + 1;
                name = (char*)malloc(needed*sizeof(char));
                snprintf(name, needed*sizeof(char), "!%satten_%d.fits", mempath, iteration);
                break;
        case 1:
                needed = snprintf(NULL, 0, "!%snoise_%d.fits", mempath, iteration) + 1;
                name = (char*)malloc(needed*sizeof(char));
                snprintf(name, needed*sizeof(char), "!%snoise_%d.fits", mempath, iteration);
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
        if(option==0 || option==1) {
                fits_update_key(fpointer, TSTRING, "BUNIT", unit, "Unit of measurement", &status);
        }

        float *host_IFITS;
        host_IFITS = (float*)malloc(M*N*sizeof(float));
        gpuErrchk(cudaMemcpy2D(host_IFITS, sizeof(float), I, sizeof(float), sizeof(float), M*N, cudaMemcpyDeviceToHost));

        float* image2D;
        image2D = (float*) malloc(M*N*sizeof(float));

        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        image2D[N*i+j] = host_IFITS[N*i+j];
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

__host__ void float2toImage(float2 *I, fitsfile *canvas, char *out_image, char*mempath, int iteration, long M, long N, float fg_scale, int option)
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

        float2 *host_2Iout = (float2*)malloc(M*N*sizeof(float2));

        gpuErrchk(cudaMemcpy2D(host_2Iout, sizeof(float2), I, sizeof(float2), sizeof(float2), M*N, cudaMemcpyDeviceToHost));

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
                        host_I_nu_0[N*i+j] = host_2Iout[N*i+j].x * fg_scale;
                        host_alpha[N*i+j] = host_2Iout[N*i+j].y * fg_scale;
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

__host__ void double2toImage(double2 *I, fitsfile *canvas, char *out_image, char*mempath, int iteration, long M, long N, double fg_scale, int samples, int option)
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
        int bitpix_value = DOUBLE_IMG;

        double2 *host_2Iout = (double2*)malloc(M*N*sizeof(double2));

        gpuErrchk(cudaMemcpy2D(host_2Iout, sizeof(double2), I, sizeof(double2), sizeof(double2), M*N, cudaMemcpyDeviceToHost));

        double *host_alpha = (double*)malloc(M*N*sizeof(double));
        double *host_I_nu_0 = (double*)malloc(M*N*sizeof(double));

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

        fits_update_key(fpointerI_nu_0, TINT, "BITPIX", &bitpix_value, "", &statusI_nu_0);
        fits_update_key(fpointeralpha, TINT, "BITPIX", &bitpix_value, "", &statusalpha);

        fits_update_key(fpointerI_nu_0, TINT, "NITER", &samples, "Number of iteration in MCMC", &statusI_nu_0);
        fits_update_key(fpointeralpha, TINT, "NITER", &samples, "Number of iteration in MCMC", &statusalpha);


        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        host_I_nu_0[N*i+j] = host_2Iout[N*i+j].x * fg_scale;
                        host_alpha[N*i+j] = host_2Iout[N*i+j].y * fg_scale;
                }
        }

        fits_write_img(fpointerI_nu_0, TDOUBLE, fpixel, elements, host_I_nu_0, &statusI_nu_0);
        fits_write_img(fpointeralpha, TDOUBLE, fpixel, elements, host_alpha, &statusalpha);

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
