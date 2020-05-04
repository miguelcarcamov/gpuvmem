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


extern long M, N, numVisibilities;
extern int iterations, iterthreadsVectorNN, blocksVectorNN, nopositivity, \
           status_mod_in, verbose_flag, apply_noise, adaptive, clip_flag, num_gpus, selected, iter, t_telescope, multigpu, firstgpu, reg_term, print_images, checkpoint, spec_idx, use_mask;

extern cufftHandle plan1GPU;
extern cufftComplex *device_V, *device_Inu;

extern float2 *device_dphi, *device_2I;
extern float *device_mask, *device_chi2, *device_dchi2, *device_S, *device_S_alpha, *device_dS, *device_dS_alpha, *device_noise_image;
extern float noise_jypix, fg_scale, noise_cut, MINPIX, \
             minpix, lambda, ftol, random_probability, final_chi2, nu_0, final_H, alpha_start, eta, epsilon;

extern dim3 threadsPerBlockNN, numBlocksNN;

extern float beam_noise, beam_bmaj, beam_bmin, beam_bpa, b_noise_aux, antenna_diameter, pb_factor, pb_cutoff, *host_noise_image;
extern double ra, dec, crpix1, crpix2, DELTAX, DELTAY, deltau, deltav;

extern freqData data;

extern char* mempath, *out_image;

extern fitsfile *mod_in;

extern Field *fields;

extern int flag_opt, valid_pixels;

extern int2 *pixels;

extern VariablesPerField *vars_per_field;

extern varsPerGPU *vars_gpu;

extern char *checkp;

extern double2 *host_M_k;
extern double2 *host_Q_k;

extern float2 *temp;
extern double2 *M_k_out, *Q_k_out;

extern FILE *outfile;
extern FILE *outfile_its;
extern FILE *shutdown_file;

extern int position_in_file;

extern int accepted_afterburndown;

extern int real_iterations;

char *shutdown_file_name;

void sig_handler(int signo)
{
        if(signo == SIGKILL || signo == SIGTERM || signo == SIGINT) {
                shutdown_file= fopen(shutdown_file_name, "w");
                printf("--------------Iteration %d-----------\n", real_iterations);
                fseek(shutdown_file,position_in_file,SEEK_SET);
                fprintf(shutdown_file, "Iterations: %d\n", real_iterations);
                fprintf(shutdown_file, "Accepted after burndown: %d\n", accepted_afterburndown);
                fflush(shutdown_file);
                double2toImage(M_k_out, mod_in, out_image, checkp, 0, M, N, 1.0, accepted_afterburndown, 1);
                double2toImage(Q_k_out, mod_in, out_image, checkp, 1, M, N, 1/(accepted_afterburndown-1), accepted_afterburndown, 1);
                float2toImage(device_2I, mod_in, out_image, checkp, 2, M, N, 1.0, 1);
                fclose(outfile);
                fclose(outfile_its);
                fclose(shutdown_file);
                cudaFree(device_2I);
                cudaFree(Q_k_out);
                cudaFree(M_k_out);
                exit(0);
        }
}

__host__ void goToError()
{
        if(num_gpus > 1) {
                for(int i=firstgpu+1; i<firstgpu + num_gpus; i++) {
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

        printf("An error has ocurred, exiting\n");
        exit(0);

}

__host__ void init_beam(int telescope)
{
        switch(telescope) {
        case 1:
                antenna_diameter = 1.4; /* CBI2 Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 90.0*RPARCM; /* radians */
                break;
        case 2:
                antenna_diameter = 12.0; /* ALMA Antenna Diameter */
                pb_factor = 1.13; /* FWHM Factor */
                pb_cutoff = 1.0*RPARCM; /* radians */
                break;
        case 3:
                antenna_diameter = 22.0; /* ATCA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 1.0*RPARCM; /* radians */
                break;
        case 4:
                antenna_diameter = 25.0; /* VLA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 20.0*RPARCM; /* radians */
                break;
        case 5:
                antenna_diameter = 3.5; /* SZA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 20.0*RPARCM; /* radians */
                break;
        case 6:
                antenna_diameter = 0.9; /* CBI Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 20.0*RPARCM; /* radians */
                break;
        default:
                printf("Telescope type not defined\n");
                goToError();
                break;
        }
}


__host__ long NearestPowerOf2(long x)
{
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
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
        if((fp = fopen(file, "r")) == NULL) {
                printf("ERROR. The input file wasn't provided by the user.\n");
                goToError();
        }else{
                while(true) {
                        int ret = fscanf(fp, "%s %e", item, &status);

                        if(ret==EOF) {
                                break;
                        }else{
                                if (strcmp(item,"lambda_entropy")==0) {
                                        if(lambda == -1) {
                                                lambda = status;
                                        }
                                }else if (strcmp(item,"noise_cut")==0) {
                                        if(noise_cut == -1) {
                                                noise_cut = status;
                                        }
                                }else if (strcmp(item,"t_telescope")==0) {
                                        t_telescope = status;
                                }else if(strcmp(item,"minpix")==0) {
                                        if(minpix == -1) {
                                                minpix = status;
                                        }
                                } else if(strcmp(item,"ftol")==0) {
                                        ftol = status;
                                } else if(strcmp(item,"random_probability")==0) {
                                        if(random_probability == -1) {
                                                random_probability = status;
                                        }
                                }else{
                                        printf("Keyword not defined in input\n");
                                        goToError();
                                }
                        }
                }
        }
}

__host__ void print_warranty() {
        printf("THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY \
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT \
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM 'AS IS' WITHOUT WARRANTY \
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, \
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR \
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM \
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF \
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.\n");
}

__host__ void print_copyright() {
        printf("   TERMS AND CONDITIONS \n"
               " \n"
               "0. Definitions. \n"
               " \n"
               "'This License' refers to version 3 of the GNU General Public License. \n"
               " \n"
               "'Copyright' also means copyright-like laws that apply to other kinds of \n"
               "works, such as semiconductor masks. \n"
               " \n"
               "'The Program' refers to any copyrightable work licensed under this \n"
               "License.  Each licensee is addressed as 'you'.  'Licensees' and \n"
               "'recipients' may be individuals or organizations. \n"
               " \n"
               "To 'modify' a work means to copy from or adapt all or part of the work \n"
               "in a fashion requiring copyright permission, other than the making of an \n"
               "exact copy.  The resulting work is called a 'modified version' of the \n"
               "earlier work or a work 'based on' the earlier work. \n"
               " \n"
               "A 'covered work' means either the unmodified Program or a work based \n"
               "on the Program. \n"
               " \n"
               "To 'propagate' a work means to do anything with it that, without \n"
               "permission, would make you directly or secondarily liable for \n"
               "infringement under applicable copyright law, except executing it on a \n"
               "computer or modifying a private copy.  Propagation includes copying, \n"
               "distribution (with or without modification), making available to the \n"
               "public, and in some countries other activities as well. \n"
               " \n"
               "To 'convey' a work means any kind of propagation that enables other \n"
               "parties to make or receive copies.  Mere interaction with a user through \n"
               "a computer network, with no transfer of a copy, is not conveying. \n"
               " \n"
               "An interactive user interface displays 'Appropriate Legal Notices' \n"
               "to the extent that it includes a convenient and prominently visible \n"
               "feature that (1) displays an appropriate copyright notice, and (2) \n"
               "tells the user that there is no warranty for the work (except to the \n"
               "extent that warranties are provided), that licensees may convey the \n"
               "work under this License, and how to view a copy of this License.  If \n"
               "the interface presents a list of user commands or options, such as a \n"
               "menu, a prominent item in the list meets this criterion. \n"
               " \n"
               "1. Source Code. \n"
               " \n"
               "The 'source code' for a work means the preferred form of the work \n"
               "for making modifications to it.  'Object code' means any non-source \n"
               "form of a work. \n"
               " \n"
               "A 'Standard Interface' means an interface that either is an official \n"
               "standard defined by a recognized standards body, or, in the case of \n"
               "interfaces specified for a particular programming language, one that \n"
               "is widely used among developers working in that language. \n"
               " \n"
               "The 'System Libraries' of an executable work include anything, other \n"
               "than the work as a whole, that (a) is included in the normal form of \n"
               "packaging a Major Component, but which is not part of that Major \n"
               "Component, and (b) serves only to enable use of the work with that \n"
               "Major Component, or to implement a Standard Interface for which an \n"
               "implementation is available to the public in source code form.  A \n"
               "'Major Component', in this context, means a major essential component \n"
               "(kernel, window system, and so on) of the specific operating system \n"
               "(if any) on which the executable work runs, or a compiler used to \n"
               "produce the work, or an object code interpreter used to run it. \n"
               " \n"
               "The 'Corresponding Source' for a work in object code form means all \n"
               "the source code needed to generate, install, and (for an executable \n"
               "work) run the object code and to modify the work, including scripts to \n"
               "control those activities.  However, it does not include the work's \n"
               "System Libraries, or general-purpose tools or generally available free \n"
               "programs which are used unmodified in performing those activities but \n"
               "which are not part of the work.  For example, Corresponding Source \n"
               "includes interface definition files associated with source files for \n"
               "the work, and the source code for shared libraries and dynamically \n"
               "linked subprograms that the work is specifically designed to require, \n"
               "such as by intimate data communication or control flow between those \n"
               "subprograms and other parts of the work. \n"
               " \n"
               "The Corresponding Source need not include anything that users \n"
               "can regenerate automatically from other parts of the Corresponding \n"
               "Source. \n"
               " \n"
               "The Corresponding Source for a work in source code form is that \n"
               "same work. \n"
               " \n"
               "2. Basic Permissions. \n"
               " \n"
               "All rights granted under this License are granted for the term of \n"
               "copyright on the Program, and are irrevocable provided the stated \n"
               "conditions are met.  This License explicitly affirms your unlimited \n"
               "permission to run the unmodified Program.  The output from running a \n"
               "covered work is covered by this License only if the output, given its \n"
               "content, constitutes a covered work.  This License acknowledges your \n"
               "rights of fair use or other equivalent, as provided by copyright law. \n"
               " \n"
               "You may make, run and propagate covered works that you do not \n"
               "convey, without conditions so long as your license otherwise remains \n"
               "in force.  You may convey covered works to others for the sole purpose \n"
               "of having them make modifications exclusively for you, or provide you \n"
               "with facilities for running those works, provided that you comply with \n"
               "the terms of this License in conveying all material for which you do \n"
               "not control copyright.  Those thus making or running the covered works \n"
               "for you must do so exclusively on your behalf, under your direction \n"
               "and control, on terms that prohibit them from making any copies of \n"
               "your copyrighted material outside their relationship with you. \n"
               " \n"
               "Conveying under any other circumstances is permitted solely under \n"
               "the conditions stated below.  Sublicensing is not allowed; section 10 \n"
               "makes it unnecessary. \n"
               " \n"
               "3. Protecting Users' Legal Rights From Anti-Circumvention Law. \n"
               " \n"
               "No covered work shall be deemed part of an effective technological \n"
               "measure under any applicable law fulfilling obligations under article \n"
               "11 of the WIPO copyright treaty adopted on 20 December 1996, or \n"
               "similar laws prohibiting or restricting circumvention of such \n"
               "measures. \n"
               " \n"
               "When you convey a covered work, you waive any legal power to forbid \n"
               "circumvention of technological measures to the extent such circumvention \n"
               "is effected by exercising rights under this License with respect to \n"
               "the covered work, and you disclaim any intention to limit operation or \n"
               "modification of the work as a means of enforcing, against the work's \n"
               "users, your or third parties' legal rights to forbid circumvention of \n"
               "technological measures. \n"
               " \n"
               "4. Conveying Verbatim Copies. \n"
               " \n"
               "You may convey verbatim copies of the Program's source code as you \n"
               "receive it, in any medium, provided that you conspicuously and \n"
               "appropriately publish on each copy an appropriate copyright notice; \n"
               "keep intact all notices stating that this License and any \n"
               "non-permissive terms added in accord with section 7 apply to the code; \n"
               "keep intact all notices of the absence of any warranty; and give all \n"
               "recipients a copy of this License along with the Program. \n"
               " \n"
               "You may charge any price or no price for each copy that you convey, \n"
               "and you may offer support or warranty protection for a fee. \n"
               " \n"
               "5. Conveying Modified Source Versions. \n"
               " \n"
               "You may convey a work based on the Program, or the modifications to \n"
               "produce it from the Program, in the form of source code under the \n"
               "terms of section 4, provided that you also meet all of these conditions: \n"
               " \n"
               "a) The work must carry prominent notices stating that you modified \n"
               "it, and giving a relevant date. \n"
               " \n"
               "b) The work must carry prominent notices stating that it is \n"
               "released under this License and any conditions added under section \n"
               "7.  This requirement modifies the requirement in section 4 to \n"
               "'keep intact all notices'. \n"
               " \n"
               "c) You must license the entire work, as a whole, under this \n"
               "License to anyone who comes into possession of a copy.  This \n"
               "License will therefore apply, along with any applicable section 7 \n"
               "additional terms, to the whole of the work, and all its parts, \n"
               "regardless of how they are packaged.  This License gives no \n"
               "permission to license the work in any other way, but it does not \n"
               "invalidate such permission if you have separately received it. \n"
               " \n"
               "d) If the work has interactive user interfaces, each must display \n"
               "Appropriate Legal Notices; however, if the Program has interactive \n"
               "interfaces that do not display Appropriate Legal Notices, your \n"
               "work need not make them do so. \n"
               " \n"
               "A compilation of a covered work with other separate and independent \n"
               "works, which are not by their nature extensions of the covered work, \n"
               "and which are not combined with it such as to form a larger program, \n"
               "in or on a volume of a storage or distribution medium, is called an \n"
               "'aggregate' if the compilation and its resulting copyright are not \n"
               "used to limit the access or legal rights of the compilation's users \n"
               "beyond what the individual works permit.  Inclusion of a covered work \n"
               "in an aggregate does not cause this License to apply to the other \n"
               "parts of the aggregate. \n"
               " \n"
               "6. Conveying Non-Source Forms. \n"
               " \n"
               "You may convey a covered work in object code form under the terms \n"
               "of sections 4 and 5, provided that you also convey the \n"
               "machine-readable Corresponding Source under the terms of this License, \n"
               "in one of these ways: \n"
               " \n"
               "a) Convey the object code in, or embodied in, a physical product \n"
               "(including a physical distribution medium), accompanied by the \n"
               "Corresponding Source fixed on a durable physical medium \n"
               "customarily used for software interchange. \n"
               " \n"
               "b) Convey the object code in, or embodied in, a physical product \n"
               "(including a physical distribution medium), accompanied by a \n"
               "written offer, valid for at least three years and valid for as \n"
               "long as you offer spare parts or customer support for that product \n"
               "model, to give anyone who possesses the object code either (1) a \n"
               "copy of the Corresponding Source for all the software in the \n"
               "product that is covered by this License, on a durable physical \n"
               "medium customarily used for software interchange, for a price no \n"
               "more than your reasonable cost of physically performing this \n"
               "conveying of source, or (2) access to copy the \n"
               "Corresponding Source from a network server at no charge. \n"
               " \n"
               "c) Convey individual copies of the object code with a copy of the \n"
               "written offer to provide the Corresponding Source.  This \n"
               "alternative is allowed only occasionally and noncommercially, and \n"
               "only if you received the object code with such an offer, in accord \n"
               "with subsection 6b. \n"
               " \n"
               "d) Convey the object code by offering access from a designated \n"
               "place (gratis or for a charge), and offer equivalent access to the \n"
               "Corresponding Source in the same way through the same place at no \n"
               "further charge.  You need not require recipients to copy the \n"
               "Corresponding Source along with the object code.  If the place to \n"
               "copy the object code is a network server, the Corresponding Source \n"
               "may be on a different server (operated by you or a third party) \n"
               "that supports equivalent copying facilities, provided you maintain \n"
               "clear directions next to the object code saying where to find the \n"
               "Corresponding Source.  Regardless of what server hosts the \n"
               "Corresponding Source, you remain obligated to ensure that it is \n"
               "available for as long as needed to satisfy these requirements. \n"
               " \n"
               "e) Convey the object code using peer-to-peer transmission, provided \n"
               "you inform other peers where the object code and Corresponding \n"
               "Source of the work are being offered to the general public at no \n"
               "charge under subsection 6d. \n"
               " \n"
               "A separable portion of the object code, whose source code is excluded \n"
               "from the Corresponding Source as a System Library, need not be \n"
               "included in conveying the object code work. \n"
               " \n"
               "A 'User Product' is either (1) a 'consumer product', which means any \n"
               "tangible personal property which is normally used for personal, family, \n"
               "or household purposes, or (2) anything designed or sold for incorporation \n"
               "into a dwelling.  In determining whether a product is a consumer product, \n"
               "doubtful cases shall be resolved in favor of coverage.  For a particular \n"
               "product received by a particular user, 'normally used' refers to a \n"
               "typical or common use of that class of product, regardless of the status \n"
               "of the particular user or of the way in which the particular user \n"
               "actually uses, or expects or is expected to use, the product.  A product \n"
               "is a consumer product regardless of whether the product has substantial \n"
               "commercial, industrial or non-consumer uses, unless such uses represent \n"
               "the only significant mode of use of the product. \n"
               " \n"
               "'Installation Information' for a User Product means any methods, \n"
               "procedures, authorization keys, or other information required to install \n"
               "and execute modified versions of a covered work in that User Product from \n"
               "a modified version of its Corresponding Source.  The information must \n"
               "suffice to ensure that the continued functioning of the modified object \n"
               "code is in no case prevented or interfered with solely because \n"
               "modification has been made. \n"
               " \n"
               "If you convey an object code work under this section in, or with, or \n"
               "specifically for use in, a User Product, and the conveying occurs as \n"
               "part of a transaction in which the right of possession and use of the \n"
               "User Product is transferred to the recipient in perpetuity or for a \n"
               "fixed term (regardless of how the transaction is characterized), the \n"
               "Corresponding Source conveyed under this section must be accompanied \n"
               "by the Installation Information.  But this requirement does not apply \n"
               "if neither you nor any third party retains the ability to install \n"
               "modified object code on the User Product (for example, the work has \n"
               "been installed in ROM). \n"
               " \n"
               "The requirement to provide Installation Information does not include a \n"
               "requirement to continue to provide support service, warranty, or updates \n"
               "for a work that has been modified or installed by the recipient, or for \n"
               "the User Product in which it has been modified or installed.  Access to a \n"
               "network may be denied when the modification itself materially and \n"
               "adversely affects the operation of the network or violates the rules and \n"
               "protocols for communication across the network. \n"
               " \n"
               "Corresponding Source conveyed, and Installation Information provided, \n"
               "in accord with this section must be in a format that is publicly \n"
               "documented (and with an implementation available to the public in \n"
               "source code form), and must require no special password or key for \n"
               "unpacking, reading or copying. \n"
               " \n"
               "7. Additional Terms. \n"
               " \n"
               "'Additional permissions' are terms that supplement the terms of this \n"
               "License by making exceptions from one or more of its conditions. \n"
               "Additional permissions that are applicable to the entire Program shall \n"
               "be treated as though they were included in this License, to the extent \n"
               "that they are valid under applicable law.  If additional permissions \n"
               "apply only to part of the Program, that part may be used separately \n"
               "under those permissions, but the entire Program remains governed by \n"
               "this License without regard to the additional permissions. \n"
               " \n"
               "When you convey a copy of a covered work, you may at your option \n"
               "remove any additional permissions from that copy, or from any part of \n"
               "it.  (Additional permissions may be written to require their own \n"
               "removal in certain cases when you modify the work.)  You may place \n"
               "additional permissions on material, added by you to a covered work, \n"
               "for which you have or can give appropriate copyright permission. \n"
               " \n"
               "Notwithstanding any other provision of this License, for material you \n"
               "add to a covered work, you may (if authorized by the copyright holders of \n"
               "that material) supplement the terms of this License with terms: \n"
               " \n"
               "a) Disclaiming warranty or limiting liability differently from the \n"
               "terms of sections 15 and 16 of this License; or \n"
               " \n"
               "b) Requiring preservation of specified reasonable legal notices or \n"
               "author attributions in that material or in the Appropriate Legal \n"
               "Notices displayed by works containing it; or \n"
               " \n"
               "c) Prohibiting misrepresentation of the origin of that material, or \n"
               "requiring that modified versions of such material be marked in \n"
               "reasonable ways as different from the original version; or \n"
               " \n"
               "d) Limiting the use for publicity purposes of names of licensors or \n"
               "authors of the material; or \n"
               " \n"
               "e) Declining to grant rights under trademark law for use of some \n"
               "trade names, trademarks, or service marks; or \n"
               " \n"
               "f) Requiring indemnification of licensors and authors of that \n"
               "material by anyone who conveys the material (or modified versions of \n"
               "it) with contractual assumptions of liability to the recipient, for \n"
               "any liability that these contractual assumptions directly impose on \n"
               "those licensors and authors. \n"
               " \n"
               "All other non-permissive additional terms are considered 'further \n"
               "restrictions' within the meaning of section 10.  If the Program as you \n"
               "received it, or any part of it, contains a notice stating that it is \n"
               "governed by this License along with a term that is a further \n"
               "restriction, you may remove that term.  If a license document contains \n"
               "a further restriction but permits relicensing or conveying under this \n"
               "License, you may add to a covered work material governed by the terms \n"
               "of that license document, provided that the further restriction does \n"
               "not survive such relicensing or conveying. \n"
               " \n"
               "If you add terms to a covered work in accord with this section, you \n"
               "must place, in the relevant source files, a statement of the \n"
               "additional terms that apply to those files, or a notice indicating \n"
               "where to find the applicable terms. \n"
               " \n"
               "Additional terms, permissive or non-permissive, may be stated in the \n"
               "form of a separately written license, or stated as exceptions; \n"
               "the above requirements apply either way. \n"
               " \n"
               "8. Termination. \n"
               " \n"
               "You may not propagate or modify a covered work except as expressly \n"
               "provided under this License.  Any attempt otherwise to propagate or \n"
               "modify it is void, and will automatically terminate your rights under \n"
               "this License (including any patent licenses granted under the third \n"
               "paragraph of section 11). \n"
               " \n"
               "However, if you cease all violation of this License, then your \n"
               "license from a particular copyright holder is reinstated (a) \n"
               "provisionally, unless and until the copyright holder explicitly and \n"
               "finally terminates your license, and (b) permanently, if the copyright \n"
               "holder fails to notify you of the violation by some reasonable means \n"
               "prior to 60 days after the cessation. \n"
               " \n"
               "Moreover, your license from a particular copyright holder is \n"
               "reinstated permanently if the copyright holder notifies you of the \n"
               "violation by some reasonable means, this is the first time you have \n"
               "received notice of violation of this License (for any work) from that \n"
               "copyright holder, and you cure the violation prior to 30 days after \n"
               "your receipt of the notice. \n"
               " \n"
               "Termination of your rights under this section does not terminate the \n"
               "licenses of parties who have received copies or rights from you under \n"
               "this License.  If your rights have been terminated and not permanently \n"
               "reinstated, you do not qualify to receive new licenses for the same \n"
               "material under section 10. \n"
               " \n"
               "9. Acceptance Not Required for Having Copies. \n"
               " \n"
               "You are not required to accept this License in order to receive or \n"
               "run a copy of the Program.  Ancillary propagation of a covered work \n"
               "occurring solely as a consequence of using peer-to-peer transmission \n"
               "to receive a copy likewise does not require acceptance.  However, \n"
               "nothing other than this License grants you permission to propagate or \n"
               "modify any covered work.  These actions infringe copyright if you do \n"
               "not accept this License.  Therefore, by modifying or propagating a \n"
               "covered work, you indicate your acceptance of this License to do so. \n"
               " \n"
               "10. Automatic Licensing of Downstream Recipients. \n"
               " \n"
               "Each time you convey a covered work, the recipient automatically \n"
               "receives a license from the original licensors, to run, modify and \n"
               "propagate that work, subject to this License.  You are not responsible \n"
               "for enforcing compliance by third parties with this License. \n"
               " \n"
               "An 'entity transaction' is a transaction transferring control of an \n"
               "organization, or substantially all assets of one, or subdividing an \n"
               "organization, or merging organizations.  If propagation of a covered \n"
               "work results from an entity transaction, each party to that \n"
               "transaction who receives a copy of the work also receives whatever \n"
               "licenses to the work the party's predecessor in interest had or could \n"
               "give under the previous paragraph, plus a right to possession of the \n"
               "Corresponding Source of the work from the predecessor in interest, if \n"
               "the predecessor has it or can get it with reasonable efforts. \n"
               " \n"
               "You may not impose any further restrictions on the exercise of the \n"
               "rights granted or affirmed under this License.  For example, you may \n"
               "not impose a license fee, royalty, or other charge for exercise of \n"
               "rights granted under this License, and you may not initiate litigation \n"
               "(including a cross-claim or counterclaim in a lawsuit) alleging that \n"
               "any patent claim is infringed by making, using, selling, offering for \n"
               "sale, or importing the Program or any portion of it. \n"
               " \n"
               "11. Patents. \n"
               " \n"
               "A 'contributor' is a copyright holder who authorizes use under this \n"
               "License of the Program or a work on which the Program is based.  The \n"
               "work thus licensed is called the contributor's 'contributor version'. \n"
               " \n"
               "A contributor's 'essential patent claims' are all patent claims \n"
               "owned or controlled by the contributor, whether already acquired or \n"
               "hereafter acquired, that would be infringed by some manner, permitted \n"
               "by this License, of making, using, or selling its contributor version, \n"
               "but do not include claims that would be infringed only as a \n"
               "consequence of further modification of the contributor version.  For \n"
               "purposes of this definition, 'control' includes the right to grant \n"
               "patent sublicenses in a manner consistent with the requirements of \n"
               "this License. \n"
               " \n"
               "Each contributor grants you a non-exclusive, worldwide, royalty-free \n"
               "patent license under the contributor's essential patent claims, to \n"
               "make, use, sell, offer for sale, import and otherwise run, modify and \n"
               "propagate the contents of its contributor version. \n"
               " \n"
               "In the following three paragraphs, a 'patent license' is any express \n"
               "agreement or commitment, however denominated, not to enforce a patent \n"
               "(such as an express permission to practice a patent or covenant not to \n"
               "sue for patent infringement).  To 'grant' such a patent license to a \n"
               "party means to make such an agreement or commitment not to enforce a \n"
               "patent against the party. \n"
               " \n"
               "If you convey a covered work, knowingly relying on a patent license, \n"
               "and the Corresponding Source of the work is not available for anyone \n"
               "to copy, free of charge and under the terms of this License, through a \n"
               "publicly available network server or other readily accessible means, \n"
               "then you must either (1) cause the Corresponding Source to be so \n"
               "available, or (2) arrange to deprive yourself of the benefit of the \n"
               "patent license for this particular work, or (3) arrange, in a manner \n"
               "consistent with the requirements of this License, to extend the patent \n"
               "license to downstream recipients.  'Knowingly relying' means you have \n"
               "actual knowledge that, but for the patent license, your conveying the \n"
               "covered work in a country, or your recipient's use of the covered work \n"
               "in a country, would infringe one or more identifiable patents in that \n"
               "country that you have reason to believe are valid. \n"
               " \n"
               "If, pursuant to or in connection with a single transaction or \n"
               "arrangement, you convey, or propagate by procuring conveyance of, a \n"
               "covered work, and grant a patent license to some of the parties \n"
               "receiving the covered work authorizing them to use, propagate, modify \n"
               "or convey a specific copy of the covered work, then the patent license \n"
               "you grant is automatically extended to all recipients of the covered \n"
               "work and works based on it. \n"
               " \n"
               "A patent license is 'discriminatory' if it does not include within \n"
               "the scope of its coverage, prohibits the exercise of, or is \n"
               "conditioned on the non-exercise of one or more of the rights that are \n"
               "specifically granted under this License.  You may not convey a covered \n"
               "work if you are a party to an arrangement with a third party that is \n"
               "in the business of distributing software, under which you make payment \n"
               "to the third party based on the extent of your activity of conveying \n"
               "the work, and under which the third party grants, to any of the \n"
               "parties who would receive the covered work from you, a discriminatory \n"
               "patent license (a) in connection with copies of the covered work \n"
               "conveyed by you (or copies made from those copies), or (b) primarily \n"
               "for and in connection with specific products or compilations that \n"
               "contain the covered work, unless you entered into that arrangement, \n"
               "or that patent license was granted, prior to 28 March 2007. \n"
               " \n"
               "Nothing in this License shall be construed as excluding or limiting \n"
               "any implied license or other defenses to infringement that may \n"
               "otherwise be available to you under applicable patent law. \n"
               " \n"
               "12. No Surrender of Others' Freedom. \n"
               " \n"
               "If conditions are imposed on you (whether by court order, agreement or \n"
               "otherwise) that contradict the conditions of this License, they do not \n"
               "excuse you from the conditions of this License.  If you cannot convey a \n"
               "covered work so as to satisfy simultaneously your obligations under this \n"
               "License and any other pertinent obligations, then as a consequence you may \n"
               "not convey it at all.  For example, if you agree to terms that obligate you \n"
               "to collect a royalty for further conveying from those to whom you convey \n"
               "the Program, the only way you could satisfy both those terms and this \n"
               "License would be to refrain entirely from conveying the Program. \n"
               " \n"
               "13. Use with the GNU Affero General Public License. \n"
               " \n"
               "Notwithstanding any other provision of this License, you have \n"
               "permission to link or combine any covered work with a work licensed \n"
               "under version 3 of the GNU Affero General Public License into a single \n"
               "combined work, and to convey the resulting work.  The terms of this \n"
               "License will continue to apply to the part which is the covered work, \n"
               "but the special requirements of the GNU Affero General Public License, \n"
               "section 13, concerning interaction through a network will apply to the \n"
               "combination as such. \n"
               " \n"
               "14. Revised Versions of this License. \n"
               " \n"
               "The Free Software Foundation may publish revised and/or new versions of \n"
               "the GNU General Public License from time to time.  Such new versions will \n"
               "be similar in spirit to the present version, but may differ in detail to \n"
               "address new problems or concerns. \n"
               " \n"
               "Each version is given a distinguishing version number.  If the \n"
               "Program specifies that a certain numbered version of the GNU General \n"
               "Public License 'or any later version' applies to it, you have the \n"
               "option of following the terms and conditions either of that numbered \n"
               "version or of any later version published by the Free Software \n"
               "Foundation.  If the Program does not specify a version number of the \n"
               "GNU General Public License, you may choose any version ever published \n"
               "by the Free Software Foundation. \n"
               " \n"
               "If the Program specifies that a proxy can decide which future \n"
               "versions of the GNU General Public License can be used, that proxy's \n"
               "public statement of acceptance of a version permanently authorizes you \n"
               "to choose that version for the Program. \n"
               " \n"
               "Later license versions may give you additional or different \n"
               "permissions.  However, no additional obligations are imposed on any \n"
               "author or copyright holder as a result of your choosing to follow a \n"
               "later version. \n"
               " \n"
               "15. Disclaimer of Warranty. \n"
               " \n"
               "THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY \n"
               "APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT \n"
               "HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM 'AS IS' WITHOUT WARRANTY \n"
               "OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, \n"
               "THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR \n"
               "PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM \n"
               "IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF \n"
               "ALL NECESSARY SERVICING, REPAIR OR CORRECTION. \n"
               " \n"
               "16. Limitation of Liability. \n"
               " \n"
               "IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING \n"
               "WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS \n"
               "THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY \n"
               "GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE \n"
               "USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF \n"
               "DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD \n"
               "PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), \n"
               "EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF \n"
               "SUCH DAMAGES. \n"
               " \n"
               "17. Interpretation of Sections 15 and 16. \n"
               " \n"
               "If the disclaimer of warranty and limitation of liability provided \n"
               "above cannot be given local legal effect according to their terms, \n"
               "reviewing courts shall apply local law that most closely approximates \n"
               "an absolute waiver of all civil liability in connection with the \n"
               "Program, unless a warranty or assumption of liability accompanies a \n"
               "copy of the Program in return for a fee. \n"
               " \n"
               " END OF TERMS AND CONDITIONS \n\n");
}

__host__ void print_help() {
        printf("Example: ./bin/gpuvmem options [ arguments ...]\n");
        printf("    -h  --help             Shows this\n");
        printf( "   -X  --blockSizeX       Block X Size for Image (Needs to be pow of 2) (Mandatory)\n");
        printf( "   -Y  --blockSizeY       Block Y Size for Image (Needs to be pow of 2) (Mandatory)\n");
        printf( "   -V  --blockSizeV       Block Size for Visibilities (Needs to be pow of 2) (Mandatory)\n");
        printf( "   -i  --input            The name of the input file of visibilities(MS) (Mandatory)\n");
        printf( "   -o  --output           The name of the output file of residual visibilities(MS) (Mandatory)\n");
        printf( "   -O  --output-image     The name of the output image FITS file (Mandatory)\n");
        printf("    -I  --inputdat         The name of the input file of parameters (Mandatory)\n");
        printf("    -m  --modin            mod_in_0 FITS (I_nu_0) image file (Mandatory)\n");
        printf("    -F  --nu_0             Frequency of reference (Mandatory)\n");
        printf("    -a  --alpha            Alpha spectral index image\n");
        printf("    -x  --minpix           Minimum positive value of a pixel (Optional)\n");
        printf("    -n  --noise            Noise Parameter (Optional)\n");
        printf("    -N  --noise-cut        Noise-cut Parameter (Optional)\n");
        printf("    -l  --lambda           Lambda Regularization Parameter (Optional)\n");
        printf("    -E  --epsilon          Epsilon Regularization Parameter (Optional)\n");
        printf("    -g  --gridding         Use gridding to decrease the number of visibilities. This is done in CPU (Need to select the CPU threads that will grid the input visibilities)\n");
        printf("    -r  --randoms          Percentage of data used when random sampling (Default = 1.0, optional)\n");
        printf("    -P  --prior            Prior used to regularize the solution (Default = 0 = Entropy)\n");
        printf("    -e  --eta              Variable that controls the minimum image value (Default eta = -1.0)\n");
        printf("    -p  --path             MEM path to save FITS images. With last / included. (Example ./../mem/)\n");
        printf("    -f  --file             Output file where final objective function values are saved (Optional)\n");
        printf("    -M  --multigpu         Number of GPUs to use multiGPU image synthesis (Default OFF => 0)\n");
        printf("    -R  --robust-parameter Robust weighting parameter when gridding. -2.0 for uniform weighting, 2.0 for natural weighting and 0.0 for a tradeoff between these two. (Default R = 2.0).\n");
        printf("    -s  --select           If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)\n");
        printf("    -t  --iterations       Number of iterations for optimization (Default = 500)\n");
        printf("    -B  --burndown_steps   Burndown iterations\n");
        printf("    -c  --copyright        Shows copyright conditions\n");
        printf("    -w  --warranty         Shows no warranty details\n");
        printf("        --nopositivity     Run gpuvmem using chi2 with no posititivy restriction\n");
        printf("        --apply-noise      Apply random gaussian noise to visibilities\n");
        printf("        --clipping         Clips the image to positive values\n");
        printf("        --print-images     Prints images per iteration\n");
        printf("        --checkpoint       Start from a certain iteration\n");
        printf("        --spec_idx       Spectral index calculation\n");
        printf("        --adaptive         MCMC Noise for each pixel is estimated using the variance of the samples\n");
        printf("        --use-mask         MCMC Noise depends of the signal of the object, more noise will be injected where is no signal\n");
        printf("        --verbose          Shows information through all the execution\n");
}

__host__ char *strip(const char *string, const char *chars)
{
        char * newstr = (char*)malloc(strlen(string) + 1);
        int counter = 0;

        for (; *string; string++) {
                if (!strchr(chars, *string)) {
                        newstr[ counter ] = *string;
                        ++counter;
                }
        }

        newstr[counter] = 0;
        return newstr;
}

void swap (int *a, int *b)
{
        int temp = *a;
        *a = *b;
        *b = temp;
}

void randomize(int2 arr[], int n)
{
        // Use a different seed value so that we don't get same
        // result each time we run this program
        SelectStream(3);

        // Start from the last element and swap one by one. We don't
        // need to run for the first element that's why i > 0
        for (int i = n-1; i > 0; i--)
        {
                // Pick a random index from 0 to i

                int j = Uniform(0, (float)i);

                // Swap arr[i] with the element at random index
                swap(&arr[i].x, &arr[j].x);
                swap(&arr[i].y, &arr[j].y);
        }
}

__host__ Vars getOptions(int argc, char **argv) {
        Vars variables;
        variables.multigpu = "NULL";
        variables.ofile = "NULL";
        variables.path = "mem/";
        variables.output_image = "mod_out";
        variables.select = 0;
        variables.blockSizeX = -1;
        variables.blockSizeY = -1;
        variables.blockSizeV = -1;
        variables.nu_0 = -1;
        variables.it_max = 500;
        variables.burndown_steps = 0;
        variables.noise = -1;
        variables.epsilon = 0.0;
        variables.lambda = -1;
        variables.randoms = -1;
        variables.noise_cut = -1;
        variables.minpix = -1;
        variables.reg_term = 0;
        variables.alpha_name = "NULL";
        variables.eta = -1.0;
        variables.gridding = 0;
        variables.robust_param = 2.0;
        variables.current_k = 1;
        variables.noise_factorInu0 = 1.0;


        long next_op;
        const char* const short_op = "hcwi:o:O:I:m:x:n:N:l:r:f:M:s:p:P:X:R:Y:V:t:F:a:e:E:B:A:g:k:z:";

        const struct option long_op[] = { //Flag for help, copyright and warranty
                {"help", 0, NULL, 'h' },
                {"warranty", 0, NULL, 'w' },
                {"copyright", 0, NULL, 'c' },
                /* These options set a flag. */
                {"verbose", 0, &verbose_flag, 1},
                {"apply-noise", 0, &apply_noise, 1},
                {"nopositivity", 0, &nopositivity, 1},
                {"clipping", 0, &clip_flag, 1},
                {"checkpoint", 0, &checkpoint, 1},
                {"print-images", 0, &print_images, 1},
                {"adaptive", 0, &adaptive, 1},
                {"use-mask", 0, &use_mask, 1},
                {"spec_idx", 0, &spec_idx, 1},
                /* These options don’t set a flag. */
                {"input", 1, NULL, 'i' }, {"output", 1, NULL, 'o'}, {"output-image", 1, NULL, 'O'},
                {"inputdat", 1, NULL, 'I'}, {"modin", 1, NULL, 'm' }, {"noise", 0, NULL, 'n' },
                {"lambda", 0, NULL, 'l' }, {"multigpu", 0, NULL, 'M'}, {"select", 1, NULL, 's'},
                {"path", 1, NULL, 'p'}, {"prior", 0, NULL, 'P'}, {"eta", 0, NULL, 'e'},
                {"blockSizeX", 1, NULL, 'X'}, {"blockSizeY", 1, NULL, 'Y'}, {"robust-parameter", 0, NULL, 'R'}, {"blockSizeV", 1, NULL, 'V'},
                {"iterations", 0, NULL, 't'}, {"burndown_steps", 1, NULL, 'B'}, {"noise-cut", 0, NULL, 'N' }, {"minpix", 0, NULL, 'x' },
                {"randoms", 0, NULL, 'r' }, {"nu_0", 1, NULL, 'F' }, {"file", 0, NULL, 'f' }, {"current_k", 1, NULL, 'k' },
                {"epsilon", 0, NULL, 'E' }, {"alpha_name", 1, NULL, 'a' }, {"alpha_value", 1, NULL, 'A' }, {"gridding", 0, NULL, 'g' },
                {"noise-factor", 0, NULL, 'z' },
                { NULL, 0, NULL, 0 }
        };

        if (argc == 1) {
                printf(
                        "ERROR. THE PROGRAM HAS BEEN EXECUTED WITHOUT THE NEEDED PARAMETERS OR OPTIONS\n");
                print_help();
                exit(EXIT_SUCCESS);
        }
        int option_index = 0;
        while (1) {
                next_op = getopt_long(argc, argv, short_op, long_op, &option_index);
                if (next_op == -1) {
                        break;
                }

                switch (next_op) {
                case 0:
                        /* If this option set a flag, do nothing else now. */
                        if (long_op[option_index].flag != 0)
                                break;
                        printf ("option %s", long_op[option_index].name);
                        if (optarg)
                                printf (" with arg %s", optarg);
                        printf ("\n");
                        break;
                case 'h':
                        print_help();
                        exit(EXIT_SUCCESS);
                case 'w':
                        print_warranty();
                        exit(EXIT_SUCCESS);
                case 'c':
                        print_copyright();
                        exit(EXIT_SUCCESS);
                case 'i':
                        variables.input = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.input, optarg);
                        break;
                case 'o':
                        variables.output = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.output, optarg);
                        break;
                case 'O':
                        variables.output_image = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.output_image, optarg);
                        break;
                case 'I':
                        variables.inputdat = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.inputdat, optarg);
                        break;
                case 'm':
                        variables.modin = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.modin, optarg);
                        break;
                case 'x':
                        variables.minpix = atof(optarg);
                        break;
                case 'e':
                        variables.eta = atof(optarg);
                        break;
                case 'E':
                        variables.epsilon = atof(optarg);
                        break;
                case 'F':
                        variables.nu_0 = atof(optarg);
                        break;
                case 'R':
                        variables.robust_param = atof(optarg);
                        break;
                case 'a':
                        variables.alpha_name = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.alpha_name, optarg);
                        break;
                case 'A':
                        variables.alpha_value = atof(optarg);
                        break;
                case 'n':
                        variables.noise = atof(optarg);
                        break;
                case 'N':
                        variables.noise_cut = atof(optarg);
                        break;
                case 'l':
                        variables.lambda = atof(optarg);
                        break;
                case 'p':
                        variables.path = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.path, optarg);
                        break;
                case 'P':
                        variables.reg_term = atoi(optarg);;
                        break;
                case 'M':
                        variables.multigpu = optarg;
                        break;
                case 'r':
                        variables.randoms = atof(optarg);
                        break;
                case 'f':
                        variables.ofile = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.ofile, optarg);
                        break;
                case 's':
                        variables.select = atoi(optarg);
                        break;
                case 'X':
                        variables.blockSizeX = atoi(optarg);
                        break;
                case 'Y':
                        variables.blockSizeY = atoi(optarg);
                        break;
                case 'V':
                        variables.blockSizeV = atoi(optarg);
                        break;
                case 'k':
                        variables.current_k = atoi(optarg);
                        break;
                case 't':
                        variables.it_max = atoi(optarg);
                        break;
                case 'B':
                        variables.burndown_steps = atoi(optarg);
                        break;
                case 'g':
                        variables.gridding = atoi(optarg);
                        break;
                case 'z':
                        variables.noise_factorInu0 = atof(optarg);
                        break;
                case '?':
                        print_help();
                        exit(EXIT_FAILURE);
                case -1:
                        break;
                default:
                        print_help();
                        exit(EXIT_FAILURE);
                }
        }

        if(variables.blockSizeX == -1 && variables.blockSizeY == -1 && variables.blockSizeV == -1 ||
           strcmp(strip(variables.input, " "),"") == 0 && strcmp(strip(variables.output, " "),"") == 0 && strcmp(strip(variables.output_image, " "),"") == 0 && strcmp(strip(variables.inputdat, " "),"") == 0 ||
           strcmp(strip(variables.modin, " "),"") == 0 && strcmp(strip(variables.path, " "),"") == 0 || variables.nu_0 == -1 || strcmp(strip(variables.alpha_name, " "),"") == 0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(!isPow2(variables.blockSizeX) && !isPow2(variables.blockSizeY) && !isPow2(variables.blockSizeV)) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(variables.randoms > 1.0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(variables.reg_term > 2) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(variables.gridding < 0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(strcmp(variables.multigpu,"NULL")!=0 && variables.select != 0) {
                print_help();
                exit(EXIT_FAILURE);
        }
        return variables;
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





__host__ float deviceReduce(float *in, long N)
{
        float *device_out;
        gpuErrchk(cudaMalloc(&device_out, sizeof(float)*1024));
        gpuErrchk(cudaMemset(device_out, 0, sizeof(float)*1024));

        int threads = 512;
        int blocks = min((int(NearestPowerOf2(N)) + threads - 1) / threads, 1024);
        int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

        bool isPower2 = isPow2(N);
        if(isPower2) {
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

__global__ void hermitianSymmetry(double3 *UVW, cufftComplex *Vo, float freq, int numVisibilities)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;

        if (i < numVisibilities) {
                if(UVW[i].x < 0.0) {
                        UVW[i].x *= -1.0;
                        UVW[i].y *= -1.0;
                        Vo[i].y *= -1.0;
                }
                UVW[i].x *= freq / LIGHTSPEED;
                UVW[i].y *= freq / LIGHTSPEED;
                UVW[i].z *= freq / LIGHTSPEED;
        }
}


__device__ float AiryDiskBeam(float distance, float lambda, float antenna_diameter, float pb_factor)
{
        float atten;
        float r = pb_factor * lambda / antenna_diameter;
        float bessel_arg = PI*distance/(r/RZ);
        float bessel_func = j1f(bessel_arg);
        if(distance == 0.0f) {
                atten = 1.0f;
        }else{
                atten = 4.0f * (bessel_func/bessel_arg) * (bessel_func/bessel_arg);
        }
        return atten;
}

__device__ float EllipticGaussianKernel(float amplitude, int x_c, int y_c, float bmaj, float bmin, float bpa, double DELTAX, double DELTAY)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float x = (j - x_c) * DELTAX * RPDEG_D;
        float y = (i - y_c) * DELTAY * RPDEG_D;
        float cos_bpa = cosf(bpa);
        float sin_bpa = sinf(bpa);
        float sin_bpa_2 = sinf(2.0*bpa);
        float a = (cos_bpa*cos_bpa)/(2.0*bmaj*bmaj) + (sin_bpa*sin_bpa)/(2.0*bmin*bmin);
        float b = sin_bpa_2/(2.0*bmaj*bmaj) - sin_bpa_2/(2.0*bmin*bmin);
        float c = (sin_bpa*sin_bpa)/(2.0*bmaj*bmaj) + (cos_bpa*cos_bpa)/(2.0*bmin*bmin);
        float G = amplitude*expf(-a*x*x - b*x*y - c*y*y);
        return G;
}

__device__ float GaussianBeam(float distance, float lambda, float antenna_diameter, float pb_factor)
{
        float fwhm = pb_factor * lambda / antenna_diameter;
        float c = 4.0*logf(2.0);
        float r = distance/fwhm;
        float atten = expf(-c*r*r);
        return atten;
}

__device__ float attenuation(float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, double DELTAX, double DELTAY)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten_result, atten;

        int x0 = xobs;
        int y0 = yobs;
        float x = (j - x0) * DELTAX * RPDEG_D;
        float y = (i - y0) * DELTAY * RPDEG_D;

        float arc = sqrtf(x*x+y*y);
        float lambda = LIGHTSPEED/freq;
        atten = GaussianBeam(arc, lambda, antenna_diameter, pb_factor);

        if(arc <= pb_cutoff) {
                atten_result = atten;
        }else{
                atten_result = 0.0f;
        }

        return atten_result;
}



__global__ void total_attenuation(float *total_atten, float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, double DELTAX, double DELTAY, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float attenPerFreq = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

        total_atten[N*i+j] += attenPerFreq;
}

__global__ void mean_attenuation(float *total_atten, int channels, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        total_atten[N*i+j] /= channels;
}

__global__ void weight_image(float *weight_image, float *total_atten, float noise_jypix, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten = total_atten[N*i+j];
        weight_image[N*i+j] += (atten / noise_jypix) * (atten / noise_jypix);
}

__global__ void noise_image(float *noise_image, float *weight_image, float noise_jypix, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float noiseval;
        noiseval = sqrtf(1.0/weight_image[N*i+j]);
        noise_image[N*i+j] = noiseval;
}

__global__ void apply_beam(float antenna_diameter, float pb_factor, float pb_cutoff, cufftComplex *image, long N, float xobs, float yobs, float fg_scale, float freq, float DELTAX, float DELTAY)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

        image[N*i+j].x = image[N*i+j].x * atten;
        image[N*i+j].y = 0.0;
}


/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(-2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float u,v, phase, c, s, re, im;
        float du = xphs/(float)M;
        float dv = yphs/(float)N;

        if(j < M/2) {
                u = du * j;
        }else{
                u = du * (j-M);
        }

        if(i < N/2) {
                v = dv * i;
        }else{
                v = dv * (i-N);
        }

        phase = 2.0*(u+v);
    #if (__CUDA_ARCH__ >= 300 )
        sincospif(phase, &s, &c);
    #else
        c = cospif(phase);
        s = sinpif(phase);
    #endif
        re = data[N*i+j].x;
        im = data[N*i+j].y;
        data[N*i+j].x = re * c - im * s;
        data[N*i+j].y = re * s + im * c;
}


/*
 * Interpolate in the visibility array to find the visibility at (u,v);
 */
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *V, double3 *UVW, float *weight, double deltau, double deltav, long numVisibilities, long N)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int i1, i2, j1, j2;
        double du, dv;
        double2 uv;
        cufftComplex v11, v12, v21, v22;
        float Zreal;
        float Zimag;

        if (i < numVisibilities) {

                uv.x = UVW[i].x/fabs(deltau);
                uv.y = UVW[i].y/deltav;

                if (fabs(uv.x) <= (N/2)+0.5 && fabs(uv.y) <= (N/2)+0.5) {

                        if(uv.x < 0.0)
                                uv.x = round(uv.x+N);


                        if(uv.y < 0.0)
                                uv.y = round(uv.y+N);


                        i1 = (int)uv.x;
                        i2 = (i1+1)%N;
                        du = uv.x - i1;

                        j1 = (int)uv.y;
                        j2 = (j1+1)%N;
                        dv = uv.y - j1;

                        if (i1 >= 0 && i1 < N && i2 >= 0 && i2 < N && j1 >= 0 && j1 < N && j2 >= 0 && j2 < N) {
                                /* Bilinear interpolation */
                                v11 = V[N*j1 + i1]; /* [i1, j1] */
                                v12 = V[N*j2 + i1]; /* [i1, j2] */
                                v21 = V[N*j1 + i2]; /* [i2, j1] */
                                v22 = V[N*j2 + i2]; /* [i2, j2] */

                                Zreal = (1-du)*(1-dv)*v11.x + (1-du)*dv*v12.x + du*(1-dv)*v21.x + du*dv*v22.x;
                                Zimag = (1-du)*(1-dv)*v11.y + (1-du)*dv*v12.y + du*(1-dv)*v21.y + du*dv*v22.y;

                                Vm[i].x = Zreal;
                                Vm[i].y = Zimag;
                        }else{
                                weight[i] = 0.0f;
                        }
                }else{
                        //Vm[i].x = 0.0f;
                        //Vm[i].y = 0.0f;
                        weight[i] = 0.0f;
                }

        }

}

__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < numVisibilities) {
                Vr[i].x = Vm[i].x - Vo[i].x;
                Vr[i].y = Vm[i].y - Vo[i].y;
        }
}
__global__ void clipWNoise(float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX, float eta)

{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] > noise_cut) {
                if(eta > 0.0) {
                        I[N*i+j].x = 0.0;
                }
                else{
                        I[N*i+j].x = -1.0 * eta * MINPIX;
                }

        }

        I[N*i+j].y = 0;
}

__global__ void clip2IWNoise(float *noise, float2 *I, long N, float noise_cut, float minpix, float alpha_start, float fg_scale, float eta)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] > noise_cut) {
                if(eta > 0.0) {
                        I[N*i+j].x = 0.0;
                }
                else{
                        I[N*i+j].x = -1.0f * eta * minpix * fg_scale;
                }

                I[N*i+j].y = 0.0f;

        }

}

__global__ void clip2I(float2 *I, long N, float minpix, float fg_scale, float eta)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        //I_nu_0
        if(I[N*i+j].x < -1.0f * eta * minpix * fg_scale) {
                I[N*i+j].x = -1.0f * eta * minpix * fg_scale;
        }
        /*/ /alpha
           if(I[N*i+j].y < minpix_alpha) {
                I[N*i+j].y = minpix_alpha;
           } */
}

__global__ void clip(cufftComplex *I, long N, float MINPIX)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        //T
        if(I[N*i+j].x < MINPIX && MINPIX >= 0.0) {
                I[N*i+j].x = MINPIX;
        }
        I[N*i+j].y = 0.0f;
}

__global__ void newP(float2 *p, float2 *xi, float xmin, long N, float minpix, float fg_scale, float eta)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[N*i+j].x *= xmin;
        xi[N*i+j].y *= xmin;
        //I_nu_0
        if(p[N*i+j].x + xi[N*i+j].x > -1.0*eta*minpix*fg_scale) {
                p[N*i+j].x += xi[N*i+j].x;
        }else{
                p[N*i+j].x = -1.0*eta*minpix*fg_scale;
                xi[N*i+j].x = 0.0;
        }

        p[N*i+j].y += xi[N*i+j].y;
        /*/ /alpha
           if(p[N*i+j].y + xi[N*i+j].y > minpix_alpha) {
                p[N*i+j].y += xi[N*i+j].y;
           }else{
                p[N*i+j].y = minpix_alpha;
                xi[N*i+j].y = 0.0;
           } */

}

__global__ void newPNoPositivity(float2 *p, float2 *xi, float xmin, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[N*i+j].x *= xmin;
        xi[N*i+j].y *= xmin;

        p[N*i+j].x += xi[N*i+j].x;
        p[N*i+j].y += xi[N*i+j].y;
}

__global__ void evaluateXt(float2 *xt, float2 *pcom, float2 *xicom, float x, long N, float minpix, float fg_scale, float eta)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        //I_nu_0
        if(pcom[N*i+j].x + x * xicom[N*i+j].x > -1.0*eta*minpix*fg_scale) {
                xt[N*i+j].x = pcom[N*i+j].x + x * xicom[N*i+j].x;
        }else{
                xt[N*i+j].x = -1.0*eta*minpix*fg_scale;
        }

        xt[N*i+j].y = pcom[N*i+j].y + x * xicom[N*i+j].y;
        //alpha
        /*if(pcom[N*i+j].y + x * xicom[N*i+j].y > minpix_alpha){
           xt[N*i+j].y = pcom[N*i+j].y + x * xicom[N*i+j].y;
           }else{
            xt[N*i+j].y = minpix_alpha;
           }*/
}


__global__ void evaluateXtNoPositivity(float2 *xt, float2 *pcom, float2 *xicom, float x, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        //I_nu_0
        xt[N*i+j].x = pcom[N*i+j].x + x * xicom[N*i+j].x;
        //alpha
        xt[N*i+j].y = pcom[N*i+j].y + x * xicom[N*i+j].y;
}


__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, long numVisibilities)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;

        if (i < numVisibilities) {
                chi2[i] =  w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
        }

}

__global__ void LVectorAlpha(float *L, float *noise, float2 *I, long N, float noise_cut)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float Dx, Dy;

        if(noise[N*i+j] <= noise_cut) {
                if((i>1 && i<N-1) && (j>1 && j<N-1)) {
                        Dx = I[N*i+(j-1)].y - 2 * I[N*i+j].y + I[N*i+(j+1)].y;
                        Dy = I[N*(i-1)+j].y - 2 * I[N*i+j].y + I[N*(i+1)+j].y;
                        L[N*i+j] = 0.5 * (Dx + Dy) * (Dx + Dy);
                }else{
                        L[N*i+j] = I[N*i+j].y;
                }
        }
}

__global__ void SVector(float *S, float *noise, float2 *I, long N, float noise_cut, float MINPIX, float fg_scale, float eta)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float entropy = 0.0;
        float I_code = I[N*i+j].x / fg_scale;
        if(noise[N*i+j] <= noise_cut) {
                entropy = I_code * logf((I_code / MINPIX) + (eta + 1.0));
        }

        S[N*i+j] = entropy;
}

__global__ void QPVector(float *Q, float *noise, float2 *I, long N, float noise_cut)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float qp = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                if((i>0 && i<N) && (j>0 && j<N)) {
                        qp = (I[N*i+j].x - I[N*i+(j-1)].x) * (I[N*i+j].x - I[N*i+(j-1)].x) +
                             (I[N*i+j].x - I[N*i+(j+1)].x) * (I[N*i+j].x - I[N*i+(j+1)].x) +
                             (I[N*i+j].x - I[N*(i-1)+j].x) * (I[N*i+j].x - I[N*(i-1)+j].x) +
                             (I[N*i+j].x - I[N*(i+1)+j].x) * (I[N*i+j].x - I[N*(i+1)+j].x);
                        qp /= 2.0;
                }else{
                        qp = I[N*i+j].x;
                }
        }

        Q[N*i+j] = qp;
}

__global__ void TVVector(float *TV, float *noise, float2 *I, long N, float noise_cut)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float tv = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                if(i!= N-1 || j!=N-1) {
                        float dx = I[N*i+j].x - I[N*i+(j+1)].x;
                        float dy = I[N*i+j].x - I[N*(i+1)+j].x;
                        tv = sqrtf((dx * dx) + (dy * dy));
                }else{
                        tv = I[N*i+j].x;
                }
        }

        TV[N*i+j] = tv;
}

__global__ void searchDirection(float2 *g, float2 *xi, float2 *h, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[N*i+j].x = -xi[N*i+j].x;
        g[N*i+j].y = -xi[N*i+j].y;

        xi[N*i+j].x = h[N*i+j].x = g[N*i+j].x;
        xi[N*i+j].y = h[N*i+j].y = g[N*i+j].y;
}

__global__ void newXi(float2 *g, float2 *xi, float2 *h, float gam, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[N*i+j].x = -xi[N*i+j].x;
        g[N*i+j].y = -xi[N*i+j].y;

        xi[N*i+j].x = h[N*i+j].x = g[N*i+j].x + gam * h[N*i+j].x;
        xi[N*i+j].y = h[N*i+j].y = g[N*i+j].y + gam * h[N*i+j].y;
}

__global__ void getGGandDGG(float *gg, float *dgg, float2 *xi, float2 *g, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float gg_Inu_0, gg_alpha;
        float dgg_Inu_0, dgg_alpha;

        gg_Inu_0 = g[N*i+j].x * g[N*i+j].x;
        gg_alpha = g[N*i+j].y * g[N*i+j].y;

        dgg_Inu_0 = (xi[N*i+j].x + g[N*i+j].x) * xi[N*i+j].x;
        dgg_alpha = (xi[N*i+j].y + g[N*i+j].y) * xi[N*i+j].y;

        gg[N*i+j] = gg_Inu_0 + gg_alpha;
        dgg[N*i+j] = dgg_Inu_0 + dgg_alpha;
}

__global__ void restartDPhi(float2 *dphi, float *dS, float *dL, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        //I_nu_0
        dphi[N*i+j].x = 0.0;
        //alpha
        dphi[N*i+j].y = 0.0;

        dS[N*i+j] = 0.0;

        dL[N*i+j] = 0.0;

}


__global__ void calculateInu(cufftComplex *I_nu, float2 *image2, float nu, float nu_0, float fg_scale, float minpix, float eta, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, nudiv_pow_alpha, nudiv;

        nudiv = nu/nu_0;

        I_nu_0 = image2[N*i+j].x;
        alpha = image2[N*i+j].y;
        //alpha = 3.5;

        nudiv_pow_alpha = powf(nudiv, alpha);

        I_nu[N*i+j].x = I_nu_0 * nudiv_pow_alpha;

        I_nu[N*i+j].y = 0.0f;
}

__global__ void random_init(unsigned int seed, curandState_t* states, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        curand_init(seed, /* the seed can be the same for each thread, here we pass the time from CPU */
                    (N*i+j), /* the sequence number should be different for each core */
                    0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                    &states[N*i+j]);
}

__global__ void calculateTheta(float2 *theta, curandState_t* states, double2 *total, double2 *total2, int samples, long N){
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float2 random_number;
        float2 stddev, avg;

        avg.x = total[N*i+j].x / samples;
        avg.y = total[N*i+j].y / samples;

        stddev.x = (2.38/(N*sqrtf(2))) * sqrtf((total2[N*i+j].x / samples) - (avg.x * avg.x));
        stddev.y = (2.38/(N*sqrtf(2))) * sqrtf((total2[N*i+j].y / samples) - (avg.y * avg.y));

        random_number.x = curand_normal(&states[N*i+j]) * stddev.x;
        random_number.y = curand_normal(&states[N*i+j]) * stddev.y;

        theta[N*i+j].x =  random_number.y;
        theta[N*i+j].x =  random_number.y;
}

__global__ void changeI(float2 *I, float2 *temp, float2 *theta, curandState_t* states, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float2 nrandom;

        nrandom.x = curand_normal(&states[N*i+j]) * theta[N*i+j].x;
        nrandom.y = curand_normal(&states[N*i+j]) * theta[N*i+j].y;

        temp[N*i+j].x = I[N*i+j].x + nrandom.x;
        temp[N*i+j].y = I[N*i+j].y + nrandom.y;

}


__host__ void do_gridding(Field *fields, freqData *data, double deltau, double deltav, int M, int N, float robust)
{
        int local_max = 0;
        int max = 0;
        float pow2_factor, S2, w_avg;
        for(int f=0; f < data->nfields; f++) {
                for(int i=0; i < data->total_frequencies; i++) {
            #pragma omp parallel for schedule(static,1)
                        for(int z=0; z < fields[f].numVisibilitiesPerFreq[i]; z++) {
                                int j, k;
                                double3 uvw;
                                float w;
                                cufftComplex Vo;

                                uvw  = fields[f].visibilities[i].uvw[z];
                                w = fields[f].visibilities[i].weight[z];
                                Vo = fields[f].visibilities[i].Vo[z];


                                // Visibilities from metres to klambda
                                uvw.x *= fields[f].visibilities[i].freq / LIGHTSPEED;
                                uvw.y *= fields[f].visibilities[i].freq / LIGHTSPEED;
                                uvw.z *= fields[f].visibilities[i].freq / LIGHTSPEED;

                                //Apply hermitian symmetry (it will be applied afterwards)
                                if(uvw.x < 0.0) {
                                        uvw.x *= -1.0;
                                        uvw.y *= -1.0;
                                        Vo.y *= -1.0;
                                }

                                j = round(uvw.x / fabs(deltau) + N/2);
                                k = round(uvw.y / fabs(deltav) + M/2);

                                if(k < M && j < N)
                                {
                    #pragma omp critical
                                        {
                                                fields[f].gridded_visibilities[i].Vo[N*k+j].x += w * Vo.x;
                                                fields[f].gridded_visibilities[i].Vo[N*k+j].y += w * Vo.y;
                                                fields[f].gridded_visibilities[i].weight[N*k+j] += w;
                                        }
                                }
                        }

                        int visCounter = 0;
                        float gridWeightSum = 0.0f;

                        for(int k=0; k<M; k++) {
                                for (int j = 0; j < N; j++) {
                                        float weight = fields[f].gridded_visibilities[i].weight[N*k+j];
                                        if(weight > 0.0f) {
                                                gridWeightSum += weight;
                                                visCounter++;
                                        }
                                }
                        }

                        // Briggs/Robust formula
                        pow2_factor = pow(10.0, -2.0*robust);
                        w_avg = gridWeightSum/visCounter;
                        S2 = 5.0f * 5.0f * pow2_factor / w_avg;

            #pragma omp parallel for schedule(static,1)
                        for(int k=0; k<M; k++) {
                                for(int j=0; j<N; j++) {
                                        double deltau_meters = fabs(deltau) * (LIGHTSPEED/fields[f].visibilities[i].freq);
                                        double deltav_meters = fabs(deltav) * (LIGHTSPEED/fields[f].visibilities[i].freq);

                                        double u_meters = (j - (N/2)) * deltau_meters;
                                        double v_meters = (k - (M/2)) * deltav_meters;

                                        fields[f].gridded_visibilities[i].uvw[N*k+j].x = u_meters;
                                        fields[f].gridded_visibilities[i].uvw[N*k+j].y = v_meters;

                                        float weight = fields[f].gridded_visibilities[i].weight[N*k+j];
                                        if(weight > 0.0f) {
                                                fields[f].gridded_visibilities[i].Vo[N*k+j].x /= weight;
                                                fields[f].gridded_visibilities[i].Vo[N*k+j].y /= weight;
                                                fields[f].gridded_visibilities[i].weight[N*k+j] /= (1 + weight * S2);
                                        }else{
                                                fields[f].gridded_visibilities[i].weight[N*k+j] = 0.0f;
                                        }
                                }
                        }

                        fields[f].visibilities[i].uvw = (double3*)realloc(fields[f].visibilities[i].uvw, visCounter*sizeof(double3));

                        fields[f].visibilities[i].Vo = (cufftComplex*)realloc(fields[f].visibilities[i].Vo, visCounter*sizeof(cufftComplex));

                        fields[f].visibilities[i].Vm = (cufftComplex*)malloc(visCounter*sizeof(cufftComplex));
                        memset(fields[f].visibilities[i].Vm, 0, visCounter*sizeof(cufftComplex));

                        fields[f].visibilities[i].weight = (float*)realloc(fields[f].visibilities[i].weight, visCounter*sizeof(float));

                        int l = 0;
                        for(int k=0; k<M; k++) {
                                for(int j=0; j<N; j++) {
                                        float weight = fields[f].gridded_visibilities[i].weight[N*k+j];
                                        if(weight > 0.0f) {
                                                fields[f].visibilities[i].uvw[l].x = fields[f].gridded_visibilities[i].uvw[N*k+j].x;
                                                fields[f].visibilities[i].uvw[l].y = fields[f].gridded_visibilities[i].uvw[N*k+j].y;
                                                fields[f].visibilities[i].Vo[l].x = fields[f].gridded_visibilities[i].Vo[N*k+j].x;
                                                fields[f].visibilities[i].Vo[l].y = fields[f].gridded_visibilities[i].Vo[N*k+j].y;
                                                fields[f].visibilities[i].weight[l] = fields[f].gridded_visibilities[i].weight[N*k+j];
                                                l++;
                                        }
                                }
                        }

                        free(fields[f].gridded_visibilities[i].uvw);
                        free(fields[f].gridded_visibilities[i].Vo);
                        free(fields[f].gridded_visibilities[i].weight);

                        if(fields[f].numVisibilitiesPerFreq[i] > 0) {
                                fields[f].numVisibilitiesPerFreq[i] = visCounter;
                        }
                }

                local_max = *std::max_element(fields[f].numVisibilitiesPerFreq,fields[f].numVisibilitiesPerFreq+data->total_frequencies);
                if(local_max > max) {
                        max = local_max;
                }
        }


        data->max_number_visibilities_in_channel = max;
}

__host__ float calculateNoise(Field *fields, freqData data, int *total_visibilities, int blockSizeV, int gridding)
{
        //Declaring block size and number of blocks for visibilities
        float sum_weights = 0.0;
        long UVpow2;

        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i< data.total_frequencies; i++) {
                        //Calculating beam noise
                        for(int j=0; j< fields[f].numVisibilitiesPerFreq[i]; j++) {
                                if(fields[f].visibilities[i].weight[j] > 0.0) {
                                        sum_weights += fields[f].visibilities[i].weight[j];
                                }
                        }
                        *total_visibilities += fields[f].numVisibilitiesPerFreq[i];
                        fields[f].visibilities[i].numVisibilities = fields[f].numVisibilitiesPerFreq[i];
                        UVpow2 = NearestPowerOf2(fields[f].visibilities[i].numVisibilities);
                        fields[f].visibilities[i].threadsPerBlockUV = blockSizeV;
                        fields[f].visibilities[i].numBlocksUV = UVpow2/fields[f].visibilities[i].threadsPerBlockUV;
                }
        }

        float sum_inverse_weight = 1.0/sum_weights;

        if(verbose_flag) {
                float aux_noise = sqrt(sum_inverse_weight);
                printf("Calculated NOISE %e\n", aux_noise);
        }

        if(beam_noise == -1 || gridding > 0)
        {
                beam_noise = sqrt(sum_inverse_weight);
                if(verbose_flag) {
                        printf("No NOISE keyword detected in header or you might be using gridding\n");
                        printf("Using NOISE: %e ...\n", beam_noise);
                }
        }else{
                printf("Using header keyword NOISE anyway...\n");
                printf("Keyword NOISE = %e\n", beam_noise);
        }

        return beam_noise;
}


__global__ void changeGibbsEllipticalMaskAlpha(float2 *temp, float2 *theta, float *mask, float normal_I_nu_0, float normal_alpha, float bmaj, float bmin, float bpa, float factor_beam, float factor_noise, float sigma, int2 pix, float DELTAX, float DELTAY, int N)
{
        float2 nrandom;
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        int c_i = pix.x;
        int c_j = pix.y;

        int idx = N*pix.x + pix.y;

        float bmaj_rad = factor_beam * bmaj * fabs(DELTAX) * RPDEG_D;
        float bmin_rad = factor_beam * bmin * fabs(DELTAY) * RPDEG_D;

        float pix_val = EllipticGaussianKernel(1.0, c_j, c_i, bmaj_rad, bmin_rad, bpa, DELTAX, DELTAY);

        nrandom.x = normal_I_nu_0 * theta[idx].x;
        nrandom.y = normal_alpha * theta[idx].y;

        if(mask[idx] <= 5.0f * sigma)
                temp[N*i+j].y += factor_noise * nrandom.y * pix_val;
        else
                temp[N*i+j].y += nrandom.y * pix_val;

        temp[N*i+j].x += nrandom.x * pix_val;
}

__global__ void changeGibbsEllipticalGaussian(float2 *temp, float2 *theta, float normal_I_nu_0, float bmaj, float bmin, float bpa, float factor, int2 pix, double DELTAX, double DELTAY, int N)
{
        float2 nrandom;
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        int c_i = pix.x;
        int c_j = pix.y;

        int idx = N*pix.x + pix.y;

        float bmaj_rad = factor * bmaj * fabs(DELTAX) * RPDEG_D;
        float bmin_rad = factor * bmin * fabs(DELTAY) * RPDEG_D;

        float pix_val = EllipticGaussianKernel(1.0, c_j, c_i, bmaj_rad, bmin_rad, bpa, DELTAX, DELTAY);

        nrandom.x = normal_I_nu_0 * theta[idx].x;

        temp[N*i+j].x += nrandom.x * pix_val;

}

__global__ void changeGibbsEllipticalGaussianSpecIdx(float2 *temp, float2 *theta, float normal_I_nu_0, float normal_alpha, float bmaj, float bmin, float bpa, float factor, int2 pix, double DELTAX, double DELTAY, int N)
{
        float2 nrandom;
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        int c_i = pix.x;
        int c_j = pix.y;

        int idx = N*pix.x + pix.y;

        float bmaj_rad = factor * bmaj * fabs(DELTAX) * RPDEG_D;
        float bmin_rad = factor * bmin * fabs(DELTAY) * RPDEG_D;

        float pix_val = EllipticGaussianKernel(1.0, c_j, c_i, bmaj_rad, bmin_rad, bpa, DELTAX, DELTAY);

        nrandom.x = normal_I_nu_0 * theta[idx].x;
        nrandom.y = normal_alpha * theta[idx].y;

        temp[N*i+j].x += nrandom.x * pix_val;
        temp[N*i+j].y += nrandom.y * pix_val;

}

__global__ void sumI(double2 *M_k, double2 *Q_k, float2 *I, int k, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        double dif_I_nu_0 = I[N*i+j].x - M_k[N*i+j].x;
        double dif_alpha = I[N*i+j].y - M_k[N*i+j].y;

        double dif_I_nu_02 = dif_I_nu_0 * dif_I_nu_0;
        double dif_alpha2 = dif_alpha * dif_alpha;

        M_k[N*i+j].x = M_k[N*i+j].x + (dif_I_nu_0/k);
        M_k[N*i+j].y = M_k[N*i+j].y + (dif_alpha/k);

        Q_k[N*i+j].x = Q_k[N*i+j].x + ((k-1) * dif_I_nu_02 / k);
        Q_k[N*i+j].y = Q_k[N*i+j].y + ((k-1) * dif_alpha2 / k);
}

__global__ void floatToDoubleKernel(double2 *a, float2 *b, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        a[N*i+j].x = b[N*i+j].x;
        a[N*i+j].y = b[N*i+j].y;
}

__global__ void updateTheta(float2 *theta, double2 *Q_k, float s_d, int samples, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        double2 cov;

        cov.x = Q_k[N*i+j].x/(samples-1);
        cov.y = Q_k[N*i+j].y/(samples-1);

        theta[N*i+j].x = sqrt(s_d * cov.x + s_d * 1E-8);
        theta[N*i+j].y = sqrt(s_d * cov.y + s_d * 1E-8);
}


__host__ float chiCuadrado(float2 *I)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        float resultPhi = 0.0;
        float resultchi2  = 0.0;
        float resultS  = 0.0;
        float resultS_alpha = 0.0;

        //if(clip_flag){
        clip2I<<<numBlocksNN, threadsPerBlockNN>>>(I, N, MINPIX, fg_scale, eta);
        gpuErrchk(cudaDeviceSynchronize());
        //}

        clip2IWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, I, N, noise_cut, MINPIX, alpha_start, fg_scale, eta);
        gpuErrchk(cudaDeviceSynchronize());

        if(epsilon) {
                LVectorAlpha<<<numBlocksNN, threadsPerBlockNN>>>(device_S_alpha, device_noise_image, I, N, noise_cut);
                gpuErrchk(cudaDeviceSynchronize());
        }

        if(iter>0 && lambda!=0.0) {
                switch(reg_term) {
                case 0:
                        SVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, I, N, noise_cut, MINPIX, fg_scale, eta);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 1:
                        QPVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, I, N, noise_cut);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 2:
                        TVVector<<<numBlocksNN, threadsPerBlockNN>>>(device_S, device_noise_image, I, N, noise_cut);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                default:
                        printf("Selected prior is not defined\n");
                        goToError();
                        break;
                }
        }


        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {

                                if(fields[f].numVisibilitiesPerFreq[i] != 0) {

                                        calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(device_Inu, I, fields[f].visibilities[i].freq, nu_0, fg_scale, MINPIX, eta, N);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        /*if(clip_flag){
                                           clip<<<numBlocksNN, threadsPerBlockNN>>>(device_Inu, N, MINPIX);
                                           gpuErrchk(cudaDeviceSynchronize());
                                           }*/

                                        apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(antenna_diameter, pb_factor, pb_cutoff, device_Inu, N, fields[f].ref_xobs, fields[f].ref_yobs, fg_scale, fields[f].visibilities[i].freq, DELTAX, DELTAY);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //FFT 2D
                                        if ((cufftExecC2C(plan1GPU, (cufftComplex*)device_Inu, (cufftComplex*)device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
                                                printf("CUFFT exec error\n");
                                                goToError();
                                        }
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //PHASE_ROTATE
                                        phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_V, M, N, fields[f].phs_xobs, fields[f].phs_yobs);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //RESIDUAL CALCULATION
                                        vis_mod<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vm, device_V, fields[f].device_visibilities[i].uvw, fields[f].device_visibilities[i].weight, deltau, deltav, fields[f].numVisibilitiesPerFreq[i], N);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
                                        gpuErrchk(cudaDeviceSynchronize());


                                        ////chi 2 VECTOR
                                        chi2Vector<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(device_chi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].weight, fields[f].numVisibilitiesPerFreq[i]);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //REDUCTIONS
                                        //chi2
                                        resultchi2  += deviceReduce(device_chi2, fields[f].numVisibilitiesPerFreq[i]);
                                }
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        #pragma omp parallel for schedule(static,1)
                        for (int i = 0; i < data.total_frequencies; i++)
                        {
                                float partial_chi2 = 0.0;

                                unsigned int j = omp_get_thread_num();
                                //unsigned int num_cpu_threads = omp_get_num_threads();
                                // set and check the CUDA device for this CPU thread
                                int gpu_id = -1;
                                cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                cudaGetDevice(&gpu_id);
                                if(fields[f].numVisibilitiesPerFreq[i] != 0) {

                                        gpuErrchk(cudaMemset(vars_gpu[i%num_gpus].device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel));

                                        calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(vars_gpu[i%num_gpus].device_Inu, I, fields[f].visibilities[i].freq, nu_0, fg_scale, MINPIX, eta, N);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        /*if(clip_flag){
                                           clip<<<numBlocksNN, threadsPerBlockNN>>>(vars_gpu[i%num_gpus].device_Inu, N, MINPIX);
                                           gpuErrchk(cudaDeviceSynchronize());
                                           }*/

                                        apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(antenna_diameter, pb_factor, pb_cutoff, vars_gpu[i%num_gpus].device_Inu, N, fields[f].ref_xobs, fields[f].ref_yobs, fg_scale, fields[f].visibilities[i].freq, DELTAX, DELTAY);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //FFT 2D
                                        if ((cufftExecC2C(vars_gpu[i%num_gpus].plan, (cufftComplex*)vars_gpu[i%num_gpus].device_Inu, (cufftComplex*)vars_gpu[i%num_gpus].device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
                                                printf("CUFFT exec error\n");
                                                //return -1 ;
                                                goToError();
                                        }
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //PHASE_ROTATE
                                        phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(vars_gpu[i%num_gpus].device_V, M, N, fields[f].phs_xobs, fields[f].phs_yobs);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //RESIDUAL CALCULATION
                                        vis_mod<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vm, vars_gpu[i%num_gpus].device_V, fields[f].device_visibilities[i].uvw, fields[f].device_visibilities[i].weight, deltau, deltav, fields[f].numVisibilitiesPerFreq[i], N);
                                        gpuErrchk(cudaDeviceSynchronize());


                                        residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        ////chi2 VECTOR
                                        chi2Vector<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(vars_gpu[i%num_gpus].device_chi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].weight, fields[f].numVisibilitiesPerFreq[i]);
                                        gpuErrchk(cudaDeviceSynchronize());


                                        partial_chi2 = deviceReduce(vars_gpu[i%num_gpus].device_chi2, fields[f].numVisibilitiesPerFreq[i]);

                                        //REDUCTIONS
                                        //chi2
                                        #pragma omp critical
                                        {
                                                resultchi2  += partial_chi2;
                                        }
                                }
                        }
                }
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        resultS = deviceReduce(device_S, M*N);

        if(epsilon) {
                resultS_alpha = deviceReduce(device_S_alpha, M*N);
        }

        resultPhi = (0.5 * resultchi2) + (lambda * resultS) + (epsilon * resultS_alpha);

        final_chi2 = resultchi2;
        final_H = resultS;
        /*printf("chi2 value = %.5f\n", resultchi2);
           printf("H value = %.5f\n", resultH);
           printf("(1/2) * chi2 value = %.5f\n", 0.5*resultchi2);
           printf("lambda * H value = %.5f\n", lambda*resultH);
           printf("Phi value = %.5f\n\n", resultPhi);*/

        return resultPhi;
}


__host__ void MetropolisHasting(float2 *I, float2 *theta, int iterations, int *burndown_steps, int *accepted)
{
        PlantSeeds(-1);
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        curandState_t *states_0;
        curandState_t *states_1;
        //*states2;
        cudaMalloc((void**)&states_0, M*N*sizeof(curandState_t));
        cudaMalloc((void**)&states_1, M*N*sizeof(curandState_t));
        //cudaMalloc((void**)&states2, N*N*sizeof(curandState_t));
        float chi2_t_0, chi2_t_1;
        float delta_chi2 = 0.0f;
        float p = 0.0f;
        float un_rand = 0.0f;
        float s_d = (2.4*2.4)/valid_pixels;
        accepted_afterburndown = *accepted;

        /**************** GPU MEMORY FOR TOTAL OUT I_nu_0 and alpha ***************/

        gpuErrchk(cudaMalloc((void**)&M_k_out, sizeof(double2)*M*N));
        gpuErrchk(cudaMalloc((void**)&Q_k_out, sizeof(double2)*M*N));

        if(checkpoint) {
                gpuErrchk(cudaMemcpy(Q_k_out, host_Q_k, sizeof(double2)*M*N, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(M_k_out, host_M_k, sizeof(double2)*M*N, cudaMemcpyHostToDevice));
        }else{
                gpuErrchk(cudaMemset(Q_k_out, 0, sizeof(double2)*M*N));
                floatToDoubleKernel<<<numBlocksNN, threadsPerBlockNN>>>(M_k_out, I, N);
                gpuErrchk(cudaDeviceSynchronize());
        }


        /**************** GPU MEMORY FOR TEMPORAL I_nu_0 and alpha ***************/
        gpuErrchk(cudaMalloc((void**)&temp, sizeof(float2)*M*N));
        gpuErrchk(cudaMemset(temp, 0, sizeof(float2)*M*N));


        random_init<<<numBlocksNN, threadsPerBlockNN>>>(time(0), states_0, N);
        gpuErrchk(cudaDeviceSynchronize());
        random_init<<<numBlocksNN, threadsPerBlockNN>>>(time(0), states_1, N);
        gpuErrchk(cudaDeviceSynchronize());
        //random_init<<<numBlocksNN, threadsPerBlockNN>>>(time(0), states2, N);
        //gpuErrchk(cudaDeviceSynchronize());
        size_t file_chi2_n = snprintf(NULL, 0, "%schi2.txt", checkp) + 1;
        size_t file_iter_n = snprintf(NULL, 0, "%siter.txt", checkp) + 1;
        size_t file_shutdown_n = snprintf(NULL, 0, "%sshutdown.txt", checkp) + 1;
        char *chi2_file_name = (char*)malloc(sizeof(char)*file_chi2_n);
        char *iter_file_name = (char*)malloc(sizeof(char)*file_iter_n);
        shutdown_file_name = (char*)malloc(sizeof(char)*file_shutdown_n);

        snprintf(chi2_file_name, file_chi2_n*sizeof(char), "%schi2.txt", checkp, 0);
        snprintf(iter_file_name, file_iter_n*sizeof(char), "%siter.txt", checkp, 0);
        snprintf(shutdown_file_name, file_shutdown_n*sizeof(char), "%sshutdown.txt", checkp, 0);

        if(checkpoint)
                outfile = fopen(chi2_file_name, "a");
        else
                outfile = fopen(chi2_file_name, "w");

        outfile_its = fopen(iter_file_name, "w");

        position_in_file = ftell(outfile_its);
        randomize(pixels, valid_pixels);

        float n_I_nu_0;
        float n_alpha;

        for(real_iterations = 1; real_iterations <= iterations; real_iterations++) {

                if(adaptive && real_iterations >= *burndown_steps + 100) {
                        //__global__ void updateTheta(float2 *theta, double2 *total, double2 *total2, float s_d, int samples, long N)
                        updateTheta<<<numBlocksNN, threadsPerBlockNN>>>(theta, Q_k_out, s_d, accepted_afterburndown, N);
                        gpuErrchk(cudaDeviceSynchronize());
                }

                if(real_iterations == *burndown_steps) {
                        if(checkpoint) {
                                gpuErrchk(cudaMemcpy(M_k_out, host_M_k, sizeof(double2)*M*N, cudaMemcpyHostToDevice));
                        }
                        else
                        {
                                floatToDoubleKernel<<<numBlocksNN, threadsPerBlockNN>>>(M_k_out, I, N);
                                gpuErrchk(cudaDeviceSynchronize());
                        }
                }

                for(int j = 0; j < valid_pixels; j++) {

                        gpuErrchk(cudaMemcpy(temp, I, M*N*sizeof(float2), cudaMemcpyDeviceToDevice));

                        if(use_mask) {
                                SelectStream(0);
                                n_I_nu_0 = Normal(0.0, 1.0);
                                SelectStream(1);
                                n_alpha = Normal(0.0, 1.0);
                                changeGibbsEllipticalMaskAlpha<<<numBlocksNN, threadsPerBlockNN>>>(temp, theta, device_mask, n_I_nu_0, n_alpha, beam_bmaj, beam_bmin, beam_bpa, (1.0/3.0), 10.0, noise_jypix, pixels[j], DELTAX, DELTAY, N);
                                gpuErrchk(cudaDeviceSynchronize());
                        }else{
                                //changeGibbs<<<1, 1>>>(temp, theta, states, pixels[j], N);
                                if(spec_idx)
                                {
                                        SelectStream(0);
                                        n_I_nu_0 = Normal(0.0, 1.0);
                                        SelectStream(1);
                                        n_alpha = Normal(0.0, 1.0);
                                        changeGibbsEllipticalGaussianSpecIdx<<<numBlocksNN, threadsPerBlockNN>>>(temp, theta, n_I_nu_0, n_alpha, beam_bmaj, beam_bmin, beam_bpa, (1.0/3.0), pixels[j], DELTAX, DELTAY, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }else{
                                        SelectStream(0);
                                        n_I_nu_0 = Normal(0.0, 1.0);
                                        changeGibbsEllipticalGaussian<<<numBlocksNN, threadsPerBlockNN>>>(temp, theta, n_I_nu_0, beam_bmaj, beam_bmin, beam_bpa, (1.0/3.0), pixels[j], DELTAX, DELTAY, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }

                        }

                        chi2_t_0 = chiCuadrado(I);
                        chi2_t_1 = chiCuadrado(temp);
                        //printf("chi2_t0: %f\n", chi2_t_0);
                        //printf("chi2_t1: %f\n", chi2_t_1);
                        fprintf(outfile, "%f\n", chi2_t_0);
                        delta_chi2 = chi2_t_1 - chi2_t_0;
                        if(delta_chi2 <= 0) {
                                //printf("Acccepted Delta chi2: %f\n", delta_chi2);
                                gpuErrchk(cudaMemcpy(I, temp, sizeof(float2)*M*N, cudaMemcpyDeviceToDevice));
                                /*if(print_images && i%3 == 0)
                                   float2toImage(I, mod_in, out_image, mempath, i, M, N, 1);*/
                                if(real_iterations >= *burndown_steps) {
                                        accepted_afterburndown++;
                                        sumI<<<numBlocksNN, threadsPerBlockNN>>>(M_k_out, Q_k_out, I, accepted_afterburndown, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }

                        }
                        else{
                                //printf("Not Accepted Delta chi2: %f\n", delta_chi2);
                                SelectStream(2);
                                un_rand = Random();
                                if(-log(un_rand) > delta_chi2) {
                                        gpuErrchk(cudaMemcpy(I, temp, sizeof(float2)*M*N, cudaMemcpyDeviceToDevice));
                                        if(real_iterations >= *burndown_steps) {
                                                accepted_afterburndown++;
                                                sumI<<<numBlocksNN, threadsPerBlockNN>>>(M_k_out, Q_k_out, I, accepted_afterburndown, N);
                                                gpuErrchk(cudaDeviceSynchronize());
                                        }
                                }
                        }
                }

                printf("--------------Iteration %d-----------\n", real_iterations);
                printf("From %d valid pixels, accepted :%d\n", valid_pixels, accepted_afterburndown);
                fseek(outfile_its,position_in_file,SEEK_SET);
                fprintf(outfile_its, "Iterations: %d\n", real_iterations);
                fprintf(outfile_its, "Accepted after burndown: %d\n", accepted_afterburndown);
                fflush(outfile_its);
                double2toImage(M_k_out, mod_in, out_image, checkp, 0, M, N, 1.0, accepted_afterburndown, 1);
                double2toImage(Q_k_out, mod_in, out_image, checkp, 1, M, N, 1.0/(accepted_afterburndown-1), accepted_afterburndown, 1);
                float2toImage(I, mod_in, out_image, checkp, 2, M, N, 1.0, 1);
                randomize(pixels, valid_pixels);
        }

        printf("ACCEPTED AFTER BURNDOWN: %d\n", accepted_afterburndown);
        fprintf(outfile_its, "%d\n", accepted_afterburndown);

        double2toImage(M_k_out, mod_in, out_image, checkp, 0, M, N, 1.0, accepted_afterburndown, 1);
        double2toImage(Q_k_out, mod_in, out_image, checkp, 1, M, N, 1.0/(accepted_afterburndown-1), accepted_afterburndown, 1);
        float2toImage(I, mod_in, out_image, checkp, 2, M, N, 1.0, 1);

        signal(SIGINT, sig_handler);
        signal(SIGTERM, sig_handler);
        signal(SIGKILL, sig_handler);

        fclose(outfile);
        fclose(outfile_its);
        cudaFree(temp);
        cudaFree(states_0);
        cudaFree(states_1);

}


__host__ void Metropolis(float2 *I, float2 *theta, int iterations, int burndown_steps, int accepted)
{
        PlantSeeds(-1);
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        curandState_t *states_0;
        curandState_t *states_1;
        //*states2;
        cudaMalloc((void**)&states_0, M*N*sizeof(curandState_t));
        cudaMalloc((void**)&states_1, M*N*sizeof(curandState_t));
        //cudaMalloc((void**)&states2, N*N*sizeof(curandState_t));
        float chi2_t_0, chi2_t_1;
        float delta_chi2 = 0.0f;
        float p = 0.0f;
        float un_rand = 0.0f;
        float s_d = (2.4*2.4)/valid_pixels;
        accepted_afterburndown = accepted;


        /**************** GPU MEMORY FOR TOTAL OUT I_nu_0 and alpha ***************/

        gpuErrchk(cudaMalloc((void**)&M_k_out, sizeof(double2)*M*N));
        gpuErrchk(cudaMalloc((void**)&Q_k_out, sizeof(double2)*M*N));

        if(checkpoint) {
                gpuErrchk(cudaMemcpy(Q_k_out, host_Q_k, sizeof(double2)*M*N, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(M_k_out, host_M_k, sizeof(double2)*M*N, cudaMemcpyHostToDevice));
        }else{
                gpuErrchk(cudaMemset(Q_k_out, 0, sizeof(double2)*M*N));
                floatToDoubleKernel<<<numBlocksNN, threadsPerBlockNN>>>(M_k_out, I, N);
                gpuErrchk(cudaDeviceSynchronize());
        }


        /**************** GPU MEMORY FOR TEMPORAL I_nu_0 and alpha ***************/
        gpuErrchk(cudaMalloc((void**)&temp, sizeof(float2)*M*N));
        gpuErrchk(cudaMemset(temp, 0, sizeof(float2)*M*N));

        random_init<<<numBlocksNN, threadsPerBlockNN>>>(time(0), states_0, N);
        gpuErrchk(cudaDeviceSynchronize());
        random_init<<<numBlocksNN, threadsPerBlockNN>>>(time(0), states_1, N);
        gpuErrchk(cudaDeviceSynchronize());
        //random_init<<<numBlocksNN, threadsPerBlockNN>>>(time(0), states2, N);
        //gpuErrchk(cudaDeviceSynchronize());
        size_t file_chi2_n = snprintf(NULL, 0, "%schi2.txt", checkp) + 1;
        size_t file_iter_n = snprintf(NULL, 0, "%siter.txt", checkp) + 1;
        size_t file_shutdown_n = snprintf(NULL, 0, "%sshutdown.txt", checkp) + 1;
        char *chi2_file_name = (char*)malloc(sizeof(char)*file_chi2_n);
        char *iter_file_name = (char*)malloc(sizeof(char)*file_iter_n);
        shutdown_file_name = (char*)malloc(sizeof(char)*file_shutdown_n);

        snprintf(chi2_file_name, file_chi2_n*sizeof(char), "%schi2.txt", checkp, 0);
        snprintf(iter_file_name, file_iter_n*sizeof(char), "%siter.txt", checkp, 0);
        snprintf(shutdown_file_name, file_shutdown_n*sizeof(char), "%sshutdown.txt", checkp, 0);

        if(checkpoint)
                outfile = fopen(chi2_file_name, "a");
        else
                outfile = fopen(chi2_file_name, "w");

        outfile_its = fopen(iter_file_name, "w");

        position_in_file = ftell(outfile_its);
        randomize(pixels, valid_pixels);

        float n_I_nu_0;
        float n_alpha;
        for(real_iterations = 1; real_iterations<= iterations; real_iterations++) {
                if(adaptive && real_iterations >= burndown_steps + 100) {
                        //__global__ void updateTheta(float2 *theta, double2 *total, double2 *total2, float s_d, int samples, long N)
                        updateTheta<<<numBlocksNN, threadsPerBlockNN>>>(theta, Q_k_out, s_d, accepted_afterburndown, N);
                        gpuErrchk(cudaDeviceSynchronize());
                }

                if(real_iterations == burndown_steps) {
                        if(checkpoint) {
                                gpuErrchk(cudaMemcpy(M_k_out, host_M_k, sizeof(double2)*M*N, cudaMemcpyHostToDevice));
                        }
                        else
                        {
                                floatToDoubleKernel<<<numBlocksNN, threadsPerBlockNN>>>(M_k_out, I, N);
                                gpuErrchk(cudaDeviceSynchronize());
                        }
                }

                for(int j = 0; j < valid_pixels; j++) {

                        //if(i>0){
                        //  calculateTheta<<<numBlocksNN, threadsPerBlockNN>>>(theta, states2, total, total2, accepted+1, N);
                        //  gpuErrchk(cudaDeviceSynchronize());
                        //}
                        gpuErrchk(cudaMemcpy(temp, I, M*N*sizeof(float2), cudaMemcpyDeviceToDevice));

                        if(use_mask) {
                                SelectStream(0);
                                n_I_nu_0 = Normal(0.0, 1.0);
                                SelectStream(1);
                                n_alpha = Normal(0.0, 1.0);
                                changeGibbsEllipticalMaskAlpha<<<numBlocksNN, threadsPerBlockNN>>>(temp, theta, device_mask, n_I_nu_0, n_alpha, beam_bmaj, beam_bmin, beam_bpa, (1.0/3.0), 10.0, noise_jypix, pixels[j], DELTAX, DELTAY, N);
                                gpuErrchk(cudaDeviceSynchronize());
                        }else{
                                //changeGibbs<<<1, 1>>>(temp, theta, states, pixels[j], N);
                                if(spec_idx)
                                {
                                        SelectStream(0);
                                        n_I_nu_0 = Normal(0.0, 1.0);
                                        SelectStream(1);
                                        n_alpha = Normal(0.0, 1.0);
                                        changeGibbsEllipticalGaussianSpecIdx<<<numBlocksNN, threadsPerBlockNN>>>(temp, theta, n_I_nu_0, n_alpha, beam_bmaj, beam_bmin, beam_bpa, (1.0/3.0), pixels[j], DELTAX, DELTAY, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }else{
                                        SelectStream(0);
                                        n_I_nu_0 = Normal(0.0, 1.0);
                                        changeGibbsEllipticalGaussian<<<numBlocksNN, threadsPerBlockNN>>>(temp, theta, n_I_nu_0, beam_bmaj, beam_bmin, beam_bpa, (1.0/3.0), pixels[j], DELTAX, DELTAY, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }

                        }


                        chi2_t_0 = chiCuadrado(I);
                        chi2_t_1 = chiCuadrado(temp);
                        //printf("chi2_t0: %f\n", chi2_t_0);
                        //printf("chi2_t1: %f\n", chi2_t_1);
                        fprintf(outfile, "%f\n", chi2_t_0);
                        fflush(outfile);
                        delta_chi2 = chi2_t_1 - chi2_t_0;
                        if(delta_chi2 <= 0.0f) {
                                //printf("Accepted Delta chi2: %f\n", delta_chi2);
                                gpuErrchk(cudaMemcpy(I, temp, sizeof(float2)*M*N, cudaMemcpyDeviceToDevice));
                                /*if(print_images && i%3 == 0)
                                   float2toImage(I, mod_in, out_image, mempath, i, M, N, 1);*/
                                if(real_iterations >= burndown_steps) {
                                        accepted_afterburndown++;
                                        sumI<<<numBlocksNN, threadsPerBlockNN>>>(M_k_out, Q_k_out, I, accepted_afterburndown, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }

                        }
                        /*else{
                                //printf("Not Accepted Delta chi2: %f\n", delta_chi2);
                                //l = exp(-delta_chi2);
                                //p = chi2_t_0 / chi2_t_1;
                                un_rand = Random();
                                if(-log(un_rand) > delta_chi2) {
                                        //printf("*****  P = %f > %f   *******\n", -log(un_rand), delta_chi2);
                                        gpuErrchk(cudaMemcpy2D(I, sizeof(float2), temp, sizeof(float2), sizeof(float2), M*N, cudaMemcpyDeviceToDevice));
                                        if(print_images && i%3 == 0)
                                           float2toImage(I, mod_in, out_image, mempath, i, M, N, 1);
                                        if(real_iterations >= burndown_steps) {
                                                sumI<<<numBlocksNN, threadsPerBlockNN>>>(total_out, total2_out, I, N);
                                                gpuErrchk(cudaDeviceSynchronize());
                                                accepted_afterburndown++;
                                        }
                                        second_pass++;

                                }
                           }*/
                }

                printf("--------------Iteration %d-----------\n", real_iterations);
                printf("From %d valid pixels, accepted :%d\n", valid_pixels, accepted_afterburndown);
                fseek(outfile_its,position_in_file,SEEK_SET);
                fprintf(outfile_its, "Iterations: %d\n", real_iterations);
                fprintf(outfile_its, "Accepted after burndown: %d\n", accepted_afterburndown);
                fflush(outfile_its);
                double2toImage(M_k_out, mod_in, out_image, checkp, 0, M, N, 1.0, accepted_afterburndown, 1);
                double2toImage(Q_k_out, mod_in, out_image, checkp, 1, M, N, 1.0/(accepted_afterburndown-1), accepted_afterburndown, 1);
                float2toImage(I, mod_in, out_image, checkp, 2, M, N, 1.0, 1);
                randomize(pixels, valid_pixels);
        }

        //avgI<<<numBlocksNN, threadsPerBlockNN>>>(total_out, total2_out, accepted_afterburndown, N);
        //gpuErrchk(cudaDeviceSynchronize());
        printf("ACCEPTED AFTER BURNDOWN: %d\n", accepted_afterburndown);
        fprintf(outfile_its, "%d\n", accepted_afterburndown);

        double2toImage(M_k_out, mod_in, out_image, checkp, 0, M, N, 1.0, accepted_afterburndown, 1);
        double2toImage(Q_k_out, mod_in, out_image, checkp, 1, M, N, 1.0/(accepted_afterburndown-1), accepted_afterburndown, 1);
        float2toImage(I, mod_in, out_image, checkp, 2, M, N, 1.0, 1);

        signal(SIGINT, sig_handler);
        signal(SIGTERM, sig_handler);
        signal(SIGKILL, sig_handler);

        fclose(outfile);
        fclose(outfile_its);
        cudaFree(temp);
        cudaFree(states_0);
        cudaFree(states_1);

}
