import sys
import os
from casatasks import tclean, exportfits, imhead, importfits, immath, imstat
from casatools import image
from casatools import quanta


def restore(model_fits="",
            residual_ms="",
            restored_image="restored",
            stokes="I",
            weighting="robust",
            robust=2.0):
    ia = image()
    qa = quanta()
    residual_image = residual_ms.partition(".ms")[0] + ".residual"
    residual_casa_image = residual_image + ".image"

    os.system("rm -rf *.log *.last " + residual_image +
              ".* mod_out convolved_mod_out convolved_mod_out.fits " +
              restored_image + " " + restored_image + ".fits")

    importfits(imagename="model_out", fitsimage=model_fits, overwrite=True)
    cdelt2 = imhead(imagename="model_out", mode="get", hdkey="cdelt2")["value"]
    cdelt_string = str(qa.convert(v=cdelt2,
                                  outunit="arcsec")["value"]) + "arcsec"
    size = imhead(imagename="model_out", mode="get", hdkey="shape")

    tclean(vis=residual_ms,
           imagename=residual_image,
           specmode='mfs',
           deconvolver='hogbom',
           niter=0,
           stokes=stokes,
           nterms=1,
           weighting=weighting,
           robust=robust,
           imsize=[size[0], size[1]],
           cell=cdelt_string,
           datacolumn='data')

    exportfits(imagename=residual_casa_image,
               fitsimage=residual_casa_image + ".fits",
               overwrite=True,
               history=False)

    ia.open(infile=residual_casa_image)
    record_beam = ia.restoringbeam()
    ia.done()
    ia.close()

    ia.open(infile="model_out")
    im2 = ia.convolve2d(outfile="convolved_model_out",
                        axes=[0, 1],
                        type='gauss',
                        major=record_beam["major"],
                        minor=record_beam["minor"],
                        pa=record_beam["positionangle"],
                        overwrite=True)
    im2.done()
    ia.done()
    ia.close()

    ia.open(infile="convolved_model_out")
    ia.setrestoringbeam(remove=True)
    ia.setrestoringbeam(beam=record_beam)
    ia.done()
    ia.close()

    image_name_list = ["convolved_model_out", residual_casa_image + ".fits"]

    immath(imagename=image_name_list,
           expr=" (IM0   + IM1) ",
           outfile=restored_image,
           imagemd=residual_casa_image + ".fits")

    exportfits(imagename=restored_image,
               fitsimage=restored_image + ".fits",
               overwrite=True,
               history=False)

    peak = imstat(imagename=restored_image)["max"][0]
    rms = imstat(imagename=residual_casa_image)["rms"][0]
    psnr = peak / rms
    return psnr, peak, rms


model_fits = sys.argv[1]
residual_ms = sys.argv[2]
restored = sys.argv[3]
weighting = sys.argv[4]
robust = float(sys.argv[5])
psnr, peak, rms = restore(model_fits=model_fits,
                          residual_ms=residual_ms,
                          restored_image=restored,
                          stokes="I",
                          weighting=weighting,
                          robust=robust)
print("PSNR: {0:0.3f}".format(psnr))
print("Peak: {0:0.3f} mJy/beam".format(peak * 1000.0))
print("RMS: {0:0.3f} mJy/beam".format(rms * 1000.0))
