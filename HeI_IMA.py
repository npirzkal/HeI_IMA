from __future__ import print_function  
import glob, sys, os
import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy import optimize

def HeI_estimate(ima_names,filt,border=25,niter=20,sigma=2,minsize = 3):
    """Estimate and subtract the HeI contribution from each IMSET extensions of a list of IMA files.
    The files need to be from the same HST visit, as this assumes a constant Zodi component level.
    ima_names: A list of IMA files
    filt: G102 or G141
    border: Number of columns on the left of the detector to Ignore. Default=25
    ninter: Maximumn number of iterations. Default=20
    sigma: Threshold over the background for a pixel to flagged as an object. Default=2
    minsize: Minimum size of an object spectrum. Default=3
    """

    ima_names0 = []
    for f in ima_names:
        filt0 = fits.open(f)[0].header["FILTER"]
        if filt0!="G102" and filt0!="G141":
            continue
        ima_names0.append(f)
    ima_names = ima_names0

    nexts = [fits.open(ima_name)[-1].header["EXTVER"] for ima_name in ima_names]

    nimas = len(ima_names)

    print("Processing:",sum(nexts),"IMSETSs in ",nimas," IMA files.")

    border = int(border)
    print("Grism:",filt)
    print("Avoiding a border of ",border," pixels")

    path = os.path.split(__file__)[0]

    # Background components for the two grisms. These are the components described in ISR WFC3-2015-17 but the broad band filter flat-field
    # was de-applied, and the default grism flatfield was applied, i.e. the quandrant gain compensation.
    if filt=="G102":
            sky_names  = [os.path.join(path,"zodi_G102_clean_np.fits"),os.path.join(path,"excess_G102_clean_np.fits")]
    if filt=="G141":
            sky_names = [os.path.join(path,"zodi_G141_clean_np.fits"),os.path.join(path,"excess_lo_G141_clean_np.fits")] 

    # Setting up DQ mask values to exclude
    bit_mask = (4+8+16+32+128+256+1024+2048+8192)

    # Reading in all DATA, ERR, and DQ extensions and create masks
    data0s = []
    err0s = []
    samp0s = []
    dq0s = []
    dqm0s = []
    for j in range(nimas):
        print("Loading ",ima_names[j],nexts[j],"IMSETs")
        data0s.append([fits.open(ima_names[j])["SCI",ext].data[5:1014+5,5:1014+5] for ext in range(1,nexts[j])])
        err0s.append([fits.open(ima_names[j])["ERR",ext].data[5:1014+5,5:1014+5] for ext in range(1,nexts[j])])
        samp0s.append([fits.open(ima_names[j])["SCI",ext].header["SAMPTIME"] for ext in range(1,nexts[j])])
        dq0s.append([fits.open(ima_names[j])["DQ",ext].data[5:1014+5,5:1014+5] for ext in range(1,nexts[j])])
    dqm0s = [[np.bitwise_and(dq0,np.zeros(np.shape(dq0),'Int16')+ bit_mask) for dq0 in dq0s[j]] for j in range(nimas)]

    # Setting up image weights
    whts = []
    for j in range(len(ima_names)):
        whts_j = []
        for i in range(len(err0s[j])):
            err = err0s[j][i]
            err[err<=1e-10] = 1e-10
            whts_j.append(1./err**2)
        whts.append(whts_j)

    bcks = [fits.open(xx)[0].data for xx in sky_names]


    zodi = bcks[0]

    npar = sum(nexts)
    print("Iteratively solving for ",npar," HeI values and 1 Zodi value...")
    
    # Setting up Sky arrays
    sky0s = []
    for j in range(len(ima_names)):
        sky0s_j = []
        for i in range(len(err0s[j])):
            data0 = data0s[j][i]
            s1 = np.ones(np.shape(data0)[0])
            s = np.median(data0,axis=0)
            sky = np.outer(s1,s)
            sky0s_j.append(sky)
        sky0s.append(sky0s_j)

    # Setting up a dillation operator
    ys,xs = np.indices([5,5])
    kernel = np.sqrt((xs-2.)**2 + (ys-2.)**2)
    kernel[kernel<=2] = 1.


    old_x = None
    # Start iteratively solving and masking objects
    for iter in range(niter):
        print("# Iteration",iter+1)
        v = np.zeros(npar)
        m = np.zeros([npar,npar])
        cwhts = []

        ii = -1
        for j in range(len(ima_names)):

            # Setup mask/weight for this ima file
            img = data0s[j][0]*1
            sky = sky0s[j][0]*1
            wht = whts[j][0]*1

            # Add pixels > sigma into a new object mask for this dataset
            sig = np.abs(img-sky)*np.sqrt(wht)
            vg = (sig > sigma) 
            objmask = np.zeros(np.shape(img))
            objmask[vg] = 1
            
            # Find all contiguous pixels and label them
            labeled,nr_objects = ndimage.label(objmask>0)

            # Remove objects/spectra that have less than minsize number of pixels
            hist = ndimage.measurements.histogram(labeled,1,nr_objects,nr_objects)
            small_objs = np.arange(1,nr_objects+1)[hist<minsize]
            labeled = np.where(np.in1d(labeled, small_objs).reshape(labeled.shape), 0, labeled)
            
            # Dillate object mask using the kernel we set up earlier.
            labeled = ndimage.morphology.binary_dilation(labeled,kernel).astype(labeled.dtype)

            # Set weight of pixels with objects/spectra to 0
            wht[labeled>0] = 0
            # Set weigth of known bad (DQ) pixels to 0
            wht[dqm0s[j][0]!=0] = 0.
            # Ignore pixels within the desired x-border
            wht[:,:border] = 0.
            
            # Save this dataset's weight
            cwhts.append(wht)
            
            for i in range(len(data0s[j])):
                ii = ii + 1
    
                img = data0s[j][i]
                sky = sky0s[j][i]
                    
                # Populate up matrix and vector
                v[ii] = np.sum(wht*data0s[j][i]*bcks[1])
                v[-1] += np.sum(wht*data0s[j][i]*zodi)


                m[ii,ii] = np.sum(wht*bcks[1]*bcks[1])
                m[ii,-1] = np.sum(wht*bcks[1]*zodi)
                m[-1,ii] = m[ii,-1]
                m[-1,-1] += np.sum(wht*zodi*zodi)


        # Solve v = m * x using positive non negative least square (nnls)
        # Solution is unique but we seek positive contributions so we use nnls.
        x,r = optimize.nnls(m,v)

        
        Zodi = x[-1]
        HeIs = {}
        ii = -1
        for j in range(len(data0s)):
            HeIs[ima_names[j]] = {}
            for i in range(len(data0s[j])):
                ii = ii + 1

                print("%s IMSET:%2d Zodi: %3.3f e-/s (%5.2f e-) He: %3.3f e-/s (%5.2f e-) Total: %5.2f e-" % (ima_names[j],i,x[-1],x[-1]*samp0s[j][i],x[ii],x[ii]*samp0s[j][i],np.median(data0s[j][i].ravel())*samp0s[j][i]))
                
                HeIs[ima_names[j]][i+1] = x[ii]
                sky0s[j][i] = x[-1]*zodi + x[ii]*bcks[1]

        # If the background and X have converged then we stop
        if np.array_equal(old_x,x):
            print("Converged!")
            break
        else:
            old_x = x 

    # Once we have HeI estimates for each IMSET of every IMA file, we subtract those
    HeI_data = bcks[1]
    for f in HeIs.keys():
        print("Updating ",f)
        fin = fits.open(f,mode="update")

        for extver in HeIs[f].keys():
            try:
                val = fin["SCI",extver].header["HeI"] # testing
                print("HeI found in ",f,"Aborting..")
                sys.exit(1)
            except:
                pass
    
            print(f,"IMSET:",extver,"subtracting",HeIs[f][extver])
            fin["SCI",extver].data[5:1014+5,5:1014+5] = fin["SCI",extver].data[5:1014+5,5:1014+5] - HeIs[f][extver]*HeI_data 
            fin["SCI",extver].header["HeI"] = (HeIs[f][extver],"HeI level subtracted (e-/s)")
    

        fin.close()
    
if __name__=="__main__":
    IMA_names = glob.glob(sys.argv[1])
    grism = sys.argv[2]

    HeI_estimate(IMA_names,grism)
