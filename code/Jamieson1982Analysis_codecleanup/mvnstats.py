import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy.stats import multivariate_normal as mvn_f
import numpy.linalg as linalg


#====================================================================
# SECT 1: Matrix Manipulation Functions
#  These are sourced from the Matrix Cookbook @Petersen2008
#====================================================================
def init_mvn( mean, cov ):
    return mvn_f( mean, cov )

#====================================================================
def product_density( mvn_l ):
    """
    Calculate product of Multivariate Normal densities.

    Return resulting mvn distribution AND weighting factor
    """

    N = len(mvn_l)

    # Extract the first mvn distribution
    mvn_prod = mvn_l[0]
    wt_prod = 1.0

    # Loop over remaining mvn distributions in list
    # combine one pair at a time
    for imvn in mvn_l[1:]:
        # Can be made more efficient
        mean1 = mvn_prod.mean
        cov1 = mvn_prod.cov
        invcov1 = linalg.inv(cov1)

        mean2 = imvn.mean
        cov2 = imvn.cov
        invcov2 = linalg.inv(cov2)

        iwt_prod = init_mvn( mean1, cov1+cov2 ).pdf( mean2 )
        wt_prod = wt_prod*iwt_prod

        icov_prod = linalg.inv(invcov1+invcov2)
        imean_prod = np.dot( icov_prod, invcov1*mean1+invcov2*mean2 )

        mvn_prod = init_mvn( imean_prod, icov_prod )

    return mvn_prod, wt_prod

#====================================================================
if __name__ == "__main__":
    main()
