# HW 3 Frequency Domain Filtering
# ECE5470 Digital Image Processing

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
#np.set_printoptions(threshold=np.inf)

#-------------------------#
# Supporting function(s)  #
#-------------------------#

# error check on user input
def get_const():
    while True:
        try:
            const = int(input("\nEnter constant between [0,255] for padding: "))
        except ValueError:
            print("\nNot a number, input constant between [0,255].\n")
        else:
            # check if constant is in range
            if const in range(0,256):
                break
            else:
                print("\nConstant not in range. Re-input constant between [0,255].\n")
    return const

# Center fourier
def center_f ( P , Q ):

    cm = np.zeros( (P,Q) )

    for i in range (0,P):
        for j in range (0,Q):
            cm[i][j] = ((-1)**(i + j))

    #return np.uint8(cm)
    return cm

# create distance matrix of size PxQ for filter gen
def d_helper( P, Q ):

    dmatrix = np.zeros((P,Q))

    xc = (P-1)/2
    yc = (Q-1)/2

    for i in range (0, P):
        for j in range (0, Q):
            dmatrix[i][j] = np.sqrt(((xc - i)**2)+((yc - j)**2))

    return dmatrix

# Generate Gaussian LPF of size PxQ
def gaussian_h ( d_matrix, sigma ):

    (p,q) = d_matrix.shape

    g_matrix = np.zeros((p,q))

    g_matrix = np.exp(-(d_matrix**2)/(2 * (sigma ** 2)))

    return g_matrix

# Generate Laplacian of size PxQ
def laplacian_h (d_matrix):

    (p,q) = d_matrix.shape

    l_matrix = np.zeros((p,q))

    l_matrix = -4 * (np.pi ** 2) * (d_matrix ** 2)

    return l_matrix

# plot fourier image
# https://github.com/damian-dz/PythonTutorials
def plot_fimg (img):
    # move the zero-frequency component to the center of the Fourier spectrum
    #fourier = np.fft.fftshift(img)
    # compute the magnitudes (absolute values) of the complex numbers
    fourier = np.abs(img)
    # compute the common logarithm of each value to reduce the dynamic range
    fourier = np.log10(fourier)
    # find the minimum value that is a finite number
    lowest = np.nanmin(fourier[np.isfinite(fourier)])
    # find the maximum value that is a finite number
    highest = np.nanmax(fourier[np.isfinite(fourier)])
    # calculate the original contrast range
    original_range = highest - lowest
    # normalize the Fourier image data ("stretch" the contrast)
    norm_fourier = (fourier - lowest) / original_range * 255
    # convert the normalized data into an image
    norm_fourier_img = Image.fromarray(norm_fourier)

    return norm_fourier_img

####################################
#    Frequency Domain Filtering    #
#            f = image             #
#      padding = padding type      #
#      "zero" = zero padding       #
#   "constant" = constant padding  #
#         - requires user input    #
#         - values from 0 to 255   #
#       "wrap" = cyclic padding    #
#    "symmetric" = mirror padding  #
#   g = filt_img = filtered image  #
####################################

def freqFilter (f, padding, sig, c):

    ########################
    #       Step 1         #
    #    P = 2M, Q = 2N    #
    ########################

    # image img, m by n, find dimensions of image
    # m rows, n columns
    (m,n) = (f.shape)

    p = 2 * m
    q = 2 * n

    ########################
    #       Step 2         #
    #   fp = padded image  #
    #   fp = size P x Q    #
    ########################

    # original image will stay in the top left corner
    # pad to the right and below image
    p_size = (0,p-m)
    q_size = (0,q-n)

    if (padding == "zero"):
        fp = np.pad(f, (p_size,q_size), "constant", constant_values=0)

    elif (padding == "constant"):
        constvalue = get_const()
        fp = np.pad(f, (p_size,q_size), padding, constant_values= constvalue)

    else:
        fp = np.pad(f, (p_size,q_size), padding)

    ########################
    #       Step 3         #
    # fp = fp * (-1)^(x+y) #
    #   to center fourier  #
    ########################

    #normalize image between [0,1]
    fp = (fp - np.min(fp)) / (np.max(fp) - np.min(fp))

    centering_matrix = center_f(p,q)

    fp = fp * centering_matrix

    ########################
    #       Step 4         #
    #  compute DFT F(u,v)  #
    #        of fp         #
    ########################

    F_p = np.fft.fft2(fp)

    fimg = plot_fimg(F_p)

    ########################
    #       Step 5         #
    #  Construct a real,   #
    #  symmetric filter    #
    #  transfer function   #
    #  H(u,v) of size PxQ  #
    # centered @ (P/2,Q/2) #
    ########################

    # generate helper matrix for distances
    dist_matrix = d_helper(p,q)

    gaus_matrix = gaussian_h(dist_matrix, sig)

    # normalize gaussian matrix between [0,1]
    gaus_matrix = (gaus_matrix - np.min(gaus_matrix)) / (np.max(gaus_matrix) - np.min(gaus_matrix))

    lap_matrix = laplacian_h(dist_matrix)

    # normalize lap matrix by dividing by max value (notes part 10)
    lap_matrix = (lap_matrix - np.min(lap_matrix)) / (np.max(lap_matrix) - np.min(lap_matrix))
    #print(lap_matrix1)

    lap_matrix = lap_matrix / np.max(lap_matrix)
    # scale between -1 to 1
    lap_matrix = 2*(lap_matrix - np.min(lap_matrix))/np.ptp(lap_matrix)-1
    #print(lap_matrix)
    ########################
    #       Step 6         #
    #  Calculate G(u,v)    #
    # G(u,v) = H(u,v)F(u,v)#
    #  using elementwise   #
    #    multiplication    #
    ########################

    # LoG: G(u,v) = F(u,v) * N(u,v) * L(u,v)
    Guv = F_p * gaus_matrix * (lap_matrix * c)

    logimg = plot_fimg(Guv)

    ########################
    #       Step 7         #
    #  obtain the filtered #
    #  image by computing  #
    #  the IDFT of G(u,v)  #
    ########################

    # discard imaginary
    gp = np.fft.ifft2(Guv).real

    # uncenter to remove checkerboard effect on image
    gp = gp * centering_matrix

    ########################
    #       Step 8         #
    #   obtain the final   #
    #  filtered image g by #
    #  extracting the MxN  #
    #  region from the top #
    #      left of gp      #
    ########################

    # unpadding image
    filt_img = abs(gp[0:m, 0:n])
    #filt_img = (gp[0:m, 0:n])

    # scale between [0,255]
    filt_img = 2 * (filt_img - np.min(filt_img)) / np.ptp(filt_img) - 1

    # display images
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.axis('off')
    plt.imshow(f, cmap="gray")

    plt.subplot(2, 2, 2)
    plt.title("Fourier Transformed Image (Padded and Centered)")
    plt.axis('off')
    plt.imshow(fimg, cmap="gray")

    plt.subplot(2, 2, 3)
    plt.title("LoG Image (Fourier Domain), $\sigma$ = %0.2f" % (sig))
    plt.axis('off')
    plt.imshow(logimg, cmap="gray")

    plt.subplot(2, 2, 4)
    plt.title("LoG Image (Spatial Domain, Uncentered, Unpadded)")
    plt.axis('off')
    plt.imshow(filt_img, cmap="gray")

    plt.show()

#image = cv2.imread('dog.jpg', 0)
image = cv2.imread('image.png', 0)
padding =  'zero'
sig = 120
c = -1

freqFilter(image, padding, sig, c)