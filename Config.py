# DESCRIPTOR CONFIGURATION

# LBP Parameters
lbp_win_shape = (16, 16)
lbp_win_step = lbp_win_shape[0]/2
lbp_radius = 1
lbp_n_points = 8 * lbp_radius
lbp_METHOD = 'nri_uniform' # "nri" means non-rotation invariant
lbp_n_bins = 59 # NRI uniform LBP has 59 values

# HOG Parameters
hog_orientations = 9
hog_pixels_per_cell = (64, 64)
hog_cells_per_block = (2, 2)
hog_cells_per_block_total = hog_cells_per_block[0]*hog_cells_per_block[1]
hog_normalise = True

# PCA Parameters
pca_ncomponents = 80# PCA Parameters
pca_ncomponents_sift = 15
pca_ncomponents_color = 10
pca_ncomponents_earlyf = 80