# DESCRIPTOR CONFIGURATION

# LBP Parameters
lbp_win_shape = (64, 64)
lbp_win_step = lbp_win_shape[0]/2
lbp_radius = 1
lbp_n_points = 8 * lbp_radius
lbp_METHOD = 'nri_uniform' # "nri" means non-rotation invariant
lbp_n_bins = 59 # NRI uniform LBP has 59 values

# HOG Parameters
hog_orientations = 9
hog_pixels_per_cell = (16, 16)
hog_cells_per_block = (2, 2)
hog_cells_per_block_total = hog_cells_per_block[0]*hog_cells_per_block[1]
hog_normalise = True