
def plot_SD_pols(arr_proj, isl_x, isl_y, isl_pols, ax = None, **kwags):
    corrected_arr = np.transpose(arr_proj)
    # Display heatmap
    ax = ax or plt.gca()
    # fig,ax = plt.subplots(1, figsize = (15,10))
    # Do some cool data transformations..
    plt.imshow(corrected_arr, cmap = 'magma', origin = "lower", **kwags)

    max_val = np.max(corrected_arr)
    max_dot_size = 4

    # half_max = max_val/2
    for xx, yy, polarity in zip(isl_x, isl_y, isl_pols):
        # print(xx, yy)
        if math.isnan(xx) is True or math.isnan(yy) is True:
            # This basically handles the nans 
            pass
        else:
            if polarity == 1:
                magnitude = corrected_arr[int(yy), int(xx)] 
                mag_percentage = magnitude/max_val
                circ = Circle((int(xx),int(yy)), max_dot_size * mag_percentage, fc = None, ec = 'b', fill = 0, zorder = 0)
                ax.add_patch(circ)
            if polarity == -1:
                magnitude = corrected_arr[int(yy), int(xx)] 
                mag_percentage = magnitude/max_val
                circ_alt = Circle((int(xx),int(yy)), max_dot_size * mag_percentage, fc = None, ec = 'r', fill = 0, zorder = 0)
                ax.add_patch(circ_alt)

    plt.show()
    # return ax
    # plt.colorbar()
    # plt.show()
# plot_SD_pols(f["STRF_Corr0"][:,:,STRF_num])

def roi_summary_plot(h5py_obj, STRF_num):
    fig, axs = plt.subplots(2, 3,  figsize=(25, 10))
    fig.suptitle("Overview for ROI")
    axs[0, 0].bar(range(len(h5py_obj.get("IslandAmpls")[:,STRF_num])), h5py_obj.get("IslandAmpls")[:,STRF_num])
    axs[0, 0].set_title("Island amplitudes")

    def polarise_amps(amps, pols):
        amps_w_pols = np.zeros(amps.shape)
        for n, amp_pol in enumerate(zip(amps, pols)):
            amps_w_pols[n] = amp_pol[0] * amp_pol[1]
        return amps_w_pols
    v = polarise_amps(h5py_obj["IslandAmpls"], h5py_obj["IslandTimeKernels_Polarities"])
    axs[0,1].bar(np.arange(0, len(v[:, STRF_num][:])), v[:, STRF_num])
    axs[0, 1].set_title("Amplitudes and polarities")

    axs[1, 0].imshow(np.rot90(h5py_obj.get("IslandTimeKernels")[:,:,STRF_num]))
    axs[1, 0].set_title("Island time kernals")

    # axs[1, 1].plot(f.get("IslandTimeKernels_Pol_rel"))  Not sure what this does
    


    axs[1,1].plot(h5py_obj["IslandsX"][:,STRF_num], label = 'x')
    axs[1,1].plot(h5py_obj["IslandsY"][:,STRF_num], label = 'y')
    axs[1,1].legend()
    axs[1,1].set_title("Island coordinates")


    axs[0, 2].scatter(h5py_obj["IslandsX"][:, STRF_num], h5py_obj["IslandsY"][:, STRF_num])
    axs[0, 2].set_title("Island coordinate scatter")

    # plot_SD_pols(f["STRF_Corr0"][:,:,STRF_num], ax = axs[1, 2])
    plot_SD_pols(h5py_obj["STRF_Corr0"][:,:,STRF_num], (h5py_obj["IslandsX"][:, STRF_num]), (h5py_obj["IslandsY"][:, STRF_num]), (h5py_obj["IslandTimeKernels_Polarities"][:, STRF_num]), ax = axs[1, 2])

    plt.show()

def STRF_summary(h5py_obj):
    n_STRFs = len(np.transpose(h5py_obj["STRF_Corr0"]))
    n_suplots_needed = np.ceil((n_STRFs + 2) / 2) * 2
    print("Need", n_suplots_needed, "subplots")
    n_cols = 5
    n_rows = round(n_suplots_needed / n_cols)
    print("Got:", n_cols, "columns and", n_rows, "rows")
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 10))

    fig.suptitle("Overview for file")
    axs[0, 0].imshow(np.average(np.rot90(h5py_obj["wDataCh0"]), axis = 2), origin = "lower", cmap = "Greys_r")
    axs[0, 0].imshow(np.rot90(h5py_obj["ROIs"]), origin = "lower", alpha = 0.25,cmap = "Reds_r")
    axs[0, 0].set_title("ROIs")
    # axs[0, 0].invert_yaxis()

    axs[0, 1].bar(np.arange(0, len(h5py_obj["nIslands_per_ROI"])), (h5py_obj["nIslands_per_ROI"]))
    axs[0, 1].set_title("# islands per ROI")

    col_count = 2
    row_count = 0
    for i in range(len(np.transpose(h5py_obj["STRF_Corr0"]))):
        if col_count == n_cols:
            col_count = 0
            row_count += 1
        if row_count == n_rows:
            # col_count += 1
            row_count = 0
        # plot_SD_pols((f["STRF_Corr0"][:, :, i]), ax = axs[row_count, col_count])
        axs[row_count, col_count].imshow(np.transpose(h5py_obj["STRF_Corr0"][:,:,i]), cmap = 'magma')
        axs[row_count, col_count].set_title("ROI {}".format(i))
        axs[row_count, col_count].invert_yaxis()
        col_count += 1
    plt.show()

def visualise_dispertion_index(bootstrap_dist_reuslts):
    fig, ax = plt.subplots(2, 4, figsize = (20, 10))
    _cached_cov_traces, _kde_cache = bootstrap_dist_reuslts
    ax[0, 0].plot(_kde_cache.T)
    ax[0, 0].set_xlabel("Covariance trace value")
    ax[0, 0].set_ylabel("Density")
    ax[0, 0].set_title("KDE for bootstrapped island number")
    
    get_max = np.max(_cached_cov_traces, axis = 1)
    for n, i in enumerate(get_max):
        ax[0, 1].scatter(n, i)
    ax[0, 1].set_title("Maximum trace per number of island")

    get_means = np.average(_cached_cov_traces, axis = 1)
    for n, i in enumerate(get_means):
        ax[1, 0].scatter(1 * n, i,)
    ax[1, 0].set_title("Mean trace per number of island")
    
    get_std = np.std(_cached_cov_traces, axis = 1)
    for n, i in enumerate(get_std):
        ax[1, 1].scatter(1 * n, i)
    ax[1, 1].set_title("STD per number of island")
   
    config_1 = np.array([(0,0), (MAX_proj.shape[1], 0), (0, MAX_proj.shape[0]), (MAX_proj.shape[1], MAX_proj.shape[0])]) 
    config_2 = np.array([(0,0), (MAX_proj.shape[0], 0), (0, MAX_proj.shape[0]), (MAX_proj.shape[0], MAX_proj.shape[0])]) 
    cords = config_1
    iterations = 15
    param_space = np.linspace(0, 1, iterations)
    decriments_trace_cache = np.zeros(iterations)
    decriments_index_cahce = np.zeros(iterations)
    cords_cache = np.zeros((iterations, 4, 2))
    for n, i in enumerate(param_space):

        cords_tL, cords_bL, cords_tR, cords_bR =((cords[0][0] + MAX_proj.shape[1] * i/2., cords[0][1] + MAX_proj.shape[0] * i/2),
                                                ( cords[1][0] - MAX_proj.shape[1] * i/2, cords[1][1] + MAX_proj.shape[0] * i/2), 
                                                ( cords[2][0] + MAX_proj.shape[1] * i/2, cords[2][1] - MAX_proj.shape[0] * i/2), 
                                                ( cords[3][0] - MAX_proj.shape[1] * i/2, cords[3][1] - MAX_proj.shape[0] * i/2))

        # cords_tL, cords_bL, cords_tR, cords_bR =((cords[0][0] + MAX_proj.shape[0] * i/2, cords[0][1] + MAX_proj.shape[0] * i/2),
        #                                         ( cords[1][0] - MAX_proj.shape[0] * i/2, cords[1][1] + MAX_proj.shape[0] * i/2), 
        #                                         ( cords[2][0] + MAX_proj.shape[0] * i/2, cords[2][1] - MAX_proj.shape[0] * i/2), 
        #                                         ( cords[3][0] - MAX_proj.shape[0] * i/2, cords[3][1] - MAX_proj.shape[0] * i/2))


        decriments_cords = np.array([cords_tL, cords_bL, cords_tR, cords_bR])
        decriments_trace_cache[n] = get_covariance_trace(decriments_cords)
        decriments_index_cahce [n] = trace_index(decriments_cords, bootstrap_dist_reuslts[0], bootstrap_dist_reuslts[1]) # Used to be two last vars were covariance_trace_stored, trace_KDE_stored
        cords_cache[n] = decriments_cords
    # custom_cmap = plt.cm.Reds.colors[0:256:math.ceil(256/iterations)]
    custom_cmap = plt.cm.Greys(np.linspace(.25, .75, iterations))
    for n, (x, y) in enumerate(zip(cords_cache[:, :, 0], cords_cache[:, :, 1])):
        ax[0,2].scatter(x, y, color = custom_cmap[n])#, c = param_space, cmap = 'magma')
    ax[0,2].set_title("Example dispertion at 10% decriments")
    ax[0,2].set_ylabel("Y")
    ax[0,2].set_xlabel("X")
    ax[1,2].scatter(param_space, decriments_trace_cache, color = custom_cmap)
    ax[1,2].set_title("Raw covariance trace scores")
    ax[1,2].set_xlabel('"Closeness"')
    ax[1,2].set_ylabel('Cov trace value')
    ax[0,3].scatter(param_space, decriments_index_cahce, color = custom_cmap)
    x = np.linspace(1, 0, 2)
    ax[0,3].plot(x, c = 'r', lw = 0.25)
    ax[0,3].set_title("Resulting index scores") 
    ax[0,3].set_xlabel('"Closeness"/sparseness')
    ax[0,3].set_ylabel("Index score")
    z = do_for_every(trace_index, covariance_trace_stored, trace_KDE_stored)
    ax[1,3].bar(np.arange(len(z)), z)
    ax[1,3].set_title("Sparseness indeces for current STRFs")
    print(min(decriments_trace_cache))

def threed_plot(elev, azim):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = (10, 20))

    # Make data.
    X = np.arange(len(corrected_arr[:, 0]))
    Y = np.arange(len(corrected_arr[0, :]))
    X, Y = np.meshgrid(X, Y)
    Z = corrected_arr.T

    x_i = (f["IslandsX"][:, STRF_num])
    y_i = (f["IslandsY"][:, STRF_num])
    pols = (f["IslandTimeKernels_Polarities"][:, STRF_num])
    coords_pols = np.stack((x_i, y_i, pols))

    # Remove gray panes and axis grid
    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    # Remove z-axis
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])

    # # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap = 'magma', antialiased=True,alpha = 0.5 )

    # for xx, yy, polarity in zip(x, y, pols):
    #     # print(xx, yy)
    #     if math.isnan(xx) is True or math.isnan(yy) is True:
    #         # This basically handles the nans 
    #         pass
    #     else:
    #         if polarity == 1:
    #             ax.scatter(yy, xx, Z[int(xx), int(yy)] + 10000, s = 200, c = 'r')

    #         if polarity == -1:
    #             ax.scatter(yy, xx, Z[int(xx), int(yy)] + 10000, s = 200, c = 'b')
            
    # Adjust plot view
    ax.view_init(elev=elev, azim=azim)
    ax.dist=10

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.25)
    # cbar.set_ticks([0, 50, 100, 150, 200])
    # cbar.set_ticklabels(['0', '50', '100', '150', '200 nm'])
    fig.canvas.flush_events()
    fig.canvas.draw()
