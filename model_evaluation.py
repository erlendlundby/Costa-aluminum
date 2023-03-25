import numpy as np
import torch
import matplotlib.pyplot as plt

#------------------------------
#Estimate states in testset
#------------------------------
def multi_models_testSet_estimate(testSet, model_list, DT, dataset_object):
    # Input
    # testSet[trajectory , timestep , state ]
    # model_list[model]
    # DT - timestep
    # dataset_object - alu_data Dataset object

    # Output
    # X_est[trajectory, model number, timestep, state]

    no_models = len(model_list)
    no_traj = testSet.shape[0]
    no_timeSteps = testSet.shape[1]

    X_est = torch.zeros((no_traj, no_models, no_timeSteps, 13))

    input_vec = torch.zeros(13)
    for i in range(no_models):
        for j in range(no_traj):
            X_est[j, i, 0:3, :] = testSet[j, 0:3, :]

    for k in range(no_traj):

        for j in range(no_models):
            for i in range(3, no_timeSteps):
                # No of time steps

                # No of

                input_vec[:] = (X_est[k, j, i - 1, :] - dataset_object.x_mean) / (
                    dataset_object.x_std
                )
                input_vec = torch.flatten(input_vec)

                xdot = (
                    model_list[j](input_vec) * dataset_object.y_std
                    + dataset_object.y_mean
                )

                X_est[k, j, i, 0:8] = X_est[k, j, i - 1, 0:8] + xdot[0, :].detach() * DT
                X_est[k, j, i, 8:] = testSet[k, i, 8:]

    return X_est

def COSTA_testSet_estimate(
    testSet,
    f_pbm,
    corr_NN_list,
    DT,
    dataset_object,
    feature_normalization="std_norm",
    output_normalization="std_norm",
):
    # Output
    # X_est[trajectory, model number, timestep, state]

    no_models = len(corr_NN_list)
    no_traj = testSet.shape[0]
    no_timeSteps = testSet.shape[1]

    X_est = torch.zeros((no_traj, no_models, no_timeSteps, 13))

    input_vec_norm = torch.zeros(13)
    input_vec_not_norm = torch.zeros(13)

    for i in range(no_models):
        for j in range(no_traj):
            X_est[j, i, 0:3, :] = testSet[j, 0:3, :]

    for k in range(no_traj):

        for j in range(no_models):
            for i in range(3, no_timeSteps):
                # No of time steps

                input_vec_not_norm[:] = X_est[k, j, i - 1, :]
                input_vec_not_norm = torch.flatten(input_vec_not_norm)
                if feature_normalization == "std_norm":
                    input_vec_norm[:] = (
                        X_est[k, j, i - 1, :] - dataset_object.x_mean
                    ) / (dataset_object.x_std)
                    input_vec_norm = torch.flatten(input_vec_norm)

                elif feature_normalization == "min_max":
                    input_vec_norm[:] = (
                        X_est[k, j, i - 1, :] - dataset_object.x_min
                    ) / (dataset_object.x_max - dataset_object.x_min)
                    input_vec_norm = torch.flatten(input_vec_norm)

                if output_normalization == "std_norm":
                    xdot = corr_NN_list[j](
                        input_vec_norm
                    ) * dataset_object.xdot_std + torch.from_numpy(
                        f_pbm(
                            input_vec_not_norm[0:8].detach().numpy(),
                            input_vec_not_norm[8:].detach().numpy(),
                        ).full()[:, 0]
                    )
                elif output_normalization == "min_max":
                    xdot = corr_NN_list[j](input_vec_norm) * (
                        dataset_object.xdot_max - dataset_object.xdot_min
                    ) + torch.from_numpy(
                        f_pbm(
                            input_vec_not_norm[0:8].detach().numpy(),
                            input_vec_not_norm[8:].detach().numpy(),
                        ).full()[:, 0]
                    )

                X_est[k, j, i, 0:8] = X_est[k, j, i - 1, 0:8] + xdot[0, :].detach() * DT
                X_est[k, j, i, 8:] = testSet[k, i, 8:]

    return X_est

#------------------------------
#Calculate AN-RFMSE values
#------------------------------
def RFMSE_w_divergence_detect(
    X_est_testSet, testSet, dataset_object, start_index, end_index
):
    no_models = X_est_testSet.shape[1]
    no_test_traj = testSet.shape[0]

    RFMSE = []
    NO_div_mat = torch.zeros(no_models)

    for i in range(no_models):
        no_div = 0
        for j in range(no_test_traj):
            norm_error = torch.mean(
                torch.abs(
                    X_est_testSet[j, i, start_index:end_index, 0:8]
                    - testSet[j, start_index:end_index, 0:8]
                )
                / dataset_object.x_std[0:8],
                dim=1,
            )

            norm_error = torch.nan_to_num(norm_error, nan=10, posinf=10, neginf=10)

            if torch.max(norm_error) > 3:
                no_div += 1
            else:
                RFMSE.append(torch.mean(norm_error))

        NO_div_mat[i] = no_div
    RFMSE_return = torch.FloatTensor(RFMSE)

    return RFMSE_return, NO_div_mat

def RFMSE_PBM_divergence_detect(
    X_est_pbm, testSet, dataset_object, start_index, end_index
):

    no_test_traj = testSet.shape[0]

    RFMSE = []

    no_div = torch.zeros(1)

    for j in range(no_test_traj):

        norm_error = torch.mean(
            torch.abs(
                X_est_pbm[j, start_index:end_index, 0:8]
                - testSet[j, start_index:end_index, 0:8]
            )
            / dataset_object.x_std[0:8],
            dim=1,
        )

        norm_error = torch.nan_to_num(norm_error, nan=10, posinf=10, neginf=10)

        if torch.max(norm_error) > 3:
            no_div[0] += 1
        else:
            RFMSE.append(torch.mean(norm_error))

        RFMSE_return = torch.FloatTensor(RFMSE)

    return RFMSE_return, no_div

#Lists of AN-RFMSE values
def lists_of_RFMSE_val(testset, hor,no_of_horizons,X_est_costa_dense, X_est_costa_sparse, X_est_DDM_dense, X_est_DDM_sparse,DATASETS, X_est_PBM,dset_no=2):
    RFMSE_multi_horizon = []
    Divergence_multi_horizon = []

    for i in range(no_of_horizons):
        a = hor[i]
        b = hor[i+1]

        RFMSE_PD,div_PD = RFMSE_w_divergence_detect(
            X_est_DDM_dense[dset_no], testset, DATASETS[dset_no], a, b
        )

        

        RFMSE_PS, div_PS = RFMSE_w_divergence_detect(
            X_est_DDM_sparse[dset_no], testset, DATASETS[dset_no], a, b
        )
        
        RFMSE_Co_PD, div_Co_PD = RFMSE_w_divergence_detect(
            X_est_costa_dense[dset_no], testset, DATASETS[dset_no], a, b
        )
        
        RFMSE_Co_PS, div_Co_PS = RFMSE_w_divergence_detect(
            X_est_costa_sparse[dset_no], testset, DATASETS[dset_no], a, b
        )
        

        RFMSE_PBM, div_pbm = RFMSE_PBM_divergence_detect(X_est_PBM, testset, DATASETS[dset_no], a, b)


        RFMSE = [RFMSE_PD.numpy(), RFMSE_PS.numpy(), RFMSE_Co_PD.numpy(), RFMSE_Co_PS.numpy(), RFMSE_PBM.numpy()]
        Divergence = [div_PD, div_PS, div_Co_PD, div_Co_PS, div_pbm]


        RFMSE_multi_horizon.append(RFMSE)
        Divergence_multi_horizon.append(Divergence)

    return RFMSE_multi_horizon, Divergence_multi_horizon
#------------------------------
#Plotting functions
#------------------------------
def testset_estimate_trajectory(test, X_est_costa, X_est_DDM, X_est_PBM,Tliq, Volt_Tliq_pbm, sim_num, dSet_no, start_ind, end_ind, DT):
    #----------------------------------------------------------
    #Costa  
    #----------------------------------------------------------
    mean_COSTA = torch.mean(X_est_costa[dSet_no][sim_num,:,:,:],0)
    std_COSTA = torch.std(X_est_costa[dSet_no][sim_num,:,:,:],0)

    upper_COSTA = mean_COSTA + 3*std_COSTA
    upper_COSTA = upper_COSTA.detach().numpy()

    lower_COSTA = mean_COSTA - 3*std_COSTA
    lower_COSTA = lower_COSTA.detach().numpy()
    #----------------------------------------------------------
    #DDM
    #---------------------------------------------------------
    mean_DDM = torch.mean(X_est_DDM[dSet_no][sim_num,:,:,:],0)
    std_DDM = torch.std(X_est_DDM[dSet_no][sim_num,:,:,:],0)

    upper_DDM = mean_DDM + 3*std_DDM
    upper_DDM = upper_DDM.detach().numpy()

    lower_DDM = mean_DDM - 3*std_DDM
    lower_DDM = lower_DDM.detach().numpy()
    #---------------------------------------------------------  
    #PBM
    #----------------------------------------------------------
    X_PBM = X_est_PBM[sim_num,:,:].detach().numpy()
    Tliq_sim = Tliq[sim_num,:].detach().numpy()
    Tliq_pbm = Volt_Tliq_pbm[sim_num,:,1].detach().numpy()

    #---------------------------------------------------------
    #Plot specifics
    #---------------------------------------------------------
    fs = 30
    fs_ax = 20
    ftl_ls =12
    csfont = {'fontname':'Times New Roman'}
    a = start_ind
    b = end_ind
    x = np.arange(a,b)*DT/3600
    #---------------------------------------------------------
    fig,ax = plt.subplots(5,3,figsize=(20,20))
    fig.tight_layout()

    k_x=0
    k_u = 0
    print(mean_COSTA.shape)
    for j in range(13):
        k_x = k_x%3
        k_u = k_u%3
        
        
        if j<8:
            #Simulated values
            ax[int(j/3),k_x].plot(x,test[sim_num,a:b,j],color='black')
            #Mean values
            ax[int(j/3),k_x].plot(x,mean_COSTA[a:b,j].detach().numpy(),color='darkorange', ls='--')
            ax[int(j/3),k_x].plot(x,mean_DDM[a:b,j].detach().numpy(),color='darkblue', ls='--')
            ax[int(j/3),k_x].plot(x,X_PBM[a:b,j],color='red', ls='--')
            #Uncertainty bounds
            ax[int(j/3),k_x].fill_between(x,lower_COSTA[a:b,j],upper_COSTA[a:b,j],facecolor='darkorange',alpha=0.4)
            ax[int(j/3),k_x].fill_between(x,lower_DDM[a:b,j],upper_DDM[a:b,j],facecolor='darkblue',alpha=0.1)
            ax[int(j/3),k_x].set_xlabel('Time (hours)', fontsize=fs_ax,**csfont)
        elif j>=8:
            ax[int((j+1)/3),k_u].plot(x,test[sim_num,a:b,j],color='black')
    
        if j<5:
            ax[int(j/3),k_x].set_ylabel('Mass (kg)', fontsize=fs_ax)
        elif j<8:
            ax[int(j/3),k_x].set_ylabel('Temp ($^\circ$C)', fontsize=fs_ax)
        #Control input plots
        elif j==8 or j==11:
            ax[int((j+1)/3),k_u].set_ylabel('Mass (kg)', fontsize=fs_ax)
            ax[int((j+1)/3),k_u].set_xlabel('Time (hours)', fontsize=fs_ax,**csfont)
        elif j==10:
            ax[int((j)/3),k_u].set_ylabel('Mass (kg)', fontsize=fs_ax)
            ax[int(j/3),k_u].set_xlabel('Time (hours)', fontsize=fs_ax,**csfont)
        elif j==9:
            ax[int((j)/3),k_u].set_ylabel('Line Current (A)', fontsize=fs_ax)
            ax[int(j/3),k_u].set_xlabel('Time (hours)', fontsize=fs_ax,**csfont)
        elif j==12:
            ax[int((j)/3),k_u].set_ylabel('Distance (m)', fontsize=fs_ax)
            ax[int(j/3),k_u].set_xlabel('Time (hours)', fontsize=fs_ax,**csfont)

        if j<8:
            k_x+=1
        else:
            k_u+=1
    #Liquidus temp plot
    ax[2,2].plot(x,Tliq_sim[a:b],color='black')
    ax[2,2].plot(x,Tliq_pbm[a:b],color='red',ls='--')
    ax[2,2].set_ylabel('Temp ($^\circ$C)', fontsize=fs_ax)
    ax[2,2].set_xlabel('Time (hours)', fontsize=fs_ax,**csfont)

    fig.delaxes(ax[4,2])
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.89, wspace=0.25, hspace=0.31)

    return fig

def RFMSE_violin_multi_horizon_plot(RFMSE_lists, horizons, log=True):
    no_horizons = len(RFMSE_lists)
    ind = np.arange(no_horizons)
    bar_W = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    colors = ["C4", "C0", "C2", "C1", "C3"]
    if log:
        ax.set_yscale("log")

    for i, _ in enumerate(RFMSE_lists[0]):  # For each architecture

        RFMSE___ = [np.array(item[i]) for item in RFMSE_lists]
        pos = ind * 1.5 + bar_W * i - 0.5
        vp = ax.violinplot(
            RFMSE___,
            widths=bar_W - 0.04,
            positions=pos,
            # flierprops={"marker": "o", "markersize": 3},
            # patch_artist=True,
        )

        for vplot in vp["bodies"]:
            vplot.set_facecolor(colors[i])
            vplot.set_alpha(0.4)

        # for bar in 
        vp["cbars"].set_color("black")
        vp["cmins"].set_color("black")
        vp["cmaxes"].set_color("black")
            # bar.set_color("black")

    ax.spines.top.set_linewidth(1)
    ax.set_xticks(1.5 * ind - 0.5 + 2 * bar_W)

    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.set_xticklabels(
        (
            str(horizons[0]) + r"$\Delta{T}$",
            str(horizons[1]) + r"$\Delta{T}$",
            str(horizons[2]) + r"$\Delta{T}$",
        )
    )
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    ax.plot(
        [0, 1],
        [0, 0],
        transform=ax.transAxes,
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    plt.grid(which="minor", alpha=0.6)

    return fig


def divergence_plot_multi_horizons(divergence_list, horizons, labelsize=20):
    no_horizons = len(divergence_list)
    no_arch = len(divergence_list[0])
    ind = np.arange(no_horizons)
    bar_W = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    colors = ["C4", "C0", "C2", "C1", "C3"]

    for i in range(no_arch):
        # For each architecture

        D = [np.sum(np.array(item[i])) for item in divergence_list]

        pos = ind * 1.5 + bar_W * i - 0.5
        ax.bar(pos, D, bar_W, color=colors[i], alpha=0.4)

    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    ax.set_xticks(1.5 * ind - 0.5 + 2 * bar_W)
    ax.set_xticklabels(
        (
            str(horizons[0]) + r"$\Delta{T}$",
            str(horizons[1]) + r"$\Delta{T}$",
            str(horizons[2]) + r"$\Delta{T}$",
        )
    )

    plt.grid(axis="y", alpha=0.6)

    return fig

