
# README file for Turi et al, 2018 Vasoactive intestinal polypeptide-expressing interneurons in the hippocampus support goal-oriented spatial learning

# for more information, refer to the comments inside the scripts or contact me in: chavlis [DOT] spiros [AT] gmail [DOT] com

# Scripts' author: S. Chavlis, PhD

################### VALIDATION OF NEURONS ##########################################################################
# Compile NEURON mod files
nrnivmodl mechanisms/


# Passive properties

# Go to ValidationTests
cd ValidationTests

# execute for cell types

../x86_64/special -c icurrent=<current_amplitude> IinjectionAt<neuron_type>.hoc

# <current_amplitude>: the injected at soma current in nA
# <neuron_type> valid values

# Axoaxonic
# Basket
# Bistratified
# OLM
# VIPCCK
# VIPCR
# Pyramidal

# e.g., ../x86_64/special -c icurrent=-0.1 IinjectionAtPyramidal.hoc

# In order to reproduce all figures with validation results, run for multiple currents from
# -0.20 to 0.40.


# Synaptic Properties

# Go to ValidationTests
cd ValidationTests

# Execute for Interneurons

../x86_64/special synaptic_validation_<neuronal_type>.hoc

# Valid values for <neuronal_type>

# Axoaxonic
# Basket
# Bistratified
# OLM
# VIPCCK
# VIPCR

# e.g., ../x86_64/special synaptic_validation_Axoaxonic.hoc

# These scripts create the folder SynapticProperties and inside the .txt files are saved.

# Execute for Pyramidal (all but NMDA)

../x86_64/special synaptic_validation_Pyramidal.hoc

# Execute for Pyramidal NMDA synapses

../x86_64/special synaptic_validation_Pyramidal_NMDA.hoc
 

# To create the plots shown in Figure S7 (A-D) after executing NEURON for all cells and NMDA synapses for Pyramidal, run

# python synaptic_validation_plots.py <neuronal_type>

# Valid values for <neuronal_type>

# Axoaxonic
# Basket
# Bistratified
# OLM
# VIPCCK
# VIPCR
# Pyramidal

## Inputs from EC LIII and CA3 SC

python synaptic_validation_plots.py EC_CA3




#################### INPUT CREATION #################################################################################
# First you have to create the Inputs, go in make_inputs_linear_track directory
cd make_inputs_linear_track

In a command line execute

python make_grid_like_inputs_speed_prelearning.py <run_number>
python make_grid_like_inputs_speed_mod.py <run_number>

# <run_number> is a specific run form one edge of the track to the other. To replicate the figures one needs 5 runs

# After the simulation of all runs create the grid-like inputs by executing
python glim_v2_prelearning.py <total_number_of_runs>
python glim_v2_speed_mod.py <total_number_of_runs> <modulation_type> # <modulation_type> is 'pos' or 'neg', e.g., python glim_v2_speed_mod.py 5 pos

# Then enter background_noise directory
cd ../backgroun_noise

# create the background noise by executing 

python poisson_input.py <total_number_of_runs> <poisson_rate>  # e.g., poisson_input.py 5 5


#################### PRELEARNING SIMULATIONS #######################################################################
# Run the prelearning simulation by executing

# return to main directory
cd ../



# Run the simulation
.x86_64/special -nogui -c nruns=<run_number> -c ntrials=<virtual_mouse_id> -c n_neuron=<deletion_type> Network_prelearning.hoc

#e.g., .x86_64/special -nogui -c nruns=1-c ntrials=1 -c n_neuron=No_VIPcells Network_prelearning.hoc

# to replicate the results of the paper you need 5 runs/trial and 10 trials and all possible deletions (see below)

# Valid deletions: 
# Control:      All connections and cells, default
# No_VIPcells:  All VIP+ neurons are removed
# No_VIPCR:     All VIP+/CR+ neurons are removed
# No_VIPCCK:    All VIP+/CCK+ neurons are removed
# No_VIPPVM:    All VIP Positive Velocity Modulated cells are removed
# No_VIPNVM:    All VIP Negative Velocity Modulated cells are removed
# No_VIPCRtoBC: All VIP/CR to Basket Cells connections are removed
# No_VIPCRtoOLM All VIP/CR to OLM Cells connections are removed


# Output of the simulation is saved into Simulation_Results/prelearning

#################### LOCOMOTION SIMULATIONS ##########################################################################
# Run the locomotion

.x86_64/special -nogui -c nruns=<run_number> -c ntrials=<virtual_mouse_id> -c n_neuron=<deletion_type> Network_locomotion.hoc

#e.g., .x86_64/special -nogui -c nruns=1-c ntrials=1 -c n_neuron=No_VIPcells Network_locomotion.hoc

# Output of the simulation is saved into Simulation_Results/locomotion

#################### ANALYSIS OF LOCOMOTION DATA BEFORE PROCEEDING #####################################################

# First, one needs to extract the spiketimes for neurons in order to analyze them
# Go to AnalysisRawData directory
cd AnalysisRawData

# Exctract spike times

python spiketimes_analysis.py <neuron_type> <learning_mode> <deletion_type> <number_of_trial> <number_of_run>

# Valid <neuron_type> values:

# _pvsoma_   : Pyramidal cells
# _aacell_   : Axoaxonic cells
# _bcell_    : Basket cells
# _bscell_   : Bistratified cells
# _olmcell_  : OLM cells
# _vipcck_   : VIP/CCK cells
# _vipcr_    : VIP/CR PVM cells
# _vipcrnvm_ : VIP/CR NVM cells


# Valid <learning_mode> values:

# prelearning
# locomotion
# reward


# After the analysis for all trials, runs and deletions execute:

python all_path_all_spiketimes.py <learning_mode> # e.g., python all_path_all_spiketimes.py locomotion

# This will create the subfolder data_analysis/locomotion/metrics_permutations
# or
# data_analysis/prelearning/metrics_permutations
# where the spiketimes and the path for all cases is stored (for better handling)

# Then execute the command below to calculate whose cell firing rate is above the threshold in the reward zone
python plasticity_indices.py

# This command will create a .txt file in Simulation_Results/<learning_mode>/<deletion_type>/Trial_<trial_number> 
# with the indices of the cells undergo plasticity


#################### REWARD+LOCOMOTION SIMULATIONS ##########################################################################
# Run the locomotion+reward

# Go to main directory 
cd ../

.x86_64/special -nogui -c nruns=<run_number> -c ntrials=<virtual_mouse_id> -c n_neuron=<deletion_type> Network_reward.hoc

#e.g., .x86_64/special -nogui -c nruns=1-c ntrials=1 -c n_neuron=No_VIPcells Network_reward.hoc

# Output of the simulation is saved into Simulation_Results/reward


#################### ANALYSIS OF FINAL DATA ##########################################################################
# Go to AnalysisRawData directory
cd AnalysisRawData

python spiketimes_analysis.py <neuron_type> <learning_mode> <deletion_type> <number_of_trial> <number_of_run> # for all runs, trials, neuronal_types etc.

python all_path_all_spiketimes.py <learning_mode> # for all learning modes, except for locomotion (see above)


python heatmaps.py <number_of_trial> <deletion_type> <learning_mode> # this produces heatmps and all metrics per deletion and per trial

python heatmaps_interneurons.py <number_of_trial> <deletion_type> <learning_mode> <interneuronal_type> # same as the above, but now for interneurons

# valid <interneuronal_type>: 

# aacell   : Axoaxonic cells
# bcell    : Basket cells
# bscell   : Bistratified cells
# olm      : OLM cells
# vipcck   : VIP/CCK cells
# vipcr    : VIP/CR PVM cells
# vipcrnvm : VIP/CR NVM cells



python all_trials_paper_all.py # produces the altogether heatmaps
python all_trials_paper_all_interneurons.py # produces the altogether heatmaps

python all_trials_enrichment.py # produces the barplot with the enrichment for all deletion types and learning modes


# for more information, refer to the comments inside the scripts or contact me in: chavlis [DOT] spiros [AT] gmail [DOT] com







