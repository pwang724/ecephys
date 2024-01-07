import os
import sys
import glob
import subprocess

from helpers import SpikeGLX_utils
from helpers import log_from_json
from helpers import run_one_probe
from create_input_json import createInputJson


# script to run CatGT, KS2, postprocessing and TPrime on data collected using
# SpikeGLX. The construction of the paths assumes data was saved with
# "Folder per probe" selected (probes stored in separate folders) AND
# that CatGT is run with the -out_prb_fld option

# -------------------------------
# -------------------------------
# User input -- Edit this section
# -------------------------------
# -------------------------------

# brain region specific params
# can add a new brain region by adding the key and value for each param
# can add new parameters -- any that are taken by create_input_json --
# by adding a new dictionary with entries for each region and setting the
# according to the new dictionary in the loop to that created json files.

refPerMS_dict = {'default': 0.5}
# threshold values appropriate for KS3.0
ksTh_dict = {'default':'[9,9]'}

# threshold values appropriate for KS2.0, 2.5
# ksTh_dict = {'default':'[10,4]'}
# -----------
# Input data
# -----------
# Name for log file for this pipeline run. Log file will be saved in the
# output destination directory catGT_dest
# If this file exists, new run data is appended to it
logName = 'ece_npx_log.csv'


# run_specs = name, gate, trigger and probes to process
# Each run_spec is a list of 4 strings:
#   undecorated run name (no g/t specifier, the run field in CatGT)
#   gate index, as a string (e.g. '0')
#   triggers to process/concatenate, as a string e.g. '0,400', '0,0 for a single file
#           can replace first limit with 'start', last with 'end'; 'start,end'
#           will concatenate all trials in the probe folder
#   probes to process, as a string, e.g. '0', '0,3', '0:3'
#   brain regions, list of strings, one per probe, to set region specific params
#           these strings must match a key in the param dictionaries above.


run_specs = [
    # ['2023_11_10_all', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.10'],
    # ['s1bot', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX2\npix2_2024.01.06'],
    ['s2bot', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX2\npix2_2024.01.06'],
    # ['s3bot', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX2\npix2_2024.01.06'],
    # ['s4bot', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX2\npix2_2024.01.06'],
    # ['s2', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX3\npix3_2024.01.06'],
    # ['s3', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX3\npix3_2024.01.06'],
    # ['s4', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX3\npix3_2024.01.06'],
]

# ---------------
# Modules List
# ---------------
# List of modules to run per probe; CatGT and TPrime are called once for each run.

run_CatGT = False   # set to False to sort/process previously processed data.
run_ap = False # use catGT to rerun probe channels
modules = [
    # 'kilosort_helper',
    # 'kilosort_postprocessing',
    # 'noise_templates', # do not include
    'mean_waveforms',
    'quality_metrics'
]

# ------------------
# Output destination
# ------------------
# Set to an existing directory; all output will be written here.
# Output will be in the standard SpikeGLX directory structure:
# run_folder/probe_folder/*.bin


# ------------
# CatGT params
# ------------

# CAR mode for CatGT. Must be equal to 'None', 'gbldmx', 'gblcar' or 'loccar'
car_mode = 'gblcar'
# inner and outer radii, in um for local comman average reference, if used
loccar_min = 40
loccar_max = 160

# flag to process lf. The depth estimation module assumes lf has been processed.
# if selected, must also include a range for filtering in the catGT_cmd_string
process_lf = False

# CatGT commands for bandpass filtering, artifact correction, and zero filling
# Note 1: directory naming in this script requires -prb_fld and -out_prb_fld
# Note 2: this command line includes specification of edge extraction
# see CatGT readme for details
# these parameters will be used for all runs
catGT_cmd_string = '-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -lffilter=butter,12,1,500 -gfix=0.4,0.10,0.02 '

ni_present = False
# ni_extract_string = '-xa=0,0,1,1,3,0 -xia=0,0,1,3,3,0 -xd=0,0,-1,1,50 -xid=0,0,-1,2,1.7 -xid=0,0,-1,3,5'
ni_extract_string = '-xa=0,0,3,3,3,0 -xia=0,0,3,3,3,0'

# ----------------------
# KS2 or KS25 parameters
# ----------------------
# parameters that will be constant for all recordings
# Template ekmplate radius and whitening, which are specified in um, will be
# translated into sites using the probe geometry.
ks_remDup = 0
ks_saveRez = 1
ks_copy_fproc = 0
ks_templateRadius_um = 163
ks_whiteningRadius_um = 163
ks_minfr_goodchannels = 0.1
ks_CAR = 0          # CAR already done in catGT
ks_nblocks = 1      # for KS2.5 and KS3; 1 for rigid registration in drift correction,
# higher numbers to allow different drift for different 'blocks' of the probe

# If running KS20_for_preprocessed_data:
# (https://github.com/jenniferColonell/KS20_for_preprocessed_data)
# can skip filtering with the doFilter parameter.
# Useful for speed when data has been filtered with CatGT.
# This parameter is not implemented in standard versions of kilosort.
ks_doFilter = 0

ks_output_tag = 'ks2'


# ----------------------
# C_Waves snr radius, um
# ----------------------
c_Waves_snr_um = 160

# ----------------------
# psth_events parameters
# ----------------------
# extract param string for psth events -- copy the CatGT params used to extract
# events that should be exported with the phy output for PSTH plots
# If not using, remove psth_events from the list of modules
event_ex_param_str = 'xd=0,0,-1,1,50'

# -----------------
# TPrime parameters
# -----------------
runTPrime = False   # set to False if not using TPrime
sync_period = 1.0   # true for SYNC wave generated by imec basestation
toStream_sync_params = 'ni' # should be ni, imec<probe index>. or obx<obx index>

#toStream_sync_params = '0,0'  # 'stream type, stream index', (js,ip) for the two stream, as a string
#niStream_sync_params = 'xia=0,0,1,3,3,0'   # copy from ni_extract_string (no spaces), set to None if no Aux data

# -----------------------
# -----------------------
# End of user input
# -----------------------
# -----------------------

# delete the existing CatGT.log
try:
    os.remove('CatGT.log')
except OSError:
    pass

# delete existing Tprime.log
try:
    os.remove('Tprime.log')
except OSError:
    pass

# delete existing C_waves.log
try:
    os.remove('C_Waves.log')
except OSError:
    pass


for spec in run_specs:
    npx_directory = spec[-1]
    catGT_dest = npx_directory
    # check for existence of log file, create if not there
    logFullPath = os.path.join(catGT_dest, logName)
    if not os.path.isfile(logFullPath):
        # create the log file, write header
        log_from_json.writeHeader(logFullPath)

    session_id = spec[0]

    # Make list of probes from the probe string
    prb_list = SpikeGLX_utils.ParseProbeStr(spec[3])

    # build path to the first probe folder; look into that folder
    # to determine the range of trials if the user specified t limits as
    # start and end
    run_folder_name = spec[0] + '_g' + spec[1]
    json_directory = os.path.join(npx_directory, run_folder_name)

    prb0_fld_name = run_folder_name + '_imec' + prb_list[0]
    prb0_fld = os.path.join(npx_directory, run_folder_name, prb0_fld_name)
    first_trig, last_trig = SpikeGLX_utils.ParseTrigStr(spec[2], prb_list[0], spec[1], prb0_fld)
    trigger_str = repr(first_trig) + ',' + repr(last_trig)

    # loop over all probes to build json files of input parameters
    # initalize lists for input and output json files
    catGT_input_json = []
    catGT_output_json = []
    module_input_json = []
    module_output_json = []
    session_id = []
    data_directory = []

    # first loop over probes creates json files containing parameters for
    # both preprocessing (CatGt) and sorting + postprocessing

    for i, prb in enumerate(prb_list):

        #create CatGT command for this probe
        print('Creating json file for CatGT on probe: ' + prb)
        # Run CatGT
        catGT_input_json.append(os.path.join(json_directory, spec[0] + prb + '_CatGT' + '-input.json'))
        catGT_output_json.append(os.path.join(json_directory, spec[0] + prb + '_CatGT' + '-output.json'))

        # build extract string for SYNC channel for this probe
        # sync_extract = '-SY=' + prb +',-1,6,500'

        # if this is the first probe proceessed, process the ni stream with it
        if run_ap:
            catGT_stream_string = '-ap '
        else:
            catGT_stream_string = ''

        if i == 0 and ni_present:
            # catGT_stream_string = '-ap -ni'
            catGT_stream_string = catGT_stream_string + '-ni'
            extract_string = ni_extract_string
        else:
            # catGT_stream_string = '-ap'
            extract_string = ''

        if process_lf:
            catGT_stream_string = catGT_stream_string + ' -lf'

        # build name of first trial to be concatenated/processed;
        # allows reaidng of the metadata
        run_str = spec[0] + '_g' + spec[1]
        run_folder = run_str
        prb_folder = run_str + '_imec' + prb
        input_data_directory = os.path.join(npx_directory, run_folder, prb_folder)
        fileName = run_str + '_t' + repr(first_trig) + '.imec' + prb + '.ap.bin'
        continuous_file = os.path.join(input_data_directory, fileName)
        metaName = run_str + '_t' + repr(first_trig) + '.imec' + prb + '.ap.meta'
        input_meta_fullpath = os.path.join(input_data_directory, metaName)

        print(input_meta_fullpath)

        info = createInputJson(catGT_input_json[i], npx_directory=npx_directory,
                               continuous_file = continuous_file,
                               kilosort_output_directory=catGT_dest,
                               spikeGLX_data = True,
                               input_meta_path = input_meta_fullpath,
                               catGT_run_name = spec[0],
                               gate_string = spec[1],
                               trigger_string = trigger_str,
                               probe_string = prb,
                               catGT_stream_string = catGT_stream_string,
                               catGT_car_mode = car_mode,
                               catGT_loccar_min_um = loccar_min,
                               catGT_loccar_max_um = loccar_max,
                               catGT_cmd_string = catGT_cmd_string + ' ' + extract_string,
                               extracted_data_directory = catGT_dest
                               )


        #create json files for the other modules
        session_id.append(spec[0] + '_imec' + prb)
        module_input_json.append(os.path.join(json_directory, session_id[i] + '-input.json'))


        # location of the binary created by CatGT, using -out_prb_fld
        run_str = spec[0] + '_g' + spec[1]
        run_folder = 'catgt_' + run_str
        prb_folder = run_str + '_imec' + prb
        data_directory.append(os.path.join(catGT_dest, run_folder, prb_folder))
        fileName = run_str + '_tcat.imec' + prb + '.ap.bin'
        continuous_file = os.path.join(data_directory[i], fileName)

        outputName = 'imec' + prb + '_' + ks_output_tag

        # kilosort_postprocessing and noise_templates moduules alter the files
        # that are input to phy. If using these modules, keep a copy of the
        # original phy output
        if ('kilosort_postprocessing' in modules) or('noise_templates' in modules):
            ks_make_copy = True
        else:
            ks_make_copy = False

        kilosort_output_dir = os.path.join(data_directory[i], outputName)

        print(data_directory[i])
        print(continuous_file)

        # clean-up of kilosort labels
        names = [
            'cluster_ContamPct.tsv',
            # 'cluster_group.tsv',
            'cluster_KSLabel.tsv']
        for name in names:
            file = os.path.join(kilosort_output_dir, name)
            if os.path.exists(file):
                os.remove(file)
                print(f'Removed: {file}')
            else:
                print(f'Does not exist: {file}')

        # cleanup of ecephys labels. Do this before running anything
        ecephys_files = [
            'clus_Table*.npy',
            'mean_waveforms*.npy',
            'cluster_snr*.npy',
            'waveform_metrics*.csv',
            'metrics*.csv',
            'cluster_snr*.tsv'
        ]
        for name in ecephys_files:
            files = glob.glob(os.path.join(kilosort_output_dir, name))
            for file in files:
                os.remove(file)
                print(f'Removed: {file}')

        # get region specific parameters
        ks_Th = ksTh_dict.get(spec[4][i])
        refPerMS = refPerMS_dict.get(spec[4][i])
        print( 'ks_Th: ' + repr(ks_Th) + ' ,refPerMS: ' + repr(refPerMS))

        info = createInputJson(module_input_json[i], npx_directory=npx_directory,
                               continuous_file = continuous_file,
                               spikeGLX_data = True,
                               input_meta_path = input_meta_fullpath,
                               kilosort_output_directory=kilosort_output_dir,
                               ks_make_copy = ks_make_copy,
                               noise_template_use_rf = False,
                               catGT_run_name = session_id[i],
                               gate_string = spec[1],
                               probe_string = spec[3],
                               ks_remDup = ks_remDup,
                               ks_finalSplits = 1,
                               ks_labelGood = 1,
                               ks_saveRez = ks_saveRez,
                               ks_copy_fproc = ks_copy_fproc,
                               ks_minfr_goodchannels = ks_minfr_goodchannels,
                               ks_whiteningRadius_um = ks_whiteningRadius_um,
                               ks_doFilter = ks_doFilter,
                               ks_Th = ks_Th,
                               ks_CSBseed = 1,
                               ks_LTseed = 1,
                               ks_templateRadius_um = ks_templateRadius_um,
                               ks_nblocks = ks_nblocks,
                               ks_CAR = ks_CAR,
                               extracted_data_directory = data_directory[i],
                               event_ex_param_str = event_ex_param_str,
                               c_Waves_snr_um = c_Waves_snr_um,
                               qm_isi_thresh = refPerMS/1000
                               )

        # copy json file to data directory as record of the input parameters

    # loop over probes for processing.
    for i, prb in enumerate(prb_list):

        run_one_probe.runOne( session_id[i],
                              json_directory,
                              data_directory[i],
                              run_CatGT,
                              catGT_input_json[i],
                              catGT_output_json[i],
                              modules,
                              module_input_json[i],
                              logFullPath )



    if runTPrime:

        # after loop over probes, run TPrime to create files of
        # event times -- edges detected in auxialliary files and spike times
        # from each probe -- all aligned to a reference stream.

        # Uncomment line belwo to create a set of all ni time points, which can be
        # corrected by TPrime. This output is used to obtain analog values
        # from the NI stream at spike times.
        # Will cause an error if no ni stream exists.
        # SpikeGLX_utils.CreateNITimeEvents(spec[0], spec[1], catGT_dest)

        # create json files for calling TPrime
        session_id = spec[0] + '_TPrime'
        input_json = os.path.join(json_directory, session_id + '-input.json')
        output_json = os.path.join(json_directory, session_id + '-output.json')

        info = createInputJson(input_json, npx_directory=npx_directory,
                               continuous_file = continuous_file,
                               spikeGLX_data = True,
                               input_meta_path = input_meta_fullpath,
                               catGT_run_name = spec[0],
                               kilosort_output_directory=kilosort_output_dir,
                               extracted_data_directory = catGT_dest,
                               tPrime_ni_ex_list = ni_extract_string,
                               event_ex_param_str = event_ex_param_str,
                               sync_period = 1.0,
                               toStream_sync_params = toStream_sync_params,
                               tPrime_3A = False,
                               toStream_path_3A = ' ',
                               fromStream_list_3A = list(),
                               ks_output_tag = ks_output_tag
                               )

        command = sys.executable + " -W ignore -m ecephys_spike_sorting.modules." + 'tPrime_helper' + " --input_json " + input_json \
                  + " --output_json " + output_json
        subprocess.check_call(command.split(' '))



