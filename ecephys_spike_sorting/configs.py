import os
import glob

BASE_PATH = r'D:\NPIX'
class baseConfig(object):
    path = ''

class NPIX1(baseConfig):
    '''
    - Processing of licking not ideal
        - paw movements in earlier days gets onto CAM1
        - grooming
        - camera moves around day to day
    - Ultimately requires DLC to segment tongue

    '''
    name = 'NPIX1'
    path = 'NPIX1'
    camera = 'CAM_1'
    fps = 40
    duration = 15 # in seconds
    # Processing on cam1 is bad for 1103 and 1104 (very)
    # 09, 10, 11, 12, 13, 14 are good. 17 back half and 18 also show anticipatory licking.

    # cam 1 roi locations and lick detection thresholds
    exp_params = {
        # '2023.11.01': [[], [], None], # odor response, too noisy can't analyze
        # '2023.11.02': [[], [], None], # odor response, too noisy, can't analyze
        # '2023.11.03': [[370, 180, 25, 25], [440, 280, 25, 25], 100], # 2 odors, done, mostly water response.
        # '2023.11.04': [[], [440, 285, 25, 25], 100], # learning, done, odor responses but no iso response. noisy
        # '2023.11.06': [[], [425, 280, 25, 25], 100], # not thirsty, phy. SS water responses.
        # '2023.11.07': [[], [435, 300, 25, 25], 100], # not thirsty, phy. odor responses.

        # '2023.11.08': [[], [440, 270, 25, 25], 100], # learning, done, weak
        # '2023.11.09': [[], [425, 270, 25, 25], 150], # good, done, strong
        '2023.11.10': [[], [625, 285, 25, 25], 150], # good, done, strongest
        # '2023.11.11': [[], [610, 285, 25, 25], 150], # good, done, weaker than 11.09
        # '2023.11.12': [[], [610, 285, 25, 25], 150], # good, done, weaker than 11.10
        # '2023.11.13': [[], [605, 285, 25, 25], 150], # good, done, same as 11.11
        # '2023.11.14': [[], [295, 280, 25, 25], 150], # good, done. weird, dominated by inhibition
        # '2023.11.15': [[], [500, 280, 25, 25], 150], # not thirsty, done, dominated by inhibition

        # '2023.11.17': [[], [480, 280, 25, 25], 150], # iso line blocked and not thirsty, phy. bad
        # '2023.11.18': [[], [390, 280, 25, 25], 150], # iso line blocked, phy. bad
    }

    # stimulus condition: [condition, odor on timing, odor off timing, US time]
    stimuli_params = {
        'ISO': ['CSP', 5, 7, 10],
        'PIN': ['CSP', 7, 9, 10],
        'EUY': ['CSM', 5, 7, None],
        'HEP': ['CSM', 5, 7, None],
        'US': ['US', None, None, 10]
    }

    run_specs = [
        ['2023_11_01__all', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.01'],
        ['2023_11_03_all', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.03'],
        ['2023_11_06_all', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.06'],
        ['2023_11_08_all', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.08'],
        ['2023_11_10_all', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.10'],
        ['2023_11_12_all', '1', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.12'],
        ['2023_11_14_all', '1', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.14'],
        ['2023_11_18_all', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.18'],
        ['2023_11_02_s4', '1', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.02'],
        ['2023_11_04_s4', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.04'],
        ['2023_11_07_s4', '1', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.07'],
        ['2023_11_09_s4', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.09'],
        ['2023_11_11_s4', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.11'],
        ['2023_11_13_s4', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.13'],
        ['2023_11_15_s4', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.15'],
        ['2023_11_17_s4', '0', '0,0', '0', ['default'], r'D:\NPIX\NPIX1\2023.11.17'],
    ]

    ni_extract_string = '-xa=0,0,1,3,3,0 -xia=0,0,1,3,3,0'

class NPIX2(baseConfig):
    name = 'NPIX1'
    path = 'NPIX1'
    camera = 'CAM_1'
    fps = 40
    duration = 15 # in seconds
    exp_params = {
    }
