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
        # '2023.11.03': [[370, 180, 25, 25], [440, 280, 25, 25], 100], # 2 odors (ISO + PIN)
        # '2023.11.04': [[], [440, 285, 25, 25], 100], # thirsty, learning
        # '2023.11.06': [[], [425, 280, 25, 25], 100], # not thirsty
        # '2023.11.07': [[], [435, 300, 25, 25], 100], # not thirsty
        # '2023.11.08': [[], [440, 270, 25, 25], 100], # thirsty, learning
        '2023.11.09': [[], [425, 270, 25, 25], 150], # good
        # '2023.11.10': [[], [625, 285, 25, 25], 150], # good
        # '2023.11.11': [[], [610, 285, 25, 25], 150], # good
        # '2023.11.12': [[], [610, 285, 25, 25], 150], # good
        # '2023.11.13': [[], [605, 285, 25, 25], 150], # good
        # '2023.11.14': [[], [295, 280, 25, 25], 150], # good
        # '2023.11.15': [[], [500, 280, 25, 25], 150], # not thirsty
        # '2023.11.17': [[], [480, 280, 25, 25], 150], # iso line blocked, not super thirsty
        # '2023.11.18': [[], [390, 280, 25, 25], 150], # iso line blocked
    }

    # stimulus condition: [condition, odor on timing, odor off timing, US time]
    stimuli_params = {
        'ISO': ['CSP', 5, 7, 10],
        'PIN': ['CSP', 7, 9, 10],
        'EUY': ['CSM', 5, 7, None],
        'HEP': ['CSM', 5, 7, None],
        'US': ['US', None, None, 10]
    }
