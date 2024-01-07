import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import cv2
import tqdm
import time
import pandas as pd
import configs
import glob
from utils import figutils
from utils import ioutils

def check_roi_video(path: str, xys: list, step_size: int=40, min_frame: int=0, max_frames:int=600):
    '''
    plots 1 frame per step_size/40 seconds of the video with the ROIs
    xys come in as [x_start, y_start, h, w]
    '''
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames != max_frames:
        print(f'{path} Total: {total_frames}, expected: {max_frames}')
    frame_count = 0
    for _ in tqdm.trange(total_frames):
        frame_recieved, frame = cap.read()
        if not frame_recieved:
            print("Error: couldn't read the frame")
        if frame_count % step_size == 0 and frame_count >= min_frame:
            time.sleep(0.2) # so pycharm will show
            # invert_frame = 255-prev_frame[:, :,0] # for use in Jupyter notebooks
            invert_frame = frame[:, :, 0]
            fig, ax = plt.subplots(1)
            ax.imshow(invert_frame, cmap='gray')
            rect = patches.Rectangle((xys[0], xys[1]), xys[3], xys[2], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()
        frame_count += 1
    cap.release()

def compute_delta(previous_frame, current_frame, min_pixel_delta=30):
    _, delta = cv2.threshold(cv2.absdiff(previous_frame, current_frame), min_pixel_delta, 1, cv2.THRESH_BINARY)
    return np.sum(delta)

def process_video(path: str, xys: list, threshold: float, plot: bool=False, max_frames:int=600):
    # Extract ROI region
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames != max_frames:
        print(f'{path} Total: {total_frames}, expected: {max_frames}')

    cropped_ROI = []
    for frame in range(total_frames):
        frame_recieved, prev_frame = cap.read()
        if not frame_recieved:
            print("Error: couldn't read the frame")
        # xys come in as [x_start, y_start, h, w]
        cropped_ROI.append(prev_frame[:, :, 0][xys[1]:xys[1] + xys[2], xys[0]:xys[0] + xys[3]])
    cap.release()

    # Compute absolute diff between frames
    deltas = [compute_delta(cropped_ROI[x], cropped_ROI[x+1]) for x in range(len(cropped_ROI) - 1)]
    deltas = np.r_[0, deltas]
    deltas_bin = [1 if x >= threshold else 0 for x in deltas]
    if plot:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(deltas)
        axs[1].plot(deltas_bin)
        plt.show()
    return deltas, deltas_bin

def _draw_layer(data, colors, condition=None, alpha=0.5, args={}):
    c = LinearSegmentedColormap.from_list('',
                                          colors,
                                          N=len(colors))
    if condition is None:
        ma = np.ma.masked_array(data, data == 0)
    else:
        ma = np.ma.masked_array(data, condition)
    plt.imshow(ma, cmap=c, interpolation='none', rasterized=True, alpha=alpha, **args)

'''
Main analysis
'''
#Define helpful variables
mouse = configs.NPIX1

folder_path = os.path.join(configs.BASE_PATH, mouse.path)
stimuli = list(mouse.stimuli_params.keys())
days = list(mouse.exp_params.keys())
threshs = [x[2] for x in mouse.exp_params.values()]
directories = [folder_path[:-1] + day for day in days]
fps = mouse.fps
total_frames = int(mouse.duration * fps)
cam = mouse.camera
if cam == 'CAM_0':
    xys = [x[0] for x in mouse.exp_params.values()]
else:
    xys = [x[1] for x in mouse.exp_params.values()]


bool_debug = 0
bool_do = 1
figure_path = os.path.join(folder_path, 'FIGURES')

ixs = np.arange(len(mouse.exp_params.keys()))

# fixed params
colors_lick = [
    [1, 0, 0, 1],
    [0, 0, 0, 1],
]
cmap = LinearSegmentedColormap.from_list('', colors_lick, N=len(colors_lick))

colors_odor = [
    [0.5, 1, 0.5, 1],
    [0, 1, 0.5, 1],
    [1, 0, 0, 1],
    [1, 0.5, 0, 1],
    [0, 1, 1, 1],
]
color_water = [0, 1, 1, 1]

if bool_debug:
    ix = ixs[0]
    current_day = days[ix]
    current_dir = os.path.join(folder_path, current_day, 'behavior')
    xys_test = xys[ix]
    thresh_test = threshs[ix]
    mp4_files = {stim: sorted(glob.glob(os.path.join(current_dir, f'*_{stim}_{cam}.mp4'))) for stim in stimuli}

    us_files = mp4_files['US']
    print(us_files[0])
    check_roi_video(us_files[0], xys_test, step_size=fps*2, min_frame=9*fps, max_frames=total_frames)
    process_video(us_files[0], xys_test, thresh_test, plot=True, max_frames=total_frames)
    print(us_files[-1])
    check_roi_video(us_files[-1], xys_test, step_size=fps*2, min_frame=9*fps, max_frames=total_frames)
    process_video(us_files[-1], xys_test, thresh_test, plot=True, max_frames=total_frames)

if bool_do:
    for ix in ixs:
        current_day = days[ix]
        current_dir = os.path.join(folder_path, current_day, 'behavior')
        xys_test = xys[ix]
        thresh_test = threshs[ix]
        mp4_files = {stim: sorted(glob.glob(os.path.join(current_dir, f'*_{stim}_CAM_1.mp4'))) for stim in stimuli}

        lick_per_stimulus = []
        indices_per_stimulus = []
        for stim in stimuli:
            current_files = mp4_files[stim]
            all_deltas = []
            trial_numbers = []
            print(stim)
            for vid in tqdm.tqdm(current_files):
                # ind_delta, _ = process_video(vid, xys_test, 200) #, plot=True) # before binarization
                _, ind_delta = process_video(vid, xys_test, thresh_test) #200)  # , plot=True)
                if len(ind_delta) != total_frames:
                    if len(ind_delta) < total_frames:
                        # Left padding seems correct to start, but later files warrant right padding, using left for now
                        ind_delta = [0.5]*(total_frames-len(ind_delta)) + ind_delta
                    else:
                        raise ValueError("File too long!")
                all_deltas.append(ind_delta)
                trial_numbers.append(int(vid.split('\\')[-1][:5]))
            if len(all_deltas):
                lick_per_stimulus.append(np.array(all_deltas))
                indices_per_stimulus.append(trial_numbers)
            else:
                lick_per_stimulus.append(None)
                indices_per_stimulus.append(None)

        save_path = os.path.join(folder_path, current_day, 'PROCESSED')
        os.makedirs(save_path, exist_ok=True)
        package = {'stim': stimuli, 'licks': lick_per_stimulus, 'trial_index': indices_per_stimulus}
        ioutils.psave(package, os.path.join(save_path, 'behavior_package'))
        print(f"Saved: {os.path.join(save_path, 'behavior_package')}")

        # Plot
        fig, axes = figutils.pretty_fig(figsize=(3*len(stimuli), 5), rows=1, cols=len(stimuli))
        fig.suptitle(f'{current_dir}')
        for i, stim in enumerate(stimuli):
            if lick_per_stimulus[i] is None:
                continue
            plt.sca(axes[i])
            # odor
            s = mouse.stimuli_params[stim][1]
            e = mouse.stimuli_params[stim][2]
            if s is not None and e is not None:
                mat = np.zeros_like(lick_per_stimulus[i])
                mat[:, int(s*fps):int(e*fps)] = 1
                _draw_layer(mat, [colors_odor[i], colors_odor[i]])
            # water
            w = mouse.stimuli_params[stim][3]
            if w is not None:
                mat = np.zeros_like(lick_per_stimulus[i])
                mat[:, int(w*fps):int((w+0.1)*fps)] = 1
                _draw_layer(mat, [color_water, color_water])
            # licks
            if len(np.unique(lick_per_stimulus[i])) == 2:
                c = [colors_lick[1], colors_lick[1]]
            else:
                c = colors_lick
            _draw_layer(lick_per_stimulus[i], c, alpha=1)

            # styling
            ticks = np.arange(0, total_frames+1, fps * 2)
            # plt.xticks(ticks, ticks//fps)
            plt.xticks(ticks, '')
            plt.yticks(np.arange(lick_per_stimulus[i].shape[0]), indices_per_stimulus[i])
            plt.title(f'{stim}')

            ax = plt.gca()
            ax.set_rasterized(True)
            ax.set_aspect('auto')
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        plt.tight_layout()
        figutils.save_fig(figure_path, current_day, transparent=False, show=True)
