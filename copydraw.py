# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:59:38 2020

@author: Daniel
"""
import numpy as np
import pandas as pd
import scipy.io
import time
import logging
import random
import utils
import json

from elements import create_element
from base import AbstractParadigm
from psychopy import core, event, clock
from psychopy.tools.monitorunittools import convertToPix  # , posToPix
from pathlib import Path
from scipy.interpolate import splprep, splev # abandon this smoothing method?
from template_image import template_to_image

logger = logging.getLogger('CopyDraw')


# logger.setLevel(logging.DEBUG)


# boxColour = [160,160,180]
# boxHeight = 200
# boxLineWidth = 6
# templateColour = [80,80,150]
# templateThickness = 3
# traceColour = [255,50,50]
# traceThickness = 1

# startTrialBoxColor = [50, 255, 255]
# textColour = [153, 153, 255]
# timeColour = [180, 180, 160]


# Try to use fewer ternary operators - so you get shorter lines (PEP8)
class ManyShapeStim:
    """
    For creating a shapestim like object,
    made of many individual psychopy shapes
    """

    def __init__(self,
                 vertices,
                 shape,
                 shape_params,
                 size=1):
        vertices = [x * size for x in vertices]
        self.shapes = [shape(pos=pos, **shape_params) for pos in vertices]
        self.verticesPix = np.array([convertToPix(vertex,
                                                  (0, 0),
                                                  shape_params['units'],
                                                  shape_params['win'])
                                     for vertex in vertices])

    def draw(self):
        for shape in self.shapes:
            shape.draw()


class CopyDraw(AbstractParadigm):

    # should there be an n_blocks arg? probably
    # some of these init argsshould be moved to the block init
    def __init__(self,
                 data_dir,
                 screen_size=(1920, 1200),  # 16:10 ratio
                 screen_ix=0,
                 flip=True,
                 lpt_address=None,
                 serial_nr=None,
                 verbose=True,
                 old_template_path='../CopyDraw_mat_repo/CopyDrawTask-master/templates/shapes'):
        super().__init__(screen_ix=screen_ix, lpt_address=lpt_address,
                         serial_nr=serial_nr)

        self.data_dir = Path(data_dir)
        self.screen_size = screen_size
        # self.trial_clock = core.Clock() # use this!
        self.flip = flip
        self.old_template_path = old_template_path

        self.results_dir = self.data_dir / 'results'

        # define attributes to be properly defined later (POLS)
        # maybe also look through and remove unnecessary ones
        self.session_name = None
        self.session_dir = None
        self.info_runs = None
        self.block_idx = None
        self.aspect_ratio = None
        # Could put a lot of these into a single params dict or sth?
        self.n_trials = None
        self.letter_time = None
        self.n_letters = None
        self.image_size = None
        self.stim_size = None
        self.finish_when_raised = None
        self.interp = None
        self.block_dir = None
        self.trials_vec = None
        self.trial_idx = None
        self.block_settings = None
        self.instructions_path = None
        self.old_stimuli = None
        self.the_box = None
        self.stimuli_fname = None
        self.stimuli_file = None
        self.n_templates = None
        self.templates = None
        self.order = None
        self.scaling_matrix = None
        self.current_stimulus = None
        self.cursor_start_point = None  # var not attribute? No.
        self.block_results = None
        self.trial_duration = None
        self.trace_vertices = None  # var not attribute? No.
        self.trial_results = None
        self.frame_elements = None
        self.block_name = None
        self.verbose = verbose  # For debugging, toggles print messages
        # Change the prints to leg messages when you figure out how to log stuff
        if self.verbose: print('initialised')

    def init_session(self, session_name=None):

        if session_name is None:
            self.session_name = time.asctime(time.localtime(time.time())).replace(':', '-')
        else:
            self.session_name = session_name
        self.session_dir = self.results_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.info_runs = self.session_dir / 'info_runs'
        self.info_runs.mkdir(exist_ok=True)
        self.block_idx = 1  # this gets +1'd every time exec block is called

    def init_block(self, block_name=None,
                   n_trials=12,
                   letter_time=2.2,
                   image_size=2,
                   finish_when_raised=True,
                   n_letters=3,
                   stim_size=35,
                   interp='smooth',  # {'s':0.001,'n_interp':300}, dict or None or 'smooth'
                   ):

        super().init_block(self.screen_size)
        self.win.color = (-1, -1, -1)
        self.aspect_ratio = self.win.size[0] / self.win.size[1]
        self.n_trials = n_trials
        self.letter_time = letter_time
        self.image_size = image_size
        self.n_letters = n_letters
        self.finish_when_raised = finish_when_raised
        self.interp = interp
        self.stim_size = stim_size

        if block_name is None:
            time_str = time.asctime(time.localtime(time.time()))
            self.block_name = f'BLOCK_{time_str.replace(":", "-")}'
        else:
            self.block_name = block_name

        # dropped frames
        self.win.recordFrameIntervals = True

        self.load_stimuli(self.data_dir / "templates",
                          short=True if self.n_letters == 2 else False,
                          size=self.stim_size)

        # instructions
        self.load_instructions(self.data_dir /
                               'instructions' /
                               'instructions.png')

        # folders for saving
        self.block_dir = self.session_dir / self.block_name
        self.block_dir.mkdir(parents=True, exist_ok=True)

        # create trials vector
        self.trials_vec = {
            'names': [],  # 231, 123 etc
            'places': [],  # paths, not needed?
            'types': [],  # not sure what this is
            'lengths': [],  # n_letters
            'id': []  # index
        }

        # trial index
        self.trial_idx = 1

        # create template order
        self.create_template_order(shuffle=True)

        # block settings
        self.block_settings = {
            'n_trials': self.n_trials,
            'letter_time': self.letter_time,
            'stim_size': self.stim_size,
            'block_name': self.block_name,
            'n_letters': self.n_letters,
            'finish_when_raised': self.finish_when_raised,
        }

        # log - use this!
        fh = logging.FileHandler(self.block_dir / 'debug.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \
                                      - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f'Block "{self.block_name}" initialised')

    def load_instructions(self, path):
        self.instructions_path = Path(path)
        assert self.instructions_path.exists(), f'Instructions file not found \
            {self.instructions_path}'
        if self.verbose: print('instructions loaded')

    def get_old_stimuli(self, stimuli_idx, temp_or_box='theBox'):
        """ get old stimuli matching the new naming convention """

        old_stim_path = Path(self.data_dir, self.old_template_path)
        assert old_stim_path.exists(), f"old Stimuli data not " \
                                       f"found: {old_stim_path}"

        # does this handle size_10 > size_1?
        # map new nomenclature to old style
        folder = (self.stimuli_fname.replace('5', '.5')
                  if '5.' in self.stimuli_fname
                  else self.stimuli_fname)[:-4]

        # get all mat files within folder
        stim_files = [f for f in old_stim_path.joinpath(folder).rglob('*.mat')]

        self.old_stimuli = scipy.io.loadmat(stim_files[stimuli_idx],
                                            simplify_cells=True)

        return self.old_stimuli[temp_or_box]

    def get_box(self, idx):
        box = self.get_old_stimuli(idx)
        self.the_box = box['boxPos'].T  # box[0,0][0].T.astype(float)
        return self.the_box

    def load_stimuli(self, path, short=True, size=35):
        self.stimuli_fname = f'Size_{"short_" if short else ""}{size}.mat'
        stimuli_path = Path(path, self.stimuli_fname)
        if self.verbose: print(f'loading stimuli: {stimuli_path}')
        assert stimuli_path.exists(), f"Stimuli data not found: {stimuli_path}"

        try:
            self.stimuli_file = scipy.io.loadmat(stimuli_path)
            if self.verbose: print('loaded mat file')
            self.templates = self.stimuli_file['new_shapes'][0]
        except Exception as e:  # rewrite
            if self.verbose: print(e)
        if self.verbose: print('stimuli loaded')

        self.n_templates = self.templates.shape[0]

    # could be in a separate file, seems like a util
    def create_template_order(self, shuffle=True):
        # requires stimuli to be loaded & n_trials to have been defined
        if self.n_templates % self.n_trials != 0:
            # change to a proper warning message?
            if self.verbose: print(f'WARNING: {self.n_trials} trials means that '
                  f'there will be an uneven number of templates')

        self.order = [(i % self.n_templates) for i in range(self.n_trials)]

        if shuffle:
            random.shuffle(self.order)

            # reshuffle to remove repeated trials showing
            while 0 in np.diff(np.array(self.order)):
                random.shuffle(self.order)

    def get_stimuli(self, stimuli_idx, scale=True):
        try:
            self.current_stimulus = self.templates[stimuli_idx].astype(float)
            self.get_box(stimuli_idx)
            self.trials_vec['id'].append(stimuli_idx)
            self.trials_vec['lengths'].append(self.n_letters)
            if scale:

                self.current_stimulus, self.scaling_matrix = \
                    utils.scale_to_norm_units(self.current_stimulus)
                self.the_box, _ = \
                    utils.scale_to_norm_units(self.the_box,
                                              scaling_matrix=self.scaling_matrix)

                # reorder
                new_box_array = np.zeros([4, 2])
                new_box_array[0:2] = self.the_box[0:2]
                new_box_array[2], new_box_array[3] = self.the_box[3],\
                                                     self.the_box[2]
                self.the_box = new_box_array

                if self.verbose: print('scaled')
                if self.flip:
                    self.current_stimulus = np.matmul(self.current_stimulus,
                                                      np.array([[1, 0], [0, -1]]))
                    self.the_box = np.matmul(self.the_box,
                                             np.array([[1, 0], [0, -1]]))

            if self.interp == 'smooth':
                self.current_stimulus = utils.smooth(self.current_stimulus,
                                                     return_df=False)

            elif self.interp is not None:
                tck, u = splprep(self.current_stimulus.T, s=self.interp['s'])

                # using a high value for n_interp along with manyshape == True
                # results in laggy performance

                u_new = np.linspace(np.min(u), np.max(u), self.interp['n_interp'])
                self.current_stimulus = np.array(splev(u_new, tck)).T
                # note this is replacing the original points, not adding

            self.cursor_start_point = self.current_stimulus[0]
            return self.current_stimulus

        except AttributeError:
            if self.verbose: print('load_stimuli must be called first')
            logger.error('Failed with AttributeError, probably due to \
                         load_stimuli not being called')

        except IndexError:
            if self.verbose: print(f'only {len(self.templates)} templates')
            logger.error('Failed with IndexError, probably related to number \
                         of templates')

    def exec_block(self):
        # call exec trial 12? times
        if self.verbose: print(f'executing block {self.block_idx}')

        self.block_results = {}

        for stimuli_idx in range(self.n_trials):
            self.exec_trial(stimuli_idx)
            self.save_trial()  # should this call be outside of here?
            self.trial_idx += 1
            # run some kind of check_trial func here?
            self.block_results[stimuli_idx] = self.trial_results

        self.save_trials_vector()
        self.save_block_settings()
        self.block_idx += 1

    def save_trial(self):
        # rudimentary atm, can ble cleaned up, flattened a little maybe
        df = pd.Series(self.trial_results)
        fname = f'tscore_{self.trial_idx}copyDraw_block{self.block_idx}.pkl'
        df.to_pickle(self.block_dir / fname)

    def save_block_settings(self):
        fname = self.info_runs / f'block_{self.block_idx}_fbsettings.json'
        with open(fname, 'w') as h:
            json.dump(self.block_settings, h)

    def save_trials_vector(self):
        # maybe change from json later, this is nice and readable fornow
        fname = self.block_dir / f'tinfo_copyDraw_block{self.block_idx}.json'
        with open(fname, 'w') as h:
            json.dump(self.trials_vec, h)

    # draw order is based on .draw() call order, consider using an ordered dict?
    def draw_and_flip(self, exclude=[]):
        for element_name, element in self.frame_elements.items():
            if element_name in exclude:
                continue
            else:
                element.draw()
        self.win.flip()  # Bottleneck point, anything that can be done?

    def create_frame(self, stimuli_idx, scale=True):

        self.frame_elements = {}

        self.frame_elements['template'] = create_element(
            'template',
            win=self.win,
            image=template_to_image(
                self.get_stimuli(stimuli_idx, scale=scale),
                f'{self.stimuli_fname[:-4]}_{stimuli_idx}',
                'template_images',
                lineWidth=15
            )
        )
        # uncomment below to draw old template to check how the image
        # matches up with it

        # # seems to top out after like 10 or 20: https://github.com/psychopy/psychopy/issues/818
        # from psychopy.visual import ShapeStim
        # self.frame_elements['old_template'] = visual.ShapeStim(win=self.win,
        #                                             vertices=self.get_stimuli(stimuli_idx,scale=scale),
        #                                             lineWidth=100,
        #                                             closeShape=False,
        #                                             interpolate=True,
        #                                             ori=0,
        #                                             pos=(0,0),
        #                                             size=1.5,
        #                                             units='norm',
        #                                             fillColor='blue',
        #                                             lineColor='blue',
        #                                             #windingRule=True,
        #                                             )
        # self.frame_elements['template'].setOpacity(0.5)

        self.frame_elements['the_box'] = create_element('the_box',
                                                        win=self.win,
                                                        vertices=self.the_box,
                                                        )

        self.frame_elements['trial_number'] = create_element(
            'trial_number',
            win=self.win,
            text=f'Trial {self.trial_idx}/{self.n_trials}',
        )

        self.frame_elements['cursor'] = create_element(
            'cursor',
            win=self.win,
            pos=convertToPix(
                self.cursor_start_point * 1.5,
                (0, 0),
                'norm',
                self.win
            )
        )

        # trace vertices
        # could also take first point from the cursor
        self.trace_vertices = [convertToPix(self.cursor_start_point * 1.5,
                                            (0, 0),
                                            'norm',
                                            self.win)]

        self.frame_elements['trace'] = create_element(
            'trace',
            win=self.win,
            vertices=self.trace_vertices,
        )

        self.frame_elements['instructions'] = create_element(
            'instructions',
            win=self.win,
            image=self.instructions_path
        )

        # start_point
        start_point_size = 0.05
        self.frame_elements['start_point'] = create_element(
            'start_point',
            win=self.win,
            size=(start_point_size, start_point_size * self.aspect_ratio)
        )

        self.frame_elements['time_bar'] = create_element(
            'time_bar',
            win=self.win
        )

    # this is too long
    # should stimuli_idx just be called trial_idx?
    # currently they are initialised differently (0 and 1)
    def exec_trial(self, stimuli_idx, scale=True):
        # display a single copydraw task

        if self.verbose: print('executing trial')

        # define duration as how long it runs for max, trial_time as actual
        # runtime (finish when raised etc)
        self.trial_duration = self.n_letters * self.letter_time

        # initialise the frame
        self.create_frame(self.order[stimuli_idx], scale=scale)

        # getting a monitor gamut error when doing rgb colours, look into it
        # self.trace.colorSpace = 'rgb255'

        # draw first frame
        self.draw_and_flip(exclude=['time_bar', 'trace'])

        # time_bar
        time_bar_x = self.frame_elements['time_bar'].size[0]

        # mouse
        mouse = event.Mouse()

        # main bit
        main_loop = True
        started_drawing = False
        while main_loop:

            # only show instructions if start_point not clicked
            if self.frame_elements['start_point'].fillColor != 'Cyan':
                self.draw_and_flip(exclude=['trace'])
            else:
                self.draw_and_flip(exclude=['trace', 'instructions'])

            # click in start_point
            if mouse.isPressedIn(self.frame_elements['start_point']):
                self.frame_elements['start_point'].fillColor = 'Cyan'
                tic = clock.getTime()
                self.draw_and_flip(exclude=['trace', 'instructions'])

            if self.frame_elements['start_point'].fillColor == 'Cyan':

                # drawing has begun
                if mouse.isPressedIn(self.frame_elements['cursor']):
                    started_drawing = True

                    trial_timer = clock.CountdownTimer(self.trial_duration)

                    # save start time
                    start_t_stamp = clock.getTime()

                    # calc pre trial time
                    ptt = start_t_stamp - tic

                # shrink time bar, draw trace, once drawing has started
                if started_drawing:

                    # for recording times associated with cursor pos
                    cursor_t = [start_t_stamp]

                    c = 0
                    # decreasing time_bar
                    while trial_timer.getTime() > 0:

                        # get remaining time
                        t_remain = trial_timer.getTime()
                        ratio = t_remain / self.trial_duration

                        # adjust time_bar size and position
                        if c % 2 == 0:  # every other frame
                            self._adjust_time_bar(ratio, time_bar_x)

                        # while mouse is pressed in, update cursor position
                        if mouse.getPressed()[0]:
                            self._move_cursor(mouse, trial_timer.getTime(),
                                              cursor_t, c)

                        # ISSUE currently cant do non continuous lines
                        # only draw every xth frame, increases recording rate
                        # erm this doesnt seem to work
                        # opposite effect seems to happen?!
                        if c % 2 == 0:
                            self.draw_and_flip(exclude=['instructions'])
                        c += 1

                        if (not mouse.getPressed()[0] and
                                self.finish_when_raised and started_drawing):
                            if self.verbose: print('mouse raised - ending')
                            main_loop = False
                            break

                    # time_bar elapsed
                    if self.verbose: print('breaking out of main')

                    main_loop = False

        # trial time is how long they were drawing for,
        # ie time elapsed during drawing
        trial_time = self.trial_duration - cursor_t[-1]
        if self.verbose: print(
            f"recorded {len(self.trace_vertices)} points at a rate of"
            f" {len(self.trace_vertices) / trial_time} points per sec")
        # should there be any unit conversions done?
        # maybe put all this stuff below in a func of its own
        # separate the main loop and data stuff
        trace_let = self.frame_elements['trace'].vertices.copy()

        # postprocessing funcs now in separate file
        # funcs for computing dtw etc on recorded data not yet written
        # scoring
        self.trial_results = {'trace_let': trace_let,
                              'trial_time': trial_time,
                              'ix_block': self.block_idx,
                              'ix_trial': self.trial_idx, 'ptt': ptt,
                              'start_t_stamp': start_t_stamp}

        # new/extra metadata
        if scale:
            self.trial_results['scaling_matrix'] = self.scaling_matrix

        self.trial_results['trial_duration'] = self.trial_duration
        self.trial_results['flip'] = self.flip
        self.trial_results['the_box'] = self.frame_elements['the_box'].vertices.copy()

        self.trial_results['theBoxPix'] = self.frame_elements['the_box'].verticesPix

        trace_let_pix = self.frame_elements['trace'].verticesPix.copy()
        if (trace_let != trace_let_pix).any():
            self.trial_results['pos_t_pix'] = trace_let_pix

        # in matlab i think this is theRect
        self.trial_results['winSize'] = self.win.size

        self.trial_results['templatePix'] = self.frame_elements['template'].verticesPix
        # do i need to add theWord? maybe - is the index enough?

    def exit(self):
        self.finish_block()
        # core.quit()

    def _adjust_time_bar(self, ratio, time_bar_x):
        """ Method for adjusting the size of time bar. Wrote
         mainly to aid in profiling. """
        new_size = [time_bar_x * ratio,  # change the x value
                    self.frame_elements['time_bar'].size[1]]
        new_pos = [(-time_bar_x * ratio / 2) + time_bar_x / 2,
                   self.frame_elements['time_bar'].pos[1]]

        self.frame_elements['time_bar'].setSize(new_size)
        self.frame_elements['time_bar'].setPos(new_pos)

    def _move_cursor(self, mouse, t_remain, cursor_t, c):
        """ Method for adjusting the position of the cursor and drawing the
        trace. Wrote mainly to aid profiling."""

        # get new position from mouse
        new_pos = convertToPix(mouse.getPos(), (0, 0), units=mouse.units,
                               win=self.win)

        # record time at which that happened
        # could maybe use t_remain here
        cursor_t.append(t_remain)

        # move cursor to that position and save
        # for drawing trace
        if c % 2 == 0:  # Only draw cursor pos change every other frame
            self.frame_elements['cursor'].pos = new_pos

        # Draw trace, or rather, add new_pos to the trace
        self.trace_vertices.append(new_pos)
        self.frame_elements[
            'trace'].vertices = self.trace_vertices


if __name__ == "__main__":
    try:
        test = CopyDraw('./',
                        )

        test.init_session('TEST_SESSION')

        test.init_block(block_name='TEST_BLOCK',
                        n_trials=2,
                        letter_time=2.7,
                        finish_when_raised=True
                        )
        test.exec_block()
        test.exit()

    # this still isnt printing the error out, why?
    except Exception as e:
        core.quit()
        print(e)
