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
from scipy.interpolate import splprep, splev  # abandon this smoothing method?
from utils import select_display, create_template_order
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
    def __init__(self,
                 data_dir,
                 screen_size=(920, 1200),  # 16:10 ratio
                 screen_ix=None,
                 flip=True,
                 lpt_address=None,
                 serial_nr=None,
                 verbose=True,

                 # MD: paths like this are configs which we should collect at a single place / file
                 # --> Plus, we really should get rid of dependencies to old data!!
                 old_template_path=
                 '../CopyDraw_mat_repo/CopyDrawTask-master/templates/shapes'):
        super().__init__(screen_ix=screen_ix, lpt_address=lpt_address,
                         serial_nr=serial_nr)
        # should these be one dict or kept separate?
        self.paths = {}
        self.names = {}
        self.stimuli = {'flip': flip}
        self.block_settings = None  # will be dict

        self.paths['data'] = Path(data_dir)
        self.screen_size = screen_size
        self.paths['old_template'] = old_template_path
        self.paths['results'] = self.paths['data'] / 'results'
        self.screen_ix = screen_ix or select_display()['screen']
        # define attributes to be properly defined later (POLS)
        # maybe also look through and remove unnecessary ones
        self.block_idx = None
        self.aspect_ratio = None
        # Could put a lot of these into a single params dict or sth?
        self.trials_vec = None
        self.trial_idx = None
        self.old_stimuli = None
        self.the_box = None
        self.templates = None
        self.current_stimulus = None
        self.cursor_start_point = None  # var not attribute? No.
        self.block_results = None
        self.trial_duration = None
        self.trace_vertices = None  # var not attribute? No.
        self.trial_results = None
        self.frame_elements = None
        self._frame_idx = None

        self._lifted = True             # track if mouse is currently lifted
        self.traces = []                # list to record multiple trace object names --> drawing interupted lines
        self._start_frame_idx = 0       # frame index of the last started trace

        self.verbose = verbose  # For debugging, toggles print messages
        # Change the prints to leg messages when you figure out how to log stuff
        if self.verbose:
            print('initialised')

    def init_session(self, session_name=None):

        if session_name is None:
            self.names['session'] =\
                time.asctime(time.localtime(time.time())).replace(':', '-')
        else:
            self.names['session'] = session_name
        self.paths['session'] = self.paths['results'] / self.names['session']
        self.paths['session'].mkdir(parents=True, exist_ok=True)

        self.paths['info_runs'] = self.paths['session'] / 'info_runs'
        self.paths['info_runs'].mkdir(exist_ok=True)
        self.block_idx = 1  # this gets +1'd every time exec block is called

    def init_block(self, block_name=None,
                   n_trials=12,
                   letter_time=2.2,
                   image_size=2,
                   finish_when_raised=True,
                   n_letters=3,
                   stim_size=35,
                   shuffle=True,
                   interp='smooth',  # {'s':0.001,'n_interp':300},None,'smooth'
                   ):

        super().init_block(self.screen_size)
        self.win.color = (-1, -1, -1)
        self.block_settings = {}
        self.aspect_ratio = self.win.size[0] / self.win.size[1]
        self.block_settings['n_trials'] = n_trials
        self.block_settings['letter_time'] = letter_time
        self.block_settings['n_letters'] = n_letters
        self.block_settings['finish_when_raised'] = finish_when_raised
        self.block_settings['interp'] = interp
        self.block_settings['stim_size'] = stim_size
        self.block_settings['shuffle'] = True

        time_str = time.asctime(time.localtime(time.time()))
        self.block_settings['block_name'] = block_name or\
                                            f'BLOCK_{time_str.replace(":", "-")}'

        # dropped frames
        self.win.recordFrameIntervals = True

        self.load_stimuli(self.paths['data'] / "templates",
                          short=True if self.block_settings['n_letters'] == 2 else False,
                          size=self.block_settings['stim_size'])

        # instructions
        self.load_instructions(self.paths['data'] /
                               'instructions' /
                               'instructions.png')

        # folders for saving
        self.paths['block'] = self.paths['session'] / self.block_settings['block_name']
        self.paths['block'].mkdir(parents=True, exist_ok=True)

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
        self.stimuli['order'] = create_template_order(self.stimuli,
                                                      self.block_settings)

        # external logger, how to integrate?
        # # log - use this!
        # fh = logging.FileHandler(self.paths['block'] / 'debug.log')
        # fh.setLevel(logging.DEBUG)
        # logger.addHandler(fh)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \
        #                               - %(message)s')
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)
        # logger.info(f'Block "{self.block_settings["block_name"]}" initialised')

    def exec_block(self, cfg, stim='off'):
        """ Will call init_block(**cfg) before calling exec trial n_trials
        times, also calling save_trial for each. Trials vector and block
         settings saved at the end. """

        self.init_block(**cfg)

        # call exec trial 12? times
        if self.verbose:
            print(f'executing block {self.block_idx}')

        self.block_results = {}

        for stimuli_idx in range(self.block_settings['n_trials']):
            self.exec_trial(stimuli_idx)
            self.save_trial()  # should this call be outside of here?
            self.trial_idx += 1
            # run some kind of check_trial func here?
            self.block_results[stimuli_idx] = self.trial_results

        self.save_trials_vector()
        self.save_block_settings()
        self.block_settings = None  # 'reset' block settings
        self.block_idx += 1

    def load_instructions(self, path):  # This seems kind of useless, remove?
        self.paths['instructions'] = Path(path)
        assert self.paths['instructions'].exists(), f'Instructions file not found \
            {self.paths["instructions"]}'
        if self.verbose:
            print('instructions loaded')

    def get_old_stimuli(self, stimuli_idx, temp_or_box='theBox'):
        """ get old stimuli matching the new naming convention """

        # MD: Actually, we should not use this old data, especially since
        # this directory is not self-sufficient with this

        old_stim_path = Path(self.paths['data'], self.paths['old_template'])
        assert old_stim_path.exists(), f"old Stimuli data not " \
                                       f"found: {old_stim_path}"

        # does this handle size_10 > size_1?
        # map new nomenclature to old style
        folder = (self.stimuli['fname'].replace('5', '.5')
                  if '5.' in self.stimuli['fname']
                  else self.stimuli['fname'])[:-4]

        # get all mat files within folder
        stim_files = [f for f in old_stim_path.joinpath(folder).rglob('*.mat')]

        self.old_stimuli = scipy.io.loadmat(stim_files[stimuli_idx],
                                            simplify_cells=True)

        return self.old_stimuli[temp_or_box]

    def get_box(self, idx):

        # MD: This is an inefficient pattern, as for each call to get box,
        # we will perform a glob / parse directories. Better get all files dirs
        # mapped to indices during initialization and later load only the path
        # --> effectively only one "glob" vs n
        # box = self.get_old_stimuli(idx)
        # self.the_box = box['boxPos'].T  # box[0,0][0].T.astype(float)

        # MD -> as this seems rather constant for a given size and n_letters ==> just hardcode for now
        tb = np.array([[285, 392],
                       [1635, 392],
                       [285, 808],
                       [1635, 808],
                       [288, 392],
                       [288, 808],
                       [1632, 392],
                       [1632, 808],
                       ])

        self.the_box = tb

        return self.the_box

    def load_stimuli(self, path, short=True, size=35):
        self.stimuli['fname'] = f'Size_{"short_" if short else ""}{size}.mat'
        self.paths['stimuli'] = Path(path, self.stimuli['fname'])
        if self.verbose:
            print(f'loading stimuli: {self.paths["stimuli"]}')
        assert self.paths['stimuli'].exists(), f"Stimuli data not found: {self.paths['stimuli']}"

        try:
            self.stimuli['file'] = scipy.io.loadmat(self.paths['stimuli'])
            if self.verbose:
                print('loaded mat file')
            self.templates = self.stimuli['file']['new_shapes'][0]
        except Exception as err:  # rewrite
            if self.verbose:
                print(err)
        if self.verbose:
            print('stimuli loaded')

        self.stimuli['n_templates'] = self.templates.shape[0]

    # # could be in a separate file, seems like a util
    # def create_template_order(self, shuffle=True):
    #     # requires stimuli to be loaded & n_trials to have been defined
    #     if self.stimuli['n_templates'] % self.block_settings['n_trials'] != 0:
    #         # change to a proper warning message?
    #         if self.verbose:
    #             print(f'WARNING: {self.block_settings["n_trials"]} trials means that '
    #                   f'there will be an uneven number of templates')
    #
    #     self.stimuli['order'] = [(i % self.stimuli['n_templates']) for i in range(self.block_settings['n_trials'])]
    #
    #     if shuffle:
    #         random.shuffle(self.stimuli['order'])
    #
    #         # reshuffle to remove repeated trials showing
    #         while 0 in np.diff(np.array(self.stimuli['order'])):
    #             random.shuffle(self.stimuli['order'])

    def get_stimuli(self, stimuli_idx, scale=True):

        try:
            self.current_stimulus = self.templates[stimuli_idx].astype(float)
            self.get_box(stimuli_idx)
            self.trials_vec['id'].append(stimuli_idx)
            self.trials_vec['lengths'].append(self.block_settings['n_letters'])
            if scale:

                self.current_stimulus, self.stimuli['scaling_matrix'] = \
                    utils.scale_to_norm_units(self.current_stimulus)
                self.the_box, _ = \
                    utils.scale_to_norm_units(self.the_box,
                                              scaling_matrix=self.stimuli['scaling_matrix'])

                # reorder
                new_box_array = np.zeros([4, 2])
                new_box_array[0:2] = self.the_box[0:2]
                new_box_array[2] = self.the_box[3]
                new_box_array[3] = self.the_box[2]

                self.the_box = new_box_array

                if self.verbose:
                    print('scaled')
                if self.stimuli['flip']:
                    self.current_stimulus = np.matmul(self.current_stimulus,
                                                      np.array([[1, 0], [0, -1]]))
                    self.the_box = np.matmul(self.the_box,
                                             np.array([[1, 0], [0, -1]]))

            if self.block_settings['interp'] == 'smooth':
                self.current_stimulus = utils.smooth(self.current_stimulus,
                                                     return_df=False)

            elif self.block_settings['interp'] is not None:
                tck, u = splprep(self.current_stimulus.T, s=self.block_settings['interp']['s'])

                # using a high value for n_interp along with manyshape == True
                # results in laggy performance

                u_new = np.linspace(np.min(u), np.max(u), self.block_settings['interp']['n_interp'])
                self.current_stimulus = np.array(splev(u_new, tck)).T
                # note this is replacing the original points, not adding

            self.cursor_start_point = self.current_stimulus[0]
            return self.current_stimulus

        except AttributeError:
            if self.verbose:
                print('load_stimuli must be called first')
            logger.error('Failed with AttributeError, probably due to \
                         load_stimuli not being called')

        except IndexError:
            if self.verbose:
                print(f'only {len(self.templates)} templates')
            logger.error('Failed with IndexError, probably related to number \
                         of templates')

    def save_trial(self):
        # rudimentary atm, can ble cleaned up, flattened a little maybe
        df = pd.Series(self.trial_results)
        fname = f'tscore_{self.trial_idx}copyDraw_block{self.block_idx}.pkl'
        df.to_pickle(self.paths['block'] / fname)

    def save_block_settings(self):
        fname = self.paths['info_runs'] / f'block_{self.block_idx}_fbsettings.json'
        with open(fname, 'w') as h:
            json.dump(self.block_settings, h)

    def save_trials_vector(self):
        # maybe change from json later, this is nice and readable fornow
        fname = self.paths['block'] / f'tinfo_copyDraw_block{self.block_idx}.json'
        with open(fname, 'w') as h:
            json.dump(self.trials_vec, h)

    # draw order is based on .draw() call order, consider using an ordered dict?
    # MD: Ordered dict would be a good idea
    def draw_and_flip(self, exclude=[]):
        """ Draws every element in the frame elements dict, excluding those
         passed in via exclude. """
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
            image=utils.template_to_image(
                self.get_stimuli(stimuli_idx, scale=scale),
                f'{self.stimuli["fname"][:-4]}_{stimuli_idx}',
                self.paths['data']/'template_images',
                lineWidth=15
            )
        )
        # uncomment below to draw old template to check how the image
        # matches up with it

        # # seems to top out after like 10 or 20:
        # https://github.com/psychopy/psychopy/issues/818
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
            text=f'Trial {self.trial_idx}/{self.block_settings["n_trials"]}',
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

        max_trace_len = 10000  # Should be more points than would ever be drawn
        self.trace_vertices = np.zeros([max_trace_len, 2])
        self.trace_vertices[0] = convertToPix(self.cursor_start_point * 1.5,
                                              (0, 0),
                                              'norm',
                                              self.win)

        self.frame_elements['trace1'] = create_element(
           'trace',                           # we will dynamically create more -> draw interupted lines
            win=self.win,
            vertices=self.trace_vertices[0:0],          # initialize empty
        )
        self.traces.append('trace1')

        self.frame_elements['instructions'] = create_element(
            'instructions',
            win=self.win,
            image=self.paths['instructions']
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

    # should stimuli_idx just be called trial_idx?
    # currently they are initialised differently (0 and 1)
    def exec_trial(self, stimuli_idx, scale=True):
        """ Top level method that executes a single trial """
        if self.verbose:
            print('executing trial')

        # define duration as how long it runs for max, trial_time as actual
        # runtime (finish when raised etc)
        self.trial_duration = self.block_settings['n_letters'] * self.block_settings['letter_time']

        # initialise the frame
        self.create_frame(self.stimuli['order'][stimuli_idx], scale=scale)

        # getting a monitor gamut error when doing rgb colours, look into it
        # self.trace.colorSpace = 'rgb255'

        # draw first frame
        self.draw_and_flip(exclude=['time_bar', 'trace1'])

        # time_bar
        time_bar_x = self.frame_elements['time_bar'].size[0]

        # main bit
        self._frame_idx = 0  # refers only to frames during drawing trace
        ptt, start_t_stamp, cursor_t = self._run_trial_main_loop(clock,
                                                                 time_bar_x)

        cursor_t = cursor_t[:self._frame_idx+1]  # truncate
        if self.verbose:
            print(f'drew {self._frame_idx} frames')
            print(f'cursor_t is: {cursor_t.shape} and'
                  f' has mean diff: {np.mean(np.diff(cursor_t))} and '
                  f'which has std: {np.std(np.diff(cursor_t))}')

        # trial time is how long they were drawing for,
        # ie time elapsed during drawing
        trial_time = self.trial_duration - cursor_t[-1]
        if self.verbose:
            print(f"recorded {self._frame_idx} points at a rate of"
                  f" {self._frame_idx / trial_time} points per sec")
        # should there be any unit conversions done?

        # # separate the main loop and data stuff
        # trace_let = self.frame_elements['trace'].vertices.copy()
        # trace_let_pix = self.frame_elements['trace'].verticesPix.copy()
        #
        # For interupted traces, we now have to concatenate
        # General TODO: --> how to deal with the jumps in terms of dtw...
        trace_let = np.concatenate([self.frame_elements[tr_n].vertices.copy()
                                    for tr_n in self.traces])
        trace_let_pix = np.concatenate([self.frame_elements[tr_n].verticesPix.copy()
                                        for tr_n in self.traces])

        self._create_trial_res(trace_let, trial_time, ptt, start_t_stamp,
                               trace_let_pix, scale, cursor_t)

    def exit(self):
        self.finish_block()  # base class method
        # core.quit()

    def _run_trial_main_loop(self, clock, time_bar_x):
        """ To run the main drawing loop """
        started_drawing = False
        cursor_t = np.zeros([10000])  # for recording times with cursor pos
        mouse = event.Mouse(win=self.win)
        #MD: Probably a better pattern -> two separate while loops,
        #   one for activating the cyan square and moving to the beginning
        #   then another one drawing and incrementing the bar
        #
        # As it is currently done, we have to evaluate all the ifs during each
        # while cycle
        # ---> actually this is what is happening...
        while True:
            self.draw_and_flip(exclude=['trace'])
            if self.verbose:
                if mouse.getPressed()[0]:
                    print(f'mouse pressed in pos: {mouse.getPos()}')
            if mouse.isPressedIn(self.frame_elements['start_point']):
                if self.verbose:
                    print('start point pressed')
                # change to cyan
                # MD: The condition down below did not work for me as fillColor returns
                # an array for me with [-1, 1, 1]
                # if self.frame_elements['start_point'].fillColor == [-1, 1, 1]:
                # DW: This throws an error for me.. I should update the yaml (it's
                # very old, incase our Psychopy versions are out of sync UPDATE:
                # I updated to Psychopy 2021.1.3 and got the same behaviour as you
                # if all(self.frame_elements['start_point'].fillColor == [-1, 1, 1]):
                self.frame_elements['start_point'].fillColor = [-1, 1, 1]
                tic = clock.getTime()
                self.draw_and_flip(exclude=['trace', 'instructions'])
                break

        while True:
            # drawing has begun
            if mouse.isPressedIn(self.frame_elements['cursor']):
                started_drawing = True
                self._lifted = False

                trial_timer = clock.CountdownTimer(self.trial_duration)

                # save start time
                start_t_stamp = clock.getTime()

                # calc pre trial time
                ptt = start_t_stamp - tic

            # shrink time bar, draw trace, once drawing has started
            if started_drawing:

                self._exec_drawing(trial_timer, mouse, time_bar_x, cursor_t)

                # time_bar elapsed
                if self.verbose:
                    print('breaking out of main')

                break

        return ptt, start_t_stamp, cursor_t

    def _create_trial_res(self, trace_let, trial_time, ptt, start_t_stamp,
                          trace_let_pix, scale, cursor_t):
        """ Creates the results dict that contains basic/essential trial info
        to be saved. """
        # original data + metadata
        self.trial_results = {'trace_let': trace_let, 'trial_time': trial_time,
                              'ix_block': self.block_idx,
                              'ix_trial': self.trial_idx, 'ptt': ptt,
                              'start_t_stamp': start_t_stamp}

        # new/extra metadata
        if scale:
            self.trial_results['scaling_matrix'] = self.stimuli['scaling_matrix']

        self.trial_results['trial_duration'] = self.trial_duration
        self.trial_results['flip'] = self.stimuli['flip']
        self.trial_results['the_box'] = self.frame_elements[
            'the_box'].vertices.copy()

        self.trial_results['theBoxPix'] = self.frame_elements[
            'the_box'].verticesPix

        self.trial_results['cursor_t'] = cursor_t

        if (trace_let != trace_let_pix).any():
            self.trial_results['pos_t_pix'] = trace_let_pix

        # in matlab i think this is theRect
        self.trial_results['winSize'] = self.win.size

        self.trial_results['template'] = self.current_stimulus
        stim_size = self.frame_elements['template'].size
        stim_pos = self.frame_elements['template'].pos
        self.trial_results['template_pix'] = convertToPix(self.current_stimulus,
                                                          units='norm',
                                                          pos=stim_pos,
                                                          win=self.win)
        self.trial_results['template_size'] = stim_size
        self.trial_results['template_pos'] = stim_pos
        # self.trial_results['templatePix'] = self.frame_elements[
        #     'template'].verticesPix
        # do i need to add theWord? maybe - is the index enough?

    def _adjust_time_bar(self, ratio, time_bar_x):
        """ Method for adjusting the size of time bar. Wrote
         mainly to aid in profiling. """
        new_size = [time_bar_x * ratio,  # change the x value
                    self.frame_elements['time_bar'].size[1]]
        new_pos = [(-time_bar_x * ratio / 2) + time_bar_x / 2,
                   self.frame_elements['time_bar'].pos[1]]

        self.frame_elements['time_bar'].setSize(new_size)
        self.frame_elements['time_bar'].setPos(new_pos)

    def _move_cursor(self, mouse, t_remain, cursor_t):
        """ Method for adjusting the position of the cursor and drawing the
        trace. Wrote mainly to aid profiling."""

        new_trace = False
        # Get new position from mouse
        if mouse.getPressed()[0]:
            if self._lifted:
                new_trace = True
                self._start_frame_idx = self._frame_idx + 1
            self._lifted = False
            new_pos = convertToPix(mouse.getPos(), (0, 0), units=mouse.units,
                                   win=self.win)
        else:
            self._lifted = True
            new_pos = self.trace_vertices[self._frame_idx]

        # Record time at which that happened
        # cursor_t.append(t_remain)
        cursor_t[self._frame_idx] = t_remain

        # Move cursor to that position and save
        self.frame_elements['cursor'].pos = new_pos

        # For drawing trace

        self._draw_trace(new_pos, new_trace=new_trace)

    def _draw_trace(self, new_pos, new_trace=False):
        """ Method that controls trace drawing (doesn't actually draw it just
        controls the vertices). - for profiling. """
        # ISSUE currently cant do non continuous lines
        # Draw trace, or rather, add new_pos to the trace
        self.trace_vertices[self._frame_idx+1] = new_pos

        if new_trace:
            tr_i = int(self.traces[-1].replace('trace', '')) + 1
            tr_i_n = 'trace' + str(tr_i)
            self.frame_elements[tr_i_n] = create_element(
                'trace',
                win=self.win,
                vertices=self.trace_vertices[-1],       # init does not matter, will be overwritten immediately after
            )
            self.traces.append(tr_i_n)

        # set the trace
        self.frame_elements[self.traces[-1]].vertices =\
            self.trace_vertices[self._start_frame_idx:self._frame_idx+1]

    def _exec_drawing(self, trial_timer, mouse, time_bar_x, cursor_t):
        """ All the drawing stuff goes in this """
        while trial_timer.getTime() > 0:

            # get remaining time
            t_remain = trial_timer.getTime()
            ratio = t_remain / self.trial_duration

            # adjust time_bar size and position
            if self._frame_idx % 2 == 0:  # every other frame
                self._adjust_time_bar(ratio, time_bar_x)

            # update cursor position
            self._move_cursor(mouse, trial_timer.getTime(), cursor_t)

            # only draw every xth frame
            if self._frame_idx % 2 == 0:
                self.draw_and_flip(exclude=['instructions'])

            if (not mouse.getPressed()[
                0] and self.block_settings['finish_when_raised']):
                if self.verbose:
                    print('mouse raised - ending')
                # main_loop = False
                break
            self._frame_idx += 1


if __name__ == "__main__":
    try:
        test = CopyDraw('./',
                        )

        test.init_session('TEST_SESSION')
        test.init_block(block_name='TEST_BLOCK',
                        n_trials=2,
                        letter_time=2.7,
                        finish_when_raised=False
                       )
        test.exec_block()
        test.exit()

    # this still isnt printing the error out, why?
    except Exception as e:
        core.quit()
        print(e)
