# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:59:38 2020

@author: Daniel
"""
import numpy as np
import pandas as pd
import scipy.io
import time
import utils
import json
import logging  # Does this need to be in here?
from elements import create_element
from base import AbstractParadigm
from psychopy import core, event, clock
from psychopy.tools.monitorunittools import convertToPix  # , posToPix
from pathlib import Path
from utils import select_display, create_template_order


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
class CopyDraw(AbstractParadigm):

    def __init__(self,
                 data_dir,
                 screen_ix=None,
                 flip=True,  # should this be passed in somewhere else?
                 lpt_address=None,
                 serial_nr=None,
                 verbose=True):
        self.log = logging.getLogger(__name__)
        self.log.info('Initialising..')
        super().__init__(screen_ix=screen_ix, lpt_address=lpt_address,
                         serial_nr=serial_nr)
        # would it be beneficial to have a method that takes a snapshot of the
        # all these settings dicts whenever called?

        # should these be one dict or kept separate?
        self.paths = {}
        self.names = {}

        self.win_settings = {}
        self.frame = {}
        self.stimuli = {'flip': flip}

        # What is better practice? initialise empty settings dict here
        # and 'reset' it to blank dict at the end of trial/block or
        # set as None here and create the dict in the trial/block methods?
        self.trial_settings = {}  # is this an appropriate name?
        self.block_settings = None  # will be dict

        self.paths['data'] = Path(data_dir)
        self.paths['results'] = self.paths['data'] / 'results'
        self.screen_ix = screen_ix or select_display()['screen']
        self.log.debug(f'Paths: {self.paths}')
        self.log.debug(f'Screen: {self.screen_ix}')

        self.block_idx = None  # move into block settings? maybe not. Currently needed after block settings reset
        self.trials_vec = None
        self.trial_idx = None
        self.stimuli['the_box'] = None  # should this go in block/trial settings?
        self.block_results = None
        self.trial_results = None


        self.verbose = verbose  # For debugging, toggles print messages
        # Change the prints to leg messages when you figure out how to self.log stuff
        if self.verbose:
            print('initialised')

    def init_session(self, session_name=None, screen_size=(1000, 600)):

        if session_name is None:
            self.names['session'] =\
                time.asctime(time.localtime(time.time())).replace(':', '-')
        else:
            self.names['session'] = session_name
        self.log.info(f'Initialised session: {self.names["session"]}')
        self.paths['session'] = self.paths['results'] / self.names['session']
        self.paths['session'].mkdir(parents=True, exist_ok=True)
        self.win_settings['screen_size'] = screen_size
        # self.paths['info_runs'] = self.paths['session'] / 'info_runs'
        # self.paths['info_runs'].mkdir(exist_ok=True)
        self.block_idx = 1  # this gets +1'd every time exec block is called

    def init_block(self, block_name=None,
                   n_trials=12,
                   # screen_size=(920, 1200),  # 16:10 ratio
                   letter_time=2.2,
                   finish_when_raised=True,
                   n_letters=3,
                   stim_size=35,  # this is the size of the templates used
                   size=1.5,  # this is the scaling factor applied to the template
                   shuffle=True,
                   interp=True,
                   win_color=(-1, -1, -1)
                   ):

        super().init_block(self.win_settings['screen_size'])
        self.block_settings = {}
        self.win_settings['aspect_ratio'] = self.win.size[0] / self.win.size[1]
        self.win_settings['color'] = win_color
        self.block_settings['n_trials'] = n_trials
        self.block_settings['letter_time'] = letter_time
        self.block_settings['n_letters'] = n_letters
        self.block_settings['finish_when_raised'] = finish_when_raised
        self.block_settings['interp'] = interp
        self.block_settings['stim_size'] = stim_size
        self.block_settings['shuffle'] = True
        self.block_settings['size'] = size
        self.win.color = self.win_settings['color']

        time_str = time.asctime(time.localtime(time.time()))
        self.block_settings['block_name'] = \
            block_name or f'BLOCK_{time_str.replace(":", "-")}'
        self.log.info(f'Initialised block: {self.block_settings}\n{self.win_settings}')
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

        # external self.logger, how to integrate?
        # # self.log - use this!
        # fh = self.logging.FileHandler(self.paths['block'] / 'debug.log')
        # fh.setLevel(self.logging.DEBUG)
        # self.logger.addHandler(fh)
        # formatter = self.logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \
        #                               - %(message)s')
        # fh.setFormatter(formatter)
        # self.logger.addHandler(fh)
        # self.logger.info(f'Block "{self.block_settings["block_name"]}" initialised')

    def exec_block(self, cfg, stim='off'):
        """ Will call init_block(**cfg) before calling exec trial n_trials
        times, also calling save_trial for each. Trials vector and block
         settings saved at the end. """

        self.init_block(**cfg)

        # call exec trial 12? times
        if self.verbose:
            print(f'executing block {self.block_idx}')
        self.log.info(f'executing block {self.block_idx}')
        self.block_results = {}

        for stimuli_idx in range(self.block_settings['n_trials']):
            self.exec_trial(stimuli_idx)
            self.save_trial()  # should this call be outside of here?
            self.trial_idx += 1
            # run some kind of check_trial func here?
            self.block_results[stimuli_idx] = self.trial_results

        self.save_trials_vector()
        self.save_block_settings()
        self.log.info('Saved block settings')
        self.block_settings = None  # 'reset' block settings
        if self.verbose:
            print('reset block settings')
        self.log.info('Resetting block settings for new block')
        self.block_idx += 1

    def load_instructions(self, path):  # This seems kind of useless, remove?
        self.paths['instructions'] = Path(path)
        assert self.paths['instructions'].exists(), f'Instructions file not found \
            {self.paths["instructions"]}'
        if self.verbose:
            print('instructions loaded')

    def get_box(self, idx):

        # MD: This is an inefficient pattern, as for each call to get box,
        # we will perform a glob / parse directories. Better get all files dirs
        # mapped to indices during initialization and later load only the path
        # --> effectively only one "glob" vs n
        # box = self.get_old_stimuli(idx)
        # self.stimuli['the_box'] = box['boxPos'].T  # box[0,0][0].T.astype(float)

        # MD -> as this seems rather constant for a given size and n_letters ==> just hardcode for now
        # DW: We could hardcode the box sizes into a config file and open from there
        # or should it be hardcoded into this func (or somewhere else)
        tb = np.array([[285, 392],
                       [1635, 392],
                       [285, 808],
                       [1635, 808],
                       [288, 392],
                       [288, 808],
                       [1632, 392],
                       [1632, 808],
                       ])

        self.stimuli['the_box'] = tb

        return self.stimuli['the_box']

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
            self.stimuli['templates'] = self.stimuli['file']['new_shapes'][0]
        except Exception as err:  # rewrite
            if self.verbose:
                print(err)
        if self.verbose:
            print('stimuli loaded')

        self.stimuli['n_templates'] = self.stimuli['templates'].shape[0]

    def get_stimuli(self, stimuli_idx, scale=True):
        # this func is a little long, how can it be broken up?
        try:
            self.stimuli['current_stim'] = self.stimuli['templates'][stimuli_idx].astype(float)
            self.get_box(stimuli_idx)
            self.trials_vec['id'].append(stimuli_idx)
            self.trials_vec['lengths'].append(self.block_settings['n_letters'])
            if scale:
                self.log.info('Scaling data stim...')
                self.stimuli['current_stim'], self.stimuli['scaling_matrix'] = \
                    utils.scale_to_norm_units(self.stimuli['current_stim'])
                self.stimuli['the_box'], _ = \
                    utils.scale_to_norm_units(self.stimuli['the_box'],
                                              scaling_matrix=self.stimuli['scaling_matrix'])

                # reorder
                new_box_array = np.zeros([4, 2])
                new_box_array[0:2] = self.stimuli['the_box'][0:2]
                new_box_array[2] = self.stimuli['the_box'][3]
                new_box_array[3] = self.stimuli['the_box'][2]

                self.stimuli['the_box'] = new_box_array

                if self.verbose:
                    print('scaled')
                if self.stimuli['flip']:
                    self.log.info('Flippinf data...')
                    self.stimuli['current_stim'] = \
                        np.matmul(self.stimuli['current_stim'],
                                  np.array([[1, 0], [0, -1]]))
                    self.stimuli['the_box'] = np.matmul(self.stimuli['the_box'],
                                                        np.array([[1, 0], [0, -1]]))

            if self.block_settings['interp']:
                self.log.info('Smoothing data...')
                if self.verbose:
                    print('smoothing')
                self.stimuli['current_stim'] = \
                    utils.smooth(self.stimuli['current_stim'], return_df=False)

            # elif self.block_settings['interp'] is not None:
            #     tck, u = splprep(self.stimuli['current_stim'].T, s=self.block_settings['interp']['s'])
            #
            #     # using a high value for n_interp along with manyshape == True
            #     # results in laggy performance
            #
            #     u_new = np.linspace(np.min(u), np.max(u), self.block_settings['interp']['n_interp'])
            #     self.stimuli['current_stim'] = np.array(splev(u_new, tck)).T
            #     # note this is replacing the original points, not adding

            self.trial_settings['cursor_start_pos'] = self.stimuli['current_stim'][0]
            return self.stimuli['current_stim']

        except AttributeError:
            if self.verbose:
                print('load_stimuli must be called first')
            self.logger.error('Failed with AttributeError, probably due to \
                         load_stimuli not being called')

        except IndexError:
            if self.verbose:
                print(f'only {len(self.stimuli["templates"])} templates')
            self.logger.error('Failed with IndexError, probably related to number \
                         of templates')

    def save_trial(self):
        # rudimentary atm, can ble cleaned up, flattened a little maybe
        df = pd.Series(self.trial_results)
        fname = f'tscore_{self.trial_idx}copyDraw_block{self.block_idx}.pkl'
        df.to_pickle(self.paths['block'] / fname)
        self.log.info(f'Saved trial: {self.paths["block"] / fname}')

    def save_block_settings(self):
        fname = self.paths['block'] / f'block_{self.block_idx}_fbsettings.json'
        with open(fname, 'w') as h:
            json.dump(self.block_settings, h)

    def save_trials_vector(self):
        # What useful relevant info should actually be saved in here, that
        # couldn't simply be put in the trial pkl file or the block settings?
        # maybe change from json later, this is nice and readable for now
        fname = self.paths['block'] / f'tinfo_copyDraw_block{self.block_idx}.json'
        with open(fname, 'w') as h:
            json.dump(self.trials_vec, h)

    # draw order is based on .draw() call order, consider using an ordered dict?
    # MD: Ordered dict would be a good idea
    def draw_and_flip(self, exclude=[]):
        """ Draws every element in the frame elements dict, excluding those
         passed in via exclude. """
        for element_name, element in self.frame['elements'].items():
            if element_name in exclude:
                continue
            else:
                element.draw()
        self.win.flip()  # Bottleneck point, anything that can be done?

    def create_frame(self, stimuli_idx, scale=True):

        self.frame['elements'] = {}

        self.frame['elements']['template'] = create_element(
            'template',
            win=self.win,
            image=utils.template_to_image(
                self.get_stimuli(stimuli_idx, scale=scale),
                f'{self.stimuli["fname"][:-4]}_{stimuli_idx}',
                self.paths['data']/'template_images',
                lineWidth=15
            ),
            size=self.block_settings['size'] * 1.6725  # scaling
        )
        # uncomment below to draw old template to check how the image
        # matches up with it

        # # seems to top out after like 10 or 20:
        # https://github.com/psychopy/psychopy/issues/818
        # from psychopy.visual import ShapeStim
        # self.frame['elements']['old_template'] = visual.ShapeStim(win=self.win,
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
        # self.frame['elements']['template'].setOpacity(0.5)

        self.frame['elements']['the_box'] = create_element('the_box',
                                                           win=self.win,
                                                           vertices=self.stimuli['the_box']
                                                           )

        self.frame['elements']['trial_number'] = create_element(
            'trial_number',
            win=self.win,
            text=f'Trial {self.trial_idx}/{self.block_settings["n_trials"]}',
        )

        self.frame['elements']['cursor'] = create_element(
            'cursor',
            win=self.win,
            pos=convertToPix(
                # so the 1.5 comes from how much the original template had to be scaled by after being converted
                # to norm units. the 1.6.. in the template section is how much the resulting image of the template had
                # to be scaled by in order to match up with the original template when drawn on top of each other
                self.trial_settings['cursor_start_pos'] * 1.5 * self.block_settings['size'],
                (0, 0),
                'norm',
                self.win
            )
        )

        max_trace_len = 10000  # Should be more points than would ever be drawn
        self.frame['trace_vertices'] = np.zeros([max_trace_len, 2])
        self.frame['trace_vertices'][0] = \
            convertToPix(self.trial_settings['cursor_start_pos'] * 1.5 * self.block_settings['size'],
                         (0, 0),
                         'norm',
                         self.win)
        # TODO: clicking NOT on cursor (post startpoint press) results in no trace being drawn - fix
        # trace starts to get drawn after sometime - some kind of lag on recording of trace vertices..
        self.frame['elements']['trace1'] = create_element(
           'trace',                           # we will dynamically create more -> draw interupted lines
            win=self.win,
            vertices=self.frame['trace_vertices'][0:1],          # initialize empty - now take first point?
        )
        self.frame['traces'].append('trace1')

        self.frame['elements']['instructions'] = create_element(
            'instructions',
            win=self.win,
            image=self.paths['instructions']
        )

        # start_point
        start_point_size = 0.05
        self.frame['elements']['start_point'] = create_element(
            'start_point',
            win=self.win,
            size=(start_point_size, start_point_size * self.win_settings['aspect_ratio'])
        )

        self.frame['elements']['time_bar'] = create_element(
            'time_bar',
            win=self.win
        )

    # should stimuli_idx just be called trial_idx?
    # currently they are initialised differently (0 and 1)
    def exec_trial(self, stimuli_idx, scale=True):
        """ Top level method that executes a single trial """

        self.frame['lifted'] = True             # track if mouse is currently lifted

        self.frame['start_frame_idx'] = 0       # frame index of the last started trace
        # Moved here to fix bug regarding number of traces in subsequent trials
        self.frame['traces'] = []                # list to record multiple trace object names --> drawing interupted lines
        self.log.info(f'Executing trial with stim idx {stimuli_idx}')
        if self.verbose:
            print('executing trial')

        # define duration as how long it runs for max, trial_time as actual
        # runtime (finish when raised etc)
        self.trial_settings['trial_duration'] = self.block_settings['n_letters'] * self.block_settings['letter_time']

        # initialise the frame
        self.create_frame(self.stimuli['order'][stimuli_idx], scale=scale)

        # getting a monitor gamut error when doing rgb colours, look into it
        # self.trace.colorSpace = 'rgb255'

        # draw first frame
        self.draw_and_flip(exclude=['time_bar', 'trace1'])

        # time_bar
        time_bar_x = self.frame['elements']['time_bar'].size[0]

        # main bit
        self.frame['idx'] = 0  # refers only to frames during drawing trace
        ptt, start_t_stamp, cursor_t = self._run_trial_main_loop(clock,
                                                                 time_bar_x)

        cursor_t = cursor_t[:self.frame['idx']+1]  # truncate

        # trial time is how long they were drawing for,
        # ie time elapsed during drawing
        trial_time = self.trial_settings['trial_duration'] - cursor_t[-1]
        self.log.info(f'Trial lasted {trial_time} seconds')
        if self.verbose:
            print(f"recorded {self.frame['idx']} points at a rate of"
                  f" {self.frame['idx'] / trial_time} points per sec")
        self.log.info(f'Drew {self.frame["idx"]} frames')
        self.log.info(f'Recording rate was {len(cursor_t) / trial_time} points per second')
        # should there be any unit conversions done?

        # # separate the main loop and data stuff
        # trace_let = self.frame['elements']['trace'].vertices.copy()
        # trace_let_pix = self.frame['elements']['trace'].verticesPix.copy()
        #
        # For interupted traces, we now have to concatenate
        # General TODO: --> how to deal with the jumps in terms of dtw...
        trace_let = np.concatenate([self.frame['elements'][tr_n].vertices.copy()
                                    for tr_n in self.frame['traces']])
        traces_pix = [self.frame['elements'][tr_n].verticesPix.copy()
                      for tr_n in self.frame['traces']]
        trace_let_pix = np.concatenate(traces_pix)

        self._create_trial_res(trace_let, trial_time, ptt, start_t_stamp,
                               trace_let_pix, scale, cursor_t, traces_pix)

    def exit(self):
        self.log.info('Exiting')
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
            if mouse.isPressedIn(self.frame['elements']['start_point']):
                self.log.debug('Mouse pressed in startpoint')
                # change to cyan
                # older psychopy versions use str instead of arr to set color
                # eg 'Cyan'
                self.frame['elements']['start_point'].fillColor = [-1, 1, 1]
                tic = clock.getTime()
                self.draw_and_flip(exclude=['trace', 'instructions'])
                break

        while True:
            # drawing has begun
            if mouse.isPressedIn(self.frame['elements']['cursor']):
                self.log.debug('Mouse started drawing with cursor')
                started_drawing = True
                self.frame['lifted'] = False

                trial_timer = clock.CountdownTimer(self.trial_settings['trial_duration'])

                # save start time
                start_t_stamp = clock.getTime()

                # calc pre trial time
                ptt = start_t_stamp - tic

            # shrink time bar, draw trace, once drawing has started
            if started_drawing:

                if self.verbose:
                    print('STARTED DRAWING')
                self._exec_drawing(trial_timer, mouse, time_bar_x, cursor_t)

                # time_bar elapsed
                if self.verbose:
                    print('breaking out of main')

                break

        return ptt, start_t_stamp, cursor_t

    def _adjust_time_bar(self, ratio, time_bar_x):
        """ Method for adjusting the size of time bar. Wrote
         mainly to aid in profiling. """
        new_size = [time_bar_x * ratio,  # change the x value
                    self.frame['elements']['time_bar'].size[1]]
        new_pos = [(-time_bar_x * ratio / 2) + time_bar_x / 2,
                   self.frame['elements']['time_bar'].pos[1]]

        self.frame['elements']['time_bar'].setSize(new_size)
        self.frame['elements']['time_bar'].setPos(new_pos)

    def _move_cursor(self, mouse, t_remain, cursor_t):
        """ Method for adjusting the position of the cursor and drawing the
        trace. Wrote mainly to aid profiling."""

        new_trace = False
        # Get new position from mouse
        if mouse.getPressed()[0]:
            if self.frame['lifted']:
                new_trace = True
                self.frame['start_frame_idx'] = self.frame['idx'] + 1
            self.frame['lifted'] = False
            new_pos = convertToPix(mouse.getPos(), (0, 0), units=mouse.units,
                                   win=self.win)
        else:

            self.frame['lifted'] = True
            new_pos = self.frame['trace_vertices'][self.frame['idx']]

        # Record time at which that happened
        # cursor_t.append(t_remain)
        cursor_t[self.frame['idx']] = t_remain

        # Move cursor to that position and save
        self.frame['elements']['cursor'].pos = new_pos

        # For drawing trace
        self._draw_trace(new_pos, new_trace=new_trace)

    def _draw_trace(self, new_pos, new_trace=False):
        """ Method that controls trace drawing (doesn't actually draw it just
        controls the vertices). - for profiling. """
        # Draw trace, or rather, add new_pos to the trace
        self.frame['trace_vertices'][self.frame['idx']+1] = new_pos

        if new_trace:
            if self.verbose:
                print('NEW TRACE')
            tr_i = int(self.frame['traces'][-1].replace('trace', '')) + 1
            tr_i_n = 'trace' + str(tr_i)
            self.frame['elements'][tr_i_n] = create_element(
                'trace',
                win=self.win,
                vertices=self.frame['trace_vertices'][-1],       # init does not matter, will be overwritten immediately after
            )
            self.frame['traces'].append(tr_i_n)

        # set the trace
        self.frame['elements'][self.frame['traces'][-1]].vertices =\
            self.frame['trace_vertices'][self.frame['start_frame_idx']:self.frame['idx']+1]

    def _exec_drawing(self, trial_timer, mouse, time_bar_x, cursor_t):
        """ All the drawing stuff goes in this """
        while trial_timer.getTime() > 0:

            # get remaining time
            t_remain = trial_timer.getTime()
            ratio = t_remain / self.trial_settings['trial_duration']

            # adjust time_bar size and position
            if self.frame['idx'] % 2 == 0:  # every other frame
                self._adjust_time_bar(ratio, time_bar_x)

            # update cursor position
            self._move_cursor(mouse, trial_timer.getTime(), cursor_t)

            # only draw every xth frame
            if self.frame['idx'] % 2 == 0:
                self.draw_and_flip(exclude=['instructions'])

            if (not mouse.getPressed()[0] and
                    self.block_settings['finish_when_raised']):
                if self.verbose:
                    print('mouse raised - ending')
                # main_loop = False
                break
            self.frame['idx'] += 1

    def _create_trial_res(self, trace_let, trial_time, ptt, start_t_stamp,
                          trace_let_pix, scale, cursor_t, traces_pix):
        """ Creates the results dict that contains basic/essential trial info
        to be saved. """

        self.log.debug('Creating trial results')

        # original data + metadata
        self.trial_results = {'trace_let': trace_let, 'trial_time': trial_time,
                              'ix_block': self.block_idx,
                              'ix_trial': self.trial_idx, 'ptt': ptt,
                              'start_t_stamp': start_t_stamp}

        # new/extra metadata
        if scale:
            self.trial_results['scaling_matrix'] = self.stimuli['scaling_matrix']
        self.trial_results['traces_pix'] = traces_pix
        self.trial_results['n_traces'] = len(self.frame['traces'])
        self.trial_results['trial_duration'] = self.trial_settings['trial_duration']
        self.trial_results['flip'] = self.stimuli['flip']
        self.trial_results['the_box'] = self.frame['elements'][
            'the_box'].vertices.copy()

        self.trial_results['theBoxPix'] = self.frame['elements'][
            'the_box'].verticesPix

        self.trial_results['cursor_t'] = cursor_t

        if (trace_let != trace_let_pix).any():
            self.trial_results['pos_t_pix'] = trace_let_pix

        # in matlab i think this is theRect
        self.trial_results['winSize'] = self.win.size

        self.trial_results['template'] = self.stimuli['current_stim']
        stim_size = self.frame['elements']['template'].size
        stim_pos = self.frame['elements']['template'].pos
        self.trial_results['template_pix'] = convertToPix(self.stimuli['current_stim'],
                                                          units='norm',
                                                          pos=stim_pos,
                                                          win=self.win)
        self.trial_results['template_size'] = stim_size
        self.trial_results['template_pos'] = stim_pos
        # self.trial_results['templatePix'] = self.frame['elements'][
        #     'template'].verticesPix
        # do i need to add theWord? maybe - is the index enough?

if __name__ == "__main__":
    try:
        import logging
        data_dir = '../'
        test_cpd = CopyDraw(data_dir,
                            verbose=True)
        log = logging.getLogger(__name__)
        log.info('Started copydraw')

        logging.basicConfig(filename='test_run.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)

        session_cfg = {
            'session_name': 'TEST_SESSION',
            'screen_size': (1000, 600)
        }
        test_cpd.init_session(**session_cfg)

        # for integration - will this be a yaml file?
        cfg = {
            'block_name': 'TEST_BLOCK_1',
            'n_trials': 2,
            'letter_time': 2.7,
            'n_letters': 3,
            'finish_when_raised': False,
            'stim_size': 35,
            'size': 1,  # move into session cfg?
        }
        test_cpd.exec_block(cfg, stim='off')

        cfg = {
            'block_name': 'TEST_BLOCK_2',
            'n_trials': 1,
            'letter_time': 2.2,
            'n_letters': 3,
            'finish_when_raised': False,
            'stim_size': 35,
            'size': 0.5,
        }
        test_cpd.exec_block(cfg, stim='off')

        test_cpd.exit()

    # this still isnt printing the error out, why?
    except Exception as e:
        core.quit()
        print(e)
