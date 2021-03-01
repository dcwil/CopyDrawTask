# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:59:38 2020

@author: Daniel
"""
import numpy as np
import pandas as pd
import scipy.io
import math
import time
import logging
import random
import utils
import json

from base import AbstractParadigm
from psychopy import visual, core, event, clock
from psychopy.tools.monitorunittools import convertToPix,posToPix
from numpy.linalg import norm
from pathlib import Path
from itertools import permutations
from scipy.interpolate import splprep, splev
from template_image import template_to_image

logger = logging.getLogger('CopyDraw')
#logger.setLevel(logging.DEBUG)


# boxColour = [160,160,180]
# boxHeight = 200
# boxLineWidth = 6
# templateColour = [80,80,150]
# templateThickness = 3
# traceColour = [255,50,50]
# traceThickness = 1

# startTrialBoxColor = [50, 255, 255]
# textColour = [153, 153, 255]
#timeCOlour = [180, 180, 160]


#Try to use fewer ternary operators - so you get shorter lines (PEP8)
class ManyShapeStim():
    """
    For creating a shapestim like object, made of many individual psychopy shapes
    """
    def __init__(self,
                 vertices,
                 shape,
                 shape_params,
                 size=1):
        
        vertices = [x*size for x in vertices]
        self.shapes = [shape(pos=pos,**shape_params) for pos in vertices]
        self.verticesPix = np.array([convertToPix(vertex,
                                                  (0,0),
                                                  shape_params['units'],
                                                  shape_params['win']) 
                                     for vertex in vertices])

    def draw(self):
        for shape in self.shapes:
            shape.draw()
            
    

class CopyDraw(AbstractParadigm):
    
    #should there be an n_blocks arg? probably
    # some of these init argsshould be moved to the block init
    def __init__(self,
                 data_dir,
                 screen_size = (1920, 1200), #16:10 ratio
                 screen_ix = 0,
                 flip=True,
                 lpt_address = None,
                 serial_nr = None,
                 old_template_path='../CopyDraw_mat_repo/CopyDrawTask-master/templates/shapes'):
        super().__init__(screen_ix=screen_ix, lpt_address=lpt_address,
                         serial_nr=serial_nr)
        
        
        self.data_dir = Path(data_dir)
        self.screen_size = screen_size
        #self.trial_clock = core.Clock() # use this!
        self.flip = flip
        self.old_template_path = old_template_path
        
        self.results_dir = self.data_dir / 'results'
        print('initialised')
        
    def init_session(self,session_name=None):
        
        if session_name == None:
            self.session_name = time.asctime( time.localtime(time.time()) ).replace(':','-')
        else:
            self.session_name = session_name
        self.session_dir = self.results_dir / self.session_name 
        self.session_dir.mkdir(parents=True,exist_ok=True)
        
        self.info_runs = self.session_dir / 'info_runs'
        self.info_runs.mkdir(exist_ok=True)
        self.block_idx = 1 # this gets +1'd every time exec block is called
        
    def init_block(self,block_name=None,
                   n_trials=12,
                 letterTime=2.2,
                 image_size=2,
                 finishWhenRaised=True,
                 n_letters=3,
                 stim_size=35,
                 manyshape=False, #new or old rendering #pretty much deprecated at this point
                 interp = 'smooth', #{'s':0.001,'n_interp':300}, dict or None or 'smooth'
                 ):
        
        super().init_block(self.screen_size)
        self.win.color = (-1,-1,-1)
        self.aspect_ratio = self.win.size[0]/self.win.size[1]
        self.msperframe,_,_ = self.win.getMsPerFrame()   #this is redundant now?
        self.n_trials = n_trials
        self.letterTime = letterTime # Start sticking to either _ or mixedCase
        self.image_size = image_size
        self.n_letters = n_letters
        self.finishWhenRaised = finishWhenRaised 
        self.manyshape = manyshape
        self.interp = interp
        self.stim_size = stim_size
        
        if block_name == None:
            time_str = time.asctime( time.localtime(time.time()) )
            self.block_name = f'BLOCK_{time_str.replace(":","-")}'
        else:
            self.block_name = block_name
        
        #dropped frames
        self.win.recordFrameIntervals = True
 
        self.load_stimuli(self.data_dir / "templates",
                          short=True if self.n_letters == 2 else False,
                          size=self.stim_size)

        ### move object creation out of here? ###
        #instructions
        self.load_instructions(self.data_dir /
                               'instructions' /
                               'instructions.png')

        
        #folders for saving
        self.block_dir = self.session_dir / self.block_name
        self.block_dir.mkdir(parents=True, exist_ok=True)
        
        #create trials vector
        self.trialsVector = {
            'names' : [], #231, 123 etc
            'places' : [],#path for each one, is this needed, there's not an individual path for each anymore
            'types' : [], #not sure what this is
            'lengths' : [], #n_letters
            'id' : [] #index
            }
        
        #trial index
        self.trial_idx = 1
        
        #create template order
        self.create_template_order(shuffle=True)
        
        #block settings
        self.block_settings = {
            'n_trials':self.n_trials,
            'letterTime':self.letterTime,
            'stim_size':self.stim_size,
            'block_name':self.block_name,
            'n_letters':self.n_letters,
            'finishWhenRaised':self.finishWhenRaised,
            }
        
        
        
        
        #log - use this!
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
        print('instructions loaded')
        
        
    def get_old_stimuli(self,stimuli_idx,scale=True,template_or_box='theBox'):
        old_stimuli_path = Path(self.data_dir,self.old_template_path)
        assert old_stimuli_path.exists(),f"old Stimuli data not found: {old_stimuli_path}"
        
        
        if self.stimuli_fname[:-4].endswith('5'):
            folder = self.stimuli_fname[:-5]+'.5'
        else:
            folder = self.stimuli_fname[:-4]
        
        idx_to_fname = {idx:''.join(p) + '.mat' 
                        for idx,p in enumerate(permutations(['1','2'] 
                                                            if 'short' 
                                                            in folder 
                                                            else ['1','2','3']))}
        
        self.old_stimuli_path = old_stimuli_path / folder / idx_to_fname[stimuli_idx]     
        self.old_stimuli = scipy.io.loadmat(self.old_stimuli_path,
                                            simplify_cells=True)
        
        return self.old_stimuli[template_or_box]
    
    
    def get_box(self,idx):
        box = self.get_old_stimuli(idx)
        self.theBox = box['boxPos'].T#box[0,0][0].T.astype(float)
        return self.theBox
        
    
    def load_stimuli(self, path, short=True, size=35):
        self.stimuli_fname = f'Size_{"short_" if short else ""}{size}.mat'
        stimuli_path = Path(path, self.stimuli_fname)
        print(f'loading stimuli: {stimuli_path}')
        assert stimuli_path.exists(), f"Stimuli data not found: {stimuli_path}"
        
        # old_stimuli_path = Path(self.data_dir,self.old_template_path)
        # assert old_stimuli_path.exists(), f"old Stimuli data not found: {old_stimuli_path}"

        try:
            self.stimuli_file = scipy.io.loadmat(stimuli_path)
            print('loaded mat file')
            self.templates = self.stimuli_file['new_shapes'][0]
        except Exception as e:
            print(e)
        print('stimuli loaded')
        
        self.n_templates = self.templates.shape[0]
       
    #could be in a separate file, seems like a util
    def create_template_order(self,shuffle=True):
        #requires stimuli to be loaded & n_trials to have been defined
        if self.n_templates % self.n_trials != 0:
            #change to a proper warning message?
            print(f'WARNING: {self.n_trials} trials means that there will be an uneven number of templates')
        
        self.order = [ (i % self.n_templates) for i in range(self.n_trials)] 
        
        if shuffle:
            random.shuffle(self.order)
            
            #reshuffle to remove repeated trials showing
            while 0 in np.diff(np.array(self.order)):
                random.shuffle(self.order)
                
            
        
        
        
        
    def get_stimuli(self, stimuli_idx,scale=True): 
        try:
            self.current_stimulus = self.templates[stimuli_idx].astype(float)
            self.get_box(stimuli_idx)
            self.trialsVector['id'].append(stimuli_idx)
            self.trialsVector['lengths'].append(self.n_letters)
            if scale:
                
                self.current_stimulus,self.scaling_matrix = \
                    utils.scale_to_norm_units(self.current_stimulus)
                self.theBox,_ = \
                    utils.scale_to_norm_units(self.theBox,
                                              scaling_matrix=self.scaling_matrix)
                
                #reorder
                newboxarray = np.zeros([4,2])
                newboxarray[0:2] = self.theBox[0:2]
                newboxarray[2],newboxarray[3] = self.theBox[3],self.theBox[2]
                self.theBox = newboxarray

                
                print('scaled')
                if self.flip:
                    self.current_stimulus = np.matmul(self.current_stimulus,
                                                      np.array([[1,0],[0,-1]]))
                    self.theBox = np.matmul(self.theBox,
                                            np.array([[1,0],[0,-1]]))
                    
            
            if self.interp == 'smooth':        
                self.current_stimulus = utils.smooth(self.current_stimulus,
                                                     return_df=False)
            
            elif self.interp is not None:
                tck, u = splprep(self.current_stimulus.T, s=self.interp['s'])
                
                
                ### using a high value for n_interp along with manyshape == True
                ### results in laggy performance
                
                u_new = np.linspace(np.min(u), np.max(u), self.interp['n_interp'])
                self.current_stimulus = np.array(splev(u_new, tck)).T
                # note this is replacing the original points, not adding
                
            
            self.cursorStartpoint = self.current_stimulus[0]
            return self.current_stimulus
        
        except AttributeError:
            print('load_stimuli must be called first')
            logger.error('Failed with AttributeError, probably due to \
                         load_stimuli not being called')
            
        except IndexError:
            print(f'only {len(self.templates)} templates')
            logger.error('Failed with IndexError, probably related to number \
                         of templates')
    
    def exec_block(self):
        #call exec trial 12? times
        print(f'executing block {self.block_idx}')
        
        self.block_results = {}
        
        for stimuli_idx in range(self.n_trials):
            self.exec_trial(stimuli_idx)
            self.save_trial() # should this call be outside of here?
            self.trial_idx += 1
            #run some kind of check_trial func here?
            self.block_results[stimuli_idx] = self.trial_results
            
        self.save_trialsVector()
        self.save_block_settings()
        self.block_idx += 1

        
    def save_trial(self):
        #rudimentary atm, can ble cleaned up, flattened a little maybe
        self.df = pd.Series(self.trial_results)
        fname = f'tscore_{self.trial_idx}copyDraw_block{self.block_idx}.pkl'
        self.df.to_pickle(self.block_dir / fname
                          )
        
    def save_block_settings(self):
        fname = self.info_runs / f'block_{self.block_idx}_fbsettings.json'
        with open(fname,'w') as h:
            json.dump(self.block_settings,h)
                
    def save_trialsVector(self):
        #maybe change from json later, this is nice and readable fornow
        fname = self.block_dir / f'tinfo_copyDraw_block{self.block_idx}.json'
        with open(fname,'w') as h:
            json.dump(self.trialsVector, h)
        

    #draw order is based on .draw() call order, consider using an ordered dict?
    def draw_and_flip(self,exclude=[]):
        for element_name,element in self.frame_elements.items():
            if element_name in exclude:
                continue
            else:
                element.draw()
        self.win.flip()
    
    #how can i make this function smaller?
    def create_frame(self,stimuli_idx,scale=True):
        
        self.frame_elements = {}
        
        if self.manyshape:
            manyshape_size = 0.035 # equates to line thickness
            self.frame_elements['template'] = ManyShapeStim(
                self.get_stimuli(stimuli_idx,scale=scale),
                                                visual.Circle,
                                                {'win':self.win,
                                                'units':'norm',
                                                'size':(manyshape_size,
                                                        manyshape_size*self.aspect_ratio),
                                                'color':'blue',
                                                'fillColor':'blue',
                                                'lineColor':'blue'
                                                },
                                                size=1.5)
        else:
            # self.frame_elements['template'] = visual.ShapeStim(win=self.win,
            #                                         vertices=self.get_stimuli(stimuli_idx,scale=scale),
            #                                         lineWidth=100, # seems to top out after like 10 or 20: https://github.com/psychopy/psychopy/issues/818
            #                                         closeShape=False,
            #                                         interpolate=True,
            #                                         ori=0,
            #                                         pos=(0,0),
            #                                         size=1.5,
            #                                         units='norm',
            #                                         fillColor='blue',
            #                                         lineColor='blue',
            #                                         #windingRule=True,
            #                                         )
            
            
            #this would replace template from above ^ 
            #if you want to run without the original template in, 
            #delete the stored files with the thick lines
            #change line width <5
            #uncomment template above and change the one below to 'template_new'
            self.frame_elements['template'] = visual.ImageStim(
                win=self.win,
                image=template_to_image(
                    self.get_stimuli(stimuli_idx,scale=scale),
                    f'{self.stimuli_fname[:-4]}_{stimuli_idx}',
                    'template_images',
                    lineWidth=15),
                #this is from visual inspection - still not quite right, 
                # but its as close as its going to get
                # seems like image needs to be stretched ever so slightly in the y
                size=1.6725,
                interpolate=True,
                pos=(-0.0025,-0.002),
                units='norm')
                                                    
        
        
        
        
        self.frame_elements['theBox'] = visual.ShapeStim(win=self.win,
                                                    vertices=self.theBox,
                                                    closeShape=True,
                                                    pos=(0,0),
                                                    size=1.5,
                                                    lineColor='white')
        
        
        self.frame_elements['trial_number'] = visual.TextStim(
                                                    win=self.win,
                                                    text=f'Trial {self.trial_idx}/{self.n_trials}',
                                                    pos=(0.9,-0.9),
                                                    units='norm',
                                                    color='white',
                                                    height=0.05)
        
        
        self.frame_elements['cursor'] = visual.Circle(
                                                    win=self.win,
                                                    units='pix',
                                                    size=(30,30),
                                                    pos=convertToPix(
                                                        self.cursorStartpoint*1.5,
                                                        (0,0),
                                                        'norm',
                                                        self.win),
                                                    color='red',
                                                    fillColor='red',
                                                    lineColor='red'
                                                    )
        
        #trace
        self.trace_vertices = [convertToPix(self.cursorStartpoint*1.5,(0,0),
                                            'norm',
                                            self.win)] # could also take this from the cursor
        self.frame_elements['trace'] = visual.ShapeStim(win=self.win,
                                          units='pix',
                                          vertices=self.trace_vertices,
                                          #colorSpace = 'rgb255',
                                          lineColor='red',
                                          lineWidth=5,
                                          interpolate=True,
                                          closeShape=False)
        
        
        self.frame_elements['instructions'] = visual.ImageStim(
            win=self.win,
            image=self.instructions_path,
            pos=(0,0.95))
        
        
        #startpoint
        startpoint_size = 0.05
        self.frame_elements['startpoint'] = visual.Rect(win=self.win,
                                 pos = (-0.8,0.7),
                                 size=(startpoint_size,
                                       startpoint_size*self.aspect_ratio),
                                 fillColor='Black',
                                 lineColor='Cyan')
        
        
        self.frame_elements['timebar'] = visual.Rect(win=self.win,
                                                      pos = (0,-0.85),
                                                      size=(1,0.025),
                                                      fillColor='gray')
        
        
        
    #this is too long
    def exec_trial(self,stimuli_idx, scale=True): #should stimuli_idx just be called trial_idx?
        #display a single copydraw task
        
        print('executing trial')

        #define duration as how long it runs for max, trialTime as actual
        #runtime (finish when raised etc)
        self.trialDuration = self.n_letters * self.letterTime

        #initialise the frame
        self.create_frame(self.order[stimuli_idx],scale=scale)
    
        #getting a monitor gamut error when doing rgb colours, look into it
        #self.trace.colorSpace = 'rgb255'
        
        #draw first frame
        self.draw_and_flip(exclude=['timebar','trace'])
         
        
        #timebar
        timebar_x = self.frame_elements['timebar'].size[0]
        
        #mouse
        self.mouse = event.Mouse()
        
        #main bit
        main_loop = True
        started_drawing = False
        while main_loop == True:
            
            #only show instructions if startpoint not clicked
            if self.frame_elements['startpoint'].fillColor != 'Cyan': 
                self.draw_and_flip(exclude=['trace'])
            else:
                self.draw_and_flip(exclude=['trace','instructions'])
            
            #click in startpoint
            if self.mouse.isPressedIn(self.frame_elements['startpoint']): 
                self.frame_elements['startpoint'].fillColor = 'Cyan'
                tic = clock.getTime()
                self.draw_and_flip(exclude=['trace','instructions'])
        
                
            if self.frame_elements['startpoint'].fillColor == 'Cyan':


                # drawing has begun
                if self.mouse.isPressedIn(self.frame_elements['cursor']):
                    
                    started_drawing = True                
                    
                    trial_timer = clock.CountdownTimer(self.trialDuration) 
                    
                    #save start time
                    self.startTStamp = clock.getTime()         
                    
                    #calc pre trial time
                    self.ptt = self.startTStamp - tic

                # shrink time bar, draw trace, once drawing has started
                if started_drawing == True:
                    
                    # for recording times associated with cursor pos
                    self.cursor_t = [self.startTStamp]
                    
                    
                    c = 0
                    #decreasing timebar
                    while trial_timer.getTime() > 0:
                        
                        #get remaining time
                        t_remain = trial_timer.getTime()
                        ratio = t_remain/self.trialDuration
                        
                        #adjust timebar size and position
                        new_size = [timebar_x*ratio, # change the x value
                                    self.frame_elements['timebar'].size[1]]
                        self.frame_elements['timebar'].setSize(new_size)
                        
                        new_pos = [(-timebar_x*ratio/2) + timebar_x/2, #shift
                                   self.frame_elements['timebar'].pos[1]]
                        self.frame_elements['timebar'].setPos(new_pos)
                        
                            
                        #while mouse is pressed in, update cursor position
                        if self.mouse.getPressed()[0]:
                            
                            #get new position from mouse
                            new_pos = convertToPix(self.mouse.getPos(),
                                                           (0,0),
                                                           units=self.mouse.units,
                                                           win=self.win)
                            
                            #record time at which that happened
                            self.cursor_t.append(trial_timer.getTime()) # could maybe use t_remain here
                            
                            #move cursor to that position and save for drawing trace
                            self.frame_elements['cursor'].pos = new_pos
                            #self.cursor_idx.append(i)
                            self.trace_vertices.append(new_pos)
                            self.frame_elements['trace'].vertices = self.trace_vertices
          
                        ### ISSUE currently cant do non continuous lines    
                        if c%2 == 0: #only draw every xth frame, increases recording rate (erm this doesnt seem to work, opposite effect seems to happen?!)
                            self.draw_and_flip(exclude=['instructions'])
                        c += 1
                        
                        if (not self.mouse.getPressed()[0] and
                            self.finishWhenRaised and started_drawing):
                            print('mouse raised - ending')
                            main_loop = False
                            break
                    
                    #timebar elapsed
                    print('breaking out of main')
                    
                    main_loop = False
       
        
       #trial time is how long they were drawing for, ie time elapsed during drawing
        self.trialTime = self.trialDuration - self.cursor_t[-1] 
        print(f'recorded {len(self.trace_vertices)} points at a rate of {len(self.trace_vertices)/self.trialTime} points per sec')
        ##### should there be any unit conversions done?
        ### maybe put all this stuff below in a func of its own
        ### separate the main loop and data stuff
        traceLet = self.frame_elements['trace'].vertices.copy()
        
        
        ## postprocessing funcs now in separate file
        ## funcs for computing dtw etc on recorded data not yet written
        #scoring
        self.trial_results = {} # self.computeScoreSingleTrial(traceLet, self.current_stimulus, self.trialTime)
        
        #trace & time
        self.trial_results['traceLet'] = traceLet
        self.trial_results['trialTime'] = self.trialTime
        
        #add metadata
        self.trial_results['ix_block'] = self.block_idx
        self.trial_results['ix_trial'] = self.trial_idx
        self.trial_results['ptt'] = self.ptt
        self.trial_results['startTStamp'] = self.startTStamp
        
        #new/extra metadata
        if scale:
            self.trial_results['scaling_matrix'] = self.scaling_matrix
        
        self.trial_results['trialDuration'] = self.trialDuration
        self.trial_results['flip'] = self.flip
        self.trial_results['theBox'] = self.frame_elements['theBox'].vertices.copy()
        

        self.trial_results['theBoxPix'] = self.frame_elements['theBox'].verticesPix
        
        
        traceLetPix = self.frame_elements['trace'].verticesPix.copy()
        if (traceLet != traceLetPix).any():
            self.trial_results['pos_t_pix'] = traceLetPix
        
        #in matlab i think this is theRect
        self.trial_results['winSize'] = self.win.size
        
        self.trial_results['templatePix'] = self.frame_elements['template'].verticesPix
        #do i need to add theWord? yes.
        
        
        
    def exit(self):
        self.finish_block()
        #core.quit()
        
    
    

        
    # #can maybe remove this
    # def store_trial(self): #what will the purpose of this be?!
    #     #add to dict or sth save all trials as block?
        
    #     #to get these keys need kin_scores (twice, once subsampled, dtw and metadata? )
    #     keys = ['dt','dt_l','w','pathlen','len','dt_norm','speed','speed_sub', 
    #     'velocity_x','velocity_x_sub','velocity_y','velocity_y_sub','isj',
    #     'isj_sub', 'isj_x', 'isj_x_sub', 'isj_y', 'isj_y_sub', 'acceleration', 
    #     'acceleration_sub', 'acceleration_x', 'acceleration_x_sub', 'acceleration_y', 
    #     'acceleration_y_sub', 'pos_t', 'pos_t_sub', 'speed_t','speed_t_sub', 
    #     'accel_t','accel_t_sub','jerk_t', 'jerk_t_sub', 'dist_t', 'ptt', 
    #     'ix_block', 'ix_trial', 'startTStamp', 'stim']
        
    #     for key in keys:
    #         #put this into try except
    #         assert key in self.trial_results.keys()
    #         #print warning - dont save?
            
    #     #then do saving/storing
        
    #     #del trial_results after? or reset to empty dict?
        
    
    

    
    # move this outta here
    def check_results(self,sample_data,template,trial_time=2.7): #might also be 2.2 check both?
        trial_results = {}
        
        pos_t = sample_data['pos_t'].copy().astype(float)
        
        
        delta_t = trial_time/len(pos_t)
        
        ##### Kinematic scores #####
        kin_scores = self.kin_scores(pos_t,delta_t)
        trial_results = {**trial_results, **kin_scores }
        
        ## sub sample ##
        mouse_pos_pix_sub = self.movingmean(pos_t,5)
        mouse_pos_pix_sub = mouse_pos_pix_sub[::3,:] # take every third point
        kin_scores_sub = self.kin_scores(mouse_pos_pix_sub,delta_t*3,sub_sampled=True)
        trial_results = {**trial_results, **kin_scores_sub}
        
        
        ##### dtw #####
        dtw_res = self.dtw_features(pos_t, template)
        trial_results = {**trial_results, **dtw_res}
        
        
        ##### misc ##### 
        # +1 on the pathlens bc matlab indexing
        trial_results['dist_t'] = np.sqrt(np.sum((template[trial_results['w'].astype(int)[:trial_results['pathlen']+1,0],:] - pos_t[trial_results['w'].astype(int)[:trial_results['pathlen']+1,1]])**2,axis=1))
        
        # normalize distance dt by length of copied template (in samples)
        trial_results['dt_norm'] = trial_results['dt_l'] / (trial_results['pathlen']+1)
        
        # get length of copied part of the template (in samples)
        trial_results['len'] = (trial_results['pathlen']+1) / len(template)
        
        
        ### no way to calculate these ###
        #its the time between touching the cyan square and starting drawing (i think)
        trial_results['ptt'] = sample_data['ptt']
        trial_results['ix_block'] = sample_data['ix_block']
        trial_results['ix_trial'] = sample_data['ix_trial']
        trial_results['startTStamp'] = sample_data['startTStamp']
        trial_results['stim'] = sample_data['stim']
        
        
        
        ### weird stuff and index changes ###
        #trial_results['pos_t'] = trial_results['pos_t'].astype('<u2')
        trial_results['pathlen'] += 1
        trial_results['w'] += 1
        #process delta t error
        trial_results['acceleration'] *= delta_t
        trial_results['acceleration_x'] *= delta_t
        trial_results['acceleration_y'] *= delta_t
        trial_results['acceleration_sub'] *= delta_t*3
        trial_results['acceleration_x_sub'] *= delta_t*3
        trial_results['acceleration_y_sub'] *= delta_t*3
        
        def check_with_tol(x,y,tol=0.0001):
            return np.abs(x-y)<tol
        
        print('calculated results - beginning checks')
        for key,data in trial_results.items():
            try:
                if isinstance(data,np.ndarray):
                    check = check_with_tol(data, sample_data[key]).all()
                    
                    if not check:
                        print(f'{key} FAILED')
                else:
                    
                    check = check_with_tol(data, sample_data[key])

                    if not check:
                        print(f'{key} FAILED: {data} should be {sample_data[key]}')
            except:
                print(f'encountered error with {key}')
                
        return trial_results
    
    
if __name__ == "__main__":
    try:
        test = CopyDraw('./',
                        )
        
        
        test.init_session('TEST_SESSION')
        
        test.init_block(block_name='TEST_BLOCK',
                        n_trials=2,
                        letterTime=2.7,                        
                        finishWhenRaised=True,
                        manyshape=False,
                        )
        test.exec_block()
        test.exit()
        
    #this still isnt printing the error out, why?
    except Exception as e:
        core.quit()
        print(e)