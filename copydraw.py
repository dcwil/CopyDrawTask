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

from base import AbstractParadigm
from psychopy import visual, core, event, clock
from psychopy.tools.monitorunittools import convertToPix,posToPix
from numpy.linalg import norm

from pathlib import Path
from itertools import permutations

from dtw import *
from dtw_matlab import dtw_matlab

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
                                                  shape_params['win']) for vertex in vertices])

    def draw(self):
        for shape in self.shapes:
            shape.draw()
            
    

class CopyDraw(AbstractParadigm):
    
    #should there be an n_blocks arg?
    def __init__(self,
                 session_name,
                 data_dir,
                 n_trials=12,
                 trialTime=2.2,
                 image_size=2,
                 n_letters=3,
                 finishWhenRaised=True,
                 screen_size = (1920, 1200), #16:10 ratio
                 screen_ix = 0,
                 flip=True,
                 lpt_address = None,
                 serial_nr = None,
                 manyshape=True, #new or old rendering
                 interp = 'smooth', #{'s':0.001,'n_interp':300}, dict of values or None or 'smooth'
                 old_template_path='../CopyDraw_mat_repo/CopyDrawTask-master/templates/shapes'):
        super().__init__(screen_ix=screen_ix, lpt_address=lpt_address,
                         serial_nr=serial_nr)
        
        self.session_name = time.asctime( time.localtime(time.time()) ).replace(':','-') if session_name == None else session_name #saving name
        self.data_dir = Path(data_dir) #templates, instructions and saves stored here
        self.n_trials = n_trials
        self.trialTime = trialTime
        self.image_size = image_size
        self.n_letters = n_letters
        self.finishWhenRaised = finishWhenRaised 
        self.screen_size = screen_size
        self.trial_clock = core.Clock() # use this!
        self.flip=flip
        self.old_template_path = old_template_path
        self.manyshape = manyshape
        self.interp = interp
        self.block_idx = 0 # this gets +1'd every time exec block is called, need a setter too?
        print('initialised')
        
        
    def init_block(self,block_name=None):
        super().init_block(self.screen_size)
        self.win.color = (-1,-1,-1)
        self.block_name =f'BLOCK_{time.asctime( time.localtime(time.time()) ).replace(":","-")}' if block_name == None else block_name
        self.aspect_ratio = self.win.size[0]/self.win.size[1]
        self.msperframe,_,_ = self.win.getMsPerFrame()   #this is redundant now?
        
        #dropped frames
        self.win.recordFrameIntervals = True
 
        self.load_stimuli(self.data_dir / "templates")
   
        
        ### move object creation out of here? ###
        #instructions
        self.load_instructions(self.data_dir / 'instructions' / 'instructions.png')

        
        #folder for saving
        self.results_dir = Path(self.data_dir,"results",self.session_name,self.block_name)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        
        
        #trial index
        self.trial_idx = 1
        
        #create template order
        self.create_template_order(shuffle=True)
        
        
        #log - use this!
        fh = logging.FileHandler(self.results_dir / 'debug.log')
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
        assert old_stimuli_path.exists(), f"old Stimuli data not found: {old_stimuli_path}"
        
        folder = self.stimuli_fname[:-5]+'.5' if self.stimuli_fname[:-4].endswith('5') else self.stimuli_fname[:-4]
        
        idx_to_fname = {idx:''.join(p) + '.mat' for idx,p in enumerate(permutations(['1','2'] if 'short' in folder else ['1','2','3']))}
        
        self.old_stimuli_path = old_stimuli_path / folder / idx_to_fname[stimuli_idx]     
        self.old_stimuli = scipy.io.loadmat(self.old_stimuli_path,simplify_cells=True)
        
        return self.old_stimuli[template_or_box]
    
    
    def get_box(self,idx):
        box = self.get_old_stimuli(idx)
        self.theBox = box['boxPos'].T#box[0,0][0].T.astype(float)
        return self.theBox
        
    
    def load_stimuli(self, path, short=True, size=35): #swap short arg for n_letters
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
                
            
        
        
        
        
    def get_stimuli(self, stimuli_idx,scale=True): #how will n_trials > len(templates) work? stratify and randomise
        #easier to ask forgiveness than permission!
        try:
            self.current_stimulus = self.templates[stimuli_idx].astype(float)
            self.get_box(stimuli_idx)

            if scale:
                
                self.current_stimulus,self.scaling_matrix = utils.scale_to_norm_units(self.current_stimulus)
                self.theBox,_ = utils.scale_to_norm_units(self.theBox,scaling_matrix=self.scaling_matrix)
                
                #reorder
                newboxarray = np.zeros([4,2])
                newboxarray[0:2] = self.theBox[0:2]
                newboxarray[2],newboxarray[3] = self.theBox[3],self.theBox[2]
                self.theBox = newboxarray

                
                print('scaled')
                if self.flip:
                    self.current_stimulus = np.matmul(self.current_stimulus, np.array([[1,0],[0,-1]]))
                    self.theBox = np.matmul(self.theBox, np.array([[1,0],[0,-1]]))
                    
            
            if self.interp == 'smooth':        
                self.current_stimulus = utils.smooth(self.current_stimulus,return_df=False)
            
            elif self.interp is not None:
                tck, u = splprep(self.current_stimulus.T, s=self.interp['s'])
                
                
                ### using a high value for n_interp along with manyshape == True
                ### results in laggy performance
                
                u_new = np.linspace(np.min(u), np.max(u), self.interp['n_interp'])
                self.current_stimulus = np.array(splev(u_new, tck)).T
                # note this is replacing the original points, not adding
                
                
                
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
            self.trial_idx += 1
            #run some kind of check_trial func here?
            self.block_results[stimuli_idx] = self.trial_results
        self.block_idx += 1
        
        
    def save_block(self):
        #rudimentary atm, can ble cleaned up, flattened a little maybe
        self.df = pd.DataFrame(self.block_results)
        
        #csv had formatting issues
        self.df.to_pickle(self.results_dir / f'scores_copyDraw_block{self.block_idx}.pkl')


    #draw order is based on .draw() call order, consider using an ordered dict?
    def draw_and_flip(self,exclude=[]):
        for element_name,element in self.frame_elements.items():
            if element_name in exclude:
                continue
            else:
                element.draw()
        self.win.flip()
    
    def create_frame(self,stimuli_idx,scale=True):
        
        self.frame_elements = {}
        
        if self.manyshape:
            manyshape_size = 0.035 # equates to line thickness
            self.frame_elements['template'] = ManyShapeStim(self.get_stimuli(stimuli_idx,scale=scale),
                                                visual.Circle,
                                                {'win':self.win,
                                                'units':'norm',
                                                'size':(manyshape_size,manyshape_size*self.aspect_ratio),
                                                'color':'blue',
                                                'fillColor':'blue',
                                                'lineColor':'blue'
                                                },
                                                size=1.5)
        else:
            self.frame_elements['template'] = visual.ShapeStim(win=self.win,
                                                    vertices=self.get_stimuli(stimuli_idx,scale=scale),
                                                    lineWidth=100, # seems to top out after like 10 or 20: https://github.com/psychopy/psychopy/issues/818
                                                    closeShape=False,
                                                    interpolate=True,
                                                    ori=0,
                                                    pos=(0,0),
                                                    size=1.5,
                                                    units='norm',
                                                    fillColor='blue',
                                                    lineColor='blue',
                                                    #windingRule=True,
                                                    )
            
            
            #this would replace template from above ^ 
            #if you want to run without the original template in, comment out and change the name below to just "template"
            self.frame_elements['template_new'] = visual.ImageStim(win=self.win,
                                                                   #currently this is rewriting the image each time - need one that opens the image if its there before remaking it
                                                               image=template_to_image(self.get_stimuli(stimuli_idx,scale=scale),
                                                                                       f'{self.stimuli_fname[:-4]}_{stimuli_idx}',
                                                                                       'template_images',
                                                                                       lineWidth=4),
                                                               
                                                               #this is from visual inspection - still not quite right
                                                               size=1.69,
                                                               interpolate=True,
                                                               pos=(-0.001,-0.002),
                                                               units='norm')
                                                    
        
        
        
        
        self.frame_elements['theBox'] = visual.ShapeStim(win=self.win,
                                                    vertices = self.theBox,
                                                    closeShape=True,
                                                    pos=(0,0),
                                                    size=1.5,
                                                    lineColor='white')
        
        
        self.frame_elements['trial_number'] = visual.TextStim(
                                                        win=self.win,
                                                        text = f'Trial {self.trial_idx}/{self.n_trials}',
                                                        pos = (0.9,-0.9),
                                                        units='norm',
                                                        color='white',
                                                        height=0.05)
        
        
        self.frame_elements['cursor'] = visual.Circle(
                                                    win=self.win,
                                                    units='pix',
                                                    size=(30,30),
                                                    pos=self.frame_elements['template'].verticesPix[0],
                                                    color='red',
                                                    fillColor='red',
                                                    lineColor='red'
                                                    )
        
        #trace
        self.trace_vertices = [self.frame_elements['template'].verticesPix[0]] # could also take this from the cursor
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
                                 size=(startpoint_size,startpoint_size*self.aspect_ratio),
                                 fillColor='Black',
                                 lineColor='Cyan')
        
        
        self.frame_elements['timebar'] = visual.Rect(win=self.win,
                                                      pos = (0,-0.85),
                                                      size=(1,0.025),
                                                      fillColor='gray')
        
        
        
        
    def exec_trial(self,stimuli_idx, scale=True): #should stimuli_idx just be called trial_idx?
        #display a single copydraw task
        
        print('executing trial')

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
                    
                    trial_timer = clock.CountdownTimer(self.trialTime) 
                    
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
                        ratio = t_remain/self.trialTime
                        
                        #adjust timebar size and position
                        self.frame_elements['timebar'].setSize([timebar_x*ratio,self.frame_elements['timebar'].size[1]])
                        self.frame_elements['timebar'].setPos([(-timebar_x*ratio/2) + timebar_x/2,self.frame_elements['timebar'].pos[1]])
                        
                            
                        #while mouse is pressed in, update cursor position
                        if self.mouse.getPressed()[0]:
                            
                            #get new position from mouse
                            new_pos = convertToPix(self.mouse.getPos(),
                                                           (0,0),
                                                           units=self.mouse.units,
                                                           win=self.win)
                            
                            #record time at which that happened
                            self.cursor_t.append(clock.getTime())
                            
                            #move cursor to that position and save for drawing trace
                            self.frame_elements['cursor'].pos = new_pos
                            #self.cursor_idx.append(i)
                            self.trace_vertices.append(new_pos)
                            self.frame_elements['trace'].vertices = self.trace_vertices
          
                        ### ISSUE currently cant do non continuous lines    
                        if c%2 == 0: #only draw every other frame, increases sf
                            self.draw_and_flip(exclude=['instructions'])
                        c += 1
                        if not self.mouse.getPressed()[0] and self.finishWhenRaised and started_drawing:
                            print('mouse raised - ending')
                            main_loop = False
                            break
                    
                    #timebar elapsed
                    print('breaking out of main')
                    print(f'recorded {len(self.trace_vertices)} points at a rate of {len(self.trace_vertices)/self.trialTime} points per sec')
                    main_loop = False
                    
            
        ##### should there be any unit conversions done?
        
        traceLet = self.frame_elements['trace'].vertices.copy()
        
        
        ## in matlab the traces and templates are stored and scoring is computed at the end of the block
        ## should this follow the same pipeline (would reduce delays between presenting trials)
        #scoring
        self.trial_results = self.computeScoreSingleTrial(traceLet, self.current_stimulus, self.trialTime)
        
        
        #add metadata
        self.trial_results['ix_block'] = self.block_idx
        self.trial_results['ix_trial'] = self.trial_idx
        self.trial_results['ptt'] = self.ptt
        self.trial_results['startTStamp'] = self.startTStamp
        
        #new/extra metadata
        if scale:
            self.trial_results['scaling_matrix'] = self.scaling_matrix
        
        self.trial_results['trialTime'] = self.trialTime
        self.trial_results['flip'] = self.flip
        self.trial_results['theBox'] = self.frame_elements['theBox'].vertices.copy()
        

        self.trial_results['theBoxPix'] = self.frame_elements['theBox'].verticesPix
        
        
        traceLetPix = self.frame_elements['trace'].verticesPix.copy()
        if (traceLet != traceLetPix).any():
            self.trial_results['pos_t_pix'] = traceLetPix
        
        #in matlab i think this is theRect
        self.trial_results['winSize'] = self.win.size
        
        self.trial_results['templatePix'] = self.frame_elements['template'].verticesPix
        #do i need to add theWord?
        
    def exit(self):
        self.finish_block()
        #core.quit()
        
    def computeScoreSingleTrial(self,traceLet,template,trialTime):
        
        trial_results = {}
        
        #compute avg delta_t
        delta_t = trialTime/traceLet.shape[0]
        
        ##### Kinematic scores #####
        kin_scores = self.kin_scores(traceLet,delta_t)
        trial_results = {**trial_results, **kin_scores }
        
        ## sub sample ##
        traceLet_sub = self.movingmean(traceLet,5)
        traceLet_sub = traceLet_sub[::3,:] # take every third point
        kin_scores_sub = self.kin_scores(traceLet_sub,(delta_t)*3,sub_sampled=True)
        trial_results = {**trial_results, **kin_scores_sub}
        
        ##### dtw #####
        print(f'template has size: {self.frame_elements["template"].verticesPix.shape}')
        print(f'stim has shape: {self.current_stimulus.shape}')
        print(f'trace has shape: {traceLet.shape}')
        
        # think about units here bound to run into issues!
        dtw_res = self.dtw_features(traceLet, template)
        trial_results = {**trial_results, **dtw_res}
        
        
        #misc
        # +1 on the pathlens bc matlab indexing
        trial_results['dist_t'] = np.sqrt(np.sum((template[trial_results['w'].astype(int)[:trial_results['pathlen']+1,0],:] - trial_results['pos_t'][trial_results['w'].astype(int)[:trial_results['pathlen']+1,1]])**2,axis=1))
        
        # normalize distance dt by length of copied template (in samples)
        trial_results['dt_norm'] = trial_results['dt_l'] / (trial_results['pathlen']+1)
        
        # get length of copied part of the template (in samples)
        trial_results['len'] = (trial_results['pathlen']+1) / len(template)
        
        
        return trial_results
        
        
        

    #should these static methods bein utils instead?
    @staticmethod
    def deriv_and_norm(var,delta_t):
        """
        Given an array (var) and timestep (delta_t), computes the derivative 
        for each timepoint and returns it (along with the magnitudes)
        
        """
        ### This is not the same as the kinematic scores in the matlab code!
        deriv_var = np.diff(var,axis=0)/delta_t
        deriv_var_norm = norm(deriv_var,axis=1)
        return deriv_var,deriv_var_norm     
    
    
    @staticmethod # again should this be in a separate file? ie utils
    def movingmean(arr,w_size):
        # trying to mimic some of the functionality from:
        # https://uk.mathworks.com/matlabcentral/fileexchange/41859-moving-average-function
        # which i think is the function used in compute_scoreSingleTrial.m (not in matlab by default)
        
        #returns an array of the same size by shrinking the window for the start and end points
        
        #round even window sizes down
        if w_size%2 == 0:
            w_size -= 1
        
        w_tail = np.floor(w_size/2)
        
        arr_sub = np.zeros_like(arr)
        
        for j,col in enumerate(arr.T): # easier to work with columns like this
            for i,val in enumerate(col):
                
                #truncate window if needed
                start = i - w_tail if i > w_tail else 0
                stop = i + w_tail + 1 if i + w_tail < len(col) else len(col)
                s = slice(int(start),int(stop))
                
                #idxs reversed bc .T
                arr_sub[i,j] = np.mean(col[s])
                
                #could probably find a way to do this both cols at the same time
        
        return arr_sub
    
    
    def kin_scores(self, var_pos, delta_t,sub_sampled=False): 
        
        kin_res = {}
        
        kin_res['pos_t'] = var_pos
        
        velocity, velocity_mag = self.deriv_and_norm(var_pos, delta_t)
        accel, accel_mag = self.deriv_and_norm(velocity, delta_t)
        jerk, jerk_mag = self.deriv_and_norm(accel, delta_t)
        
        N = len(var_pos)
        
        #average
        ## divided by number of timepoints, because delta t was used to calc instead of total t
        kin_res['speed'] = np.sum(velocity_mag) / N
        kin_res['acceleration'] = np.sum(accel_mag) / N
        kin_res['velocity_x'] = np.sum(np.abs(velocity[:,0])) / N
        kin_res['velocity_y'] = np.sum(np.abs(velocity[:,1])) / N
        kin_res['acceleration_x'] = np.sum(np.abs(accel[:,0])) / N
        kin_res['acceleration_y'] = np.sum(np.abs(accel[:,1])) / N #matlab code does not compute y values, incorrect indexing
        
        #isj
        # in matlab this variable is overwritten
        isj_ = np.sum((jerk * (delta_t)**3)**2,axis=0)
        kin_res['isj_x'],kin_res['isj_y'] = isj_[0],isj_[1]
        kin_res['isj'] = np.mean(isj_)
        
        kin_res['speed_t'] = velocity * (delta_t)
        kin_res['accel_t']= accel * (delta_t)**2
        kin_res['jerk_t'] = jerk * (delta_t)**3
        
        if sub_sampled:
            kin_res = {f'{k}_sub':v for k,v in kin_res.items()}
        
        return kin_res
        
    #can maybe remove this
    def store_trial(self): #what will the purpose of this be?!
        #add to dict or sth save all trials as block?
        
        #to get these keys need kin_scores (twice, once subsampled, dtw and metadata? )
        keys = ['dt','dt_l','w','pathlen','len','dt_norm','speed','speed_sub', 
        'velocity_x','velocity_x_sub','velocity_y','velocity_y_sub','isj',
        'isj_sub', 'isj_x', 'isj_x_sub', 'isj_y', 'isj_y_sub', 'acceleration', 
        'acceleration_sub', 'acceleration_x', 'acceleration_x_sub', 'acceleration_y', 
        'acceleration_y_sub', 'pos_t', 'pos_t_sub', 'speed_t','speed_t_sub', 
        'accel_t','accel_t_sub','jerk_t', 'jerk_t_sub', 'dist_t', 'ptt', 
        'ix_block', 'ix_trial', 'startTStamp', 'stim']
        
        for key in keys:
            #put this into try except
            assert key in self.trial_results.keys()
            #print warning - dont save?
            
        #then do saving/storing
        
        #del trial_results after? or reset to empty dict?
        
    
    
    @staticmethod # again, should this be in a separate file?
    def dtw_features(trace,template,step_pattern='MATLAB'):
        res = {}
        if step_pattern != 'MATLAB':
            alignment = dtw(trace,template,step_pattern=step_pattern,keep_internals=True)
            
            idx_min = np.argmin(alignment.localCostMatrix[-1,1:]) #called pathlen briefly in .m files
            
            res['w'] = np.stack([alignment.index1,alignment.index2],axis=1)
            res['pathlen'] = min([idx_min,template.shape[0]]) #bug in matlab code here, chooses wrong template axis
            res['dt'] = alignment.distance
            res['dt_l'] = alignment.costMatrix[-1,idx_min]
        else:
            res['dt'],res['dt_l'],res['w'],pathlen = dtw_matlab(trace,template)
            res['pathlen'] = min([pathlen,template.shape[0]]) #bug in matlab code here, chooses wrong template axis
            
        return res
    
    #@staticmethod    
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
        test = CopyDraw('TEST_SESSION','./',
                        n_trials=2,
                        finishWhenRaised=True,
                        manyshape=False,
                        trialTime=2.7)
        
        test.init_block(block_name='TEST_BLOCK')
        test.exec_block()
        test.save_block()
        test.exit()
        
    #this still isnt printing the error out, why?
    except Exception as e:
        core.quit()
        print(e)