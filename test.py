# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:14:23 2020

@author: Daniel
"""

#### is there anything useful left in here? ####




# from psychopy import visual, core
# win = visual.Window([400,400])
# message = visual.TextStim(win, text='hello')
# message.autoDraw = True  # Automatically draw every frame
# win.flip()
# core.wait(20.0)
# message.text = 'world'  # Change properties of existing stim
# win.flip()
# core.wait(2.0)
import scipy.io
import math
import numpy as np
from psychopy import visual, core, event, monitors# import some libraries from PsychoPy
from pathlib import Path
from psychopy.tools.monitorunittools import posToPix,convertToPix,pix2cm
from numpy.linalg import norm
from dtw import dtw

mon = monitors.Monitor('LG')

#create a window
mywin = visual.Window([1280,720], monitor=mon, units="norm")
#print(mywin.monitorFramePeriod)
aspect_ratio = mywin.size[0]/mywin.size[1]
#create some stimuli
# grating = visual.GratingStim(win=mywin, mask="circle", size=3, pos=[-4,0], sf=3)

imfile = scipy.io.loadmat('templates/Size_35.mat')
#imfile['templateLet'][:,0]
six_templates = imfile['new_shapes'][0]
template_choice = 2

instruct_path = Path('instructions','instructions.png')



#flip
flip = True


template = six_templates[template_choice].astype(float)

#scale between 0 and 1
template[:,0] -= np.min(template[:,0])
template[:,1] -= np.min(template[:,1])

template[:,0] /= np.max(template[:,0])
template[:,1] /= np.max(template[:,1])

#print(template)
fixation = visual.ShapeStim(win=mywin,
                            vertices=template,
                            lineWidth=30,
                            closeShape=False,
                            interpolate=False,
                            ori=180 if flip else 0,
                            pos=(0.75,0.75),
                            size=1.5,
                            units='norm'
                            )

instructions = visual.ImageStim(
    win=mywin,
    image=instruct_path,
    pos=(0,0.85))

startpoint_size = 0.05
startpoint = visual.Rect(win=mywin,
                         pos = (-0.8,0.7),
                         size=(startpoint_size,startpoint_size*aspect_ratio),
                         fillColor='Green')


timebar = visual.Rect(win=mywin,
                      pos = (0,-0.85),
                      size=(0.6,0.1),
                      fillColor='Blue')


mouse = event.Mouse()

trial_dur = 12 #seconds

#fixation = visual.GratingStim(win=mywin, size=0.5, pos=[0,0], sf=0, rgb=-1)

#draw the stimuli and update the window
# grating.draw()
msperframe,_,_ = mywin.getMsPerFrame()

instructions.draw()
fixation.draw()
startpoint.draw()
#timebar.draw()

mywin.update()


trial_dur_frames = math.ceil(trial_dur / (msperframe/10**3))
timebar_x = timebar.size[0]
timebar_x_vals = np.linspace(timebar_x,0,num=trial_dur_frames)
timebar_shift_perframe = timebar_x/trial_dur_frames
while True:

    if mouse.isPressedIn(startpoint): #click in startpoint

        while startpoint.contains(mouse.getPos()): #start when mouse leaves startpoint
            instructions.draw()
            fixation.draw()
            startpoint.draw()
            mywin.update()
            
        #timebar decreasing& recording mouse pos
        mouse_pos = []
        #mouse_pos_pix = []
        for frame_n in range(trial_dur_frames):
            timebar.setSize([timebar_x_vals[frame_n],timebar.size[1]])
            timebar.setPos([(timebar_x_vals[frame_n]/2) - timebar_x/2,timebar.pos[1]])
            
            mouse_pos.append(mouse.getPos())
            #mouse_pos_pix.append(posToPix(mouse))
            timebar.draw()
            instructions.draw()
            fixation.draw()
            startpoint.draw()
            mywin.flip()
    
        break        




#convert mouse pos into pix
mouse_pos = np.array(mouse_pos)
mouse_pos_pix = convertToPix(mouse_pos,win=mywin,units='norm',pos=(0,0))
mouse_pos_cm = pix2cm(mouse_pos_pix, mon)

# #features
# def deriv_and_norm(var,delta_t):
#     deriv_var = np.diff(var,axis=0)/delta_t
#     deriv_var_norm = norm(deriv_var,axis=1)
#     return deriv_var,deriv_var_norm

# velocity, velocity_mag = deriv_and_norm(mouse_pos_cm, (msperframe/10**3))
# acceleration, acceleration_mag = deriv_and_norm(velocity, (msperframe/10**3))
# jerk, jerk_mag = deriv_and_norm(acceleration, (msperframe/10**3))

# #dtw
# reference = fixation.verticesPix
# query = mouse_pos_pix

# cost_matrix = np.zeros([len(query),len(reference)])
# for i in range(cost_matrix.shape[0]):
#     for j in range(cost_matrix.shape[1]):
#         diff = query[i] - reference[j]
#         cost_matrix[i,j] = norm(diff)
        

# alignment = dtw(cost_matrix,keep_internals=True)

mywin.close()
core.quit()
