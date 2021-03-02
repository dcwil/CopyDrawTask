# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:00:46 2020

@author: Daniel
"""

# code taken from https://github.com/bsdlab/MDDInfer

import abc
#import pdb


from psychopy import parallel, visual
#from ..utils import VPPort
#from .. import logger

class AbstractParadigm(abc.ABC):
    def __init__(self, screen_ix = 0, lpt_address=None, serial_nr = None):

        if lpt_address:
            self.pport = parallel.ParallelPort(address=lpt_address)
        else:
            self.pport = None
        if serial_nr:
            print('VPPort stuff goes here')
            #self.vpp = VPPort(serial_nr)
        else:
            self.vpp = None
        self.screen_ix = screen_ix

    def get_fixation(self):
        return visual.Circle(self.win, radius=5, color='red', interpolate=True, units='pix')

    def send_marker(self, val):
        if type(val) == int and val < 256:
            if self.vpp:
                self.vpp.write([val])
            if self.pport:
                self.pport.setData(val)
        else:
            raise ValueError("Please provide an integer value < 256 to be written as a marker")
        #logger.info('marker-%d' % val)

    # @abc.abstractmethod
    # def exec_subtrial(self):
    #     pass

    @abc.abstractmethod
    def exec_trial(self):
        pass

    # def init_block(self):
    #     self.win = visual.Window(fullscr=True, screen=self.screen_ix,units='norm')
    #     self.fixation = self.get_fixation()
    def init_block(self,size=None): #this is different from the original base class
        self.win = visual.Window(screen=self.screen_ix,
                                 units='norm',
                                 fullscr=True if size==None else False,
                                 size=size if size!=None else (800,600)) # default is 800,600

    def finish_block(self):
        self.win.close()

    @abc.abstractmethod
    def exec_block(self, n_trials):
        pass