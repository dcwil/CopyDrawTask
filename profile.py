# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:37:46 2021

@author: Daniel
"""

import cProfile
import pstats
fname = 'profile.prof'

pr = cProfile.Profile()

pr.enable()

from copydraw import CopyDraw 


test = CopyDraw('./',
                )


test.init_session('TEST_SESSION')

test.init_block(block_name='TEST_BLOCK',
                n_trials=3,
                letterTime=2.2,                        
                finishWhenRaised=True,
                manyshape=False,)
test.exec_block()

test.exit()

pr.disable()
pr.dump_stats(fname)

# p = pstats.Stats('profile.prof')
# p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)