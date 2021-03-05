# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:37:46 2021

@author: Daniel
"""

import cProfile
import pstats
import io
from copydraw import CopyDraw
from pstats import SortKey
from functools import wraps


# https://towardsdatascience.com/how-to-profile-your-code-in-python-e70c834fad89
def profile(output_file=None):
    def inner(func):
        @wraps(func)  # This preserves the funcs info after being wrapped
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)
            # Use SnakeViz from hereon instead
            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE  # 'cumulative'
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())
            return retval

        return wrapper

    return inner


if __name__ == '__main__':

    # This syntax means that run_test_session = profile(run_test_session)
    @profile(output_file='profile.prof')
    def run_test_session():
        cd = CopyDraw('./', verbose=True)
        cd.init_session('TEST_SESSION')
        cd.init_block(block_name='TEST_BLOCK', n_trials=3, letter_time=2.2,
                      finish_when_raised=False)
        cd.exec_block()
        cd.exit()
    run_test_session()

# Seems _move_cursor, draw_and_flip & create_frame need the most time
