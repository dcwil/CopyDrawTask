#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Have a test run of the copy draw task

from copydraw import CopyDraw


def run_copy_draw():
    data_dir = '../'
    test_cpd = CopyDraw(data_dir,
                        screen_size=(1000, 600),
                        verbose=True)

    test_cpd.init_session('TEST_SESSION')

    # for integration - will this be a yaml file?
    cfg = {
        'block_name': 'TEST_BLOCK_1',
        'n_trials': 2,
        'letter_time': 2.7,
        'n_letters': 3,
        'finish_when_raised': False
    }

    test_cpd.exec_block(cfg, stim='off')

    test_cpd.exec_block(cfg, stim='off')

    test_cpd.exit()


if __name__ == "__main__":
    # As of now, this is only a very basic run -> might be more
    # complexe later, testing setting of screens etc...
    run_copy_draw()
