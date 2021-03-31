#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Have a test run of the copy draw task

from copydraw import CopyDraw

# testing stim sizes - should stim_size be hardcoded, as it doesnt effect size of displayed tempalte?


def run_copy_draw():
    data_dir = '../'
    test_cpd = CopyDraw(data_dir,
                        verbose=True)
    log = logging.getLogger(__name__)
    log.info('Started test_run_copydraw')

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


if __name__ == "__main__":
    # As of now, this is only a very basic run -> might be more
    # complexe later, testing setting of screens etc...

    import logging

    logging.basicConfig(filename='test_run.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG,
                        filemore='w')
    run_copy_draw()
