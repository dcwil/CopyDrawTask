#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Have a test run of the copy draw task

from copydraw import CopyDraw

import pyglet as pg

from psychopy.visual import Window


def select_display():

    """ Select the screen to display the task to

    NOTE: Looking at screens[0].display.x_screen and
    Looking at screens[0].display.x_screen is seems
    like a setup with an extended screen has only
    one number (screen index) with shifted coords,
    as can be accessed by screen[0].y or .x

    """
    screens = pg.canvas.Display().get_screens()

    if len(screens) > 1:
        print(f"Found {len(screens)} screens with the following settings:"
              ''.join([f"\nScreen {i}: \n {s}" for i, s in enumerate(screens)]))
        resp = -1
        while resp not in list(range(len(screens))):
            inp = input("\nPlease select one of "
                        + str(list(range(len(screens))))
                        + " : ")
            resp = int(inp)

        ix_scr = resp
    else:
        ix_scr = 0

    screen_conf = {'screen': ix_scr}
    # check if the internal indeces agree, if yes -> shift if necessary
    # to the correct screen
    if all([s.display.x_screen == screens[0].display.x_screen
            for s in screens]):
        # now assume that .x and .y of the selected screen correspond
        # to the offsets to the main display
        screen_conf['pos'] = [screens[ix_scr].x, screens[ix_scr].y]

    return screen_conf


def run_copy_draw():
    data_dir = '.'
    test_cpd = CopyDraw(data_dir,
                        old_template_path='./templates',
                        screen_size=(1000, 600))

    test_cpd.init_session('TEST_SESSION')

    test_cpd.init_block(
        block_name='TEST_BLOCK',
        n_trials=2,
        letter_time=2.7,
        finish_when_raised=False
    )

    test_cpd.exec_block()
    test_cpd.exit()


if __name__ == "__main__":
    # As of now, this is only a very basic run -> might be more
    # complexe later, testing setting of screens etc...
    run_copy_draw()
