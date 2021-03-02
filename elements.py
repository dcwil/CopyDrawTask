from psychopy.visual import ShapeStim, ImageStim, Circle,\
    TextStim, Rect

classes = {
    'theBox': ShapeStim,
    'trial_number': TextStim,
    'cursor': Circle,
    'trace': ShapeStim,
    'instructions': ImageStim,
    'time_bar': Rect,
    'start_point': Rect,
    'template': ImageStim
}

static_params = {
    'theBox': {
        'closeShape': True,
        'pos': (0, 0),
        'size': 1.5,
        'lineColor': 'white'
    },
    'trial_number': {
        'pos': (0.9, 0.9),
        'units': 'norm',
        'color': 'white',
        'height': 0.05
    },
    'cursor': {
        'units': 'pix',
        'size': (30, 30),
        'color': 'red',
        'fillColor': 'red',
        'lineColor': 'red'
    },
    'trace': {
        'units': 'pix',
        'lineColor': 'red',
        'lineWidth': 5,
        'interpolate': True,
        'closeShape': False,
    },
    'instructions': {
        'pos': (0, 0.95)
    },
    'start_point': {
        'pos': (-0.8, 0.7),
        'fillColor': 'Black',
        'lineColor': 'Cyan'
    },
    'time_bar': {
        'pos': (0, -0.85),
        'size': (1, 0.025),
        'fillColor': 'gray'
    },
    'template': {
        'units':'norm',
        'pos': (-0.0025, -0.002),
        'interpolate': True,
        'size':1.6725
    }
}


def create_element(name, **kwargs):
    params = {**kwargs, **static_params[name]}
    return classes[name](**params)
