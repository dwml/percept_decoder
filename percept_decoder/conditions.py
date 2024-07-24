from enum import Enum


class Condition(Enum):
    off_med_on_stim = 1
    off_med_off_stim = 2
    on_med_50p_off_stim = 3
    on_med_100p_off_stim = 4
    on_med_on_stim = 5
