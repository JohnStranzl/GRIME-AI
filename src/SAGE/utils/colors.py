# sam2_gui/utils/colors.py
_COLOR_CYCLE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 127, 0),
    (127, 0, 255),
    (0, 127, 255),
]


def get_color_for_index(idx: int):
    return _COLOR_CYCLE[idx % len(_COLOR_CYCLE)]
