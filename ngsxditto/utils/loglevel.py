import logging
import ipywidgets as widgets
from IPython.display import display

def loggingSlider(name=None, default_level=logging.INFO):
    """
    Creates an interactive slider to set logging levels.
    - name=None  -> controls the root logger (global)
    - name=str   -> controls the logger with the given name
    This also creates a basic logger if the logger has no handlers yet.
    """

    # Determine target logger
    if name is None or name == "":
        logger = logging.getLogger()  # Root logger → global
        label_prefix = "global log level"
    else:
        logger = logging.getLogger(name)
        label_prefix = name + " log level"

    # Configure basic logging if root has no handlers
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    # Map slider positions to logging levels
    level_map = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING,
        3: logging.ERROR,
        4: logging.CRITICAL
    }
    labels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    slider_value = 1
    for i in range(5):
        if default_level == level_map[i]:
            slider_value = i
        if default_level == labels[i]:
            slider_value = i
        if default_level == i:
            slider_value = i

    logger.setLevel(level_map[slider_value])

    # Create the slider
    slider = widgets.IntSlider(
        value=slider_value, min=0, max=4, step=1,
        description=f"{label_prefix}: {labels[slider_value]}",
        continuous_update=True,
        readout=False,
        layout=widgets.Layout(width='400px'),
        style={'description_width': '250px'}
    )
    slider.style.handle_color = 'lightblue'

    # Callback to update logging level
    def update_level(change):
        new_val = change['new']
        new_level = level_map[new_val]
        slider.description = f"{label_prefix}: {labels[new_val]}"
        logger.setLevel(new_level)
        logger.log(new_level, f"Logging level set to {labels[new_val]}")

    slider.observe(update_level, names='value')

    return display(slider)


def test_logging(name=None):
    """
    Function to test logging output.
    - name=None -> global/root logger
    - name=str  -> logger with given name
    """
    if name is None or name == "":
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)

    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")