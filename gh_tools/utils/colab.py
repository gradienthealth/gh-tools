import time
from IPython.display import display, Javascript
import hashlib
import os

def save_notebook():
    assert os.path.exists(os.environ["NOTEBOOK_FILEPATH"]), "The filepath `{}` does not exist".format(os.environ["NOTEBOOK_FILEPATH"])
    file_path = os.environ["NOTEBOOK_FILEPATH"]
    start_md5 = hashlib.md5(open(file_path,'rb').read()).hexdigest()
    display(Javascript('IPython.notebook.save_checkpoint();'))
    current_md5 = start_md5
    
    while start_md5 == current_md5:
        time.sleep(1)
        current_md5 = hashlib.md5(open(file_path,'rb').read()).hexdigest()

def isColabNotebook():
    try:
        module_name = get_ipython().__class__.__module__
        shell = get_ipython().__class__.__name__
        if module_name == "google.colab._shell":
            return True   # Colab Notebook
        elif shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
