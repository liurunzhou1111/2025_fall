import os
import sys

def init_paths(base_dir=None, project_name=None):
    if base_dir is None:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        cur_dir = base_dir
    python_paths = [
        os.path.join(cur_dir, 'dreamer-datasets'),
        os.path.join(cur_dir, 'dreamer-models'),
        os.path.join(cur_dir, 'dreamer-train'),
    ]
    if project_name is not None:
        python_paths.append(os.path.join(cur_dir, 'dreamer-train', 'projects', project_name))
        
    for python_path in python_paths:
        sys.path.insert(0, python_path)
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] += ':{}'.format(python_path)
        else:
            os.environ['PYTHONPATH'] = python_path
            
    # Cache paths from drivedreamer conda environment
    # These paths are based on the default cache locations used by torch and transformers in the drivedreamer environment
    os.environ['TORCH_HOME'] = os.environ.get('PATH_TO_TORCH_HOME', '/mnt/home/yf2578/.cache/torch')
    os.environ['TRANSFORMERS_CACHE'] = os.environ.get('PATH_TO_TRANSFORMERS_CACHE', '/mnt/home/yf2578/.cache/huggingface')
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ.get('PATH_TO_HUGGINGFACE_HUB_CACHE', '/mnt/home/yf2578/.cache/huggingface')
    os.environ['XDG_CACHE_HOME'] = os.environ.get('PATH_TO_XDG_CACHE', '/mnt/home/yf2578/.cache')
        
    # For users with huggingface network issues, uncomment the following lines for speedding up downloading
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # uncomment the following lines if you have already downloaded huggingface models, and you want set up offline mode (use the local models)
    # os.environ['HF_HUB_OFFLINE'] = '1'
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'
