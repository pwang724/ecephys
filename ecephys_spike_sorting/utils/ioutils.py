import pickle
import yaml
import os
import logging
import json

def psave(obj, filename):
    if filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def pload(filename):
    if filename[-4:] != '.pkl':
        filename += '.pkl'
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def ysave(obj, filename):
    with open(filename + '.yaml', "w") as outfile:
        yaml.dump(obj, outfile)

def yload(filename):
    if filename[-5:] != '.yaml':
        filename += '.yaml'
    with open(filename, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def jsave(obj, filename):
    with open(filename + '.json', "w") as outfile:
        json.dump(obj, outfile)

def jload(filename):
    if filename[-5:] != '.json':
        filename += '.json'
    with open(filename, "r") as file:
        return json.load(file
                         # , Loader=yaml.FullLoader
                         )



def setupLogger(logName):
    if os.path.exists(logName):
        os.remove(logName)
    pn = os.path.join(logName)
    logger = logging.getLogger('experiment')
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(pn)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def log_dict(logger, dict, name):
    logger.info('='*10)
    logger.info(f'***{name}***')
    for k, v in dict.items():
        logger.info(f'{k} : {v}')
    logger.info('='*10)