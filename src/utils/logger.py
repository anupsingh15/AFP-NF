import logging

def setup_logger(name, log_file, level=logging.INFO, mode='a'):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file,mode=mode)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')      
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

    