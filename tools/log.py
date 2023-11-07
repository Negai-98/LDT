import csv
from datetime import datetime
import logging
import os
from tools import io


def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


class logger(object):
    def __init__(self, cfg):
        io.makedirs(cfg.log.save_path)
        logging.basicConfig(filename=os.path.join(cfg.log.save_path,
                            "log_{:}.txt".format(str(datetime.now().strftime('%Y_%m_%d_%H_%M')))),
                            level=logging.INFO, filemode="a")
        self.log = get_logger(logpath=os.path.join(cfg.log.save_path, 'logs'))
        self.log.setLevel(logging.INFO)
        self.info(cfg)
        self.trainlogpath = os.path.join(cfg.log.save_path, 'training.csv')
        self.testlogpath = os.path.join(cfg.log.save_path, 'test.csv')
        self.evallogpath = os.path.join(cfg.log.save_path, 'eval.csv')
        assert len(cfg.log.trainformat) == len(cfg.log.traincolumns)
        assert len(cfg.log.evalformat) == len(cfg.log.evalcolumns)
        self.traincolumns = cfg.log.traincolumns
        self.trainformat = cfg.log.trainformat
        self.evalformat = cfg.log.evalformat
        self.evalcolumns = cfg.log.evalcolumns
        if not os.path.exists(self.trainlogpath):
            with open(self.trainlogpath, 'w') as f:
                csvlogger = csv.DictWriter(f, cfg.log.traincolumns)
                csvlogger.writeheader()
        if not os.path.exists(self.evallogpath):
            with open(self.evallogpath, 'w') as f:
                csvlogger = csv.DictWriter(f, cfg.log.evalcolumns)
                csvlogger.writeheader()
        if not os.path.exists(self.testlogpath):
            with open(self.testlogpath, 'w') as f:
                csvlogger = csv.DictWriter(f, cfg.log.traincolumns)
                csvlogger.writeheader()

    def info(self, message):
        self.log.info(message)

    def write(self, message, mode="train"):
        assert mode in ["train", "test", "eval"]
        if mode == "train":
            assert len(message) == len(self.traincolumns)
            logpath = self.trainlogpath
            columns = self.traincolumns
            form = self.trainformat
        elif mode == 'eval':
            assert len(message) == len(self.evalcolumns)
            logpath = self.evallogpath
            columns = self.evalcolumns
            form = self.evalformat
        else:
            assert len(message) == len(self.traincolumns)
            logpath = self.testlogpath
            columns = self.traincolumns
            form = self.trainformat
        # try:
        logdict = {
            columns[i]: (message[i] if form[i] is None else
                                   form[i].format(message[i]))
            for i in range(len(message))
        }
        with open(logpath, 'a') as f:
            csvlogger = csv.DictWriter(f, columns)
            csvlogger.writerow(logdict)
