__all__ = ['Logger']


import logging


class Logger:
    def __init__(self, name, filename, level=logging.DEBUG):
        self.__name = name
        self.__filename = filename
        self.__level = level

    @property
    def name(self):
        return self.__name

    @property
    def filename(self):
        return self.__filename

    @property
    def level(self):
        return self.__level

    def set_level(self, level):
        self.__level = level

    def __write_message(self, level, name, message):
        with open(self.__filename, 'a', ) as f:
            f.write("{}:{}:{}\n".format(level, name, message))

    def blank(self):
        with open(self.__filename, 'a', ) as f:
            f.write("\n")

    def separator(self):
        with open(self.__filename, 'a', ) as f:
            f.write("# -------------------- #\n")

    def debug(self, message):
        if self.__level <= logging.DEBUG:
            self.__write_message('DEBUG', self.__name, message)

    def info(self, message):
        if self.__level <= logging.INFO:
            self.__write_message('INFO', self.__name, message)

    def warning(self, message):
        if self.__level <= logging.WARNING:
            self.__write_message('WARNING', self.__name, message)

    def error(self, message):
        if self.__level <= logging.ERROR:
            self.__write_message('ERROR', self.__name, message)
