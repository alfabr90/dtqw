import logging

__all__ = ['Logger']


class Logger:
    def __init__(self, name, filename, level=logging.DEBUG):
        self._name = name
        self._filename = filename
        self._level = level

    @property
    def name(self):
        return self._name

    @property
    def filename(self):
        return self._filename

    @property
    def level(self):
        return self._level

    def set_level(self, level):
        self._level = level

    def __write_message(self, level, name, message):
        with open(self._filename, 'a') as f:
            f.write("{}:{}:{}\n".format(level, name, message))

    def blank(self):
        with open(self._filename, 'a') as f:
            f.write("\n")

    def separator(self):
        with open(self._filename, 'a') as f:
            f.write("# -------------------- #\n")

    def debug(self, message):
        if self._level <= logging.DEBUG:
            self.__write_message('DEBUG', self._name, message)

    def info(self, message):
        if self._level <= logging.INFO:
            self.__write_message('INFO', self._name, message)

    def warning(self, message):
        if self._level <= logging.WARNING:
            self.__write_message('WARNING', self._name, message)

    def error(self, message):
        if self._level <= logging.ERROR:
            self.__write_message('ERROR', self._name, message)
