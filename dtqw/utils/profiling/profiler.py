import csv
import json
from datetime import datetime
from urllib import request, error

from dtqw.utils.logger import is_logger

__all__ = ['Profiler']


class Profiler:
    """Profile and export the resources consumed by Spark."""

    def __init__(self, base_url='http://localhost:4040/api/v1/'):
        """
        Build a Profiler object.

        Parameters
        ----------
        base_url: str, optional
            The base URL for getting information about the resources consumed. Default is http://localhost:4040/api/v1/.
        """
        self._base_url = base_url

        self._times = None
        self._rdd = None
        self._resources = None
        self._executors = None

        self._logger = None

    @property
    def times(self):
        return self._times

    @property
    def rdd(self):
        return self._rdd

    @property
    def executors(self):
        return self._executors

    @property
    def resources(self):
        return self._resources

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        """
        Parameters
        ----------
        logger : Logger
            A Logger object or None to disable logging.

        Raises
        ------
        TypeError

        """
        if is_logger(logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError('logger instance expected, not "{}"'.format(type(logger)))

    @staticmethod
    def _default_rdd():
        return {'memoryUsed': 0, 'diskUsed': 0}

    @staticmethod
    def _default_resources():
        return {'totalShuffleWrite': [], 'totalShuffleRead': [], 'diskUsed': [], 'memoryUsed': []}

    @staticmethod
    def _default_executor():
        return {'totalShuffleWrite': [], 'totalShuffleRead': [], 'diskUsed': [], 'memoryUsed': []}

    @staticmethod
    def _export_values(values, fieldnames, filename, extension='csv'):
        if extension == 'csv':
            f = filename + '.' + extension

            with open(f, 'w') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

                for v in values:
                    w.writerow(v)
        else:
            raise Exception("unsupported file extension!")

    def __str__(self):
        return self.__class__.__name__

    def to_string(self):
        return self.__str__()

    def _request(self, url_suffix=''):
        if self.logger:
            self.logger.info('performing request to "{}"...'.format(self._base_url + 'applications' + url_suffix))

        t1 = datetime.now()

        try:
            with request.urlopen(self._base_url + 'applications' + url_suffix) as response:
                result = response.read()

            if self.logger:
                self.logger.debug('request performed in {}s'.format((datetime.now() - t1).total_seconds()))
        except error.URLError as e:
            if self.logger:
                self.logger.warning('request failed with the following error: "{}" and no data will be returned'.format(e.reason))
            return None

        if result is not None:
            result = json.loads(result.decode('utf-8'))
        else:
            if self.logger:
                self.logger.warning('the response is empty'.format(self._base_url + 'applications' + url_suffix))
        return result

    def request_applications(self):
        """
        Request applications info.

        Returns
        -------
        list
            A list with applications info.

        """
        return self._request()

    def request_jobs(self, app_id, job_id=None):
        """
        Request an application's jobs info.

        Parameters
        ----------
        app_id : int
            The application's id.
        job_id : int, optional
            The job's id.

        Returns
        -------
        list or dict
            A list with all application's jobs info or a dict with the job info. None when an error occurred.

        """
        if job_id is None:
            return self._request('/{}/jobs'.format(app_id))
        else:
            return self._request('/{}/jobs/{}'.format(app_id, job_id))

    def request_stages(self, app_id, stage_id=None):
        """
        Request an application's stages info.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int, optional
            The stage's id.

        Returns
        -------
        list or dict
            A list with all application's stages info or a dict with the stage info. None when an error occurred.

        """
        if stage_id is None:
            return self._request('/{}/stages'.format(app_id))
        else:
            return self._request('/{}/stages/{}'.format(app_id, stage_id))

    def request_stageattempt(self, app_id, stage_id, stageattempt_id):
        """
        Request an application's stage attempts info.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int
            The stage's id.
        stageattempt_id : int
            The stage attempt's id.

        Returns
        -------
        dict
            A dict with an application's stage attempt info. None when an error occurred.

        """
        return self._request('/{}/stages/{}/{}'.format(app_id, stage_id, stageattempt_id))

    def request_stageattempt_tasksummary(self, app_id, stage_id, stageattempt_id):
        """
        Request the task summary of a stage attempt info.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int
            The stage's id.
        stageattempt_id : int
            The stage attempt's id.

        Returns
        -------
        dict
            A dict with the task summary of a stage attempt. None when an error occurred.

        """
        return self._request('/{}/stages/{}/{}/taskSummary'.format(app_id, stage_id, stageattempt_id))

    def request_stageattempt_tasklist(self, app_id, stage_id, stageattempt_id):
        """
        Request the task list of a stage attempt info.

        Parameters
        ----------
        app_id : int
            The application's id.
        stage_id : int
            The stage's id.
        stageattempt_id : int
            The stage attempt's id.

        Returns
        -------
        list
            A list with the task list of a stage attempt. None when an error occurred.

        """
        return self._request('/{}/stages/{}/{}/taskList'.format(app_id, stage_id, stageattempt_id))

    def request_executors(self, app_id):
        """
        Request an application's active executors info.

        Parameters
        ----------
        app_id : int
            The application's id.

        Returns
        -------
        list
            A list with all application's active executors info. None when an error occurred.

        """
        return self._request('/{}/executors'.format(app_id))

    def request_allexecutors(self, app_id):
        """
        Request an application's executors info.

        Parameters
        ----------
        app_id : int
            The application's id.

        Returns
        -------
        list
            A list with all application's executors info. None when an error occurred.

        """
        return self._request('/{}/allexecutors'.format(app_id))

    def request_rdd(self, app_id, rdd_id=None):
        """
        Request an application's RDD info.

        Parameters
        ----------
        app_id : int
            The application's id.
        rdd_id : int, optional
            The RDD's id.

        Returns
        -------
        list or dict
            A list with all application's RDD info or a dict with the RDD info. None when an error occurred.

        """
        if rdd_id is None:
            return self._request('/{}/storage/rdd'.format(app_id))
        else:
            return self._request('/{}/storage/rdd/{}'.format(app_id, rdd_id))

    def start(self):
        """Reset the profiler attributes to get info for a new profiling round."""
        self._times = {}
        self._rdd = {}
        self._resources = self._default_resources()
        self._executors = {}

    def log_executors(self, data=None, app_id=None):
        """
        Log all executors info into the log file.

        When no data is provided, the application's id is used to request those data.

        Parameters
        ----------
        data : list, optional
            The executors data.
        app_id : int
            The application's id.

        Returns
        -------
        None

        """
        if self.logger:
            if data is None:
                if app_id is None:
                    self.logger.error('expected an application id, not "{}"'.format(type(app_id)))
                    raise ValueError('expected an application id, not "{}"'.format(type(app_id)))
                data = self.request_executors(app_id)

            if data is not None:
                self.logger.info("printing executors data...")
                for d in data:
                    for k, v in d.items():
                        self.logger.info("{}: {}".format(k, v))
        else:
            print('no logger has been defined')

    def log_rdd(self, data=None, app_id=None, rdd_id=None):
        """
        Log all RDD info into the log file.

        When no data is provided, the application's id is used to request all its RDD data.
        If the RDD's id are also provided, they are used to get its data.

        Parameters
        ----------
        data : list, optional
            The executors data.
        app_id : int, optional
            The application's id.
        rdd_id : int, optional
            The RDD's id.

        Returns
        -------
        None

        """
        if self.logger:
            if data is None:
                if app_id is None:
                    self.logger.error('expected an application id, not "{}"'.format(type(app_id)))
                    raise ValueError('expected an application id, not "{}"'.format(type(app_id)))
                else:
                    if rdd_id is None:
                        data = self.request_rdd(app_id)

                        if data is not None:
                            self.logger.info("printing RDDs data...")
                            for d in data:
                                for k, v in d.items():
                                    if k != 'partitions':
                                        self.logger.info("{}: {}".format(k, v))
                    else:
                        data = self.request_rdd(app_id, rdd_id)

                        if data is not None:
                            self.logger.info("printing RDD (id {}) data...".format(rdd_id))
                            for k, v in data.items():
                                if k != 'partitions':
                                    self.logger.info("{}: {}".format(k, v))
        else:
            print('No logger has been defined')

    def profile_times(self, name, value):
        """
        Store the execution or building time for a named quantum walk element.

        Parameters
        ----------
        name : str
            A name for the element.
        value : float
            The measured execution or building time of the element.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.info('profiling time for "{}"...'.format(name))

        self._times[name] = value

    def profile_rdd(self, name, app_id, rdd_id):
        """
        Store information about a RDD that represents a quantum walk element.

        Parameters
        ----------
        name : str
            A name for the element.
        app_id : int
            The application's id.
        rdd_id : int
            The RDD's id.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.info('profiling RDD for "{}"...'.format(name))

        if name not in self._rdd:
            self._rdd[name] = self._default_rdd()

        data = self.request_rdd(app_id, rdd_id)

        if data is not None:
            for k, v in data.items():
                if k in self._rdd[name]:
                    self._rdd[name][k] = v

    def profile_resources(self, app_id):
        """
        Store information about the resources consumed by the application.

        Parameters
        ----------
        app_id : int
            The application's id.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.info('profiling application resources...')

        data = self.request_executors(app_id)

        for k in self._resources:
            self._resources[k].append(0)

        if data is not None:
            for d in data:
                for k, v in d.items():
                    if k in self._resources:
                        self._resources[k][-1] += v

    def profile_executors(self, app_id, exec_id=None):
        """
        Store all executors info.

        When no executor's id is provided, all the application's executors info are requested and stored.

        Parameters
        ----------
        app_id : int
            The application's id.
        exec_id : int, optional
            The executor's id.

        Returns
        -------
        None

        """
        if self.logger:
            if exec_id is None:
                self.logger.info('profiling resources of executors...')
            else:
                self.logger.info('profiling resources of executor {}...'.format(exec_id))

        data = self.request_executors(app_id)

        if data is not None:
            if exec_id is None:
                for d in data:
                    if d['id'] not in self._executors:
                        self._executors[d['id']] = self._default_executor()

                    for k, v in d.items():
                        if k in self._executors[d['id']]:
                            self._executors[d['id']][k].append(v)
            else:
                for d in data:
                    if d['id'] == exec_id:
                        if d['id'] not in self._executors:
                            self._executors[d['id']] = self._default_executor()

                        for k, v in d.items():
                            if k in self._executors[d['id']]:
                                self._executors[d['id']][k].append(v)
                        break

    def get_times(self, name=None):
        """
        Get all measured times or for a specific quantum walk element.

        Parameters
        ----------
        name : str, optional
            A name for the element.

        Returns
        -------
        dict or float
            A dict with all measured times or the measured time for the specific quantum walk element.

        """
        if len(self._times):
            if name is None:
                return self._times.copy()
            else:
                if name not in self._times:
                    if self.logger:
                        self.logger.warning('no measurement of time has been done for "{}"'.format(name))
                    return {}
                return self._times[name]
        else:
            if self.logger:
                self.logger.warning('no measurement of time has been done')
            return {}

    def get_rdd(self, name=None):
        """
        Get all RDD resources measurements or for a specific quantum walk element.

        Parameters
        ----------
        name : str, optional
            A name for the element.

        Returns
        -------
        dict
            A dict with all RDD resources measurements or the measured time for the specific quantum walk element.

        """
        if len(self._rdd):
            if name is None:
                return self._rdd.copy()
            else:
                if name not in self._rdd:
                    if self.logger:
                        self.logger.warning('no measurement of RDD resources has been done for "{}"'.format(name))
                    return {}
                return self._rdd[name]
        else:
            if self.logger:
                self.logger.warning('no measurement of RDD resources has been done')
            return {}

    def get_resources(self, key=None):
        """
        Get all or a specific resource measurement.

        Parameters
        ----------
        key : str, optional
            The key representing a resource.

        Returns
        -------
        dict
            A dict with all resource measurements or the specific one.

        """
        if len(self._resources):
            if key is None:
                return self._resources.copy()
            else:
                if key not in self._default_resources():
                    if self.logger:
                        self.logger.warning('no measurement of resources has been done for "{}"'.format(key))
                    return {}
                return self._resources[key]
        else:
            if self.logger:
                self.logger.warning('no measurement of resources has been done')
            return {}

    def get_executors(self, exec_id=None):
        """
        Get all executors' resources measurements or for a specific executor.

        Parameters
        ----------
        exec_id : int, optional
            An executor id.

        Returns
        -------
        dict
            A dict with all executors' resources or the specific executor's.

        """
        if len(self._executors):
            if exec_id is None:
                return self._executors.copy()
            else:
                if exec_id not in self._executors:
                    if self.logger:
                        self.logger.warning('no measurement of resources has been done for executor {}'.format(exec_id))
                    return {}
                return self._executors[exec_id]
        else:
            if self.logger:
                self.logger.warning('no measurement of executors resources has been done')
            return self._executors

    def export_times(self, path, extension='csv'):
        """
        Export all stored execution and/or building times.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv', as only CSV files are supported for now.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.info("exporting times in {} format...".format(extension))

        if len(self._times):
            self._export_values([self._times], self._times.keys(), path + 'times', extension)

            if self.logger:
                self.logger.info("times successfully exported")
        else:
            if self.logger:
                self.logger.warning('no measurement of time has been done')

    def export_rdd(self, path, extension='csv'):
        """
        Export all stored RDD resources informations.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv', as only CSV files are supported for now.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.info("exporting RDD resources in {} format...".format(extension))

        if len(self._rdd):
            rdd = []

            for k, v in self._rdd.items():
                tmp = v.copy()
                tmp['rdd'] = k
                rdd.append(tmp)

            self._export_values(rdd, rdd[-1].keys(), path + 'rdd', extension)

            if self.logger:
                self.logger.info("RDD resources successfully exported")
        else:
            if self.logger:
                self.logger.warning('no measurement of RDD resources has been done')

    def export_resources(self, path, extension='csv'):
        """
        Export all stored resources informations.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv', as only CSV files are supported for now.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.info("exporting resources in {} format...".format(extension))

        if len(self._resources):
            resources = []
            size = 0

            for k, v in self._resources.items():
                size = len(v)
                break

            for i in range(size):
                tmp = self._default_resources()

                for k in tmp:
                    tmp[k] = self._resources[k][i]

                resources.append(tmp)

            self._export_values(resources, resources[-1].keys(), path + 'resources', extension)

            if self.logger:
                self.logger.info("resources successfully exported")
        else:
            if self.logger:
                self.logger.warning('no measurement of resources has been done')

    def export_executors(self, path, extension='csv'):
        """
        Export all stored executors' resources informations.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv', as only CSV files are supported for now.

        Returns
        -------
        None

        """
        if self.logger:
            self.logger.info("exporting executors resources in {} format...".format(extension))

        if len(self._executors):
            for k1, v1 in self._executors.items():
                executors = []
                size = 0

                for k2, v2 in v1.items():
                    size = len(v2)
                    break

                for i in range(size):
                    tmp = self._default_resources()

                    for k2 in tmp:
                        tmp[k2] = v1[k2][i]

                    executors.append(tmp)

                self._export_values(executors, executors[-1].keys(), "{}executor_{}".format(path, k1), extension)

            if self.logger:
                self.logger.info("executors resources successfully exported")
        else:
            if self.logger:
                self.logger.warning('no measurement of executors resources has been done')

    def export(self, path, extension='csv'):
        """
        Export all stored profiling information.

        Parameters
        ----------
        path: str
            The location of the files.
        extension: str, optional
            The extension of the files. Default value is 'csv', as only CSV files are supported for now.

        Returns
        -------
        None

        """
        self.export_times(path, extension)
        self.export_rdd(path, extension)
        self.export_resources(path, extension)
        self.export_executors(path, extension)


def is_profiler(obj):
    """
    Check whether argument is a Profiler object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a Profiler object, False otherwise.

    """
    return isinstance(obj, Profiler)
