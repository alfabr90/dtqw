import csv
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import statistics as st
from datetime import datetime
from urllib import request, error
from mpl_toolkits.mplot3d import Axes3D

__all__ = ['Profiler']


class Profiler:
    def __init__(self, base_url='http://localhost:4040/api/v1/'):
        self._base_url = base_url

        self._times = {}
        self._rdd = {}
        self._resources = self._default_resources()
        self._executors = {}

        self.logger = None

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
    def _export_values(values, filename, extension='csv'):
        if extension == 'csv':
            f = filename + extension

            fieldnames = values.keys()

            with open(f, 'w') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerow(values)
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
        return self._request()

    def request_jobs(self, app_id, job_id=None):
        if job_id is None:
            return self._request('/{}/jobs'.format(app_id))
        else:
            return self._request('/{}/jobs/{}'.format(app_id, job_id))

    def request_stages(self, app_id, stage_id=None):
        if stage_id is None:
            return self._request('/{}/stages'.format(app_id))
        else:
            return self._request('/{}/stages/{}'.format(app_id, stage_id))

    def request_stageattempt(self, app_id, stage_id, stageattempt_id):
        return self._request('/{}/stages/{}/{}'.format(app_id, stage_id, stageattempt_id))

    def request_stageattempt_tasksummary(self, app_id, stage_id, stageattempt_id):
        return self._request('/{}/stages/{}/{}/taskSummary'.format(app_id, stage_id, stageattempt_id))

    def request_stageattempt_tasklist(self, app_id, stage_id, stageattempt_id):
        return self._request('/{}/stages/{}/{}/taskList'.format(app_id, stage_id, stageattempt_id))

    def request_executors(self, app_id):
        return self._request('/{}/executors'.format(app_id))

    def request_allexecutors(self, app_id):
        return self._request('/{}/allexecutors'.format(app_id))

    def request_rdd(self, app_id, rdd_id=None):
        if rdd_id is None:
            return self._request('/{}/storage/rdd'.format(app_id))
        else:
            return self._request('/{}/storage/rdd/{}'.format(app_id, rdd_id))

    def log_executors(self, data=None, app_id=None):
        if self.logger:
            if data is None:
                if app_id is None:
                    self.logger.error('expected an application id, not "{}"'.format(type(app_id)))
                    raise ValueError('expected an application id, not "{}"'.format(type(app_id)))
                data = self.request_executors(app_id)

            if data is not None:
                self.logger.info("Printing executors data...")
                for d in data:
                    for k, v in d.items():
                        self.logger.info("{}: {}".format(k, v))
        else:
            print('No logger has been defined')

    def log_rdd(self, data=None, app_id=None, rdd_id=None):
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
        self._times[name] = value

    def profile_rdd(self, name, app_id, rdd_id):
        if self.logger:
            self.logger.info('profiling RDD for "{}"...'.format(name))

        if not (name in self._rdd):
            self._rdd[name] = self._default_rdd()

        if isinstance(rdd_id, (list, tuple)):
            for rdd in rdd_id:
                data = self.request_rdd(app_id, rdd)

                if data is not None:
                    for k, v in data.items():
                        if k in self._rdd[name].keys():
                            self._rdd[name][k] += v
        else:
            data = self.request_rdd(app_id, rdd_id)

            if data is not None:
                for k, v in data.items():
                    if k in self._rdd[name].keys():
                        self._rdd[name][k] = v

    def profile_resources(self, app_id):
        if self.logger:
            self.logger.info('profiling application resources...')

        data = self.request_executors(app_id)

        for k in self._resources.keys():
            self._resources[k].append(0)

        if data is not None:
            for d in data:
                for k, v in d.items():
                    if k in self._resources.keys():
                        self._resources[k][-1] += v

    def profile_executors(self, app_id, exec_id=None):
        if self.logger:
            if exec_id is None:
                self.logger.info('profiling resources of executors...')
            else:
                self.logger.info('profiling resources of executor {}...'.format(exec_id))

        data = self.request_executors(app_id)

        if data is not None:
            if exec_id is None:
                for d in data:
                    if not (d['id'] in self._executors):
                        self._executors[d['id']] = self._default_executor()

                    for k, v in d.items():
                        if k in self._executors[d['id']]:
                            self._executors[d['id']][k].append(v)
            else:
                for d in data:
                    if d['id'] == exec_id:
                        if not (d['id'] in self._executors):
                            self._executors[d['id']] = self._default_executor()

                        for k, v in d.items():
                            if k in self._executors[d['id']]:
                                self._executors[d['id']][k].append(v)
                        break

    def get_time(self, key):
        if not (key in self._times):
            if self.logger:
                self.logger.warning('key "{}" not present'.format(key))
            return None
        return self._times[key]

    def get_rdd(self, name=None, key=None):
        if name is None:
            if key is None:
                return self._rdd
            else:
                rdd = {}

                for k1, v1 in self._rdd.items():
                    rdd[k1] = {}
                    for k2, v2 in v1.items():
                        if key == k2:
                            rdd[k1][k2] = v2
                return rdd
        else:
            if not (name in self._rdd):
                if self.logger:
                    self.logger.warning('key "{}" not present'.format(key))
                return None

            if key is None:
                return self._rdd[name]
            else:
                if not (key in self._rdd[name]):
                    if self.logger:
                        self.logger.warning('key "{}" not present'.format(key))
                    return None
                return self._rdd[name][key]

    def get_resources(self, key=None):
        if key is None:
            resources = self._default_resources()

            for k in resources.keys():
                resources[k] = self._resources[k][-1]

            return resources
        else:
            if not (key in self._resources[key]):
                if self.logger:
                    self.logger.warning('key "{}" not present'.format(key))
                return None
            return self._resources[key][-1]

    def get_executors(self, exec_id=None, key=None):
        if exec_id is None:
            if key is None:
                return self._rdd
            else:
                executors = {}

                for k1, v1 in self._executors.items():
                    executors[k1] = {}
                    for k2, v2 in v1.items():
                        if key == k2:
                            executors[k1][k2] = v2
                return executors
        else:
            if not (exec_id in self._executors):
                if self.logger:
                    self.logger.warning('key "{}" not present'.format(key))
                return None

            if key is None:
                resources = self._default_executor()

                for k in resources.keys():
                    resources[k] = self._executors[exec_id][k][-1]

                return resources
            else:
                if not (key in self._executors[exec_id]):
                    if self.logger:
                        self.logger.warning('key "{}" not present'.format(key))
                    return None
                return self._executors[exec_id][key][-1]

    def export_times(self, filename, extension='csv'):
        self._export_values(self._times, filename, extension)

    def export_mean_times(self, filename, extension='csv'):
        times = {}

        for k, v in self._times.items():
            times[k] = st.mean(v)

        self._export_values(times, filename, extension)

    def export_pstdev_times(self, filename, extension='csv'):
        times = {}

        for k, v in self._times.items():
            times[k] = st.pstdev(v)

        self._export_values(times, filename, extension)

    def export_mean_resources(self, filename, extension='csv'):
        resources = {}

        for k, v in self._resources.items():
            resources[k] = st.mean(v)

        self._export_values(resources, filename, extension)

    def export_pstdev_resources(self, filename, extension='csv'):
        resources = {}

        for k, v in self._resources.items():
            resources[k] = st.pstdev(v)

        self._export_values(resources, filename, extension)

    def plot_times(self, title, filename, **kwargs):
        mean_times = []
        pstdev_times = []
        keys = self._times.keys()

        for k in keys:
            mean_times.append(st.mean(self._times[k]))
            pstdev_times.append(st.pstdev(self._times[k]))

        fig, ax = plt.subplots()

        index = np.arange(len(keys))

        plt.bar(index, mean_times, 0.35, color='r', yerr=pstdev_times)

        plt.xlabel('Operations')
        plt.ylabel('Time (s)')
        plt.xticks(index, keys, rotation='vertical')
        plt.title(title)

        plt.tight_layout()
        plt.savefig(filename, kwargs=kwargs)

    def plot_resources(self, title, filename, **kwargs):
        mean_resources = []
        pstdev_resources = []
        keys = Profiler.resources_base2.keys()

        print(self._resources.keys())
        print(keys)

        for k in keys:
            mean_resources.append(st.mean(self._resources[k]))
            pstdev_resources.append(st.pstdev(self._resources[k]))

        fig, ax = plt.subplots()

        index = np.arange(len(keys))

        plt.bar(index, mean_resources, 0.35, color='b', yerr=pstdev_resources)

        plt.xlabel('Resources')
        plt.ylabel('Bytes')
        plt.xticks(index, keys, rotation='vertical')
        plt.title(title)

        plt.tight_layout()
        plt.savefig(filename, kwargs=kwargs)

    def plot_executor(self, title, filename, **kwargs):
        mean_resources = {}
        pstdev_resources = {}
        e_keys = self._executors.keys()
        r_keys = Profiler.resources_base2.keys()

        for k1 in r_keys:
            mean_resources[k1] = []
            pstdev_resources[k1] = []

            for k2 in e_keys:
                mean_resources[k1].append(st.mean(self._executors[k2][k1]))
                pstdev_resources[k1].append(st.pstdev(self._executors[k2][k1]))

        fig, ax = plt.subplots()

        index = np.arange(len(e_keys))

        i = 0
        for k in r_keys:
            plt.bar(index + 0.35 * i, mean_resources[k], 0.35, yerr=pstdev_resources[k], label=k)
            i += 1

        plt.xlabel('Resources by executors')
        plt.ylabel('Bytes')
        plt.xticks(index + 0.35 * len(r_keys) / 2, e_keys)
        plt.title(title)

        plt.tight_layout()
        plt.savefig(filename, kwargs=kwargs)
