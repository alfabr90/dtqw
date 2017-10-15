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

        self._times = []
        self._rdd = []
        self._resources = []
        self._executors = []

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

    def start_round(self):
        self._times.append({})
        self._rdd.append({})
        self._resources.append(self._default_resources())
        self._executors.append({})

    def profile_times(self, name, value):
        self._times[-1][name] = value

    def profile_rdd(self, name, app_id, rdd_id):
        if self.logger:
            self.logger.info('profiling RDD for "{}"...'.format(name))

        if not (name in self._rdd[-1]):
            self._rdd[-1][name] = self._default_rdd()

        data = self.request_rdd(app_id, rdd_id)

        if data is not None:
            for k, v in data.items():
                if k in self._rdd[-1][name]:
                    self._rdd[-1][name][k] = v

    def profile_resources(self, app_id):
        if self.logger:
            self.logger.info('profiling application resources...')

        data = self.request_executors(app_id)

        for k in self._resources[-1]:
            self._resources[-1][k].append(0)

        if data is not None:
            for d in data:
                for k, v in d.items():
                    if k in self._resources[-1]:
                        self._resources[-1][k][-1] += v

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
                    if not (d['id'] in self._executors[-1]):
                        self._executors[-1][d['id']] = self._default_executor()

                    for k, v in d.items():
                        if k in self._executors[-1][d['id']]:
                            self._executors[-1][d['id']][k].append(v)
            else:
                for d in data:
                    if d['id'] == exec_id:
                        if not (d['id'] in self._executors):
                            self._executors[-1][d['id']] = self._default_executor()

                        for k, v in d.items():
                            if k in self._executors[-1][d['id']]:
                                self._executors[-1][d['id']][k].append(v)
                        break

    def get_times(self, func=None, name=None):
        if len(self._times):
            if name is not None:
                if not (name in self._times[-1]):
                    if self.logger:
                        self.logger.warning('key "{}" not present'.format(name))
                    return None

            if func is None:
                if name is None:
                    return self._times[-1]
                else:
                    return self._times[-1][name]
            else:
                if name is None:
                    times = {}

                    for k in self._times[-1]:
                        times[k] = func([t[k] for t in self._times])

                    return times
                else:
                    return func([t[name] for t in self._times])
        else:
            if self.logger:
                self.logger('No measurement of time has been done')
            return self._times

    def get_rdd(self, func=None, name=None, key=None):
        if len(self._rdd):
            if name is not None:
                if not (name in self._rdd[-1]):
                    if self.logger:
                        self.logger.warning('key "{}" not present'.format(name))
                    return None

            if key is not None:
                if not (key in self._default_rdd()):
                    if self.logger:
                        self.logger.warning('key "{}" not present'.format(key))
                    return None

            if func is None:
                if name is None:
                    if key is None:
                        return self._rdd[-1]
                    else:
                        rdd = {}

                        for k, v in self._rdd[-1].items():
                            rdd[k] = v[key]

                        return rdd
                else:
                    if key is None:
                        return self._rdd[-1][name]
                    else:
                        return self._rdd[-1][name][key]
            else:
                keys = []

                for r in self._rdd:
                    keys = keys + [k for k in r.keys()]

                keys = set(keys)

                if name is None:
                    if key is None:
                        rdd = {}

                        for k1 in keys:
                            rdd[k1] = {}
                            for k2 in self._default_rdd():
                                tmp = []
                                for r in self._rdd:
                                    if k1 in r:
                                        if k2 in r[k1]:
                                            tmp.append(r[k1][k2])
                                rdd[k1][k2] = func(tmp)

                        return rdd
                    else:
                        rdd = {}

                        for k in keys:
                            tmp = []
                            for r in self._rdd:
                                if k in r:
                                    if key in r[k]:
                                        tmp.append(r[k][key])
                            rdd[k] = func(tmp)

                        return rdd
                else:
                    if key is None:
                        rdd = {}

                        for k in self._default_rdd():
                            tmp = []
                            for r in self._rdd:
                                if name in r:
                                    if k in r[name]:
                                        tmp.append(r[name][k])
                            rdd[k] = func(tmp)

                        return rdd
                    else:
                        rdd = []

                        for r in self._rdd:
                            rdd.append(r[name][key])

                        return func(rdd)
        else:
            if self.logger:
                self.logger('No measurement of RDD has been done')
            return self._rdd

    def get_resources(self, func=None, key=None):
        if len(self._resources):
            if key is not None:
                if not (key in self._default_resources()):
                    if self.logger:
                        self.logger.warning('key "{}" not present'.format(key))
                    return None

            if func is None:
                if key is None:
                    resources = self._default_resources()

                    for k in resources:
                        resources[k] = self._resources[-1][k][-1]

                    return resources
                else:
                    return self._resources[-1][key][-1]
            else:
                size = len([v for v in self._resources[-1].values()][0])

                if key is None:
                    resources = self._default_resources()

                    for k in resources:
                        for i in range(size):
                            resources[k].append(func([r[k][i] for r in self._resources]))

                    return resources
                else:
                    resources = []

                    for i in range(size):
                        resources.append(func([r[key][i] for r in self._resources]))

                    return resources
        else:
            if self.logger:
                self.logger('No measurement of general resources has been done')
            return self._resources

    def get_executors(self, func=None, exec_id=None, key=None):
        if len(self._executors):
            keys = []

            for e in self._executors:
                keys = keys + [k for k in e.keys()]

            keys = set(keys)

            if exec_id is not None:
                if not (exec_id in keys):
                    if self.logger:
                        self.logger.warning('key "{}" not present'.format(exec_id))
                    return None

            if key is not None:
                if not (key in self._default_executor()):
                    if self.logger:
                        self.logger.warning('key "{}" not present'.format(key))
                    return None

            if func is None:
                if exec_id is None:
                    if key is None:
                        executors = {}

                        for k1 in keys:
                            executors[k1] = self._default_resources()

                            for e in self._executors[::-1]:
                                if k1 in e:
                                    for k2 in executors[k1]:
                                        executors[k1][k2] = e[k1][k2][-1]
                                    break

                        return executors
                    else:
                        executors = {}

                        for k in keys:
                            executors[k] = self._default_resources()

                            for e in self._executors[::-1]:
                                if k in e:
                                    executors[k] = e[k][key][-1]
                                    break

                        return executors
                else:
                    if key is None:
                        executor = self._default_executor()

                        for e in self._executors[::-1]:
                            if exec_id in e:
                                for k in executor:
                                    executor[k] = e[exec_id][k][-1]
                                break

                        return executor
                    else:
                        for e in self._executors[::-1]:
                            if exec_id in e:
                                return e[exec_id][key][-1]
            else:
                if exec_id is None:
                    if key is None:
                        executors = {}

                        for k1 in keys:
                            executors[k1] = self._default_resources()

                            for e1 in self._executors[::-1]:
                                if k1 in e1:
                                    size = len([v for v in e1[k1].values()][0])

                                    for k2 in executors[k1]:
                                        for i in range(size):
                                            tmp = []
                                            for e2 in self._executors:
                                                if k1 in e2:
                                                    tmp.append(e2[k1][k2][i])
                                            executors[k1][k2].append(func(tmp))

                                    break

                        return executors
                    else:
                        executors = {}

                        for k in keys:
                            executors[k] = self._default_resources()

                            for e1 in self._executors[::-1]:
                                if k in e1:
                                    size = len([v for v in e1[k].values()][0])

                                    for key in executors[k]:
                                        for i in range(size):
                                            tmp = []
                                            for e2 in self._executors:
                                                if k in e2:
                                                    tmp.append(e2[k][key][i])
                                            executors[k][key].append(func(tmp))

                                    break

                        return executors
                else:
                    if key is None:
                        executors = self._default_resources()

                        for e1 in self._executors[::-1]:
                            if exec_id in e1:
                                size = len([v for v in e1[exec_id].values()][0])

                                for k in executors:
                                    for i in range(size):
                                        tmp = []
                                        for e2 in self._executors:
                                            if exec_id in e2:
                                                tmp.append(e2[exec_id][k][i])
                                        executors[k].append(func(tmp))

                                break

                        return executors
                    else:
                        executors = []

                        for e1 in self._executors[::-1]:
                            if exec_id in e1:
                                size = len(e1[exec_id][key])

                                for i in range(size):
                                    tmp = []
                                    for e2 in self._executors:
                                        if exec_id in e2:
                                            tmp.append(e2[exec_id][key][i])
                                    executors.append(func(tmp))

                                break

                        return executors
        else:
            if self.logger:
                self.logger('No measurement of executors resources has been done')
            return self._executors

    def export_times(self, filename, func=None, extension='csv'):
        if len(self._times):
            if func is None:
                times = self._times
            else:
                times = [self.get_times(func)]

            self._export_values(times, times[-1].keys(), filename, extension)
        else:
            if self.logger:
                self.logger('No measurement of time has been done')

    def export_rdd(self, filename, func, extension='csv'):
        if len(self._rdd):
            rdd = []

            for k, v in self.get_rdd(func).items():
                tmp = v.copy()
                tmp['rdd'] = k
                rdd.append(tmp)

            self._export_values(rdd, rdd[-1].keys(), filename, extension)
        else:
            if self.logger:
                self.logger('No measurement of RDD has been done')

    def export_resources(self, filename, func, extension='csv'):
        if len(self._resources):
            resources = []
            rsrc = self.get_resources(func)
            size = 0

            for k, v in rsrc.items():
                size = len(v)
                break

            for i in range(size):
                tmp = self._default_resources()

                for k in tmp:
                    tmp[k] = rsrc[k][i]

                resources.append(tmp)

            self._export_values(resources, resources[-1].keys(), filename, extension)
        else:
            if self.logger:
                self.logger('No measurement of general resources has been done')

    def export_executors(self, filename, func, extension='csv'):
        if len(self._executors):
            for k1, v1 in self.get_executors(func).items():
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

                self._export_values(executors, executors[-1].keys(), "{}_{}".format(filename, k1), extension)
        else:
            if self.logger:
                self.logger('No measurement of executors resources has been done')

    def plot_times(self, title, filename, func=None, **kwargs):
        if len(self._times):
            keys = self._times[-1].keys()
            index = np.arange(len(keys))
            width = 0.35

            fig, ax = plt.subplots()

            if func is None:
                if len(self._times) == 1:
                    times = []

                    for k in keys:
                        times.append(self._times[-1][k])

                    plt.bar(index, times, width, color='b')
                else:
                    mean_times = []
                    pstdev_times = []

                    for k, v in self.get_times(st.mean).items():
                        mean_times.append(v)

                    for k, v in self.get_times(st.pstdev).items():
                        pstdev_times.append(v)

                    plt.bar(index, mean_times, width, color='b', yerr=pstdev_times)
            else:
                times = []

                for k, v in self.get_times(func).items():
                    times.append(v)

                plt.bar(index, times, 0.35, color='b')

            plt.xlabel('Operations')
            plt.ylabel('Time (s)')
            plt.xticks(index, keys, rotation='vertical')
            plt.title(title)

            plt.tight_layout()
            plt.savefig(filename, kwargs=kwargs)
        else:
            if self.logger:
                self.logger('No measurement of time has been done')

    def plot_resources(self, title, filename, func=None, **kwargs):
        if len(self._resources):
            keys = self._default_resources().keys()

            for k in keys:
                x = np.linspace(1, len(self._resources[-1][k]), len(self._resources[-1][k]))
                break

            fig, ax = plt.subplots()

            if func is None:
                if len(self._resources) == 1:
                    for k, v in self._resources[-1].items():
                        plt.plot(x, v, marker='s', label=k)
                else:
                    resources = {}

                    for k, v in self.get_resources(st.mean).items():
                        resources[k] = {}
                        resources[k]['mean'] = v.copy()

                    for k, v in self.get_resources(st.pstdev).items():
                        resources[k]['pstdev'] = v.copy()

                    for k, v in resources.items():
                        plt.errorbar(x, v['mean'], yerr=v['pstdev'], marker='s', label=k)
            else:
                for k, v in self.get_resources(func).items():
                    plt.plot(x, v, marker='s', label=k)

            plt.xlabel('Measurements')
            plt.ylabel('Bytes')
            plt.xticks(x, x)
            plt.title(title)
            plt.legend()

            plt.tight_layout()
            plt.savefig(filename, kwargs=kwargs)
        else:
            if self.logger:
                self.logger('No measurement of general resources has been done')
