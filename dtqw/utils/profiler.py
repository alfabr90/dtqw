import csv
import json
import urllib.request
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import statistics as st
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from dtqw.utils.utils import create_dir

__all__ = ['Profiler']


class Profiler:
    resources_base1 = {'memoryUsed': 0, 'diskUsed': 0}
    resources_base2 = {'totalShuffleWrite': 0, 'totalShuffleRead': 0, 'diskUsed': 0, 'memoryUsed': 0}

    def __init__(self, base_url='http://localhost:4040/api/v1/'):
        self._base_url = base_url
        self._times = {}
        self._operators = {}
        self._executors = {}
        self._resources = {}

        self.logger = None

    @property
    def times(self):
        return self._times

    @property
    def operators(self):
        return self._operators

    @property
    def executors(self):
        return self._executors

    @property
    def resources(self):
        return self._resources

    def __str__(self):
        return self.__class__.__name__

    def to_string(self):
        return self.__str__()

    def _request(self, url_suffix=''):
        if self.logger:
            self.logger.info('performing request to "{}"...'.format(self._base_url + 'applications' + url_suffix))

        t1 = datetime.now()

        with urllib.request.urlopen(self._base_url + 'applications' + url_suffix) as response:
            result = response.read()

        if self.logger:
            self.logger.debug('request performed in {}s'.format((datetime.now() - t1).total_seconds()))

        if result is not None:
            result = json.loads(result.decode('utf-8'))
        else:
            if self.logger:
                self.logger.warning('the response is empty'.format(self._base_url + 'applications' + url_suffix))

        return result

    def get_applications(self):
        return self._request()

    def get_jobs(self, app_id):
        return self._request('/{}/jobs'.format(app_id))

    def get_job(self, app_id, job_id):
        return self._request('/{}/jobs/{}'.format(app_id, job_id))

    def get_stages(self, app_id):
        return self._request('/{}/stages'.format(app_id))

    def get_stage(self, app_id, stage_id):
        return self._request('/{}/stages/{}'.format(app_id, stage_id))

    def get_stageattempt(self, app_id, stage_id, stageattempt_id):
        return self._request('/{}/stages/{}/{}'.format(app_id, stage_id, stageattempt_id))

    def get_stageattempt_tasksummary(self, app_id, stage_id, stageattempt_id):
        return self._request('/{}/stages/{}/{}/taskSummary'.format(app_id, stage_id, stageattempt_id))

    def get_stageattempt_tasklist(self, app_id, stage_id, stageattempt_id):
        return self._request('/{}/stages/{}/{}/taskList'.format(app_id, stage_id, stageattempt_id))

    def get_executors(self, app_id):
        return self._request('/{}/executors'.format(app_id))

    def get_allexecutors(self, app_id):
        return self._request('/{}/allexecutors'.format(app_id))

    def get_rdds(self, app_id):
        return self._request('/{}/storage/rdd'.format(app_id))

    def get_rdd(self, app_id, rdd_id):
        return self._request('/{}/storage/rdd/{}'.format(app_id, rdd_id))

    def log_executors(self, data=None, app_id=None):
        if self.logger:
            if data is None:
                if app_id is None:
                    self.logger.error('expected an application id, not "{}"'.format(type(app_id)))
                    raise ValueError('expected an application id, not "{}"'.format(type(app_id)))
                data = self.get_executors(app_id)

            if len(data) > 0:
                self.logger.info("Printing executors data...")
                for d in data:
                    for k, v in d.items():
                        self.logger.info("{}: {}".format(k, v))
        else:
            print('No logger has been defined')

    def log_rdds(self, data=None, app_id=None):
        if self.logger:
            if data is None:
                if app_id is None:
                    self.logger.error('expected an application id, not "{}"'.format(type(app_id)))
                    raise ValueError('expected an application id, not "{}"'.format(type(app_id)))
                data = self.get_rdds(app_id)

            if len(data) > 0:
                self.logger.info("printing RDDs data...")
                for d in data:
                    for k, v in d.items():
                        if k != 'partitions':
                            self.logger.info("{}: {}".format(k, v))
        else:
            print('No logger has been defined')

    def log_rdd(self, data=None, app_id=None, rdd_id=None):
        if self.logger:
            if data is None:
                if app_id is None:
                    self.logger.error('expected an application id, not "{}"'.format(type(app_id)))
                    raise ValueError('expected an application id, not "{}"'.format(type(app_id)))
                if rdd_id is None:
                    self.logger.error('expected a RDD id, not "{}"'.format(type(rdd_id)))
                    raise ValueError('expected a RDD id, not "{}"'.format(type(rdd_id)))
                data = self.get_rdd(app_id, rdd_id)

            self.logger.info("printing RDD (id {}) data...".format(rdd_id))
            for k, v in data.items():
                if k != 'partitions':
                    self.logger.info("{}: {}".format(k, v))
        else:
            print('No logger has been defined')

    def profile_times(self, key, value):
        if not (key in self._times):
            self._times[key] = []
        self._times[key].append(value)

    def profile_operator(self, app_id, rdd_id, op_name):
        resources = Profiler.resources_base1

        if not (op_name in self._operators):
            self._operators[op_name] = {}

            for k in resources.keys():
                self._operators[op_name][k] = []

        if isinstance(rdd_id, (list, tuple)):
            for rdd in rdd_id:
                data = self.get_rdd(app_id, rdd)

                for k, v in data.items():
                    if k in resources.keys():
                        resources[k] += v

                for k, v in resources.items():
                    self._operators[op_name][k].append(v)
        else:
            data = self.get_rdd(app_id, rdd_id)

            for k, v in data.items():
                if k in resources.keys():
                    self._operators[op_name][k].append(v)

    def profile_resources(self, app_id):
        data = self.get_executors(app_id)

        if len(data) > 0:
            resources = Profiler.resources_base2

            for k in resources.keys():
                if not (k in self._resources):
                    self._resources[k] = []

            for d in data:
                for k, v in d.items():
                    if k in resources.keys():
                        resources[k] += d['id'][k]

            for k, v in resources.items():
                self._resources[k].append(v)

    def profile_executors(self, app_id):
        data = self.get_executors(app_id)

        if len(data) > 0:
            resources = Profiler.resources_base2.keys()

            for d in data:
                if not (d['id'] in self._executors):
                    self._executors[d['id']] = {}
                    for k in resources:
                        self._executors[d['id']][k] = []

                for k, v in d.items():
                    if k in resources:
                        self._executors[d['id']][k].append(d['id'][k])

    def get_last_time(self, key):
        if not (key in self._times):
            return None
        return self._times[key][-1]

    def get_last_operator(self, op_name, key=None):
        if not (op_name in self._operators):
            return None

        if key is None:
            resources = Profiler.resources_base1

            for k in resources.keys():
                resources[k] = self._operators[op_name][k][-1]

            return resources
        else:
            if not (key in self._operators[op_name][key]):
                return None

            return self._operators[op_name][key][-1]

    def get_last_resource(self, key=None):
        if key is None:
            resources = Profiler.resources_base2

            for k in resources.keys():
                resources[k] = self._resources[k][-1]

            return resources
        else:
            if not (key in self._resources[key]):
                return None

            return self._resources[key][-1]

    def get_last_executor(self, exec_id, key=None):
        if not (exec_id in self._executors):
            return None

        if key is None:
            resources = Profiler.resources_base2

            for k in resources.keys():
                resources[k] = self._executors[exec_id][k][-1]

            return resources
        else:
            if not (key in self._executors[exec_id][key]):
                return None

            return self._executors[exec_id][key][-1]

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
