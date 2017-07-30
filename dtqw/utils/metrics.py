import urllib.request
import json

from .logger import Logger

__all__ = ['Metrics']


class Metrics:
    def __init__(self, base_url='http://localhost:4040/api/v1/', log_filename='log.txt'):
        self.__base_url = base_url
        self.__logger = Logger(__name__, log_filename)

    def __request(self, url_suffix=''):
        with urllib.request.urlopen(self.__base_url + 'applications' + url_suffix) as response:
            result = response.read()

        if result is not None:
            result = json.loads(result.decode('utf-8'))

        return result

    def get_applications(self):
        return self.__request()

    def get_jobs(self, app_id):
        return self.__request('/{}/jobs'.format(app_id))

    def get_job(self, app_id, job_id):
        return self.__request('/{}/jobs/{}'.format(app_id, job_id))

    def get_stages(self, app_id):
        return self.__request('/{}/stages'.format(app_id))

    def get_stage(self, app_id, stage_id):
        return self.__request('/{}/stages/{}'.format(app_id, stage_id))

    def get_stageattempt(self, app_id, stage_id, stageattempt_id):
        return self.__request('/{}/stages/{}/{}'.format(app_id, stage_id, stageattempt_id))

    def get_stageattempt_tasksummary(self, app_id, stage_id, stageattempt_id):
        return self.__request('/{}/stages/{}/{}/taskSummary'.format(app_id, stage_id, stageattempt_id))

    def get_stageattempt_tasklist(self, app_id, stage_id, stageattempt_id):
        return self.__request('/{}/stages/{}/{}/taskList'.format(app_id, stage_id, stageattempt_id))

    def get_executors(self, app_id):
        return self.__request('/{}/executors'.format(app_id))

    def get_allexecutors(self, app_id):
        return self.__request('/{}/allexecutors'.format(app_id))

    def get_rdds(self, app_id):
        return self.__request('/{}/storage/rdd'.format(app_id))

    def get_rdd(self, app_id, rdd_id):
        return self.__request('/{}/storage/rdd/{}'.format(app_id, rdd_id))

    def log_executors(self, data=None, app_id=None):
        if data is None:
            if app_id is None:
                raise ValueError("expected an application id")
            data = self.get_executors(app_id)

        self.__logger.info("Printing executors data...")
        if len(data) > 0:
            for d in data:
                for k, v in d.items():
                    self.__logger.info("{}: {}".format(k, v))
        else:
            self.__logger.info("No executors found")

    def log_rdds(self, data=None, app_id=None):
        if data is None:
            if app_id is None:
                raise ValueError("expected an application id")
            data = self.get_rdds(app_id)

        self.__logger.info("Printing RDDs data...")
        if len(data) > 0:
            for d in data:
                for k, v in d.items():
                    if k != 'partitions':
                        self.__logger.info("{}: {}".format(k, v))
        else:
            self.__logger.info("No RDD found")

    def log_rdd(self, data=None, app_id=None, rdd_id=None):
        if data is None:
            if app_id is None or rdd_id is None:
                raise ValueError("expected an application id and a RDD id")
            data = self.get_rdd(app_id, rdd_id)

        self.__logger.info("Printing RDD (id {}) data...".format(rdd_id))
        for k, v in data.items():
            if k != 'partitions':
                self.__logger.info("{}: {}".format(k, v))
