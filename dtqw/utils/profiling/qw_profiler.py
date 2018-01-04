from dtqw.utils.profiling.profiler import Profiler
from dtqw.math.operator import is_operator
from dtqw.math.state import is_state
from dtqw.math.statistics.cdf import is_cdf

__all__ = ['QWProfiler']


class QWProfiler(Profiler):
    """Profile and export the resources consumed by Spark for Quantum Walks."""

    def __init__(self, base_url='http://localhost:4040/api/v1/'):
        """
        Build a Quantum Walk Profiler object.

        Parameters
        ----------
        base_url: str, optional
            The base URL for getting information about the resources consumed. Default is http://localhost:4040/api/v1/.
        """
        super().__init__(base_url)

        self._operators = None
        self._states = None
        self._cdfs = None

    @staticmethod
    def _default_operator():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0}

    @staticmethod
    def _default_state():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0, 'numElements': 0, 'numNonzeroElements': 0}

    @staticmethod
    def _default_cdf():
        return {'buildingTime': 0.0, 'diskUsed': 0, 'memoryUsed': 0, 'numElements': 0, 'numNonzeroElements': 0}

    def start(self):
        """Reset the profiler attributes to get info for a new profiling round."""
        super().start()

        self._operators = {}
        self._states = {}
        self._cdfs = {}

    def profile_operator(self, name, operator, time):
        """
        Store building time and resources information for a named quantum walk operator.

        Parameters
        ----------
        name : str
            A name for the operator.
        operator : :obj:Operator
            The operator object.
        time : float
            The measured building time of the operator.

        Returns
        -------
        dict
            The resources information measured for the operator.

        Raises
        -----
        TypeError

        """
        if self.logger:
            self.logger.info('profiling operator data for "{}"...'.format(name))

        if not is_operator(operator):
            if self._logger:
                self._logger.error('Operator instance expected, not "{}"'.format(type(operator)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(operator)))

        if name not in self._operators:
            self._operators[name] = []

        self._operators[name].append(self._default_operator())

        app_id = operator.spark_context.applicationId
        rdd_id = operator.data.id()
        data = self.request_rdd(app_id, rdd_id)

        if data is not None:
            for k, v in data.items():
                if k in self._default_operator():
                    self._operators[name][-1][k] = v

        self._operators[name][-1]['buildingTime'] = time

        return self._operators[name][-1]

    def profile_state(self, name, state, time):
        """
        Store building time and resources information for a named quantum walk system state.

        Parameters
        ----------
        name : str
            A name for the state.
        state : :obj:State
            The operator state.
        time : float
            The measured building time of the state.

        Returns
        -------
        dict
            The resources information measured for the state.

        Raises
        -----
        TypeError

        """
        if self._logger:
            self._logger.info('profiling quantum system state data for "{}"...'.format(name))

        if not is_state(state):
            if self._logger:
                self._logger.error('State instance expected, not "{}"'.format(type(state)))
            raise TypeError('State instance expected, not "{}"'.format(type(state)))

        if name not in self._states:
            self._states[name] = []

        self._states[name].append(self._default_state())

        app_id = state.spark_context.applicationId
        rdd_id = state.data.id()
        data = self.request_rdd(app_id, rdd_id)

        if data is not None:
            for k, v in data.items():
                if k in self._default_state():
                    self._states[name][-1][k] = v

        self._states[name][-1]['buildingTime'] = time

        return self._states[name][-1]

    def profile_cdf(self, name, cdf, time):
        """
        Store building time and resources information for a named measurement (CDF).

        Parameters
        ----------
        name : str
            A name for the cdf.
        cdf : :obj:CDF
            The cdf object.
        time : float
            The measured building time of the cdf.

        Returns
        -------
        dict
            The resources information measured for the cdf.

        Raises
        -----
        TypeError

        """
        if self._logger:
            self._logger.info('profiling CDF data for "{}"...'.format(name))

        if not is_cdf(cdf):
            if self._logger:
                self._logger.error('CDF instance expected, not "{}"'.format(type(cdf)))
            raise TypeError('CDF instance expected, not "{}"'.format(type(cdf)))

        if name not in self._cdfs:
            self._cdfs[name] = []

        self._cdfs[name].append(self._default_cdf())

        app_id = cdf.spark_context.applicationId
        rdd_id = cdf.data.id()
        data = self.request_rdd(app_id, rdd_id)

        if data is not None:
            for k, v in data.items():
                if k in self._default_cdf():
                    self._cdfs[name][-1][k] = v

        self._cdfs[name][-1]['buildingTime'] = time

        return self._cdfs[name][-1]

    def get_operators(self, name=None):
        """
        Get all operators' resources information.

        Parameters
        ----------
        name : str, optional
            The name used for a operator.

        Returns
        -------
        dict or list
            A dict with all operators' resources information or a list of the provided operator's resources information.

        """
        if len(self._operators):
            if name is None:
                return self._operators.copy()
            else:
                if name not in self._operators:
                    if self.logger:
                        self.logger.warning('no resources information for operator "{}"'.format(name))
                    return {}
                return self._operators[name]
        else:
            if self.logger:
                self.logger.warning('no resources information for operators have been gotten')
            return {}

    def get_states(self, name=None):
        """
        Get all states' resources information.

        Parameters
        ----------
        name : str, optional
            The name used for a operator.

        Returns
        -------
        dict or list
            A dict with all states' resources information or a list of the provided state's resources information.

        """
        if len(self._states):
            if name is None:
                return self._states.copy()
            else:
                if name not in self._states:
                    if self.logger:
                        self.logger.warning('no resources information for state "{}"'.format(name))
                    return {}
                return self._states[name]
        else:
            if self.logger:
                self.logger.warning('no resources information for states have been gotten')
            return {}

    def get_cdfs(self, name=None):
        """
        Get all cdfs' resources information.

        Parameters
        ----------
        name : str, optional
            The name used for a operator.

        Returns
        -------
        dict or list
            A dict with all cdfs' resources information or a list of the provided cdf's resources information.

        """
        if len(self._cdfs):
            if name is None:
                return self._cdfs.copy()
            else:
                if name not in self._cdfs:
                    if self.logger:
                        self.logger.warning('no resources information for cdf "{}"'.format(name))
                    return {}
                return self._cdfs[name]
        else:
            if self.logger:
                self.logger.warning('no resources information for cdfs have been gotten')
            return {}

    def export_operators(self, path, extension='csv'):
        """
        Export all stored operators' resources.

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
            self.logger.info("exporting operators' resources in {} format...".format(extension))

        if len(self._operators):
            operator = []

            for k, v in self._operators.items():
                for i in v:
                    operator.append(i.copy())
                    operator[-1]['name'] = k

            self._export_values(operator, operator[-1].keys(), path + 'operators', extension)

            if self.logger:
                self.logger.info("operator's resources successfully exported")
        else:
            if self.logger:
                self.logger.warning("no measurement of operators' resources has been done")

    def export_states(self, path, extension='csv'):
        """
        Export all stored state' resources.

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
            self.logger.info("exporting states' resources in {} format...".format(extension))

        if len(self._states):
            states = []

            for k, v in self._states.items():
                for i in v:
                    states.append(i.copy())
                    states[-1]['name'] = k

            self._export_values(states, states[-1].keys(), path + 'states', extension)

            if self.logger:
                self.logger.info("states' resources successfully exported")
        else:
            if self.logger:
                self.logger.warning("no measurement of states' resources has been done")

    def export_cdfs(self, path, extension='csv'):
        """
        Export all stored cdfs' resources.

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
            self.logger.info("exporting cdfs' resources in {} format...".format(extension))

        if len(self._operators):
            cdfs = []

            for k, v in self._cdfs.items():
                for i in v:
                    cdfs.append(i.copy())
                    cdfs[-1]['name'] = k

            self._export_values(cdfs, cdfs[-1].keys(), path + 'cdfs', extension)

            if self.logger:
                self.logger.info("cdfs' resources successfully exported")
        else:
            if self.logger:
                self.logger.warning("no measurement of cdfs' resources has been done")

    def export(self, path, extension='csv'):
        """
        Export all stored profiling information of quantum walks.

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
        super().export(path, extension)

        self.export_operators(path, extension)
        self.export_states(path, extension)
        self.export_cdfs(path, extension)
