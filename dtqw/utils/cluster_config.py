
MUL_MODE_BROADCAST = 0
MUL_MODE_BLOCK = 1

class ClusterConfig:
    def __init__(self, num_executor, driver_conf, executor_conf,
                 memory_fraction=0.8, storage_fraction=0.6, partitions_multiplier=2):
        self.__partitions_multiplier = partitions_multiplier
        self.__num_executor = num_executor
        self.__memory_fraction = memory_fraction
        self.__storage_fraction = storage_fraction

        self.__driver_conf = {
            'cores': 0,
            'memory': 0
        }
        self.__executor_conf = {
            'cores': 0,
            'memory': 0
        }

        for k, v in driver_conf.items():
            self.__driver_conf[k] = v
        for k, v in executor_conf.items():
            self.__executor_conf[k] = v

    def get_num_partitions(self):
        return 8

    def get_multiplication_mode(self):
        return MUL_MODE_BROADCAST

    def get_num_blocks(self):
        return 2
