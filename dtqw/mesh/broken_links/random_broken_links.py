import random

from dtqw.mesh.broken_links.broken_links import BrokenLinks
from dtqw.utils.utils import Utils

__all__ = ['RandomBrokenLinks']


class RandomBrokenLinks(BrokenLinks):
    """Class for generating random broken links for a mesh."""

    def __init__(self, spark_context, probability):
        """
        Build a random broken links generator object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        probability : float
            Probability of the occurences of broken links in the mesh.
        """
        super().__init__(spark_context)

        if probability <= 0:
            # self.logger.error('probability of broken links must be positive')
            raise ValueError('probability of broken links must be positive')

        self._probability = probability

    @property
    def probability(self):
        return self._probability

    def generate(self, num_edges):
        """
        Yield broken links for the mesh based on its probability to have a broken link/edge.

        Returns
        -------
        RDD or Broadcast
            The RDD or Broadcast dict which keys are the numbered edges that are broken.

        Raises
        ------
        ValueError

        """
        probability = self._probability
        seed = Utils.getConf(self._spark_context, 'dtqw.randomBrokenLinks.seed', default=None)

        def __map(e):
            random.seed(seed)
            return e, random.random() < probability

        rdd = self._spark_context.range(
            num_edges
        ).map(
            __map
        ).filter(
            lambda m: m[1] is True
        )

        generation_mode = Utils.getConf(self._spark_context, 'dtqw.mesh.brokenLinks.generationMode', default='rdd')

        if generation_mode == 'rdd':
            return rdd
        elif generation_mode == 'broadcast':
            return rdd.collectAsMap()
        else:
            raise ValueError("invalid broken links generation mode")
