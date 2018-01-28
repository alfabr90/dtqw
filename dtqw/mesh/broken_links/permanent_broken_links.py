import random

from dtqw.mesh.broken_links.broken_links import BrokenLinks

__all__ = ['PermanentBrokenLinks']


class PermanentBrokenLinks(BrokenLinks):
    """Class for permanent broken links."""

    def __init__(self, spark_context, edges):
        """
        Build a permanent broken links object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        edges : collection
            Collection of the edges that are broken.
        """
        super().__init__(spark_context)

        if not len(edges):
            # self.logger.error('probability of broken links must be positive')
            raise ValueError('there must be at least one broken edge')

        self._edges = edges

    @property
    def edges(self):
        return self._edges

    def generate(self, num_edges):
        """
        Yield broken links for the mesh based on the informed broken edges.

        Returns
        -------
        RDD
            The RDD which keys are the numbered edges that are broken.

        """
        if isinstance(self._edges, range):
            if self._edges.start < 0 or self._edges.stop > num_edges:
                raise ValueError(
                    'invalid edges for broken links. Mesh supports edges from {} to {}'.format(
                        0, num_edges
                    )
                )

            rdd = self._spark_context.range(
                edges
            )
        else:
            if min(self._edges) < 0 or max(self._edges) >= num_edges:
                raise ValueError(
                    'invalid edges for broken links. Mesh supports edges from {} to {}'.format(
                        0, num_edges
                    )
                )

            rdd = self._spark_context.parallelize(
                edges
            )

        return rdd.map(
            lambda m: (m, True)
        )
