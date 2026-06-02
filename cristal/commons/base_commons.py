"""Contains Base class for all commons classes. Useful to bind configuration dependencies to the class."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.detector_config import DetectorConfig


class BaseCommons:
    """Base class for all commons classes. Useful to bind configuration dependencies to the class.

    Attributes
    ----------
    requires : list[str]
        List of required dependency names

    Examples
    --------
    >>> class Test(BaseCommons):
    >>>     requires = ["dep1", "dep2"]
    >>> test = Test()
    >>> test.bind(config)  # Binds the dependencies to the object. Must contain "dep1" AND "dep2".
    >>> test.dep1  # Accesses the bound dependency.
    """

    requires: list[str] = []  #: List of required dependency names
    _dependencies = None  #: Internal storage for dependencies

    def bind(self, config: "DetectorConfig"):
        """Bind the dependencies in :attr:`requires` to the class so that the object can access them.

        Parameters
        ----------
        config : DetectorConfig
            The configuration containing the dependencies.

        Raises
        ------
        ValueError
            If a dependency required by the class is missing in :attr:`config`.

        Examples
        --------
        >>> class Test(BaseCommons):
        >>>     requires = ["dep1", "dep2"]
        >>> test = Test()
        >>> test.bind(config)  # Binds the dependencies to the object. Must contain "dep1" AND "dep2".
        >>> test.dep1  # Accesses the bound dependency.
        """
        self._dependencies = {}

        for dep_name in self.requires:
            if not hasattr(config, dep_name):
                raise ValueError(f"Missing dependency: {dep_name} for {self.__class__.__name__}.")

            self._dependencies[dep_name] = getattr(config, dep_name)

    def __getattr__(self, name):
        # If no attribute is bound
        if self._dependencies is None and name in self.requires:
            return None
        # Allows direct access to self.dependencies by name
        if self._dependencies is not None and name in self._dependencies:
            return self._dependencies[name]
        raise AttributeError(name)
