class BaseCommons:
    requires = []  #: List of required dependency names
    _dependencies = None  #: Internal storage for dependencies

    def bind(self, config):
        self._dependencies = {}

        for dep_name in self.requires:
            if not hasattr(config, dep_name):
                raise ValueError(f"Missing dependency: {dep_name} for {self.__class__.__name__}")

            self._dependencies[dep_name] = getattr(config, dep_name)

    def __getattr__(self, name):
        # Allows direct access to self.dependencies by name
        if self._dependencies is not None and name in self._dependencies:
            return self._dependencies[name]
        raise AttributeError(name)
