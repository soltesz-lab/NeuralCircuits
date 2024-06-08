import os, sys
import importlib
from collections import defaultdict
import yaml
from yaml.representer import Representer
yaml.add_representer(defaultdict, Representer.represent_dict)

class IncludeLoader(yaml.Loader):
    """
    YAML loader with `!include` handler.
    """

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        yaml.Loader.__init__(self, stream)

    def include(self, node):
        """

        :param node:
        :return:
        """
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, IncludeLoader)

    def envsubst(self, node):
        """

        :param node:
        :return:
        """

        s = self.construct_scalar(node)
        s = envsubst(s)
        return s


IncludeLoader.add_constructor('!include', IncludeLoader.include)
IncludeLoader.add_constructor('!envsubst', IncludeLoader.envsubst)

class ExplicitDumper(yaml.SafeDumper):
    """
    YAML dumper that will never emit aliases.
    """

    def ignore_aliases(self, data):
        return True

def write_to_yaml(file_path, data, default_flow_style=False):
    """

    :param file_path: str (should end in '.yaml')
    :param data: dict
    :param convert_scalars: bool
    :return:
    """
    with open(file_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=default_flow_style, Dumper=ExplicitDumper)


def read_from_yaml(file_path, include_loader=None):
    """

    :param file_path: str (should end in '.yaml')
    :return:
    """
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            if include_loader is None:
                Loader = yaml.FullLoader
            else:
                Loader = include_loader
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise IOError(f'read_from_yaml: invalid file_path: {file_path}')


def yaml_envsubst(full, val=None, initial=True):
    val = val or full if initial else val
    if isinstance(val, dict):
        for k, v in val.items():
            val[k] = yaml_envsubst(full, v, False)
    elif isinstance(val, list):
        for idx, i in enumerate(val):
            val[idx] = yaml_envsubst(full, i, False)
    elif isinstance(val, str):
        val = envsubst(val.format(**full))

    return val    


def import_object_by_path(path):
    module_path, _, obj_name = path.rpartition(".")
    print(path)
    print(module_path)
    print(obj_name)
    
    if module_path == "__main__" or module_path == "":
        module = sys.modules["__main__"]
    else:
        module = importlib.import_module(module_path)
    return getattr(module, obj_name)
