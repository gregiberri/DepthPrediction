#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import multiprocessing
from types import SimpleNamespace
from warnings import warn

import yaml

yaml.Dumper.ignore_aliases = lambda *args: True


class ConfigNameSpace(SimpleNamespace):
    def __init__(self, **kwargs):
        """
        Make config namespace from dict.
        """
        for kwarg in kwargs.items():
            if isinstance(kwarg[1], dict):
                self.__dict__.update({kwarg[0]: ConfigNameSpace(**kwarg[1])})
            else:
                self.__dict__.update({kwarg[0]: kwarg[1]})

    def __getattr__(self, name):
        """
        Overwrite getattr: return None for nonexistent attributes.
        """
        try:
            super(ConfigNameSpace, self).__getattr__()
        except AttributeError:
            logging.warning(f'The required config attribute: `{name}` does not exist, returned `None` instead.')
            return None

    def __len__(self):
        return len(self.__dict__)

    def dict(self):
        """
        Make dict from the namespace
        """
        config_dict = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNameSpace):
                subdict = v.dict()
                config_dict[k] = subdict
            else:
                config_dict[k] = v
        return config_dict

    def load(self, path):
        """
        Load namespace from yaml file
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        self.__init__(**config)
        return self

    def save(self, path):
        """
        Save namespace into yaml file.
        """
        assert path[-3:] == 'yml' or path[-4:] == 'yaml', 'The file should be a yml file: *.yml or *.yaml'

        with open(path, 'w') as nf:
            config_dict = self.dict()
            yaml.dump(config_dict, nf)

    def str(self, step='  '):
        """
        Make string from the namespace for easier printing
        """
        object_str = ''
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNameSpace):
                new_step = step + '  '
                object_str += step + k + ': \n' + v.str(new_step)
            else:
                object_str += step + k + ': ' + str(v) + '\n'

        return object_str

    def __repr__(self):
        return 'config: \n' + self.str()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def update(self, other):
        if isinstance(other, ConfigNameSpace):
            self.__dict__.update(other.__dict__)
        elif isinstance(other, dict):
            self.__dict__.update(other)
        else:
            raise ValueError('other object used for update should be Namespace of dict.')