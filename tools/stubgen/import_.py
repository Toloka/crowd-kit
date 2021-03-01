import os
import sys
from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, resolve_name, spec_from_file_location
from pkgutil import walk_packages
from typing import Optional, Tuple
from _frozen_importlib_external import _NamespaceLoader


class SourceFinder(MetaPathFinder):

    def __init__(self, module_root, sources_path):
        self.sources_path = sources_path
        self.module_root = module_root

    def find_spec(self, fullname, path, target=None):

        # To absolute import
        fullname = resolve_name(fullname, path)

        # Checking if fullname is module_root or its submodule
        if fullname == self.module_root:
            path_prefix = self.sources_path
        elif fullname.startswith(self.sources_path + '.'):
            tokens = fullname[len(self.sources_path) + 1]
            path_prefix = os.path.join(self.sources_path, *tokens)
        else:
            return None

        # Trying to guess a file
        if os.path.exists(path_prefix + '.py'):
            path = path_prefix + '.py'
        elif os.path.exists(os.path.join(path_prefix, '__init__.py')):
            path = os.path.join(path_prefix, '__init__.py')
        else:
            return None

        # Creating spec from a file
        return spec_from_file_location(fullname, path)


class VirtualPackageFinder(MetaPathFinder):

    def __init__(self, module_root):
        self.module_root = module_root

    def find_spec(self, fullname, path, target=None):
        if self.module_root.startswith(fullname + '.'):
            name = fullname.split('.')[0]
            loader = _NamespaceLoader(name, path, self)
            return ModuleSpec(name=name, loader=loader, is_package=True)

        return None


def override_module_import_path(module, sources_path):
    sys.meta_path.insert(0, SourceFinder(module, sources_path))
    sys.meta_path.append(VirtualPackageFinder(module))


def traverse_modules(module_root, sources_path):
    yield module_root, import_module(module_root)
    for _, module_name, _ in walk_packages([sources_path], prefix=module_root + '.'):
        yield module_name, import_module(module_name)


def split_qulaname(qualname) -> Tuple[str, Optional[str]]:
    """
    Splits a qualname into a module_name and a nested_name
    Args:
        qualname: a __qualname__ of an object

    Returns: A tuple of module_name and a nested_name where
        * module_name is the largest subqualname that resolves to a module
        * nested_name is the rest of the qualname or None if module_name is qualname
    """

    module_name = qualname
    while module_name:
        try:
            find_spec(module_name)
        except ImportError:
            module_name = module_name[:module_name.rfind('.')]
        else:
            break

    if not module_name:
        raise ValueError(f'Qualname {qualname:}')

    return module_name, qualname[len(module_name) + 1:] or None
