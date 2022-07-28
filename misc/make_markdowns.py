#!/usr/bin/env python3

import importlib.util
import inspect
import os
import sys
import logging
import json
from argparse import ArgumentParser
from stubmaker.builder import override_module_import_path, traverse_modules
from stubmaker.builder.common import BaseDefinition

from stubmaker.builder.representations_tree_builder import RepresentationsTreeBuilder
from stubmaker.viewers.markdown_viewer import MarkdownViewer


class GitHubSourceFinder:
    def __init__(self, source_root_url, handle_unknown_source='raise'):
        self.handle_unknown_source = handle_unknown_source
        self.source_root_url = source_root_url

    def __call__(self, definition: BaseDefinition):
        definition_module = inspect.getmodule(definition.obj)

        # handle package.__init__.py files
        if definition_module.__name__.split('.')[-1] != definition_module.__file__.split('/')[-1].replace('.py', ''):
            definition_module_name = definition.module_name + '.__init__'
        else:
            definition_module_name = definition.module_name

        file_path = '/'.join(definition_module_name.split('.')[1:]) + '.py'
        try:
            obj = definition.obj
            while hasattr(obj, '__wrapped__'):
                obj = obj.__wrapped__
            line = inspect.findsource(obj)[1]
        except OSError as exc:
            msg = f"Can't find source for {definition.obj}"
            if self.handle_unknown_source == 'ignore':
                logging.warning(msg)
                return None
            raise RuntimeError(msg) from exc
        return f'{self.source_root_url}/{file_path}#L{line}'


def make_markdowns(
    module_root, src_root, output_dir, skip_modules, modules_aliases_mapping,
    github_source_url, described_objects,
):
    # Making sure our module is imported from provided src-root even
    # another version of the module is installed in the system
    override_module_import_path(module_root, src_root)

    for module_name, module in traverse_modules(module_root, src_root, skip_modules=skip_modules):
        # Normalizing paths before comparison and ensuring dst directory's existence
        dst_dir = os.path.abspath(output_dir)
        os.makedirs(dst_dir, exist_ok=True)

        builder = RepresentationsTreeBuilder(
            module_name, module, module_root, modules_aliases_mapping=modules_aliases_mapping,
            described_objects=described_objects, preserve_forward_references=False,
        )
        viewer = MarkdownViewer(
            source_link_finder=GitHubSourceFinder(
                github_source_url, handle_unknown_source='ignore'
            ) if github_source_url else None
        )

        for name, markdown in viewer.get_markdown_files_for_module(builder.module_rep):

            dst_path = f'{dst_dir}/{name}.md'
            logging.info(f'Creating: {dst_path}')
            with open(dst_path, 'w') as md_flo:
                md_flo.write(markdown)


def main():
    parser = ArgumentParser()
    parser.add_argument('--module-root', type=str, required=True, help='Module name to import these sources as')
    parser.add_argument('--src-root', type=os.path.abspath, required=True, help='Path to source files to process')
    parser.add_argument('--output-dir', type=os.path.abspath, required=True)
    parser.add_argument(
        '--skip-modules', type=str, nargs='*', help='Module names to skip when generating markdowns',
        default=['crowdkit.datasets._base', 'crowdkit.datasets._loaders'],
    )
    parser.add_argument(
        '--modules-aliases', type=os.path.abspath, required=False,
        help='Path to module names to aliases mapping in json format.',
    )
    parser.add_argument(
        '--described-objects', type=os.path.abspath, required=False,
        help='A dictionary from python objects to tuples consisting of module and qualname for each object (e.g. '
             "ModuleType: (types, ModuleType). Such objects' names and modules will not be deduced based on runtime "
             'data and provided names and modules will be used instead. Provided python file must have '
             'DESCRIBED_OBJECTS dictionary on the module level.',
    )
    parser.add_argument(
        '--logging-level', type=lambda x: str(x).upper(), required=False,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='WARN',
    )
    parser.add_argument(
        '--github-source-url', type=str, required=False,
        help='Url of GitHub source root. If provided links to source code will be generated.',
    )

    args = parser.parse_args()

    logging.basicConfig(format='%(message)s', level=getattr(logging, args.logging_level), stream=sys.stderr)

    if args.described_objects:
        spec = importlib.util.spec_from_file_location(
            "described_objects", os.path.join(os.getcwd(), args.described_objects)
        )
        described_objects = importlib.util.module_from_spec(spec)
        sys.modules["described_objects"] = described_objects
        spec.loader.exec_module(described_objects)
        described_objects = getattr(described_objects, 'DESCRIBED_OBJECTS', None)
        if described_objects is None:
            raise RuntimeError('DESCRIBED_OBJECTS dict should be defined in python file specified in described-objects')

    make_markdowns(
        module_root=args.module_root,
        src_root=args.src_root,
        output_dir=args.output_dir,
        skip_modules=args.skip_modules,
        modules_aliases_mapping=args.modules_aliases and json.load(open(args.modules_aliases)),
        described_objects=args.described_objects and described_objects,
        github_source_url=args.github_source_url,
    )


if __name__ == '__main__':
    main()
