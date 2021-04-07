from argparse import ArgumentParser

from stubgen.ast_builder import ASTBuilder
from stubgen.import_ import override_module_import_path, traverse_modules


def main():
    parser = ArgumentParser()
    parser.add_argument('--src-root', type=str, required=True, help='Path to source files to process')
    parser.add_argument('--module-root', type=str, required=False, help='Module name to import these sources as')
    args = parser.parse_args()

    override_module_import_path(args.module_root, args.src_root)

    for module_name, module in traverse_modules(args.module_root, args.src_root):

        stub_path = module.__file__ + 'i'
        with open(stub_path, 'w') as stub_flo:
            stub_flo.write(str(ASTBuilder(module_name, module)))


if __name__ == '__main__':
    main()
