import click
import json
import os
import subprocess

from glob import glob

HOME = os.path.expanduser('~')


def pull_results_from_repo(results_dir):
    print('pull from results repository')
    subprocess.call(['git', 'checkout', 'gh-pages'], cwd=results_dir)
    subprocess.call(['git', 'pull'], cwd=results_dir)


def load_results_from_repo(asv_dir, results_dir):
    print('load results from repository')
    subprocess.call(['rm', '-r', asv_dir])
    subprocess.call(['mkdir', asv_dir])
    subprocess.call(['cp', '-r', os.path.join(results_dir, 'results'), asv_dir])


def recreate_machine_config(asv_dir):
    print('recreating machine config')
    with open(os.path.join(glob(os.path.join(asv_dir, 'results', '*[!.json]'))[0], 'machine.json')) as f:
        machine_config = json.load(f)

    machine_config = {
        'version': machine_config.pop('version'),
        machine_config['machine']: machine_config
    }
    with open(os.path.join(HOME, '.asv-machine.json'), 'w') as f:
        json.dump(machine_config, f)


def run_asv(bench_dir):
    print('running asv')
    subprocess.call(['asv', 'run', 'NEW'], cwd=bench_dir)
    subprocess.call(['asv', 'publish'], cwd=bench_dir)


def update_results_repo(asv_dir, results_dir):
    print('push results to repository')
    subprocess.call([f'cp -r {os.path.join(asv_dir, "html", "*")} {results_dir}'], shell=True)
    subprocess.call(['cp', '-r', os.path.join(asv_dir, 'results'), results_dir])
    subprocess.call(['git', 'add',  '.'], cwd=results_dir)
    subprocess.call(['git', 'commit', '-m', '"new results"'], cwd=results_dir)
    subprocess.call(['git', 'push'], cwd=results_dir)


@click.command()
@click.option('--crowdkit-dir')
@click.option('--results-dir')
def main(crowdkit_dir, results_dir):
    bench_dir = os.path.join(crowdkit_dir, 'benchmarks')
    asv_dir = os.path.join(bench_dir, '.asv')

    pull_results_from_repo(results_dir)
    load_results_from_repo(asv_dir, results_dir)
    recreate_machine_config(asv_dir)
    run_asv(bench_dir)
    update_results_repo(asv_dir, results_dir)


if __name__ == '__main__':
    main()
