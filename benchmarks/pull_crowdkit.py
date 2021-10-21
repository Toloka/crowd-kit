import click
import os
import subprocess


HOME = os.path.expanduser('~')


def pull_crowdkit_from_repo(crowdkit_dir):
    print('pull from crowd-kit repository')
    subprocess.call(['git', 'checkout', 'main'], cwd=crowdkit_dir)
    subprocess.call(['git', 'pull'], cwd=crowdkit_dir)


@click.command()
@click.option('--crowdkit-dir')
def main(crowdkit_dir):
    pull_crowdkit_from_repo(crowdkit_dir)


if __name__ == '__main__':
    main()
