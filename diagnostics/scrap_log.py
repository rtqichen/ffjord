import os
import re
import csv


def log_to_csv(log_filename, csv_filename):
    with open(log_filename, 'r') as f:
        lines = f.readlines()

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = None
        writer = None

        for line in lines:
            if line.startswith('Iter'):
                # A dictionary of quantity : value.
                quants = _line_to_dict(line)
                # Create writer and write header.
                if fieldnames is None:
                    fieldnames = quants.keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                # Write a line.
                writer.writerow(quants)


def _line_to_dict(line):
    line = re.sub(':', '', line)  # strip colons.
    line = re.sub('\([^)]*\)', '', line)  # strip running averages.

    quants = {}
    for quant_str in line.split('|'):
        quant_str = quant_str.strip()  # strip beginning and ending whitespaces.
        key, val = quant_str.split(' ')
        quants[key] = val

    return quants


def plot_pairplot(csv_filename, fig_filename, top=None):
    import seaborn as sns
    import pandas as pd

    sns.set(style="ticks", color_codes=True)
    quants = pd.read_csv(csv_filename)
    if top is not None:
        quants = quants[:top]

    g = sns.pairplot(quants, kind='reg', diag_kind='kde', markers='.')
    g.savefig(fig_filename)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    parser.add_argument('--top_iters', type=int, default=None)
    args = parser.parse_args()

    print('Parsing log into csv.')
    log_to_csv(args.log, args.log + '.csv')
    print('Creating correlation plot.')
    plot_pairplot(args.log + '.csv', os.path.join(os.path.dirname(args.log), 'quants.png'), args.top_iters)
