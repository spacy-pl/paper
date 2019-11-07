import json
from collections import defaultdict
import click


def make_flag_word_dict(splitted):
    words_dict = defaultdict(list)
    for word, flags in splitted:
        for f in flags:
            words_dict[f].append(word)
    return words_dict


def decode_and_split(lines):
    lines = [l.decode('iso-8859-2').strip() for l in lines]
    splitted = [(l.split('/')) if '/' in l else (l, "NO_FLAG")
                for l in lines]
    return splitted


@click.command()
@click.argument('ispell_all', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
def main(ispell_all, output):
    """Assigns lemmas to lemmatization rule flags"""

    with open(ispell_all, "rb") as f:
        lines = f.readlines()

    splitted = decode_and_split(lines)
    words_dict = make_flag_word_dict(splitted)

    with open(output, "w") as f:
        json.dump(words_dict, f, indent=4)


if __name__ == "__main__":
    main()
