import json
import os
from collections import defaultdict
import click


def get_rule(aff_code):
    lemma_suf, ends = aff_code.split('>')
    if ',' in ends:
        lemma_end, word_end = ends.split(',')
        lemma_end = lemma_end.strip()[1:]
        word_end = word_end.strip()
        word_suf = lemma_suf[:-len(lemma_end)] + word_end
    else:
        word_end = ends
        word_suf = lemma_suf + word_end
    return [word_suf, lemma_suf]


def read_flag(lines):
    flag = "NO_FLAG"
    for aff_code, comment in lines:
        if aff_code.startswith('flag*'):
            flag = aff_code[5]
        else:
            yield flag, aff_code, comment


def extract_information(lines):
    lines = [l.strip() for l in lines]
    lines = skip_uninteresting(lines)
    lines = [l.replace('\t', '') for l in lines]
    # remove empty
    lines = [l for l in lines if l.replace(' ', '') != '']
    lines = [split_on_comment(line) for line in lines]
    lines = [l for l in lines if l[0] != '' or l[1] != '']
    return lines


def skip_uninteresting(lines):
    i = 0
    while "suffixes" not in lines[i]:
        i += 1
    return lines[(i+1):]


def split_on_comment(line):
    if line[0] == '#':
        comment = line[1:]
        aff_code = ''
    else:
        splitted = line.split("#")
        aff_code = splitted[0]
        if len(splitted) > 1:
            comment = splitted[1]
        else:
            comment = ''
    return aff_code.strip().replace(' ', ''), comment.strip()


def parse_aff_lines(lines):
    processed = extract_information(lines)

    rule_groups = defaultdict(list)
    comments = defaultdict(list)
    for flag, aff_code, comment in read_flag(processed):
        # case: comment line
        if aff_code == '' and comment != '':
            comments[flag].append(comment)
        # case: rule line
        elif aff_code != '':
            rule = get_rule(aff_code)
            rule_groups[flag].append(rule)
        else:
            raise Exception(
                "Unexpected empty line! (on flag: {})".format(flag))

    return rule_groups, comments


@click.command()
@click.argument('ispell_rules', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
def main(ispell_rules, output):
    """Extract rules from ispell aff file to more convenient form"""

    with open(ispell_rules, "rb") as f:
        lines = f.readlines()
    lines = [l.decode('iso-8859-2') for l in lines]

    rule_groups, comments = parse_aff_lines(lines)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(rule_groups, f, indent=4)


if __name__ == '__main__':
    main()
