import click
import json
from maps.flag_to_pos import MAP


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
def main(input_file, output_file):
    with open(input_file, "r") as f:
        dict_to_convert = json.load(f)

    result_dict = {}
    for flag, value in dict_to_convert.items():
        try:
            pos = MAP[flag]
            result_dict[pos] = value
        except KeyError:
            print(f"No mapping for flag {flag}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)


if __name__ == '__main__':
    main()
