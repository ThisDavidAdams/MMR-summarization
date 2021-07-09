"""
@author: David Adams
"""
import os
import jsonlines
import argparse


def main():
    # Define arguments
    parser = argparse.ArgumentParser(description='DEC-to-JSONL Converter')
    parser.add_argument('--input', type=str, default='multinews_output/dec/',
                        help='The folder containing all DEC files to be converted to a JSONL file.'
                             'This path must end with a / character.')
    parser.add_argument('--output', type=str, default='matchsum_output.jsonl',
                        help='JSONL file to create by converting all the DEC files.'
                             'Should be placed in matchusm_output folder for simple use of other scripts.')
    args = parser.parse_args()

    # Path of dec input files.
    input_path = args.input

    # List of files in input path.
    filenames = os.listdir(input_path)
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))

    # Output file.
    open('matchsum_output.jsonl', "w+").close()
    sd = jsonlines.open('matchsum_output.jsonl', 'w')

    # For each file in the input path:
    for file in sorted_filenames:
        # Open the next input file and write it to the output file.
        input_file = open(input_path + file, 'r', encoding='utf-8')
        sd.write({"ext_summary": input_file.read()})


if __name__ == "__main__":
    main()
