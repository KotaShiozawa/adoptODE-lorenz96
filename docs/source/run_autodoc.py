#!/usr/bin/env python3

import glob
import logging
import os
import subprocess as sp

logging.basicConfig(level=logging.INFO)

OUTPUT_PATH = "packages"
# REPLACEMENTS = {
#     "Submodules\n----------\n\n": "",
#     "Subpackages\n-----------": "**Subpackages:**",
#     "drrc package\n============": "Reference manual\n================",
# }
REPLACEMENTS = {
    "Submodules\n----------\n\n": "",
    "Subpackages\n-----------": "",
    "adoptODE package\n============": "adoptODE\n====",
}


def replace_in_file(infile, replacements, outfile=None):
    """reads in a file, replaces the given data using python formatting and
    writes back the result to a file.

    Args:
        infile (str):
            File to be read
        replacements (dict):
            The replacements old => new in a dictionary format {old: new}
        outfile (str):
            Output file to which the data is written. If it is omitted, the
            input file will be overwritten instead

    """
    if outfile is None:
        outfile = infile

    with open(infile, "r", encoding="UTF-8") as fp:
        content = fp.read()

    for key, value in replacements.items():
        content = content.replace(key, value)

    with open(outfile, "w", encoding="UTF-8") as fp:
        fp.write(content)


def main(folder="adoptODE"):
    # remove old files
    for path in glob.glob(f"{OUTPUT_PATH}/*.rst"):
        logging.info("Remove file `%s`", path)
        os.remove(path)

    # run sphinx-apidoc
    sp.check_call(
        [
            "sphinx-apidoc",
            "--maxdepth",
            "4",
            "-o",
            ".",
            "--module-first",
            f"../../{folder}",  # path of the package
        ]
    )

    # replace unwanted information
    for path in glob.glob(f"{OUTPUT_PATH}/*.rst"):
        logging.info("Patch file `%s`", path)
        replace_in_file(path, REPLACEMENTS)


if __name__ == "__main__":
    main()
