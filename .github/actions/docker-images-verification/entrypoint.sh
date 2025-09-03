#!/bin/sh -l
python3 oci-cvmfs.py . https://raw.githubusercontent.com/$GITHUB_REPOSITORY/$GITHUB_REF/images.txt --verify
