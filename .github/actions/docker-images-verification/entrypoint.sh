#!/bin/sh -l
python3 oci-cvmfs.py . https://raw.githubusercontent.com/nrp-nautilus/cvmfs-oci/refs/$GITHUB_REF/main/images.txt --verify
