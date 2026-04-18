#!/usr/bin/env bash

cat <<EOF >> ${PREFIX}/.messages.txt

**************************************************************
* NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE:         *
* Checkpoint files for SAM2 and YOLO11 models *must* be      *
* downloaded before running GRIME-AI. Please activate the    *
* environment and run the commands                           *
* 'download-sam2-checkpoints' followed by                    *
* 'download-yolo-models' prior to using GRIME-AI. These      *
* commands will fetch the needed files and save them to the  *
* proper locations.                                          *
**************************************************************

EOF
