#!/usr/bin/env bash

cat <<EOF >> ${PREFIX}/.messages.txt

**************************************************************************************
* NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE:                                 *
* Checkpoint files for SAM2 *must* be downloaded before running GRIME-AI.            *
* Please run the command 'download-sam2-checkpoints' prior to using GRIME-AI.        *
* This command will fetch the checkpoint files and save them to the proper location. *
**************************************************************************************

EOF
