#!/usr/bin/bash
conda create torch
conda activate torch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge cudatoolkit-dev 