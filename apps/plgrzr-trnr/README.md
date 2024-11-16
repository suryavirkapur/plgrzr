# Instructions

install micromamba with homebrew: brew install micromamba
create a new conda environment: micromamba env create -f env.yml
activate the new environment: micromamba activate tch-rs
create a symlink in this repo: ln -sf ~/micromamba/envs/tch-rs/lib/python3.10/site-packages/torch/ torch
run: cargo run
