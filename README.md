# uwarevoice
This is a headless server that provides low latency ("real time") AI voice conversion over websockets. The server is intended to be accessed using a client (like [uwavclient](https://github.com/Uwaan/uwavclient)), allowing you to run server and client on different computers over your local network and offload the GPU-intensive AI stuff somewhere other than your main PC.

(You can certainly run it on the same computer as the client if you want, though, I don't care. I'm not the computer police. It's just hard to run a game and also a voice converter on the same GPU at the same time.)

This is heavily WIP and is currently hard-coded to host a single RVC v2 model *(yours)* for a single client *(you)*. I intend to experiment with different models and different engines once it's complete, but the "one client at a time" part is unlikely to ever change.
## Work in progress
Currently, this program lacks a lot of nice-ities like streamlined install scripts or built-in huggingface downloaders that you might be used to with other local AI projects. For the current RVC functionality to work, you will need to source your own <span>hubert</span>.pt, and <span>rmvpe</span>.pt, and others. I can't provide downloads for these, and even if I could, you shouldn't trust them because they're not safetensors.

At some point, this server will presumably mature and better installation instructions will be provided. Until then, the assumption is that you've downloaded other local AI stuff, and that you've played with it enough to know how this kind of thing typically works, and that you are capable of setting up a python venv and figuring out what needs to be installed using `pip`.

## Installation instructions
A few notes, though, because I'm not updating things like requirements.txt until things are more stable, and my implementation of RVC is a mess: RVC v2 is based on facebook's Hubert model, which is basically stone age technology in terms of AI, and many of the tools facebook made for it are so deprecated that pip won't even download them any more.
```
python3.10 -m venv venv
source venv/bin/activate
pip install --force pip==24.0
```
`faiss` and `fairseq` in particular have a bunch of stuff going on that needs old versions of pip/python to run, and their installation will give you errors unless you do this.

## Disclosure
This repository contains vibe code. Or, at least, its skeleton was completely vibe coded because I never had a reason to touch python before starting any of this. I still wouldn't consider myself very good with python, but this has at least gotten me to look at enough of it for long enough to be able to "write" a voice conversion engine (by stealing code from the RVC project the old fashioned way).
