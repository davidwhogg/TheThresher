#!/usr/bin/env python

# FIXME: test

from distutils.core import setup

desc = open("README.md").read()
with open("requirements.txt") as f:
    required = f.readlines()

setup(
    name="thresher",
    version="0.0.1",
    author="David W. Hogg and Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    url="http://davidwhogg.github.com/TheThresher",
    packages=["thresher"],
    scripts=["bin/thresh", "bin/thresh-plot", "bin/lucky"],
    install_requires=required,
    license="GPLv2",
    description="we Don't Throw Away Data (tm).",
    long_description=desc,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    package_data={"thresher": ["test/*"]},
)
