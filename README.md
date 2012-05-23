# The Thresher

Luck isn't a factor when you are using **The Thresher**.

While we recognize that you have other options available for your
fast-camera imaging solution needs, *We Don't Throw Away Data (tm)*.

### Authors

- **David W. Hogg**, New York University
- **Dan Foreman-Mackey**, New York University
- **Federica Bianco**, Las Cumbres Observatory

### License

Copyright 2012 David W. Hogg & Dan Foreman-Mackey.

The Thresher is free software licensed under the **GNU General Public
License Version 2**.  See the file LICENSE for the full terms of this
license.

### Contributors

- **BJ Fulton**, Las Cumbres Observatory
- **Dustin Lang**, Princeton University Observatory
- **Phil Marshall**, Oxford University

### Notes

**The Thresher** is based on a mash-up of
- Hirsch et al (2011, A&A, 26, 531)
<http://adsabs.harvard.edu/abs/2011A%26A...531A...9H> and
- Magain et al (1998, ApJ 494, 472)
<http://adsabs.harvard.edu/abs/1998ApJ...494..472M>.

## Usage

The data should be in a set of `FITS` files in a directory somewhere.
To run the pipeline on this directory, run `bin/lucky` as follows:

```bash
bin/lucky /path/to/your/data -o /path/to/outputs --size 64
```

You can run `bin/lucky -h` for more command line options.

To plot the inference in real time, start the `bin/lucky-plot` daemon
which will monitor the output directory and generate the plots:

```bash
bin/lucky-plot /path/to/outputs -m -o /path/to/plots
```

To generate all of the plots for an existing run, use:

```bash
bin/lucky-plot /path/to/outputs -o /path/to/plots
```
