# Continuous checks

`repro.yml` installs `requirements.txt` on a clean runner and then does two things. It runs `repro.py`, which reprints the 12 model melting-law table from the committed data. It also runs `src/validate.py`, which checks the thermodynamics against analytic results, including the two-level Schottky heat capacity and the identity between free energy, internal energy and entropy.

Neither step downloads a model, so the whole workflow finishes in under a minute. It runs on Python 3.11 and 3.13 so that a version problem in the pinned dependencies is caught rather than left for a reader to hit.
