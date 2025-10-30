 # GST spectroscopy
 
This project is intended to serve the purpose of working with and extracting
information from optical spectra obtained by FTIR spectroscopy.

So far, it allows one to obtain spectra of optical constants (n&k) of thin film
materials. The employed approach is called "inversion" below, meaning that we
solve the inverse problem — computation of n&k using R&T spectra using formulae
intended to compute R&T from n&k. Mathematically, a system of two non-linear
equations is solved; those equations are:

R(n, k) = R_exp
T(n, k) = T_exp

Here, R(n, k) and T(n, k) are models from film optics, while R_exp and T_exp are
experimental data. Thus, n and k must be the only two unknowns so that solution
could be found.

Initial data for the computation is two pairs of R&T spectra: of the film on the
substrate and of the substrate alone. Computation is only semi-automatic, so
some manual work is needed. Once aforementioned spectra are in posession,
proceed as follows:

1. First, you may want to look at the spectra. That is what `plot R+T.py` is
   for. It also strips the spectral range so that only interesting part (which
   is fundamental absorption edge of course) participates in computation.
   Additionally, some information about the spectra is printed to the console.
2. Once we are confident that the spectrum pair is the one we intend to use for
   computation, we proceed to the next step — computation of optical constants
   of the substrate material. Use `substrate inversion.py` for this. It will
   compute n&k of substrate material automatically. Mathematical model utilized
   disregards interference in substrate completely, which is the case when
   incoherent light source was used for spectra acquisition.
3. Next step involves working with the thin film spectrum. The script named
   `film-on-substrate inversion.py` solves the inversion problem for a thin film
   on a substrate. Interference for the film is attributed in mathematical
   model, while interference in substrate is excluded from consideration due to
   reason mentioned above. This script also needs the data obtained in the
   previous step. The inversion of substrate data was performed automatically,
   because in this case only one root can be obtained. However, for the film on
   the substrate problem multiple solutions will be obtained. All of them are
   mathematically correct, while only one branch is physically sound. And it is
   the task of researcher to identify it. But before diving in this process, it
   should be noted, that the algorithm allows to estimate also thin film
   thickness. When thickness, given to the `film-on-substrate inversion.py`
   script fits model+data well, the picture of all the mathematical roots will
   contain sharply intersecting branches. It takes trial-and-error process to
   determine the film thickness using this criterion. Once this criterion is
   satisfied, next step is to remove unphysical roots.
4. The `remove unphysical roots.py` does just that. It is not a computation
   script, but rather a mini-app to simplify roots sorting. It shows all the
   roots found during previous step in n versus k plot — in that way all the
   different branches of the solution are recognizable. All you need to do is
   to mark unphysical roots by drawing a line around them. Holding "Shift" will
   revert selection back. After only physical branch remains press "Enter" and
   it will be saved to the file. This file is the inversion result.
5. Finally, another mini-app — `Eg by approximation.py` — can be used to compute
   optical band gap from the obtained result. It will display the Tauc plot so
   that characteristic linear range should be readily recognizeable. All you
   need to do after that is to select it using mouse and approximation result
   will be immediately shown. Repeat if necessary.
