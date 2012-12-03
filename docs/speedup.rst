Speeding up PyPhysim
====================

There a re a number of alternatives to speed-up python code.

- you can use the weave module (inline or blitz methods) from scipy to
  spped up things here. See
  http://docs.scipy.org/doc/scipy/reference/tutorial/weave.html
  and
  http://www.scipy.org/PerformancePython
- You could use Cython
- You could try numexpr
  http://code.google.com/p/numexpr/
- You could try Numba
  http://jakevdp.github.com/blog/2012/08/24/numba-vs-cython/
- Use smart numpy broadcast tricks to avoid loops
  This is fast, but uses more memory. See :meth:`.Modulator.demodulate`
