Typing Support in PyPhysim
==========================

PyPhysim has an increasing support for static typing checking.

Ideally everything in PyPhysim should be type checked without errors by `mypy
<http://mypy-lang.org>`_ and any new code should ideally have typing information
as well.


.. Note:: There are other type checkers that can be used, such as `pytype (from
          Google) <https://github.com/google/pytype>`_ or `pyre (from Facebook)
          <https://www.facebook.com/notes/protect-the-graph/pyre-fast-type-checking-for-python/2048520695388071/>`_


Some useful information:
########################

- `What is the difference between TypeVar and NewType? <https://stackoverflow.com/questions/58755948/what-is-the-difference-between-typevar-and-newtype>`_
- `Covariance, Contravariance, and Invariance — The Ultimate Python Guide <https://blog.daftcode.pl/covariance-contravariance-and-invariance-the-ultimate-python-guide-8fabc0c24278>`_
- `Python Type Checking (Guide) <https://realpython.com/python-type-checking/>`_
- Accepting any derived class: If the number of subclasses is fixed, just create
  a Union with all of them. If it is not and you truly want "any subclass of
  Base", then try "Generic[Base]" as the argument type.

.. code-block:: python

   FadingGenerator = TypeVar('FadingGenerator',
                             bound=FadingSampleGenerator)

- Inspecting the type of variables: Use :code:`reveal_type(expr)` to see the
  type of :code:`expr`. It only works in `mypy` and you get a :code:`NameError`
  if you try to run code with it in Python.


