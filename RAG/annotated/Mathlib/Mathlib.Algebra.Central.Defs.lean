/--
For a commutative ring `K` and a `K`-algebra `D`, we say that `D` is a central algebra over `K` if
the center of `D` is the image of `K` in `D`.
-/
class Algebra.IsCentral
    (K : Type u) [CommSemiring K] (D : Type v) [Semiring D] [Algebra K D] : Prop where
  out : Subalgebra.center K D ≤ ⊥

