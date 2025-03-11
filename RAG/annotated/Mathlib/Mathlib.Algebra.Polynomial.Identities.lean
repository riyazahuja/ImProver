/-- `(x + y)^n` can be expressed as `x^n + n*x^(n-1)*y + k * y^2` for some `k` in the ring.
-/
def powAddExpansion {R : Type*} [CommSemiring R] (x y : R) :
    ∀ n : ℕ, { k // (x + y) ^ n = x ^ n + n * x ^ (n - 1) * y + k * y ^ 2 }
                /-
                  R✝ : Type u
                  S : Type v
                  T : Type w
                  ι : Type x
                  k : Type y
                  A : Type z
                  a b : R✝
                  m n : Nat
                  R : Type u_1
                  inst✝ : CommSemiring R
                  x y : R
                  ⊢ Eq (HPow.hPow (HAdd.hAdd x y) 0) (HAdd.hAdd (HAdd.hAdd (HPow.hPow x 0) (HMul …
                -/
  | 0 => ⟨0, by simp⟩
                /-
                  🎉 no goals
                -/
                /-
                  R✝ : Type u
                  S : Type v
                  T : Type w
                  ι : Type x
                  k : Type y
                  A : Type z
                  a b : R✝
                  m n : Nat
                  R : Type u_1
                  inst✝ : CommSemiring R
                  x y : R
                  ⊢ Eq (HPow.hPow (HAdd.hAdd x y) 1) (HAdd.hAdd (HAdd.hAdd (HPow.hPow x 1) (HMul …
                -/
  | 1 => ⟨0, by simp⟩
                /-
                  🎉 no goals
                -/
  | n + 2 => by
    /-
      R✝ : Type u
      S : Type v
      T : Type w
      ι : Type x
      k : Type y
      A : Type z
      a b : R✝
      m n✝ : Nat
      R : Type u_1
      inst✝ : CommSemiring R
      x y : R
      n : Nat
      ⊢ Subtype fun k => Eq (HPow.hPow (HAdd.hAdd x y) (HAdd.hAdd n 2)) (HAdd.hAdd ( …
    -/
    cases' (powAddExpansion x y (n + 1)) with z hz
    /-
      case mk
      R✝ : Type u
      S : Type v
      T : Type w
      ι : Type x
      k : Type y
      A : Type z
      a b : R✝
      m n✝ : Nat
      R : Type u_1
      inst✝ : CommSemiring R
      x y : R
      n : Nat
      z : R
      hz : Eq (HPow.hPow (HAdd.hAdd x y) (HAdd.hAdd n 1)) (HAdd.hAdd (HAdd.hAdd (HPo …
      ⊢ Subtype fun k => Eq (HPow.hPow (HAdd.hAdd x y) (HAdd.hAdd n 2)) (HAdd.hAdd ( …
    -/
    exists x * z + (n + 1) * x ^ n + z * y
    calc
      (x + y) ^ (n + 2) = (x + y) * (x + y) ^ (n + 1) := by ring
      _ = (x + y) * (x ^ (n + 1) + ↑(n + 1) * x ^ (n + 1 - 1) * y + z * y ^ 2) := by rw [hz]
      _ = x ^ (n + 2) + ↑(n + 2) * x ^ (n + 1) * y + (x * z + (n + 1) * x ^ n + z * y) * y ^ 2 := by
        push_cast
        ring!


private def polyBinomAux1 (x y : R) (e : ℕ) (a : R) :
    { k : R // a * (x + y) ^ e = a * (x ^ e + e * x ^ (e - 1) * y + k * y ^ 2) } := by
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a✝ b : R
    m n : Nat
    inst✝ : CommRing R
    x y : R
    e : Nat
    a : R
    ⊢ Subtype fun k => Eq (HMul.hMul a (HPow.hPow (HAdd.hAdd x y) e)) (HMul.hMul a …
  -/
  exists (powAddExpansion x y e).val
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a✝ b : R
    m n : Nat
    inst✝ : CommRing R
    x y : R
    e : Nat
    a : R
    ⊢ Eq (HMul.hMul a (HPow.hPow (HAdd.hAdd x y) e)) (HMul.hMul a (HAdd.hAdd (HAdd …
  -/
  congr
  /-
    case e_a
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a✝ b : R
    m n : Nat
    inst✝ : CommRing R
    x y : R
    e : Nat
    a : R
    ⊢ Eq (HPow.hPow (HAdd.hAdd x y) e) (HAdd.hAdd (HAdd.hAdd (HPow.hPow x e) (HMul …
  -/
  apply (powAddExpansion _ _ _).property
  /-
    🎉 no goals
  -/


private theorem poly_binom_aux2 (f : R[X]) (x y : R) :
    f.eval (x + y) =
      f.sum fun e a => a * (x ^ e + e * x ^ (e - 1) * y + (polyBinomAux1 x y e a).val * y ^ 2) := by
  /-
    R : Type u
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (Polynomial.eval (HAdd.hAdd x y) f) (f.sum fun e a => HMul.hMul a (HAdd.h …
  -/
  unfold eval; rw [eval₂_eq_sum]; congr with (n z)
  /-
    case e_f.h.h
    R : Type u
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    n : Nat
    z : R
    ⊢ Eq (HMul.hMul ((RingHom.id R) z) (HPow.hPow (HAdd.hAdd x y) n)) (HMul.hMul z …
  -/
  apply (polyBinomAux1 x y _ _).property
  /-
    🎉 no goals
  -/


private theorem poly_binom_aux3 (f : R[X]) (x y : R) :
    f.eval (x + y) =
      ((f.sum fun e a => a * x ^ e) + f.sum fun e a => a * e * x ^ (e - 1) * y) +
        f.sum fun e a => a * (polyBinomAux1 x y e a).val * y ^ 2 := by
  /-
    R : Type u
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (Polynomial.eval (HAdd.hAdd x y) f) (HAdd.hAdd (HAdd.hAdd (f.sum fun e a  …
  -/
  rw [poly_binom_aux2]
  /-
    R : Type u
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (f.sum fun e a => HMul.hMul a (HAdd.hAdd (HAdd.hAdd (HPow.hPow x e) (HMul …
  -/
  simp [left_distrib, sum_add, mul_assoc]
  /-
    🎉 no goals
  -/


/-- A polynomial `f` evaluated at `x + y` can be expressed as
the evaluation of `f` at `x`, plus `y` times the (polynomial) derivative of `f` at `x`,
plus some element `k : R` times `y^2`.
-/
def binomExpansion (f : R[X]) (x y : R) :
    { k : R // f.eval (x + y) = f.eval x + f.derivative.eval x * y + k * y ^ 2 } := by
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Subtype fun k => Eq (Polynomial.eval (HAdd.hAdd x y) f) (HAdd.hAdd (HAdd.hAd …
  -/
  exists f.sum fun e a => a * (polyBinomAux1 x y e a).val
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (Polynomial.eval (HAdd.hAdd x y) f) (HAdd.hAdd (HAdd.hAdd (Polynomial.eva …
  -/
  rw [poly_binom_aux3]
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (f.sum fun e a => HMul.hMul a (HPow.hPow x e)) (f.s …
  -/
  congr
    /-
      case e_a.e_a
      R : Type u
      S : Type v
      T : Type w
      ι : Type x
      k : Type y
      A : Type z
      a b : R
      m n : Nat
      inst✝ : CommRing R
      f : Polynomial R
      x y : R
      ⊢ Eq (f.sum fun e a => HMul.hMul a (HPow.hPow x e)) (Polynomial.eval x f)
    -/
  · rw [← eval_eq_sum]
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_a
      R : Type u
      S : Type v
      T : Type w
      ι : Type x
      k : Type y
      A : Type z
      a b : R
      m n : Nat
      inst✝ : CommRing R
      f : Polynomial R
      x y : R
      ⊢ Eq (f.sum fun e a => HMul.hMul (HMul.hMul (HMul.hMul a ↑e) (HPow.hPow x (HSu …
    -/
  · rw [derivative_eval]
    /-
      case e_a.e_a
      R : Type u
      S : Type v
      T : Type w
      ι : Type x
      k : Type y
      A : Type z
      a b : R
      m n : Nat
      inst✝ : CommRing R
      f : Polynomial R
      x y : R
      ⊢ Eq (f.sum fun e a => HMul.hMul (HMul.hMul (HMul.hMul a ↑e) (HPow.hPow x (HSu …
    -/
    exact (Finset.sum_mul ..).symm
    /-
      🎉 no goals
    -/
    /-
      case e_a
      R : Type u
      S : Type v
      T : Type w
      ι : Type x
      k : Type y
      A : Type z
      a b : R
      m n : Nat
      inst✝ : CommRing R
      f : Polynomial R
      x y : R
      ⊢ Eq (f.sum fun e a => HMul.hMul (HMul.hMul a ↑(Polynomial.polyBinomAux1 x y e …
    -/
  · exact (Finset.sum_mul ..).symm
    /-
      🎉 no goals
    -/


/-- `x^n - y^n` can be expressed as `z * (x - y)` for some `z` in the ring.
-/
def powSubPowFactor (x y : R) : ∀ i : ℕ, { z : R // x ^ i - y ^ i = z * (x - y) }
                /-
                  R : Type u
                  S : Type v
                  T : Type w
                  ι : Type x
                  k : Type y
                  A : Type z
                  a b : R
                  m n : Nat
                  inst✝ : CommRing R
                  x y : R
                  ⊢ Eq (HSub.hSub (HPow.hPow x 0) (HPow.hPow y 0)) (HMul.hMul 0 (HSub.hSub x y))
                -/
  | 0 => ⟨0, by simp⟩
                /-
                  🎉 no goals
                -/
                /-
                  R : Type u
                  S : Type v
                  T : Type w
                  ι : Type x
                  k : Type y
                  A : Type z
                  a b : R
                  m n : Nat
                  inst✝ : CommRing R
                  x y : R
                  ⊢ Eq (HSub.hSub (HPow.hPow x 1) (HPow.hPow y 1)) (HMul.hMul 1 (HSub.hSub x y))
                -/
  | 1 => ⟨1, by simp⟩
                /-
                  🎉 no goals
                -/
  | k + 2 => by
    /-
      R : Type u
      S : Type v
      T : Type w
      ι : Type x
      k✝ : Type y
      A : Type z
      a b : R
      m n : Nat
      inst✝ : CommRing R
      x y : R
      k : Nat
      ⊢ Subtype fun z => Eq (HSub.hSub (HPow.hPow x (HAdd.hAdd k 2)) (HPow.hPow y (H …
    -/
    cases' @powSubPowFactor x y (k + 1) with z hz
    /-
      case mk
      R : Type u
      S : Type v
      T : Type w
      ι : Type x
      k✝ : Type y
      A : Type z
      a b : R
      m n : Nat
      inst✝ : CommRing R
      x y : R
      k : Nat
      z : R
      hz : Eq (HSub.hSub (HPow.hPow x (HAdd.hAdd k 1)) (HPow.hPow y (HAdd.hAdd k 1)) …
      ⊢ Subtype fun z => Eq (HSub.hSub (HPow.hPow x (HAdd.hAdd k 2)) (HPow.hPow y (H …
    -/
    exists z * x + y ^ (k + 1)
    /-
      case mk
      R : Type u
      S : Type v
      T : Type w
      ι : Type x
      k✝ : Type y
      A : Type z
      a b : R
      m n : Nat
      inst✝ : CommRing R
      x y : R
      k : Nat
      z : R
      hz : Eq (HSub.hSub (HPow.hPow x (HAdd.hAdd k 1)) (HPow.hPow y (HAdd.hAdd k 1)) …
      ⊢ Eq (HSub.hSub (HPow.hPow x (HAdd.hAdd k 2)) (HPow.hPow y (HAdd.hAdd k 2))) ( …
    -/
    linear_combination (norm := ring) x * hz
    /-
      🎉 no goals
    -/


/-- For any polynomial `f`, `f.eval x - f.eval y` can be expressed as `z * (x - y)`
for some `z` in the ring.
-/
def evalSubFactor (f : R[X]) (x y : R) : { z : R // f.eval x - f.eval y = z * (x - y) } := by
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Subtype fun z => Eq (HSub.hSub (Polynomial.eval x f) (Polynomial.eval y f))  …
  -/
  refine ⟨f.sum fun i r => r * (powSubPowFactor x y i).val, ?_⟩
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (HSub.hSub (Polynomial.eval x f) (Polynomial.eval y f)) (HMul.hMul (f.sum …
  -/
  delta eval; rw [eval₂_eq_sum, eval₂_eq_sum]
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (HSub.hSub (f.sum fun e a => HMul.hMul ((RingHom.id R) a) (HPow.hPow x e) …
  -/
  simp only [sum, ← Finset.sum_sub_distrib, Finset.sum_mul]
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (f.support.sum fun x_1 => HSub.hSub (HMul.hMul ((RingHom.id R) (f.coeff x …
  -/
  dsimp
  /-
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    ⊢ Eq (f.support.sum fun x_1 => HSub.hSub (HMul.hMul (f.coeff x_1) (HPow.hPow x …
  -/
  congr with i
  /-
    case e_f.h
    R : Type u
    S : Type v
    T : Type w
    ι : Type x
    k : Type y
    A : Type z
    a b : R
    m n : Nat
    inst✝ : CommRing R
    f : Polynomial R
    x y : R
    i : Nat
    ⊢ Eq (HSub.hSub (HMul.hMul (f.coeff i) (HPow.hPow x i)) (HMul.hMul (f.coeff i) …
  -/
  rw [mul_assoc, ← (powSubPowFactor x y _).prop, mul_sub]
  /-
    🎉 no goals
  -/


