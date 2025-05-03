/-- `WittVector p R` is the ring of `p`-typical Witt vectors over the commutative ring `R`,
where `p` is a prime number.

If `p` is invertible in `R`, this ring is isomorphic to `ℕ → R` (the product of `ℕ` copies of `R`).
If `R` is a ring of characteristic `p`, then `WittVector p R` is a ring of characteristic `0`.
The canonical example is `WittVector p (ZMod p)`,
which is isomorphic to the `p`-adic integers `ℤ_[p]`. -/
structure WittVector (p : ℕ) (R : Type*) where mk' ::
  /-- `x.coeff n` is the `n`th coefficient of the Witt vector `x`.

  This concept does not have a standard name in the literature.
  -/
  coeff : ℕ → R

-- Porting note: added to make the `p` argument explicit

/-- Construct a Witt vector `mk p x : 𝕎 R` from a sequence `x` of elements of `R`. -/
def WittVector.mk (p : ℕ) {R : Type*} (coeff : ℕ → R) : WittVector p R := mk' coeff


local notation "𝕎" => WittVector p -- type as `\bbW`


@[ext]
theorem ext {x y : 𝕎 R} (h : ∀ n, x.coeff n = y.coeff n) : x = y := by
  /-
    p : Nat
    R : Type u_1
    x y : WittVector p R
    h : ∀ (n : Nat), Eq (x.coeff n) (y.coeff n)
    ⊢ Eq x y
  -/
  cases x
  /-
    case mk'
    p : Nat
    R : Type u_1
    y : WittVector p R
    coeff✝ : Nat → R
    h : ∀ (n : Nat), Eq ({ coeff := coeff✝ }.coeff n) (y.coeff n)
    ⊢ Eq { coeff := coeff✝ } y
  -/
  cases y
  /-
    case mk'.mk'
    p : Nat
    R : Type u_1
    coeff✝¹ coeff✝ : Nat → R
    h : ∀ (n : Nat), Eq ({ coeff := coeff✝¹ }.coeff n) ({ coeff := coeff✝ }.coeff n)
    ⊢ Eq { coeff := coeff✝¹ } { coeff := coeff✝ }
  -/
  simp only at h
  /-
    case mk'.mk'
    p : Nat
    R : Type u_1
    coeff✝¹ coeff✝ : Nat → R
    h : ∀ (n : Nat), Eq (coeff✝¹ n) (coeff✝ n)
    ⊢ Eq { coeff := coeff✝¹ } { coeff := coeff✝ }
  -/
  simp [funext_iff, h]
  /-
    🎉 no goals
  -/


theorem coeff_mk (x : ℕ → R) : (mk p x).coeff = x :=
  rfl

/- These instances are not needed for the rest of the development,
but it is interesting to establish early on that `WittVector p` is a lawful functor. -/

instance : Functor (WittVector p) where
  map f v := mk p (f ∘ v.coeff)
  mapConst a _ := mk p fun _ => a


instance : LawfulFunctor (WittVector p) where
  map_const := rfl
  -- Porting note: no longer needs to deconstruct `v` to conclude `{coeff := v.coeff} = v`
  id_map _ := rfl
  comp_map _ _ _ := rfl


/-- The polynomials used for defining the element `0` of the ring of Witt vectors. -/
def wittZero : ℕ → MvPolynomial (Fin 0 × ℕ) ℤ :=
  wittStructureInt p 0


/-- The polynomials used for defining the element `1` of the ring of Witt vectors. -/
def wittOne : ℕ → MvPolynomial (Fin 0 × ℕ) ℤ :=
  wittStructureInt p 1


/-- The polynomials used for defining the addition of the ring of Witt vectors. -/
def wittAdd : ℕ → MvPolynomial (Fin 2 × ℕ) ℤ :=
  wittStructureInt p (X 0 + X 1)


/-- The polynomials used for defining repeated addition of the ring of Witt vectors. -/
def wittNSMul (n : ℕ) : ℕ → MvPolynomial (Fin 1 × ℕ) ℤ :=
  wittStructureInt p (n • X (0 : (Fin 1)))


/-- The polynomials used for defining repeated addition of the ring of Witt vectors. -/
def wittZSMul (n : ℤ) : ℕ → MvPolynomial (Fin 1 × ℕ) ℤ :=
  wittStructureInt p (n • X (0 : (Fin 1)))


/-- The polynomials used for describing the subtraction of the ring of Witt vectors. -/
def wittSub : ℕ → MvPolynomial (Fin 2 × ℕ) ℤ :=
  wittStructureInt p (X 0 - X 1)


/-- The polynomials used for defining the multiplication of the ring of Witt vectors. -/
def wittMul : ℕ → MvPolynomial (Fin 2 × ℕ) ℤ :=
  wittStructureInt p (X 0 * X 1)


/-- The polynomials used for defining the negation of the ring of Witt vectors. -/
def wittNeg : ℕ → MvPolynomial (Fin 1 × ℕ) ℤ :=
  wittStructureInt p (-X 0)


/-- The polynomials used for defining repeated addition of the ring of Witt vectors. -/
def wittPow (n : ℕ) : ℕ → MvPolynomial (Fin 1 × ℕ) ℤ :=
  wittStructureInt p (X 0 ^ n)


/-- An auxiliary definition used in `WittVector.eval`.
Evaluates a polynomial whose variables come from the disjoint union of `k` copies of `ℕ`,
with a curried evaluation `x`.
This can be defined more generally but we use only a specific instance here. -/
def peval {k : ℕ} (φ : MvPolynomial (Fin k × ℕ) ℤ) (x : Fin k → ℕ → R) : R :=
  aeval (Function.uncurry x) φ


/-- Let `φ` be a family of polynomials, indexed by natural numbers, whose variables come from the
disjoint union of `k` copies of `ℕ`, and let `xᵢ` be a Witt vector for `0 ≤ i < k`.

`eval φ x` evaluates `φ` mapping the variable `X_(i, n)` to the `n`th coefficient of `xᵢ`.

Instantiating `φ` with certain polynomials defined in
`Mathlib/RingTheory/WittVector/StructurePolynomial.lean` establishes the
ring operations on `𝕎 R`. For example, `WittVector.wittAdd` is such a `φ` with `k = 2`;
evaluating this at `(x₀, x₁)` gives us the sum of two Witt vectors `x₀ + x₁`.
-/
def eval {k : ℕ} (φ : ℕ → MvPolynomial (Fin k × ℕ) ℤ) (x : Fin k → 𝕎 R) : 𝕎 R :=
  mk p fun n => peval (φ n) fun i => (x i).coeff


instance : Zero (𝕎 R) :=
  ⟨eval (wittZero p) ![]⟩


instance : Inhabited (𝕎 R) :=
  ⟨0⟩


instance : One (𝕎 R) :=
  ⟨eval (wittOne p) ![]⟩


instance : Add (𝕎 R) :=
  ⟨fun x y => eval (wittAdd p) ![x, y]⟩


instance : Sub (𝕎 R) :=
  ⟨fun x y => eval (wittSub p) ![x, y]⟩


instance hasNatScalar : SMul ℕ (𝕎 R) :=
  ⟨fun n x => eval (wittNSMul p n) ![x]⟩


instance hasIntScalar : SMul ℤ (𝕎 R) :=
  ⟨fun n x => eval (wittZSMul p n) ![x]⟩


instance : Mul (𝕎 R) :=
  ⟨fun x y => eval (wittMul p) ![x, y]⟩


instance : Neg (𝕎 R) :=
  ⟨fun x => eval (wittNeg p) ![x]⟩


instance hasNatPow : Pow (𝕎 R) ℕ :=
  ⟨fun x n => eval (wittPow p n) ![x]⟩


instance : NatCast (𝕎 R) :=
  ⟨Nat.unaryCast⟩


instance : IntCast (𝕎 R) :=
  ⟨Int.castDef⟩


@[simp]
theorem wittZero_eq_zero (n : ℕ) : wittZero p n = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (WittVector.wittZero p n) 0
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [wittZero, wittStructureRat, bind₁, aeval_zero', constantCoeff_xInTermsOfW, map_zero,
    map_wittStructureInt]


@[simp]
theorem wittOne_zero_eq_one : wittOne p 0 = 1 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (WittVector.wittOne p 0) 1
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [wittOne, wittStructureRat, xInTermsOfW_zero, map_one, bind₁_X_right,
    map_wittStructureInt]


@[simp]
theorem wittOne_pos_eq_zero (n : ℕ) (hn : 0 < n) : wittOne p n = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (WittVector.wittOne p n) 0
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [wittOne, wittStructureRat, RingHom.map_zero, map_one, RingHom.map_one,
    map_wittStructureInt]
  /-
    case a
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq ((MvPolynomial.bind₁ fun k => 1) (xInTermsOfW p Rat n)) 0
  -/
  induction n using Nat.strong_induction_on with | h n IH => ?_
  /-
    case a.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
    hn : LT.lt 0 n
    ⊢ Eq ((MvPolynomial.bind₁ fun k => 1) (xInTermsOfW p Rat n)) 0
  -/
  rw [xInTermsOfW_eq]
  simp only [map_mul, map_sub, map_sum, map_pow, bind₁_X_right,
    bind₁_C_right]
  /-
    case a.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
    hn : LT.lt 0 n
    ⊢ Eq (HMul.hMul (HSub.hSub 1 ((Finset.range n).sum fun x => HMul.hMul (HPow.hP …
  -/
  rw [sub_mul, one_mul]
  /-
    case a.h
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
    hn : LT.lt 0 n
    ⊢ Eq (HSub.hSub (HPow.hPow (MvPolynomial.C (Invertible.invOf ↑p)) n) (HMul.hMu …
  -/
  rw [Finset.sum_eq_single 0]
    /-
      case a.h
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
      hn : LT.lt 0 n
      ⊢ Eq (HSub.hSub (HPow.hPow (MvPolynomial.C (Invertible.invOf ↑p)) n) (HMul.hMu …
    -/
  · simp only [invOf_eq_inv, one_mul, inv_pow, tsub_zero, RingHom.map_one, pow_zero]
    /-
      case a.h
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
      hn : LT.lt 0 n
      ⊢ Eq (HSub.hSub (HPow.hPow (MvPolynomial.C (Inv.inv ↑p)) n) (HMul.hMul (HPow.h …
    -/
    simp only [one_pow, one_mul, xInTermsOfW_zero, sub_self, bind₁_X_right]
    /-
      🎉 no goals
    -/
    /-
      case a.h.h₀
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
      hn : LT.lt 0 n
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range n) b → Ne b 0 → Eq (HMul.hMul (HPo …
    -/
  · intro i hin hi0
    /-
      case a.h.h₀
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
      hn : LT.lt 0 n
      i : Nat
      hin : Membership.mem (Finset.range n) i
      hi0 : Ne i 0
      ⊢ Eq (HMul.hMul (HPow.hPow (MvPolynomial.C ↑p) i) (HPow.hPow ((MvPolynomial.bi …
    -/
    rw [Finset.mem_range] at hin
    /-
      case a.h.h₀
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
      hn : LT.lt 0 n
      i : Nat
      hin : LT.lt i n
      hi0 : Ne i 0
      ⊢ Eq (HMul.hMul (HPow.hPow (MvPolynomial.C ↑p) i) (HPow.hPow ((MvPolynomial.bi …
    -/
    rw [IH _ hin (Nat.pos_of_ne_zero hi0), zero_pow (pow_ne_zero _ hp.1.ne_zero), mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case a.h.h₁
      p : Nat
      hp : Fact (Nat.Prime p)
      n : Nat
      IH : ∀ (m : Nat), LT.lt m n → LT.lt 0 m → Eq ((MvPolynomial.bind₁ fun k => 1)  …
      hn : LT.lt 0 n
      ⊢ Not (Membership.mem (Finset.range n) 0) → Eq (HMul.hMul (HPow.hPow (MvPolyno …
    -/
  · rw [Finset.mem_range]; intro; contradiction
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem wittAdd_zero : wittAdd p 0 = X (0, 0) + X (1, 0) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (WittVector.wittAdd p 0) (HAdd.hAdd (MvPolynomial.X { fst := 0, snd := 0  …
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [wittAdd, wittStructureRat, map_add, rename_X, xInTermsOfW_zero, map_X,
    wittPolynomial_zero, bind₁_X_right, map_wittStructureInt]


@[simp]
theorem wittSub_zero : wittSub p 0 = X (0, 0) - X (1, 0) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (WittVector.wittSub p 0) (HSub.hSub (MvPolynomial.X { fst := 0, snd := 0  …
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [wittSub, wittStructureRat, map_sub, rename_X, xInTermsOfW_zero, map_X,
    wittPolynomial_zero, bind₁_X_right, map_wittStructureInt]


@[simp]
theorem wittMul_zero : wittMul p 0 = X (0, 0) * X (1, 0) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (WittVector.wittMul p 0) (HMul.hMul (MvPolynomial.X { fst := 0, snd := 0  …
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [wittMul, wittStructureRat, rename_X, xInTermsOfW_zero, map_X, wittPolynomial_zero,
    map_mul, bind₁_X_right, map_wittStructureInt]


@[simp]
theorem wittNeg_zero : wittNeg p 0 = -X (0, 0) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ Eq (WittVector.wittNeg p 0) (Neg.neg (MvPolynomial.X { fst := 0, snd := 0 }))
  -/
  apply MvPolynomial.map_injective (Int.castRingHom ℚ) Int.cast_injective
  simp only [wittNeg, wittStructureRat, rename_X, xInTermsOfW_zero, map_X, wittPolynomial_zero,
    map_neg, bind₁_X_right, map_wittStructureInt]


@[simp]
theorem constantCoeff_wittAdd (n : ℕ) : constantCoeff (wittAdd p n) = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (WittVector.wittAdd p n)) 0
  -/
  apply constantCoeff_wittStructureInt p _ _ n
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (HAdd.hAdd (MvPolynomial.X 0) (MvPolynomial.X …
  -/
  simp only [add_zero, RingHom.map_add, constantCoeff_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_wittSub (n : ℕ) : constantCoeff (wittSub p n) = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (WittVector.wittSub p n)) 0
  -/
  apply constantCoeff_wittStructureInt p _ _ n
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (HSub.hSub (MvPolynomial.X 0) (MvPolynomial.X …
  -/
  simp only [sub_zero, RingHom.map_sub, constantCoeff_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_wittMul (n : ℕ) : constantCoeff (wittMul p n) = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (WittVector.wittMul p n)) 0
  -/
  apply constantCoeff_wittStructureInt p _ _ n
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (HMul.hMul (MvPolynomial.X 0) (MvPolynomial.X …
  -/
  simp only [mul_zero, RingHom.map_mul, constantCoeff_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_wittNeg (n : ℕ) : constantCoeff (wittNeg p n) = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (WittVector.wittNeg p n)) 0
  -/
  apply constantCoeff_wittStructureInt p _ _ n
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (Neg.neg (MvPolynomial.X 0))) 0
  -/
  simp only [neg_zero, RingHom.map_neg, constantCoeff_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_wittNSMul (m : ℕ) (n : ℕ) : constantCoeff (wittNSMul p m n) = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (WittVector.wittNSMul p m n)) 0
  -/
  apply constantCoeff_wittStructureInt p _ _ n
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    m n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (HSMul.hSMul m (MvPolynomial.X 0))) 0
  -/
  simp only [smul_zero, map_nsmul, constantCoeff_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_wittZSMul (z : ℤ) (n : ℕ) : constantCoeff (wittZSMul p z n) = 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    z : Int
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (WittVector.wittZSMul p z n)) 0
  -/
  apply constantCoeff_wittStructureInt p _ _ n
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    z : Int
    n : Nat
    ⊢ Eq (MvPolynomial.constantCoeff (HSMul.hSMul z (MvPolynomial.X 0))) 0
  -/
  simp only [smul_zero, map_zsmul, constantCoeff_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_coeff (n : ℕ) : (0 : 𝕎 R).coeff n = 0 :=
                                           /-
                                             p : Nat
                                             R : Type u_1
                                             hp : Fact (Nat.Prime p)
                                             inst✝ : CommRing R
                                             n : Nat
                                             ⊢ Eq ((MvPolynomial.aeval (Function.uncurry fun i => (Matrix.vecEmpty i).coeff …
                                           -/
  show (aeval _ (wittZero p n) : R) = 0 by simp only [wittZero_eq_zero, map_zero]
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem one_coeff_zero : (1 : 𝕎 R).coeff 0 = 1 :=
                                          /-
                                            p : Nat
                                            R : Type u_1
                                            hp : Fact (Nat.Prime p)
                                            inst✝ : CommRing R
                                            ⊢ Eq ((MvPolynomial.aeval (Function.uncurry fun i => (Matrix.vecEmpty i).coeff …
                                          -/
  show (aeval _ (wittOne p 0) : R) = 1 by simp only [wittOne_zero_eq_one, map_one]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem one_coeff_eq_of_pos (n : ℕ) (hn : 0 < n) : coeff (1 : 𝕎 R) n = 0 :=
                                          /-
                                            p : Nat
                                            R : Type u_1
                                            hp : Fact (Nat.Prime p)
                                            inst✝ : CommRing R
                                            n : Nat
                                            hn : LT.lt 0 n
                                            ⊢ Eq ((MvPolynomial.aeval (Function.uncurry fun i => (Matrix.vecEmpty i).coeff …
                                          -/
  show (aeval _ (wittOne p n) : R) = 0 by simp only [hn, wittOne_pos_eq_zero, map_zero]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem v2_coeff {p' R'} (x y : WittVector p' R') (i : Fin 2) :
                                                    /-
                                                      p' : Nat
                                                      R' : Type u_2
                                                      x y : WittVector p' R'
                                                      i : Fin 2
                                                      ⊢ Eq (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpty) i).coeff (Matrix.vec …
                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    (![x, y] i).coeff = ![x.coeff, y.coeff] i := by fin_cases i <;> simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

-- Porting note: the lemmas below needed `coeff_mk` added to the `simp` calls


theorem add_coeff (x y : 𝕎 R) (n : ℕ) :
    (x + y).coeff n = peval (wittAdd p n) ![x.coeff, y.coeff] := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x y : WittVector p R
    n : Nat
    ⊢ Eq ((HAdd.hAdd x y).coeff n) (WittVector.peval (WittVector.wittAdd p n) (Mat …
  -/
  simp [(· + ·), Add.add, eval, coeff_mk]
  /-
    🎉 no goals
  -/


theorem sub_coeff (x y : 𝕎 R) (n : ℕ) :
    (x - y).coeff n = peval (wittSub p n) ![x.coeff, y.coeff] := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x y : WittVector p R
    n : Nat
    ⊢ Eq ((HSub.hSub x y).coeff n) (WittVector.peval (WittVector.wittSub p n) (Mat …
  -/
  simp [(· - ·), Sub.sub, eval, coeff_mk]
  /-
    🎉 no goals
  -/


theorem mul_coeff (x y : 𝕎 R) (n : ℕ) :
    (x * y).coeff n = peval (wittMul p n) ![x.coeff, y.coeff] := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x y : WittVector p R
    n : Nat
    ⊢ Eq ((HMul.hMul x y).coeff n) (WittVector.peval (WittVector.wittMul p n) (Mat …
  -/
  simp [(· * ·), Mul.mul, eval, coeff_mk]
  /-
    🎉 no goals
  -/


theorem neg_coeff (x : 𝕎 R) (n : ℕ) : (-x).coeff n = peval (wittNeg p n) ![x.coeff] := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x : WittVector p R
    n : Nat
    ⊢ Eq ((Neg.neg x).coeff n) (WittVector.peval (WittVector.wittNeg p n) (Matrix. …
  -/
  simp [Neg.neg, eval, Matrix.cons_fin_one, coeff_mk]
  /-
    🎉 no goals
  -/


theorem nsmul_coeff (m : ℕ) (x : 𝕎 R) (n : ℕ) :
    (m • x).coeff n = peval (wittNSMul p m n) ![x.coeff] := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    m : Nat
    x : WittVector p R
    n : Nat
    ⊢ Eq ((HSMul.hSMul m x).coeff n) (WittVector.peval (WittVector.wittNSMul p m n …
  -/
  simp [(· • ·), SMul.smul, eval, Matrix.cons_fin_one, coeff_mk]
  /-
    🎉 no goals
  -/


theorem zsmul_coeff (m : ℤ) (x : 𝕎 R) (n : ℕ) :
    (m • x).coeff n = peval (wittZSMul p m n) ![x.coeff] := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    m : Int
    x : WittVector p R
    n : Nat
    ⊢ Eq ((HSMul.hSMul m x).coeff n) (WittVector.peval (WittVector.wittZSMul p m n …
  -/
  simp [(· • ·), SMul.smul, eval, Matrix.cons_fin_one, coeff_mk]
  /-
    🎉 no goals
  -/


theorem pow_coeff (m : ℕ) (x : 𝕎 R) (n : ℕ) :
    (x ^ m).coeff n = peval (wittPow p m n) ![x.coeff] := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    m : Nat
    x : WittVector p R
    n : Nat
    ⊢ Eq ((HPow.hPow x m).coeff n) (WittVector.peval (WittVector.wittPow p m n) (M …
  -/
  simp [(· ^ ·), Pow.pow, eval, Matrix.cons_fin_one, coeff_mk]
  /-
    🎉 no goals
  -/


theorem add_coeff_zero (x y : 𝕎 R) : (x + y).coeff 0 = x.coeff 0 + y.coeff 0 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x y : WittVector p R
    ⊢ Eq ((HAdd.hAdd x y).coeff 0) (HAdd.hAdd (x.coeff 0) (y.coeff 0))
  -/
  simp [add_coeff, peval, Function.uncurry]
  /-
    🎉 no goals
  -/


theorem mul_coeff_zero (x y : 𝕎 R) : (x * y).coeff 0 = x.coeff 0 * y.coeff 0 := by
  /-
    p : Nat
    R : Type u_1
    hp : Fact (Nat.Prime p)
    inst✝ : CommRing R
    x y : WittVector p R
    ⊢ Eq ((HMul.hMul x y).coeff 0) (HMul.hMul (x.coeff 0) (y.coeff 0))
  -/
  simp [mul_coeff, peval, Function.uncurry]
  /-
    🎉 no goals
  -/


theorem wittAdd_vars (n : ℕ) : (wittAdd p n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) :=
  wittStructureInt_vars _ _ _


theorem wittSub_vars (n : ℕ) : (wittSub p n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) :=
  wittStructureInt_vars _ _ _


theorem wittMul_vars (n : ℕ) : (wittMul p n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) :=
  wittStructureInt_vars _ _ _


theorem wittNeg_vars (n : ℕ) : (wittNeg p n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) :=
  wittStructureInt_vars _ _ _


theorem wittNSMul_vars (m : ℕ) (n : ℕ) :
    (wittNSMul p m n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) :=
  wittStructureInt_vars _ _ _


theorem wittZSMul_vars (m : ℤ) (n : ℕ) :
    (wittZSMul p m n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) :=
  wittStructureInt_vars _ _ _


theorem wittPow_vars (m : ℕ) (n : ℕ) : (wittPow p m n).vars ⊆ Finset.univ ×ˢ Finset.range (n + 1) :=
  wittStructureInt_vars _ _ _


