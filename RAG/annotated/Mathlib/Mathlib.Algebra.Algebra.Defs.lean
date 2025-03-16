/-- An associative unital `R`-algebra is a semiring `A` equipped with a map into its center `R → A`.

See the implementation notes in this file for discussion of the details of this definition.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): unsupported @[nolint has_nonempty_instance]
class Algebra (R : Type u) (A : Type v) [CommSemiring R] [Semiring A] extends SMul R A,
  R →+* A where
  commutes' : ∀ r x, toRingHom r * x = x * toRingHom r
  smul_def' : ∀ r x, r • x = toRingHom r * x


/-- Embedding `R →+* A` given by `Algebra` structure. -/
def algebraMap (R : Type u) (A : Type v) [CommSemiring R] [Semiring A] [Algebra R A] : R →+* A :=
  Algebra.toRingHom


/-- Coercion from a commutative semiring to an algebra over this semiring. -/
@[coe, reducible]
def Algebra.cast {R A : Type*} [CommSemiring R] [Semiring A] [Algebra R A] : R → A :=
  algebraMap R A


scoped instance coeHTCT (R A : Type*) [CommSemiring R] [Semiring A] [Algebra R A] :
    CoeHTCT R A :=
  ⟨Algebra.cast⟩


@[norm_cast]
theorem coe_zero : (↑(0 : R) : A) = 0 :=
  map_zero (algebraMap R A)


@[norm_cast]
theorem coe_one : (↑(1 : R) : A) = 1 :=
  map_one (algebraMap R A)


@[norm_cast]
theorem coe_natCast (a : ℕ) : (↑(a : R) : A) = a :=
  map_natCast (algebraMap R A) a


@[norm_cast]
theorem coe_add (a b : R) : (↑(a + b : R) : A) = ↑a + ↑b :=
  map_add (algebraMap R A) a b


@[norm_cast]
theorem coe_mul (a b : R) : (↑(a * b : R) : A) = ↑a * ↑b :=
  map_mul (algebraMap R A) a b


@[norm_cast]
theorem coe_pow (a : R) (n : ℕ) : (↑(a ^ n : R) : A) = (a : A) ^ n :=
  map_pow (algebraMap R A) _ _


@[norm_cast]
theorem coe_neg (x : R) : (↑(-x : R) : A) = -↑x :=
  map_neg (algebraMap R A) x


@[norm_cast]
theorem coe_sub (a b : R) :
    (↑(a - b : R) : A) = ↑a - ↑b :=
  map_sub (algebraMap R A) a b


/-- Creating an algebra from a morphism to the center of a semiring. -/
def RingHom.toAlgebra' {R S} [CommSemiring R] [Semiring S] (i : R →+* S)
    (h : ∀ c x, i c * x = x * i c) : Algebra R S where
  smul c x := i c * x
  commutes' := h
  smul_def' _ _ := rfl
  toRingHom := i

-- just simple lemmas for a declaration that is itself primed, no need for docstrings

set_option linter.docPrime false in
theorem RingHom.smul_toAlgebra' {R S} [CommSemiring R] [Semiring S] (i : R →+* S)
    (h : ∀ c x, i c * x = x * i c) (r : R) (s : S) :
    let _ := RingHom.toAlgebra' i h
    r • s = i r * s := rfl


set_option linter.docPrime false in
theorem RingHom.algebraMap_toAlgebra' {R S} [CommSemiring R] [Semiring S] (i : R →+* S)
    (h : ∀ c x, i c * x = x * i c) :
    @algebraMap R S _ _ (i.toAlgebra' h) = i :=
  rfl


/-- Creating an algebra from a morphism to a commutative semiring. -/
def RingHom.toAlgebra {R S} [CommSemiring R] [CommSemiring S] (i : R →+* S) : Algebra R S :=
  i.toAlgebra' fun _ => mul_comm _


theorem RingHom.algebraMap_toAlgebra {R S} [CommSemiring R] [CommSemiring S] (i : R →+* S) :
    @algebraMap R S _ _ i.toAlgebra = i :=
  rfl


/-- Let `R` be a commutative semiring, let `A` be a semiring with a `Module R` structure.
If `(r • 1) * x = x * (r • 1) = r • x` for all `r : R` and `x : A`, then `A` is an `Algebra`
over `R`.

See note [reducible non-instances]. -/
abbrev ofModule' [CommSemiring R] [Semiring A] [Module R A]
    (h₁ : ∀ (r : R) (x : A), r • (1 : A) * x = r • x)
    (h₂ : ∀ (r : R) (x : A), x * r • (1 : A) = r • x) : Algebra R A where
  toFun r := r • (1 : A)
  map_one' := one_smul _ _
                       /-
                         R : Type u
                         S : Type v
                         A : Type w
                         B : Type u_1
                         inst✝² : CommSemiring R
                         inst✝¹ : Semiring A
                         inst✝ : Module R A
                         h₁ : ∀ (r : R) (x : A), Eq (HMul.hMul (HSMul.hSMul r 1) x) (HSMul.hSMul r x)
                         h₂ : ∀ (r : R) (x : A), Eq (HMul.hMul x (HSMul.hSMul r 1)) (HSMul.hSMul r x)
                         r₁ r₂ : R
                         ⊢ Eq ({ toFun := fun r => HSMul.hSMul r 1, map_one' := ⋯ }.toFun (HMul.hMul r₁ …
                       -/
  map_mul' r₁ r₂ := by simp only [h₁, mul_smul]
                       /-
                         🎉 no goals
                       -/
  map_zero' := zero_smul _ _
  map_add' r₁ r₂ := add_smul r₁ r₂ 1
                      /-
                        R : Type u
                        S : Type v
                        A : Type w
                        B : Type u_1
                        inst✝² : CommSemiring R
                        inst✝¹ : Semiring A
                        inst✝ : Module R A
                        h₁ : ∀ (r : R) (x : A), Eq (HMul.hMul (HSMul.hSMul r 1) x) (HSMul.hSMul r x)
                        h₂ : ∀ (r : R) (x : A), Eq (HMul.hMul x (HSMul.hSMul r 1)) (HSMul.hSMul r x)
                        r : R
                        x : A
                        ⊢ Eq (HMul.hMul ({ toFun := fun r => HSMul.hSMul r 1, map_one' := ⋯, map_mul'  …
                      -/
  commutes' r x := by simp [h₁, h₂]
                      /-
                        🎉 no goals
                      -/
                      /-
                        R : Type u
                        S : Type v
                        A : Type w
                        B : Type u_1
                        inst✝² : CommSemiring R
                        inst✝¹ : Semiring A
                        inst✝ : Module R A
                        h₁ : ∀ (r : R) (x : A), Eq (HMul.hMul (HSMul.hSMul r 1) x) (HSMul.hSMul r x)
                        h₂ : ∀ (r : R) (x : A), Eq (HMul.hMul x (HSMul.hSMul r 1)) (HSMul.hSMul r x)
                        r : R
                        x : A
                        ⊢ Eq (HSMul.hSMul r x) (HMul.hMul ({ toFun := fun r => HSMul.hSMul r 1, map_on …
                      -/
  smul_def' r x := by simp [h₁]
                      /-
                        🎉 no goals
                      -/


/-- Let `R` be a commutative semiring, let `A` be a semiring with a `Module R` structure.
If `(r • x) * y = x * (r • y) = r • (x * y)` for all `r : R` and `x y : A`, then `A`
is an `Algebra` over `R`.

See note [reducible non-instances]. -/
abbrev ofModule [CommSemiring R] [Semiring A] [Module R A]
    (h₁ : ∀ (r : R) (x y : A), r • x * y = r • (x * y))
    (h₂ : ∀ (r : R) (x y : A), x * r • y = r • (x * y)) : Algebra R A :=
                           /-
                             R : Type u
                             S : Type v
                             A : Type w
                             B : Type u_1
                             inst✝² : CommSemiring R
                             inst✝¹ : Semiring A
                             inst✝ : Module R A
                             h₁ : ∀ (r : R) (x y : A), Eq (HMul.hMul (HSMul.hSMul r x) y) (HSMul.hSMul r (H …
                             h₂ : ∀ (r : R) (x y : A), Eq (HMul.hMul x (HSMul.hSMul r y)) (HSMul.hSMul r (H …
                             r : R
                             x : A
                             ⊢ Eq (HMul.hMul (HSMul.hSMul r 1) x) (HSMul.hSMul r x)
                           -/
                           /-
                             🎉 no goals
                           -/
  ofModule' (fun r x => by rw [h₁, one_mul]) fun r x => by rw [h₂, mul_one]
                                                           /-
                                                             🎉 no goals
                                                           -/


/-- To prove two algebra structures on a fixed `[CommSemiring R] [Semiring A]` agree,
it suffices to check the `algebraMap`s agree.
-/
@[ext]
theorem algebra_ext {R : Type*} [CommSemiring R] {A : Type*} [Semiring A] (P Q : Algebra R A)
    (h : ∀ r : R, (haveI := P; algebraMap R A r) = haveI := Q; algebraMap R A r) :
    P = Q := by
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    A : Type u_3
    inst✝ : Semiring A
    P Q : Algebra R A
    h : ∀ (r : R), Eq ((algebraMap R A) r) ((algebraMap R A) r)
    ⊢ Eq P Q
  -/
  replace h : P.toRingHom = Q.toRingHom := DFunLike.ext _ _ h
  have h' : (haveI := P; (· • ·) : R → A → A) = (haveI := Q; (· • ·) : R → A → A) := by
    funext r a
    rw [P.smul_def', Q.smul_def', h]
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    A : Type u_3
    inst✝ : Semiring A
    P Q : Algebra R A
    h : Eq Algebra.toRingHom Algebra.toRingHom
    h' : Eq (fun x1 x2 => HSMul.hSMul x1 x2) fun x1 x2 => HSMul.hSMul x1 x2
    ⊢ Eq P Q
  -/
  rcases P with @⟨⟨P⟩⟩
  /-
    case mk.mk
    R : Type u_2
    inst✝¹ : CommSemiring R
    A : Type u_3
    inst✝ : Semiring A
    Q : Algebra R A
    toRingHom✝ : RingHom R A
    commutes'✝ : ∀ (r : R) (x : A), Eq (HMul.hMul (toRingHom✝ r) x) (HMul.hMul x ( …
    P : R → A → A
    smul_def'✝ : ∀ (r : R) (x : A), Eq (HSMul.hSMul r x) (HMul.hMul (toRingHom✝ r) …
    h : Eq Algebra.toRingHom Algebra.toRingHom
    h' : Eq (fun x1 x2 => HSMul.hSMul x1 x2) fun x1 x2 => HSMul.hSMul x1 x2
    ⊢ Eq (Algebra.mk toRingHom✝ commutes'✝ smul_def'✝) Q
  -/
  rcases Q with @⟨⟨Q⟩⟩
  /-
    case mk.mk.mk.mk
    R : Type u_2
    inst✝¹ : CommSemiring R
    A : Type u_3
    inst✝ : Semiring A
    toRingHom✝¹ : RingHom R A
    commutes'✝¹ : ∀ (r : R) (x : A), Eq (HMul.hMul (toRingHom✝¹ r) x) (HMul.hMul x …
    P : R → A → A
    smul_def'✝¹ : ∀ (r : R) (x : A), Eq (HSMul.hSMul r x) (HMul.hMul (toRingHom✝¹  …
    toRingHom✝ : RingHom R A
    commutes'✝ : ∀ (r : R) (x : A), Eq (HMul.hMul (toRingHom✝ r) x) (HMul.hMul x ( …
    Q : R → A → A
    smul_def'✝ : ∀ (r : R) (x : A), Eq (HSMul.hSMul r x) (HMul.hMul (toRingHom✝ r) …
    h : Eq Algebra.toRingHom Algebra.toRingHom
    h' : Eq (fun x1 x2 => HSMul.hSMul x1 x2) fun x1 x2 => HSMul.hSMul x1 x2
    ⊢ Eq (Algebra.mk toRingHom✝¹ commutes'✝¹ smul_def'✝¹) (Algebra.mk toRingHom✝ c …
  -/
  congr
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 200) toModule {R A} {_ : CommSemiring R} {_ : Semiring A} [Algebra R A] :
    Module R A where
                   /-
                     R✝ : Type u
                     S : Type v
                     A✝ : Type w
                     B : Type u_1
                     inst✝⁶ : CommSemiring R✝
                     inst✝⁵ : CommSemiring S
                     inst✝⁴ : Semiring A✝
                     inst✝³ : Algebra R✝ A✝
                     inst✝² : Semiring B
                     inst✝¹ : Algebra R✝ B
                     R : Type ?u.17344
                     A : Type ?u.17347
                     x✝² : CommSemiring R
                     x✝¹ : Semiring A
                     inst✝ : Algebra R A
                     x✝ : A
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  one_smul _ := by simp [smul_def']
                   /-
                     🎉 no goals
                   -/
                 /-
                   R✝ : Type u
                   S : Type v
                   A✝ : Type w
                   B : Type u_1
                   inst✝⁶ : CommSemiring R✝
                   inst✝⁵ : CommSemiring S
                   inst✝⁴ : Semiring A✝
                   inst✝³ : Algebra R✝ A✝
                   inst✝² : Semiring B
                   inst✝¹ : Algebra R✝ B
                   R : Type ?u.17344
                   A : Type ?u.17347
                   x✝¹ : CommSemiring R
                   x✝ : Semiring A
                   inst✝ : Algebra R A
                   ⊢ ∀ (x y : R) (b : A), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMul.hSMul x (HSMu …
                 -/
  mul_smul := by simp [smul_def', mul_assoc]
                 /-
                   🎉 no goals
                 -/
                 /-
                   R✝ : Type u
                   S : Type v
                   A✝ : Type w
                   B : Type u_1
                   inst✝⁶ : CommSemiring R✝
                   inst✝⁵ : CommSemiring S
                   inst✝⁴ : Semiring A✝
                   inst✝³ : Algebra R✝ A✝
                   inst✝² : Semiring B
                   inst✝¹ : Algebra R✝ B
                   R : Type ?u.17344
                   A : Type ?u.17347
                   x✝¹ : CommSemiring R
                   x✝ : Semiring A
                   inst✝ : Algebra R A
                   ⊢ ∀ (a : R) (x y : A), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hS …
                 -/
                  /-
                    R✝ : Type u
                    S : Type v
                    A✝ : Type w
                    B : Type u_1
                    inst✝⁶ : CommSemiring R✝
                    inst✝⁵ : CommSemiring S
                    inst✝⁴ : Semiring A✝
                    inst✝³ : Algebra R✝ A✝
                    inst✝² : Semiring B
                    inst✝¹ : Algebra R✝ B
                    R : Type ?u.17344
                    A : Type ?u.17347
                    x✝¹ : CommSemiring R
                    x✝ : Semiring A
                    inst✝ : Algebra R A
                    ⊢ ∀ (a : R), Eq (HSMul.hSMul a 0) 0
                  -/
  smul_add := by simp [smul_def', mul_add]
                  /-
                    🎉 no goals
                  -/
                 /-
                   🎉 no goals
                 -/
  smul_zero := by simp [smul_def']
                 /-
                   R✝ : Type u
                   S : Type v
                   A✝ : Type w
                   B : Type u_1
                   inst✝⁶ : CommSemiring R✝
                   inst✝⁵ : CommSemiring S
                   inst✝⁴ : Semiring A✝
                   inst✝³ : Algebra R✝ A✝
                   inst✝² : Semiring B
                   inst✝¹ : Algebra R✝ B
                   R : Type ?u.17344
                   A : Type ?u.17347
                   x✝¹ : CommSemiring R
                   x✝ : Semiring A
                   inst✝ : Algebra R A
                   ⊢ ∀ (r s : R) (x : A), Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hS …
                 -/
  add_smul := by simp [smul_def', add_mul]
                 /-
                   🎉 no goals
                 -/
                  /-
                    R✝ : Type u
                    S : Type v
                    A✝ : Type w
                    B : Type u_1
                    inst✝⁶ : CommSemiring R✝
                    inst✝⁵ : CommSemiring S
                    inst✝⁴ : Semiring A✝
                    inst✝³ : Algebra R✝ A✝
                    inst✝² : Semiring B
                    inst✝¹ : Algebra R✝ B
                    R : Type ?u.17344
                    A : Type ?u.17347
                    x✝¹ : CommSemiring R
                    x✝ : Semiring A
                    inst✝ : Algebra R A
                    ⊢ ∀ (x : A), Eq (HSMul.hSMul 0 x) 0
                  -/
  zero_smul := by simp [smul_def']
                  /-
                    🎉 no goals
                  -/

-- Porting note: this caused deterministic timeouts later in mathlib3 but not in mathlib 4.
-- attribute [instance 0] Algebra.toSMul


theorem smul_def (r : R) (x : A) : r • x = algebraMap R A r * x :=
  Algebra.smul_def' r x


theorem algebraMap_eq_smul_one (r : R) : algebraMap R A r = r • (1 : A) :=
  calc
    algebraMap R A r = algebraMap R A r * 1 := (mul_one _).symm
    _ = r • (1 : A) := (Algebra.smul_def r 1).symm


theorem algebraMap_eq_smul_one' : ⇑(algebraMap R A) = fun r => r • (1 : A) :=
  funext algebraMap_eq_smul_one


/-- `mul_comm` for `Algebra`s when one element is from the base ring. -/
theorem commutes (r : R) (x : A) : algebraMap R A r * x = x * algebraMap R A r :=
  Algebra.commutes' r x


lemma commute_algebraMap_left (r : R) (x : A) : Commute (algebraMap R A r) x :=
  Algebra.commutes r x


lemma commute_algebraMap_right (r : R) (x : A) : Commute x (algebraMap R A r) :=
  (Algebra.commutes r x).symm


/-- `mul_left_comm` for `Algebra`s when one element is from the base ring. -/
theorem left_comm (x : A) (r : R) (y : A) :
    x * (algebraMap R A r * y) = algebraMap R A r * (x * y) := by
  /-
    R : Type u
    A : Type w
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    r : R
    y : A
    ⊢ Eq (HMul.hMul x (HMul.hMul ((algebraMap R A) r) y)) (HMul.hMul ((algebraMap  …
  -/
  rw [← mul_assoc, ← commutes, mul_assoc]
  /-
    🎉 no goals
  -/


/-- `mul_right_comm` for `Algebra`s when one element is from the base ring. -/
theorem right_comm (x : A) (r : R) (y : A) :
    x * algebraMap R A r * y = x * y * algebraMap R A r := by
  /-
    R : Type u
    A : Type w
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    r : R
    y : A
    ⊢ Eq (HMul.hMul (HMul.hMul x ((algebraMap R A) r)) y) (HMul.hMul (HMul.hMul x  …
  -/
  rw [mul_assoc, commutes, ← mul_assoc]
  /-
    🎉 no goals
  -/


instance _root_.IsScalarTower.right : IsScalarTower R A A :=
                   /-
                     R : Type u
                     S : Type v
                     A : Type w
                     B : Type u_1
                     inst✝⁵ : CommSemiring R
                     inst✝⁴ : CommSemiring S
                     inst✝³ : Semiring A
                     inst✝² : Algebra R A
                     inst✝¹ : Semiring B
                     inst✝ : Algebra R B
                     x : R
                     y z : A
                     ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
                   -/
  ⟨fun x y z => by rw [smul_eq_mul, smul_eq_mul, smul_def, smul_def, mul_assoc]⟩
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem _root_.RingHom.smulOneHom_eq_algebraMap : RingHom.smulOneHom = algebraMap R A :=
  RingHom.ext fun r => (algebraMap_eq_smul_one r).symm

-- TODO: set up `IsScalarTower.smulCommClass` earlier so that we can actually prove this using
-- `mul_smul_comm s x y`.


/-- This is just a special case of the global `mul_smul_comm` lemma that requires less typeclass
search (and was here first). -/
@[simp]
protected theorem mul_smul_comm (s : R) (x y : A) : x * s • y = s • (x * y) := by
  /-
    R : Type u
    A : Type w
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : R
    x y : A
    ⊢ Eq (HMul.hMul x (HSMul.hSMul s y)) (HSMul.hSMul s (HMul.hMul x y))
  -/
  rw [smul_def, smul_def, left_comm]
  /-
    🎉 no goals
  -/


/-- This is just a special case of the global `smul_mul_assoc` lemma that requires less typeclass
search (and was here first). -/
@[simp]
protected theorem smul_mul_assoc (r : R) (x y : A) : r • x * y = r • (x * y) :=
  smul_mul_assoc r x y


@[simp]
theorem _root_.smul_algebraMap {α : Type*} [Monoid α] [MulDistribMulAction α A]
    [SMulCommClass α R A] (a : α) (r : R) : a • algebraMap R A r = algebraMap R A r := by
  /-
    R : Type u
    A : Type w
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    α : Type u_2
    inst✝² : Monoid α
    inst✝¹ : MulDistribMulAction α A
    inst✝ : SMulCommClass α R A
    a : α
    r : R
    ⊢ Eq (HSMul.hSMul a ((algebraMap R A) r)) ((algebraMap R A) r)
  -/
  rw [algebraMap_eq_smul_one, smul_comm a r (1 : A), smul_one]
  /-
    🎉 no goals
  -/


/--
Compose an `Algebra` with a `RingHom`, with action `f s • m`.

This is the algebra version of `Module.compHom`.
-/
abbrev compHom : Algebra S A where
  smul s a := f s • a
  toRingHom := (algebraMap R A).comp f
  commutes' _ _ := Algebra.commutes _ _
  smul_def' _ _ := Algebra.smul_def _ _


theorem compHom_smul_def (s : S) (x : A) :
    letI := compHom A f
    s • x = f s • x := rfl


theorem compHom_algebraMap_eq :
    letI := compHom A f
    algebraMap S A = (algebraMap R A).comp f := rfl


theorem compHom_algebraMap_apply (s : S) :
    letI := compHom A f
    algebraMap S A s = (algebraMap R A) (f s) := rfl


/-- The canonical ring homomorphism `algebraMap R A : R →+* A` for any `R`-algebra `A`,
packaged as an `R`-linear map.
-/
protected def linearMap : R →ₗ[R] A :=
                                                   /-
                                                     R : Type u
                                                     S : Type v
                                                     A : Type w
                                                     B : Type u_1
                                                     inst✝⁵ : CommSemiring R
                                                     inst✝⁴ : CommSemiring S
                                                     inst✝³ : Semiring A
                                                     inst✝² : Algebra R A
                                                     inst✝¹ : Semiring B
                                                     inst✝ : Algebra R B
                                                     x y : R
                                                     ⊢ Eq ({ toFun := (↑↑__src✝).toFun, map_add' := ⋯ }.toFun (HSMul.hSMul x y)) (H …
                                                   -/
  { algebraMap R A with map_smul' := fun x y => by simp [Algebra.smul_def] }
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem linearMap_apply (r : R) : Algebra.linearMap R A r = algebraMap R A r :=
  rfl


theorem coe_linearMap : ⇑(Algebra.linearMap R A) = algebraMap R A :=
  rfl


/-- The identity map inducing an `Algebra` structure. -/
instance (priority := 1100) id : Algebra R R where
  -- We override `toFun` and `toSMul` because `RingHom.id` is not reducible and cannot
  -- be made so without a significant performance hit.
  -- see library note [reducible non-instances].
  toFun x := x
  toSMul := Mul.toSMul _
  __ := (RingHom.id R).toAlgebra


@[simp]
theorem map_eq_id : algebraMap R R = RingHom.id _ :=
  rfl


theorem map_eq_self (x : R) : algebraMap R R x = x :=
  rfl


@[simp]
theorem smul_eq_mul (x y : R) : x • y = x * y :=
  rfl


@[norm_cast]
theorem algebraMap.coe_smul (A B C : Type*) [SMul A B] [CommSemiring B] [Semiring C] [Algebra B C]
    [SMul A C] [IsScalarTower A B C] (a : A) (b : B) : (a • b : B) = a • (b : C) := calc
  ((a • b : B) : C) = (a • b) • 1 := Algebra.algebraMap_eq_smul_one _
  _ = a • (b • 1) := smul_assoc ..
  _ = a • (b : C) := congrArg _ (Algebra.algebraMap_eq_smul_one b).symm

