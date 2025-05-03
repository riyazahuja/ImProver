/-- This inductive type is used to express representatives of the free algebra.
-/
inductive Pre
  | of : X → Pre
  | ofScalar : R → Pre
  | add : Pre → Pre → Pre
  | mul : Pre → Pre → Pre


instance : Inhabited (Pre R X) := ⟨ofScalar 0⟩

-- Note: These instances are only used to simplify the notation.

/-- Coercion from `X` to `Pre R X`. Note: Used for notation only. -/
def hasCoeGenerator : Coe X (Pre R X) := ⟨of⟩


/-- Coercion from `R` to `Pre R X`. Note: Used for notation only. -/
def hasCoeSemiring : Coe R (Pre R X) := ⟨ofScalar⟩


/-- Multiplication in `Pre R X` defined as `Pre.mul`. Note: Used for notation only. -/
def hasMul : Mul (Pre R X) := ⟨mul⟩


/-- Addition in `Pre R X` defined as `Pre.add`. Note: Used for notation only. -/
def hasAdd : Add (Pre R X) := ⟨add⟩


/-- Zero in `Pre R X` defined as the image of `0` from `R`. Note: Used for notation only. -/
def hasZero : Zero (Pre R X) := ⟨ofScalar 0⟩


/-- One in `Pre R X` defined as the image of `1` from `R`. Note: Used for notation only. -/
def hasOne : One (Pre R X) := ⟨ofScalar 1⟩


/-- Scalar multiplication defined as multiplication by the image of elements from `R`.
Note: Used for notation only.
-/
def hasSMul : SMul R (Pre R X) := ⟨fun r m ↦ mul (ofScalar r) m⟩


/-- Given a function from `X` to an `R`-algebra `A`, `lift_fun` provides a lift of `f` to a function
from `Pre R X` to `A`. This is mainly used in the construction of `FreeAlgebra.lift`.
-/
-- Porting note: recOn was replaced to preserve computability, see https://github.com/leanprover/lean4/issues/2049
def liftFun {A : Type*} [Semiring A] [Algebra R A] (f : X → A) :
    Pre R X → A
  | .of t => f t
  | .add a b => liftFun f a + liftFun f b
  | .mul a b => liftFun f a * liftFun f b
  | .ofScalar c => algebraMap _ _ c


/-- An inductively defined relation on `Pre R X` used to force the initial algebra structure on
the associated quotient.
-/
inductive Rel : Pre R X → Pre R X → Prop
  -- force `ofScalar` to be a central semiring morphism
  | add_scalar {r s : R} : Rel (↑(r + s)) (↑r + ↑s)
  | mul_scalar {r s : R} : Rel (↑(r * s)) (↑r * ↑s)
  | central_scalar {r : R} {a : Pre R X} : Rel (r * a) (a * r)

  -- commutative additive semigroup
  | add_assoc {a b c : Pre R X} : Rel (a + b + c) (a + (b + c))
  | add_comm {a b : Pre R X} : Rel (a + b) (b + a)
  | zero_add {a : Pre R X} : Rel (0 + a) a

  -- multiplicative monoid
  | mul_assoc {a b c : Pre R X} : Rel (a * b * c) (a * (b * c))
  | one_mul {a : Pre R X} : Rel (1 * a) a
  | mul_one {a : Pre R X} : Rel (a * 1) a

  -- distributivity
  | left_distrib {a b c : Pre R X} : Rel (a * (b + c)) (a * b + a * c)
  | right_distrib {a b c : Pre R X} :
      Rel ((a + b) * c) (a * c + b * c)

  -- other relations needed for semiring
  | zero_mul {a : Pre R X} : Rel (0 * a) 0
  | mul_zero {a : Pre R X} : Rel (a * 0) 0

  -- compatibility
  | add_compat_left {a b c : Pre R X} : Rel a b → Rel (a + c) (b + c)
  | add_compat_right {a b c : Pre R X} : Rel a b → Rel (c + a) (c + b)
  | mul_compat_left {a b c : Pre R X} : Rel a b → Rel (a * c) (b * c)
  | mul_compat_right {a b c : Pre R X} : Rel a b → Rel (c * a) (c * b)


/-- The free algebra for the type `X` over the commutative semiring `R`.
-/
def FreeAlgebra :=
  Quot (FreeAlgebra.Rel R X)


instance instSMul {A} [CommSemiring A] [Algebra R A] : SMul R (FreeAlgebra A X) where
  smul r := Quot.map (HMul.hMul (algebraMap R A r : Pre A X)) fun _ _ ↦ Rel.mul_compat_right


instance instZero : Zero (FreeAlgebra R X) where zero := Quot.mk _ 0


instance instOne : One (FreeAlgebra R X) where one := Quot.mk _ 1


instance instAdd : Add (FreeAlgebra R X) where
  add := Quot.map₂ HAdd.hAdd (fun _ _ _ ↦ Rel.add_compat_right) fun _ _ _ ↦ Rel.add_compat_left


instance instMul : Mul (FreeAlgebra R X) where
  mul := Quot.map₂ HMul.hMul (fun _ _ _ ↦ Rel.mul_compat_right) fun _ _ _ ↦ Rel.mul_compat_left

-- `Quot.mk` is an implementation detail of `FreeAlgebra`, so this lemma is private

private theorem mk_mul (x y : Pre R X) :
    Quot.mk (Rel R X) (x * y) = (HMul.hMul (self := instHMul (α := FreeAlgebra R X))
    (Quot.mk (Rel R X) x) (Quot.mk (Rel R X) y)) :=
  rfl


instance instMonoidWithZero : MonoidWithZero (FreeAlgebra R X) where
  mul_assoc := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a b c : FreeAlgebra R X), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a ( …
    -/
    rintro ⟨⟩ ⟨⟩ ⟨⟩
    /-
      case mk.mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝³ : FreeAlgebra R X
      a✝² : FreeAlgebra.Pre R X
      b✝ : FreeAlgebra R X
      a✝¹ : FreeAlgebra.Pre R X
      c✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HMul.hMul (HMul.hMul (Quot.mk (FreeAlgebra.Rel R X) a✝²) (Quot.mk (FreeA …
    -/
    exact Quot.sound Rel.mul_assoc
    /-
      🎉 no goals
    -/
  one := Quot.mk _ 1
  one_mul := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a : FreeAlgebra R X), Eq (HMul.hMul 1 a) a
    -/
    rintro ⟨⟩
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝¹ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HMul.hMul 1 (Quot.mk (FreeAlgebra.Rel R X) a✝)) (Quot.mk (FreeAlgebra.Re …
    -/
    exact Quot.sound Rel.one_mul
    /-
      🎉 no goals
    -/
  mul_one := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a : FreeAlgebra R X), Eq (HMul.hMul a 1) a
    -/
    rintro ⟨⟩
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝¹ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HMul.hMul (Quot.mk (FreeAlgebra.Rel R X) a✝) 1) (Quot.mk (FreeAlgebra.Re …
    -/
    exact Quot.sound Rel.mul_one
    /-
      🎉 no goals
    -/
  zero_mul := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a : FreeAlgebra R X), Eq (HMul.hMul 0 a) 0
    -/
    rintro ⟨⟩
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝¹ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HMul.hMul 0 (Quot.mk (FreeAlgebra.Rel R X) a✝)) 0
    -/
    exact Quot.sound Rel.zero_mul
    /-
      🎉 no goals
    -/
  mul_zero := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a : FreeAlgebra R X), Eq (HMul.hMul a 0) 0
    -/
    rintro ⟨⟩
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝¹ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HMul.hMul (Quot.mk (FreeAlgebra.Rel R X) a✝) 0) 0
    -/
    exact Quot.sound Rel.mul_zero
    /-
      🎉 no goals
    -/


instance instDistrib : Distrib (FreeAlgebra R X) where
  left_distrib := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a b c : FreeAlgebra R X), Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HM …
    -/
    rintro ⟨⟩ ⟨⟩ ⟨⟩
    /-
      case mk.mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝³ : FreeAlgebra R X
      a✝² : FreeAlgebra.Pre R X
      b✝ : FreeAlgebra R X
      a✝¹ : FreeAlgebra.Pre R X
      c✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HMul.hMul (Quot.mk (FreeAlgebra.Rel R X) a✝²) (HAdd.hAdd (Quot.mk (FreeA …
    -/
    exact Quot.sound Rel.left_distrib
    /-
      🎉 no goals
    -/
  right_distrib := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a b c : FreeAlgebra R X), Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HM …
    -/
    rintro ⟨⟩ ⟨⟩ ⟨⟩
    /-
      case mk.mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝³ : FreeAlgebra R X
      a✝² : FreeAlgebra.Pre R X
      b✝ : FreeAlgebra R X
      a✝¹ : FreeAlgebra.Pre R X
      c✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HMul.hMul (HAdd.hAdd (Quot.mk (FreeAlgebra.Rel R X) a✝²) (Quot.mk (FreeA …
    -/
    exact Quot.sound Rel.right_distrib
    /-
      🎉 no goals
    -/


instance instAddCommMonoid : AddCommMonoid (FreeAlgebra R X) where
  add_assoc := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a b c : FreeAlgebra R X), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a ( …
    -/
    rintro ⟨⟩ ⟨⟩ ⟨⟩
    /-
      case mk.mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝³ : FreeAlgebra R X
      a✝² : FreeAlgebra.Pre R X
      b✝ : FreeAlgebra R X
      a✝¹ : FreeAlgebra.Pre R X
      c✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Quot.mk (FreeAlgebra.Rel R X) a✝²) (Quot.mk (FreeA …
    -/
    exact Quot.sound Rel.add_assoc
    /-
      🎉 no goals
    -/
  zero_add := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a : FreeAlgebra R X), Eq (HAdd.hAdd 0 a) a
    -/
    rintro ⟨⟩
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝¹ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HAdd.hAdd 0 (Quot.mk (FreeAlgebra.Rel R X) a✝)) (Quot.mk (FreeAlgebra.Re …
    -/
    exact Quot.sound Rel.zero_add
    /-
      🎉 no goals
    -/
  add_zero := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a : FreeAlgebra R X), Eq (HAdd.hAdd a 0) a
    -/
    rintro ⟨⟩
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝¹ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HAdd.hAdd (Quot.mk (FreeAlgebra.Rel R X) a✝) 0) (Quot.mk (FreeAlgebra.Re …
    -/
    change Quot.mk _ _ = _
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝¹ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (Quot.mk (FreeAlgebra.Rel R X) (HAdd.hAdd a✝ 0)) (Quot.mk (FreeAlgebra.Re …
    -/
    rw [Quot.sound Rel.add_comm, Quot.sound Rel.zero_add]
    /-
      🎉 no goals
    -/
  add_comm := by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (a b : FreeAlgebra R X), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
    -/
    rintro ⟨⟩ ⟨⟩
    /-
      case mk.mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      a✝² : FreeAlgebra R X
      a✝¹ : FreeAlgebra.Pre R X
      b✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (HAdd.hAdd (Quot.mk (FreeAlgebra.Rel R X) a✝¹) (Quot.mk (FreeAlgebra.Rel  …
    -/
    exact Quot.sound Rel.add_comm
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      ⊢ ∀ (x : FreeAlgebra R X), Eq ((fun x1 x2 => HSMul.hSMul x1 x2) 0 x) 0
    -/
    /-
      🎉 no goals
    -/
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      x✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) 0 (Quot.mk (FreeAlgebra.Rel R X) a✝)) 0
    -/
  nsmul := (· • ·)
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      x✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (Quot.mk (FreeAlgebra.Rel R X) (HMul.hMul (FreeAlgebra.Pre.ofScalar ((alg …
    -/
  nsmul_zero := by
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      x✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq (Quot.mk (FreeAlgebra.Rel R X) (HMul.hMul (FreeAlgebra.Pre.ofScalar 0) a✝ …
    -/
    rintro ⟨⟩
    /-
      🎉 no goals
    -/
    change Quot.mk _ (_ * _) = _
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      n : Nat
      ⊢ ∀ (x : FreeAlgebra R X), Eq ((fun x1 x2 => HSMul.hSMul x1 x2) (HAdd.hAdd n 1 …
    -/
    rw [map_zero]
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      n : Nat
      x✝ : FreeAlgebra R X
      a : FreeAlgebra.Pre R X
      ⊢ Eq ((fun x1 x2 => HSMul.hSMul x1 x2) (HAdd.hAdd n 1) (Quot.mk (FreeAlgebra.R …
    -/
    exact Quot.sound Rel.zero_mul
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      n : Nat
      x✝ : FreeAlgebra R X
      a : FreeAlgebra.Pre R X
      ⊢ Eq (Quot.mk (FreeAlgebra.Rel R X) (HMul.hMul (FreeAlgebra.Pre.ofScalar ((alg …
    -/
  nsmul_succ n := by
    /-
      case mk
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      n : Nat
      x✝ : FreeAlgebra R X
      a : FreeAlgebra.Pre R X
      ⊢ Eq (HMul.hMul (Quot.mk (FreeAlgebra.Rel R X) (FreeAlgebra.Pre.ofScalar (HAdd …
    -/
    rintro ⟨a⟩
    /-
      case mk.e_a
      R : Type u_1
      inst✝ : CommSemiring R
      X : Type u_2
      n : Nat
      x✝ : FreeAlgebra R X
      a : FreeAlgebra.Pre R X
      ⊢ Eq (Quot.mk (FreeAlgebra.Rel R X) (FreeAlgebra.Pre.ofScalar (HAdd.hAdd ((alg …
    -/
    dsimp only [HSMul.hSMul, instSMul, Quot.map]
    /-
      🎉 no goals
    -/
    rw [map_add, map_one, mk_mul, mk_mul, ← add_one_mul (_ : FreeAlgebra R X)]
    congr 1
    exact Quot.sound Rel.add_scalar


instance : Semiring (FreeAlgebra R X) where
  __ := instMonoidWithZero R X
  __ := instAddCommMonoid R X
  __ := instDistrib R X
  natCast n := Quot.mk _ (n : R)
                     /-
                       R : Type u_1
                       inst✝ : CommSemiring R
                       X : Type u_2
                       ⊢ Eq (NatCast.natCast 0) 0
                     -/
  natCast_zero := by simp; rfl
                           /-
                             🎉 no goals
                           -/
                       /-
                         R : Type u_1
                         inst✝ : CommSemiring R
                         X : Type u_2
                         n : Nat
                         ⊢ Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCast n) 1)
                       -/
  natCast_succ n := by simpa using Quot.sound Rel.add_scalar
                       /-
                         🎉 no goals
                       -/


instance : Inhabited (FreeAlgebra R X) :=
  ⟨0⟩


instance instAlgebra {A} [CommSemiring A] [Algebra R A] : Algebra R (FreeAlgebra A X) where
  toRingHom := ({
      toFun := fun r => Quot.mk _ r
      map_one' := rfl
      map_mul' := fun _ _ => Quot.sound Rel.mul_scalar
      map_zero' := rfl
      map_add' := fun _ _ => Quot.sound Rel.add_scalar } : A →+* FreeAlgebra A X).comp
      (algebraMap R A)
  commutes' _ := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type ?u.21926
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      x✝ : R
      ⊢ ∀ (x : FreeAlgebra A X), Eq (HMul.hMul (({ toFun := fun r => Quot.mk (FreeAl …
    -/
    rintro ⟨⟩
    /-
      case mk
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type ?u.21926
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      x✝¹ : R
      x✝ : FreeAlgebra A X
      a✝ : FreeAlgebra.Pre A X
      ⊢ Eq (HMul.hMul (({ toFun := fun r => Quot.mk (FreeAlgebra.Rel A X) (FreeAlgeb …
    -/
    exact Quot.sound Rel.central_scalar
    /-
      🎉 no goals
    -/
  smul_def' _ _ := rfl

-- verify there is no diamond at `default` transparency but we will need
-- `reducible_and_instances` which currently fails https://github.com/leanprover-community/mathlib4/issues/10906

instance {R S A} [CommSemiring R] [CommSemiring S] [CommSemiring A]
    [SMul R S] [Algebra R A] [Algebra S A] [IsScalarTower R S A] :
    IsScalarTower R S (FreeAlgebra A X) where
  smul_assoc r s x := by
    /-
      R✝ : Type u_1
      inst✝⁷ : CommSemiring R✝
      X : Type u_2
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : CommSemiring A
      inst✝³ : SMul R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      r : R
      s : S
      x : FreeAlgebra A X
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r s) x) (HSMul.hSMul r (HSMul.hSMul s x))
    -/
    change algebraMap S A (r • s) • x = algebraMap R A _ • (algebraMap S A _ • x)
    /-
      R✝ : Type u_1
      inst✝⁷ : CommSemiring R✝
      X : Type u_2
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : CommSemiring A
      inst✝³ : SMul R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      r : R
      s : S
      x : FreeAlgebra A X
      ⊢ Eq (HSMul.hSMul ((algebraMap S A) (HSMul.hSMul r s)) x) (HSMul.hSMul ((algeb …
    -/
    rw [← smul_assoc]
    /-
      R✝ : Type u_1
      inst✝⁷ : CommSemiring R✝
      X : Type u_2
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : CommSemiring A
      inst✝³ : SMul R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      r : R
      s : S
      x : FreeAlgebra A X
      ⊢ Eq (HSMul.hSMul ((algebraMap S A) (HSMul.hSMul r s)) x) (HSMul.hSMul (HSMul. …
    -/
    congr
    /-
      case e_a
      R✝ : Type u_1
      inst✝⁷ : CommSemiring R✝
      X : Type u_2
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : CommSemiring A
      inst✝³ : SMul R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      r : R
      s : S
      x : FreeAlgebra A X
      ⊢ Eq ((algebraMap S A) (HSMul.hSMul r s)) (HSMul.hSMul ((algebraMap R A) r) (( …
    -/
    simp only [Algebra.algebraMap_eq_smul_one, smul_eq_mul]
    /-
      case e_a
      R✝ : Type u_1
      inst✝⁷ : CommSemiring R✝
      X : Type u_2
      R : Type u_3
      S : Type u_4
      A : Type u_5
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : CommSemiring A
      inst✝³ : SMul R S
      inst✝² : Algebra R A
      inst✝¹ : Algebra S A
      inst✝ : IsScalarTower R S A
      r : R
      s : S
      x : FreeAlgebra A X
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r s) 1) (HMul.hMul (HSMul.hSMul r 1) (HSMul.hSM …
    -/
    rw [smul_assoc, ← smul_one_mul]
    /-
      🎉 no goals
    -/


instance {R S A} [CommSemiring R] [CommSemiring S] [CommSemiring A] [Algebra R A] [Algebra S A] :
    SMulCommClass R S (FreeAlgebra A X) where
  smul_comm r s x := smul_comm (algebraMap R A r) (algebraMap S A s) x


instance {S : Type*} [CommRing S] : Ring (FreeAlgebra S X) :=
  Algebra.semiringToRing S

-- verify there is no diamond but we will need
-- `reducible_and_instances` which currently fails https://github.com/leanprover-community/mathlib4/issues/10906

/-- The canonical function `X → FreeAlgebra R X`.
-/
irreducible_def ι : X → FreeAlgebra R X := fun m ↦ Quot.mk _ m


@[simp]
                                                                             /-
                                                                               R : Type u_1
                                                                               inst✝ : CommSemiring R
                                                                               X : Type u_2
                                                                               m : X
                                                                               ⊢ Eq (Quot.mk (FreeAlgebra.Rel R X) (FreeAlgebra.Pre.of m)) (FreeAlgebra.ι R m)
                                                                             -/
theorem quot_mk_eq_ι (m : X) : Quot.mk (FreeAlgebra.Rel R X) m = ι R m := by rw [ι_def]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- Internal definition used to define `lift` -/
private def liftAux (f : X → A) : FreeAlgebra R X →ₐ[R] A where
  toFun a :=
    Quot.liftOn a (liftFun _ _ f) fun a b h ↦ by
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : X → A
        a✝ : FreeAlgebra R X
        a b : FreeAlgebra.Pre R X
        h : FreeAlgebra.Rel R X a b
        ⊢ Eq (FreeAlgebra.liftFun R X f a) (FreeAlgebra.liftFun R X f b)
      -/
      induction h
        /-
          case add_scalar
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝ : FreeAlgebra R X
          a b : FreeAlgebra.Pre R X
          r✝ s✝ : R
          ⊢ Eq (FreeAlgebra.liftFun R X f (FreeAlgebra.Pre.ofScalar (HAdd.hAdd r✝ s✝)))  …
        -/
      · exact (algebraMap R A).map_add _ _
        /-
          🎉 no goals
        -/
        /-
          case mul_scalar
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝ : FreeAlgebra R X
          a b : FreeAlgebra.Pre R X
          r✝ s✝ : R
          ⊢ Eq (FreeAlgebra.liftFun R X f (FreeAlgebra.Pre.ofScalar (HMul.hMul r✝ s✝)))  …
        -/
      · exact (algebraMap R A).map_mul _ _
        /-
          🎉 no goals
        -/
        /-
          case central_scalar
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b : FreeAlgebra.Pre R X
          r✝ : R
          a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HMul.hMul (FreeAlgebra.Pre.ofScalar r✝) a✝))  …
        -/
      · apply Algebra.commutes
        /-
          🎉 no goals
        -/
        /-
          case add_assoc
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ c✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HAdd.hAdd (HAdd.hAdd a✝ b✝) c✝)) (FreeAlgebra …
        -/
      · change _ + _ + _ = _ + (_ + _)
        /-
          case add_assoc
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ c✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HAdd.hAdd (HAdd.hAdd (FreeAlgebra.liftFun R X f a✝) (FreeAlgebra.liftFun …
        -/
        rw [add_assoc]
        /-
          🎉 no goals
        -/
        /-
          case add_comm
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HAdd.hAdd a✝ b✝)) (FreeAlgebra.liftFun R X f  …
        -/
      · change _ + _ = _ + _
        /-
          case add_comm
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HAdd.hAdd (FreeAlgebra.liftFun R X f a✝) (FreeAlgebra.liftFun R X f b✝)) …
        -/
        rw [add_comm]
        /-
          🎉 no goals
        -/
        /-
          case zero_add
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HAdd.hAdd 0 a✝)) (FreeAlgebra.liftFun R X f a✝)
        -/
      · change algebraMap _ _ _ + liftFun R X f _ = liftFun R X f _
        /-
          case zero_add
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HAdd.hAdd ((algebraMap R A) 0) (FreeAlgebra.liftFun R X f a✝)) (FreeAlge …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case mul_assoc
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ c✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HMul.hMul (HMul.hMul a✝ b✝) c✝)) (FreeAlgebra …
        -/
      · change _ * _ * _ = _ * (_ * _)
        /-
          case mul_assoc
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ c✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HMul.hMul (HMul.hMul (FreeAlgebra.liftFun R X f a✝) (FreeAlgebra.liftFun …
        -/
        rw [mul_assoc]
        /-
          🎉 no goals
        -/
        /-
          case one_mul
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HMul.hMul 1 a✝)) (FreeAlgebra.liftFun R X f a✝)
        -/
      · change algebraMap _ _ _ * liftFun R X f _ = liftFun R X f _
        /-
          case one_mul
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HMul.hMul ((algebraMap R A) 1) (FreeAlgebra.liftFun R X f a✝)) (FreeAlge …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case mul_one
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HMul.hMul a✝ 1)) (FreeAlgebra.liftFun R X f a✝)
        -/
      · change liftFun R X f _ * algebraMap _ _ _ = liftFun R X f _
        /-
          case mul_one
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HMul.hMul (FreeAlgebra.liftFun R X f a✝) ((algebraMap R A) 1)) (FreeAlge …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case left_distrib
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ c✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HMul.hMul a✝ (HAdd.hAdd b✝ c✝))) (FreeAlgebra …
        -/
      · change _ * (_ + _) = _ * _ + _ * _
        /-
          case left_distrib
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ c✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HMul.hMul (FreeAlgebra.liftFun R X f a✝) (HAdd.hAdd (FreeAlgebra.liftFun …
        -/
        rw [left_distrib]
        /-
          🎉 no goals
        -/
        /-
          case right_distrib
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ c✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HMul.hMul (HAdd.hAdd a✝ b✝) c✝)) (FreeAlgebra …
        -/
      · change (_ + _) * _ = _ * _ + _ * _
        /-
          case right_distrib
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ b✝ c✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HMul.hMul (HAdd.hAdd (FreeAlgebra.liftFun R X f a✝) (FreeAlgebra.liftFun …
        -/
        rw [right_distrib]
        /-
          🎉 no goals
        -/
        /-
          case zero_mul
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HMul.hMul 0 a✝)) (FreeAlgebra.liftFun R X f 0)
        -/
      · change algebraMap _ _ _ * _ = algebraMap _ _ _
        /-
          case zero_mul
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HMul.hMul ((algebraMap R A) 0) (FreeAlgebra.liftFun R X f a✝)) ((algebra …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case mul_zero
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (FreeAlgebra.liftFun R X f (HMul.hMul a✝ 0)) (FreeAlgebra.liftFun R X f 0)
        -/
      · change _ * algebraMap _ _ _ = algebraMap _ _ _
        /-
          case mul_zero
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          f : X → A
          a✝¹ : FreeAlgebra R X
          a b a✝ : FreeAlgebra.Pre R X
          ⊢ Eq (HMul.hMul (FreeAlgebra.liftFun R X f a✝) ((algebraMap R A) 0)) ((algebra …
        -/
        simp
        /-
          🎉 no goals
        -/
      repeat
        change liftFun R X f _ + liftFun R X f _ = _
        simp only [*]
        rfl
      repeat
        change liftFun R X f _ * liftFun R X f _ = _
        simp only [*]
        rfl
  map_one' := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      ⊢ Eq ((fun a => Quot.liftOn a (FreeAlgebra.liftFun R X f) ⋯) 1) 1
    -/
    change algebraMap _ _ _ = _
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      ⊢ Eq ((algebraMap R A) 1) 1
    -/
    simp
    /-
      🎉 no goals
    -/
  map_mul' := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      ⊢ ∀ (x y : FreeAlgebra R X), Eq ({ toFun := fun a => Quot.liftOn a (FreeAlgebr …
    -/
    rintro ⟨⟩ ⟨⟩
    /-
      case mk.mk
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      x✝ : FreeAlgebra R X
      a✝¹ : FreeAlgebra.Pre R X
      y✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq ({ toFun := fun a => Quot.liftOn a (FreeAlgebra.liftFun R X f) ⋯, map_one …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_zero' := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      ⊢ Eq ((↑{ toFun := fun a => Quot.liftOn a (FreeAlgebra.liftFun R X f) ⋯, map_o …
    -/
    dsimp
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      ⊢ Eq (Quot.liftOn 0 (FreeAlgebra.liftFun R X f) ⋯) 0
    -/
    change algebraMap _ _ _ = _
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      ⊢ Eq ((algebraMap R A) 0) 0
    -/
    simp
    /-
      🎉 no goals
    -/
  map_add' := by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      ⊢ ∀ (x y : FreeAlgebra R X), Eq ((↑{ toFun := fun a => Quot.liftOn a (FreeAlge …
    -/
    rintro ⟨⟩ ⟨⟩
    /-
      case mk.mk
      R : Type u_1
      inst✝² : CommSemiring R
      X : Type u_2
      A : Type u_3
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : X → A
      x✝ : FreeAlgebra R X
      a✝¹ : FreeAlgebra.Pre R X
      y✝ : FreeAlgebra R X
      a✝ : FreeAlgebra.Pre R X
      ⊢ Eq ((↑{ toFun := fun a => Quot.liftOn a (FreeAlgebra.liftFun R X f) ⋯, map_o …
    -/
    rfl
    /-
      🎉 no goals
    -/
                  /-
                    R : Type u_1
                    inst✝² : CommSemiring R
                    X : Type u_2
                    A : Type u_3
                    inst✝¹ : Semiring A
                    inst✝ : Algebra R A
                    f : X → A
                    ⊢ ∀ (r : R), Eq ((↑↑{ toFun := fun a => Quot.liftOn a (FreeAlgebra.liftFun R X …
                  -/
  commutes' := by tauto
                  /-
                    🎉 no goals
                  -/


/-- Given a function `f : X → A` where `A` is an `R`-algebra, `lift R f` is the unique lift
of `f` to a morphism of `R`-algebras `FreeAlgebra R X → A`.
-/
@[irreducible]
def lift : (X → A) ≃ (FreeAlgebra R X →ₐ[R] A) :=
  { toFun := liftAux R
    invFun := fun F ↦ F ∘ ι R
    left_inv := fun f ↦ by
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : X → A
        ⊢ Eq ((fun F => Function.comp (⇑F) (FreeAlgebra.ι R)) (FreeAlgebra.liftAux R f …
      -/
      ext
      /-
        case h
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : X → A
        x✝ : X
        ⊢ Eq ((fun F => Function.comp (⇑F) (FreeAlgebra.ι R)) (FreeAlgebra.liftAux R f …
      -/
      simp only [Function.comp_apply, ι_def]
      /-
        case h
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : X → A
        x✝ : X
        ⊢ Eq ((FreeAlgebra.liftAux R f) (Quot.mk (FreeAlgebra.Rel R X) (FreeAlgebra.Pr …
      -/
      rfl
      /-
        🎉 no goals
      -/
    right_inv := fun F ↦ by
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        F : AlgHom R (FreeAlgebra R X) A
        ⊢ Eq (FreeAlgebra.liftAux R ((fun F => Function.comp (⇑F) (FreeAlgebra.ι R)) F …
      -/
      ext t
      /-
        case H
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        F : AlgHom R (FreeAlgebra R X) A
        t : FreeAlgebra R X
        ⊢ Eq ((FreeAlgebra.liftAux R ((fun F => Function.comp (⇑F) (FreeAlgebra.ι R))  …
      -/
      rcases t with ⟨x⟩
      induction x with
      | of =>
        change ((F : FreeAlgebra R X → A) ∘ ι R) _ = _
        simp only [Function.comp_apply, ι_def]
      | ofScalar x =>
        change algebraMap _ _ x = F (algebraMap _ _ x)
        rw [AlgHom.commutes F _]
      | add a b ha hb =>
        -- Porting note: it is necessary to declare fa and fb explicitly otherwise Lean refuses
        -- to consider `Quot.mk (Rel R X) ·` as element of FreeAlgebra R X
        let fa : FreeAlgebra R X := Quot.mk (Rel R X) a
        let fb : FreeAlgebra R X := Quot.mk (Rel R X) b
        change liftAux R (F ∘ ι R) (fa + fb) = F (fa + fb)
        rw [map_add, map_add, ha, hb]
      | mul a b ha hb =>
        let fa : FreeAlgebra R X := Quot.mk (Rel R X) a
        let fb : FreeAlgebra R X := Quot.mk (Rel R X) b
        change liftAux R (F ∘ ι R) (fa * fb) = F (fa * fb)
        rw [map_mul, map_mul, ha, hb] }


@[simp]
theorem liftAux_eq (f : X → A) : liftAux R f = lift R f := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    ⊢ Eq (FreeAlgebra.liftAux R f) ((FreeAlgebra.lift R) f)
  -/
  rw [lift]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    ⊢ Eq (FreeAlgebra.liftAux R f) ({ toFun := FreeAlgebra.liftAux R, invFun := fu …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_symm_apply (F : FreeAlgebra R X →ₐ[R] A) : (lift R).symm F = F ∘ ι R := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    F : AlgHom R (FreeAlgebra R X) A
    ⊢ Eq ((FreeAlgebra.lift R).symm F) (Function.comp (⇑F) (FreeAlgebra.ι R))
  -/
  rw [lift]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    F : AlgHom R (FreeAlgebra R X) A
    ⊢ Eq ({ toFun := FreeAlgebra.liftAux R, invFun := fun F => Function.comp (⇑F)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_comp_lift (f : X → A) : (lift R f : FreeAlgebra R X → A) ∘ ι R = f := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    ⊢ Eq (Function.comp (⇑((FreeAlgebra.lift R) f)) (FreeAlgebra.ι R)) f
  -/
  ext
  /-
    case h
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    x✝ : X
    ⊢ Eq (Function.comp (⇑((FreeAlgebra.lift R) f)) (FreeAlgebra.ι R) x✝) (f x✝)
  -/
  rw [Function.comp_apply, ι_def, lift]
  /-
    case h
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    x✝ : X
    ⊢ Eq (({ toFun := FreeAlgebra.liftAux R, invFun := fun F => Function.comp (⇑F) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_ι_apply (f : X → A) (x) : lift R f (ι R x) = f x := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    x : X
    ⊢ Eq (((FreeAlgebra.lift R) f) (FreeAlgebra.ι R x)) (f x)
  -/
  rw [ι_def, lift]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    x : X
    ⊢ Eq (({ toFun := FreeAlgebra.liftAux R, invFun := fun F => Function.comp (⇑F) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_unique (f : X → A) (g : FreeAlgebra R X →ₐ[R] A) :
    (g : FreeAlgebra R X → A) ∘ ι R = f ↔ g = lift R f := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    g : AlgHom R (FreeAlgebra R X) A
    ⊢ Iff (Eq (Function.comp (⇑g) (FreeAlgebra.ι R)) f) (Eq g ((FreeAlgebra.lift R …
  -/
  rw [← (lift R).symm_apply_eq, lift]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : X → A
    g : AlgHom R (FreeAlgebra R X) A
    ⊢ Iff (Eq (Function.comp (⇑g) (FreeAlgebra.ι R)) f) (Eq ({ toFun := FreeAlgebr …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_comp_ι (g : FreeAlgebra R X →ₐ[R] A) :
    lift R ((g : FreeAlgebra R X → A) ∘ ι R) = g := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    g : AlgHom R (FreeAlgebra R X) A
    ⊢ Eq ((FreeAlgebra.lift R) (Function.comp (⇑g) (FreeAlgebra.ι R))) g
  -/
  rw [← lift_symm_apply]
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    g : AlgHom R (FreeAlgebra R X) A
    ⊢ Eq ((FreeAlgebra.lift R) ((FreeAlgebra.lift R).symm g)) g
  -/
  exact (lift R).apply_symm_apply g
  /-
    🎉 no goals
  -/


/-- See note [partially-applied ext lemmas]. -/
@[ext high]
theorem hom_ext {f g : FreeAlgebra R X →ₐ[R] A}
    (w : (f : FreeAlgebra R X → A) ∘ ι R = (g : FreeAlgebra R X → A) ∘ ι R) : f = g := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (FreeAlgebra R X) A
    w : Eq (Function.comp (⇑f) (FreeAlgebra.ι R)) (Function.comp (⇑g) (FreeAlgebra …
    ⊢ Eq f g
  -/
  rw [← lift_symm_apply, ← lift_symm_apply] at w
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    X : Type u_2
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R (FreeAlgebra R X) A
    w : Eq ((FreeAlgebra.lift R).symm f) ((FreeAlgebra.lift R).symm g)
    ⊢ Eq f g
  -/
  exact (lift R).symm.injective w
  /-
    🎉 no goals
  -/


/-- The free algebra on `X` is "just" the monoid algebra on the free monoid on `X`.

This would be useful when constructing linear maps out of a free algebra,
for example.
-/
noncomputable def equivMonoidAlgebraFreeMonoid :
    FreeAlgebra R X ≃ₐ[R] MonoidAlgebra R (FreeMonoid X) :=
  AlgEquiv.ofAlgHom (lift R fun x ↦ (MonoidAlgebra.of R (FreeMonoid X)) (FreeMonoid.of x))
    ((MonoidAlgebra.lift R (FreeMonoid X) (FreeAlgebra R X)) (FreeMonoid.lift (ι R)))
    (by
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        ⊢ Eq (((FreeAlgebra.lift R) fun x => (MonoidAlgebra.of R (FreeMonoid X)) (Free …
      -/
      apply MonoidAlgebra.algHom_ext; intro x
      /-
        case h
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        x : FreeMonoid X
        ⊢ Eq ((((FreeAlgebra.lift R) fun x => (MonoidAlgebra.of R (FreeMonoid X)) (Fre …
      -/
      refine FreeMonoid.recOn x ?_ ?_
        /-
          case h.refine_1
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          x : FreeMonoid X
          ⊢ Eq ((((FreeAlgebra.lift R) fun x => (MonoidAlgebra.of R (FreeMonoid X)) (Fre …
        -/
      · simp
        /-
          case h.refine_1
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          x : FreeMonoid X
          ⊢ Eq 1 (MonoidAlgebra.single 1 1)
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case h.refine_2
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          x : FreeMonoid X
          ⊢ ∀ (x : X) (xs : FreeMonoid X), Eq ((((FreeAlgebra.lift R) fun x => (MonoidAl …
        -/
      · intro x y ih
        /-
          case h.refine_2
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          x✝ : FreeMonoid X
          x : X
          y : FreeMonoid X
          ih : Eq ((((FreeAlgebra.lift R) fun x => (MonoidAlgebra.of R (FreeMonoid X)) ( …
          ⊢ Eq ((((FreeAlgebra.lift R) fun x => (MonoidAlgebra.of R (FreeMonoid X)) (Fre …
        -/
        simp at ih
        /-
          case h.refine_2
          R : Type u_1
          inst✝² : CommSemiring R
          X : Type u_2
          A : Type u_3
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          x✝ : FreeMonoid X
          x : X
          y : FreeMonoid X
          ih : Eq (((FreeAlgebra.lift R) fun x => MonoidAlgebra.single (FreeMonoid.of x) …
          ⊢ Eq ((((FreeAlgebra.lift R) fun x => (MonoidAlgebra.of R (FreeMonoid X)) (Fre …
        -/
        simp [ih])
        /-
          🎉 no goals
        -/
    (by
      /-
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        ⊢ Eq (((MonoidAlgebra.lift R (FreeMonoid X) (FreeAlgebra R X)) (FreeMonoid.lif …
      -/
      ext
      /-
        case w.h
        R : Type u_1
        inst✝² : CommSemiring R
        X : Type u_2
        A : Type u_3
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        x✝ : X
        ⊢ Eq (Function.comp (⇑(((MonoidAlgebra.lift R (FreeMonoid X) (FreeAlgebra R X) …
      -/
      simp)
      /-
        🎉 no goals
      -/


/-- `FreeAlgebra R X` is nontrivial when `R` is. -/
instance [Nontrivial R] : Nontrivial (FreeAlgebra R X) :=
  equivMonoidAlgebraFreeMonoid.surjective.nontrivial


/-- `FreeAlgebra R X` has no zero-divisors when `R` has no zero-divisors. -/
instance instNoZeroDivisors [NoZeroDivisors R] : NoZeroDivisors (FreeAlgebra R X) :=
  equivMonoidAlgebraFreeMonoid.toMulEquiv.noZeroDivisors


/-- `FreeAlgebra R X` is a domain when `R` is an integral domain. -/
instance instIsDomain {R X} [CommRing R] [IsDomain R] : IsDomain (FreeAlgebra R X) :=
  NoZeroDivisors.to_isDomain _


/-- The left-inverse of `algebraMap`. -/
def algebraMapInv : FreeAlgebra R X →ₐ[R] R :=
  lift R (0 : X → R)


theorem algebraMap_leftInverse :
    Function.LeftInverse algebraMapInv (algebraMap R <| FreeAlgebra R X) := fun x ↦ by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    X : Type u_2
    x : R
    ⊢ Eq (FreeAlgebra.algebraMapInv ((algebraMap R (FreeAlgebra R X)) x)) x
  -/
  simp [algebraMapInv]
  /-
    🎉 no goals
  -/


@[simp]
theorem algebraMap_inj (x y : R) :
    algebraMap R (FreeAlgebra R X) x = algebraMap R (FreeAlgebra R X) y ↔ x = y :=
  algebraMap_leftInverse.injective.eq_iff


@[simp]
theorem algebraMap_eq_zero_iff (x : R) : algebraMap R (FreeAlgebra R X) x = 0 ↔ x = 0 :=
  map_eq_zero_iff (algebraMap _ _) algebraMap_leftInverse.injective


@[simp]
theorem algebraMap_eq_one_iff (x : R) : algebraMap R (FreeAlgebra R X) x = 1 ↔ x = 1 :=
  map_eq_one_iff (algebraMap _ _) algebraMap_leftInverse.injective

-- this proof is copied from the approach in `FreeAbelianGroup.of_injective`

theorem ι_injective [Nontrivial R] : Function.Injective (ι R : X → FreeAlgebra R X) :=
  fun x y hoxy ↦
  by_contradiction <| by
    classical exact fun hxy : x ≠ y ↦
        let f : FreeAlgebra R X →ₐ[R] R := lift R fun z ↦ if x = z then (1 : R) else 0
        have hfx1 : f (ι R x) = 1 := (lift_ι_apply _ _).trans <| if_pos rfl
        have hfy1 : f (ι R y) = 1 := hoxy ▸ hfx1
        have hfy0 : f (ι R y) = 0 := (lift_ι_apply _ _).trans <| if_neg hxy
        one_ne_zero <| hfy1.symm.trans hfy0


@[simp]
theorem ι_inj [Nontrivial R] (x y : X) : ι R x = ι R y ↔ x = y :=
  ι_injective.eq_iff


@[simp]
theorem ι_ne_algebraMap [Nontrivial R] (x : X) (r : R) : ι R x ≠ algebraMap R _ r := fun h ↦ by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    X : Type u_2
    inst✝ : Nontrivial R
    x : X
    r : R
    h : Eq (FreeAlgebra.ι R x) ((algebraMap R (FreeAlgebra R X)) r)
    ⊢ False
  -/
  let f0 : FreeAlgebra R X →ₐ[R] R := lift R 0
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    X : Type u_2
    inst✝ : Nontrivial R
    x : X
    r : R
    h : Eq (FreeAlgebra.ι R x) ((algebraMap R (FreeAlgebra R X)) r)
    f0 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 0
    ⊢ False
  -/
  let f1 : FreeAlgebra R X →ₐ[R] R := lift R 1
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    X : Type u_2
    inst✝ : Nontrivial R
    x : X
    r : R
    h : Eq (FreeAlgebra.ι R x) ((algebraMap R (FreeAlgebra R X)) r)
    f0 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 0
    f1 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 1
    ⊢ False
  -/
  have hf0 : f0 (ι R x) = 0 := lift_ι_apply _ _
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    X : Type u_2
    inst✝ : Nontrivial R
    x : X
    r : R
    h : Eq (FreeAlgebra.ι R x) ((algebraMap R (FreeAlgebra R X)) r)
    f0 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 0
    f1 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 1
    hf0 : Eq (f0 (FreeAlgebra.ι R x)) 0
    ⊢ False
  -/
  have hf1 : f1 (ι R x) = 1 := lift_ι_apply _ _
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    X : Type u_2
    inst✝ : Nontrivial R
    x : X
    r : R
    h : Eq (FreeAlgebra.ι R x) ((algebraMap R (FreeAlgebra R X)) r)
    f0 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 0
    f1 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 1
    hf0 : Eq (f0 (FreeAlgebra.ι R x)) 0
    hf1 : Eq (f1 (FreeAlgebra.ι R x)) 1
    ⊢ False
  -/
  rw [h, f0.commutes, Algebra.id.map_eq_self] at hf0
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    X : Type u_2
    inst✝ : Nontrivial R
    x : X
    r : R
    h : Eq (FreeAlgebra.ι R x) ((algebraMap R (FreeAlgebra R X)) r)
    f0 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 0
    f1 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 1
    hf0 : Eq r 0
    hf1 : Eq (f1 (FreeAlgebra.ι R x)) 1
    ⊢ False
  -/
  rw [h, f1.commutes, Algebra.id.map_eq_self] at hf1
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    X : Type u_2
    inst✝ : Nontrivial R
    x : X
    r : R
    h : Eq (FreeAlgebra.ι R x) ((algebraMap R (FreeAlgebra R X)) r)
    f0 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 0
    f1 : AlgHom R (FreeAlgebra R X) R := (FreeAlgebra.lift R) 1
    hf0 : Eq r 0
    hf1 : Eq r 1
    ⊢ False
  -/
  exact zero_ne_one (hf0.symm.trans hf1)
  /-
    🎉 no goals
  -/


@[simp]
theorem ι_ne_zero [Nontrivial R] (x : X) : ι R x ≠ 0 :=
  ι_ne_algebraMap x 0


@[simp]
theorem ι_ne_one [Nontrivial R] (x : X) : ι R x ≠ 1 :=
  ι_ne_algebraMap x 1


/-- An induction principle for the free algebra.

If `C` holds for the `algebraMap` of `r : R` into `FreeAlgebra R X`, the `ι` of `x : X`, and is
preserved under addition and multiplication, then it holds for all of `FreeAlgebra R X`.
-/
@[elab_as_elim, induction_eliminator]
theorem induction {C : FreeAlgebra R X → Prop}
    (h_grade0 : ∀ r, C (algebraMap R (FreeAlgebra R X) r)) (h_grade1 : ∀ x, C (ι R x))
    (h_mul : ∀ a b, C a → C b → C (a * b)) (h_add : ∀ a b, C a → C b → C (a + b))
    (a : FreeAlgebra R X) : C a := by
  -- the arguments are enough to construct a subalgebra, and a mapping into it from X
  let s : Subalgebra R (FreeAlgebra R X) :=
    { carrier := C
      mul_mem' := h_mul _ _
      add_mem' := h_add _ _
      algebraMap_mem' := h_grade0 }
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    X : Type u_2
    C : FreeAlgebra R X → Prop
    h_grade0 : ∀ (r : R), C ((algebraMap R (FreeAlgebra R X)) r)
    h_grade1 : ∀ (x : X), C (FreeAlgebra.ι R x)
    h_mul : ∀ (a b : FreeAlgebra R X), C a → C b → C (HMul.hMul a b)
    h_add : ∀ (a b : FreeAlgebra R X), C a → C b → C (HAdd.hAdd a b)
    a : FreeAlgebra R X
    s : Subalgebra R (FreeAlgebra R X) := { carrier := C, mul_mem' := ⋯, one_mem'  …
    ⊢ C a
  -/
  let of : X → s := Subtype.coind (ι R) h_grade1
  -- the mapping through the subalgebra is the identity
  have of_id : AlgHom.id R (FreeAlgebra R X) = s.val.comp (lift R of) := by
    ext
    simp [of, Subtype.coind]
  -- finding a proof is finding an element of the subalgebra
  suffices a = lift R of a by
    rw [this]
    exact Subtype.prop (lift R of a)
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    X : Type u_2
    C : FreeAlgebra R X → Prop
    h_grade0 : ∀ (r : R), C ((algebraMap R (FreeAlgebra R X)) r)
    h_grade1 : ∀ (x : X), C (FreeAlgebra.ι R x)
    h_mul : ∀ (a b : FreeAlgebra R X), C a → C b → C (HMul.hMul a b)
    h_add : ∀ (a b : FreeAlgebra R X), C a → C b → C (HAdd.hAdd a b)
    a : FreeAlgebra R X
    s : Subalgebra R (FreeAlgebra R X) := { carrier := C, mul_mem' := ⋯, one_mem'  …
    of : X → Subtype fun x => Membership.mem s x := Subtype.coind (FreeAlgebra.ι R …
    of_id : Eq (AlgHom.id R (FreeAlgebra R X)) (s.val.comp ((FreeAlgebra.lift R) o …
    ⊢ Eq a ↑(((FreeAlgebra.lift R) of) a)
  -/
  simp [AlgHom.ext_iff] at of_id
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    X : Type u_2
    C : FreeAlgebra R X → Prop
    h_grade0 : ∀ (r : R), C ((algebraMap R (FreeAlgebra R X)) r)
    h_grade1 : ∀ (x : X), C (FreeAlgebra.ι R x)
    h_mul : ∀ (a b : FreeAlgebra R X), C a → C b → C (HMul.hMul a b)
    h_add : ∀ (a b : FreeAlgebra R X), C a → C b → C (HAdd.hAdd a b)
    a : FreeAlgebra R X
    s : Subalgebra R (FreeAlgebra R X) := { carrier := C, mul_mem' := ⋯, one_mem'  …
    of : X → Subtype fun x => Membership.mem s x := Subtype.coind (FreeAlgebra.ι R …
    of_id : ∀ (x : FreeAlgebra R X), Eq x ↑(((FreeAlgebra.lift R) of) x)
    ⊢ Eq a ↑(((FreeAlgebra.lift R) of) a)
  -/
  exact of_id a
  /-
    🎉 no goals
  -/


@[simp]
theorem adjoin_range_ι : Algebra.adjoin R (Set.range (ι R : X → FreeAlgebra R X)) = ⊤ := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    X : Type u_2
    ⊢ Eq (Algebra.adjoin R (Set.range (FreeAlgebra.ι R))) Top.top
  -/
  set S := Algebra.adjoin R (Set.range (ι R : X → FreeAlgebra R X))
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    X : Type u_2
    S : Subalgebra R (FreeAlgebra R X) := Algebra.adjoin R (Set.range (FreeAlgebra …
    ⊢ Eq S Top.top
  -/
  refine top_unique fun x hx => ?_; clear hx
  induction x with
  | h_grade0 => exact S.algebraMap_mem _
  | h_add x y hx hy => exact S.add_mem hx hy
  | h_mul x y hx hy => exact S.mul_mem hx hy
  | h_grade1 x => exact Algebra.subset_adjoin (Set.mem_range_self _)


/-- Noncommutative version of `Algebra.adjoin_range_eq_range_aeval`. -/
theorem _root_.Algebra.adjoin_range_eq_range_freeAlgebra_lift (f : X → A) :
    Algebra.adjoin R (Set.range f) = (FreeAlgebra.lift R f).range := by
  simp only [← Algebra.map_top, ← adjoin_range_ι, AlgHom.map_adjoin, ← Set.range_comp,
    Function.comp_def, lift_ι_apply]


/-- Noncommutative version of `Algebra.adjoin_range_eq_range`. -/
theorem _root_.Algebra.adjoin_eq_range_freeAlgebra_lift (s : Set A) :
    Algebra.adjoin R s = (FreeAlgebra.lift R ((↑) : s → A)).range := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    A : Type u_3
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s : Set A
    ⊢ Eq (Algebra.adjoin R s) ((FreeAlgebra.lift R) Subtype.val).range
  -/
  rw [← Algebra.adjoin_range_eq_range_freeAlgebra_lift, Subtype.range_coe]
  /-
    🎉 no goals
  -/


