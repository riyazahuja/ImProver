/-- A bialgebra over a commutative (semi)ring `R` is both an algebra and a coalgebra over `R`, such
that the counit and comultiplication are algebra morphisms. -/
class Bialgebra (R : Type u) (A : Type v) [CommSemiring R] [Semiring A] extends
    Algebra R A, Coalgebra R A where
  -- The counit is an algebra morphism
  /-- The counit on a bialgebra preserves 1. -/
  counit_one : counit 1 = 1
  /-- The counit on a bialgebra preserves multiplication. Note that this is written
  in a rather obscure way: it says that two bilinear maps `A →ₗ[R] A →ₗ[R]` are equal.
  The two corresponding equal linear maps `A ⊗[R] A →ₗ[R]`
  are the following: the first factors through `A` and is multiplication on `A` followed
  by `counit`. The second factors through `R ⊗[R] R`, and is `counit ⊗ counit` followed by
  multiplication on `R`.

  See `Bialgebra.mk'` for a constructor for bialgebras which uses
  the more familiar but mathematically equivalent `counit (a * b) = counit a * counit b`. -/
  mul_compr₂_counit : (LinearMap.mul R A).compr₂ counit = (LinearMap.mul R R).compl₁₂ counit counit
  -- The comultiplication is an algebra morphism
  /-- The comultiplication on a bialgebra preserves `1`. -/
  comul_one : comul 1 = 1
  /-- The comultiplication on a bialgebra preserves multiplication. This is written in
  a rather obscure way: it says that two bilinear maps `A →ₗ[R] A →ₗ[R] (A ⊗[R] A)`
  are equal. The corresponding equal linear maps `A ⊗[R] A →ₗ[R] A ⊗[R] A`
  are firstly multiplication followed by `comul`, and secondly `comul ⊗ comul` followed
  by multiplication on `A ⊗[R] A`.

  See `Bialgebra.mk'` for a constructor for bialgebras which uses the more familiar
  but mathematically equivalent `comul (a * b) = comul a * comul b`. -/
  mul_compr₂_comul :
    (LinearMap.mul R A).compr₂ comul = (LinearMap.mul R (A ⊗[R] A)).compl₁₂ comul comul


lemma counit_mul (a b : A) : counit (R := R) (a * b) = counit a * counit b :=
  DFunLike.congr_fun (DFunLike.congr_fun mul_compr₂_counit a) b


lemma comul_mul (a b : A) : comul (R := R) (a * b) = comul a * comul b :=
  DFunLike.congr_fun (DFunLike.congr_fun mul_compr₂_comul a) b


/-- If `R` is a field (or even a commutative semiring) and `A`
is an `R`-algebra with a coalgebra structure, then `Bialgebra.mk'`
consumes proofs that the counit and comultiplication preserve
the identity and multiplication, and produces a bialgebra
structure on `A`. -/
def mk' (R : Type u) (A : Type v) [CommSemiring R] [Semiring A]
    [Algebra R A] [C : Coalgebra R A] (counit_one : C.counit 1 = 1)
    (counit_mul : ∀ {a b}, C.counit (a * b) = C.counit a * C.counit b)
    (comul_one : C.comul 1 = 1)
    (comul_mul : ∀ {a b}, C.comul (a * b) = C.comul a * C.comul b) :
    Bialgebra R A where
  counit_one := counit_one
                          /-
                            R✝ : Type u
                            A✝ : Type v
                            inst✝⁵ : CommSemiring R✝
                            inst✝⁴ : Semiring A✝
                            inst✝³ : Bialgebra R✝ A✝
                            R : Type u
                            A : Type v
                            inst✝² : CommSemiring R
                            inst✝¹ : Semiring A
                            inst✝ : Algebra R A
                            C : Coalgebra R A
                            counit_one : Eq (CoalgebraStruct.counit 1) 1
                            counit_mul : ∀ {a b : A}, Eq (CoalgebraStruct.counit (HMul.hMul a b)) (HMul.hM …
                            comul_one : Eq (CoalgebraStruct.comul 1) 1
                            comul_mul : ∀ {a b : A}, Eq (CoalgebraStruct.comul (HMul.hMul a b)) (HMul.hMul …
                            ⊢ Eq ((LinearMap.mul R A).compr₂ CoalgebraStruct.counit) ((LinearMap.mul R R). …
                          -/
  mul_compr₂_counit := by ext; exact counit_mul
                               /-
                                 🎉 no goals
                               -/
  comul_one := comul_one
                         /-
                           R✝ : Type u
                           A✝ : Type v
                           inst✝⁵ : CommSemiring R✝
                           inst✝⁴ : Semiring A✝
                           inst✝³ : Bialgebra R✝ A✝
                           R : Type u
                           A : Type v
                           inst✝² : CommSemiring R
                           inst✝¹ : Semiring A
                           inst✝ : Algebra R A
                           C : Coalgebra R A
                           counit_one : Eq (CoalgebraStruct.counit 1) 1
                           counit_mul : ∀ {a b : A}, Eq (CoalgebraStruct.counit (HMul.hMul a b)) (HMul.hM …
                           comul_one : Eq (CoalgebraStruct.comul 1) 1
                           comul_mul : ∀ {a b : A}, Eq (CoalgebraStruct.comul (HMul.hMul a b)) (HMul.hMul …
                           ⊢ Eq ((LinearMap.mul R A).compr₂ CoalgebraStruct.comul) ((LinearMap.mul R (Ten …
                         -/
  mul_compr₂_comul := by ext; exact comul_mul
                              /-
                                🎉 no goals
                              -/


/-- `counitAlgHom R A` is the counit of the `R`-bialgebra `A`, as an `R`-algebra map. -/
@[simps!]
def counitAlgHom : A →ₐ[R] R :=
  .ofLinearMap counit counit_one counit_mul


/-- `comulAlgHom R A` is the comultiplication of the `R`-bialgebra `A`, as an `R`-algebra map. -/
@[simps!]
def comulAlgHom : A →ₐ[R] A ⊗[R] A :=
  .ofLinearMap comul comul_one comul_mul


@[simp] lemma counit_algebraMap (r : R) : counit (R := R) (algebraMap R A r) = r :=
  (counitAlgHom R A).commutes r


@[simp] lemma comul_algebraMap (r : R) :
    comul (R := R) (algebraMap R A r) = algebraMap R (A ⊗[R] A) r :=
  (comulAlgHom R A).commutes r


@[simp] lemma counit_natCast (n : ℕ) : counit (R := R) (n : A) = n :=
  map_natCast (counitAlgHom R A) _


@[simp] lemma comul_natCast (n : ℕ) : comul (R := R) (n : A) = n :=
  map_natCast (comulAlgHom R A) _


@[simp] lemma counit_pow (a : A) (n : ℕ) : counit (R := R) (a ^ n) = counit a ^ n :=
  map_pow (counitAlgHom R A) a n


@[simp] lemma comul_pow (a : A) (n : ℕ) : comul (R := R) (a ^ n) = comul a ^ n :=
  map_pow (comulAlgHom R A) a n


/-- Every commutative (semi)ring is a bialgebra over itself -/
noncomputable
instance toBialgebra : Bialgebra R R where
                          /-
                            R : Type u
                            inst✝ : CommSemiring R
                            ⊢ Eq ((LinearMap.mul R R).compr₂ CoalgebraStruct.counit) ((LinearMap.mul R R). …
                          -/
  mul_compr₂_counit := by ext; simp
                               /-
                                 🎉 no goals
                               -/
  counit_one := rfl
                         /-
                           R : Type u
                           inst✝ : CommSemiring R
                           ⊢ Eq ((LinearMap.mul R R).compr₂ CoalgebraStruct.comul) ((LinearMap.mul R (Ten …
                         -/
  mul_compr₂_comul := by ext; simp
                              /-
                                🎉 no goals
                              -/
  comul_one := rfl


