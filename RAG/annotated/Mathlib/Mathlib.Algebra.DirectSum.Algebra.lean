/-- A graded version of `Algebra`. An instance of `DirectSum.GAlgebra R A` endows `(⨁ i, A i)`
with an `R`-algebra structure. -/
class GAlgebra where
  toFun : R →+ A 0
  map_one : toFun 1 = GradedMonoid.GOne.one
  map_mul :
    ∀ r s, GradedMonoid.mk _ (toFun (r * s)) = .mk _ (GradedMonoid.GMul.mul (toFun r) (toFun s))
  commutes : ∀ (r) (x : GradedMonoid A), .mk _ (toFun r) * x = x * .mk _ (toFun r)
  smul_def : ∀ (r) (x : GradedMonoid A), r • x = .mk _ (toFun r) * x


instance _root_.GradedMonoid.smulCommClass_right :
    SMulCommClass R (GradedMonoid A) (GradedMonoid A) where
  smul_comm s x y := by
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : ι) → AddCommMonoid (A i)
      inst✝⁵ : (i : ι) → Module R (A i)
      inst✝⁴ : AddMonoid ι
      inst✝³ : DirectSum.GSemiring A
      inst✝² : Semiring B
      inst✝¹ : DirectSum.GAlgebra R A
      inst✝ : Algebra R B
      s : R
      x y : GradedMonoid A
      ⊢ Eq (HSMul.hSMul s (HSMul.hSMul x y)) (HSMul.hSMul x (HSMul.hSMul s y))
    -/
    dsimp
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : ι) → AddCommMonoid (A i)
      inst✝⁵ : (i : ι) → Module R (A i)
      inst✝⁴ : AddMonoid ι
      inst✝³ : DirectSum.GSemiring A
      inst✝² : Semiring B
      inst✝¹ : DirectSum.GAlgebra R A
      inst✝ : Algebra R B
      s : R
      x y : GradedMonoid A
      ⊢ Eq (HSMul.hSMul s (HMul.hMul x y)) (HMul.hMul x (HSMul.hSMul s y))
    -/
    rw [GAlgebra.smul_def, GAlgebra.smul_def, ← mul_assoc, GAlgebra.commutes, mul_assoc]
    /-
      🎉 no goals
    -/


instance _root_.GradedMonoid.isScalarTower_right :
    IsScalarTower R (GradedMonoid A) (GradedMonoid A) where
  smul_assoc s x y := by
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : ι) → AddCommMonoid (A i)
      inst✝⁵ : (i : ι) → Module R (A i)
      inst✝⁴ : AddMonoid ι
      inst✝³ : DirectSum.GSemiring A
      inst✝² : Semiring B
      inst✝¹ : DirectSum.GAlgebra R A
      inst✝ : Algebra R B
      s : R
      x y : GradedMonoid A
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul s x) y) (HSMul.hSMul s (HSMul.hSMul x y))
    -/
    dsimp
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁷ : CommSemiring R
      inst✝⁶ : (i : ι) → AddCommMonoid (A i)
      inst✝⁵ : (i : ι) → Module R (A i)
      inst✝⁴ : AddMonoid ι
      inst✝³ : DirectSum.GSemiring A
      inst✝² : Semiring B
      inst✝¹ : DirectSum.GAlgebra R A
      inst✝ : Algebra R B
      s : R
      x y : GradedMonoid A
      ⊢ Eq (HMul.hMul (HSMul.hSMul s x) y) (HSMul.hSMul s (HMul.hMul x y))
    -/
    rw [GAlgebra.smul_def, GAlgebra.smul_def, ← mul_assoc]
    /-
      🎉 no goals
    -/


instance : Algebra R (⨁ i, A i) where
  toFun := (DirectSum.of A 0).comp GAlgebra.toFun
  map_zero' := AddMonoidHom.map_zero _
  map_add' := AddMonoidHom.map_add _
  map_one' := DFunLike.congr_arg (DirectSum.of A 0) GAlgebra.map_one
  map_mul' a b := by
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      a b : R
      ⊢ Eq ({ toFun := ⇑((DirectSum.of A 0).comp DirectSum.GAlgebra.toFun), map_one' …
    -/
    simp only [AddMonoidHom.comp_apply]
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      a b : R
      ⊢ Eq ((DirectSum.of A 0) (DirectSum.GAlgebra.toFun (HMul.hMul a b))) (HMul.hMu …
    -/
    rw [of_mul_of]
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      a b : R
      ⊢ Eq ((DirectSum.of A 0) (DirectSum.GAlgebra.toFun (HMul.hMul a b))) ((DirectS …
    -/
    apply DFinsupp.single_eq_of_sigma_eq (GAlgebra.map_mul a b)
    /-
      🎉 no goals
    -/
  commutes' r x := by
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      ⊢ Eq (HMul.hMul ({ toFun := ⇑((DirectSum.of A 0).comp DirectSum.GAlgebra.toFun …
    -/
    change AddMonoidHom.mul (DirectSum.of _ _ _) x = AddMonoidHom.mul.flip (DirectSum.of _ _ _) x
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      ⊢ Eq ((AddMonoidHom.mul ((DirectSum.of A 0) (DirectSum.GAlgebra.toFun r))) x)  …
    -/
    apply DFunLike.congr_fun _ x
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      ⊢ Eq (AddMonoidHom.mul ((DirectSum.of A 0) (DirectSum.GAlgebra.toFun r))) (Add …
    -/
    ext i xi : 2
    /-
      case H.h
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      i : ι
      xi : A i
      ⊢ Eq (((AddMonoidHom.mul ((DirectSum.of A 0) (DirectSum.GAlgebra.toFun r))).co …
    -/
    dsimp only [AddMonoidHom.comp_apply, AddMonoidHom.mul_apply, AddMonoidHom.flip_apply]
    /-
      case H.h
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      i : ι
      xi : A i
      ⊢ Eq (HMul.hMul ((DirectSum.of A 0) (DirectSum.GAlgebra.toFun r)) ((DirectSum. …
    -/
    rw [of_mul_of, of_mul_of]
    /-
      case H.h
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      i : ι
      xi : A i
      ⊢ Eq ((DirectSum.of A (HAdd.hAdd 0 i)) (GradedMonoid.GMul.mul (DirectSum.GAlge …
    -/
    apply DFinsupp.single_eq_of_sigma_eq (GAlgebra.commutes r ⟨i, xi⟩)
    /-
      🎉 no goals
    -/
  smul_def' r x := by
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      ⊢ Eq (HSMul.hSMul r x) (HMul.hMul ({ toFun := ⇑((DirectSum.of A 0).comp Direct …
    -/
    change DistribMulAction.toAddMonoidHom _ r x = AddMonoidHom.mul (DirectSum.of _ _ _) x
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      ⊢ Eq ((DistribMulAction.toAddMonoidHom (DirectSum ι fun i => A i) r) x) ((AddM …
    -/
    apply DFunLike.congr_fun _ x
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      ⊢ Eq (DistribMulAction.toAddMonoidHom (DirectSum ι fun i => A i) r) (AddMonoid …
    -/
    ext i xi : 2
    dsimp only [AddMonoidHom.comp_apply, DistribMulAction.toAddMonoidHom_apply,
      AddMonoidHom.mul_apply]
    /-
      case H.h
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      i : ι
      xi : A i
      ⊢ Eq (HSMul.hSMul r ((DirectSum.of A i) xi)) (HMul.hMul ((DirectSum.of A 0) (D …
    -/
    rw [DirectSum.of_mul_of, ← of_smul]
    /-
      case H.h
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      r : R
      x : DirectSum ι fun i => A i
      i : ι
      xi : A i
      ⊢ Eq ((DirectSum.of A i) (HSMul.hSMul r xi)) ((DirectSum.of A (HAdd.hAdd 0 i)) …
    -/
    apply DFinsupp.single_eq_of_sigma_eq (GAlgebra.smul_def r ⟨i, xi⟩)
    /-
      🎉 no goals
    -/


theorem algebraMap_apply (r : R) :
    algebraMap R (⨁ i, A i) r = DirectSum.of A 0 (GAlgebra.toFun r) :=
  rfl


theorem algebraMap_toAddMonoid_hom :
    ↑(algebraMap R (⨁ i, A i)) = (DirectSum.of A 0).comp (GAlgebra.toFun : R →+ A 0) :=
  rfl


/-- A family of `LinearMap`s preserving `DirectSum.GOne.one` and `DirectSum.GMul.mul`
describes an `AlgHom` on `⨁ i, A i`. This is a stronger version of `DirectSum.toSemiring`.

Of particular interest is the case when `A i` are bundled subojects, `f` is the family of
coercions such as `Submodule.subtype (A i)`, and the `[GMonoid A]` structure originates from
`DirectSum.GMonoid.ofAddSubmodules`, in which case the proofs about `GOne` and `GMul`
can be discharged by `rfl`. -/
@[simps]
def toAlgebra (f : ∀ i, A i →ₗ[R] B) (hone : f _ GradedMonoid.GOne.one = 1)
    (hmul : ∀ {i j} (ai : A i) (aj : A j), f _ (GradedMonoid.GMul.mul ai aj) = f _ ai * f _ aj) :
    (⨁ i, A i) →ₐ[R] B :=
  { toSemiring (fun i => (f i).toAddMonoidHom) hone @hmul with
    toFun := toSemiring (fun i => (f i).toAddMonoidHom) hone @hmul
    commutes' := fun r => by
      /-
        ι : Type uι
        R : Type uR
        A : ι → Type uA
        B : Type uB
        inst✝⁸ : CommSemiring R
        inst✝⁷ : (i : ι) → AddCommMonoid (A i)
        inst✝⁶ : (i : ι) → Module R (A i)
        inst✝⁵ : AddMonoid ι
        inst✝⁴ : DirectSum.GSemiring A
        inst✝³ : Semiring B
        inst✝² : DirectSum.GAlgebra R A
        inst✝¹ : Algebra R B
        inst✝ : DecidableEq ι
        f : (i : ι) → LinearMap (RingHom.id R) (A i) B
        hone : Eq ((f 0) GradedMonoid.GOne.one) 1
        hmul : ∀ {i j : ι} (ai : A i) (aj : A j), Eq ((f (HAdd.hAdd i j)) (GradedMonoi …
        r : R
        ⊢ Eq ((↑↑{ toFun := ⇑(DirectSum.toSemiring (fun i => (f i).toAddMonoidHom) hon …
      -/
      show toModule R _ _ f (algebraMap R _ r) = _
      rw [Algebra.algebraMap_eq_smul_one, Algebra.algebraMap_eq_smul_one, map_smul, one_def,
        ← lof_eq_of R, toModule_lof, hone] }


/-- Two `AlgHom`s out of a direct sum are equal if they agree on the generators.

See note [partially-applied ext lemmas]. -/
@[ext]
theorem algHom_ext' ⦃f g : (⨁ i, A i) →ₐ[R] B⦄
    (h : ∀ i, f.toLinearMap.comp (lof _ _ A i) = g.toLinearMap.comp (lof _ _ A i)) : f = g :=
  AlgHom.toLinearMap_injective <| DirectSum.linearMap_ext _ h


theorem algHom_ext ⦃f g : (⨁ i, A i) →ₐ[R] B⦄ (h : ∀ i x, f (of A i x) = g (of A i x)) : f = g :=
  algHom_ext' R A fun i => LinearMap.ext <| h i


/-- The piecewise multiplication from the `Mul` instance, as a bundled linear homomorphism.

This is the graded version of `LinearMap.mul`, and the linear version of `DirectSum.gMulHom` -/
@[simps]
def gMulLHom {i j} : A i →ₗ[R] A j →ₗ[R] A (i + j) where
  toFun a :=
    { toFun := fun b => GradedMonoid.GMul.mul a b
      map_smul' := fun r x => by
        /-
          ι : Type uι
          R : Type uR
          A : ι → Type uA
          B : Type uB
          inst✝⁸ : CommSemiring R
          inst✝⁷ : (i : ι) → AddCommMonoid (A i)
          inst✝⁶ : (i : ι) → Module R (A i)
          inst✝⁵ : AddMonoid ι
          inst✝⁴ : DirectSum.GSemiring A
          inst✝³ : Semiring B
          inst✝² : DirectSum.GAlgebra R A
          inst✝¹ : Algebra R B
          inst✝ : DecidableEq ι
          i j : ι
          a : A i
          r : R
          x : A j
          ⊢ Eq ({ toFun := fun b => GradedMonoid.GMul.mul a b, map_add' := ⋯ }.toFun (HS …
        -/
        injection (smul_comm r (GradedMonoid.mk _ a) (GradedMonoid.mk _ x)).symm
        /-
          🎉 no goals
        -/
      map_add' := GNonUnitalNonAssocSemiring.mul_add _ }
  map_smul' r x := LinearMap.ext fun y => by
    /-
      ι : Type uι
      R : Type uR
      A : ι → Type uA
      B : Type uB
      inst✝⁸ : CommSemiring R
      inst✝⁷ : (i : ι) → AddCommMonoid (A i)
      inst✝⁶ : (i : ι) → Module R (A i)
      inst✝⁵ : AddMonoid ι
      inst✝⁴ : DirectSum.GSemiring A
      inst✝³ : Semiring B
      inst✝² : DirectSum.GAlgebra R A
      inst✝¹ : Algebra R B
      inst✝ : DecidableEq ι
      i j : ι
      r : R
      x : A i
      y : A j
      ⊢ Eq (({ toFun := fun a => { toFun := fun b => GradedMonoid.GMul.mul a b, map_ …
    -/
    injection smul_assoc r (GradedMonoid.mk _ x) (GradedMonoid.mk _ y)
    /-
      🎉 no goals
    -/
  map_add' _ _ := LinearMap.ext fun _ => GNonUnitalNonAssocSemiring.add_mul _ _ _


/-- A direct sum of copies of an `Algebra` inherits the algebra structure.

-/
@[simps]
instance Algebra.directSumGAlgebra {R A : Type*} [AddMonoid ι] [CommSemiring R]
    [Semiring A] [Algebra R A] : DirectSum.GAlgebra R fun _ : ι => A where
  toFun := (algebraMap R A).toAddMonoidHom
  map_one := (algebraMap R A).map_one
  map_mul a b := Sigma.ext (zero_add _).symm (heq_of_eq <| (algebraMap R A).map_mul a b)
  commutes := fun _ ⟨_, _⟩ =>
    Sigma.ext ((zero_add _).trans (add_zero _).symm) (heq_of_eq <| Algebra.commutes _ _)
  smul_def := fun _ ⟨_, _⟩ => Sigma.ext (zero_add _).symm (heq_of_eq <| Algebra.smul_def _ _)

