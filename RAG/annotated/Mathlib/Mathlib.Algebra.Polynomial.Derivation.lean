/-- `Polynomial.derivative` as a derivation. -/
@[simps]
def derivative' : Derivation R R[X] R[X] where
  toFun := derivative
  map_add' _ _ := derivative_add
  map_smul' := derivative_smul
  map_one_eq_zero' := derivative_one
                     /-
                       R : Type u_1
                       A : Type u_2
                       inst✝ : CommSemiring R
                       f g : Polynomial R
                       ⊢ Eq ({ toFun := ⇑Polynomial.derivative, map_add' := ⋯, map_smul' := ⋯ } (HMul …
                     -/
  leibniz' f g := by simp [mul_comm, add_comm, derivative_mul]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem derivation_C (D : Derivation R R[X] A) (a : R) : D (C a) = 0 :=
  D.map_algebraMap a


@[simp]
theorem C_smul_derivation_apply (D : Derivation R R[X] A) (a : R) (f : R[X]) :
    C a • D f = a • D f := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Module (Polynomial R) A
    D : Derivation R (Polynomial R) A
    a : R
    f : Polynomial R
    ⊢ Eq (HSMul.hSMul (Polynomial.C a) (D f)) (HSMul.hSMul a (D f))
  -/
  have : C a • D f = D (C a * f) := by simp
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid A
    inst✝¹ : Module R A
    inst✝ : Module (Polynomial R) A
    D : Derivation R (Polynomial R) A
    a : R
    f : Polynomial R
    this : Eq (HSMul.hSMul (Polynomial.C a) (D f)) (D (HMul.hMul (Polynomial.C a)  …
    ⊢ Eq (HSMul.hSMul (Polynomial.C a) (D f)) (HSMul.hSMul a (D f))
  -/
  rw [this, C_mul', D.map_smul]
  /-
    🎉 no goals
  -/


@[ext]
theorem derivation_ext {D₁ D₂ : Derivation R R[X] A} (h : D₁ X = D₂ X) : D₁ = D₂ :=
  Derivation.ext fun f => Derivation.eqOn_adjoin (Set.eqOn_singleton.2 h) <| by
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid A
      inst✝¹ : Module R A
      inst✝ : Module (Polynomial R) A
      D₁ D₂ : Derivation R (Polynomial R) A
      h : Eq (D₁ Polynomial.X) (D₂ Polynomial.X)
      f : Polynomial R
      ⊢ Membership.mem (↑(Algebra.adjoin R (Singleton.singleton Polynomial.X))) f
    -/
    simp only [adjoin_X, Algebra.coe_top, Set.mem_univ]
    /-
      🎉 no goals
    -/


/-- The derivation on `R[X]` that takes the value `a` on `X`. -/
def mkDerivation : A →ₗ[R] Derivation R R[X] A where
  toFun := fun a ↦ (LinearMap.toSpanSingleton R[X] A a).compDer derivative'
                           /-
                             R : Type u_1
                             A : Type u_2
                             inst✝⁴ : CommSemiring R
                             inst✝³ : AddCommMonoid A
                             inst✝² : Module R A
                             inst✝¹ : Module (Polynomial R) A
                             inst✝ : IsScalarTower R (Polynomial R) A
                             a b : A
                             ⊢ Eq ((fun a => (LinearMap.toSpanSingleton (Polynomial R) A a).compDer Polynom …
                           -/
  map_add' := fun a b ↦ by ext; simp
                                /-
                                  🎉 no goals
                                -/
                            /-
                              R : Type u_1
                              A : Type u_2
                              inst✝⁴ : CommSemiring R
                              inst✝³ : AddCommMonoid A
                              inst✝² : Module R A
                              inst✝¹ : Module (Polynomial R) A
                              inst✝ : IsScalarTower R (Polynomial R) A
                              t : R
                              a : A
                              ⊢ Eq ({ toFun := fun a => (LinearMap.toSpanSingleton (Polynomial R) A a).compD …
                            -/
  map_smul' := fun t a ↦ by ext; simp
                                 /-
                                   🎉 no goals
                                 -/


lemma mkDerivation_apply (a : A) (f : R[X]) :
    mkDerivation R a f = derivative f • a := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid A
    inst✝² : Module R A
    inst✝¹ : Module (Polynomial R) A
    inst✝ : IsScalarTower R (Polynomial R) A
    a : A
    f : Polynomial R
    ⊢ Eq (((Polynomial.mkDerivation R) a) f) (HSMul.hSMul (Polynomial.derivative f …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
                                                              /-
                                                                R : Type u_1
                                                                A : Type u_2
                                                                inst✝⁴ : CommSemiring R
                                                                inst✝³ : AddCommMonoid A
                                                                inst✝² : Module R A
                                                                inst✝¹ : Module (Polynomial R) A
                                                                inst✝ : IsScalarTower R (Polynomial R) A
                                                                a : A
                                                                ⊢ Eq (((Polynomial.mkDerivation R) a) Polynomial.X) a
                                                              -/
theorem mkDerivation_X (a : A) : mkDerivation R a X = a := by simp [mkDerivation_apply]
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma mkDerivation_one_eq_derivative' : mkDerivation R (1 : R[X]) = derivative' := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq ((Polynomial.mkDerivation R) 1) Polynomial.derivative'
  -/
  ext : 1
  /-
    case h
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (((Polynomial.mkDerivation R) 1) Polynomial.X) (Polynomial.derivative' Po …
  -/
  simp [derivative']
  /-
    🎉 no goals
  -/


lemma mkDerivation_one_eq_derivative (f : R[X]) : mkDerivation R (1 : R[X]) f = derivative f := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : Polynomial R
    ⊢ Eq (((Polynomial.mkDerivation R) 1) f) (Polynomial.derivative f)
  -/
  rw [mkDerivation_one_eq_derivative']
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    f : Polynomial R
    ⊢ Eq (Polynomial.derivative' f) (Polynomial.derivative f)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Polynomial.mkDerivation` as a linear equivalence. -/
def mkDerivationEquiv : A ≃ₗ[R] Derivation R R[X] A :=
  LinearEquiv.symm <|
    { invFun := mkDerivation R
      toFun := fun D => D X
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl
      left_inv := fun _ => derivation_ext <| mkDerivation_X _ _
      right_inv := fun _ => mkDerivation_X _ _ }


@[simp] lemma mkDerivationEquiv_apply (a : A) :
    mkDerivationEquiv R a = mkDerivation R a := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid A
    inst✝² : Module R A
    inst✝¹ : Module (Polynomial R) A
    inst✝ : IsScalarTower R (Polynomial R) A
    a : A
    ⊢ Eq ((Polynomial.mkDerivationEquiv R) a) ((Polynomial.mkDerivation R) a)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma mkDerivationEquiv_symm_apply (D : Derivation R R[X] A) :
    (mkDerivationEquiv R).symm D = D X := rfl


/--
For a derivation `d : A → M` and an element `a : A`, `d.compAEval a` is the
derivation of `R[X]` which takes a polynomial `f` to `d(aeval a f)`.

This derivation takes values in `Module.AEval R M a`, which is `M`, regarded as an
`R[X]`-module, with the action of a polynomial `f` defined by `f • m = (aeval a f) • m`.
-/
/-
Note: `compAEval` is not defined using `Derivation.compAlgebraMap`.
This because `A` is not an `R[X]` algebra and it would be messy to create an algebra instance
within the definition.
-/
@[simps]
def compAEval : Derivation R R[X] <| AEval R M a where
  toFun f          := AEval.of R M a (d (aeval a f))
                         /-
                           R : Type u_1
                           A : Type u_2
                           M : Type u_3
                           inst✝⁶ : CommSemiring R
                           inst✝⁵ : CommSemiring A
                           inst✝⁴ : Algebra R A
                           inst✝³ : AddCommMonoid M
                           inst✝² : Module A M
                           inst✝¹ : Module R M
                           inst✝ : IsScalarTower R A M
                           d : Derivation R A M
                           a : A
                           ⊢ ∀ (x y : Polynomial R), Eq ((fun f => (Module.AEval.of R M a) (d ((Polynomia …
                         -/
  map_add'         := by simp
                         /-
                           🎉 no goals
                         -/
                         /-
                           R : Type u_1
                           A : Type u_2
                           M : Type u_3
                           inst✝⁶ : CommSemiring R
                           inst✝⁵ : CommSemiring A
                           inst✝⁴ : Algebra R A
                           inst✝³ : AddCommMonoid M
                           inst✝² : Module A M
                           inst✝¹ : Module R M
                           inst✝ : IsScalarTower R A M
                           d : Derivation R A M
                           a : A
                           ⊢ ∀ (m : R) (x : Polynomial R), Eq ({ toFun := fun f => (Module.AEval.of R M a …
                         -/
  map_smul'        := by simp
                         /-
                           🎉 no goals
                         -/
                         /-
                           R : Type u_1
                           A : Type u_2
                           M : Type u_3
                           inst✝⁶ : CommSemiring R
                           inst✝⁵ : CommSemiring A
                           inst✝⁴ : Algebra R A
                           inst✝³ : AddCommMonoid M
                           inst✝² : Module A M
                           inst✝¹ : Module R M
                           inst✝ : IsScalarTower R A M
                           d : Derivation R A M
                           a : A
                           ⊢ ∀ (a_1 b : Polynomial R), Eq ({ toFun := fun f => (Module.AEval.of R M a) (d …
                         -/
                         /-
                           R : Type u_1
                           A : Type u_2
                           M : Type u_3
                           inst✝⁶ : CommSemiring R
                           inst✝⁵ : CommSemiring A
                           inst✝⁴ : Algebra R A
                           inst✝³ : AddCommMonoid M
                           inst✝² : Module A M
                           inst✝¹ : Module R M
                           inst✝ : IsScalarTower R A M
                           d : Derivation R A M
                           a : A
                           ⊢ Eq ({ toFun := fun f => (Module.AEval.of R M a) (d ((Polynomial.aeval a) f)) …
                         -/
  leibniz'         := by simp [AEval.of_aeval_smul, -Derivation.map_aeval]
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
  map_one_eq_zero' := by simp


/--
  A form of the chain rule: if `f` is a polynomial over `R`
  and `d : A → M` is an `R`-derivation then for all `a : A` we have
  $$ d(f(a)) = f' (a) d a. $$
  The equation is in the `R[X]`-module `Module.AEval R M a`.
  For the same equation in `M`, see `Derivation.compAEval_eq`.
-/
theorem compAEval_eq (d : Derivation R A M) (f : R[X]) :
    d.compAEval a f = derivative f • (AEval.of R M a (d a)) := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    a : A
    d : Derivation R A M
    f : Polynomial R
    ⊢ Eq ((d.compAEval a) f) (HSMul.hSMul (Polynomial.derivative f) ((Module.AEval …
  -/
  rw [← mkDerivation_apply]
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    a : A
    d : Derivation R A M
    f : Polynomial R
    ⊢ Eq ((d.compAEval a) f) (((Polynomial.mkDerivation R) ((Module.AEval.of R M a …
  -/
  congr
  /-
    case e_a
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    a : A
    d : Derivation R A M
    f : Polynomial R
    ⊢ Eq (d.compAEval a) ((Polynomial.mkDerivation R) ((Module.AEval.of R M a) (d  …
  -/
  apply derivation_ext
  /-
    case e_a.h
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    a : A
    d : Derivation R A M
    f : Polynomial R
    ⊢ Eq ((d.compAEval a) Polynomial.X) (((Polynomial.mkDerivation R) ((Module.AEv …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
  A form of the chain rule: if `f` is a polynomial over `R`
  and `d : A → M` is an `R`-derivation then for all `a : A` we have
  $$ d(f(a)) = f' (a) d a. $$
  The equation is in `M`. For the same equation in `Module.AEval R M a`,
  see `Derivation.compAEval_eq`.
-/
theorem comp_aeval_eq (d : Derivation R A M) (f : R[X]) :
    d (aeval a f) = aeval a (derivative f) • d a :=
  calc
    _ = (AEval.of R M a).symm (d.compAEval a f) := rfl
                /-
                  R : Type u_1
                  A : Type u_2
                  M : Type u_3
                  inst✝⁶ : CommSemiring R
                  inst✝⁵ : CommSemiring A
                  inst✝⁴ : Algebra R A
                  inst✝³ : AddCommMonoid M
                  inst✝² : Module A M
                  inst✝¹ : Module R M
                  inst✝ : IsScalarTower R A M
                  a : A
                  d : Derivation R A M
                  f : Polynomial R
                  ⊢ Eq ((Module.AEval.of R M a).symm ((d.compAEval a) f)) (HSMul.hSMul ((Polynom …
                -/
    _ = _ := by simp [-compAEval_apply, compAEval_eq]
                /-
                  🎉 no goals
                -/


