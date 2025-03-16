/-- `baseChange A f` for `f : M →ₗ[R] N` is the `A`-linear map `A ⊗[R] M →ₗ[A] A ⊗[R] N`.

This "base change" operation is also known as "extension of scalars". -/
def baseChange (f : M →ₗ[R] N) : A ⊗[R] M →ₗ[A] A ⊗[R] N :=
  AlgebraTensorModule.map (LinearMap.id : A →ₗ[A] A) f


@[simp]
theorem baseChange_tmul (a : A) (x : M) : f.baseChange A (a ⊗ₜ x) = a ⊗ₜ f x :=
  rfl


theorem baseChange_eq_ltensor : (f.baseChange A : A ⊗ M → A ⊗ N) = f.lTensor A :=
  rfl


@[simp]
theorem baseChange_add : (f + g).baseChange A = f.baseChange A + g.baseChange A := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f g : LinearMap (RingHom.id R) M N
    ⊢ Eq (LinearMap.baseChange A (HAdd.hAdd f g)) (HAdd.hAdd (LinearMap.baseChange …
  -/
  ext
  -- Porting note: added `-baseChange_tmul`
  /-
    case a.h.h
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    f g : LinearMap (RingHom.id R) M N
    x✝ : M
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (LinearMap.baseChange A (HAdd. …
  -/
  simp [baseChange_eq_ltensor, -baseChange_tmul]
  /-
    🎉 no goals
  -/


@[simp]
theorem baseChange_zero : baseChange A (0 : M →ₗ[R] N) = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    ⊢ Eq (LinearMap.baseChange A 0) 0
  -/
  ext
  /-
    case a.h.h
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    x✝ : M
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (LinearMap.baseChange A 0)) 1) …
  -/
  simp [baseChange_eq_ltensor]
  /-
    🎉 no goals
  -/


@[simp]
theorem baseChange_smul : (r • f).baseChange A = r • f.baseChange A := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    r : R
    f : LinearMap (RingHom.id R) M N
    ⊢ Eq (LinearMap.baseChange A (HSMul.hSMul r f)) (HSMul.hSMul r (LinearMap.base …
  -/
  ext
  /-
    case a.h.h
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    r : R
    f : LinearMap (RingHom.id R) M N
    x✝ : M
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (LinearMap.baseChange A (HSMul …
  -/
  simp [baseChange_tmul]
  /-
    🎉 no goals
  -/


@[simp]
lemma baseChange_id : (.id : M →ₗ[R] M).baseChange A = .id := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (LinearMap.baseChange A LinearMap.id) LinearMap.id
  -/
  ext; simp
       /-
         🎉 no goals
       -/


lemma baseChange_comp (g : N →ₗ[R] P) :
    (g ∘ₗ f).baseChange A = g.baseChange A ∘ₗ f.baseChange A := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    P : Type u_6
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : AddCommMonoid P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.baseChange A (g.comp f)) ((LinearMap.baseChange A g).comp (Lin …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


variable (R M) in
@[simp]
lemma baseChange_one : (1 : Module.End R M).baseChange A = 1 := baseChange_id


lemma baseChange_mul (f g : Module.End R M) :
    (f * g).baseChange A = f.baseChange A * g.baseChange A := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : Module.End R M
    ⊢ Eq (LinearMap.baseChange A (HMul.hMul f g)) (HMul.hMul (LinearMap.baseChange …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- `baseChange A e` for `e : M ≃ₗ[R] N` is the `A`-linear map `A ⊗[R] M ≃ₗ[A] A ⊗[R] N`. -/
def _root_.LinearEquiv.baseChange (e : M ≃ₗ[R] N) : A ⊗[R] M ≃ₗ[A] A ⊗[R] N :=
  AlgebraTensorModule.congr (.refl _ _) e


/-- `baseChange` as a linear map.

When `M = N`, this is true more strongly as `Module.End.baseChangeHom`. -/
@[simps]
def baseChangeHom : (M →ₗ[R] N) →ₗ[R] A ⊗[R] M →ₗ[A] A ⊗[R] N where
  toFun := baseChange A
  map_add' := baseChange_add
  map_smul' := baseChange_smul


/-- `baseChange` as an `AlgHom`. -/
@[simps!]
def _root_.Module.End.baseChangeHom : Module.End R M →ₐ[R] Module.End A (A ⊗[R] M) :=
  .ofLinearMap (LinearMap.baseChangeHom _ _ _ _) (baseChange_one _ _) baseChange_mul


lemma baseChange_pow (f : Module.End R M) (n : ℕ) :
    (f ^ n).baseChange A = f.baseChange A ^ n :=
  map_pow (Module.End.baseChangeHom _ _ _) f n


@[simp]
theorem baseChange_sub : (f - g).baseChange A = f.baseChange A - g.baseChange A := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommRing R
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f g : LinearMap (RingHom.id R) M N
    ⊢ Eq (LinearMap.baseChange A (HSub.hSub f g)) (HSub.hSub (LinearMap.baseChange …
  -/
  ext
  -- Porting note: `tmul_sub` wasn't needed in mathlib3
  /-
    case a.h.h
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommRing R
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f g : LinearMap (RingHom.id R) M N
    x✝ : M
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (LinearMap.baseChange A (HSub. …
  -/
  simp [baseChange_eq_ltensor, tmul_sub]
  /-
    🎉 no goals
  -/


@[simp]
theorem baseChange_neg : (-f).baseChange A = -f.baseChange A := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommRing R
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    ⊢ Eq (LinearMap.baseChange A (Neg.neg f)) (Neg.neg (LinearMap.baseChange A f))
  -/
  ext
  -- Porting note: `tmul_neg` wasn't needed in mathlib3
  /-
    case a.h.h
    R : Type u_1
    A : Type u_2
    M : Type u_4
    N : Type u_5
    inst✝⁶ : CommRing R
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    x✝ : M
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (LinearMap.baseChange A (Neg.n …
  -/
  simp [baseChange_eq_ltensor, tmul_neg]
  /-
    🎉 no goals
  -/


/--
If `M` is an `R`-module and `N` is an `A`-module, then `A`-linear maps `A ⊗[R] M →ₗ[A] N`
correspond to `R` linear maps `M →ₗ[R] N` by composing with `M → A ⊗ M`, `x ↦ 1 ⊗ x`.
-/
noncomputable
def liftBaseChangeEquiv : (M →ₗ[R] N) ≃ₗ[A] (A ⊗[R] M →ₗ[A] N) :=
  (LinearMap.ringLmapEquivSelf _ _ _).symm.trans (AlgebraTensorModule.lift.equiv _ _ _ _ _ _)


/-- If `N` is an `A` module, we may lift a linear map `M →ₗ[R] N` to `A ⊗[R] M →ₗ[A] N` -/
noncomputable
abbrev liftBaseChange (l : M →ₗ[R] N) : A ⊗[R] M →ₗ[A] N :=
  LinearMap.liftBaseChangeEquiv A l


@[simp]
lemma liftBaseChange_tmul (l : M →ₗ[R] N) (x y) : l.liftBaseChange A (x ⊗ₜ y) = x • l y := rfl


                                                                                            /-
                                                                                              R : Type u_1
                                                                                              M : Type u_2
                                                                                              N : Type u_3
                                                                                              A : Type u_4
                                                                                              inst✝⁸ : CommSemiring R
                                                                                              inst✝⁷ : CommSemiring A
                                                                                              inst✝⁶ : Algebra R A
                                                                                              inst✝⁵ : AddCommMonoid M
                                                                                              inst✝⁴ : AddCommMonoid N
                                                                                              inst✝³ : Module R M
                                                                                              inst✝² : Module R N
                                                                                              inst✝¹ : Module A N
                                                                                              inst✝ : IsScalarTower R A N
                                                                                              l : LinearMap (RingHom.id R) M N
                                                                                              y : M
                                                                                              ⊢ Eq ((LinearMap.liftBaseChange A l) (TensorProduct.tmul R 1 y)) (l y)
                                                                                            -/
lemma liftBaseChange_one_tmul (l : M →ₗ[R] N) (y) : l.liftBaseChange A (1 ⊗ₜ y) = l y := by simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
lemma liftBaseChangeEquiv_symm_apply (l : A ⊗[R] M →ₗ[A] N) (x) :
    (liftBaseChangeEquiv A).symm l x = l (1 ⊗ₜ x) := rfl


lemma liftBaseChange_comp {P} [AddCommMonoid P] [Module A P] [Module R P] [IsScalarTower R A P]
    (l : M →ₗ[R] N) (l' : N →ₗ[A] P) :
      l' ∘ₗ l.liftBaseChange A = (l'.restrictScalars R ∘ₗ l).liftBaseChange A := by
  /-
    R : Type u_3
    M : Type u_4
    N : Type u_5
    A : Type u_2
    inst✝¹² : CommSemiring R
    inst✝¹¹ : CommSemiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    inst✝⁵ : Module A N
    inst✝⁴ : IsScalarTower R A N
    P : Type u_1
    inst✝³ : AddCommMonoid P
    inst✝² : Module A P
    inst✝¹ : Module R P
    inst✝ : IsScalarTower R A P
    l : LinearMap (RingHom.id R) M N
    l' : LinearMap (RingHom.id A) N P
    ⊢ Eq (l'.comp (LinearMap.liftBaseChange A l)) (LinearMap.liftBaseChange A ((↑R …
  -/
  ext
  /-
    case a.h.h
    R : Type u_3
    M : Type u_4
    N : Type u_5
    A : Type u_2
    inst✝¹² : CommSemiring R
    inst✝¹¹ : CommSemiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : AddCommMonoid N
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    inst✝⁵ : Module A N
    inst✝⁴ : IsScalarTower R A N
    P : Type u_1
    inst✝³ : AddCommMonoid P
    inst✝² : Module A P
    inst✝¹ : Module R P
    inst✝ : IsScalarTower R A P
    l : LinearMap (RingHom.id R) M N
    l' : LinearMap (RingHom.id A) N P
    x✝ : M
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (l'.comp (LinearMap.liftBaseCh …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma range_liftBaseChange (l : M →ₗ[R] N) :
    LinearMap.range (l.liftBaseChange A) = Submodule.span A (LinearMap.range l) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    A : Type u_4
    inst✝⁸ : CommSemiring R
    inst✝⁷ : CommSemiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module A N
    inst✝ : IsScalarTower R A N
    l : LinearMap (RingHom.id R) M N
    ⊢ Eq (LinearMap.range (LinearMap.liftBaseChange A l)) (Submodule.span A ↑(Line …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      M : Type u_2
      N : Type u_3
      A : Type u_4
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module A N
      inst✝ : IsScalarTower R A N
      l : LinearMap (RingHom.id R) M N
      ⊢ LE.le (LinearMap.range (LinearMap.liftBaseChange A l)) (Submodule.span A ↑(L …
    -/
  · rintro _ ⟨x, rfl⟩
    /-
      case a.intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      A : Type u_4
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module A N
      inst✝ : IsScalarTower R A N
      l : LinearMap (RingHom.id R) M N
      x : TensorProduct R A M
      ⊢ Membership.mem (Submodule.span A ↑(LinearMap.range l)) ((LinearMap.liftBaseC …
    -/
    induction x using TensorProduct.induction_on
      /-
        case a.intro.zero
        R : Type u_1
        M : Type u_2
        N : Type u_3
        A : Type u_4
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : Module R M
        inst✝² : Module R N
        inst✝¹ : Module A N
        inst✝ : IsScalarTower R A N
        l : LinearMap (RingHom.id R) M N
        ⊢ Membership.mem (Submodule.span A ↑(LinearMap.range l)) ((LinearMap.liftBaseC …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case a.intro.tmul
        R : Type u_1
        M : Type u_2
        N : Type u_3
        A : Type u_4
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : Module R M
        inst✝² : Module R N
        inst✝¹ : Module A N
        inst✝ : IsScalarTower R A N
        l : LinearMap (RingHom.id R) M N
        x✝ : A
        y✝ : M
        ⊢ Membership.mem (Submodule.span A ↑(LinearMap.range l)) ((LinearMap.liftBaseC …
      -/
    · rw [LinearMap.liftBaseChange_tmul]
      /-
        case a.intro.tmul
        R : Type u_1
        M : Type u_2
        N : Type u_3
        A : Type u_4
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : Module R M
        inst✝² : Module R N
        inst✝¹ : Module A N
        inst✝ : IsScalarTower R A N
        l : LinearMap (RingHom.id R) M N
        x✝ : A
        y✝ : M
        ⊢ Membership.mem (Submodule.span A ↑(LinearMap.range l)) (HSMul.hSMul x✝ (l y✝))
      -/
      exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨_, rfl⟩)
      /-
        🎉 no goals
      -/
      /-
        case a.intro.add
        R : Type u_1
        M : Type u_2
        N : Type u_3
        A : Type u_4
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : Module R M
        inst✝² : Module R N
        inst✝¹ : Module A N
        inst✝ : IsScalarTower R A N
        l : LinearMap (RingHom.id R) M N
        x✝ y✝ : TensorProduct R A M
        a✝¹ : Membership.mem (Submodule.span A ↑(LinearMap.range l)) ((LinearMap.liftB …
        a✝ : Membership.mem (Submodule.span A ↑(LinearMap.range l)) ((LinearMap.liftBa …
        ⊢ Membership.mem (Submodule.span A ↑(LinearMap.range l)) ((LinearMap.liftBaseC …
      -/
    · rw [map_add]
      /-
        case a.intro.add
        R : Type u_1
        M : Type u_2
        N : Type u_3
        A : Type u_4
        inst✝⁸ : CommSemiring R
        inst✝⁷ : CommSemiring A
        inst✝⁶ : Algebra R A
        inst✝⁵ : AddCommMonoid M
        inst✝⁴ : AddCommMonoid N
        inst✝³ : Module R M
        inst✝² : Module R N
        inst✝¹ : Module A N
        inst✝ : IsScalarTower R A N
        l : LinearMap (RingHom.id R) M N
        x✝ y✝ : TensorProduct R A M
        a✝¹ : Membership.mem (Submodule.span A ↑(LinearMap.range l)) ((LinearMap.liftB …
        a✝ : Membership.mem (Submodule.span A ↑(LinearMap.range l)) ((LinearMap.liftBa …
        ⊢ Membership.mem (Submodule.span A ↑(LinearMap.range l)) (HAdd.hAdd ((LinearMa …
      -/
      exact add_mem ‹_› ‹_›
      /-
        🎉 no goals
      -/
    /-
      case a
      R : Type u_1
      M : Type u_2
      N : Type u_3
      A : Type u_4
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module A N
      inst✝ : IsScalarTower R A N
      l : LinearMap (RingHom.id R) M N
      ⊢ LE.le (Submodule.span A ↑(LinearMap.range l)) (LinearMap.range (LinearMap.li …
    -/
  · rw [Submodule.span_le]
    /-
      case a
      R : Type u_1
      M : Type u_2
      N : Type u_3
      A : Type u_4
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module A N
      inst✝ : IsScalarTower R A N
      l : LinearMap (RingHom.id R) M N
      ⊢ HasSubset.Subset ↑(LinearMap.range l) ↑(LinearMap.range (LinearMap.liftBaseC …
    -/
    rintro _ ⟨x, rfl⟩
    /-
      case a.intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      A : Type u_4
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid N
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module A N
      inst✝ : IsScalarTower R A N
      l : LinearMap (RingHom.id R) M N
      x : M
      ⊢ Membership.mem (↑(LinearMap.range (LinearMap.liftBaseChange A l))) (l x)
    -/
    exact ⟨1 ⊗ₜ x, by simp⟩
    /-
      🎉 no goals
    -/


instance : One (A ⊗[R] B) where one := 1 ⊗ₜ 1


theorem one_def : (1 : A ⊗[R] B) = (1 : A) ⊗ₜ (1 : B) :=
  rfl


instance instAddCommMonoidWithOne : AddCommMonoidWithOne (A ⊗[R] B) where
  natCast n := n ⊗ₜ 1
                     /-
                       R : Type uR
                       S : Type uS
                       A : Type uA
                       B : Type uB
                       C : Type uC
                       D : Type uD
                       E : Type uE
                       F : Type uF
                       inst✝⁴ : CommSemiring R
                       inst✝³ : AddCommMonoidWithOne A
                       inst✝² : Module R A
                       inst✝¹ : AddCommMonoidWithOne B
                       inst✝ : Module R B
                       ⊢ Eq (NatCast.natCast 0) 0
                     -/
  natCast_zero := by simp
                     /-
                       🎉 no goals
                     -/
                       /-
                         R : Type uR
                         S : Type uS
                         A : Type uA
                         B : Type uB
                         C : Type uC
                         D : Type uD
                         E : Type uE
                         F : Type uF
                         inst✝⁴ : CommSemiring R
                         inst✝³ : AddCommMonoidWithOne A
                         inst✝² : Module R A
                         inst✝¹ : AddCommMonoidWithOne B
                         inst✝ : Module R B
                         n : Nat
                         ⊢ Eq (NatCast.natCast (HAdd.hAdd n 1)) (HAdd.hAdd (NatCast.natCast n) 1)
                       -/
  natCast_succ n := by simp [add_tmul, one_def]
                       /-
                         🎉 no goals
                       -/
  add_comm := add_comm


theorem natCast_def (n : ℕ) : (n : A ⊗[R] B) = (n : A) ⊗ₜ (1 : B) := rfl


theorem natCast_def' (n : ℕ) : (n : A ⊗[R] B) = (1 : A) ⊗ₜ (n : B) := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoidWithOne A
    inst✝² : Module R A
    inst✝¹ : AddCommMonoidWithOne B
    inst✝ : Module R B
    n : Nat
    ⊢ Eq (↑n) (TensorProduct.tmul R 1 ↑n)
  -/
  rw [natCast_def, ← nsmul_one, smul_tmul, nsmul_one]
  /-
    🎉 no goals
  -/


/-- (Implementation detail)
The multiplication map on `A ⊗[R] B`,
as an `R`-bilinear map.
-/
@[irreducible]
def mul : A ⊗[R] B →ₗ[R] A ⊗[R] B →ₗ[R] A ⊗[R] B :=
  TensorProduct.map₂ (LinearMap.mul R A) (LinearMap.mul R B)


unseal mul in
@[simp]
theorem mul_apply (a₁ a₂ : A) (b₁ b₂ : B) :
    mul (a₁ ⊗ₜ[R] b₁) (a₂ ⊗ₜ[R] b₂) = (a₁ * a₂) ⊗ₜ[R] (b₁ * b₂) :=
  rfl

-- providing this instance separately makes some downstream code substantially faster

instance instMul : Mul (A ⊗[R] B) where
  mul a b := mul a b


unseal mul in
@[simp]
theorem tmul_mul_tmul (a₁ a₂ : A) (b₁ b₂ : B) :
    a₁ ⊗ₜ[R] b₁ * a₂ ⊗ₜ[R] b₂ = (a₁ * a₂) ⊗ₜ[R] (b₁ * b₂) :=
  rfl


unseal mul in
theorem _root_.SemiconjBy.tmul {a₁ a₂ a₃ : A} {b₁ b₂ b₃ : B}
    (ha : SemiconjBy a₁ a₂ a₃) (hb : SemiconjBy b₁ b₂ b₃) :
    SemiconjBy (a₁ ⊗ₜ[R] b₁) (a₂ ⊗ₜ[R] b₂) (a₃ ⊗ₜ[R] b₃) :=
  congr_arg₂ (· ⊗ₜ[R] ·) ha.eq hb.eq


nonrec theorem _root_.Commute.tmul {a₁ a₂ : A} {b₁ b₂ : B}
    (ha : Commute a₁ a₂) (hb : Commute b₁ b₂) :
    Commute (a₁ ⊗ₜ[R] b₁) (a₂ ⊗ₜ[R] b₂) :=
  ha.tmul hb


instance instNonUnitalNonAssocSemiring : NonUnitalNonAssocSemiring (A ⊗[R] B) where
                           /-
                             R : Type uR
                             S : Type uS
                             A : Type uA
                             B : Type uB
                             C : Type uC
                             D : Type uD
                             E : Type uE
                             F : Type uF
                             inst✝⁸ : CommSemiring R
                             inst✝⁷ : NonUnitalNonAssocSemiring A
                             inst✝⁶ : Module R A
                             inst✝⁵ : SMulCommClass R A A
                             inst✝⁴ : IsScalarTower R A A
                             inst✝³ : NonUnitalNonAssocSemiring B
                             inst✝² : Module R B
                             inst✝¹ : SMulCommClass R B B
                             inst✝ : IsScalarTower R B B
                             a b c : TensorProduct R A B
                             ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
                           -/
  left_distrib a b c := by simp [HMul.hMul, Mul.mul]
                           /-
                             🎉 no goals
                           -/
                            /-
                              R : Type uR
                              S : Type uS
                              A : Type uA
                              B : Type uB
                              C : Type uC
                              D : Type uD
                              E : Type uE
                              F : Type uF
                              inst✝⁸ : CommSemiring R
                              inst✝⁷ : NonUnitalNonAssocSemiring A
                              inst✝⁶ : Module R A
                              inst✝⁵ : SMulCommClass R A A
                              inst✝⁴ : IsScalarTower R A A
                              inst✝³ : NonUnitalNonAssocSemiring B
                              inst✝² : Module R B
                              inst✝¹ : SMulCommClass R B B
                              inst✝ : IsScalarTower R B B
                              a b c : TensorProduct R A B
                              ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
                            -/
  right_distrib a b c := by simp [HMul.hMul, Mul.mul]
                            /-
                              🎉 no goals
                            -/
                   /-
                     R : Type uR
                     S : Type uS
                     A : Type uA
                     B : Type uB
                     C : Type uC
                     D : Type uD
                     E : Type uE
                     F : Type uF
                     inst✝⁸ : CommSemiring R
                     inst✝⁷ : NonUnitalNonAssocSemiring A
                     inst✝⁶ : Module R A
                     inst✝⁵ : SMulCommClass R A A
                     inst✝⁴ : IsScalarTower R A A
                     inst✝³ : NonUnitalNonAssocSemiring B
                     inst✝² : Module R B
                     inst✝¹ : SMulCommClass R B B
                     inst✝ : IsScalarTower R B B
                     a : TensorProduct R A B
                     ⊢ Eq (HMul.hMul 0 a) 0
                   -/
  zero_mul a := by simp [HMul.hMul, Mul.mul]
                   /-
                     🎉 no goals
                   -/
                   /-
                     R : Type uR
                     S : Type uS
                     A : Type uA
                     B : Type uB
                     C : Type uC
                     D : Type uD
                     E : Type uE
                     F : Type uF
                     inst✝⁸ : CommSemiring R
                     inst✝⁷ : NonUnitalNonAssocSemiring A
                     inst✝⁶ : Module R A
                     inst✝⁵ : SMulCommClass R A A
                     inst✝⁴ : IsScalarTower R A A
                     inst✝³ : NonUnitalNonAssocSemiring B
                     inst✝² : Module R B
                     inst✝¹ : SMulCommClass R B B
                     inst✝ : IsScalarTower R B B
                     a : TensorProduct R A B
                     ⊢ Eq (HMul.hMul a 0) 0
                   -/
  mul_zero a := by simp [HMul.hMul, Mul.mul]
                   /-
                     🎉 no goals
                   -/

-- we want `isScalarTower_right` to take priority since it's better for unification elsewhere

instance (priority := 100) isScalarTower_right [Monoid S] [DistribMulAction S A]
    [IsScalarTower S A A] [SMulCommClass R S A] : IsScalarTower S (A ⊗[R] B) (A ⊗[R] B) where
  smul_assoc r x y := by
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹² : CommSemiring R
      inst✝¹¹ : NonUnitalNonAssocSemiring A
      inst✝¹⁰ : Module R A
      inst✝⁹ : SMulCommClass R A A
      inst✝⁸ : IsScalarTower R A A
      inst✝⁷ : NonUnitalNonAssocSemiring B
      inst✝⁶ : Module R B
      inst✝⁵ : SMulCommClass R B B
      inst✝⁴ : IsScalarTower R B B
      inst✝³ : Monoid S
      inst✝² : DistribMulAction S A
      inst✝¹ : IsScalarTower S A A
      inst✝ : SMulCommClass R S A
      r : S
      x y : TensorProduct R A B
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r x) y) (HSMul.hSMul r (HSMul.hSMul x y))
    -/
    change r • x * y = r • (x * y)
    induction y with
    | zero => simp [smul_zero]
    | tmul a b => induction x with
      | zero => simp [smul_zero]
      | tmul a' b' =>
        dsimp
        rw [TensorProduct.smul_tmul', TensorProduct.smul_tmul', tmul_mul_tmul, smul_mul_assoc]
      | add x y hx hy => simp [smul_add, add_mul _, *]
    | add x y hx hy => simp [smul_add, mul_add _, *]

-- we want `Algebra.to_smulCommClass` to take priority since it's better for unification elsewhere

instance (priority := 100) sMulCommClass_right [Monoid S] [DistribMulAction S A]
    [SMulCommClass S A A] [SMulCommClass R S A] : SMulCommClass S (A ⊗[R] B) (A ⊗[R] B) where
  smul_comm r x y := by
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹² : CommSemiring R
      inst✝¹¹ : NonUnitalNonAssocSemiring A
      inst✝¹⁰ : Module R A
      inst✝⁹ : SMulCommClass R A A
      inst✝⁸ : IsScalarTower R A A
      inst✝⁷ : NonUnitalNonAssocSemiring B
      inst✝⁶ : Module R B
      inst✝⁵ : SMulCommClass R B B
      inst✝⁴ : IsScalarTower R B B
      inst✝³ : Monoid S
      inst✝² : DistribMulAction S A
      inst✝¹ : SMulCommClass S A A
      inst✝ : SMulCommClass R S A
      r : S
      x y : TensorProduct R A B
      ⊢ Eq (HSMul.hSMul r (HSMul.hSMul x y)) (HSMul.hSMul x (HSMul.hSMul r y))
    -/
    change r • (x * y) = x * r • y
    induction y with
    | zero => simp [smul_zero]
    | tmul a b => induction x with
      | zero => simp [smul_zero]
      | tmul a' b' =>
        dsimp
        rw [TensorProduct.smul_tmul', TensorProduct.smul_tmul', tmul_mul_tmul, mul_smul_comm]
      | add x y hx hy => simp [smul_add, add_mul _, *]
    | add x y hx hy => simp [smul_add, mul_add _, *]


protected theorem one_mul (x : A ⊗[R] B) : mul (1 ⊗ₜ 1) x = x := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁸ : CommSemiring R
    inst✝⁷ : NonAssocSemiring A
    inst✝⁶ : Module R A
    inst✝⁵ : SMulCommClass R A A
    inst✝⁴ : IsScalarTower R A A
    inst✝³ : NonAssocSemiring B
    inst✝² : Module R B
    inst✝¹ : SMulCommClass R B B
    inst✝ : IsScalarTower R B B
    x : TensorProduct R A B
    ⊢ Eq ((Algebra.TensorProduct.mul (TensorProduct.tmul R 1 1)) x) x
  -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  refine TensorProduct.induction_on x ?_ ?_ ?_ <;> simp +contextual
                                                   /-
                                                     🎉 no goals
                                                   -/


protected theorem mul_one (x : A ⊗[R] B) : mul x (1 ⊗ₜ 1) = x := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁸ : CommSemiring R
    inst✝⁷ : NonAssocSemiring A
    inst✝⁶ : Module R A
    inst✝⁵ : SMulCommClass R A A
    inst✝⁴ : IsScalarTower R A A
    inst✝³ : NonAssocSemiring B
    inst✝² : Module R B
    inst✝¹ : SMulCommClass R B B
    inst✝ : IsScalarTower R B B
    x : TensorProduct R A B
    ⊢ Eq ((Algebra.TensorProduct.mul x) (TensorProduct.tmul R 1 1)) x
  -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  refine TensorProduct.induction_on x ?_ ?_ ?_ <;> simp +contextual
                                                   /-
                                                     🎉 no goals
                                                   -/


instance instNonAssocSemiring : NonAssocSemiring (A ⊗[R] B) where
  one_mul := Algebra.TensorProduct.one_mul
  mul_one := Algebra.TensorProduct.mul_one
  toNonUnitalNonAssocSemiring := instNonUnitalNonAssocSemiring
  __ := instAddCommMonoidWithOne


unseal mul in
protected theorem mul_assoc (x y z : A ⊗[R] B) : mul (mul x y) z = mul x (mul y z) := by
  -- restate as an equality of morphisms so that we can use `ext`
  suffices LinearMap.llcomp R _ _ _ mul ∘ₗ mul =
      (LinearMap.llcomp R _ _ _ LinearMap.lflip <| LinearMap.llcomp R _ _ _ mul.flip ∘ₗ mul).flip by
    exact DFunLike.congr_fun (DFunLike.congr_fun (DFunLike.congr_fun this x) y) z
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁸ : CommSemiring R
    inst✝⁷ : NonUnitalSemiring A
    inst✝⁶ : Module R A
    inst✝⁵ : SMulCommClass R A A
    inst✝⁴ : IsScalarTower R A A
    inst✝³ : NonUnitalSemiring B
    inst✝² : Module R B
    inst✝¹ : SMulCommClass R B B
    inst✝ : IsScalarTower R B B
    x y z : TensorProduct R A B
    ⊢ Eq (((LinearMap.llcomp R (TensorProduct R A B) (TensorProduct R A B) (Linear …
  -/
  ext xa xb ya yb za zb
  /-
    case a.h.h.a.h.h.a.h.h
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁸ : CommSemiring R
    inst✝⁷ : NonUnitalSemiring A
    inst✝⁶ : Module R A
    inst✝⁵ : SMulCommClass R A A
    inst✝⁴ : IsScalarTower R A A
    inst✝³ : NonUnitalSemiring B
    inst✝² : Module R B
    inst✝¹ : SMulCommClass R B B
    inst✝ : IsScalarTower R B B
    x y z : TensorProduct R A B
    xa : A
    xb : B
    ya : A
    yb : B
    za : A
    zb : B
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
  -/
  exact congr_arg₂ (· ⊗ₜ ·) (mul_assoc xa ya za) (mul_assoc xb yb zb)
  /-
    🎉 no goals
  -/


instance instNonUnitalSemiring : NonUnitalSemiring (A ⊗[R] B) where
  mul_assoc := Algebra.TensorProduct.mul_assoc


instance instSemiring : Semiring (A ⊗[R] B) where
                           /-
                             R : Type uR
                             S : Type uS
                             A : Type uA
                             B : Type uB
                             C : Type uC
                             D : Type uD
                             E : Type uE
                             F : Type uF
                             inst✝⁶ : CommSemiring R
                             inst✝⁵ : Semiring A
                             inst✝⁴ : Algebra R A
                             inst✝³ : Semiring B
                             inst✝² : Algebra R B
                             inst✝¹ : Semiring C
                             inst✝ : Algebra R C
                             a b c : TensorProduct R A B
                             ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
                           -/
  left_distrib a b c := by simp [HMul.hMul, Mul.mul]
                           /-
                             🎉 no goals
                           -/
                            /-
                              R : Type uR
                              S : Type uS
                              A : Type uA
                              B : Type uB
                              C : Type uC
                              D : Type uD
                              E : Type uE
                              F : Type uF
                              inst✝⁶ : CommSemiring R
                              inst✝⁵ : Semiring A
                              inst✝⁴ : Algebra R A
                              inst✝³ : Semiring B
                              inst✝² : Algebra R B
                              inst✝¹ : Semiring C
                              inst✝ : Algebra R C
                              a b c : TensorProduct R A B
                              ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
                            -/
  right_distrib a b c := by simp [HMul.hMul, Mul.mul]
                            /-
                              🎉 no goals
                            -/
                   /-
                     R : Type uR
                     S : Type uS
                     A : Type uA
                     B : Type uB
                     C : Type uC
                     D : Type uD
                     E : Type uE
                     F : Type uF
                     inst✝⁶ : CommSemiring R
                     inst✝⁵ : Semiring A
                     inst✝⁴ : Algebra R A
                     inst✝³ : Semiring B
                     inst✝² : Algebra R B
                     inst✝¹ : Semiring C
                     inst✝ : Algebra R C
                     a : TensorProduct R A B
                     ⊢ Eq (HMul.hMul 0 a) 0
                   -/
  zero_mul a := by simp [HMul.hMul, Mul.mul]
                   /-
                     🎉 no goals
                   -/
                   /-
                     R : Type uR
                     S : Type uS
                     A : Type uA
                     B : Type uB
                     C : Type uC
                     D : Type uD
                     E : Type uE
                     F : Type uF
                     inst✝⁶ : CommSemiring R
                     inst✝⁵ : Semiring A
                     inst✝⁴ : Algebra R A
                     inst✝³ : Semiring B
                     inst✝² : Algebra R B
                     inst✝¹ : Semiring C
                     inst✝ : Algebra R C
                     a : TensorProduct R A B
                     ⊢ Eq (HMul.hMul a 0) 0
                   -/
  mul_zero a := by simp [HMul.hMul, Mul.mul]
                   /-
                     🎉 no goals
                   -/
  mul_assoc := Algebra.TensorProduct.mul_assoc
  one_mul := Algebra.TensorProduct.one_mul
  mul_one := Algebra.TensorProduct.mul_one
  natCast_zero := AddMonoidWithOne.natCast_zero
  natCast_succ := AddMonoidWithOne.natCast_succ


@[simp]
theorem tmul_pow (a : A) (b : B) (k : ℕ) : a ⊗ₜ[R] b ^ k = (a ^ k) ⊗ₜ[R] (b ^ k) := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    a : A
    b : B
    k : Nat
    ⊢ Eq (HPow.hPow (TensorProduct.tmul R a b) k) (TensorProduct.tmul R (HPow.hPow …
  -/
  induction' k with k ih
    /-
      case zero
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : Semiring B
      inst✝ : Algebra R B
      a : A
      b : B
      ⊢ Eq (HPow.hPow (TensorProduct.tmul R a b) 0) (TensorProduct.tmul R (HPow.hPow …
    -/
  · simp [one_def]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type uR
      A : Type uA
      B : Type uB
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : Semiring B
      inst✝ : Algebra R B
      a : A
      b : B
      k : Nat
      ih : Eq (HPow.hPow (TensorProduct.tmul R a b) k) (TensorProduct.tmul R (HPow.h …
      ⊢ Eq (HPow.hPow (TensorProduct.tmul R a b) (HAdd.hAdd k 1)) (TensorProduct.tmu …
    -/
  · simp [pow_succ, ih]
    /-
      🎉 no goals
    -/


/-- The ring morphism `A →+* A ⊗[R] B` sending `a` to `a ⊗ₜ 1`. -/
@[simps]
def includeLeftRingHom : A →+* A ⊗[R] B where
  toFun a := a ⊗ₜ 1
                  /-
                    R : Type uR
                    S : Type uS
                    A : Type uA
                    B : Type uB
                    C : Type uC
                    D : Type uD
                    E : Type uE
                    F : Type uF
                    inst✝⁶ : CommSemiring R
                    inst✝⁵ : Semiring A
                    inst✝⁴ : Algebra R A
                    inst✝³ : Semiring B
                    inst✝² : Algebra R B
                    inst✝¹ : Semiring C
                    inst✝ : Algebra R C
                    ⊢ Eq ((↑{ toFun := fun a => TensorProduct.tmul R a 1, map_one' := ⋯, map_mul'  …
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
                 /-
                   R : Type uR
                   S : Type uS
                   A : Type uA
                   B : Type uB
                   C : Type uC
                   D : Type uD
                   E : Type uE
                   F : Type uF
                   inst✝⁶ : CommSemiring R
                   inst✝⁵ : Semiring A
                   inst✝⁴ : Algebra R A
                   inst✝³ : Semiring B
                   inst✝² : Algebra R B
                   inst✝¹ : Semiring C
                   inst✝ : Algebra R C
                   ⊢ ∀ (x y : A), Eq ({ toFun := fun a => TensorProduct.tmul R a 1, map_one' := ⋯ …
                 -/
                 /-
                   R : Type uR
                   S : Type uS
                   A : Type uA
                   B : Type uB
                   C : Type uC
                   D : Type uD
                   E : Type uE
                   F : Type uF
                   inst✝⁶ : CommSemiring R
                   inst✝⁵ : Semiring A
                   inst✝⁴ : Algebra R A
                   inst✝³ : Semiring B
                   inst✝² : Algebra R B
                   inst✝¹ : Semiring C
                   inst✝ : Algebra R C
                   ⊢ ∀ (x y : A), Eq ((↑{ toFun := fun a => TensorProduct.tmul R a 1, map_one' := …
                 -/
                 /-
                   🎉 no goals
                 -/
  map_add' := by simp [add_tmul]
                 /-
                   🎉 no goals
                 -/
  map_one' := rfl
  map_mul' := by simp


instance leftAlgebra [SMulCommClass R S A] : Algebra S (A ⊗[R] B) :=
  { commutes' := fun r x => by
      /-
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁹ : CommSemiring R
        inst✝⁸ : Semiring A
        inst✝⁷ : Algebra R A
        inst✝⁶ : Semiring B
        inst✝⁵ : Algebra R B
        inst✝⁴ : Semiring C
        inst✝³ : Algebra R C
        inst✝² : CommSemiring S
        inst✝¹ : Algebra S A
        inst✝ : SMulCommClass R S A
        r : S
        x : TensorProduct R A B
        ⊢ Eq (HMul.hMul ((Algebra.TensorProduct.includeLeftRingHom.comp (algebraMap S  …
      -/
      dsimp only [RingHom.toFun_eq_coe, RingHom.comp_apply, includeLeftRingHom_apply]
      rw [algebraMap_eq_smul_one, ← smul_tmul', ← one_def, mul_smul_comm, smul_mul_assoc, mul_one,
        one_mul]
    smul_def' := fun r x => by
      /-
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁹ : CommSemiring R
        inst✝⁸ : Semiring A
        inst✝⁷ : Algebra R A
        inst✝⁶ : Semiring B
        inst✝⁵ : Algebra R B
        inst✝⁴ : Semiring C
        inst✝³ : Algebra R C
        inst✝² : CommSemiring S
        inst✝¹ : Algebra S A
        inst✝ : SMulCommClass R S A
        r : S
        x : TensorProduct R A B
        ⊢ Eq (HSMul.hSMul r x) (HMul.hMul ((Algebra.TensorProduct.includeLeftRingHom.c …
      -/
      dsimp only [RingHom.toFun_eq_coe, RingHom.comp_apply, includeLeftRingHom_apply]
      /-
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁹ : CommSemiring R
        inst✝⁸ : Semiring A
        inst✝⁷ : Algebra R A
        inst✝⁶ : Semiring B
        inst✝⁵ : Algebra R B
        inst✝⁴ : Semiring C
        inst✝³ : Algebra R C
        inst✝² : CommSemiring S
        inst✝¹ : Algebra S A
        inst✝ : SMulCommClass R S A
        r : S
        x : TensorProduct R A B
        ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (TensorProduct.tmul R ((algebraMap S A) r) 1 …
      -/
      rw [algebraMap_eq_smul_one, ← smul_tmul', smul_mul_assoc, ← one_def, one_mul]
      /-
        🎉 no goals
      -/
    toRingHom := TensorProduct.includeLeftRingHom.comp (algebraMap S A) }


/-- The tensor product of two `R`-algebras is an `R`-algebra. -/
instance instAlgebra : Algebra R (A ⊗[R] B) :=
  inferInstance


@[simp]
theorem algebraMap_apply [SMulCommClass R S A] (r : S) :
    algebraMap S (A ⊗[R] B) r = (algebraMap S A) r ⊗ₜ 1 :=
  rfl


theorem algebraMap_apply' (r : R) :
    algebraMap R (A ⊗[R] B) r = 1 ⊗ₜ algebraMap R B r := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    r : R
    ⊢ Eq ((algebraMap R (TensorProduct R A B)) r) (TensorProduct.tmul R 1 ((algebr …
  -/
  rw [algebraMap_apply, Algebra.algebraMap_eq_smul_one, Algebra.algebraMap_eq_smul_one, smul_tmul]
  /-
    🎉 no goals
  -/


/-- The `R`-algebra morphism `A →ₐ[R] A ⊗[R] B` sending `a` to `a ⊗ₜ 1`. -/
def includeLeft [SMulCommClass R S A] : A →ₐ[S] A ⊗[R] B :=
                                            /-
                                              R : Type uR
                                              S : Type uS
                                              A : Type uA
                                              B : Type uB
                                              C : Type uC
                                              D : Type uD
                                              E : Type uE
                                              F : Type uF
                                              inst✝⁹ : CommSemiring R
                                              inst✝⁸ : Semiring A
                                              inst✝⁷ : Algebra R A
                                              inst✝⁶ : Semiring B
                                              inst✝⁵ : Algebra R B
                                              inst✝⁴ : Semiring C
                                              inst✝³ : Algebra R C
                                              inst✝² : CommSemiring S
                                              inst✝¹ : Algebra S A
                                              inst✝ : SMulCommClass R S A
                                              ⊢ ∀ (r : S), Eq ((↑↑__src✝).toFun ((algebraMap S A) r)) ((algebraMap S (Tensor …
                                            -/
  { includeLeftRingHom with commutes' := by simp }
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem includeLeft_apply [SMulCommClass R S A] (a : A) :
    (includeLeft : A →ₐ[S] A ⊗[R] B) a = a ⊗ₜ 1 :=
  rfl


/-- The algebra morphism `B →ₐ[R] A ⊗[R] B` sending `b` to `1 ⊗ₜ b`. -/
def includeRight : B →ₐ[R] A ⊗[R] B where
  toFun b := 1 ⊗ₜ b
                  /-
                    R : Type uR
                    S : Type uS
                    A : Type uA
                    B : Type uB
                    C : Type uC
                    D : Type uD
                    E : Type uE
                    F : Type uF
                    inst✝⁸ : CommSemiring R
                    inst✝⁷ : Semiring A
                    inst✝⁶ : Algebra R A
                    inst✝⁵ : Semiring B
                    inst✝⁴ : Algebra R B
                    inst✝³ : Semiring C
                    inst✝² : Algebra R C
                    inst✝¹ : CommSemiring S
                    inst✝ : Algebra S A
                    ⊢ Eq ((↑{ toFun := fun b => TensorProduct.tmul R 1 b, map_one' := ⋯, map_mul'  …
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
                 /-
                   R : Type uR
                   S : Type uS
                   A : Type uA
                   B : Type uB
                   C : Type uC
                   D : Type uD
                   E : Type uE
                   F : Type uF
                   inst✝⁸ : CommSemiring R
                   inst✝⁷ : Semiring A
                   inst✝⁶ : Algebra R A
                   inst✝⁵ : Semiring B
                   inst✝⁴ : Algebra R B
                   inst✝³ : Semiring C
                   inst✝² : Algebra R C
                   inst✝¹ : CommSemiring S
                   inst✝ : Algebra S A
                   ⊢ ∀ (x y : B), Eq ({ toFun := fun b => TensorProduct.tmul R 1 b, map_one' := ⋯ …
                 -/
                 /-
                   R : Type uR
                   S : Type uS
                   A : Type uA
                   B : Type uB
                   C : Type uC
                   D : Type uD
                   E : Type uE
                   F : Type uF
                   inst✝⁸ : CommSemiring R
                   inst✝⁷ : Semiring A
                   inst✝⁶ : Algebra R A
                   inst✝⁵ : Semiring B
                   inst✝⁴ : Algebra R B
                   inst✝³ : Semiring C
                   inst✝² : Algebra R C
                   inst✝¹ : CommSemiring S
                   inst✝ : Algebra S A
                   ⊢ ∀ (x y : B), Eq ((↑{ toFun := fun b => TensorProduct.tmul R 1 b, map_one' := …
                 -/
                 /-
                   🎉 no goals
                 -/
  map_add' := by simp [tmul_add]
                 /-
                   🎉 no goals
                 -/
  map_one' := rfl
  map_mul' := by simp
                    /-
                      R : Type uR
                      S : Type uS
                      A : Type uA
                      B : Type uB
                      C : Type uC
                      D : Type uD
                      E : Type uE
                      F : Type uF
                      inst✝⁸ : CommSemiring R
                      inst✝⁷ : Semiring A
                      inst✝⁶ : Algebra R A
                      inst✝⁵ : Semiring B
                      inst✝⁴ : Algebra R B
                      inst✝³ : Semiring C
                      inst✝² : Algebra R C
                      inst✝¹ : CommSemiring S
                      inst✝ : Algebra S A
                      r : R
                      ⊢ Eq ((↑↑{ toFun := fun b => TensorProduct.tmul R 1 b, map_one' := ⋯, map_mul' …
                    -/
  commutes' r := by simp only [algebraMap_apply']
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem includeRight_apply (b : B) : (includeRight : B →ₐ[R] A ⊗[R] B) b = 1 ⊗ₜ b :=
  rfl


theorem includeLeftRingHom_comp_algebraMap :
    (includeLeftRingHom.comp (algebraMap R A) : R →+* A ⊗[R] B) =
      includeRight.toRingHom.comp (algebraMap R B) := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    ⊢ Eq (Algebra.TensorProduct.includeLeftRingHom.comp (algebraMap R A)) (Algebra …
  -/
  ext
  /-
    case a
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    x✝ : R
    ⊢ Eq ((Algebra.TensorProduct.includeLeftRingHom.comp (algebraMap R A)) x✝) ((A …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A version of `TensorProduct.ext` for `AlgHom`.

Using this as the `@[ext]` lemma instead of `Algebra.TensorProduct.ext'` allows `ext` to apply
lemmas specific to `A →ₐ[S] _` and `B →ₐ[R] _`; notably this allows recursion into nested tensor
products of algebras.

See note [partially-applied ext lemmas]. -/
@[ext high]
theorem ext ⦃f g : (A ⊗[R] B) →ₐ[S] C⦄
    (ha : f.comp includeLeft = g.comp includeLeft)
    (hb : (f.restrictScalars R).comp includeRight = (g.restrictScalars R).comp includeRight) :
    f = g := by
  /-
    R : Type uR
    S : Type uS
    A : Type uA
    B : Type uB
    C : Type uC
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : Semiring B
    inst✝⁸ : Algebra R B
    inst✝⁷ : Semiring C
    inst✝⁶ : Algebra R C
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra S A
    inst✝³ : Algebra R S
    inst✝² : Algebra S C
    inst✝¹ : IsScalarTower R S A
    inst✝ : IsScalarTower R S C
    f g : AlgHom S (TensorProduct R A B) C
    ha : Eq (f.comp Algebra.TensorProduct.includeLeft) (g.comp Algebra.TensorProdu …
    hb : Eq ((AlgHom.restrictScalars R f).comp Algebra.TensorProduct.includeRight) …
    ⊢ Eq f g
  -/
  apply AlgHom.toLinearMap_injective
  /-
    case a
    R : Type uR
    S : Type uS
    A : Type uA
    B : Type uB
    C : Type uC
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : Semiring B
    inst✝⁸ : Algebra R B
    inst✝⁷ : Semiring C
    inst✝⁶ : Algebra R C
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra S A
    inst✝³ : Algebra R S
    inst✝² : Algebra S C
    inst✝¹ : IsScalarTower R S A
    inst✝ : IsScalarTower R S C
    f g : AlgHom S (TensorProduct R A B) C
    ha : Eq (f.comp Algebra.TensorProduct.includeLeft) (g.comp Algebra.TensorProdu …
    hb : Eq ((AlgHom.restrictScalars R f).comp Algebra.TensorProduct.includeRight) …
    ⊢ Eq f.toLinearMap g.toLinearMap
  -/
  ext a b
  /-
    case a.a.h.h
    R : Type uR
    S : Type uS
    A : Type uA
    B : Type uB
    C : Type uC
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : Semiring B
    inst✝⁸ : Algebra R B
    inst✝⁷ : Semiring C
    inst✝⁶ : Algebra R C
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra S A
    inst✝³ : Algebra R S
    inst✝² : Algebra S C
    inst✝¹ : IsScalarTower R S A
    inst✝ : IsScalarTower R S C
    f g : AlgHom S (TensorProduct R A B) C
    ha : Eq (f.comp Algebra.TensorProduct.includeLeft) (g.comp Algebra.TensorProdu …
    hb : Eq ((AlgHom.restrictScalars R f).comp Algebra.TensorProduct.includeRight) …
    a : A
    b : B
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry f.toLinearMap) a) b) (((Tensor …
  -/
  have := congr_arg₂ HMul.hMul (AlgHom.congr_fun ha a) (AlgHom.congr_fun hb b)
  /-
    case a.a.h.h
    R : Type uR
    S : Type uS
    A : Type uA
    B : Type uB
    C : Type uC
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : Semiring B
    inst✝⁸ : Algebra R B
    inst✝⁷ : Semiring C
    inst✝⁶ : Algebra R C
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra S A
    inst✝³ : Algebra R S
    inst✝² : Algebra S C
    inst✝¹ : IsScalarTower R S A
    inst✝ : IsScalarTower R S C
    f g : AlgHom S (TensorProduct R A B) C
    ha : Eq (f.comp Algebra.TensorProduct.includeLeft) (g.comp Algebra.TensorProdu …
    hb : Eq ((AlgHom.restrictScalars R f).comp Algebra.TensorProduct.includeRight) …
    a : A
    b : B
    this : Eq (HMul.hMul ((f.comp Algebra.TensorProduct.includeLeft) a) (((AlgHom. …
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry f.toLinearMap) a) b) (((Tensor …
  -/
  dsimp at *
  /-
    case a.a.h.h
    R : Type uR
    S : Type uS
    A : Type uA
    B : Type uB
    C : Type uC
    inst✝¹² : CommSemiring R
    inst✝¹¹ : Semiring A
    inst✝¹⁰ : Algebra R A
    inst✝⁹ : Semiring B
    inst✝⁸ : Algebra R B
    inst✝⁷ : Semiring C
    inst✝⁶ : Algebra R C
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra S A
    inst✝³ : Algebra R S
    inst✝² : Algebra S C
    inst✝¹ : IsScalarTower R S A
    inst✝ : IsScalarTower R S C
    f g : AlgHom S (TensorProduct R A B) C
    ha : Eq (f.comp Algebra.TensorProduct.includeLeft) (g.comp Algebra.TensorProdu …
    hb : Eq ((AlgHom.restrictScalars R f).comp Algebra.TensorProduct.includeRight) …
    a : A
    b : B
    this : Eq (HMul.hMul (f (TensorProduct.tmul R a 1)) (f (TensorProduct.tmul R 1 …
    ⊢ Eq (f (TensorProduct.tmul R a b)) (g (TensorProduct.tmul R a b))
  -/
  rwa [← map_mul, ← map_mul, tmul_mul_tmul, one_mul, mul_one] at this
  /-
    🎉 no goals
  -/


theorem ext' {g h : A ⊗[R] B →ₐ[S] C} (H : ∀ a b, g (a ⊗ₜ b) = h (a ⊗ₜ b)) : g = h :=
  ext (AlgHom.ext fun _ => H _ _) (AlgHom.ext fun _ => H _ _)


instance instAddCommGroupWithOne : AddCommGroupWithOne (A ⊗[R] B) where
  toAddCommGroup := TensorProduct.addCommGroup
  __ := instAddCommMonoidWithOne
  intCast z := z ⊗ₜ (1 : B)
                        /-
                          R : Type uR
                          S : Type uS
                          A : Type uA
                          B : Type uB
                          C : Type uC
                          D : Type uD
                          E : Type uE
                          F : Type uF
                          inst✝⁴ : CommSemiring R
                          inst✝³ : AddCommGroupWithOne A
                          inst✝² : Module R A
                          inst✝¹ : AddCommGroupWithOne B
                          inst✝ : Module R B
                          n : Nat
                          ⊢ Eq (IntCast.intCast ↑n) ↑n
                        -/
  intCast_ofNat n := by simp [natCast_def]
                        /-
                          🎉 no goals
                        -/
                          /-
                            R : Type uR
                            S : Type uS
                            A : Type uA
                            B : Type uB
                            C : Type uC
                            D : Type uD
                            E : Type uE
                            F : Type uF
                            inst✝⁴ : CommSemiring R
                            inst✝³ : AddCommGroupWithOne A
                            inst✝² : Module R A
                            inst✝¹ : AddCommGroupWithOne B
                            inst✝ : Module R B
                            n : Nat
                            ⊢ Eq (IntCast.intCast (Int.negSucc n)) (Neg.neg ↑(HAdd.hAdd n 1))
                          -/
  intCast_negSucc n := by simp [natCast_def, add_tmul, neg_tmul, one_def]
                          /-
                            🎉 no goals
                          -/


theorem intCast_def (z : ℤ) : (z : A ⊗[R] B) = (z : A) ⊗ₜ (1 : B) := rfl


instance instNonUnitalNonAssocRing : NonUnitalNonAssocRing (A ⊗[R] B) where
  toAddCommGroup := TensorProduct.addCommGroup
  __ := instNonUnitalNonAssocSemiring


instance instNonAssocRing : NonAssocRing (A ⊗[R] B) where
  toAddCommGroup := TensorProduct.addCommGroup
  __ := instNonAssocSemiring
  __ := instAddCommGroupWithOne


instance instNonUnitalRing : NonUnitalRing (A ⊗[R] B) where
  toAddCommGroup := TensorProduct.addCommGroup
  __ := instNonUnitalSemiring


instance instCommSemiring : CommSemiring (A ⊗[R] B) where
  toSemiring := inferInstance
  mul_comm x y := by
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝⁴ : CommSemiring R
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      x y : TensorProduct R A B
      ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
    -/
    refine TensorProduct.induction_on x ?_ ?_ ?_
      /-
        case refine_1
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring A
        inst✝² : Algebra R A
        inst✝¹ : CommSemiring B
        inst✝ : Algebra R B
        x y : TensorProduct R A B
        ⊢ Eq (HMul.hMul 0 y) (HMul.hMul y 0)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring A
        inst✝² : Algebra R A
        inst✝¹ : CommSemiring B
        inst✝ : Algebra R B
        x y : TensorProduct R A B
        ⊢ ∀ (x : A) (y_1 : B), Eq (HMul.hMul (TensorProduct.tmul R x y_1) y) (HMul.hMu …
      -/
    · intro a₁ b₁
      /-
        case refine_2
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring A
        inst✝² : Algebra R A
        inst✝¹ : CommSemiring B
        inst✝ : Algebra R B
        x y : TensorProduct R A B
        a₁ : A
        b₁ : B
        ⊢ Eq (HMul.hMul (TensorProduct.tmul R a₁ b₁) y) (HMul.hMul y (TensorProduct.tm …
      -/
      refine TensorProduct.induction_on y ?_ ?_ ?_
        /-
          case refine_2.refine_1
          R : Type uR
          S : Type uS
          A : Type uA
          B : Type uB
          C : Type uC
          D : Type uD
          E : Type uE
          F : Type uF
          inst✝⁴ : CommSemiring R
          inst✝³ : CommSemiring A
          inst✝² : Algebra R A
          inst✝¹ : CommSemiring B
          inst✝ : Algebra R B
          x y : TensorProduct R A B
          a₁ : A
          b₁ : B
          ⊢ Eq (HMul.hMul (TensorProduct.tmul R a₁ b₁) 0) (HMul.hMul 0 (TensorProduct.tm …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case refine_2.refine_2
          R : Type uR
          S : Type uS
          A : Type uA
          B : Type uB
          C : Type uC
          D : Type uD
          E : Type uE
          F : Type uF
          inst✝⁴ : CommSemiring R
          inst✝³ : CommSemiring A
          inst✝² : Algebra R A
          inst✝¹ : CommSemiring B
          inst✝ : Algebra R B
          x y : TensorProduct R A B
          a₁ : A
          b₁ : B
          ⊢ ∀ (x : A) (y : B), Eq (HMul.hMul (TensorProduct.tmul R a₁ b₁) (TensorProduct …
        -/
      · intro a₂ b₂
        /-
          case refine_2.refine_2
          R : Type uR
          S : Type uS
          A : Type uA
          B : Type uB
          C : Type uC
          D : Type uD
          E : Type uE
          F : Type uF
          inst✝⁴ : CommSemiring R
          inst✝³ : CommSemiring A
          inst✝² : Algebra R A
          inst✝¹ : CommSemiring B
          inst✝ : Algebra R B
          x y : TensorProduct R A B
          a₁ : A
          b₁ : B
          a₂ : A
          b₂ : B
          ⊢ Eq (HMul.hMul (TensorProduct.tmul R a₁ b₁) (TensorProduct.tmul R a₂ b₂)) (HM …
        -/
        simp [mul_comm]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.refine_3
          R : Type uR
          S : Type uS
          A : Type uA
          B : Type uB
          C : Type uC
          D : Type uD
          E : Type uE
          F : Type uF
          inst✝⁴ : CommSemiring R
          inst✝³ : CommSemiring A
          inst✝² : Algebra R A
          inst✝¹ : CommSemiring B
          inst✝ : Algebra R B
          x y : TensorProduct R A B
          a₁ : A
          b₁ : B
          ⊢ ∀ (x y : TensorProduct R A B), Eq (HMul.hMul (TensorProduct.tmul R a₁ b₁) x) …
        -/
      · intro a₂ b₂ ha hb
        /-
          case refine_2.refine_3
          R : Type uR
          S : Type uS
          A : Type uA
          B : Type uB
          C : Type uC
          D : Type uD
          E : Type uE
          F : Type uF
          inst✝⁴ : CommSemiring R
          inst✝³ : CommSemiring A
          inst✝² : Algebra R A
          inst✝¹ : CommSemiring B
          inst✝ : Algebra R B
          x y : TensorProduct R A B
          a₁ : A
          b₁ : B
          a₂ b₂ : TensorProduct R A B
          ha : Eq (HMul.hMul (TensorProduct.tmul R a₁ b₁) a₂) (HMul.hMul a₂ (TensorProdu …
          hb : Eq (HMul.hMul (TensorProduct.tmul R a₁ b₁) b₂) (HMul.hMul b₂ (TensorProdu …
          ⊢ Eq (HMul.hMul (TensorProduct.tmul R a₁ b₁) (HAdd.hAdd a₂ b₂)) (HMul.hMul (HA …
        -/
        simp [mul_add, add_mul, ha, hb]
        /-
          🎉 no goals
        -/
      /-
        case refine_3
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring A
        inst✝² : Algebra R A
        inst✝¹ : CommSemiring B
        inst✝ : Algebra R B
        x y : TensorProduct R A B
        ⊢ ∀ (x y_1 : TensorProduct R A B), Eq (HMul.hMul x y) (HMul.hMul y x) → Eq (HM …
      -/
    · intro x₁ x₂ h₁ h₂
      /-
        case refine_3
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring A
        inst✝² : Algebra R A
        inst✝¹ : CommSemiring B
        inst✝ : Algebra R B
        x y x₁ x₂ : TensorProduct R A B
        h₁ : Eq (HMul.hMul x₁ y) (HMul.hMul y x₁)
        h₂ : Eq (HMul.hMul x₂ y) (HMul.hMul y x₂)
        ⊢ Eq (HMul.hMul (HAdd.hAdd x₁ x₂) y) (HMul.hMul y (HAdd.hAdd x₁ x₂))
      -/
      simp [mul_add, add_mul, h₁, h₂]
      /-
        🎉 no goals
      -/


instance instRing : Ring (A ⊗[R] B) where
  toSemiring := instSemiring
  __ := TensorProduct.addCommGroup
  __ := instNonAssocRing


theorem intCast_def' (z : ℤ) : (z : A ⊗[R] B) = (1 : A) ⊗ₜ (z : B) := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommRing R
    inst✝³ : Ring A
    inst✝² : Algebra R A
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    z : Int
    ⊢ Eq (↑z) (TensorProduct.tmul R 1 ↑z)
  -/
  rw [intCast_def, ← zsmul_one, smul_tmul, zsmul_one]
  /-
    🎉 no goals
  -/

-- verify there are no diamonds

instance instCommRing : CommRing (A ⊗[R] B) :=
  { toRing := inferInstance
    mul_comm := mul_comm }


/-- `S ⊗[R] T` has a `T`-algebra structure. This is not a global instance or else the action of
`S` on `S ⊗[R] S` would be ambiguous. -/
abbrev rightAlgebra : Algebra B (A ⊗[R] B) :=
  includeRight.toRingHom.toAlgebra' fun b x => by
    suffices LinearMap.mulLeft R (includeRight b) = LinearMap.mulRight R (includeRight b) from
      congr($this x)
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      b : B
      x : TensorProduct R A B
      ⊢ Eq (LinearMap.mulLeft R (Algebra.TensorProduct.includeRight b)) (LinearMap.m …
    -/
    ext xa xb
    /-
      case a.h.h
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      b : B
      x : TensorProduct R A B
      xa : A
      xb : B
      ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (LinearMap.mulLeft R (Algebra. …
    -/
    simp [mul_comm]
    /-
      🎉 no goals
    -/


instance right_isScalarTower : IsScalarTower R B (A ⊗[R] B) :=
  IsScalarTower.of_algebraMap_eq fun r => (Algebra.TensorProduct.includeRight.commutes r).symm


/-- Build an algebra morphism from a linear map out of a tensor product, and evidence that on pure
tensors, it preserves multiplication and the identity.

Note that we state `h_one` using `1 ⊗ₜ[R] 1` instead of `1` so that lemmas about `f` applied to pure
tensors can be directly applied by the caller (without needing `TensorProduct.one_def`).
-/
def algHomOfLinearMapTensorProduct (f : A ⊗[R] B →ₗ[S] C)
    (h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B), f ((a₁ * a₂) ⊗ₜ (b₁ * b₂)) = f (a₁ ⊗ₜ b₁) * f (a₂ ⊗ₜ b₂))
    (h_one : f (1 ⊗ₜ[R] 1) = 1) : A ⊗[R] B →ₐ[S] C :=
  #adaptation_note
  /--
  After https://github.com/leanprover/lean4/pull/4119 we either need to specify
  the `(R := S) (A := A ⊗[R] B)` arguments, or use `set_option maxSynthPendingDepth 2 in`.
  -/
  AlgHom.ofLinearMap f h_one <| (f.map_mul_iff (R := S) (A := A ⊗[R] B)).2 <| by
    -- these instances are needed by the statement of `ext`, but not by the current definition.
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Algebra S A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra S C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : LinearMap (RingHom.id S) (TensorProduct R A B) C
      h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B), Eq (f (TensorProduct.tmul R (HMul.hMul a₁ a …
      h_one : Eq (f (TensorProduct.tmul R 1 1)) 1
      ⊢ Eq ((LinearMap.mul S (TensorProduct R A B)).compr₂ f) (((LinearMap.mul S C). …
    -/
    letI : Algebra R C := RestrictScalars.algebra R S C
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Algebra S A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra S C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : LinearMap (RingHom.id S) (TensorProduct R A B) C
      h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B), Eq (f (TensorProduct.tmul R (HMul.hMul a₁ a …
      h_one : Eq (f (TensorProduct.tmul R 1 1)) 1
      this : Algebra R C := RestrictScalars.algebra R S C
      ⊢ Eq ((LinearMap.mul S (TensorProduct R A B)).compr₂ f) (((LinearMap.mul S C). …
    -/
    letI : IsScalarTower R S C := RestrictScalars.isScalarTower R S C
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Algebra S A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra S C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : LinearMap (RingHom.id S) (TensorProduct R A B) C
      h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B), Eq (f (TensorProduct.tmul R (HMul.hMul a₁ a …
      h_one : Eq (f (TensorProduct.tmul R 1 1)) 1
      this✝ : Algebra R C := RestrictScalars.algebra R S C
      this : IsScalarTower R S C := RestrictScalars.isScalarTower R S C
      ⊢ Eq ((LinearMap.mul S (TensorProduct R A B)).compr₂ f) (((LinearMap.mul S C). …
    -/
    ext
    /-
      case a.h.h.a.h.h
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Algebra S A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra S C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : LinearMap (RingHom.id S) (TensorProduct R A B) C
      h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B), Eq (f (TensorProduct.tmul R (HMul.hMul a₁ a …
      h_one : Eq (f (TensorProduct.tmul R 1 1)) 1
      this✝ : Algebra R C := RestrictScalars.algebra R S C
      this : IsScalarTower R S C := RestrictScalars.isScalarTower R S C
      x✝³ : A
      x✝² : B
      x✝¹ : A
      x✝ : B
      ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((TensorProduct.AlgebraTensor …
    -/
    dsimp
    /-
      case a.h.h.a.h.h
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Semiring A
      inst✝⁸ : Algebra R A
      inst✝⁷ : Algebra S A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra S C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : LinearMap (RingHom.id S) (TensorProduct R A B) C
      h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B), Eq (f (TensorProduct.tmul R (HMul.hMul a₁ a …
      h_one : Eq (f (TensorProduct.tmul R 1 1)) 1
      this✝ : Algebra R C := RestrictScalars.algebra R S C
      this : IsScalarTower R S C := RestrictScalars.isScalarTower R S C
      x✝³ : A
      x✝² : B
      x✝¹ : A
      x✝ : B
      ⊢ Eq (f (TensorProduct.tmul R (HMul.hMul x✝³ x✝¹) (HMul.hMul x✝² x✝))) (HMul.h …
    -/
    exact h_mul _ _ _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem algHomOfLinearMapTensorProduct_apply (f h_mul h_one x) :
    (algHomOfLinearMapTensorProduct f h_mul h_one : A ⊗[R] B →ₐ[S] C) x = f x :=
  rfl


/-- Build an algebra equivalence from a linear equivalence out of a tensor product, and evidence
that on pure tensors, it preserves multiplication and the identity.

Note that we state `h_one` using `1 ⊗ₜ[R] 1` instead of `1` so that lemmas about `f` applied to pure
tensors can be directly applied by the caller (without needing `TensorProduct.one_def`).
-/
def algEquivOfLinearEquivTensorProduct (f : A ⊗[R] B ≃ₗ[S] C)
    (h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B), f ((a₁ * a₂) ⊗ₜ (b₁ * b₂)) = f (a₁ ⊗ₜ b₁) * f (a₂ ⊗ₜ b₂))
    (h_one : f (1 ⊗ₜ[R] 1) = 1) : A ⊗[R] B ≃ₐ[S] C :=
  { algHomOfLinearMapTensorProduct (f : A ⊗[R] B →ₗ[S] C) h_mul h_one, f with }


@[simp]
theorem algEquivOfLinearEquivTensorProduct_apply (f h_mul h_one x) :
    (algEquivOfLinearEquivTensorProduct f h_mul h_one : A ⊗[R] B ≃ₐ[S] C) x = f x :=
  rfl


/-- Build an algebra equivalence from a linear equivalence out of a triple tensor product,
and evidence of multiplicativity on pure tensors.
-/
def algEquivOfLinearEquivTripleTensorProduct (f : (A ⊗[R] B) ⊗[R] C ≃ₗ[R] D)
    (h_mul :
      ∀ (a₁ a₂ : A) (b₁ b₂ : B) (c₁ c₂ : C),
        f ((a₁ * a₂) ⊗ₜ (b₁ * b₂) ⊗ₜ (c₁ * c₂)) = f (a₁ ⊗ₜ b₁ ⊗ₜ c₁) * f (a₂ ⊗ₜ b₂ ⊗ₜ c₂))
    (h_one : f (((1 : A) ⊗ₜ[R] (1 : B)) ⊗ₜ[R] (1 : C)) = 1) :
    (A ⊗[R] B) ⊗[R] C ≃ₐ[R] D :=
  AlgEquiv.ofLinearEquiv f h_one <| f.map_mul_iff.2 <| by
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹³ : CommSemiring R
      inst✝¹² : CommSemiring S
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Semiring A
      inst✝⁹ : Algebra R A
      inst✝⁸ : Algebra S A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R B
      inst✝⁴ : Semiring C
      inst✝³ : Algebra S C
      inst✝² : Semiring D
      inst✝¹ : Algebra R D
      inst✝ : Algebra R C
      f : LinearEquiv (RingHom.id R) (TensorProduct R (TensorProduct R A B) C) D
      h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B) (c₁ c₂ : C), Eq (f (TensorProduct.tmul R (Te …
      h_one : Eq (f (TensorProduct.tmul R (TensorProduct.tmul R 1 1) 1)) 1
      ⊢ Eq ((LinearMap.mul R (TensorProduct R (TensorProduct R A B) C)).compr₂ ↑f) ( …
    -/
    ext
    /-
      case a.a.h.h.h.a.a.h.h.h
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹³ : CommSemiring R
      inst✝¹² : CommSemiring S
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Semiring A
      inst✝⁹ : Algebra R A
      inst✝⁸ : Algebra S A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R B
      inst✝⁴ : Semiring C
      inst✝³ : Algebra S C
      inst✝² : Semiring D
      inst✝¹ : Algebra R D
      inst✝ : Algebra R C
      f : LinearEquiv (RingHom.id R) (TensorProduct R (TensorProduct R A B) C) D
      h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B) (c₁ c₂ : C), Eq (f (TensorProduct.tmul R (Te …
      h_one : Eq (f (TensorProduct.tmul R (TensorProduct.tmul R 1 1) 1)) 1
      x✝⁵ : A
      x✝⁴ : B
      x✝³ : C
      x✝² : A
      x✝¹ : B
      x✝ : C
      ⊢ Eq ((((TensorProduct.AlgebraTensorModule.curry (TensorProduct.AlgebraTensorM …
    -/
    dsimp
    /-
      case a.a.h.h.h.a.a.h.h.h
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹³ : CommSemiring R
      inst✝¹² : CommSemiring S
      inst✝¹¹ : Algebra R S
      inst✝¹⁰ : Semiring A
      inst✝⁹ : Algebra R A
      inst✝⁸ : Algebra S A
      inst✝⁷ : IsScalarTower R S A
      inst✝⁶ : Semiring B
      inst✝⁵ : Algebra R B
      inst✝⁴ : Semiring C
      inst✝³ : Algebra S C
      inst✝² : Semiring D
      inst✝¹ : Algebra R D
      inst✝ : Algebra R C
      f : LinearEquiv (RingHom.id R) (TensorProduct R (TensorProduct R A B) C) D
      h_mul : ∀ (a₁ a₂ : A) (b₁ b₂ : B) (c₁ c₂ : C), Eq (f (TensorProduct.tmul R (Te …
      h_one : Eq (f (TensorProduct.tmul R (TensorProduct.tmul R 1 1) 1)) 1
      x✝⁵ : A
      x✝⁴ : B
      x✝³ : C
      x✝² : A
      x✝¹ : B
      x✝ : C
      ⊢ Eq (f (TensorProduct.tmul R (TensorProduct.tmul R (HMul.hMul x✝⁵ x✝²) (HMul. …
    -/
    exact h_mul _ _ _ _ _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem algEquivOfLinearEquivTripleTensorProduct_apply (f h_mul h_one x) :
    (algEquivOfLinearEquivTripleTensorProduct f h_mul h_one : (A ⊗[R] B) ⊗[R] C ≃ₐ[R] D) x = f x :=
  rfl


/-- The forward direction of the universal property of tensor products of algebras; any algebra
morphism from the tensor product can be factored as the product of two algebra morphisms that
commute.

See `Algebra.TensorProduct.liftEquiv` for the fact that every morphism factors this way. -/
def lift (f : A →ₐ[S] C) (g : B →ₐ[R] C) (hfg : ∀ x y, Commute (f x) (g y)) : (A ⊗[R] B) →ₐ[S] C :=
  algHomOfLinearMapTensorProduct
    (AlgebraTensorModule.lift <|
      letI restr : (C →ₗ[S] C) →ₗ[S] _ :=
        { toFun := (·.restrictScalars R)
          map_add' := fun _ _ => LinearMap.ext fun _ => rfl
          map_smul' := fun _ _ => LinearMap.ext fun _ => rfl }
      LinearMap.flip <| (restr ∘ₗ LinearMap.mul S C ∘ₗ f.toLinearMap).flip ∘ₗ g)
    (fun a₁ a₂ b₁ b₂ => show f (a₁ * a₂) * g (b₁ * b₂) = f a₁ * g b₁ * (f a₂ * g b₂) by
      /-
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝¹⁴ : CommSemiring R
        inst✝¹³ : CommSemiring S
        inst✝¹² : Algebra R S
        inst✝¹¹ : Semiring A
        inst✝¹⁰ : Algebra R A
        inst✝⁹ : Algebra S A
        inst✝⁸ : IsScalarTower R S A
        inst✝⁷ : Semiring B
        inst✝⁶ : Algebra R B
        inst✝⁵ : Semiring C
        inst✝⁴ : Algebra S C
        inst✝³ : Semiring D
        inst✝² : Algebra R D
        inst✝¹ : Algebra R C
        inst✝ : IsScalarTower R S C
        f : AlgHom S A C
        g : AlgHom R B C
        hfg : ∀ (x : A) (y : B), Commute (f x) (g y)
        a₁ a₂ : A
        b₁ b₂ : B
        ⊢ Eq (HMul.hMul (f (HMul.hMul a₁ a₂)) (g (HMul.hMul b₁ b₂))) (HMul.hMul (HMul. …
      -/
      rw [map_mul, map_mul, (hfg a₂ b₁).mul_mul_mul_comm])
      /-
        🎉 no goals
      -/
                           /-
                             R : Type uR
                             S : Type uS
                             A : Type uA
                             B : Type uB
                             C : Type uC
                             D : Type uD
                             E : Type uE
                             F : Type uF
                             inst✝¹⁴ : CommSemiring R
                             inst✝¹³ : CommSemiring S
                             inst✝¹² : Algebra R S
                             inst✝¹¹ : Semiring A
                             inst✝¹⁰ : Algebra R A
                             inst✝⁹ : Algebra S A
                             inst✝⁸ : IsScalarTower R S A
                             inst✝⁷ : Semiring B
                             inst✝⁶ : Algebra R B
                             inst✝⁵ : Semiring C
                             inst✝⁴ : Algebra S C
                             inst✝³ : Semiring D
                             inst✝² : Algebra R D
                             inst✝¹ : Algebra R C
                             inst✝ : IsScalarTower R S C
                             f : AlgHom S A C
                             g : AlgHom R B C
                             hfg : ∀ (x : A) (y : B), Commute (f x) (g y)
                             ⊢ Eq (HMul.hMul (f 1) (g 1)) 1
                           -/
    (show f 1 * g 1 = 1 by rw [map_one, map_one, one_mul])
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem lift_tmul (f : A →ₐ[S] C) (g : B →ₐ[R] C) (hfg : ∀ x y, Commute (f x) (g y))
    (a : A) (b : B) :
    lift f g hfg (a ⊗ₜ b) = f a * g b :=
  rfl


@[simp]
theorem lift_includeLeft_includeRight :
    lift includeLeft includeRight (fun _ _ => (Commute.one_right _).tmul (Commute.one_left _)) =
      .id S (A ⊗[R] B) := by
  /-
    R : Type uR
    S : Type uS
    A : Type uA
    B : Type uB
    inst✝⁸ : CommSemiring R
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Semiring A
    inst✝⁴ : Algebra R A
    inst✝³ : Algebra S A
    inst✝² : IsScalarTower R S A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    ⊢ Eq (Algebra.TensorProduct.lift Algebra.TensorProduct.includeLeft Algebra.Ten …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


@[simp]
theorem lift_comp_includeLeft (f : A →ₐ[S] C) (g : B →ₐ[R] C) (hfg : ∀ x y, Commute (f x) (g y)) :
    (lift f g hfg).comp includeLeft = f :=
                   /-
                     R : Type uR
                     S : Type uS
                     A : Type uA
                     B : Type uB
                     C : Type uC
                     inst✝¹² : CommSemiring R
                     inst✝¹¹ : CommSemiring S
                     inst✝¹⁰ : Algebra R S
                     inst✝⁹ : Semiring A
                     inst✝⁸ : Algebra R A
                     inst✝⁷ : Algebra S A
                     inst✝⁶ : IsScalarTower R S A
                     inst✝⁵ : Semiring B
                     inst✝⁴ : Algebra R B
                     inst✝³ : Semiring C
                     inst✝² : Algebra S C
                     inst✝¹ : Algebra R C
                     inst✝ : IsScalarTower R S C
                     f : AlgHom S A C
                     g : AlgHom R B C
                     hfg : ∀ (x : A) (y : B), Commute (f x) (g y)
                     ⊢ ∀ (x : A), Eq (((Algebra.TensorProduct.lift f g hfg).comp Algebra.TensorProd …
                   -/
  AlgHom.ext <| by simp
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem lift_comp_includeRight (f : A →ₐ[S] C) (g : B →ₐ[R] C) (hfg : ∀ x y, Commute (f x) (g y)) :
    ((lift f g hfg).restrictScalars R).comp includeRight = g :=
                   /-
                     R : Type uR
                     S : Type uS
                     A : Type uA
                     B : Type uB
                     C : Type uC
                     inst✝¹² : CommSemiring R
                     inst✝¹¹ : CommSemiring S
                     inst✝¹⁰ : Algebra R S
                     inst✝⁹ : Semiring A
                     inst✝⁸ : Algebra R A
                     inst✝⁷ : Algebra S A
                     inst✝⁶ : IsScalarTower R S A
                     inst✝⁵ : Semiring B
                     inst✝⁴ : Algebra R B
                     inst✝³ : Semiring C
                     inst✝² : Algebra S C
                     inst✝¹ : Algebra R C
                     inst✝ : IsScalarTower R S C
                     f : AlgHom S A C
                     g : AlgHom R B C
                     hfg : ∀ (x : A) (y : B), Commute (f x) (g y)
                     ⊢ ∀ (x : B), Eq (((AlgHom.restrictScalars R (Algebra.TensorProduct.lift f g hf …
                   -/
  AlgHom.ext <| by simp
                   /-
                     🎉 no goals
                   -/


/-- The universal property of the tensor product of algebras.

Pairs of algebra morphisms that commute are equivalent to algebra morphisms from the tensor product.

This is `Algebra.TensorProduct.lift` as an equivalence.

See also `GradedTensorProduct.liftEquiv` for an alternative commutativity requirement for graded
algebra. -/
@[simps]
def liftEquiv : {fg : (A →ₐ[S] C) × (B →ₐ[R] C) // ∀ x y, Commute (fg.1 x) (fg.2 y)}
    ≃ ((A ⊗[R] B) →ₐ[S] C) where
  toFun fg := lift fg.val.1 fg.val.2 fg.prop
  invFun f' := ⟨(f'.comp includeLeft, (f'.restrictScalars R).comp includeRight), fun _ _ =>
    ((Commute.one_right _).tmul (Commute.one_left _)).map f'⟩
                    /-
                      R : Type uR
                      S : Type uS
                      A : Type uA
                      B : Type uB
                      C : Type uC
                      D : Type uD
                      E : Type uE
                      F : Type uF
                      inst✝¹⁴ : CommSemiring R
                      inst✝¹³ : CommSemiring S
                      inst✝¹² : Algebra R S
                      inst✝¹¹ : Semiring A
                      inst✝¹⁰ : Algebra R A
                      inst✝⁹ : Algebra S A
                      inst✝⁸ : IsScalarTower R S A
                      inst✝⁷ : Semiring B
                      inst✝⁶ : Algebra R B
                      inst✝⁵ : Semiring C
                      inst✝⁴ : Algebra S C
                      inst✝³ : Semiring D
                      inst✝² : Algebra R D
                      inst✝¹ : Algebra R C
                      inst✝ : IsScalarTower R S C
                      fg : Subtype fun fg => ∀ (x : A) (y : B), Commute (fg.1 x) (fg.2 y)
                      ⊢ Eq ((fun f' => ⟨{ fst := f'.comp Algebra.TensorProduct.includeLeft, snd := ( …
                    -/
                            /-
                              🎉 no goals
                            -/
  left_inv fg := by ext <;> simp
                            /-
                              🎉 no goals
                            -/
                     /-
                       R : Type uR
                       S : Type uS
                       A : Type uA
                       B : Type uB
                       C : Type uC
                       D : Type uD
                       E : Type uE
                       F : Type uF
                       inst✝¹⁴ : CommSemiring R
                       inst✝¹³ : CommSemiring S
                       inst✝¹² : Algebra R S
                       inst✝¹¹ : Semiring A
                       inst✝¹⁰ : Algebra R A
                       inst✝⁹ : Algebra S A
                       inst✝⁸ : IsScalarTower R S A
                       inst✝⁷ : Semiring B
                       inst✝⁶ : Algebra R B
                       inst✝⁵ : Semiring C
                       inst✝⁴ : Algebra S C
                       inst✝³ : Semiring D
                       inst✝² : Algebra R D
                       inst✝¹ : Algebra R C
                       inst✝ : IsScalarTower R S C
                       f' : AlgHom S (TensorProduct R A B) C
                       ⊢ Eq ((fun fg => Algebra.TensorProduct.lift (↑fg).1 (↑fg).2 ⋯) ((fun f' => ⟨{  …
                     -/
                             /-
                               🎉 no goals
                             -/
  right_inv f' := by ext <;> simp
                             /-
                               🎉 no goals
                             -/


/-- The base ring is a left identity for the tensor product of algebra, up to algebra isomorphism.
-/
protected nonrec def lid : R ⊗[R] A ≃ₐ[R] A :=
  algEquivOfLinearEquivTensorProduct (TensorProduct.lid R A) (by
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹⁸ : CommSemiring R
      inst✝¹⁷ : CommSemiring S
      inst✝¹⁶ : Algebra R S
      inst✝¹⁵ : Semiring A
      inst✝¹⁴ : Algebra R A
      inst✝¹³ : Algebra S A
      inst✝¹² : IsScalarTower R S A
      inst✝¹¹ : Semiring B
      inst✝¹⁰ : Algebra R B
      inst✝⁹ : Algebra S B
      inst✝⁸ : IsScalarTower R S B
      inst✝⁷ : Semiring C
      inst✝⁶ : Algebra R C
      inst✝⁵ : Semiring D
      inst✝⁴ : Algebra R D
      inst✝³ : Semiring E
      inst✝² : Algebra R E
      inst✝¹ : Semiring F
      inst✝ : Algebra R F
      ⊢ ∀ (a₁ a₂ : R) (b₁ b₂ : A), Eq ((_root_.TensorProduct.lid R A) (TensorProduct …
    -/
    simp only [mul_smul, lid_tmul, Algebra.smul_mul_assoc, Algebra.mul_smul_comm]
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹⁸ : CommSemiring R
      inst✝¹⁷ : CommSemiring S
      inst✝¹⁶ : Algebra R S
      inst✝¹⁵ : Semiring A
      inst✝¹⁴ : Algebra R A
      inst✝¹³ : Algebra S A
      inst✝¹² : IsScalarTower R S A
      inst✝¹¹ : Semiring B
      inst✝¹⁰ : Algebra R B
      inst✝⁹ : Algebra S B
      inst✝⁸ : IsScalarTower R S B
      inst✝⁷ : Semiring C
      inst✝⁶ : Algebra R C
      inst✝⁵ : Semiring D
      inst✝⁴ : Algebra R D
      inst✝³ : Semiring E
      inst✝² : Algebra R E
      inst✝¹ : Semiring F
      inst✝ : Algebra R F
      ⊢ ∀ (a₁ a₂ : R) (b₁ b₂ : A), Eq (HSMul.hSMul a₁ (HSMul.hSMul a₂ (HMul.hMul b₁  …
    -/
    simp_rw [← mul_smul, mul_comm]
    /-
      R : Type uR
      S : Type uS
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      E : Type uE
      F : Type uF
      inst✝¹⁸ : CommSemiring R
      inst✝¹⁷ : CommSemiring S
      inst✝¹⁶ : Algebra R S
      inst✝¹⁵ : Semiring A
      inst✝¹⁴ : Algebra R A
      inst✝¹³ : Algebra S A
      inst✝¹² : IsScalarTower R S A
      inst✝¹¹ : Semiring B
      inst✝¹⁰ : Algebra R B
      inst✝⁹ : Algebra S B
      inst✝⁸ : IsScalarTower R S B
      inst✝⁷ : Semiring C
      inst✝⁶ : Algebra R C
      inst✝⁵ : Semiring D
      inst✝⁴ : Algebra R D
      inst✝³ : Semiring E
      inst✝² : Algebra R E
      inst✝¹ : Semiring F
      inst✝ : Algebra R F
      ⊢ R → R → A → A → True
    -/
    simp)
    /-
      🎉 no goals
    -/
        /-
          R : Type uR
          S : Type uS
          A : Type uA
          B : Type uB
          C : Type uC
          D : Type uD
          E : Type uE
          F : Type uF
          inst✝¹⁸ : CommSemiring R
          inst✝¹⁷ : CommSemiring S
          inst✝¹⁶ : Algebra R S
          inst✝¹⁵ : Semiring A
          inst✝¹⁴ : Algebra R A
          inst✝¹³ : Algebra S A
          inst✝¹² : IsScalarTower R S A
          inst✝¹¹ : Semiring B
          inst✝¹⁰ : Algebra R B
          inst✝⁹ : Algebra S B
          inst✝⁸ : IsScalarTower R S B
          inst✝⁷ : Semiring C
          inst✝⁶ : Algebra R C
          inst✝⁵ : Semiring D
          inst✝⁴ : Algebra R D
          inst✝³ : Semiring E
          inst✝² : Algebra R E
          inst✝¹ : Semiring F
          inst✝ : Algebra R F
          ⊢ Eq ((_root_.TensorProduct.lid R A) (TensorProduct.tmul R 1 1)) 1
        -/
    (by simp [Algebra.smul_def])
        /-
          🎉 no goals
        -/


@[simp] theorem lid_toLinearEquiv :
    (TensorProduct.lid R A).toLinearEquiv = _root_.TensorProduct.lid R A := rfl


variable {R} {A} in
@[simp]
theorem lid_tmul (r : R) (a : A) : TensorProduct.lid R A (r ⊗ₜ a) = r • a := rfl


variable {A} in
@[simp]
theorem lid_symm_apply (a : A) : (TensorProduct.lid R A).symm a = 1 ⊗ₜ a := rfl


/-- The base ring is a right identity for the tensor product of algebra, up to algebra isomorphism.

Note that if `A` is commutative this can be instantiated with `S = A`.
-/
protected nonrec def rid : A ⊗[R] R ≃ₐ[S] A :=
  algEquivOfLinearEquivTensorProduct (AlgebraTensorModule.rid R S A)
    (fun a₁ a₂ r₁ r₂ => smul_mul_smul_comm r₁ a₁ r₂ a₂ |>.symm)
    (one_smul R _)


@[simp] theorem rid_toLinearEquiv :
    (TensorProduct.rid R S A).toLinearEquiv = AlgebraTensorModule.rid R S A := rfl


variable {R A} in
@[simp]
theorem rid_tmul (r : R) (a : A) : TensorProduct.rid R S A (a ⊗ₜ r) = r • a := rfl


variable {A} in
@[simp]
theorem rid_symm_apply (a : A) : (TensorProduct.rid R S A).symm a = a ⊗ₜ 1 := rfl


/-- If A and B are both R- and S-algebras and their actions on them commute,
and if the S-action on `A ⊗[R] B` can switch between the two factors, then there is a
canonical S-algebra homomorphism from `A ⊗[S] B` to `A ⊗[R] B`. -/
def mapOfCompatibleSMul : A ⊗[S] B →ₐ[S] A ⊗[R] B :=
  .ofLinearMap (_root_.TensorProduct.mapOfCompatibleSMul R S A B) rfl fun x ↦
                       /-
                         R✝ : Type uR
                         S✝ : Type uS
                         A✝ : Type uA
                         B✝ : Type uB
                         C : Type uC
                         D : Type uD
                         E : Type uE
                         F : Type uF
                         inst✝²⁸ : CommSemiring R✝
                         inst✝²⁷ : CommSemiring S✝
                         inst✝²⁶ : Algebra R✝ S✝
                         inst✝²⁵ : Semiring A✝
                         inst✝²⁴ : Algebra R✝ A✝
                         inst✝²³ : Algebra S✝ A✝
                         inst✝²² : IsScalarTower R✝ S✝ A✝
                         inst✝²¹ : Semiring B✝
                         inst✝²⁰ : Algebra R✝ B✝
                         inst✝¹⁹ : Algebra S✝ B✝
                         inst✝¹⁸ : IsScalarTower R✝ S✝ B✝
                         inst✝¹⁷ : Semiring C
                         inst✝¹⁶ : Algebra R✝ C
                         inst✝¹⁵ : Semiring D
                         inst✝¹⁴ : Algebra R✝ D
                         inst✝¹³ : Semiring E
                         inst✝¹² : Algebra R✝ E
                         inst✝¹¹ : Semiring F
                         inst✝¹⁰ : Algebra R✝ F
                         R : Type u_1
                         S : Type u_2
                         A : Type u_3
                         B : Type u_4
                         inst✝⁹ : CommSemiring R
                         inst✝⁸ : CommSemiring S
                         inst✝⁷ : Semiring A
                         inst✝⁶ : Semiring B
                         inst✝⁵ : Algebra R A
                         inst✝⁴ : Algebra R B
                         inst✝³ : Algebra S A
                         inst✝² : Algebra S B
                         inst✝¹ : SMulCommClass R S A
                         inst✝ : TensorProduct.CompatibleSMul R S A B
                         x : TensorProduct S A B
                         ⊢ ∀ (y : TensorProduct S A B), Eq ((_root_.TensorProduct.mapOfCompatibleSMul R …
                       -/
                       /-
                         🎉 no goals
                       -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    x.induction_on (by simp) (fun _ _ y ↦ y.induction_on (by simp) (by simp)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                        /-
                          R✝ : Type uR
                          S✝ : Type uS
                          A✝ : Type uA
                          B✝ : Type uB
                          C : Type uC
                          D : Type uD
                          E : Type uE
                          F : Type uF
                          inst✝²⁸ : CommSemiring R✝
                          inst✝²⁷ : CommSemiring S✝
                          inst✝²⁶ : Algebra R✝ S✝
                          inst✝²⁵ : Semiring A✝
                          inst✝²⁴ : Algebra R✝ A✝
                          inst✝²³ : Algebra S✝ A✝
                          inst✝²² : IsScalarTower R✝ S✝ A✝
                          inst✝²¹ : Semiring B✝
                          inst✝²⁰ : Algebra R✝ B✝
                          inst✝¹⁹ : Algebra S✝ B✝
                          inst✝¹⁸ : IsScalarTower R✝ S✝ B✝
                          inst✝¹⁷ : Semiring C
                          inst✝¹⁶ : Algebra R✝ C
                          inst✝¹⁵ : Semiring D
                          inst✝¹⁴ : Algebra R✝ D
                          inst✝¹³ : Semiring E
                          inst✝¹² : Algebra R✝ E
                          inst✝¹¹ : Semiring F
                          inst✝¹⁰ : Algebra R✝ F
                          R : Type u_1
                          S : Type u_2
                          A : Type u_3
                          B : Type u_4
                          inst✝⁹ : CommSemiring R
                          inst✝⁸ : CommSemiring S
                          inst✝⁷ : Semiring A
                          inst✝⁶ : Semiring B
                          inst✝⁵ : Algebra R A
                          inst✝⁴ : Algebra R B
                          inst✝³ : Algebra S A
                          inst✝² : Algebra S B
                          inst✝¹ : SMulCommClass R S A
                          inst✝ : TensorProduct.CompatibleSMul R S A B
                          x : TensorProduct S A B
                          x✝³ : A
                          x✝² : B
                          y x✝¹ x✝ : TensorProduct S A B
                          h : Eq ((_root_.TensorProduct.mapOfCompatibleSMul R S A B) (HMul.hMul (TensorP …
                          h' : Eq ((_root_.TensorProduct.mapOfCompatibleSMul R S A B) (HMul.hMul (Tensor …
                          ⊢ Eq ((_root_.TensorProduct.mapOfCompatibleSMul R S A B) (HMul.hMul (TensorPro …
                        -/
      fun _ _ h h' ↦ by simp only [mul_add, map_add, h, h'])
                        /-
                          🎉 no goals
                        -/
                          /-
                            R✝ : Type uR
                            S✝ : Type uS
                            A✝ : Type uA
                            B✝ : Type uB
                            C : Type uC
                            D : Type uD
                            E : Type uE
                            F : Type uF
                            inst✝²⁸ : CommSemiring R✝
                            inst✝²⁷ : CommSemiring S✝
                            inst✝²⁶ : Algebra R✝ S✝
                            inst✝²⁵ : Semiring A✝
                            inst✝²⁴ : Algebra R✝ A✝
                            inst✝²³ : Algebra S✝ A✝
                            inst✝²² : IsScalarTower R✝ S✝ A✝
                            inst✝²¹ : Semiring B✝
                            inst✝²⁰ : Algebra R✝ B✝
                            inst✝¹⁹ : Algebra S✝ B✝
                            inst✝¹⁸ : IsScalarTower R✝ S✝ B✝
                            inst✝¹⁷ : Semiring C
                            inst✝¹⁶ : Algebra R✝ C
                            inst✝¹⁵ : Semiring D
                            inst✝¹⁴ : Algebra R✝ D
                            inst✝¹³ : Semiring E
                            inst✝¹² : Algebra R✝ E
                            inst✝¹¹ : Semiring F
                            inst✝¹⁰ : Algebra R✝ F
                            R : Type u_1
                            S : Type u_2
                            A : Type u_3
                            B : Type u_4
                            inst✝⁹ : CommSemiring R
                            inst✝⁸ : CommSemiring S
                            inst✝⁷ : Semiring A
                            inst✝⁶ : Semiring B
                            inst✝⁵ : Algebra R A
                            inst✝⁴ : Algebra R B
                            inst✝³ : Algebra S A
                            inst✝² : Algebra S B
                            inst✝¹ : SMulCommClass R S A
                            inst✝ : TensorProduct.CompatibleSMul R S A B
                            x x✝² x✝¹ : TensorProduct S A B
                            h : ∀ (y : TensorProduct S A B), Eq ((_root_.TensorProduct.mapOfCompatibleSMul …
                            h' : ∀ (y : TensorProduct S A B), Eq ((_root_.TensorProduct.mapOfCompatibleSMu …
                            x✝ : TensorProduct S A B
                            ⊢ Eq ((_root_.TensorProduct.mapOfCompatibleSMul R S A B) (HMul.hMul (HAdd.hAdd …
                          -/
      fun _ _ h h' _ ↦ by simp only [add_mul, map_add, h, h']
                          /-
                            🎉 no goals
                          -/


@[simp] theorem mapOfCompatibleSMul_tmul (m n) : mapOfCompatibleSMul R S A B (m ⊗ₜ n) = m ⊗ₜ n :=
  rfl


theorem mapOfCompatibleSMul_surjective : Function.Surjective (mapOfCompatibleSMul R S A B) :=
  _root_.TensorProduct.mapOfCompatibleSMul_surjective R S A B


/-- `mapOfCompatibleSMul R S A B` is also A-linear. -/
def mapOfCompatibleSMul' : A ⊗[S] B →ₐ[R] A ⊗[R] B :=
  .ofLinearMap (_root_.TensorProduct.mapOfCompatibleSMul' R S A B) rfl
    (map_mul <| mapOfCompatibleSMul R S A B)


/-- If the R- and S-actions on A and B satisfy `CompatibleSMul` both ways,
then `A ⊗[S] B` is canonically isomorphic to `A ⊗[R] B`. -/
def equivOfCompatibleSMul [CompatibleSMul S R A B] : A ⊗[S] B ≃ₐ[S] A ⊗[R] B where
  __ := mapOfCompatibleSMul R S A B
  invFun := mapOfCompatibleSMul S R A B
  __ := _root_.TensorProduct.equivOfCompatibleSMul R S A B


/-- If the R- and S- action on S and A satisfy `CompatibleSMul` both ways,
then `S ⊗[R] A` is canonically isomorphic to `A`. -/
def lidOfCompatibleSMul : S ⊗[R] A ≃ₐ[S] A :=
  (equivOfCompatibleSMul R S S A).symm.trans (TensorProduct.lid _ _)


theorem lidOfCompatibleSMul_tmul (s a) : lidOfCompatibleSMul R S A (s ⊗ₜ[R] a) = s • a := rfl


unseal mul in
/-- The tensor product of R-algebras is commutative, up to algebra isomorphism.
-/
protected def comm : A ⊗[R] B ≃ₐ[R] B ⊗[R] A :=
  algEquivOfLinearEquivTensorProduct (_root_.TensorProduct.comm R A B) (fun _ _ _ _ => rfl) rfl


@[simp] theorem comm_toLinearEquiv :
    (Algebra.TensorProduct.comm R A B).toLinearEquiv = _root_.TensorProduct.comm R A B := rfl


variable {A B} in
@[simp]
theorem comm_tmul (a : A) (b : B) :
    TensorProduct.comm R A B (a ⊗ₜ b) = b ⊗ₜ a :=
  rfl


variable {A B} in
@[simp]
theorem comm_symm_tmul (a : A) (b : B) :
    (TensorProduct.comm R A B).symm (b ⊗ₜ a) = a ⊗ₜ b :=
  rfl


theorem comm_symm :
    (TensorProduct.comm R A B).symm = TensorProduct.comm R B A := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    inst✝¹ : Semiring B
    inst✝ : Algebra R B
    ⊢ Eq (Algebra.TensorProduct.comm R A B).symm (Algebra.TensorProduct.comm R B A)
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


theorem adjoin_tmul_eq_top : adjoin R { t : A ⊗[R] B | ∃ a b, a ⊗ₜ[R] b = t } = ⊤ :=
  top_le_iff.mp <| (top_le_iff.mpr <| span_tmul_eq_top R A B).trans (span_le_adjoin R _)


unseal mul in
theorem assoc_aux_1 (a₁ a₂ : A) (b₁ b₂ : B) (c₁ c₂ : C) :
    (TensorProduct.assoc R A B C) (((a₁ * a₂) ⊗ₜ[R] (b₁ * b₂)) ⊗ₜ[R] (c₁ * c₂)) =
      (TensorProduct.assoc R A B C) ((a₁ ⊗ₜ[R] b₁) ⊗ₜ[R] c₁) *
        (TensorProduct.assoc R A B C) ((a₂ ⊗ₜ[R] b₂) ⊗ₜ[R] c₂) :=
  rfl


theorem assoc_aux_2 : (TensorProduct.assoc R A B C) ((1 ⊗ₜ[R] 1) ⊗ₜ[R] 1) = 1 :=
  rfl


/-- The associator for tensor product of R-algebras, as an algebra isomorphism. -/
protected def assoc : (A ⊗[R] B) ⊗[R] C ≃ₐ[R] A ⊗[R] B ⊗[R] C :=
  algEquivOfLinearEquivTripleTensorProduct
    (_root_.TensorProduct.assoc R A B C)
    Algebra.TensorProduct.assoc_aux_1
    Algebra.TensorProduct.assoc_aux_2


@[simp] theorem assoc_toLinearEquiv :
  (Algebra.TensorProduct.assoc R A B C).toLinearEquiv = _root_.TensorProduct.assoc R A B C := rfl


@[simp]
theorem assoc_tmul (a : A) (b : B) (c : C) :
    Algebra.TensorProduct.assoc R A B C ((a ⊗ₜ b) ⊗ₜ c) = a ⊗ₜ (b ⊗ₜ c) :=
  rfl


@[simp]
theorem assoc_symm_tmul (a : A) (b : B) (c : C) :
    (Algebra.TensorProduct.assoc R A B C).symm (a ⊗ₜ (b ⊗ₜ c)) = (a ⊗ₜ b) ⊗ₜ c :=
  rfl


/-- The tensor product of a pair of algebra morphisms. -/
def map (f : A →ₐ[S] B) (g : C →ₐ[R] D) : A ⊗[R] C →ₐ[S] B ⊗[R] D :=
                                                                                           /-
                                                                                             R : Type uR
                                                                                             S : Type uS
                                                                                             A : Type uA
                                                                                             B : Type uB
                                                                                             C : Type uC
                                                                                             D : Type uD
                                                                                             E : Type uE
                                                                                             F : Type uF
                                                                                             inst✝¹⁸ : CommSemiring R
                                                                                             inst✝¹⁷ : CommSemiring S
                                                                                             inst✝¹⁶ : Algebra R S
                                                                                             inst✝¹⁵ : Semiring A
                                                                                             inst✝¹⁴ : Algebra R A
                                                                                             inst✝¹³ : Algebra S A
                                                                                             inst✝¹² : IsScalarTower R S A
                                                                                             inst✝¹¹ : Semiring B
                                                                                             inst✝¹⁰ : Algebra R B
                                                                                             inst✝⁹ : Algebra S B
                                                                                             inst✝⁸ : IsScalarTower R S B
                                                                                             inst✝⁷ : Semiring C
                                                                                             inst✝⁶ : Algebra R C
                                                                                             inst✝⁵ : Semiring D
                                                                                             inst✝⁴ : Algebra R D
                                                                                             inst✝³ : Semiring E
                                                                                             inst✝² : Algebra R E
                                                                                             inst✝¹ : Semiring F
                                                                                             inst✝ : Algebra R F
                                                                                             f : AlgHom S A B
                                                                                             g : AlgHom R C D
                                                                                             ⊢ ∀ (a₁ a₂ : A) (b₁ b₂ : C), Eq ((TensorProduct.AlgebraTensorModule.map f.toLi …
                                                                                           -/
  algHomOfLinearMapTensorProduct (AlgebraTensorModule.map f.toLinearMap g.toLinearMap) (by simp)
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
        /-
          R : Type uR
          S : Type uS
          A : Type uA
          B : Type uB
          C : Type uC
          D : Type uD
          E : Type uE
          F : Type uF
          inst✝¹⁸ : CommSemiring R
          inst✝¹⁷ : CommSemiring S
          inst✝¹⁶ : Algebra R S
          inst✝¹⁵ : Semiring A
          inst✝¹⁴ : Algebra R A
          inst✝¹³ : Algebra S A
          inst✝¹² : IsScalarTower R S A
          inst✝¹¹ : Semiring B
          inst✝¹⁰ : Algebra R B
          inst✝⁹ : Algebra S B
          inst✝⁸ : IsScalarTower R S B
          inst✝⁷ : Semiring C
          inst✝⁶ : Algebra R C
          inst✝⁵ : Semiring D
          inst✝⁴ : Algebra R D
          inst✝³ : Semiring E
          inst✝² : Algebra R E
          inst✝¹ : Semiring F
          inst✝ : Algebra R F
          f : AlgHom S A B
          g : AlgHom R C D
          ⊢ Eq ((TensorProduct.AlgebraTensorModule.map f.toLinearMap g.toLinearMap) (Ten …
        -/
    (by simp [one_def])
        /-
          🎉 no goals
        -/


@[simp]
theorem map_tmul (f : A →ₐ[S] B) (g : C →ₐ[R] D) (a : A) (c : C) : map f g (a ⊗ₜ c) = f a ⊗ₜ g c :=
  rfl


@[simp]
theorem map_id : map (.id S A) (.id R C) = .id S _ :=
  ext (AlgHom.ext fun _ => rfl) (AlgHom.ext fun _ => rfl)


theorem map_comp [Algebra S C] [IsScalarTower R S C]
    (f₂ : B →ₐ[S] C) (f₁ : A →ₐ[S] B) (g₂ : E →ₐ[R] F) (g₁ : D →ₐ[R] E) :
    map (f₂.comp f₁) (g₂.comp g₁) = (map f₂ g₂).comp (map f₁ g₁) :=
  ext (AlgHom.ext fun _ => rfl) (AlgHom.ext fun _ => rfl)


lemma map_id_comp (g₂ : E →ₐ[R] F) (g₁ : D →ₐ[R] E) :
    map (AlgHom.id S A) (g₂.comp g₁) = (map (AlgHom.id S A) g₂).comp (map (AlgHom.id S A) g₁) :=
  ext (AlgHom.ext fun _ => rfl) (AlgHom.ext fun _ => rfl)


lemma map_comp_id [Algebra S C] [IsScalarTower R S C]
    (f₂ : B →ₐ[S] C) (f₁ : A →ₐ[S] B) :
    map (f₂.comp f₁) (AlgHom.id R E) = (map f₂ (AlgHom.id R E)).comp (map f₁ (AlgHom.id R E)) :=
  ext (AlgHom.ext fun _ => rfl) (AlgHom.ext fun _ => rfl)


@[simp]
theorem map_comp_includeLeft (f : A →ₐ[S] B) (g : C →ₐ[R] D) :
    (map f g).comp includeLeft = includeLeft.comp f :=
                   /-
                     R : Type uR
                     S : Type uS
                     A : Type uA
                     B : Type uB
                     C : Type uC
                     D : Type uD
                     inst✝¹⁴ : CommSemiring R
                     inst✝¹³ : CommSemiring S
                     inst✝¹² : Algebra R S
                     inst✝¹¹ : Semiring A
                     inst✝¹⁰ : Algebra R A
                     inst✝⁹ : Algebra S A
                     inst✝⁸ : IsScalarTower R S A
                     inst✝⁷ : Semiring B
                     inst✝⁶ : Algebra R B
                     inst✝⁵ : Algebra S B
                     inst✝⁴ : IsScalarTower R S B
                     inst✝³ : Semiring C
                     inst✝² : Algebra R C
                     inst✝¹ : Semiring D
                     inst✝ : Algebra R D
                     f : AlgHom S A B
                     g : AlgHom R C D
                     ⊢ ∀ (x : A), Eq (((Algebra.TensorProduct.map f g).comp Algebra.TensorProduct.i …
                   -/
  AlgHom.ext <| by simp
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem map_restrictScalars_comp_includeRight (f : A →ₐ[S] B) (g : C →ₐ[R] D) :
    ((map f g).restrictScalars R).comp includeRight = includeRight.comp g :=
                   /-
                     R : Type uR
                     S : Type uS
                     A : Type uA
                     B : Type uB
                     C : Type uC
                     D : Type uD
                     inst✝¹⁴ : CommSemiring R
                     inst✝¹³ : CommSemiring S
                     inst✝¹² : Algebra R S
                     inst✝¹¹ : Semiring A
                     inst✝¹⁰ : Algebra R A
                     inst✝⁹ : Algebra S A
                     inst✝⁸ : IsScalarTower R S A
                     inst✝⁷ : Semiring B
                     inst✝⁶ : Algebra R B
                     inst✝⁵ : Algebra S B
                     inst✝⁴ : IsScalarTower R S B
                     inst✝³ : Semiring C
                     inst✝² : Algebra R C
                     inst✝¹ : Semiring D
                     inst✝ : Algebra R D
                     f : AlgHom S A B
                     g : AlgHom R C D
                     ⊢ ∀ (x : C), Eq (((AlgHom.restrictScalars R (Algebra.TensorProduct.map f g)).c …
                   -/
  AlgHom.ext <| by simp
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem map_comp_includeRight (f : A →ₐ[R] B) (g : C →ₐ[R] D) :
    (map f g).comp includeRight = includeRight.comp g :=
  map_restrictScalars_comp_includeRight f g


theorem map_range (f : A →ₐ[R] B) (g : C →ₐ[R] D) :
    (map f g).range = (includeLeft.comp f).range ⊔ (includeRight.comp g).range := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    C : Type uC
    D : Type uD
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Semiring B
    inst✝⁴ : Algebra R B
    inst✝³ : Semiring C
    inst✝² : Algebra R C
    inst✝¹ : Semiring D
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    ⊢ Eq (Algebra.TensorProduct.map f g).range (Max.max (Algebra.TensorProduct.inc …
  -/
  apply le_antisymm
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra R C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : AlgHom R A B
      g : AlgHom R C D
      ⊢ LE.le (Algebra.TensorProduct.map f g).range (Max.max (Algebra.TensorProduct. …
    -/
  · rw [← map_top, ← adjoin_tmul_eq_top, ← adjoin_image, adjoin_le_iff]
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra R C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : AlgHom R A B
      g : AlgHom R C D
      ⊢ HasSubset.Subset (Set.image (⇑(Algebra.TensorProduct.map f g)) (setOf fun t  …
    -/
    rintro _ ⟨_, ⟨a, b, rfl⟩, rfl⟩
    /-
      case a.intro.intro.intro.intro
      R : Type uR
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra R C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : AlgHom R A B
      g : AlgHom R C D
      a : A
      b : C
      ⊢ Membership.mem (↑(Max.max (Algebra.TensorProduct.includeLeft.comp f).range ( …
    -/
    rw [map_tmul, ← mul_one (f a), ← one_mul (g b), ← tmul_mul_tmul]
    /-
      case a.intro.intro.intro.intro
      R : Type uR
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra R C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : AlgHom R A B
      g : AlgHom R C D
      a : A
      b : C
      ⊢ Membership.mem (↑(Max.max (Algebra.TensorProduct.includeLeft.comp f).range ( …
    -/
    exact mul_mem_sup (AlgHom.mem_range_self _ a) (AlgHom.mem_range_self _ b)
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra R C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : AlgHom R A B
      g : AlgHom R C D
      ⊢ LE.le (Max.max (Algebra.TensorProduct.includeLeft.comp f).range (Algebra.Ten …
    -/
  · rw [← map_comp_includeLeft f g, ← map_comp_includeRight f g]
    /-
      case a
      R : Type uR
      A : Type uA
      B : Type uB
      C : Type uC
      D : Type uD
      inst✝⁸ : CommSemiring R
      inst✝⁷ : Semiring A
      inst✝⁶ : Algebra R A
      inst✝⁵ : Semiring B
      inst✝⁴ : Algebra R B
      inst✝³ : Semiring C
      inst✝² : Algebra R C
      inst✝¹ : Semiring D
      inst✝ : Algebra R D
      f : AlgHom R A B
      g : AlgHom R C D
      ⊢ LE.le (Max.max ((Algebra.TensorProduct.map f g).comp Algebra.TensorProduct.i …
    -/
    exact sup_le (AlgHom.range_comp_le_range _ _) (AlgHom.range_comp_le_range _ _)
    /-
      🎉 no goals
    -/


/-- Construct an isomorphism between tensor products of an S-algebra with an R-algebra
from S- and R- isomorphisms between the tensor factors.
-/
def congr (f : A ≃ₐ[S] B) (g : C ≃ₐ[R] D) : A ⊗[R] C ≃ₐ[S] B ⊗[R] D :=
  AlgEquiv.ofAlgHom (map f g) (map f.symm g.symm)
                        /-
                          R : Type uR
                          S : Type uS
                          A : Type uA
                          B : Type uB
                          C : Type uC
                          D : Type uD
                          E : Type uE
                          F : Type uF
                          inst✝¹⁸ : CommSemiring R
                          inst✝¹⁷ : CommSemiring S
                          inst✝¹⁶ : Algebra R S
                          inst✝¹⁵ : Semiring A
                          inst✝¹⁴ : Algebra R A
                          inst✝¹³ : Algebra S A
                          inst✝¹² : IsScalarTower R S A
                          inst✝¹¹ : Semiring B
                          inst✝¹⁰ : Algebra R B
                          inst✝⁹ : Algebra S B
                          inst✝⁸ : IsScalarTower R S B
                          inst✝⁷ : Semiring C
                          inst✝⁶ : Algebra R C
                          inst✝⁵ : Semiring D
                          inst✝⁴ : Algebra R D
                          inst✝³ : Semiring E
                          inst✝² : Algebra R E
                          inst✝¹ : Semiring F
                          inst✝ : Algebra R F
                          f : AlgEquiv S A B
                          g : AlgEquiv R C D
                          b : B
                          d : D
                          ⊢ Eq (((Algebra.TensorProduct.map ↑f ↑g).comp (Algebra.TensorProduct.map ↑f.sy …
                        -/
                        /-
                          🎉 no goals
                        -/
    (ext' fun b d => by simp) (ext' fun a c => by simp)
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp] theorem congr_toLinearEquiv (f : A ≃ₐ[S] B) (g : C ≃ₐ[R] D) :
    (Algebra.TensorProduct.congr f g).toLinearEquiv =
      TensorProduct.AlgebraTensorModule.congr f.toLinearEquiv g.toLinearEquiv := rfl


@[simp]
theorem congr_apply (f : A ≃ₐ[S] B) (g : C ≃ₐ[R] D) (x) :
    congr f g x = (map (f : A →ₐ[S] B) (g : C →ₐ[R] D)) x :=
  rfl


@[simp]
theorem congr_symm_apply (f : A ≃ₐ[S] B) (g : C ≃ₐ[R] D) (x) :
    (congr f g).symm x = (map (f.symm : B →ₐ[S] A) (g.symm : D →ₐ[R] C)) x :=
  rfl


@[simp]
theorem congr_refl : congr (.refl : A ≃ₐ[S] A) (.refl : C ≃ₐ[R] C) = .refl :=
  AlgEquiv.coe_algHom_injective <| map_id


theorem congr_trans [Algebra S C] [IsScalarTower R S C]
    (f₁ : A ≃ₐ[S] B) (f₂ : B ≃ₐ[S] C) (g₁ : D ≃ₐ[R] E) (g₂ : E ≃ₐ[R] F) :
    congr (f₁.trans f₂) (g₁.trans g₂) = (congr f₁ g₁).trans (congr f₂ g₂) :=
  AlgEquiv.coe_algHom_injective <| map_comp f₂.toAlgHom f₁.toAlgHom g₂.toAlgHom g₁.toAlgHom


theorem congr_symm (f : A ≃ₐ[S] B) (g : C ≃ₐ[R] D) : congr f.symm g.symm = (congr f g).symm := rfl


variable (R A B C) in
/-- Tensor product of algebras analogue of `mul_left_comm`.

This is the algebra version of `TensorProduct.leftComm`. -/
def leftComm : A ⊗[R] B ⊗[R] C ≃ₐ[R] B ⊗[R] A ⊗[R] C :=
  let e₁ := (Algebra.TensorProduct.assoc R A B C).symm
  let e₂ := congr (Algebra.TensorProduct.comm R A B) (1 : C ≃ₐ[R] C)
  let e₃ := Algebra.TensorProduct.assoc R B A C
  e₁.trans (e₂.trans e₃)


@[simp]
theorem leftComm_tmul (m : A) (n : B) (p : C) :
    leftComm R A B C (m ⊗ₜ (n ⊗ₜ p)) = n ⊗ₜ (m ⊗ₜ p) :=
  rfl


@[simp]
theorem leftComm_symm_tmul (m : A) (n : B) (p : C) :
    (leftComm R A B C).symm (n ⊗ₜ (m ⊗ₜ p)) = m ⊗ₜ (n ⊗ₜ p) :=
  rfl


@[simp]
theorem leftComm_toLinearEquiv :
    (leftComm R A B C : _ ≃ₗ[R] _) = _root_.TensorProduct.leftComm R A B C := rfl


variable (R A B C D) in
/-- Tensor product of algebras analogue of `mul_mul_mul_comm`.

This is the algebra version of `TensorProduct.tensorTensorTensorComm`. -/
def tensorTensorTensorComm : (A ⊗[R] B) ⊗[R] C ⊗[R] D ≃ₐ[R] (A ⊗[R] C) ⊗[R] B ⊗[R] D :=
  let e₁ := Algebra.TensorProduct.assoc R A B (C ⊗[R] D)
  let e₂ := congr (1 : A ≃ₐ[R] A) (leftComm R B C D)
  let e₃ := (Algebra.TensorProduct.assoc R A C (B ⊗[R] D)).symm
  e₁.trans (e₂.trans e₃)


@[simp]
theorem tensorTensorTensorComm_tmul (m : A) (n : B) (p : C) (q : D) :
    tensorTensorTensorComm R A B C D (m ⊗ₜ n ⊗ₜ (p ⊗ₜ q)) = m ⊗ₜ p ⊗ₜ (n ⊗ₜ q) :=
  rfl


@[simp]
theorem tensorTensorTensorComm_symm :
    (tensorTensorTensorComm R A B C D).symm = tensorTensorTensorComm R A C B D := by
  /-
    R : Type uR
    A : Type uA
    B : Type uB
    C : Type uC
    D : Type uD
    inst✝⁸ : CommSemiring R
    inst✝⁷ : Semiring A
    inst✝⁶ : Algebra R A
    inst✝⁵ : Semiring B
    inst✝⁴ : Algebra R B
    inst✝³ : Semiring C
    inst✝² : Algebra R C
    inst✝¹ : Semiring D
    inst✝ : Algebra R D
    ⊢ Eq (Algebra.TensorProduct.tensorTensorTensorComm R A B C D).symm (Algebra.Te …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp]
theorem tensorTensorTensorComm_toLinearEquiv :
    (tensorTensorTensorComm R A B C D : _ ≃ₗ[R] _) =
      _root_.TensorProduct.tensorTensorTensorComm R A B C D := rfl


/-- If `A`, `B`, `C` are `R`-algebras, `A` and `C` are also `S`-algebras (forming a tower as
`·/S/R`), then the product map of `f : A →ₐ[S] C` and `g : B →ₐ[R] C` is an `S`-algebra
homomorphism.

This is just a special case of `Algebra.TensorProduct.lift` for when `C` is commutative. -/
abbrev productLeftAlgHom (f : A →ₐ[S] C) (g : B →ₐ[R] C) : A ⊗[R] B →ₐ[S] C :=
  lift f g (fun _ _ => Commute.all _ _)


/-- `LinearMap.mul'` as an `AlgHom` over the algebra. -/
def lmul'' : S ⊗[R] S →ₐ[S] S :=
  algHomOfLinearMapTensorProduct
    { __ := LinearMap.mul' R S
                                                /-
                                                  R : Type uR
                                                  S : Type uS
                                                  A : Type uA
                                                  B : Type uB
                                                  C : Type uC
                                                  D : Type uD
                                                  E : Type uE
                                                  F : Type uF
                                                  inst✝⁶ : CommSemiring R
                                                  inst✝⁵ : Semiring A
                                                  inst✝⁴ : Semiring B
                                                  inst✝³ : CommSemiring S
                                                  inst✝² : Algebra R A
                                                  inst✝¹ : Algebra R B
                                                  inst✝ : Algebra R S
                                                  f : AlgHom R A S
                                                  g : AlgHom R B S
                                                  s : S
                                                  x : TensorProduct R S S
                                                  ⊢ Eq (__spread✝⁻⁰.toFun (HSMul.hSMul s 0)) (HSMul.hSMul ((RingHom.id S) s) (__ …
                                                -/
      map_smul' := fun s x ↦ x.induction_on (by simp)
                                                /-
                                                  🎉 no goals
                                                -/
                      /-
                        R : Type uR
                        S : Type uS
                        A : Type uA
                        B : Type uB
                        C : Type uC
                        D : Type uD
                        E : Type uE
                        F : Type uF
                        inst✝⁶ : CommSemiring R
                        inst✝⁵ : Semiring A
                        inst✝⁴ : Semiring B
                        inst✝³ : CommSemiring S
                        inst✝² : Algebra R A
                        inst✝¹ : Algebra R B
                        inst✝ : Algebra R S
                        f : AlgHom R A S
                        g : AlgHom R B S
                        s : S
                        x : TensorProduct R S S
                        x✝¹ x✝ : S
                        ⊢ Eq (__spread✝⁻⁰.toFun (HSMul.hSMul s (TensorProduct.tmul R x✝¹ x✝))) (HSMul. …
                      -/
        (fun _ _ ↦ by simp [TensorProduct.smul_tmul', mul_assoc])
                      /-
                        🎉 no goals
                      -/
                           /-
                             R : Type uR
                             S : Type uS
                             A : Type uA
                             B : Type uB
                             C : Type uC
                             D : Type uD
                             E : Type uE
                             F : Type uF
                             inst✝⁶ : CommSemiring R
                             inst✝⁵ : Semiring A
                             inst✝⁴ : Semiring B
                             inst✝³ : CommSemiring S
                             inst✝² : Algebra R A
                             inst✝¹ : Algebra R B
                             inst✝ : Algebra R S
                             f : AlgHom R A S
                             g : AlgHom R B S
                             s : S
                             x✝ x y : TensorProduct R S S
                             hx : Eq (__spread✝⁻⁰.toFun (HSMul.hSMul s x)) (HSMul.hSMul ((RingHom.id S) s)  …
                             hy : Eq (__spread✝⁻⁰.toFun (HSMul.hSMul s y)) (HSMul.hSMul ((RingHom.id S) s)  …
                             ⊢ Eq (__spread✝⁻⁰.toFun (HSMul.hSMul s (HAdd.hAdd x y))) (HSMul.hSMul ((RingHo …
                           -/
        fun x y hx hy ↦ by simp_all [hx, hy, mul_add] }
                           /-
                             🎉 no goals
                           -/
                           /-
                             R : Type uR
                             S : Type uS
                             A : Type uA
                             B : Type uB
                             C : Type uC
                             D : Type uD
                             E : Type uE
                             F : Type uF
                             inst✝⁶ : CommSemiring R
                             inst✝⁵ : Semiring A
                             inst✝⁴ : Semiring B
                             inst✝³ : CommSemiring S
                             inst✝² : Algebra R A
                             inst✝¹ : Algebra R B
                             inst✝ : Algebra R S
                             f : AlgHom R A S
                             g : AlgHom R B S
                             a₁ a₂ b₁ b₂ : S
                             ⊢ Eq
                                 ((let __spread.0 := LinearMap.mul' R S;
                                   { toAddHom := __spread.0.toAddHom, map_smul' := ⋯ })
                                   (TensorProduct.tmul R (HMul.hMul a₁ a₂) (HMul.hMul b₁ b₂)))
                                 (HMul.hMul
                                   ((let __spread.0 := LinearMap.mul' R S;
                                     { toAddHom := __spread.0.toAddHom, map_smul' := ⋯ })
                                     (TensorProduct.tmul R a₁ b₁))
                                   ((let __spread.0 := LinearMap.mul' R S;
                                     { toAddHom := __spread.0.toAddHom, map_smul' := ⋯ })
                                     (TensorProduct.tmul R a₂ b₂)))
                           -/
                           /-
                             🎉 no goals
                           -/
    (fun a₁ a₂ b₁ b₂ => by simp [mul_mul_mul_comm]) <| by simp
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem lmul''_eq_lid_comp_mapOfCompatibleSMul :
    lmul'' R = (TensorProduct.lid S S).toAlgHom.comp (mapOfCompatibleSMul' _ _ _ _) := by
  /-
    R : Type uR
    S : Type uS
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    ⊢ Eq (Algebra.TensorProduct.lmul'' R) ((↑(Algebra.TensorProduct.lid S S)).comp …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- `LinearMap.mul'` as an `AlgHom` over the base ring. -/
def lmul' : S ⊗[R] S →ₐ[R] S := (lmul'' R).restrictScalars R


theorem lmul'_toLinearMap : (lmul' R : _ →ₐ[R] S).toLinearMap = LinearMap.mul' R S :=
  rfl


@[simp]
theorem lmul'_apply_tmul (a b : S) : lmul' (S := S) R (a ⊗ₜ[R] b) = a * b :=
  rfl


@[simp]
theorem lmul'_comp_includeLeft : (lmul' R : _ →ₐ[R] S).comp includeLeft = AlgHom.id R S :=
  AlgHom.ext <| mul_one


@[simp]
theorem lmul'_comp_includeRight : (lmul' R : _ →ₐ[R] S).comp includeRight = AlgHom.id R S :=
  AlgHom.ext <| one_mul


variable (R S) in
/-- If multiplication by elements of S can switch between the two factors of `S ⊗[R] S`,
then `lmul''` is an isomorphism. -/
def lmulEquiv [CompatibleSMul R S S S] : S ⊗[R] S ≃ₐ[S] S :=
  .ofAlgHom (lmul'' R) includeLeft lmul'_comp_includeLeft <| AlgHom.ext fun x ↦ x.induction_on
        /-
          R : Type uR
          S : Type uS
          A : Type uA
          B : Type uB
          C : Type uC
          D : Type uD
          E : Type uE
          F : Type uF
          inst✝⁷ : CommSemiring R
          inst✝⁶ : Semiring A
          inst✝⁵ : Semiring B
          inst✝⁴ : CommSemiring S
          inst✝³ : Algebra R A
          inst✝² : Algebra R B
          inst✝¹ : Algebra R S
          f : AlgHom R A S
          g : AlgHom R B S
          inst✝ : TensorProduct.CompatibleSMul R S S S
          x : TensorProduct R S S
          ⊢ Eq ((Algebra.TensorProduct.includeLeft.comp (Algebra.TensorProduct.lmul'' R) …
        -/
    (by simp) (fun x y ↦ show (x * y) ⊗ₜ[R] 1 = x ⊗ₜ[R] y by
        /-
          🎉 no goals
        -/
      /-
        R : Type uR
        S : Type uS
        A : Type uA
        B : Type uB
        C : Type uC
        D : Type uD
        E : Type uE
        F : Type uF
        inst✝⁷ : CommSemiring R
        inst✝⁶ : Semiring A
        inst✝⁵ : Semiring B
        inst✝⁴ : CommSemiring S
        inst✝³ : Algebra R A
        inst✝² : Algebra R B
        inst✝¹ : Algebra R S
        f : AlgHom R A S
        g : AlgHom R B S
        inst✝ : TensorProduct.CompatibleSMul R S S S
        x✝ : TensorProduct R S S
        x y : S
        ⊢ Eq (TensorProduct.tmul R (HMul.hMul x y) 1) (TensorProduct.tmul R x y)
      -/
      rw [mul_comm, ← smul_eq_mul, smul_tmul, smul_eq_mul, mul_one])
      /-
        🎉 no goals
      -/
                       /-
                         R : Type uR
                         S : Type uS
                         A : Type uA
                         B : Type uB
                         C : Type uC
                         D : Type uD
                         E : Type uE
                         F : Type uF
                         inst✝⁷ : CommSemiring R
                         inst✝⁶ : Semiring A
                         inst✝⁵ : Semiring B
                         inst✝⁴ : CommSemiring S
                         inst✝³ : Algebra R A
                         inst✝² : Algebra R B
                         inst✝¹ : Algebra R S
                         f : AlgHom R A S
                         g : AlgHom R B S
                         inst✝ : TensorProduct.CompatibleSMul R S S S
                         x x✝¹ x✝ : TensorProduct R S S
                         hx : Eq ((Algebra.TensorProduct.includeLeft.comp (Algebra.TensorProduct.lmul'' …
                         hy : Eq ((Algebra.TensorProduct.includeLeft.comp (Algebra.TensorProduct.lmul'' …
                         ⊢ Eq ((Algebra.TensorProduct.includeLeft.comp (Algebra.TensorProduct.lmul'' R) …
                       -/
    fun _ _ hx hy ↦ by simp_all [hx, hy, add_tmul]
                       /-
                         🎉 no goals
                       -/


theorem lmulEquiv_eq_lidOfCompatibleSMul [CompatibleSMul R S S S] :
    lmulEquiv R S = lidOfCompatibleSMul R S S :=
                                      /-
                                        R : Type uR
                                        S : Type uS
                                        inst✝³ : CommSemiring R
                                        inst✝² : CommSemiring S
                                        inst✝¹ : Algebra R S
                                        inst✝ : TensorProduct.CompatibleSMul R S S S
                                        ⊢ Eq ↑(Algebra.TensorProduct.lmulEquiv R S) ↑(Algebra.TensorProduct.lidOfCompa …
                                      -/
  AlgEquiv.coe_algHom_injective <| by ext; rfl
                                           /-
                                             🎉 no goals
                                           -/


/-- If `S` is commutative, for a pair of morphisms `f : A →ₐ[R] S`, `g : B →ₐ[R] S`,
We obtain a map `A ⊗[R] B →ₐ[R] S` that commutes with `f`, `g` via `a ⊗ b ↦ f(a) * g(b)`.

This is a special case of `Algebra.TensorProduct.productLeftAlgHom` for when the two base rings are
the same.
-/
def productMap : A ⊗[R] B →ₐ[R] S := productLeftAlgHom f g


theorem productMap_eq_comp_map : productMap f g = (lmul' R).comp (TensorProduct.map f g) := by
  /-
    R : Type uR
    S : Type uS
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : CommSemiring S
    inst✝² : Algebra R A
    inst✝¹ : Algebra R B
    inst✝ : Algebra R S
    f : AlgHom R A S
    g : AlgHom R B S
    ⊢ Eq (Algebra.TensorProduct.productMap f g) ((Algebra.TensorProduct.lmul' R).c …
  -/
          /-
            🎉 no goals
          -/
  ext <;> rfl
          /-
            🎉 no goals
          -/


@[simp]
theorem productMap_apply_tmul (a : A) (b : B) : productMap f g (a ⊗ₜ b) = f a * g b := rfl


theorem productMap_left_apply (a : A) : productMap f g (a ⊗ₜ 1) = f a := by
  /-
    R : Type uR
    S : Type uS
    A : Type uA
    B : Type uB
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    inst✝⁴ : Semiring B
    inst✝³ : CommSemiring S
    inst✝² : Algebra R A
    inst✝¹ : Algebra R B
    inst✝ : Algebra R S
    f : AlgHom R A S
    g : AlgHom R B S
    a : A
    ⊢ Eq ((Algebra.TensorProduct.productMap f g) (TensorProduct.tmul R a 1)) (f a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem productMap_left : (productMap f g).comp includeLeft = f :=
  lift_comp_includeLeft _ _ (fun _ _ => Commute.all _ _)


theorem productMap_right_apply (b : B) :
                                        /-
                                          R : Type uR
                                          S : Type uS
                                          A : Type uA
                                          B : Type uB
                                          inst✝⁶ : CommSemiring R
                                          inst✝⁵ : Semiring A
                                          inst✝⁴ : Semiring B
                                          inst✝³ : CommSemiring S
                                          inst✝² : Algebra R A
                                          inst✝¹ : Algebra R B
                                          inst✝ : Algebra R S
                                          f : AlgHom R A S
                                          g : AlgHom R B S
                                          b : B
                                          ⊢ Eq ((Algebra.TensorProduct.productMap f g) (TensorProduct.tmul R 1 b)) (g b)
                                        -/
    productMap f g (1 ⊗ₜ b) = g b := by simp
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem productMap_right : (productMap f g).comp includeRight = g :=
  lift_comp_includeRight _ _ (fun _ _ => Commute.all _ _)


theorem productMap_range : (productMap f g).range = f.range ⊔ g.range := by
  rw [productMap_eq_comp_map, AlgHom.range_comp, map_range, map_sup, ← AlgHom.range_comp,
    ← AlgHom.range_comp,
    ← AlgHom.comp_assoc, ← AlgHom.comp_assoc, lmul'_comp_includeLeft, lmul'_comp_includeRight,
    AlgHom.id_comp, AlgHom.id_comp]


lemma Algebra.baseChange_lmul {R B : Type*} [CommRing R] [CommRing B] [Algebra R B]
    {A : Type*} [CommRing A] [Algebra R A] (f : B) :
    (Algebra.lmul R B f).baseChange A = Algebra.lmul A (A ⊗[R] B) (1 ⊗ₜ f) := by
  /-
    R : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f : B
    ⊢ Eq (LinearMap.baseChange A ((Algebra.lmul R B) f)) ((Algebra.lmul A (TensorP …
  -/
  ext i
  /-
    case a.h.h
    R : Type u_1
    B : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing B
    inst✝² : Algebra R B
    A : Type u_3
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    f i : B
    ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (LinearMap.baseChange A ((Alge …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The natural linear map $A ⊗ \text{Hom}_R(M, N) → \text{Hom}_A (M_A, N_A)$,
where $M_A$ and $N_A$ are the respective modules over $A$ obtained by extension of scalars.

See `LinearMap.tensorProductEnd` for this map specialized to endomorphisms,
and bundled as `A`-algebra homomorphism. -/
@[simps!]
noncomputable
def tensorProduct : A ⊗[R] (M →ₗ[R] N) →ₗ[A] (A ⊗[R] M) →ₗ[A] (A ⊗[R] N) :=
  TensorProduct.AlgebraTensorModule.lift <|
  { toFun := fun a ↦ a • baseChangeHom R A M N
                   /-
                     R : Type u_1
                     A : Type u_2
                     M : Type u_3
                     N : Type u_4
                     inst✝⁶ : CommRing R
                     inst✝⁵ : CommRing A
                     inst✝⁴ : Algebra R A
                     inst✝³ : AddCommGroup M
                     inst✝² : Module R M
                     inst✝¹ : AddCommGroup N
                     inst✝ : Module R N
                     ⊢ ∀ (x y : A), Eq ((fun a => HSMul.hSMul a (LinearMap.baseChangeHom R A M N))  …
                   -/
    map_add' := by simp only [add_smul, forall_true_iff]
                   /-
                     🎉 no goals
                   -/
                    /-
                      R : Type u_1
                      A : Type u_2
                      M : Type u_3
                      N : Type u_4
                      inst✝⁶ : CommRing R
                      inst✝⁵ : CommRing A
                      inst✝⁴ : Algebra R A
                      inst✝³ : AddCommGroup M
                      inst✝² : Module R M
                      inst✝¹ : AddCommGroup N
                      inst✝ : Module R N
                      ⊢ ∀ (m x : A), Eq ({ toFun := fun a => HSMul.hSMul a (LinearMap.baseChangeHom  …
                    -/
    map_smul' := by simp only [smul_assoc, RingHom.id_apply, forall_true_iff] }
                    /-
                      🎉 no goals
                    -/


/-- The natural `A`-algebra homomorphism $A ⊗ (\text{End}_R M) → \text{End}_A (A ⊗ M)$,
where `M` is an `R`-module, and `A` an `R`-algebra. -/
@[simps!]
noncomputable
def tensorProductEnd : A ⊗[R] (End R M) →ₐ[A] End A (A ⊗[R] M) :=
  Algebra.TensorProduct.algHomOfLinearMapTensorProduct
    (LinearMap.tensorProduct R A M M)
    (fun a b f g ↦ by
      /-
        R : Type u_1
        A : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        a b : A
        f g : Module.End R M
        ⊢ Eq ((LinearMap.tensorProduct R A M M) (TensorProduct.tmul R (HMul.hMul a b)  …
      -/
      apply LinearMap.ext
      /-
        case h
        R : Type u_1
        A : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        a b : A
        f g : Module.End R M
        ⊢ ∀ (x : TensorProduct R A M), Eq (((LinearMap.tensorProduct R A M M) (TensorP …
      -/
      intro x
      simp only [tensorProduct, mul_comm a b, mul_eq_comp,
        TensorProduct.AlgebraTensorModule.lift_apply, TensorProduct.lift.tmul, coe_restrictScalars,
        coe_mk, AddHom.coe_mk, mul_smul, smul_apply, baseChangeHom_apply, baseChange_comp,
        comp_apply, Algebra.mul_smul_comm, Algebra.smul_mul_assoc])
    (by
      /-
        R : Type u_1
        A : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        ⊢ Eq ((LinearMap.tensorProduct R A M M) (TensorProduct.tmul R 1 1)) 1
      -/
      apply LinearMap.ext
      /-
        case h
        R : Type u_1
        A : Type u_2
        M : Type u_3
        N : Type u_4
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : AddCommGroup N
        inst✝ : Module R N
        ⊢ ∀ (x : TensorProduct R A M), Eq (((LinearMap.tensorProduct R A M M) (TensorP …
      -/
      intro x
      simp only [tensorProduct, TensorProduct.AlgebraTensorModule.lift_apply,
        TensorProduct.lift.tmul, coe_restrictScalars, coe_mk, AddHom.coe_mk, one_smul,
        baseChangeHom_apply, baseChange_eq_ltensor, one_apply, one_eq_id, lTensor_id,
        LinearMap.id_apply])


/-- The algebra homomorphism from `End M ⊗ End N` to `End (M ⊗ N)` sending `f ⊗ₜ g` to
the `TensorProduct.map f g`, the tensor product of the two maps.

This is an `AlgHom` version of `TensorProduct.AlgebraTensorModule.homTensorHomMap`. Like that
definition, this is generalized across many different rings; namely a tower of algebras `A/S/R`. -/
def endTensorEndAlgHom : End A M ⊗[R] End R N →ₐ[S] End A (M ⊗[R] N) :=
  Algebra.TensorProduct.algHomOfLinearMapTensorProduct
    (AlgebraTensorModule.homTensorHomMap R A S M N M N)
    (fun _f₁ _f₂ _g₁ _g₂ => AlgebraTensorModule.ext fun _m _n => rfl)
    (AlgebraTensorModule.ext fun _m _n => rfl)


theorem endTensorEndAlgHom_apply (f : End A M) (g : End R N) :
    endTensorEndAlgHom (R := R) (S := S) (A := A) (M := M) (N := N) (f ⊗ₜ[R] g)
      = AlgebraTensorModule.map f g :=
  rfl


/-- An auxiliary definition, used for constructing the `Module (A ⊗[R] B) M` in
`TensorProduct.Algebra.module` below. -/
def moduleAux : A ⊗[R] B →ₗ[R] M →ₗ[R] M :=
  TensorProduct.lift
    { toFun := fun a => a • (Algebra.lsmul R R M : B →ₐ[R] Module.End R M).toLinearMap
      map_add' := fun r t => by
        /-
          R : Type u_1
          A : Type u_2
          B : Type u_3
          M : Type u_4
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : AddCommMonoid M
          inst✝⁸ : Module R M
          inst✝⁷ : Semiring A
          inst✝⁶ : Semiring B
          inst✝⁵ : Module A M
          inst✝⁴ : Module B M
          inst✝³ : Algebra R A
          inst✝² : Algebra R B
          inst✝¹ : IsScalarTower R A M
          inst✝ : IsScalarTower R B M
          r t : A
          ⊢ Eq ((fun a => HSMul.hSMul a (Algebra.lsmul R R M).toLinearMap) (HAdd.hAdd r  …
        -/
        ext
        /-
          case h.h
          R : Type u_1
          A : Type u_2
          B : Type u_3
          M : Type u_4
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : AddCommMonoid M
          inst✝⁸ : Module R M
          inst✝⁷ : Semiring A
          inst✝⁶ : Semiring B
          inst✝⁵ : Module A M
          inst✝⁴ : Module B M
          inst✝³ : Algebra R A
          inst✝² : Algebra R B
          inst✝¹ : IsScalarTower R A M
          inst✝ : IsScalarTower R B M
          r t : A
          x✝¹ : B
          x✝ : M
          ⊢ Eq ((((fun a => HSMul.hSMul a (Algebra.lsmul R R M).toLinearMap) (HAdd.hAdd  …
        -/
        simp only [add_smul, LinearMap.add_apply]
        /-
          🎉 no goals
        -/
      map_smul' := fun n r => by
        /-
          R : Type u_1
          A : Type u_2
          B : Type u_3
          M : Type u_4
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : AddCommMonoid M
          inst✝⁸ : Module R M
          inst✝⁷ : Semiring A
          inst✝⁶ : Semiring B
          inst✝⁵ : Module A M
          inst✝⁴ : Module B M
          inst✝³ : Algebra R A
          inst✝² : Algebra R B
          inst✝¹ : IsScalarTower R A M
          inst✝ : IsScalarTower R B M
          n : R
          r : A
          ⊢ Eq ({ toFun := fun a => HSMul.hSMul a (Algebra.lsmul R R M).toLinearMap, map …
        -/
        ext
        /-
          case h.h
          R : Type u_1
          A : Type u_2
          B : Type u_3
          M : Type u_4
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : AddCommMonoid M
          inst✝⁸ : Module R M
          inst✝⁷ : Semiring A
          inst✝⁶ : Semiring B
          inst✝⁵ : Module A M
          inst✝⁴ : Module B M
          inst✝³ : Algebra R A
          inst✝² : Algebra R B
          inst✝¹ : IsScalarTower R A M
          inst✝ : IsScalarTower R B M
          n : R
          r : A
          x✝¹ : B
          x✝ : M
          ⊢ Eq ((({ toFun := fun a => HSMul.hSMul a (Algebra.lsmul R R M).toLinearMap, m …
        -/
        simp only [RingHom.id_apply, LinearMap.smul_apply, smul_assoc] }
        /-
          🎉 no goals
        -/


theorem moduleAux_apply (a : A) (b : B) (m : M) : moduleAux (a ⊗ₜ[R] b) m = a • b • m :=
  rfl


/-- If `M` is a representation of two different `R`-algebras `A` and `B` whose actions commute,
then it is a representation the `R`-algebra `A ⊗[R] B`.

An important example arises from a semiring `S`; allowing `S` to act on itself via left and right
multiplication, the roles of `R`, `A`, `B`, `M` are played by `ℕ`, `S`, `Sᵐᵒᵖ`, `S`. This example
is important because a submodule of `S` as a `Module` over `S ⊗[ℕ] Sᵐᵒᵖ` is a two-sided ideal.

NB: This is not an instance because in the case `B = A` and `M = A ⊗[R] A` we would have a diamond
of `smul` actions. Furthermore, this would not be a mere definitional diamond but a true
mathematical diamond in which `A ⊗[R] A` had two distinct scalar actions on itself: one from its
multiplication, and one from this would-be instance. Arguably we could live with this but in any
case the real fix is to address the ambiguity in notation, probably along the lines outlined here:
https://leanprover.zulipchat.com/#narrow/stream/144837-PR-reviews/topic/.234773.20base.20change/near/240929258
-/
protected def module : Module (A ⊗[R] B) M where
  smul x m := moduleAux x m
                    /-
                      R : Type u_1
                      A : Type u_2
                      B : Type u_3
                      M : Type u_4
                      inst✝¹¹ : CommSemiring R
                      inst✝¹⁰ : AddCommMonoid M
                      inst✝⁹ : Module R M
                      inst✝⁸ : Semiring A
                      inst✝⁷ : Semiring B
                      inst✝⁶ : Module A M
                      inst✝⁵ : Module B M
                      inst✝⁴ : Algebra R A
                      inst✝³ : Algebra R B
                      inst✝² : IsScalarTower R A M
                      inst✝¹ : IsScalarTower R B M
                      inst✝ : SMulCommClass A B M
                      m : M
                      ⊢ Eq (HSMul.hSMul 0 m) 0
                    -/
                    /-
                      R : Type u_1
                      A : Type u_2
                      B : Type u_3
                      M : Type u_4
                      inst✝¹¹ : CommSemiring R
                      inst✝¹⁰ : AddCommMonoid M
                      inst✝⁹ : Module R M
                      inst✝⁸ : Semiring A
                      inst✝⁷ : Semiring B
                      inst✝⁶ : Module A M
                      inst✝⁵ : Module B M
                      inst✝⁴ : Algebra R A
                      inst✝³ : Algebra R B
                      inst✝² : IsScalarTower R A M
                      inst✝¹ : IsScalarTower R B M
                      inst✝ : SMulCommClass A B M
                      x : TensorProduct R A B
                      ⊢ Eq (HSMul.hSMul x 0) 0
                    -/
  zero_smul m := by simp only [(· • ·), map_zero, LinearMap.zero_apply]
                    /-
                      🎉 no goals
                    -/
                         /-
                           R : Type u_1
                           A : Type u_2
                           B : Type u_3
                           M : Type u_4
                           inst✝¹¹ : CommSemiring R
                           inst✝¹⁰ : AddCommMonoid M
                           inst✝⁹ : Module R M
                           inst✝⁸ : Semiring A
                           inst✝⁷ : Semiring B
                           inst✝⁶ : Module A M
                           inst✝⁵ : Module B M
                           inst✝⁴ : Algebra R A
                           inst✝³ : Algebra R B
                           inst✝² : IsScalarTower R A M
                           inst✝¹ : IsScalarTower R B M
                           inst✝ : SMulCommClass A B M
                           x : TensorProduct R A B
                           m₁ m₂ : M
                           ⊢ Eq (HSMul.hSMul x (HAdd.hAdd m₁ m₂)) (HAdd.hAdd (HSMul.hSMul x m₁) (HSMul.hS …
                         -/
                    /-
                      🎉 no goals
                    -/
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      M : Type u_4
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : Module R M
      inst✝⁸ : Semiring A
      inst✝⁷ : Semiring B
      inst✝⁶ : Module A M
      inst✝⁵ : Module B M
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R B M
      inst✝ : SMulCommClass A B M
      m : M
      ⊢ Eq (HSMul.hSMul 1 m) m
    -/
                         /-
                           🎉 no goals
                         -/
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      M : Type u_4
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : Module R M
      inst✝⁸ : Semiring A
      inst✝⁷ : Semiring B
      inst✝⁶ : Module A M
      inst✝⁵ : Module B M
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R B M
      inst✝ : SMulCommClass A B M
      m : M
      ⊢ Eq ((TensorProduct.Algebra.moduleAux (TensorProduct.tmul R 1 1)) m) m
    -/
                       /-
                         R : Type u_1
                         A : Type u_2
                         B : Type u_3
                         M : Type u_4
                         inst✝¹¹ : CommSemiring R
                         inst✝¹⁰ : AddCommMonoid M
                         inst✝⁹ : Module R M
                         inst✝⁸ : Semiring A
                         inst✝⁷ : Semiring B
                         inst✝⁶ : Module A M
                         inst✝⁵ : Module B M
                         inst✝⁴ : Algebra R A
                         inst✝³ : Algebra R B
                         inst✝² : IsScalarTower R A M
                         inst✝¹ : IsScalarTower R B M
                         inst✝ : SMulCommClass A B M
                         x y : TensorProduct R A B
                         m : M
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd x y) m) (HAdd.hAdd (HSMul.hSMul x m) (HSMul.hSMul …
                       -/
    /-
      🎉 no goals
    -/
  smul_zero x := by simp only [(· • ·), map_zero]
    /-
      R : Type u_1
      A : Type u_2
      B : Type u_3
      M : Type u_4
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : Module R M
      inst✝⁸ : Semiring A
      inst✝⁷ : Semiring B
      inst✝⁶ : Module A M
      inst✝⁵ : Module B M
      inst✝⁴ : Algebra R A
      inst✝³ : Algebra R B
      inst✝² : IsScalarTower R A M
      inst✝¹ : IsScalarTower R B M
      inst✝ : SMulCommClass A B M
      x y : TensorProduct R A B
      m : M
      ⊢ Eq (HSMul.hSMul (HMul.hMul x y) m) (HSMul.hSMul x (HSMul.hSMul y m))
    -/
                       /-
                         🎉 no goals
                       -/
      /-
        case refine_1.refine_1
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ Eq (HSMul.hSMul (HMul.hMul 0 0) m) (HSMul.hSMul 0 (HSMul.hSMul 0 m))
      -/
  smul_add x m₁ m₂ := by simp only [(· • ·), map_add]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ ∀ (x : A) (y : B), Eq (HSMul.hSMul (HMul.hMul 0 (TensorProduct.tmul R x y))  …
      -/
  add_smul x y m := by simp only [(· • ·), map_add, LinearMap.add_apply]
      /-
        case refine_1.refine_2
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        a : A
        b : B
        ⊢ Eq (HSMul.hSMul (HMul.hMul 0 (TensorProduct.tmul R a b)) m) (HSMul.hSMul 0 ( …
      -/
  one_smul m := by
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_3
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ ∀ (x y : TensorProduct R A B), Eq (HSMul.hSMul (HMul.hMul 0 x) m) (HSMul.hSM …
      -/
    -- Porting note: was one `simp only`, not two
      /-
        case refine_1.refine_3
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        z w : TensorProduct R A B
        a✝¹ : Eq (HSMul.hSMul (HMul.hMul 0 z) m) (HSMul.hSMul 0 (HSMul.hSMul z m))
        a✝ : Eq (HSMul.hSMul (HMul.hMul 0 w) m) (HSMul.hSMul 0 (HSMul.hSMul w m))
        ⊢ Eq (HSMul.hSMul (HMul.hMul 0 (HAdd.hAdd z w)) m) (HSMul.hSMul 0 (HSMul.hSMul …
      -/
    simp only [(· • ·), Algebra.TensorProduct.one_def]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_1
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ ∀ (x : A) (y : B), Eq (HSMul.hSMul (HMul.hMul (TensorProduct.tmul R x y) 0)  …
      -/
    simp only [moduleAux_apply, one_smul]
      /-
        case refine_2.refine_1
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        a : A
        b : B
        ⊢ Eq (HSMul.hSMul (HMul.hMul (TensorProduct.tmul R a b) 0) m) (HSMul.hSMul (Te …
      -/
  mul_smul x y m := by
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ ∀ (x : A) (y : B) (x_1 : A) (y_1 : B), Eq (HSMul.hSMul (HMul.hMul (TensorPro …
      -/
    refine TensorProduct.induction_on x ?_ ?_ ?_ <;> refine TensorProduct.induction_on y ?_ ?_ ?_
    · simp only [(· • ·), mul_zero, map_zero, LinearMap.zero_apply]
      /-
        case refine_2.refine_2
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        a₁ : A
        b₁ : B
        a₂ : A
        b₂ : B
        ⊢ Eq (HSMul.hSMul (HMul.hMul (TensorProduct.tmul R a₂ b₂) (TensorProduct.tmul  …
      -/
    · intro a b
      /-
        case refine_2.refine_2
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        a₁ : A
        b₁ : B
        a₂ : A
        b₂ : B
        ⊢ Eq ((TensorProduct.Algebra.moduleAux (TensorProduct.tmul R (HMul.hMul a₂ a₁) …
      -/
      simp only [(· • ·), zero_mul, map_zero, LinearMap.zero_apply]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_3
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ ∀ (x y : TensorProduct R A B), (∀ (x_1 : A) (y : B), Eq (HSMul.hSMul (HMul.h …
      -/
    · intro z w _ _
      simp only [(· • ·), zero_mul, map_zero, LinearMap.zero_apply]
      /-
        case refine_2.refine_3
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        z w : TensorProduct R A B
        hz : ∀ (x : A) (y : B), Eq (HSMul.hSMul (HMul.hMul (TensorProduct.tmul R x y)  …
        hw : ∀ (x : A) (y : B), Eq (HSMul.hSMul (HMul.hMul (TensorProduct.tmul R x y)  …
        a : A
        b : B
        ⊢ Eq (HSMul.hSMul (HMul.hMul (TensorProduct.tmul R a b) (HAdd.hAdd z w)) m) (H …
      -/
    · intro a b
      simp only [(· • ·), mul_zero, map_zero, LinearMap.zero_apply]
    · intro a₁ b₁ a₂ b₂
      /-
        case refine_3.refine_1
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ ∀ (x y : TensorProduct R A B), Eq (HSMul.hSMul (HMul.hMul x 0) m) (HSMul.hSM …
      -/
      -- Porting note: was one `simp only`, not two
      /-
        case refine_3.refine_1
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        z w : TensorProduct R A B
        a✝¹ : Eq (HSMul.hSMul (HMul.hMul z 0) m) (HSMul.hSMul z (HSMul.hSMul 0 m))
        a✝ : Eq (HSMul.hSMul (HMul.hMul w 0) m) (HSMul.hSMul w (HSMul.hSMul 0 m))
        ⊢ Eq (HSMul.hSMul (HMul.hMul (HAdd.hAdd z w) 0) m) (HSMul.hSMul (HAdd.hAdd z w …
      -/
      simp only [(· • ·), Algebra.TensorProduct.tmul_mul_tmul]
      /-
        🎉 no goals
      -/
      /-
        case refine_3.refine_2
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ ∀ (x : A) (y : B) (x_1 y_1 : TensorProduct R A B), Eq (HSMul.hSMul (HMul.hMu …
      -/
      simp only [moduleAux_apply, mul_smul, smul_comm a₁ b₂]
      /-
        case refine_3.refine_2
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        a : A
        b : B
        z w : TensorProduct R A B
        hz : Eq (HSMul.hSMul (HMul.hMul z (TensorProduct.tmul R a b)) m) (HSMul.hSMul  …
        hw : Eq (HSMul.hSMul (HMul.hMul w (TensorProduct.tmul R a b)) m) (HSMul.hSMul  …
        ⊢ Eq (HSMul.hSMul (HMul.hMul (HAdd.hAdd z w) (TensorProduct.tmul R a b)) m) (H …
      -/
    · intro z w hz hw a b
      /-
        case refine_3.refine_2
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        a : A
        b : B
        z w : TensorProduct R A B
        hz : Eq ((TensorProduct.Algebra.moduleAux (HMul.hMul z (TensorProduct.tmul R a …
        hw : Eq ((TensorProduct.Algebra.moduleAux (HMul.hMul w (TensorProduct.tmul R a …
        ⊢ Eq ((TensorProduct.Algebra.moduleAux (HMul.hMul (HAdd.hAdd z w) (TensorProdu …
      -/
      -- Porting note: was one `simp only`, but random stuff doesn't work
      /-
        🎉 no goals
      -/
      /-
        case refine_3.refine_3
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        ⊢ ∀ (x y : TensorProduct R A B), (∀ (x_1 y : TensorProduct R A B), Eq (HSMul.h …
      -/
      simp only [(· • ·)] at hz hw ⊢
      /-
        case refine_3.refine_3
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        u v : TensorProduct R A B
        a✝¹ : ∀ (x y : TensorProduct R A B), Eq (HSMul.hSMul (HMul.hMul x u) m) (HSMul …
        a✝ : ∀ (x y : TensorProduct R A B), Eq (HSMul.hSMul (HMul.hMul x v) m) (HSMul. …
        z w : TensorProduct R A B
        hz : Eq (HSMul.hSMul (HMul.hMul z (HAdd.hAdd u v)) m) (HSMul.hSMul z (HSMul.hS …
        hw : Eq (HSMul.hSMul (HMul.hMul w (HAdd.hAdd u v)) m) (HSMul.hSMul w (HSMul.hS …
        ⊢ Eq (HSMul.hSMul (HMul.hMul (HAdd.hAdd z w) (HAdd.hAdd u v)) m) (HSMul.hSMul  …
      -/
      simp only [moduleAux_apply, mul_add, LinearMap.map_add,
      /-
        case refine_3.refine_3
        R : Type u_1
        A : Type u_2
        B : Type u_3
        M : Type u_4
        inst✝¹¹ : CommSemiring R
        inst✝¹⁰ : AddCommMonoid M
        inst✝⁹ : Module R M
        inst✝⁸ : Semiring A
        inst✝⁷ : Semiring B
        inst✝⁶ : Module A M
        inst✝⁵ : Module B M
        inst✝⁴ : Algebra R A
        inst✝³ : Algebra R B
        inst✝² : IsScalarTower R A M
        inst✝¹ : IsScalarTower R B M
        inst✝ : SMulCommClass A B M
        x y : TensorProduct R A B
        m : M
        u v : TensorProduct R A B
        a✝¹ : ∀ (x y : TensorProduct R A B), Eq (HSMul.hSMul (HMul.hMul x u) m) (HSMul …
        a✝ : ∀ (x y : TensorProduct R A B), Eq (HSMul.hSMul (HMul.hMul x v) m) (HSMul. …
        z w : TensorProduct R A B
        hz : Eq ((TensorProduct.Algebra.moduleAux (HMul.hMul z (HAdd.hAdd u v))) m) (( …
        hw : Eq ((TensorProduct.Algebra.moduleAux (HMul.hMul w (HAdd.hAdd u v))) m) (( …
        ⊢ Eq ((TensorProduct.Algebra.moduleAux (HMul.hMul (HAdd.hAdd z w) (HAdd.hAdd u …
      -/
        LinearMap.add_apply, moduleAux_apply, hz, hw, smul_add]
      /-
        🎉 no goals
      -/
    · intro z w _ _
      simp only [(· • ·), mul_zero, map_zero, LinearMap.zero_apply]
    · intro a b z w hz hw
      simp only [(· • ·)] at hz hw ⊢
      simp only [LinearMap.map_add, add_mul, LinearMap.add_apply, hz, hw]
    · intro u v _ _ z w hz hw
      simp only [(· • ·)] at hz hw ⊢
      simp only [add_mul, LinearMap.map_add, LinearMap.add_apply, hz, hw, add_add_add_comm]


theorem smul_def (a : A) (b : B) (m : M) : a ⊗ₜ[R] b • m = a • b • m :=
  rfl


