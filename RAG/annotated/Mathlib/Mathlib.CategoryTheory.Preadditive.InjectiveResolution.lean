/--
An `InjectiveResolution Z` consists of a bundled `ℕ`-indexed cochain complex of injective objects,
along with a quasi-isomorphism from the complex consisting of just `Z` supported in degree `0`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure InjectiveResolution (Z : C) where
  /-- the cochain complex involved in the resolution -/
  cocomplex : CochainComplex C ℕ
  /-- the cochain complex must be degreewise injective -/
  injective : ∀ n, Injective (cocomplex.X n) := by infer_instance
  /-- the cochain complex must have homology -/
  [hasHomology : ∀ i, cocomplex.HasHomology i]
  /-- the morphism from the single cochain complex with `Z` in degree `0` -/
  ι : (single₀ C).obj Z ⟶ cocomplex
  /-- the morphism from the single cochain complex with `Z` in degree `0` is a quasi-isomorphism -/
  quasiIso : QuasiIso ι := by infer_instance


/-- An object admits an injective resolution. -/
class HasInjectiveResolution (Z : C) : Prop where
  out : Nonempty (InjectiveResolution Z)


/-- You will rarely use this typeclass directly: it is implied by the combination
`[EnoughInjectives C]` and `[Abelian C]`. -/
class HasInjectiveResolutions : Prop where
  out : ∀ Z : C, HasInjectiveResolution Z


lemma cocomplex_exactAt_succ (n : ℕ) :
    I.cocomplex.ExactAt (n + 1) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    I : CategoryTheory.InjectiveResolution Z
    n : Nat
    ⊢ HomologicalComplex.ExactAt I.cocomplex (HAdd.hAdd n 1)
  -/
  rw [← quasiIsoAt_iff_exactAt I.ι (n + 1) (exactAt_succ_single_obj _ _)]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    I : CategoryTheory.InjectiveResolution Z
    n : Nat
    ⊢ QuasiIsoAt I.ι (HAdd.hAdd n 1)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma exact_succ (n : ℕ) :
    (ShortComplex.mk _ _ (I.cocomplex.d_comp_d n (n + 1) (n + 2))).Exact :=
                                                           /-
                                                             C : Type u
                                                             inst✝² : CategoryTheory.Category.{v, u} C
                                                             inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                             Z : C
                                                             I : CategoryTheory.InjectiveResolution Z
                                                             n : Nat
                                                             ⊢ Eq ((ComplexShape.up Nat).prev (HAdd.hAdd n 1)) n
                                                           -/
  (HomologicalComplex.exactAt_iff' _ n (n + 1) (n + 2) (by simp)
                                                           /-
                                                             🎉 no goals
                                                           -/
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          Z : C
          I : CategoryTheory.InjectiveResolution Z
          n : Nat
          ⊢ Eq ((ComplexShape.up Nat).next (HAdd.hAdd n 1)) (HAdd.hAdd n 2)
        -/
    (by simp only [CochainComplex.next]; rfl)).1 (I.cocomplex_exactAt_succ n)
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem ι_f_succ (n : ℕ) : I.ι.f (n + 1) = 0 :=
                                   /-
                                     C : Type u
                                     inst✝² : CategoryTheory.Category.{v, u} C
                                     inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                     Z : C
                                     I : CategoryTheory.InjectiveResolution Z
                                     n : Nat
                                     ⊢ Ne (HAdd.hAdd n 1) 0
                                   -/
  (isZero_single_obj_X _ _ _ _ (by simp)).eq_of_src _ _
                                   /-
                                     🎉 no goals
                                   -/


@[reassoc]
theorem ι_f_zero_comp_complex_d :
    I.ι.f 0 ≫ I.cocomplex.d 0 1 = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    I : CategoryTheory.InjectiveResolution Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (I.ι.f 0) (I.cocomplex.d 0 1)) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem complex_d_comp (n : ℕ) :
    I.cocomplex.d n (n + 1) ≫ I.cocomplex.d (n + 1) (n + 2) = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    I : CategoryTheory.InjectiveResolution Z
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (I.cocomplex.d n (HAdd.hAdd n 1)) (I. …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The (limit) kernel fork given by the composition
`Z ⟶ I.cocomplex.X 0 ⟶ I.cocomplex.X 1` when `I : InjectiveResolution Z`. -/
@[simp]
def kernelFork : KernelFork (I.cocomplex.d 0 1) :=
  KernelFork.ofι _ I.ι_f_zero_comp_complex_d


/-- `Z` is the kernel of `I.cocomplex.X 0 ⟶ I.cocomplex.X 1` when `I : InjectiveResolution Z`. -/
def isLimitKernelFork : IsLimit (I.kernelFork) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    I : CategoryTheory.InjectiveResolution Z
    ⊢ CategoryTheory.Limits.IsLimit I.kernelFork
  -/
  refine IsLimit.ofIsoLimit (I.cocomplex.cyclesIsKernel 0 1 (by simp)) (Iso.symm ?_)
  refine Fork.ext ((singleObjHomologySelfIso _ _ _).symm ≪≫
    isoOfQuasiIsoAt I.ι 0 ≪≫ I.cocomplex.isoHomologyπ₀.symm) ?_
  rw [← cancel_epi (singleObjHomologySelfIso (ComplexShape.up ℕ) _ _).hom,
    ← cancel_epi (isoHomologyπ₀ _).hom,
    ← cancel_epi (singleObjCyclesSelfIso (ComplexShape.up ℕ) _ _).inv]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    I : CategoryTheory.InjectiveResolution Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.singleObjCyclesSe …
  -/
  simp
  /-
    🎉 no goals
  -/


instance (n : ℕ) : Mono (I.ι.f n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    I : CategoryTheory.InjectiveResolution Z
    n : Nat
    ⊢ CategoryTheory.Mono (I.ι.f n)
  -/
  cases n
    /-
      case zero
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      Z : C
      I : CategoryTheory.InjectiveResolution Z
      ⊢ CategoryTheory.Mono (I.ι.f 0)
    -/
  · exact mono_of_isLimit_fork I.isLimitKernelFork
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      Z : C
      I : CategoryTheory.InjectiveResolution Z
      n✝ : Nat
      ⊢ CategoryTheory.Mono (I.ι.f (HAdd.hAdd n✝ 1))
    -/
  · rw [ι_f_succ]; infer_instance
                   /-
                     🎉 no goals
                   -/


/-- An injective object admits a trivial injective resolution: itself in degree 0. -/
@[simps]
def self [Injective Z] : InjectiveResolution Z where
  cocomplex := (CochainComplex.single₀ C).obj Z
  ι := 𝟙 ((CochainComplex.single₀ C).obj Z)
  injective n := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      Z : C
      I : CategoryTheory.InjectiveResolution Z
      inst✝ : CategoryTheory.Injective Z
      n : Nat
      ⊢ CategoryTheory.Injective (((CochainComplex.single₀ C).obj Z).X n)
    -/
    cases n
      /-
        case zero
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        Z : C
        I : CategoryTheory.InjectiveResolution Z
        inst✝ : CategoryTheory.Injective Z
        ⊢ CategoryTheory.Injective (((CochainComplex.single₀ C).obj Z).X 0)
      -/
    · simpa
      /-
        🎉 no goals
      -/
      /-
        case succ
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        Z : C
        I : CategoryTheory.InjectiveResolution Z
        inst✝ : CategoryTheory.Injective Z
        n✝ : Nat
        ⊢ CategoryTheory.Injective (((CochainComplex.single₀ C).obj Z).X (HAdd.hAdd n✝ …
      -/
    · apply IsZero.injective
      /-
        case succ.h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        Z : C
        I : CategoryTheory.InjectiveResolution Z
        inst✝ : CategoryTheory.Injective Z
        n✝ : Nat
        ⊢ CategoryTheory.Limits.IsZero (((CochainComplex.single₀ C).obj Z).X (HAdd.hAd …
      -/
      apply HomologicalComplex.isZero_single_obj_X
      /-
        case succ.h.hi
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        Z : C
        I : CategoryTheory.InjectiveResolution Z
        inst✝ : CategoryTheory.Injective Z
        n✝ : Nat
        ⊢ Ne (HAdd.hAdd n✝ 1) 0
      -/
      simp
      /-
        🎉 no goals
      -/


