/--
A `ProjectiveResolution Z` consists of a bundled `ℕ`-indexed chain complex of projective objects,
along with a quasi-isomorphism to the complex consisting of just `Z` supported in degree `0`.
-/
structure ProjectiveResolution (Z : C) where
  /-- the chain complex involved in the resolution -/
  complex : ChainComplex C ℕ
  /-- the chain complex must be degreewise projective -/
  projective : ∀ n, Projective (complex.X n) := by infer_instance
  /-- the chain complex must have homology -/
  [hasHomology : ∀ i, complex.HasHomology i]
  /-- the morphism to the single chain complex with `Z` in degree `0` -/
  π : complex ⟶ (ChainComplex.single₀ C).obj Z
  /-- the morphism to the single chain complex with `Z` in degree `0` is a quasi-isomorphism -/
  quasiIso : QuasiIso π := by infer_instance


/-- An object admits a projective resolution.
-/
class HasProjectiveResolution (Z : C) : Prop where
  out : Nonempty (ProjectiveResolution Z)


/-- You will rarely use this typeclass directly: it is implied by the combination
`[EnoughProjectives C]` and `[Abelian C]`.
By itself it's enough to set up the basic theory of derived functors.
-/
class HasProjectiveResolutions : Prop where
  out : ∀ Z : C, HasProjectiveResolution Z


lemma complex_exactAt_succ (n : ℕ) :
    P.complex.ExactAt (n + 1) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    P : CategoryTheory.ProjectiveResolution Z
    n : Nat
    ⊢ HomologicalComplex.ExactAt P.complex (HAdd.hAdd n 1)
  -/
  rw [← quasiIsoAt_iff_exactAt' P.π (n + 1) (exactAt_succ_single_obj _ _)]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    P : CategoryTheory.ProjectiveResolution Z
    n : Nat
    ⊢ QuasiIsoAt P.π (HAdd.hAdd n 1)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma exact_succ (n : ℕ) :
    (ShortComplex.mk _ _ (P.complex.d_comp_d (n + 2) (n + 1) n)).Exact :=
                                                             /-
                                                               C : Type u
                                                               inst✝² : CategoryTheory.Category.{v, u} C
                                                               inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                               inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                               Z : C
                                                               P : CategoryTheory.ProjectiveResolution Z
                                                               n : Nat
                                                               ⊢ Eq ((ComplexShape.down Nat).prev (HAdd.hAdd n 1)) (HAdd.hAdd n 2)
                                                             -/
  ((HomologicalComplex.exactAt_iff' _ (n + 2) (n + 1) n) (by simp only [prev]; rfl)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          Z : C
          P : CategoryTheory.ProjectiveResolution Z
          n : Nat
          ⊢ Eq ((ComplexShape.down Nat).next (HAdd.hAdd n 1)) n
        -/
    (by simp)).1 (P.complex_exactAt_succ n)
        /-
          🎉 no goals
        -/


@[simp]
theorem π_f_succ (n : ℕ) : P.π.f (n + 1) = 0 :=
                                   /-
                                     C : Type u
                                     inst✝² : CategoryTheory.Category.{v, u} C
                                     inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                     Z : C
                                     P : CategoryTheory.ProjectiveResolution Z
                                     n : Nat
                                     ⊢ Ne (HAdd.hAdd n 1) 0
                                   -/
  (isZero_single_obj_X _ _ _ _ (by simp)).eq_of_tgt _ _
                                   /-
                                     🎉 no goals
                                   -/


@[reassoc (attr := simp)]
theorem complex_d_comp_π_f_zero :
    P.complex.d 1 0 ≫ P.π.f 0 = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    P : CategoryTheory.ProjectiveResolution Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.complex.d 1 0) (P.π.f 0)) 0
  -/
  rw [← P.π.comm 1 0, single_obj_d, comp_zero]
  /-
    🎉 no goals
  -/


theorem complex_d_succ_comp (n : ℕ) :
    P.complex.d n (n + 1) ≫ P.complex.d (n + 1) (n + 2) = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    P : CategoryTheory.ProjectiveResolution Z
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.complex.d n (HAdd.hAdd n 1)) (P.co …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The (limit) cokernel cofork given by the composition
`P.complex.X 1 ⟶ P.complex.X 0 ⟶ Z` when `P : ProjectiveResolution Z`. -/
@[simp]
noncomputable def cokernelCofork : CokernelCofork (P.complex.d 1 0) :=
  CokernelCofork.ofπ _ P.complex_d_comp_π_f_zero


/-- `Z` is the cokernel of `P.complex.X 1 ⟶ P.complex.X 0` when `P : ProjectiveResolution Z`. -/
noncomputable def isColimitCokernelCofork : IsColimit (P.cokernelCofork) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    P : CategoryTheory.ProjectiveResolution Z
    ⊢ CategoryTheory.Limits.IsColimit P.cokernelCofork
  -/
  refine IsColimit.ofIsoColimit (P.complex.opcyclesIsCokernel 1 0 (by simp)) ?_
  refine Cofork.ext (P.complex.isoHomologyι₀.symm ≪≫ isoOfQuasiIsoAt P.π 0 ≪≫
    singleObjHomologySelfIso _ _ _) ?_
  rw [← cancel_mono (singleObjHomologySelfIso (ComplexShape.down ℕ) 0 _).inv,
    ← cancel_mono (isoHomologyι₀ _).hom]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    P : CategoryTheory.ProjectiveResolution Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  simp only [isoHomologyι₀_inv_naturality_assoc, p_opcyclesMap_assoc, single₀_obj_zero, assoc,
    Iso.hom_inv_id, comp_id, isoHomologyι_inv_hom_id, singleObjHomologySelfIso_inv_homologyι,
    singleObjOpcyclesSelfIso_hom, single₀ObjXSelf, Iso.refl_inv, id_comp]


instance (n : ℕ) : Epi (P.π.f n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    Z : C
    P : CategoryTheory.ProjectiveResolution Z
    n : Nat
    ⊢ CategoryTheory.Epi (P.π.f n)
  -/
  cases n
    /-
      case zero
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasZeroObject C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      Z : C
      P : CategoryTheory.ProjectiveResolution Z
      ⊢ CategoryTheory.Epi (P.π.f 0)
    -/
  · exact epi_of_isColimit_cofork P.isColimitCokernelCofork
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
      P : CategoryTheory.ProjectiveResolution Z
      n✝ : Nat
      ⊢ CategoryTheory.Epi (P.π.f (HAdd.hAdd n✝ 1))
    -/
  · rw [π_f_succ]; infer_instance
                   /-
                     🎉 no goals
                   -/


/-- A projective object admits a trivial projective resolution: itself in degree 0. -/
@[simps]
noncomputable def self [Projective Z] : ProjectiveResolution Z where
  complex := (ChainComplex.single₀ C).obj Z
  π := 𝟙 ((ChainComplex.single₀ C).obj Z)
  projective n := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      Z : C
      P : CategoryTheory.ProjectiveResolution Z
      inst✝ : CategoryTheory.Projective Z
      n : Nat
      ⊢ CategoryTheory.Projective (((ChainComplex.single₀ C).obj Z).X n)
    -/
    cases n
      /-
        case zero
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        Z : C
        P : CategoryTheory.ProjectiveResolution Z
        inst✝ : CategoryTheory.Projective Z
        ⊢ CategoryTheory.Projective (((ChainComplex.single₀ C).obj Z).X 0)
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
        P : CategoryTheory.ProjectiveResolution Z
        inst✝ : CategoryTheory.Projective Z
        n✝ : Nat
        ⊢ CategoryTheory.Projective (((ChainComplex.single₀ C).obj Z).X (HAdd.hAdd n✝  …
      -/
    · apply IsZero.projective
      /-
        case succ.h
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        Z : C
        P : CategoryTheory.ProjectiveResolution Z
        inst✝ : CategoryTheory.Projective Z
        n✝ : Nat
        ⊢ CategoryTheory.Limits.IsZero (((ChainComplex.single₀ C).obj Z).X (HAdd.hAdd  …
      -/
      apply HomologicalComplex.isZero_single_obj_X
      /-
        case succ.h.hi
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroObject C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        Z : C
        P : CategoryTheory.ProjectiveResolution Z
        inst✝ : CategoryTheory.Projective Z
        n✝ : Nat
        ⊢ Ne (HAdd.hAdd n✝ 1) 0
      -/
      simp
      /-
        🎉 no goals
      -/


