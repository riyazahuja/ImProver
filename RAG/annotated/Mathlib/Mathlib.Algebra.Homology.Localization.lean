lemma HomologicalComplex.homologyFunctor_inverts_quasiIso (i : ι) :
    (quasiIso C c).IsInvertedBy (homologyFunctor C c i) := fun _ _ _ hf => by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.CategoryWithHomology C
    i : ι
    x✝² x✝¹ : HomologicalComplex C c
    x✝ : Quiver.Hom x✝² x✝¹
    hf : HomologicalComplex.quasiIso C c x✝
    ⊢ CategoryTheory.IsIso ((HomologicalComplex.homologyFunctor C c i).map x✝)
  -/
  rw [mem_quasiIso_iff] at hf
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.CategoryWithHomology C
    i : ι
    x✝² x✝¹ : HomologicalComplex C c
    x✝ : Quiver.Hom x✝² x✝¹
    hf : QuasiIso x✝
    ⊢ CategoryTheory.IsIso ((HomologicalComplex.homologyFunctor C c i).map x✝)
  -/
  dsimp
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.CategoryWithHomology C
    i : ι
    x✝² x✝¹ : HomologicalComplex C c
    x✝ : Quiver.Hom x✝² x✝¹
    hf : QuasiIso x✝
    ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap x✝ i)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The category of homological complexes up to quasi-isomorphisms. -/
abbrev HomologicalComplexUpToQuasiIso := (HomologicalComplex.quasiIso C c).Localization'


variable {C c} in
/-- The localization functor `HomologicalComplex C c ⥤ HomologicalComplexUpToQuasiIso C c`. -/
abbrev HomologicalComplexUpToQuasiIso.Q :
    HomologicalComplex C c ⥤ HomologicalComplexUpToQuasiIso C c :=
  (HomologicalComplex.quasiIso C c).Q'


/-- The homology functor `HomologicalComplexUpToQuasiIso C c ⥤ C` for each `i : ι`. -/
noncomputable def homologyFunctor (i : ι) : HomologicalComplexUpToQuasiIso C c ⥤ C :=
  Localization.lift _ (HomologicalComplex.homologyFunctor_inverts_quasiIso C c i) Q


/-- The homology functor on `HomologicalComplexUpToQuasiIso C c` is induced by
the homology functor on `HomologicalComplex C c`. -/
noncomputable def homologyFunctorFactors (i : ι) :
    Q ⋙ homologyFunctor C c i ≅ HomologicalComplex.homologyFunctor C c i :=
  Localization.fac _ (HomologicalComplex.homologyFunctor_inverts_quasiIso C c i) Q


lemma isIso_Q_map_iff_mem_quasiIso {K L : HomologicalComplex C c} (f : K ⟶ L) :
    IsIso (Q.map f) ↔ HomologicalComplex.quasiIso C c f := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.CategoryWithHomology C
    inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    ⊢ Iff (CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f)) (Homolog …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      ⊢ CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f) → HomologicalC …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      h : CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f)
      ⊢ HomologicalComplex.quasiIso C c f
    -/
    rw [HomologicalComplex.mem_quasiIso_iff, quasiIso_iff]
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      h : CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f)
      ⊢ ∀ (i : ι), QuasiIsoAt f i
    -/
    intro i
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      h : CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f)
      i : ι
      ⊢ QuasiIsoAt f i
    -/
    rw [quasiIsoAt_iff_isIso_homologyMap]
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      h : CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f)
      i : ι
      ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap f i)
    -/
    refine (NatIso.isIso_map_iff (homologyFunctorFactors C c i) f).1 ?_
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      h : CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f)
      i : ι
      ⊢ CategoryTheory.IsIso ((HomologicalComplexUpToQuasiIso.Q.comp (HomologicalCom …
    -/
    dsimp
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      h : CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f)
      i : ι
      ⊢ CategoryTheory.IsIso ((HomologicalComplexUpToQuasiIso.homologyFunctor C c i) …
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      ⊢ HomologicalComplex.quasiIso C c f → CategoryTheory.IsIso (HomologicalComplex …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝¹ : CategoryTheory.CategoryWithHomology C
      inst✝ : (HomologicalComplex.quasiIso C c).HasLocalization
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      h : HomologicalComplex.quasiIso C c f
      ⊢ CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map f)
    -/
    exact Localization.inverts Q (HomologicalComplex.quasiIso C c) _ h
    /-
      🎉 no goals
    -/


lemma HomologicalComplexUpToQuasiIso.Q_inverts_homotopyEquivalences
    [(HomologicalComplex.quasiIso C c).HasLocalization] :
    (HomologicalComplex.homotopyEquivalences C c).IsInvertedBy
      HomologicalComplexUpToQuasiIso.Q :=
  MorphismProperty.IsInvertedBy.of_le _ _ _
    (Localization.inverts Q (HomologicalComplex.quasiIso C c))
    (homotopyEquivalences_le_quasiIso C c)


/-- The class of quasi-isomorphisms in the homotopy category. -/
def quasiIso : MorphismProperty (HomotopyCategory C c) :=
  fun _ _ f => ∀ (i : ι), IsIso ((homologyFunctor C c i).map f)


lemma mem_quasiIso_iff {X Y : HomotopyCategory C c} (f : X ⟶ Y) :
    quasiIso C c f ↔ ∀ (n : ι), IsIso ((homologyFunctor _ _ n).map f) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    X Y : HomotopyCategory C c
    f : Quiver.Hom X Y
    ⊢ Iff (HomotopyCategory.quasiIso C c f) (∀ (n : ι), CategoryTheory.IsIso ((Hom …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma quotient_map_mem_quasiIso_iff {K L : HomologicalComplex C c} (f : K ⟶ L) :
    quasiIso C c ((quotient C c).map f) ↔ HomologicalComplex.quasiIso C c f := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    ⊢ Iff (HomotopyCategory.quasiIso C c ((HomotopyCategory.quotient C c).map f))  …
  -/
  have eq := fun (i : ι) => NatIso.isIso_map_iff (homologyFunctorFactors C c i) f
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    eq : ∀ (i : ι), Iff (CategoryTheory.IsIso (((HomotopyCategory.quotient C c).co …
    ⊢ Iff (HomotopyCategory.quasiIso C c ((HomotopyCategory.quotient C c).map f))  …
  -/
  dsimp at eq
  simp only [HomologicalComplex.mem_quasiIso_iff, mem_quasiIso_iff, quasiIso_iff,
    quasiIsoAt_iff_isIso_homologyMap, eq]


instance respectsIso_quasiIso : (quasiIso C c).RespectsIso := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    ⊢ (HomotopyCategory.quasiIso C c).RespectsIso
  -/
  apply MorphismProperty.RespectsIso.of_respects_arrow_iso
  /-
    case hP
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    ⊢ ∀ (f g : CategoryTheory.Arrow (HomotopyCategory C c)), CategoryTheory.Iso f  …
  -/
  intro f g e hf i
  exact ((MorphismProperty.isomorphisms C).arrow_mk_iso_iff
    ((homologyFunctor C c i).mapArrow.mapIso e)).1 (hf i)


lemma homologyFunctor_inverts_quasiIso (i : ι) :
    (quasiIso C c).IsInvertedBy (homologyFunctor C c i) := fun _ _ _ hf => hf i


lemma quasiIso_eq_quasiIso_map_quotient :
    quasiIso C c = (HomologicalComplex.quasiIso C c).map (quotient C c) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    ⊢ Eq (HomotopyCategory.quasiIso C c) ((HomologicalComplex.quasiIso C c).map (H …
  -/
  ext ⟨K⟩ ⟨L⟩ f
  /-
    case h.mk.mk
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : HomologicalComplex C c
    f : Quiver.Hom { as := K } { as := L }
    ⊢ Iff (HomotopyCategory.quasiIso C c f) ((HomologicalComplex.quasiIso C c).map …
  -/
  obtain ⟨f, rfl⟩ := (HomotopyCategory.quotient C c).map_surjective f
  /-
    case h.mk.mk.intro
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.CategoryWithHomology C
    K L : HomologicalComplex C c
    f : Quiver.Hom K L
    ⊢ Iff (HomotopyCategory.quasiIso C c ((HomotopyCategory.quotient C c).map f))  …
  -/
  constructor
    /-
      case h.mk.mk.intro.mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      ⊢ HomotopyCategory.quasiIso C c ((HomotopyCategory.quotient C c).map f) → (Hom …
    -/
  · intro hf
    /-
      case h.mk.mk.intro.mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      hf : HomotopyCategory.quasiIso C c ((HomotopyCategory.quotient C c).map f)
      ⊢ (HomologicalComplex.quasiIso C c).map (HomotopyCategory.quotient C c) ((Homo …
    -/
    rw [quotient_map_mem_quasiIso_iff] at hf
    /-
      case h.mk.mk.intro.mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      hf : HomologicalComplex.quasiIso C c f
      ⊢ (HomologicalComplex.quasiIso C c).map (HomotopyCategory.quotient C c) ((Homo …
    -/
    exact MorphismProperty.map_mem_map _ _ _ hf
    /-
      🎉 no goals
    -/
    /-
      case h.mk.mk.intro.mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      ⊢ (HomologicalComplex.quasiIso C c).map (HomotopyCategory.quotient C c) ((Homo …
    -/
  · rintro ⟨K', L', g, h, ⟨e⟩⟩
    /-
      case h.mk.mk.intro.mpr.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      K' L' : HomologicalComplex C c
      g : Quiver.Hom K' L'
      h : HomologicalComplex.quasiIso C c g
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk ((HomotopyCategory.quotient C  …
      ⊢ HomotopyCategory.quasiIso C c ((HomotopyCategory.quotient C c).map f)
    -/
    rw [← quotient_map_mem_quasiIso_iff] at h
    /-
      case h.mk.mk.intro.mpr.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : HomologicalComplex C c
      f : Quiver.Hom K L
      K' L' : HomologicalComplex C c
      g : Quiver.Hom K' L'
      h : HomotopyCategory.quasiIso C c ((HomotopyCategory.quotient C c).map g)
      e : CategoryTheory.Iso (CategoryTheory.Arrow.mk ((HomotopyCategory.quotient C  …
      ⊢ HomotopyCategory.quasiIso C c ((HomotopyCategory.quotient C c).map f)
    -/
    exact ((quasiIso C c).arrow_mk_iso_iff e).1 h
    /-
      🎉 no goals
    -/


/-- The condition on a complex shape `c` saying that homotopic maps become equal in
the localized category with respect to quasi-isomorphisms. -/
class ComplexShape.QFactorsThroughHomotopy {ι : Type*} (c : ComplexShape ι)
    (C : Type*) [Category C] [Preadditive C]
    [CategoryWithHomology C] : Prop where
  areEqualizedByLocalization {K L : HomologicalComplex C c} {f g : K ⟶ L} (h : Homotopy f g) :
    AreEqualizedByLocalization (HomologicalComplex.quasiIso C c) f g


lemma Q_map_eq_of_homotopy {K L : HomologicalComplex C c} {f g : K ⟶ L} (h : Homotopy f g) :
    Q.map f = Q.map g :=
  (ComplexShape.QFactorsThroughHomotopy.areEqualizedByLocalization h).map_eq Q


/-- The functor `HomotopyCategory C c ⥤ HomologicalComplexUpToQuasiIso C c` from the homotopy
category to the localized category with respect to quasi-isomorphisms. -/
def Qh : HomotopyCategory C c ⥤ HomologicalComplexUpToQuasiIso C c :=
  CategoryTheory.Quotient.lift _ HomologicalComplexUpToQuasiIso.Q (by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.23060, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : CategoryTheory.CategoryWithHomology C
      inst✝¹ : (HomologicalComplex.quasiIso C c).HasLocalization
      inst✝ : c.QFactorsThroughHomotopy C
      ⊢ ∀ (x y : HomologicalComplex C c) (f₁ f₂ : Quiver.Hom x y), homotopic C c f₁  …
    -/
    intro K L f g ⟨h⟩
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.23060, u_1} C
      ι : Type u_2
      c : ComplexShape ι
      inst✝³ : CategoryTheory.Preadditive C
      inst✝² : CategoryTheory.CategoryWithHomology C
      inst✝¹ : (HomologicalComplex.quasiIso C c).HasLocalization
      inst✝ : c.QFactorsThroughHomotopy C
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      h : Homotopy f g
      ⊢ Eq (HomologicalComplexUpToQuasiIso.Q.map f) (HomologicalComplexUpToQuasiIso. …
    -/
    exact Q_map_eq_of_homotopy h)
    /-
      🎉 no goals
    -/


/-- The canonical isomorphism `HomotopyCategory.quotient C c ⋙ Qh ≅ Q`. -/
def quotientCompQhIso : HomotopyCategory.quotient C c ⋙ Qh ≅ Q := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.24791, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.CategoryWithHomology C
    inst✝¹ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝ : c.QFactorsThroughHomotopy C
    ⊢ CategoryTheory.Iso ((HomotopyCategory.quotient C c).comp HomologicalComplexU …
  -/
  apply Quotient.lift.isLift
  /-
    🎉 no goals
  -/


lemma Qh_inverts_quasiIso : (HomotopyCategory.quasiIso C c).IsInvertedBy Qh := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.CategoryWithHomology C
    inst✝¹ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝ : c.QFactorsThroughHomotopy C
    ⊢ (HomotopyCategory.quasiIso C c).IsInvertedBy HomologicalComplexUpToQuasiIso.Qh
  -/
  rintro ⟨K⟩ ⟨L⟩ φ
  /-
    case mk.mk
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.CategoryWithHomology C
    inst✝¹ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝ : c.QFactorsThroughHomotopy C
    K L : HomologicalComplex C c
    φ : Quiver.Hom { as := K } { as := L }
    ⊢ HomotopyCategory.quasiIso C c φ → CategoryTheory.IsIso (HomologicalComplexUp …
  -/
  obtain ⟨φ, rfl⟩ := (HomotopyCategory.quotient C c).map_surjective φ
  rw [HomotopyCategory.quotient_map_mem_quasiIso_iff φ,
    ← HomologicalComplexUpToQuasiIso.isIso_Q_map_iff_mem_quasiIso]
  /-
    case mk.mk.intro
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    ι : Type u_2
    c : ComplexShape ι
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.CategoryWithHomology C
    inst✝¹ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝ : c.QFactorsThroughHomotopy C
    K L : HomologicalComplex C c
    φ : Quiver.Hom K L
    ⊢ CategoryTheory.IsIso (HomologicalComplexUpToQuasiIso.Q.map φ) → CategoryTheo …
  -/
  exact (NatIso.isIso_map_iff (quotientCompQhIso C c) φ).2
  /-
    🎉 no goals
  -/


instance : (HomotopyCategory.quotient C c ⋙ Qh).IsLocalization
    (HomologicalComplex.quasiIso C c) :=
  Functor.IsLocalization.of_iso _ (quotientCompQhIso C c).symm


/-- The homology functor on `HomologicalComplexUpToQuasiIso C c` is induced by
the homology functor on `HomotopyCategory C c`. -/
noncomputable def homologyFunctorFactorsh (i : ι ) :
    Qh ⋙ homologyFunctor C c i ≅ HomotopyCategory.homologyFunctor C c i :=
  Quotient.natIsoLift _ ((Functor.associator _ _ _).symm ≪≫
    isoWhiskerRight (quotientCompQhIso C c) _ ≪≫
    homologyFunctorFactors C c i  ≪≫ (HomotopyCategory.homologyFunctorFactors C c i).symm)


/-- The category `HomologicalComplexUpToQuasiIso C c` which was defined as a localization of
`HomologicalComplex C c` with respect to quasi-isomorphisms also identify to a localization
of the homotopy category with respect ot quasi-isomorphisms. -/
instance : HomologicalComplexUpToQuasiIso.Qh.IsLocalization (HomotopyCategory.quasiIso C c) :=
  Functor.IsLocalization.of_comp (HomotopyCategory.quotient C c)
    Qh (HomologicalComplex.homotopyEquivalences C c)
    (HomotopyCategory.quasiIso C c) (HomologicalComplex.quasiIso C c)
    (homotopyEquivalences_le_quasiIso C c)
    (HomotopyCategory.quasiIso_eq_quasiIso_map_quotient C c)


/-- The homotopy category satisfies the universal property of the localized category
with respect to homotopy equivalences. -/
def ComplexShape.strictUniversalPropertyFixedTargetQuotient (E : Type*) [Category E] :
    Localization.StrictUniversalPropertyFixedTarget (HomotopyCategory.quotient C c)
      (HomologicalComplex.homotopyEquivalences C c) E where
  inverts := HomotopyCategory.quotient_inverts_homotopyEquivalences C c
  lift F hF := CategoryTheory.Quotient.lift _ F (by
    /-
      ι : Type u_1
      c : ComplexShape ι
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.35851, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.35915, u_3} E
      F : CategoryTheory.Functor (HomologicalComplex C c) E
      hF : (HomologicalComplex.homotopyEquivalences C c).IsInvertedBy F
      ⊢ ∀ (x y : HomologicalComplex C c) (f₁ f₂ : Quiver.Hom x y), homotopic C c f₁  …
    -/
    intro K L f g ⟨h⟩
    /-
      ι : Type u_1
      c : ComplexShape ι
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.35851, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.35915, u_3} E
      F : CategoryTheory.Functor (HomologicalComplex C c) E
      hF : (HomologicalComplex.homotopyEquivalences C c).IsInvertedBy F
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      h : Homotopy f g
      ⊢ Eq (F.map f) (F.map g)
    -/
    have : DecidableRel c.Rel := by classical infer_instance
    /-
      ι : Type u_1
      c : ComplexShape ι
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.35851, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.35915, u_3} E
      F : CategoryTheory.Functor (HomologicalComplex C c) E
      hF : (HomologicalComplex.homotopyEquivalences C c).IsInvertedBy F
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      h : Homotopy f g
      this : DecidableRel c.Rel
      ⊢ Eq (F.map f) (F.map g)
    -/
    exact h.map_eq_of_inverts_homotopyEquivalences hc F hF)
    /-
      🎉 no goals
    -/
  fac _ _ := rfl
  uniq _ _ h := Quotient.lift_unique' _ _ _ h


lemma ComplexShape.quotient_isLocalization :
    (HomotopyCategory.quotient C c).IsLocalization
      (HomologicalComplex.homotopyEquivalences _ _) := by
  /-
    ι : Type u_1
    c : ComplexShape ι
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    ⊢ (HomotopyCategory.quotient C c).IsLocalization (HomologicalComplex.homotopyE …
  -/
  apply Functor.IsLocalization.mk'
  /-
    case h₁
    ι : Type u_1
    c : ComplexShape ι
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    ⊢ CategoryTheory.Localization.StrictUniversalPropertyFixedTarget (HomotopyCate …
  -/
  all_goals apply c.strictUniversalPropertyFixedTargetQuotient hc
  /-
    🎉 no goals
  -/


lemma ComplexShape.QFactorsThroughHomotopy_of_exists_prev [CategoryWithHomology C] :
    c.QFactorsThroughHomotopy C where
  areEqualizedByLocalization {K L f g} h := by
    /-
      ι : Type u_1
      c : ComplexShape ι
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_2} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      inst✝ : CategoryTheory.CategoryWithHomology C
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      h : Homotopy f g
      ⊢ CategoryTheory.AreEqualizedByLocalization (HomologicalComplex.quasiIso C c)  …
    -/
    have : DecidableRel c.Rel := by classical infer_instance
    exact h.map_eq_of_inverts_homotopyEquivalences hc _
      (MorphismProperty.IsInvertedBy.of_le _ _ _
        (Localization.inverts _ (HomologicalComplex.quasiIso C _))
        (homotopyEquivalences_le_quasiIso C _))


instance : (HomotopyCategory.quotient C (ComplexShape.down ι)).IsLocalization
    (HomologicalComplex.homotopyEquivalences _ _) :=
  (ComplexShape.down ι).quotient_isLocalization (fun _ => ⟨_, rfl⟩) C


instance : (ComplexShape.down ι).QFactorsThroughHomotopy C :=
  (ComplexShape.down ι).QFactorsThroughHomotopy_of_exists_prev (fun _ => ⟨_, rfl⟩) C


instance : (HomotopyCategory.quotient C (ComplexShape.up ℤ)).IsLocalization
    (HomologicalComplex.homotopyEquivalences _ _) :=
                                                                   /-
                                                                     C : Type u_1
                                                                     inst✝² : CategoryTheory.Category.{u_3, u_1} C
                                                                     ι : Type u_2
                                                                     inst✝¹ : CategoryTheory.Preadditive C
                                                                     inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                                                     n : Int
                                                                     ⊢ (ComplexShape.up Int).Rel (HSub.hSub n 1) n
                                                                   -/
  (ComplexShape.up ℤ).quotient_isLocalization (fun n => ⟨n - 1, by simp⟩) C
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance : (ComplexShape.up ℤ).QFactorsThroughHomotopy C :=
                                                                                  /-
                                                                                    C : Type u_1
                                                                                    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
                                                                                    ι : Type u_2
                                                                                    inst✝² : CategoryTheory.Preadditive C
                                                                                    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
                                                                                    inst✝ : CategoryTheory.CategoryWithHomology C
                                                                                    n : Int
                                                                                    ⊢ (ComplexShape.up Int).Rel (HSub.hSub n 1) n
                                                                                  -/
  (ComplexShape.up ℤ).QFactorsThroughHomotopy_of_exists_prev (fun n => ⟨n - 1, by simp⟩) C
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- The localizer morphism which expresses that `F.mapHomologicalComplex c` preserves
quasi-isomorphisms. -/
@[simps]
def mapHomologicalComplexUpToQuasiIsoLocalizerMorphism :
    LocalizerMorphism (HomologicalComplex.quasiIso C c) (HomologicalComplex.quasiIso D c) where
  functor := F.mapHomologicalComplex c
  map _ _ f (_ : QuasiIso f) := HomologicalComplex.quasiIso_map_of_preservesHomology _ _


lemma mapHomologicalComplex_upToQuasiIso_Q_inverts_quasiIso :
    (HomologicalComplex.quasiIso C c).IsInvertedBy
      (F.mapHomologicalComplex c ⋙ HomologicalComplexUpToQuasiIso.Q) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    ι : Type u_3
    c : ComplexShape ι
    inst✝⁶ : CategoryTheory.Preadditive C
    inst✝⁵ : CategoryTheory.Preadditive D
    inst✝⁴ : CategoryTheory.CategoryWithHomology C
    inst✝³ : CategoryTheory.CategoryWithHomology D
    inst✝² : (HomologicalComplex.quasiIso D c).HasLocalization
    inst✝¹ : F.Additive
    inst✝ : F.PreservesHomology
    ⊢ (HomologicalComplex.quasiIso C c).IsInvertedBy ((F.mapHomologicalComplex c). …
  -/
  apply (F.mapHomologicalComplexUpToQuasiIsoLocalizerMorphism c).inverts
  /-
    🎉 no goals
  -/


/-- The functor `HomologicalComplexUpToQuasiIso C c ⥤ HomologicalComplexUpToQuasiIso D c`
induced by a functor `F : C ⥤ D` which preserves homology. -/
noncomputable def mapHomologicalComplexUpToQuasiIso :
    HomologicalComplexUpToQuasiIso C c ⥤ HomologicalComplexUpToQuasiIso D c :=
  (F.mapHomologicalComplexUpToQuasiIsoLocalizerMorphism c).localizedFunctor
    HomologicalComplexUpToQuasiIso.Q HomologicalComplexUpToQuasiIso.Q


noncomputable instance :
    Localization.Lifting HomologicalComplexUpToQuasiIso.Q
      (HomologicalComplex.quasiIso C c)
      (F.mapHomologicalComplex c ⋙ HomologicalComplexUpToQuasiIso.Q)
      (F.mapHomologicalComplexUpToQuasiIso c) :=
  (F.mapHomologicalComplexUpToQuasiIsoLocalizerMorphism c).liftingLocalizedFunctor _ _


/-- The functor `F.mapHomologicalComplexUpToQuasiIso c` is induced by
`F.mapHomologicalComplex c`. -/
noncomputable def mapHomologicalComplexUpToQuasiIsoFactors :
    HomologicalComplexUpToQuasiIso.Q ⋙ F.mapHomologicalComplexUpToQuasiIso c ≅
      F.mapHomologicalComplex c ⋙ HomologicalComplexUpToQuasiIso.Q :=
  Localization.Lifting.iso HomologicalComplexUpToQuasiIso.Q
      (HomologicalComplex.quasiIso C c) _ _


/-- The functor `F.mapHomologicalComplexUpToQuasiIso c` is induced by
`F.mapHomotopyCategory c`. -/
noncomputable def mapHomologicalComplexUpToQuasiIsoFactorsh :
    HomologicalComplexUpToQuasiIso.Qh ⋙ F.mapHomologicalComplexUpToQuasiIso c ≅
      F.mapHomotopyCategory c ⋙ HomologicalComplexUpToQuasiIso.Qh :=
  Localization.liftNatIso (HomotopyCategory.quotient C c)
    (HomologicalComplex.homotopyEquivalences C c)
    (HomotopyCategory.quotient C c ⋙ HomologicalComplexUpToQuasiIso.Qh ⋙
      F.mapHomologicalComplexUpToQuasiIso c)
    (HomotopyCategory.quotient C c ⋙ F.mapHomotopyCategory c ⋙
      HomologicalComplexUpToQuasiIso.Qh) _ _
      (F.mapHomologicalComplexUpToQuasiIsoFactors c)


noncomputable instance :
    Localization.Lifting HomologicalComplexUpToQuasiIso.Qh (HomotopyCategory.quasiIso C c)
      (F.mapHomotopyCategory c ⋙ HomologicalComplexUpToQuasiIso.Qh)
      (F.mapHomologicalComplexUpToQuasiIso c) :=
  ⟨F.mapHomologicalComplexUpToQuasiIsoFactorsh c⟩


@[reassoc]
lemma mapHomologicalComplexUpToQuasiIsoFactorsh_hom_app (K : HomologicalComplex C c) :
    (F.mapHomologicalComplexUpToQuasiIsoFactorsh c).hom.app
        ((HomotopyCategory.quotient _ _).obj K) =
      (F.mapHomologicalComplexUpToQuasiIso c).map
          ((HomologicalComplexUpToQuasiIso.quotientCompQhIso C c).hom.app K) ≫
        (F.mapHomologicalComplexUpToQuasiIsoFactors c).hom.app K ≫
          (HomologicalComplexUpToQuasiIso.quotientCompQhIso D c).inv.app _ ≫
            HomologicalComplexUpToQuasiIso.Qh.map
              ((F.mapHomotopyCategoryFactors c).inv.app K) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹¹ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    ι : Type u_3
    c : ComplexShape ι
    inst✝¹⁰ : CategoryTheory.Preadditive C
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : CategoryTheory.CategoryWithHomology C
    inst✝⁷ : CategoryTheory.CategoryWithHomology D
    inst✝⁶ : (HomologicalComplex.quasiIso D c).HasLocalization
    inst✝⁵ : F.Additive
    inst✝⁴ : F.PreservesHomology
    inst✝³ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝² : c.QFactorsThroughHomotopy C
    inst✝¹ : c.QFactorsThroughHomotopy D
    inst✝ : (HomotopyCategory.quotient C c).IsLocalization (HomologicalComplex.hom …
    K : HomologicalComplex C c
    ⊢ Eq ((F.mapHomologicalComplexUpToQuasiIsoFactorsh c).hom.app ((HomotopyCatego …
  -/
  dsimp [mapHomologicalComplexUpToQuasiIsoFactorsh]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹¹ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    ι : Type u_3
    c : ComplexShape ι
    inst✝¹⁰ : CategoryTheory.Preadditive C
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : CategoryTheory.CategoryWithHomology C
    inst✝⁷ : CategoryTheory.CategoryWithHomology D
    inst✝⁶ : (HomologicalComplex.quasiIso D c).HasLocalization
    inst✝⁵ : F.Additive
    inst✝⁴ : F.PreservesHomology
    inst✝³ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝² : c.QFactorsThroughHomotopy C
    inst✝¹ : c.QFactorsThroughHomotopy D
    inst✝ : (HomotopyCategory.quotient C c).IsLocalization (HomologicalComplex.hom …
    K : HomologicalComplex C c
    ⊢ Eq ((CategoryTheory.Localization.liftNatTrans (HomotopyCategory.quotient C c …
  -/
  rw [Localization.liftNatTrans_app]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹¹ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    ι : Type u_3
    c : ComplexShape ι
    inst✝¹⁰ : CategoryTheory.Preadditive C
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : CategoryTheory.CategoryWithHomology C
    inst✝⁷ : CategoryTheory.CategoryWithHomology D
    inst✝⁶ : (HomologicalComplex.quasiIso D c).HasLocalization
    inst✝⁵ : F.Additive
    inst✝⁴ : F.PreservesHomology
    inst✝³ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝² : c.QFactorsThroughHomotopy C
    inst✝¹ : c.QFactorsThroughHomotopy D
    inst✝ : (HomotopyCategory.quotient C c).IsLocalization (HomologicalComplex.hom …
    K : HomologicalComplex C c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Localization.Lifting …
  -/
  dsimp
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹¹ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    ι : Type u_3
    c : ComplexShape ι
    inst✝¹⁰ : CategoryTheory.Preadditive C
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : CategoryTheory.CategoryWithHomology C
    inst✝⁷ : CategoryTheory.CategoryWithHomology D
    inst✝⁶ : (HomologicalComplex.quasiIso D c).HasLocalization
    inst✝⁵ : F.Additive
    inst✝⁴ : F.PreservesHomology
    inst✝³ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝² : c.QFactorsThroughHomotopy C
    inst✝¹ : c.QFactorsThroughHomotopy D
    inst✝ : (HomotopyCategory.quotient C c).IsLocalization (HomologicalComplex.hom …
    K : HomologicalComplex C c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((F …
  -/
  simp only [Category.comp_id, Category.id_comp]
  change _ = (F.mapHomologicalComplexUpToQuasiIso c).map (𝟙 _) ≫ _ ≫ 𝟙 _ ≫
    HomologicalComplexUpToQuasiIso.Qh.map (𝟙 _)
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹² : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹¹ : CategoryTheory.Category.{u_6, u_2} D
    F : CategoryTheory.Functor C D
    ι : Type u_3
    c : ComplexShape ι
    inst✝¹⁰ : CategoryTheory.Preadditive C
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : CategoryTheory.CategoryWithHomology C
    inst✝⁷ : CategoryTheory.CategoryWithHomology D
    inst✝⁶ : (HomologicalComplex.quasiIso D c).HasLocalization
    inst✝⁵ : F.Additive
    inst✝⁴ : F.PreservesHomology
    inst✝³ : (HomologicalComplex.quasiIso C c).HasLocalization
    inst✝² : c.QFactorsThroughHomotopy C
    inst✝¹ : c.QFactorsThroughHomotopy D
    inst✝ : (HomotopyCategory.quotient C c).IsLocalization (HomologicalComplex.hom …
    K : HomologicalComplex C c
    ⊢ Eq ((F.mapHomologicalComplexUpToQuasiIsoFactors c).hom.app K) (CategoryTheor …
  -/
  simp only [map_id, Category.comp_id, Category.id_comp]
  /-
    🎉 no goals
  -/


