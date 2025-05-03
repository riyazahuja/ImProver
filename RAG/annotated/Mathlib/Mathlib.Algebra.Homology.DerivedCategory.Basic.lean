/-- The triangulated subcategory of `HomotopyCategory C (ComplexShape.up ℤ)` consisting
of acyclic complexes. -/
def subcategoryAcyclic : Triangulated.Subcategory (HomotopyCategory C (ComplexShape.up ℤ)) :=
  (homologyFunctor C (ComplexShape.up ℤ) 0).homologicalKernel


instance : ClosedUnderIsomorphisms (subcategoryAcyclic C).P := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    ⊢ CategoryTheory.ClosedUnderIsomorphisms (HomotopyCategory.subcategoryAcyclic  …
  -/
  dsimp [subcategoryAcyclic]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    ⊢ CategoryTheory.ClosedUnderIsomorphisms (HomotopyCategory.homologyFunctor C ( …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma mem_subcategoryAcyclic_iff (X : HomotopyCategory C (ComplexShape.up ℤ)) :
    (subcategoryAcyclic C).P X ↔ ∀ (n : ℤ), IsZero ((homologyFunctor _ _ n).obj X) :=
  Functor.mem_homologicalKernel_iff _ X


lemma quotient_obj_mem_subcategoryAcyclic_iff_exactAt (K : CochainComplex C ℤ) :
    (subcategoryAcyclic C).P ((quotient _ _).obj K) ↔ ∀ (n : ℤ), K.ExactAt n := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    K : CochainComplex C Int
    ⊢ Iff ((HomotopyCategory.subcategoryAcyclic C).P ((HomotopyCategory.quotient C …
  -/
  rw [mem_subcategoryAcyclic_iff]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    K : CochainComplex C Int
    ⊢ Iff (∀ (n : Int), CategoryTheory.Limits.IsZero ((HomotopyCategory.homologyFu …
  -/
  refine forall_congr' (fun n => ?_)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    K : CochainComplex C Int
    n : Int
    ⊢ Iff (CategoryTheory.Limits.IsZero ((HomotopyCategory.homologyFunctor C (Comp …
  -/
  simp only [HomologicalComplex.exactAt_iff_isZero_homology]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    K : CochainComplex C Int
    n : Int
    ⊢ Iff (CategoryTheory.Limits.IsZero ((HomotopyCategory.homologyFunctor C (Comp …
  -/
  exact ((homologyFunctorFactors C (ComplexShape.up ℤ) n).app K).isZero_iff
  /-
    🎉 no goals
  -/


lemma quasiIso_eq_subcategoryAcyclic_W :
    quasiIso C (ComplexShape.up ℤ) = (subcategoryAcyclic C).W := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    ⊢ Eq (HomotopyCategory.quasiIso C (ComplexShape.up Int)) (HomotopyCategory.sub …
  -/
  ext K L f
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    K L : HomotopyCategory C (ComplexShape.up Int)
    f : Quiver.Hom K L
    ⊢ Iff (HomotopyCategory.quasiIso C (ComplexShape.up Int) f) ((HomotopyCategory …
  -/
  exact ((homologyFunctor C (ComplexShape.up ℤ) 0).mem_homologicalKernel_W_iff f).symm
  /-
    🎉 no goals
  -/


/-- The assumption that a localized category for
`(HomologicalComplex.quasiIso C (ComplexShape.up ℤ))` has been chosen, and that the morphisms
in this chosen category are in `Type w`. -/
abbrev HasDerivedCategory := MorphismProperty.HasLocalization.{w}
  (HomologicalComplex.quasiIso C (ComplexShape.up ℤ))


/-- The derived category obtained using the constructed localized category of cochain complexes
with respect to quasi-isomorphisms. This should be used only while proving statements
which do not involve the derived category. -/
def HasDerivedCategory.standard : HasDerivedCategory.{max u v} C :=
  MorphismProperty.HasLocalization.standard _


/-- The derived category of an abelian category. -/
def DerivedCategory : Type (max u v) := HomologicalComplexUpToQuasiIso C (ComplexShape.up ℤ)


instance : Category.{w} (DerivedCategory C) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ CategoryTheory.Category.{w, max u v} (DerivedCategory C)
  -/
  dsimp [DerivedCategory]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ CategoryTheory.Category.{w, max u v} (HomologicalComplexUpToQuasiIso C (Comp …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The localization functor `CochainComplex C ℤ ⥤ DerivedCategory C`. -/
def Q : CochainComplex C ℤ ⥤ DerivedCategory C := HomologicalComplexUpToQuasiIso.Q


instance : (Q (C := C)).IsLocalization
    (HomologicalComplex.quasiIso C (ComplexShape.up ℤ)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ DerivedCategory.Q.IsLocalization (HomologicalComplex.quasiIso C (ComplexShap …
  -/
  dsimp only [Q, DerivedCategory]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ HomologicalComplexUpToQuasiIso.Q.IsLocalization (HomologicalComplex.quasiIso …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {K L : CochainComplex C ℤ} (f : K ⟶ L) [QuasiIso f] :
    IsIso (Q.map f) :=
  Localization.inverts Q (HomologicalComplex.quasiIso C (ComplexShape.up ℤ)) _
    (inferInstanceAs (QuasiIso f))


/-- The localization functor `HomotopyCategory C (ComplexShape.up ℤ) ⥤ DerivedCategory C`. -/
def Qh : HomotopyCategory C (ComplexShape.up ℤ) ⥤ DerivedCategory C :=
  HomologicalComplexUpToQuasiIso.Qh


/-- The natural isomorphism `HomotopyCategory.quotient C (ComplexShape.up ℤ) ⋙ Qh ≅ Q`. -/
def quotientCompQhIso : HomotopyCategory.quotient C (ComplexShape.up ℤ) ⋙ Qh ≅ Q :=
  HomologicalComplexUpToQuasiIso.quotientCompQhIso C (ComplexShape.up ℤ)


instance : Qh.IsLocalization (HomotopyCategory.quasiIso C (ComplexShape.up ℤ)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ DerivedCategory.Qh.IsLocalization (HomotopyCategory.quasiIso C (ComplexShape …
  -/
  dsimp [Qh, DerivedCategory]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ HomologicalComplexUpToQuasiIso.Qh.IsLocalization (HomotopyCategory.quasiIso  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : Qh.IsLocalization (HomotopyCategory.subcategoryAcyclic C).W := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ DerivedCategory.Qh.IsLocalization (HomotopyCategory.subcategoryAcyclic C).W
  -/
  rw [← HomotopyCategory.quasiIso_eq_subcategoryAcyclic_W]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ DerivedCategory.Qh.IsLocalization (HomotopyCategory.quasiIso C (ComplexShape …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance : Preadditive (DerivedCategory C) :=
  Localization.preadditive Qh (HomotopyCategory.subcategoryAcyclic C).W


instance : (Qh (C := C)).Additive :=
  Localization.functor_additive Qh (HomotopyCategory.subcategoryAcyclic C).W


instance : (Q (C := C)).Additive :=
  Functor.additive_of_iso (quotientCompQhIso C)


noncomputable instance : HasZeroObject (DerivedCategory C) :=
  Q.hasZeroObject_of_additive


noncomputable instance : HasShift (DerivedCategory C) ℤ :=
  HasShift.localized Qh (HomotopyCategory.subcategoryAcyclic C).W ℤ


noncomputable instance : (Qh (C := C)).CommShift ℤ :=
  Functor.CommShift.localized Qh (HomotopyCategory.subcategoryAcyclic C).W ℤ


noncomputable instance : (Q (C := C)).CommShift ℤ :=
  Functor.CommShift.ofIso (quotientCompQhIso C) ℤ


instance : NatTrans.CommShift (quotientCompQhIso C).hom ℤ :=
  Functor.CommShift.ofIso_compatibility (quotientCompQhIso C) ℤ


instance (n : ℤ) : (shiftFunctor (DerivedCategory C) n).Additive := by
  rw [Localization.functor_additive_iff
    Qh (HomotopyCategory.subcategoryAcyclic C).W]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    ⊢ (DerivedCategory.Qh.comp (CategoryTheory.shiftFunctor (DerivedCategory C) n) …
  -/
  exact Functor.additive_of_iso (Qh.commShiftIso n)
  /-
    🎉 no goals
  -/


noncomputable instance : Pretriangulated (DerivedCategory C) :=
  Triangulated.Localization.pretriangulated
    Qh (HomotopyCategory.subcategoryAcyclic C).W


instance : (Qh (C := C)).IsTriangulated :=
  Triangulated.Localization.isTriangulated_functor
    Qh (HomotopyCategory.subcategoryAcyclic C).W


noncomputable instance : IsTriangulated (DerivedCategory C) :=
  Triangulated.Localization.isTriangulated
    Qh (HomotopyCategory.subcategoryAcyclic C).W


instance : (Qh (C := C)).mapArrow.EssSurj :=
  Localization.essSurj_mapArrow _ (HomotopyCategory.subcategoryAcyclic C).W


instance {D : Type*} [Category D] : ((whiskeringLeft _ _ D).obj (Qh (C := C))).Full :=
  inferInstanceAs
    (Localization.whiskeringLeftFunctor' _ (HomotopyCategory.quasiIso _ _) D).Full


instance {D : Type*} [Category D] : ((whiskeringLeft _ _ D).obj (Qh (C := C))).Faithful :=
  inferInstanceAs
    (Localization.whiskeringLeftFunctor' _ (HomotopyCategory.quasiIso _ _) D).Faithful


variable {C} in
lemma mem_distTriang_iff (T : Triangle (DerivedCategory C)) :
    (T ∈ distTriang (DerivedCategory C)) ↔ ∃ (X Y : CochainComplex C ℤ) (f : X ⟶ Y),
      Nonempty (T ≅ Q.mapTriangle.obj (CochainComplex.mappingCone.triangle f)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    T : CategoryTheory.Pretriangulated.Triangle (DerivedCategory C)
    ⊢ Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T) …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      T : CategoryTheory.Pretriangulated.Triangle (DerivedCategory C)
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T → Exi …
    -/
  · rintro ⟨T', e, ⟨X, Y, f, ⟨e'⟩⟩⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      T : CategoryTheory.Pretriangulated.Triangle (DerivedCategory C)
      T' : CategoryTheory.Pretriangulated.Triangle (HomotopyCategory C (ComplexShape …
      e : CategoryTheory.Iso T (DerivedCategory.Qh.mapTriangle.obj T')
      X Y : CochainComplex C Int
      f : Quiver.Hom X Y
      e' : CategoryTheory.Iso T' (CochainComplex.mappingCone.triangleh f)
      ⊢ Exists fun X => Exists fun Y => Exists fun f => Nonempty (CategoryTheory.Iso …
    -/
    refine ⟨_, _, f, ⟨?_⟩⟩
    exact e ≪≫ Qh.mapTriangle.mapIso e' ≪≫
      (Functor.mapTriangleCompIso (HomotopyCategory.quotient C _) Qh).symm.app _ ≪≫
      (Functor.mapTriangleIso (quotientCompQhIso C)).app _
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      T : CategoryTheory.Pretriangulated.Triangle (DerivedCategory C)
      ⊢ (Exists fun X => Exists fun Y => Exists fun f => Nonempty (CategoryTheory.Is …
    -/
  · rintro ⟨X, Y, f, ⟨e⟩⟩
    refine isomorphic_distinguished _ (Qh.map_distinguished _ ?_) _
      (e ≪≫ (Functor.mapTriangleIso (quotientCompQhIso C)).symm.app _ ≪≫
      (Functor.mapTriangleCompIso (HomotopyCategory.quotient C _) Qh).app _)
    /-
      case mpr.intro.intro.intro.intro
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      T : CategoryTheory.Pretriangulated.Triangle (DerivedCategory C)
      X Y : CochainComplex C Int
      f : Quiver.Hom X Y
      e : CategoryTheory.Iso T (DerivedCategory.Q.mapTriangle.obj (CochainComplex.ma …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles ((Homot …
    -/
    exact ⟨_, _, f, ⟨Iso.refl _⟩⟩
    /-
      🎉 no goals
    -/


/-- The single functors `C ⥤ DerivedCategory C` for all `n : ℤ` along with
their compatibilities with shifts. -/
noncomputable def singleFunctors : SingleFunctors C (DerivedCategory C) ℤ :=
  (HomotopyCategory.singleFunctors C).postcomp Qh


/-- The shift functor `C ⥤ DerivedCategory C` which sends `X : C` to the
single cochain complex with `X` sitting in degree `n : ℤ`. -/
noncomputable abbrev singleFunctor (n : ℤ) := (singleFunctors C).functor n


instance (n : ℤ) : (singleFunctor C n).Additive := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    ⊢ (DerivedCategory.singleFunctor C n).Additive
  -/
  dsimp [singleFunctor, singleFunctors]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    ⊢ (((HomotopyCategory.singleFunctors C).functor n).comp DerivedCategory.Qh).Ad …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The isomorphism
`DerivedCategory.singleFunctors C ≅ (HomotopyCategory.singleFunctors C).postcomp Qh` given
by the definition of `DerivedCategory.singleFunctors`. -/
noncomputable def singleFunctorsPostcompQhIso :
    singleFunctors C ≅ (HomotopyCategory.singleFunctors C).postcomp Qh :=
  Iso.refl _


/-- The isomorphism
`DerivedCategory.singleFunctors C ≅ (CochainComplex.singleFunctors C).postcomp Q`. -/
noncomputable def singleFunctorsPostcompQIso :
    singleFunctors C ≅ (CochainComplex.singleFunctors C).postcomp Q :=
  (SingleFunctors.postcompFunctor C ℤ (Qh : _ ⥤ DerivedCategory C)).mapIso
    (HomotopyCategory.singleFunctorsPostcompQuotientIso C) ≪≫
      (CochainComplex.singleFunctors C).postcompPostcompIso (HomotopyCategory.quotient _ _) Qh ≪≫
      SingleFunctors.postcompIsoOfIso
        (CochainComplex.singleFunctors C) (quotientCompQhIso C)


lemma singleFunctorsPostcompQIso_hom_hom (n : ℤ) :
    (singleFunctorsPostcompQIso C).hom.hom n = 𝟙 _ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    ⊢ Eq ((DerivedCategory.singleFunctorsPostcompQIso C).hom.hom n) (CategoryTheor …
  -/
  ext X
  dsimp [singleFunctorsPostcompQIso, HomotopyCategory.singleFunctorsPostcompQuotientIso,
    quotientCompQhIso, HomologicalComplexUpToQuasiIso.quotientCompQhIso]
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.SingleFunctors.pos …
  -/
  rw [CategoryTheory.Functor.map_id, SingleFunctors.id_hom, NatTrans.id_app]
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((( …
  -/
  erw [Category.id_comp, Category.id_comp]
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.id (HomologicalComplexUpToQuasiIso.Q.obj ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma singleFunctorsPostcompQIso_inv_hom (n : ℤ) :
    (singleFunctorsPostcompQIso C).inv.hom n = 𝟙 _ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    ⊢ Eq ((DerivedCategory.singleFunctorsPostcompQIso C).inv.hom n) (CategoryTheor …
  -/
  ext X
  dsimp [singleFunctorsPostcompQIso, HomotopyCategory.singleFunctorsPostcompQuotientIso,
    quotientCompQhIso, HomologicalComplexUpToQuasiIso.quotientCompQhIso]
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [CategoryTheory.Functor.map_id]
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [SingleFunctors.id_hom, NatTrans.id_app]
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [Category.id_comp, Category.id_comp]
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    n : Int
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((((CategoryTheory.SingleFunctors.postc …
  -/
  rfl
  /-
    🎉 no goals
  -/


