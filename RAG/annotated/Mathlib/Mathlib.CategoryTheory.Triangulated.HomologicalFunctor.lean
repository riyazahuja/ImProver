/-- A functor from a pretriangulated category to an abelian category is an homological functor
if it sends distinguished triangles to exact sequences. -/
class IsHomological extends F.PreservesZeroMorphisms : Prop where
  exact (T : Triangle C) (hT : T ∈ distTriang C) :
    ((shortComplexOfDistTriangle T hT).map F).Exact


lemma map_distinguished_exact [F.IsHomological] (T : Triangle C) (hT : T ∈ distTriang C) :
    ((shortComplexOfDistTriangle T hT).map F).Exact :=
  IsHomological.exact _ hT


instance (L : C ⥤ D) (F : D ⥤ A) [L.CommShift ℤ] [L.IsTriangulated] [F.IsHomological] :
    (L ⋙ F).IsHomological where
  exact T hT := F.map_distinguished_exact _ (L.map_distinguished T hT)


lemma IsHomological.mk' [F.PreservesZeroMorphisms]
    (hF : ∀ (T : Pretriangulated.Triangle C) (hT : T ∈ distTriang C),
      ∃ (T' : Pretriangulated.Triangle C) (e : T ≅ T'),
      ((shortComplexOfDistTriangle T' (isomorphic_distinguished _ hT _ e.symm)).map F).Exact) :
    F.IsHomological where
  exact T hT := by
    /-
      C : Type u_1
      A : Type u_3
      inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁷ : CategoryTheory.HasShift C Int
      inst✝⁶ : CategoryTheory.Category.{u_5, u_3} A
      F : CategoryTheory.Functor C A
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝² : CategoryTheory.Pretriangulated C
      inst✝¹ : CategoryTheory.Abelian A
      inst✝ : F.PreservesZeroMorphisms
      hF : ∀ (T : CategoryTheory.Pretriangulated.Triangle C) (hT : Membership.mem Ca …
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ ((CategoryTheory.Pretriangulated.shortComplexOfDistTriangle T hT).map F).Exact
    -/
    obtain ⟨T', e, h'⟩ := hF T hT
    exact (ShortComplex.exact_iff_of_iso
      (F.mapShortComplex.mapIso ((shortComplexOfDistTriangleIsoOfIso e hT)))).2 h'


lemma IsHomological.of_iso {F₁ F₂ : C ⥤ A} [F₁.IsHomological] (e : F₁ ≅ F₂) :
    F₂.IsHomological :=
  have := preservesZeroMorphisms_of_iso e
  ⟨fun T hT => ShortComplex.exact_of_iso (ShortComplex.mapNatIso _ e)
    (F₁.map_distinguished_exact T hT)⟩


/-- The kernel of a homological functor `F : C ⥤ A` is the strictly full
triangulated subcategory consisting of objects `X` such that
for all `n : ℤ`, `F.obj (X⟦n⟧)` is zero. -/
def homologicalKernel [F.IsHomological] :
    Triangulated.Subcategory C := Triangulated.Subcategory.mk'
  (fun X => ∀ (n : ℤ), IsZero (F.obj (X⟦n⟧)))
  (fun n => by
    rw [IsZero.iff_id_eq_zero, ← F.map_id, ← Functor.map_id,
      id_zero, Functor.map_zero, Functor.map_zero])
  (fun X a hX b => IsZero.of_iso (hX (a + b)) (F.mapIso ((shiftFunctorAdd C a b).app X).symm))
  (fun T hT h₁ h₃ n => (F.map_distinguished_exact _
    (Triangle.shift_distinguished T hT n)).isZero_of_both_zeros
      (IsZero.eq_of_src (h₁ n) _ _) (IsZero.eq_of_tgt (h₃ n) _ _))


instance [F.IsHomological] : ClosedUnderIsomorphisms F.homologicalKernel.P := by
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.Category.{?u.17199, u_2} D
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁰ : CategoryTheory.HasShift D Int
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁷ : CategoryTheory.Pretriangulated D
    inst✝⁶ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.Abelian A
    inst✝ : F.IsHomological
    ⊢ CategoryTheory.ClosedUnderIsomorphisms F.homologicalKernel.P
  -/
  dsimp only [homologicalKernel]
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.Category.{?u.17199, u_2} D
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁰ : CategoryTheory.HasShift D Int
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁷ : CategoryTheory.Pretriangulated D
    inst✝⁶ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.Abelian A
    inst✝ : F.IsHomological
    ⊢ CategoryTheory.ClosedUnderIsomorphisms (CategoryTheory.Triangulated.Subcateg …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma mem_homologicalKernel_iff [F.IsHomological] [F.ShiftSequence ℤ] (X : C) :
    F.homologicalKernel.P X ↔ ∀ (n : ℤ), IsZero ((F.shift n).obj X) := by
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X : C
    ⊢ Iff (F.homologicalKernel.P X) (∀ (n : Int), CategoryTheory.Limits.IsZero ((F …
  -/
  simp only [← fun (n : ℤ) => Iso.isZero_iff ((F.isoShift n).app X)]
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X : C
    ⊢ Iff (F.homologicalKernel.P X) (∀ (n : Int), CategoryTheory.Limits.IsZero ((( …
  -/
  rfl
  /-
    🎉 no goals
  -/


noncomputable instance (priority := 100) [F.IsHomological] :
    PreservesLimitsOfShape (Discrete WalkingPair) F := by
  suffices ∀ (X₁ X₂ : C), PreservesLimit (pair X₁ X₂) F from
    ⟨fun {X} => preservesLimit_of_iso_diagram F (diagramIsoPair X).symm⟩
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.Category.{?u.19832, u_2} D
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁰ : CategoryTheory.HasShift D Int
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁷ : CategoryTheory.Pretriangulated D
    inst✝⁶ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.Abelian A
    inst✝ : F.IsHomological
    ⊢ ∀ (X₁ X₂ : C), CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.p …
  -/
  intro X₁ X₂
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.Category.{?u.19832, u_2} D
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁰ : CategoryTheory.HasShift D Int
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁷ : CategoryTheory.Pretriangulated D
    inst✝⁶ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.Abelian A
    inst✝ : F.IsHomological
    X₁ X₂ : C
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X₁ X₂) F
  -/
  have : HasBinaryBiproduct (F.obj X₁) (F.obj X₂) := HasBinaryBiproducts.has_binary_biproduct _ _
  have : Mono (F.biprodComparison X₁ X₂) := by
    rw [mono_iff_cancel_zero]
    intro Z f hf
    let S := (ShortComplex.mk _ _ (biprod.inl_snd (X := X₁) (Y := X₂))).map F
    have : Mono S.f := by dsimp [S]; infer_instance
    have ex : S.Exact := F.map_distinguished_exact _ (binaryBiproductTriangle_distinguished X₁ X₂)
    obtain ⟨g, rfl⟩ := ex.lift' f (by simpa using hf =≫ biprod.snd)
    dsimp [S] at hf ⊢
    replace hf := hf =≫ biprod.fst
    simp only [assoc, biprodComparison_fst, zero_comp, ← F.map_comp, biprod.inl_fst,
      F.map_id, comp_id] at hf
    rw [hf, zero_comp]
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.Category.{?u.19832, u_2} D
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁰ : CategoryTheory.HasShift D Int
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁷ : CategoryTheory.Pretriangulated D
    inst✝⁶ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.Abelian A
    inst✝ : F.IsHomological
    X₁ X₂ : C
    this✝ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X₁) (F.obj X₂)
    this : CategoryTheory.Mono (F.biprodComparison X₁ X₂)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X₁ X₂) F
  -/
  have : PreservesBinaryBiproduct X₁ X₂ F := preservesBinaryBiproduct_of_mono_biprodComparison _
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.Category.{?u.19832, u_2} D
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹⁰ : CategoryTheory.HasShift D Int
    inst✝⁹ : CategoryTheory.Preadditive D
    inst✝⁸ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝⁷ : CategoryTheory.Pretriangulated D
    inst✝⁶ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.Abelian A
    inst✝ : F.IsHomological
    X₁ X₂ : C
    this✝¹ : CategoryTheory.Limits.HasBinaryBiproduct (F.obj X₁) (F.obj X₂)
    this✝ : CategoryTheory.Mono (F.biprodComparison X₁ X₂)
    this : CategoryTheory.Limits.PreservesBinaryBiproduct X₁ X₂ F
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.pair X₁ X₂) F
  -/
  apply Limits.preservesBinaryProduct_of_preservesBinaryBiproduct
  /-
    🎉 no goals
  -/


instance (priority := 100) [F.IsHomological] : F.Additive :=
  F.additive_of_preserves_binary_products


lemma isHomological_of_localization (L : C ⥤ D)
    [L.CommShift ℤ] [L.IsTriangulated] [L.mapArrow.EssSurj] (F : D ⥤ A)
    (G : C ⥤ A) (e : L ⋙ F ≅ G) [G.IsHomological] :
    F.IsHomological := by
  have : F.PreservesZeroMorphisms := preservesZeroMorphisms_of_map_zero_object
    (F.mapIso L.mapZeroObject.symm ≪≫ e.app _ ≪≫ G.mapZeroObject)
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁶ : CategoryTheory.HasShift C Int
    inst✝¹⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹³ : CategoryTheory.HasShift D Int
    inst✝¹² : CategoryTheory.Preadditive D
    inst✝¹¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝¹⁰ : CategoryTheory.Pretriangulated D
    inst✝⁹ : CategoryTheory.Category.{u_6, u_3} A
    inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Abelian A
    L : CategoryTheory.Functor C D
    inst✝³ : L.CommShift Int
    inst✝² : L.IsTriangulated
    inst✝¹ : L.mapArrow.EssSurj
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    inst✝ : G.IsHomological
    this : F.PreservesZeroMorphisms
    ⊢ F.IsHomological
  -/
  have : (L ⋙ F).IsHomological := IsHomological.of_iso e.symm
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁶ : CategoryTheory.HasShift C Int
    inst✝¹⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹³ : CategoryTheory.HasShift D Int
    inst✝¹² : CategoryTheory.Preadditive D
    inst✝¹¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝¹⁰ : CategoryTheory.Pretriangulated D
    inst✝⁹ : CategoryTheory.Category.{u_6, u_3} A
    inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Abelian A
    L : CategoryTheory.Functor C D
    inst✝³ : L.CommShift Int
    inst✝² : L.IsTriangulated
    inst✝¹ : L.mapArrow.EssSurj
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    inst✝ : G.IsHomological
    this✝ : F.PreservesZeroMorphisms
    this : (L.comp F).IsHomological
    ⊢ F.IsHomological
  -/
  refine IsHomological.mk' _ (fun T hT => ?_)
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁶ : CategoryTheory.HasShift C Int
    inst✝¹⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹³ : CategoryTheory.HasShift D Int
    inst✝¹² : CategoryTheory.Preadditive D
    inst✝¹¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝¹⁰ : CategoryTheory.Pretriangulated D
    inst✝⁹ : CategoryTheory.Category.{u_6, u_3} A
    inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Abelian A
    L : CategoryTheory.Functor C D
    inst✝³ : L.CommShift Int
    inst✝² : L.IsTriangulated
    inst✝¹ : L.mapArrow.EssSurj
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    inst✝ : G.IsHomological
    this✝ : F.PreservesZeroMorphisms
    this : (L.comp F).IsHomological
    T : CategoryTheory.Pretriangulated.Triangle D
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Exists fun T' => Exists fun e => ((CategoryTheory.Pretriangulated.shortCompl …
  -/
  rw [L.distTriang_iff] at hT
  /-
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁶ : CategoryTheory.HasShift C Int
    inst✝¹⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹³ : CategoryTheory.HasShift D Int
    inst✝¹² : CategoryTheory.Preadditive D
    inst✝¹¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝¹⁰ : CategoryTheory.Pretriangulated D
    inst✝⁹ : CategoryTheory.Category.{u_6, u_3} A
    inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Abelian A
    L : CategoryTheory.Functor C D
    inst✝³ : L.CommShift Int
    inst✝² : L.IsTriangulated
    inst✝¹ : L.mapArrow.EssSurj
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e : CategoryTheory.Iso (L.comp F) G
    inst✝ : G.IsHomological
    this✝ : F.PreservesZeroMorphisms
    this : (L.comp F).IsHomological
    T : CategoryTheory.Pretriangulated.Triangle D
    hT✝ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    hT : Membership.mem L.essImageDistTriang T
    ⊢ Exists fun T' => Exists fun e => ((CategoryTheory.Pretriangulated.shortCompl …
  -/
  obtain ⟨T₀, e, hT₀⟩ := hT
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    A : Type u_3
    inst✝¹⁷ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁶ : CategoryTheory.HasShift C Int
    inst✝¹⁵ : CategoryTheory.Category.{u_5, u_2} D
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroObject D
    inst✝¹³ : CategoryTheory.HasShift D Int
    inst✝¹² : CategoryTheory.Preadditive D
    inst✝¹¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝¹⁰ : CategoryTheory.Pretriangulated D
    inst✝⁹ : CategoryTheory.Category.{u_6, u_3} A
    inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.Abelian A
    L : CategoryTheory.Functor C D
    inst✝³ : L.CommShift Int
    inst✝² : L.IsTriangulated
    inst✝¹ : L.mapArrow.EssSurj
    F : CategoryTheory.Functor D A
    G : CategoryTheory.Functor C A
    e✝ : CategoryTheory.Iso (L.comp F) G
    inst✝ : G.IsHomological
    this✝ : F.PreservesZeroMorphisms
    this : (L.comp F).IsHomological
    T : CategoryTheory.Pretriangulated.Triangle D
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    T₀ : CategoryTheory.Pretriangulated.Triangle C
    e : CategoryTheory.Iso T (L.mapTriangle.obj T₀)
    hT₀ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₀
    ⊢ Exists fun T' => Exists fun e => ((CategoryTheory.Pretriangulated.shortCompl …
  -/
  exact ⟨L.mapTriangle.obj T₀, e, (L ⋙ F).map_distinguished_exact _ hT₀⟩
  /-
    🎉 no goals
  -/


/-- The connecting homomorphism in the long exact sequence attached to an homological
functor and a distinguished triangle. -/
noncomputable def homologySequenceδ
    [F.ShiftSequence ℤ] (T : Triangle C) (n₀ n₁ : ℤ) (h : n₀ + 1 = n₁) :
    (F.shift n₀).obj T.obj₃ ⟶ (F.shift n₁).obj T.obj₁ :=
                              /-
                                C : Type u_1
                                D : Type u_2
                                A : Type u_3
                                inst✝⁹ : CategoryTheory.Category.{?u.36525, u_1} C
                                inst✝⁸ : CategoryTheory.HasShift C Int
                                inst✝⁷ : CategoryTheory.Category.{?u.36554, u_2} D
                                inst✝⁶ : CategoryTheory.Limits.HasZeroObject D
                                inst✝⁵ : CategoryTheory.HasShift D Int
                                inst✝⁴ : CategoryTheory.Preadditive D
                                inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
                                inst✝² : CategoryTheory.Pretriangulated D
                                inst✝¹ : CategoryTheory.Category.{?u.36788, u_3} A
                                F : CategoryTheory.Functor C A
                                inst✝ : F.ShiftSequence Int
                                T : CategoryTheory.Pretriangulated.Triangle C
                                n₀ n₁ : Int
                                h : Eq (HAdd.hAdd n₀ 1) n₁
                                ⊢ Eq (HAdd.hAdd 1 n₀) n₁
                              -/
  F.shiftMap T.mor₃ n₀ n₁ (by rw [add_comm 1, h])
                              /-
                                🎉 no goals
                              -/


@[reassoc]
lemma homologySequenceδ_naturality
    [F.ShiftSequence ℤ] (T T' : Triangle C) (φ : T ⟶ T') (n₀ n₁ : ℤ) (h : n₀ + 1 = n₁) :
    (F.shift n₀).map φ.hom₃ ≫ F.homologySequenceδ T' n₀ n₁ h =
      F.homologySequenceδ T n₀ n₁ h ≫ (F.shift n₁).map φ.hom₁ := by
  /-
    C : Type u_1
    A : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝ : F.ShiftSequence Int
    T T' : CategoryTheory.Pretriangulated.Triangle C
    φ : Quiver.Hom T T'
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift n₀).map φ.hom₃) (F.homology …
  -/
  dsimp only [homologySequenceδ]
  /-
    C : Type u_1
    A : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝ : F.ShiftSequence Int
    T T' : CategoryTheory.Pretriangulated.Triangle C
    φ : Quiver.Hom T T'
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift n₀).map φ.hom₃) (F.shiftMap …
  -/
  rw [← shiftMap_comp', ← φ.comm₃, shiftMap_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma comp_homologySequenceδ :
    (F.shift n₀).map T.mor₂ ≫ F.homologySequenceδ T n₀ n₁ h = 0 := by
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_4, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift n₀).map T.mor₂) (F.homology …
  -/
  dsimp only [homologySequenceδ]
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_4, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift n₀).map T.mor₂) (F.shiftMap …
  -/
  rw [← F.shiftMap_comp', comp_distTriang_mor_zero₂₃ _ hT, shiftMap_zero]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homologySequenceδ_comp :
    F.homologySequenceδ T n₀ n₁ h ≫ (F.shift n₁).map T.mor₁ = 0 := by
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_4, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.homologySequenceδ T n₀ n₁ h) ((F.s …
  -/
  dsimp only [homologySequenceδ]
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_4, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.shiftMap T.mor₃ n₀ n₁ ⋯) ((F.shift …
  -/
  rw [← F.shiftMap_comp, comp_distTriang_mor_zero₃₁ _ hT, shiftMap_zero]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homologySequence_comp  :
    (F.shift n₀).map T.mor₁ ≫ (F.shift n₀).map T.mor₂ = 0 := by
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_4, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n₀ : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.shift n₀).map T.mor₁) ((F.shift n …
  -/
  rw [← Functor.map_comp, comp_distTriang_mor_zero₁₂ _ hT, Functor.map_zero]
  /-
    🎉 no goals
  -/


lemma homologySequence_exact₂ :
    (ShortComplex.mk _ _ (F.homologySequence_comp T hT n₀)).Exact := by
  refine ShortComplex.exact_of_iso ?_ (F.map_distinguished_exact _
    (Triangle.shift_distinguished _ hT n₀))
  exact ShortComplex.isoMk ((F.isoShift n₀).app _)
    (n₀.negOnePow • ((F.isoShift n₀).app _)) ((F.isoShift n₀).app _)
    (by dsimp; simp) (by dsimp; simp)


lemma homologySequence_exact₃ :
    (ShortComplex.mk _ _ (F.comp_homologySequenceδ T hT _ _ h)).Exact := by
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_4, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ (CategoryTheory.ShortComplex.mk ((F.shift n₀).map T.mor₂) (F.homologySequenc …
  -/
  refine ShortComplex.exact_of_iso ?_ (F.homologySequence_exact₂ _ (rot_of_distTriang _ hT) n₀)
  exact ShortComplex.isoMk (Iso.refl _) (Iso.refl _)
    ((F.shiftIso 1 n₀ n₁ (by omega)).app _) (by simp) (by simp [homologySequenceδ, shiftMap])


lemma homologySequence_exact₁ :
    (ShortComplex.mk _ _ (F.homologySequenceδ_comp T hT _ _ h)).Exact := by
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_4, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ (CategoryTheory.ShortComplex.mk (F.homologySequenceδ T n₀ n₁ h) ((F.shift n₁ …
  -/
  refine ShortComplex.exact_of_iso ?_ (F.homologySequence_exact₂ _ (inv_rot_of_distTriang _ hT) n₁)
  refine ShortComplex.isoMk (-((F.shiftIso (-1) n₁ n₀ (by omega)).app _))
    (Iso.refl _) (Iso.refl _) ?_ (by simp)
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_4, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Neg.neg ((F.shiftIso (-1) n₁ n₀ ⋯).a …
  -/
  dsimp
  simp only [homologySequenceδ, neg_comp, map_neg, comp_id,
    F.shiftIso_hom_app_comp_shiftMap_of_add_eq_zero T.mor₃ (-1) (neg_add_cancel 1) n₀ n₁ (by omega)]


lemma homologySequence_epi_shift_map_mor₁_iff :
    Epi ((F.shift n₀).map T.mor₁) ↔ (F.shift n₀).map T.mor₂ = 0 :=
  (F.homologySequence_exact₂ T hT n₀).epi_f_iff


lemma homologySequence_mono_shift_map_mor₁_iff :
    Mono ((F.shift n₁).map T.mor₁) ↔ F.homologySequenceδ T n₀ n₁ h = 0 :=
  (F.homologySequence_exact₁ T hT n₀ n₁ h).mono_g_iff


lemma homologySequence_epi_shift_map_mor₂_iff :
    Epi ((F.shift n₀).map T.mor₂) ↔ F.homologySequenceδ T n₀ n₁ h = 0 :=
  (F.homologySequence_exact₃ T hT n₀ n₁ h).epi_f_iff


lemma homologySequence_mono_shift_map_mor₂_iff :
    Mono ((F.shift n₀).map T.mor₂) ↔ (F.shift n₀).map T.mor₁ = 0 :=
  (F.homologySequence_exact₂ T hT n₀).mono_g_iff

lemma mem_homologicalKernel_W_iff {X Y : C} (f : X ⟶ Y) :
    F.homologicalKernel.W f ↔ ∀ (n : ℤ), IsIso ((F.shift n).map f) := by
  /-
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Iff (F.homologicalKernel.W f) (∀ (n : Int), CategoryTheory.IsIso ((F.shift n …
  -/
  obtain ⟨Z, g, h, hT⟩ := distinguished_cocone_triangle f
  /-
    case intro.intro.intro
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
    ⊢ Iff (F.homologicalKernel.W f) (∀ (n : Int), CategoryTheory.IsIso ((F.shift n …
  -/
  apply (F.homologicalKernel.mem_W_iff_of_distinguished _ hT).trans
  /-
    case intro.intro.intro
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
    ⊢ Iff (F.homologicalKernel.P (CategoryTheory.Pretriangulated.Triangle.mk f g h …
  -/
  have h₁ := fun n => (F.homologySequence_exact₃ _ hT n _ rfl).isZero_X₂_iff
  /-
    case intro.intro.intro
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
    h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero (CategoryTheory.ShortCompl …
    ⊢ Iff (F.homologicalKernel.P (CategoryTheory.Pretriangulated.Triangle.mk f g h …
  -/
  have h₂ := fun n => F.homologySequence_mono_shift_map_mor₁_iff _ hT n _ rfl
  /-
    case intro.intro.intro
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
    h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero (CategoryTheory.ShortCompl …
    h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map (Cat …
    ⊢ Iff (F.homologicalKernel.P (CategoryTheory.Pretriangulated.Triangle.mk f g h …
  -/
  have h₃ := fun n => F.homologySequence_epi_shift_map_mor₁_iff _ hT n
  /-
    case intro.intro.intro
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
    h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero (CategoryTheory.ShortCompl …
    h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map (Cat …
    h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map (CategoryTheory.Pre …
    ⊢ Iff (F.homologicalKernel.P (CategoryTheory.Pretriangulated.Triangle.mk f g h …
  -/
  dsimp at h₁ h₂ h₃ ⊢
  /-
    case intro.intro.intro
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
    h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
    h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
    h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
    ⊢ Iff (F.homologicalKernel.P Z) (∀ (n : Int), CategoryTheory.IsIso ((F.shift n …
  -/
  simp only [mem_homologicalKernel_iff, h₁, ← h₂, ← h₃]
  /-
    case intro.intro.intro
    C : Type u_1
    A : Type u_3
    inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
    F : CategoryTheory.Functor C A
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝³ : CategoryTheory.Pretriangulated C
    inst✝² : CategoryTheory.Abelian A
    inst✝¹ : F.IsHomological
    inst✝ : F.ShiftSequence Int
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
    h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
    h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
    h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
    ⊢ Iff (∀ (n : Int), And (CategoryTheory.Epi ((F.shift n).map f)) (CategoryTheo …
  -/
  constructor
    /-
      case intro.intro.intro.mp
      C : Type u_1
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.HasShift C Int
      inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
      F : CategoryTheory.Functor C A
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : F.IsHomological
      inst✝ : F.ShiftSequence Int
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
      h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
      h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
      ⊢ (∀ (n : Int), And (CategoryTheory.Epi ((F.shift n).map f)) (CategoryTheory.M …
    -/
  · intro h n
    /-
      case intro.intro.intro.mp
      C : Type u_1
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.HasShift C Int
      inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
      F : CategoryTheory.Functor C A
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : F.IsHomological
      inst✝ : F.ShiftSequence Int
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h✝ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
      h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
      h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
      h : ∀ (n : Int), And (CategoryTheory.Epi ((F.shift n).map f)) (CategoryTheory. …
      n : Int
      ⊢ CategoryTheory.IsIso ((F.shift n).map f)
    -/
    obtain ⟨m, rfl⟩ : ∃ (m : ℤ), n = m + 1 := ⟨n - 1, by simp⟩
    /-
      case intro.intro.intro.mp.intro
      C : Type u_1
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.HasShift C Int
      inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
      F : CategoryTheory.Functor C A
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : F.IsHomological
      inst✝ : F.ShiftSequence Int
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h✝ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
      h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
      h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
      h : ∀ (n : Int), And (CategoryTheory.Epi ((F.shift n).map f)) (CategoryTheory. …
      m : Int
      ⊢ CategoryTheory.IsIso ((F.shift (HAdd.hAdd m 1)).map f)
    -/
    have := (h (m + 1)).1
    /-
      case intro.intro.intro.mp.intro
      C : Type u_1
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.HasShift C Int
      inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
      F : CategoryTheory.Functor C A
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : F.IsHomological
      inst✝ : F.ShiftSequence Int
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h✝ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
      h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
      h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
      h : ∀ (n : Int), And (CategoryTheory.Epi ((F.shift n).map f)) (CategoryTheory. …
      m : Int
      this : CategoryTheory.Epi ((F.shift (HAdd.hAdd m 1)).map f)
      ⊢ CategoryTheory.IsIso ((F.shift (HAdd.hAdd m 1)).map f)
    -/
    have := (h m).2
    /-
      case intro.intro.intro.mp.intro
      C : Type u_1
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.HasShift C Int
      inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
      F : CategoryTheory.Functor C A
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : F.IsHomological
      inst✝ : F.ShiftSequence Int
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h✝ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
      h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
      h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
      h : ∀ (n : Int), And (CategoryTheory.Epi ((F.shift n).map f)) (CategoryTheory. …
      m : Int
      this✝ : CategoryTheory.Epi ((F.shift (HAdd.hAdd m 1)).map f)
      this : CategoryTheory.Mono ((F.shift (HAdd.hAdd m 1)).map f)
      ⊢ CategoryTheory.IsIso ((F.shift (HAdd.hAdd m 1)).map f)
    -/
    apply isIso_of_mono_of_epi
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.mpr
      C : Type u_1
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.HasShift C Int
      inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
      F : CategoryTheory.Functor C A
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : F.IsHomological
      inst✝ : F.ShiftSequence Int
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
      h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
      h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
      ⊢ (∀ (n : Int), CategoryTheory.IsIso ((F.shift n).map f)) → ∀ (n : Int), And ( …
    -/
  · intros
    /-
      case intro.intro.intro.mpr
      C : Type u_1
      A : Type u_3
      inst✝⁹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁸ : CategoryTheory.HasShift C Int
      inst✝⁷ : CategoryTheory.Category.{u_5, u_3} A
      F : CategoryTheory.Functor C A
      inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝³ : CategoryTheory.Pretriangulated C
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : F.IsHomological
      inst✝ : F.ShiftSequence Int
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      g : Quiver.Hom Y Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      h₁ : ∀ (n : Int), Iff (CategoryTheory.Limits.IsZero ((F.shift n).obj Z)) (And  …
      h₂ : ∀ (n : Int), Iff (CategoryTheory.Mono ((F.shift (HAdd.hAdd n 1)).map f))  …
      h₃ : ∀ (n : Int), Iff (CategoryTheory.Epi ((F.shift n).map f)) (Eq ((F.shift n …
      a✝ : ∀ (n : Int), CategoryTheory.IsIso ((F.shift n).map f)
      n✝ : Int
      ⊢ And (CategoryTheory.Epi ((F.shift n✝).map f)) (CategoryTheory.Mono ((F.shift …
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> infer_instance
                    /-
                      🎉 no goals
                    -/


/-- The exact sequence with six terms starting from `(F.shift n₀).obj T.obj₁` until
`(F.shift n₁).obj T.obj₃` when `T` is a distinguished triangle and `F` a homological functor. -/
@[simp] noncomputable def homologySequenceComposableArrows₅ : ComposableArrows A 5 :=
  mk₅ ((F.shift n₀).map T.mor₁) ((F.shift n₀).map T.mor₂)
    (F.homologySequenceδ T n₀ n₁ h) ((F.shift n₁).map T.mor₁) ((F.shift n₁).map T.mor₂)


include hT in
lemma homologySequenceComposableArrows₅_exact :
    (F.homologySequenceComposableArrows₅ T n₀ n₁ h).Exact :=
  exact_of_δ₀ (F.homologySequence_exact₂ T hT n₀).exact_toComposableArrows
    (exact_of_δ₀ (F.homologySequence_exact₃ T hT n₀ n₁ h).exact_toComposableArrows
      (exact_of_δ₀ (F.homologySequence_exact₁ T hT n₀ n₁ h).exact_toComposableArrows
        (F.homologySequence_exact₂ T hT n₁).exact_toComposableArrows))


