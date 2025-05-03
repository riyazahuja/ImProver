/-- The class of morphisms of presheaves which become isomorphisms after sheafification.
(See `GrothendieckTopology.W_iff`.) -/
abbrev W : MorphismProperty (Cᵒᵖ ⥤ A) := LeftBousfield.W (Presheaf.IsSheaf J)


variable (A) in
lemma W_eq_W_range_sheafToPresheaf_obj :
    J.W = LeftBousfield.W (· ∈ Set.range (sheafToPresheaf J A).obj) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝ : CategoryTheory.Category.{u_4, u_2} A
    ⊢ Eq J.W (CategoryTheory.Localization.LeftBousfield.W fun x => Membership.mem  …
  -/
  apply congr_arg
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝ : CategoryTheory.Category.{u_4, u_2} A
    ⊢ Eq (CategoryTheory.Presheaf.IsSheaf J) fun x => Membership.mem (Set.range (C …
  -/
  ext P
  /-
    case h.h.a
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝ : CategoryTheory.Category.{u_4, u_2} A
    P : CategoryTheory.Functor (Opposite C) A
    ⊢ Iff (CategoryTheory.Presheaf.IsSheaf J P) (Membership.mem (Set.range (Catego …
  -/
  constructor
    /-
      case h.h.a.mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} A
      P : CategoryTheory.Functor (Opposite C) A
      ⊢ CategoryTheory.Presheaf.IsSheaf J P → Membership.mem (Set.range (CategoryThe …
    -/
  · intro hP
    /-
      case h.h.a.mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} A
      P : CategoryTheory.Functor (Opposite C) A
      hP : CategoryTheory.Presheaf.IsSheaf J P
      ⊢ Membership.mem (Set.range (CategoryTheory.sheafToPresheaf J A).obj) P
    -/
    exact ⟨⟨P, hP⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.a.mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} A
      P : CategoryTheory.Functor (Opposite C) A
      ⊢ Membership.mem (Set.range (CategoryTheory.sheafToPresheaf J A).obj) P → Cate …
    -/
  · rintro ⟨F, rfl⟩
    /-
      case h.h.a.mpr.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_2
      inst✝ : CategoryTheory.Category.{u_4, u_2} A
      F : CategoryTheory.Sheaf J A
      ⊢ CategoryTheory.Presheaf.IsSheaf J ((CategoryTheory.sheafToPresheaf J A).obj F)
    -/
    exact F.cond
    /-
      🎉 no goals
    -/


lemma W_sheafToPreheaf_map_iff_isIso {F₁ F₂ : Sheaf J A} (φ : F₁ ⟶ F₂) :
    J.W ((sheafToPresheaf J A).map φ) ↔ IsIso φ := by
  rw [W_eq_W_range_sheafToPresheaf_obj, LeftBousfield.W_iff_isIso _ _ ⟨_, rfl⟩ ⟨_, rfl⟩,
    isIso_iff_of_reflects_iso]


lemma W_adj_unit_app (adj : G ⊣ sheafToPresheaf J A) (P : Cᵒᵖ ⥤ A) : J.W (adj.unit.app P) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} A
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryTh …
    adj : CategoryTheory.Adjunction G (CategoryTheory.sheafToPresheaf J A)
    P : CategoryTheory.Functor (Opposite C) A
    ⊢ J.W (adj.unit.app P)
  -/
  rw [W_eq_W_range_sheafToPresheaf_obj]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} A
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryTh …
    adj : CategoryTheory.Adjunction G (CategoryTheory.sheafToPresheaf J A)
    P : CategoryTheory.Functor (Opposite C) A
    ⊢ CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (Set.ra …
  -/
  exact LeftBousfield.W_adj_unit_app adj P
  /-
    🎉 no goals
  -/


lemma W_iff_isIso_map_of_adjunction (adj : G ⊣ sheafToPresheaf J A)
    {P₁ P₂ : Cᵒᵖ ⥤ A} (f : P₁ ⟶ P₂) :
    J.W f ↔ IsIso (G.map f) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} A
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryTh …
    adj : CategoryTheory.Adjunction G (CategoryTheory.sheafToPresheaf J A)
    P₁ P₂ : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P₁ P₂
    ⊢ Iff (J.W f) (CategoryTheory.IsIso (G.map f))
  -/
  rw [W_eq_W_range_sheafToPresheaf_obj]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} A
    G : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryTh …
    adj : CategoryTheory.Adjunction G (CategoryTheory.sheafToPresheaf J A)
    P₁ P₂ : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P₁ P₂
    ⊢ Iff (CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (S …
  -/
  exact LeftBousfield.W_iff_isIso_map adj f
  /-
    🎉 no goals
  -/


lemma W_eq_inverseImage_isomorphisms_of_adjunction (adj : G ⊣ sheafToPresheaf J A) :
    J.W = (MorphismProperty.isomorphisms _).inverseImage G := by
  rw [W_eq_W_range_sheafToPresheaf_obj,
    LeftBousfield.W_eq_inverseImage_isomorphisms adj]


lemma W_toSheafify (P : Cᵒᵖ ⥤ A) : J.W (toSheafify J P) :=
  J.W_adj_unit_app (sheafificationAdjunction J A) P


lemma W_iff {P₁ P₂ : Cᵒᵖ ⥤ A} (f : P₁ ⟶ P₂) :
    J.W f ↔ IsIso ((presheafToSheaf J A).map f) :=
  J.W_iff_isIso_map_of_adjunction (sheafificationAdjunction J A) f


variable (A) in
lemma W_eq_inverseImage_isomorphisms :
    J.W = (MorphismProperty.isomorphisms _).inverseImage (presheafToSheaf J A) :=
  J.W_eq_inverseImage_isomorphisms_of_adjunction (sheafificationAdjunction J A)


instance : (presheafToSheaf J A).IsLocalization J.W := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} A
    inst✝ : CategoryTheory.HasWeakSheafify J A
    ⊢ (CategoryTheory.presheafToSheaf J A).IsLocalization J.W
  -/
  rw [W_eq_inverseImage_isomorphisms]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} A
    inst✝ : CategoryTheory.HasWeakSheafify J A
    ⊢ (CategoryTheory.presheafToSheaf J A).IsLocalization ((CategoryTheory.Morphis …
  -/
  exact (sheafificationAdjunction J A).isLocalization
  /-
    🎉 no goals
  -/


