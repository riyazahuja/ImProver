/-- A functor `F : A ⥤ B` preserves the sheafification for the Grothendieck
topology `J` on a category `C` if whenever a morphism of presheaves `f : P₁ ⟶ P₂`
in `Cᵒᵖ ⥤ A` is such that becomes an iso after sheafification, then it is
also the case of `whiskerRight f F : P₁ ⋙ F ⟶ P₂ ⋙ F`. -/
class PreservesSheafification : Prop where
  le : J.W ≤ J.W.inverseImage ((whiskeringRight Cᵒᵖ A B).obj F)


lemma W_of_preservesSheafification
    {P₁ P₂ : Cᵒᵖ ⥤ A} (f : P₁ ⟶ P₂) (hf : J.W f) :
    J.W (whiskerRight f F) :=
  PreservesSheafification.le _ hf


lemma W_isInvertedBy_whiskeringRight_presheafToSheaf :
    J.W.IsInvertedBy (((whiskeringRight Cᵒᵖ A B).obj F) ⋙ presheafToSheaf J B) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} A
    inst✝² : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝¹ : J.PreservesSheafification F
    inst✝ : CategoryTheory.HasWeakSheafify J B
    ⊢ J.W.IsInvertedBy (((CategoryTheory.whiskeringRight (Opposite C) A B).obj F). …
  -/
  intro P₁ P₂ f hf
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} A
    inst✝² : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝¹ : J.PreservesSheafification F
    inst✝ : CategoryTheory.HasWeakSheafify J B
    P₁ P₂ : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P₁ P₂
    hf : J.W f
    ⊢ CategoryTheory.IsIso ((((CategoryTheory.whiskeringRight (Opposite C) A B).ob …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} A
    inst✝² : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝¹ : J.PreservesSheafification F
    inst✝ : CategoryTheory.HasWeakSheafify J B
    P₁ P₂ : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P₁ P₂
    hf : J.W f
    ⊢ CategoryTheory.IsIso ((CategoryTheory.presheafToSheaf J B).map (CategoryTheo …
  -/
  rw [← W_iff]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} A
    inst✝² : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝¹ : J.PreservesSheafification F
    inst✝ : CategoryTheory.HasWeakSheafify J B
    P₁ P₂ : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P₁ P₂
    hf : J.W f
    ⊢ J.W (CategoryTheory.whiskerRight f F)
  -/
  exact J.W_of_preservesSheafification F _ hf
  /-
    🎉 no goals
  -/


/-- This is the functor sending a sheaf `X : Sheaf J A` to the sheafification
of `X.val ⋙ F`. -/
noncomputable abbrev Sheaf.composeAndSheafify : Sheaf J A ⥤ Sheaf J B :=
  sheafToPresheaf J A ⋙ (whiskeringRight _ _ _).obj F ⋙ presheafToSheaf J B


/-- The canonical natural transformation from
`(whiskeringRight Cᵒᵖ A B).obj F ⋙ presheafToSheaf J B` to
`presheafToSheaf J A ⋙ Sheaf.composeAndSheafify J F`. -/
@[simps!]
noncomputable def toPresheafToSheafCompComposeAndSheafify :
    (whiskeringRight Cᵒᵖ A B).obj F ⋙ presheafToSheaf J B ⟶
      presheafToSheaf J A ⋙ Sheaf.composeAndSheafify J F :=
  whiskerRight (sheafificationAdjunction J A).unit
    ((whiskeringRight _ _ _).obj F ⋙ presheafToSheaf J B)


instance : IsIso (toPresheafToSheafCompComposeAndSheafify J F) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} A
    inst✝³ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ : J.PreservesSheafification F
    ⊢ CategoryTheory.IsIso (CategoryTheory.toPresheafToSheafCompComposeAndSheafify …
  -/
  have : J.PreservesSheafification F := inferInstance
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} A
    inst✝³ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ this : J.PreservesSheafification F
    ⊢ CategoryTheory.IsIso (CategoryTheory.toPresheafToSheafCompComposeAndSheafify …
  -/
  rw [NatTrans.isIso_iff_isIso_app]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} A
    inst✝³ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ this : J.PreservesSheafification F
    ⊢ ∀ (X : CategoryTheory.Functor (Opposite C) A), CategoryTheory.IsIso ((Catego …
  -/
  intro X
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} A
    inst✝³ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ this : J.PreservesSheafification F
    X : CategoryTheory.Functor (Opposite C) A
    ⊢ CategoryTheory.IsIso ((CategoryTheory.toPresheafToSheafCompComposeAndSheafif …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} A
    inst✝³ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : CategoryTheory.HasWeakSheafify J A
    inst✝ this : J.PreservesSheafification F
    X : CategoryTheory.Functor (Opposite C) A
    ⊢ CategoryTheory.IsIso ((CategoryTheory.presheafToSheaf J B).map (CategoryTheo …
  -/
  simpa only [← J.W_iff] using J.W_of_preservesSheafification F _ (J.W_toSheafify X)
  /-
    🎉 no goals
  -/


/-- The canonical isomorphism between `presheafToSheaf J A ⋙ Sheaf.composeAndSheafify J F`
and `(whiskeringRight Cᵒᵖ A B).obj F ⋙ presheafToSheaf J B` when `F : A ⥤ B`
preserves sheafification. -/
@[simps! inv_app]
noncomputable def presheafToSheafCompComposeAndSheafifyIso :
    presheafToSheaf J A ⋙ Sheaf.composeAndSheafify J F ≅
      (whiskeringRight Cᵒᵖ A B).obj F ⋙ presheafToSheaf J B :=
  (asIso (toPresheafToSheafCompComposeAndSheafify J F)).symm


noncomputable instance : Localization.Lifting (presheafToSheaf J A) J.W
    ((whiskeringRight Cᵒᵖ A B).obj F ⋙ presheafToSheaf J B) (Sheaf.composeAndSheafify J F) :=
  ⟨presheafToSheafCompComposeAndSheafifyIso J F⟩


lemma GrothendieckTopology.preservesSheafification_iff_of_adjunctions
    (adj₂ : G₂ ⊣ sheafToPresheaf J B) :
    J.PreservesSheafification F ↔ ∀ (P : Cᵒᵖ ⥤ A),
      IsIso (G₂.map (whiskerRight (adj₁.unit.app P) F)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
    inst✝ : CategoryTheory.Category.{u_3, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    ⊢ Iff (J.PreservesSheafification F) (∀ (P : CategoryTheory.Functor (Opposite C …
  -/
  simp only [← J.W_iff_isIso_map_of_adjunction adj₂]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
    inst✝ : CategoryTheory.Category.{u_3, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    ⊢ Iff (J.PreservesSheafification F) (∀ (P : CategoryTheory.Functor (Opposite C …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      ⊢ J.PreservesSheafification F → ∀ (P : CategoryTheory.Functor (Opposite C) A), …
    -/
  · intro _ P
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      a✝ : J.PreservesSheafification F
      P : CategoryTheory.Functor (Opposite C) A
      ⊢ J.W (CategoryTheory.whiskerRight (adj₁.unit.app P) F)
    -/
    apply W_of_preservesSheafification
    /-
      case mp.hf
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      a✝ : J.PreservesSheafification F
      P : CategoryTheory.Functor (Opposite C) A
      ⊢ J.W (adj₁.unit.app P)
    -/
    rw [J.W_iff_isIso_map_of_adjunction adj₁]
    /-
      case mp.hf
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      a✝ : J.PreservesSheafification F
      P : CategoryTheory.Functor (Opposite C) A
      ⊢ CategoryTheory.IsIso (G₁.map (adj₁.unit.app P))
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      ⊢ (∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whiskerR …
    -/
  · intro h
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      ⊢ J.PreservesSheafification F
    -/
    constructor
    /-
      case mpr.le
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      ⊢ LE.le J.W (J.W.inverseImage ((CategoryTheory.whiskeringRight (Opposite C) A  …
    -/
    intro P₁ P₂ f hf
    /-
      case mpr.le
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      P₁ P₂ : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P₁ P₂
      hf : J.W f
      ⊢ J.W.inverseImage ((CategoryTheory.whiskeringRight (Opposite C) A B).obj F) f
    -/
    rw [J.W_iff_isIso_map_of_adjunction adj₁] at hf
    /-
      case mpr.le
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      P₁ P₂ : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P₁ P₂
      hf : CategoryTheory.IsIso (G₁.map f)
      ⊢ J.W.inverseImage ((CategoryTheory.whiskeringRight (Opposite C) A B).obj F) f
    -/
    dsimp [MorphismProperty.inverseImage]
    /-
      case mpr.le
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      P₁ P₂ : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P₁ P₂
      hf : CategoryTheory.IsIso (G₁.map f)
      ⊢ J.W (CategoryTheory.whiskerRight f F)
    -/
    rw [← (W _).postcomp_iff _ _ (h P₂), ← whiskerRight_comp]
    /-
      case mpr.le
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      P₁ P₂ : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P₁ P₂
      hf : CategoryTheory.IsIso (G₁.map f)
      ⊢ J.W (CategoryTheory.whiskerRight (CategoryTheory.CategoryStruct.comp f (adj₁ …
    -/
    erw [adj₁.unit.naturality f]
    /-
      case mpr.le
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      P₁ P₂ : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P₁ P₂
      hf : CategoryTheory.IsIso (G₁.map f)
      ⊢ J.W (CategoryTheory.whiskerRight (CategoryTheory.CategoryStruct.comp (adj₁.u …
    -/
    dsimp only [Functor.comp_map]
    /-
      case mpr.le
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      P₁ P₂ : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P₁ P₂
      hf : CategoryTheory.IsIso (G₁.map f)
      ⊢ J.W (CategoryTheory.whiskerRight (CategoryTheory.CategoryStruct.comp (adj₁.u …
    -/
    rw [whiskerRight_comp, (W _).precomp_iff _ _ (h P₁)]
    /-
      case mpr.le
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} A
      inst✝ : CategoryTheory.Category.{u_3, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      h : ∀ (P : CategoryTheory.Functor (Opposite C) A), J.W (CategoryTheory.whisker …
      P₁ P₂ : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P₁ P₂
      hf : CategoryTheory.IsIso (G₁.map f)
      ⊢ J.W (CategoryTheory.whiskerRight ((CategoryTheory.sheafToPresheaf J A).map ( …
    -/
    apply Localization.LeftBousfield.W_of_isIso
    /-
      🎉 no goals
    -/


/-- The canonical natural transformation
`(whiskeringRight Cᵒᵖ A B).obj F ⋙ G₂ ⟶ G₁ ⋙ sheafCompose J F`
when `F : A ⥤ B` is such that `J.HasSheafCompose F`, and that `G₁` and `G₂` are
left adjoints to the forget functors `sheafToPresheaf`. -/
def sheafComposeNatTrans :
    (whiskeringRight Cᵒᵖ A B).obj F ⋙ G₂ ⟶ G₁ ⋙ sheafCompose J F where
  app P := (adj₂.homEquiv _ _).symm (whiskerRight (adj₁.unit.app P) F)
  naturality {P Q} f := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝² : CategoryTheory.Category.{?u.31758, u_1} A
      inst✝¹ : CategoryTheory.Category.{?u.31762, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      inst✝ : J.HasSheafCompose F
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.whiskeringRight (O …
    -/
    dsimp
    erw [← adj₂.homEquiv_naturality_left_symm,
      ← adj₂.homEquiv_naturality_right_symm]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝² : CategoryTheory.Category.{?u.31758, u_1} A
      inst✝¹ : CategoryTheory.Category.{?u.31762, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      inst✝ : J.HasSheafCompose F
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      ⊢ Eq ((adj₂.homEquiv (P.comp F) ((CategoryTheory.sheafCompose J F).obj (G₁.obj …
    -/
    dsimp
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝² : CategoryTheory.Category.{?u.31758, u_1} A
      inst✝¹ : CategoryTheory.Category.{?u.31762, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      inst✝ : J.HasSheafCompose F
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      ⊢ Eq ((adj₂.homEquiv (P.comp F) ((CategoryTheory.sheafCompose J F).obj (G₁.obj …
    -/
    rw [← whiskerRight_comp, ← whiskerRight_comp]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝² : CategoryTheory.Category.{?u.31758, u_1} A
      inst✝¹ : CategoryTheory.Category.{?u.31762, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      inst✝ : J.HasSheafCompose F
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      ⊢ Eq ((adj₂.homEquiv (P.comp F) ((CategoryTheory.sheafCompose J F).obj (G₁.obj …
    -/
    erw [adj₁.unit.naturality f]
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u_1
      B : Type u_2
      inst✝² : CategoryTheory.Category.{?u.31758, u_1} A
      inst✝¹ : CategoryTheory.Category.{?u.31762, u_2} B
      F : CategoryTheory.Functor A B
      G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
      adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
      G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
      adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
      inst✝ : J.HasSheafCompose F
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      ⊢ Eq ((adj₂.homEquiv (P.comp F) ((CategoryTheory.sheafCompose J F).obj (G₁.obj …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma sheafComposeNatTrans_fac (P : Cᵒᵖ ⥤ A) :
    adj₂.unit.app (P ⋙ F) ≫
      (sheafToPresheaf J B).map ((sheafComposeNatTrans J F adj₁ adj₂).app P) =
        whiskerRight (adj₁.unit.app P) F  := by
  simp [sheafComposeNatTrans, -sheafToPresheaf_obj, -sheafToPresheaf_map,
    Adjunction.homEquiv_counit]


lemma sheafComposeNatTrans_app_uniq (P : Cᵒᵖ ⥤ A)
    (α : G₂.obj (P ⋙ F) ⟶ (sheafCompose J F).obj (G₁.obj P))
    (hα : adj₂.unit.app (P ⋙ F) ≫ (sheafToPresheaf J B).map α =
        whiskerRight (adj₁.unit.app P) F) :
    α = (sheafComposeNatTrans J F adj₁ adj₂).app P := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝ : J.HasSheafCompose F
    P : CategoryTheory.Functor (Opposite C) A
    α : Quiver.Hom (G₂.obj (P.comp F)) ((CategoryTheory.sheafCompose J F).obj (G₁. …
    hα : Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (P.comp F)) ((Categ …
    ⊢ Eq α ((CategoryTheory.sheafComposeNatTrans J F adj₁ adj₂).app P)
  -/
  apply (adj₂.homEquiv _ _).injective
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝ : J.HasSheafCompose F
    P : CategoryTheory.Functor (Opposite C) A
    α : Quiver.Hom (G₂.obj (P.comp F)) ((CategoryTheory.sheafCompose J F).obj (G₁. …
    hα : Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (P.comp F)) ((Categ …
    ⊢ Eq ((adj₂.homEquiv (P.comp F) ((CategoryTheory.sheafCompose J F).obj (G₁.obj …
  -/
  dsimp [sheafComposeNatTrans]
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝ : J.HasSheafCompose F
    P : CategoryTheory.Functor (Opposite C) A
    α : Quiver.Hom (G₂.obj (P.comp F)) ((CategoryTheory.sheafCompose J F).obj (G₁. …
    hα : Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (P.comp F)) ((Categ …
    ⊢ Eq ((adj₂.homEquiv (P.comp F) ((CategoryTheory.sheafCompose J F).obj (G₁.obj …
  -/
  erw [Equiv.apply_symm_apply]
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝ : J.HasSheafCompose F
    P : CategoryTheory.Functor (Opposite C) A
    α : Quiver.Hom (G₂.obj (P.comp F)) ((CategoryTheory.sheafCompose J F).obj (G₁. …
    hα : Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (P.comp F)) ((Categ …
    ⊢ Eq ((adj₂.homEquiv (P.comp F) ((CategoryTheory.sheafCompose J F).obj (G₁.obj …
  -/
  rw [← hα]
  /-
    case a
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝ : J.HasSheafCompose F
    P : CategoryTheory.Functor (Opposite C) A
    α : Quiver.Hom (G₂.obj (P.comp F)) ((CategoryTheory.sheafCompose J F).obj (G₁. …
    hα : Eq (CategoryTheory.CategoryStruct.comp (adj₂.unit.app (P.comp F)) ((Categ …
    ⊢ Eq ((adj₂.homEquiv (P.comp F) ((CategoryTheory.sheafCompose J F).obj (G₁.obj …
  -/
  apply adj₂.homEquiv_unit
  /-
    🎉 no goals
  -/


lemma GrothendieckTopology.preservesSheafification_iff_of_adjunctions_of_hasSheafCompose :
    J.PreservesSheafification F ↔ IsIso (sheafComposeNatTrans J F adj₁ adj₂) := by
  rw [J.preservesSheafification_iff_of_adjunctions F adj₁ adj₂,
    NatTrans.isIso_iff_isIso_app]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝ : J.HasSheafCompose F
    ⊢ Iff (∀ (P : CategoryTheory.Functor (Opposite C) A), CategoryTheory.IsIso (G₂ …
  -/
  apply forall_congr'
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} A
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝ : J.HasSheafCompose F
    ⊢ ∀ (a : CategoryTheory.Functor (Opposite C) A), Iff (CategoryTheory.IsIso (G₂ …
  -/
  intro P
  rw [← J.W_iff_isIso_map_of_adjunction adj₂, ← J.W_sheafToPreheaf_map_iff_isIso,
    ← sheafComposeNatTrans_fac J F adj₁ adj₂,
    (W _).precomp_iff _ _ (J.W_adj_unit_app adj₂ (P ⋙ F))]


instance : IsIso (sheafComposeNatTrans J F adj₁ adj₂) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} A
    inst✝² : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝¹ : J.HasSheafCompose F
    inst✝ : J.PreservesSheafification F
    ⊢ CategoryTheory.IsIso (CategoryTheory.sheafComposeNatTrans J F adj₁ adj₂)
  -/
  rw [← J.preservesSheafification_iff_of_adjunctions_of_hasSheafCompose]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} A
    inst✝² : CategoryTheory.Category.{u_4, u_2} B
    F : CategoryTheory.Functor A B
    G₁ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) A) (CategoryT …
    adj₁ : CategoryTheory.Adjunction G₁ (CategoryTheory.sheafToPresheaf J A)
    G₂ : CategoryTheory.Functor (CategoryTheory.Functor (Opposite C) B) (CategoryT …
    adj₂ : CategoryTheory.Adjunction G₂ (CategoryTheory.sheafToPresheaf J B)
    inst✝¹ : J.HasSheafCompose F
    inst✝ : J.PreservesSheafification F
    ⊢ J.PreservesSheafification F
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The canonical natural isomorphism
`(whiskeringRight Cᵒᵖ A B).obj F ⋙ G₂ ≅ G₁ ⋙ sheafCompose J F`
when `F : A ⥤ B` preserves sheafification, and that `G₁` and `G₂` are
left adjoints to the forget functors `sheafToPresheaf`. -/
noncomputable def sheafComposeNatIso :
    (whiskeringRight Cᵒᵖ A B).obj F ⋙ G₂ ≅ G₁ ⋙ sheafCompose J F :=
  asIso (sheafComposeNatTrans J F adj₁ adj₂)


/-- The canonical isomorphism `sheafify J (P ⋙ F) ≅ sheafify J P ⋙ F` when
`F` preserves the sheafification. -/
noncomputable def sheafifyComposeIso :
    sheafify J (P ⋙ F) ≅ sheafify J P ⋙ F :=
  (sheafToPresheaf J B).mapIso
    ((sheafComposeNatIso J F (sheafificationAdjunction J A) (sheafificationAdjunction J B)).app P)


@[reassoc (attr := simp)]
lemma sheafComposeIso_hom_fac :
    toSheafify J (P ⋙ F) ≫ (sheafifyComposeIso J F P).hom =
      whiskerRight (toSheafify J P) F :=
  sheafComposeNatTrans_fac J F (sheafificationAdjunction J A) (sheafificationAdjunction J B) P


@[reassoc (attr := simp)]
lemma sheafComposeIso_inv_fac :
    whiskerRight (toSheafify J P) F ≫ (sheafifyComposeIso J F P).inv =
      toSheafify J (P ⋙ F) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} A
    inst✝⁴ : CategoryTheory.Category.{u_3, u_2} B
    F : CategoryTheory.Functor A B
    inst✝³ : CategoryTheory.HasWeakSheafify J A
    inst✝² : CategoryTheory.HasWeakSheafify J B
    inst✝¹ : J.HasSheafCompose F
    inst✝ : J.PreservesSheafification F
    P : CategoryTheory.Functor (Opposite C) A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (Categor …
  -/
  rw [← sheafComposeIso_hom_fac, assoc, Iso.hom_inv_id, comp_id]
  /-
    🎉 no goals
  -/


lemma sheafToPresheaf_map_sheafComposeNatTrans_eq_sheafifyCompIso_inv (P : Cᵒᵖ ⥤ D) :
    (sheafToPresheaf J E).map
      ((sheafComposeNatTrans J F (plusPlusAdjunction J D) (plusPlusAdjunction J E)).app P) =
      (sheafifyCompIso J F P).inv := by
  suffices (sheafComposeNatTrans J F (plusPlusAdjunction J D) (plusPlusAdjunction J E)).app P =
    ⟨(sheafifyCompIso J F P).inv⟩ by
    rw [this]
    rfl
  /-
    C : Type u
    inst✝¹⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_3
    E : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{max v u, u_3} D
    inst✝¹⁴ : CategoryTheory.Category.{max v u, u_4} E
    F : CategoryTheory.Functor D E
    inst✝¹³ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝¹⁰ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝⁹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁸ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
    inst✝ : (CategoryTheory.forget E).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq ((CategoryTheory.sheafComposeNatTrans J F (CategoryTheory.plusPlusAdjunct …
  -/
  apply ((plusPlusAdjunction J E).homEquiv _ _).injective
  /-
    case a
    C : Type u
    inst✝¹⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_3
    E : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{max v u, u_3} D
    inst✝¹⁴ : CategoryTheory.Category.{max v u, u_4} E
    F : CategoryTheory.Functor D E
    inst✝¹³ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝¹⁰ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝⁹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁸ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
    inst✝ : (CategoryTheory.forget E).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Eq (((CategoryTheory.plusPlusAdjunction J E).homEquiv (((CategoryTheory.whis …
  -/
  convert sheafComposeNatTrans_fac J F (plusPlusAdjunction J D) (plusPlusAdjunction J E) P
  /-
    case h.e'_3.h
    C : Type u
    inst✝¹⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_3
    E : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{max v u, u_3} D
    inst✝¹⁴ : CategoryTheory.Category.{max v u, u_4} E
    F : CategoryTheory.Functor D E
    inst✝¹³ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝¹⁰ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝⁹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁸ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
    inst✝ : (CategoryTheory.forget E).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    e_1✝ : Eq (Quiver.Hom (((CategoryTheory.whiskeringRight (Opposite C) D E).obj  …
    ⊢ Eq (((CategoryTheory.plusPlusAdjunction J E).homEquiv (((CategoryTheory.whis …
  -/
  dsimp [plusPlusAdjunction]
  /-
    case h.e'_3.h
    C : Type u
    inst✝¹⁶ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u_3
    E : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{max v u, u_3} D
    inst✝¹⁴ : CategoryTheory.Category.{max v u, u_4} E
    F : CategoryTheory.Functor D E
    inst✝¹³ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝¹⁰ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝⁹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁸ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
    inst✝ : (CategoryTheory.forget E).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    e_1✝ : Eq (Quiver.Hom (((CategoryTheory.whiskeringRight (Opposite C) D E).obj  …
    ⊢ Eq (((CategoryTheory.Adjunction.mkOfHomEquiv { homEquiv := fun P Q => { toFu …
  -/
  simp
  /-
    🎉 no goals
  -/


instance (P : Cᵒᵖ ⥤ D) :
    IsIso ((sheafComposeNatTrans J F (plusPlusAdjunction J D) (plusPlusAdjunction J E)).app P) := by
  rw [← isIso_iff_of_reflects_iso _ (sheafToPresheaf J E),
    sheafToPresheaf_map_sheafComposeNatTrans_eq_sheafifyCompIso_inv]
  /-
    C : Type u
    inst✝¹⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝¹⁷ : CategoryTheory.Category.{?u.111571, u_1} A
    inst✝¹⁶ : CategoryTheory.Category.{?u.111575, u_2} B
    F✝ : CategoryTheory.Functor A B
    D : Type u_3
    E : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{max v u, u_3} D
    inst✝¹⁴ : CategoryTheory.Category.{max v u, u_4} E
    F : CategoryTheory.Functor D E
    inst✝¹³ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝¹⁰ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝⁹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁸ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
    inst✝ : (CategoryTheory.forget E).ReflectsIsomorphisms
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ CategoryTheory.IsIso (J.sheafifyCompIso F P).inv
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : IsIso (sheafComposeNatTrans J F (plusPlusAdjunction J D) (plusPlusAdjunction J E)) :=
  NatIso.isIso_of_isIso_app _


instance : PreservesSheafification J F := by
  rw [preservesSheafification_iff_of_adjunctions_of_hasSheafCompose _ _
    (plusPlusAdjunction J D) (plusPlusAdjunction J E)]
  /-
    C : Type u
    inst✝¹⁸ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u_1
    B : Type u_2
    inst✝¹⁷ : CategoryTheory.Category.{?u.141608, u_1} A
    inst✝¹⁶ : CategoryTheory.Category.{?u.141612, u_2} B
    F✝ : CategoryTheory.Functor A B
    D : Type u_3
    E : Type u_4
    inst✝¹⁵ : CategoryTheory.Category.{max v u, u_3} D
    inst✝¹⁴ : CategoryTheory.Category.{max v u, u_4} E
    F : CategoryTheory.Functor D E
    inst✝¹³ : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹² : ∀ (α β : Type (max v u)) (fst snd : β → α), CategoryTheory.Limits.Ha …
    inst✝¹¹ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝¹⁰ : ∀ (X : C), CategoryTheory.Limits.HasColimitsOfShape (Opposite (J.Cov …
    inst✝⁹ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁸ : ∀ (X : C) (W : J.Cover X) (P : CategoryTheory.Functor (Opposite C) D) …
    inst✝⁷ : CategoryTheory.ConcreteCategory D
    inst✝⁶ : CategoryTheory.ConcreteCategory E
    inst✝⁵ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝⁴ : ∀ (X : C), CategoryTheory.Limits.PreservesColimitsOfShape (Opposite ( …
    inst✝³ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget D)
    inst✝² : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget E)
    inst✝¹ : (CategoryTheory.forget D).ReflectsIsomorphisms
    inst✝ : (CategoryTheory.forget E).ReflectsIsomorphisms
    ⊢ CategoryTheory.IsIso (CategoryTheory.sheafComposeNatTrans J F (CategoryTheor …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


