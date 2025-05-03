/-- A morphism `φ : S₁ ⟶ S₂` of short complexes that have homology is a quasi-isomorphism if
the induced map `homologyMap φ : S₁.homology ⟶ S₂.homology` is an isomorphism. -/
class QuasiIso (φ : S₁ ⟶ S₂) : Prop where
  /-- the homology map is an isomorphism -/
  isIso' : IsIso (homologyMap φ)


instance QuasiIso.isIso (φ : S₁ ⟶ S₂) [QuasiIso φ] : IsIso (homologyMap φ) := QuasiIso.isIso'


lemma quasiIso_iff (φ : S₁ ⟶ S₂) :
    QuasiIso φ ↔ IsIso (homologyMap φ) := by
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso (Category …
  -/
  constructor
    /-
      case mp
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      ⊢ CategoryTheory.ShortComplex.QuasiIso φ → CategoryTheory.IsIso (CategoryTheor …
    -/
  · intro h
    /-
      case mp
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.QuasiIso φ
      ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ) → CategoryT …
    -/
  · intro h
    /-
      case mpr
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
      ⊢ CategoryTheory.ShortComplex.QuasiIso φ
    -/
    exact ⟨h⟩
    /-
      🎉 no goals
    -/


instance quasiIso_of_isIso (φ : S₁ ⟶ S₂) [IsIso φ] : QuasiIso φ :=
  ⟨(homologyMapIso (asIso φ)).isIso_hom⟩


instance quasiIso_comp (φ : S₁ ⟶ S₂) (φ' : S₂ ⟶ S₃) [hφ : QuasiIso φ] [hφ' : QuasiIso φ'] :
    QuasiIso (φ ≫ φ') := by
  /-
    C : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_1, u_2} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    inst✝³ : S₁.HasHomology
    inst✝² : S₂.HasHomology
    inst✝¹ : S₃.HasHomology
    inst✝ : S₄.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ : CategoryTheory.ShortComplex.QuasiIso φ
    hφ' : CategoryTheory.ShortComplex.QuasiIso φ'
    ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
  -/
  rw [quasiIso_iff] at hφ hφ' ⊢
  /-
    C : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_1, u_2} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    inst✝³ : S₁.HasHomology
    inst✝² : S₂.HasHomology
    inst✝¹ : S₃.HasHomology
    inst✝ : S₄.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
    hφ' : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ')
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap (CategoryTheor …
  -/
  rw [homologyMap_comp]
  /-
    C : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_1, u_2} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    inst✝³ : S₁.HasHomology
    inst✝² : S₂.HasHomology
    inst✝¹ : S₃.HasHomology
    inst✝ : S₄.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
    hφ' : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ')
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Sho …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIso_of_comp_left (φ : S₁ ⟶ S₂) (φ' : S₂ ⟶ S₃)
    [hφ : QuasiIso φ] [hφφ' : QuasiIso (φ ≫ φ')] :
    QuasiIso φ' := by
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : S₃.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ : CategoryTheory.ShortComplex.QuasiIso φ
    hφφ' : CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.com …
    ⊢ CategoryTheory.ShortComplex.QuasiIso φ'
  -/
  rw [quasiIso_iff] at hφ hφφ' ⊢
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : S₃.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
    hφφ' : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap (Category …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ')
  -/
  rw [homologyMap_comp] at hφφ'
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : S₃.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
    hφφ' : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheor …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ')
  -/
  exact IsIso.of_isIso_comp_left (homologyMap φ) (homologyMap φ')
  /-
    🎉 no goals
  -/


lemma quasiIso_iff_comp_left (φ : S₁ ⟶ S₂) (φ' : S₂ ⟶ S₃) [hφ : QuasiIso φ] :
    QuasiIso (φ ≫ φ') ↔ QuasiIso φ' := by
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : S₃.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.com …
  -/
  constructor
    /-
      case mp
      C : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝² : S₁.HasHomology
      inst✝¹ : S₂.HasHomology
      inst✝ : S₃.HasHomology
      φ : Quiver.Hom S₁ S₂
      φ' : Quiver.Hom S₂ S₃
      hφ : CategoryTheory.ShortComplex.QuasiIso φ
      ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.comp φ φ …
    -/
  · intro
    /-
      case mp
      C : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝² : S₁.HasHomology
      inst✝¹ : S₂.HasHomology
      inst✝ : S₃.HasHomology
      φ : Quiver.Hom S₁ S₂
      φ' : Quiver.Hom S₂ S₃
      hφ : CategoryTheory.ShortComplex.QuasiIso φ
      a✝ : CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.comp  …
      ⊢ CategoryTheory.ShortComplex.QuasiIso φ'
    -/
    exact quasiIso_of_comp_left φ φ'
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝² : S₁.HasHomology
      inst✝¹ : S₂.HasHomology
      inst✝ : S₃.HasHomology
      φ : Quiver.Hom S₁ S₂
      φ' : Quiver.Hom S₂ S₃
      hφ : CategoryTheory.ShortComplex.QuasiIso φ
      ⊢ CategoryTheory.ShortComplex.QuasiIso φ' → CategoryTheory.ShortComplex.QuasiI …
    -/
  · intro
    /-
      case mpr
      C : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝² : S₁.HasHomology
      inst✝¹ : S₂.HasHomology
      inst✝ : S₃.HasHomology
      φ : Quiver.Hom S₁ S₂
      φ' : Quiver.Hom S₂ S₃
      hφ : CategoryTheory.ShortComplex.QuasiIso φ
      a✝ : CategoryTheory.ShortComplex.QuasiIso φ'
      ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
    -/
    exact quasiIso_comp φ φ'
    /-
      🎉 no goals
    -/


lemma quasiIso_of_comp_right (φ : S₁ ⟶ S₂) (φ' : S₂ ⟶ S₃)
    [hφ' : QuasiIso φ'] [hφφ' : QuasiIso (φ ≫ φ')] :
    QuasiIso φ := by
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : S₃.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ' : CategoryTheory.ShortComplex.QuasiIso φ'
    hφφ' : CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.com …
    ⊢ CategoryTheory.ShortComplex.QuasiIso φ
  -/
  rw [quasiIso_iff] at hφ' hφφ' ⊢
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : S₃.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ' : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ')
    hφφ' : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap (Category …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  rw [homologyMap_comp] at hφφ'
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : S₃.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ' : CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ')
    hφφ' : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheor …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.homologyMap φ)
  -/
  exact IsIso.of_isIso_comp_right (homologyMap φ) (homologyMap φ')
  /-
    🎉 no goals
  -/


lemma quasiIso_iff_comp_right (φ : S₁ ⟶ S₂) (φ' : S₂ ⟶ S₃) [hφ' : QuasiIso φ'] :
    QuasiIso (φ ≫ φ') ↔ QuasiIso φ := by
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : S₃.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₂ S₃
    hφ' : CategoryTheory.ShortComplex.QuasiIso φ'
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.com …
  -/
  constructor
    /-
      case mp
      C : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝² : S₁.HasHomology
      inst✝¹ : S₂.HasHomology
      inst✝ : S₃.HasHomology
      φ : Quiver.Hom S₁ S₂
      φ' : Quiver.Hom S₂ S₃
      hφ' : CategoryTheory.ShortComplex.QuasiIso φ'
      ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.comp φ φ …
    -/
  · intro
    /-
      case mp
      C : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝² : S₁.HasHomology
      inst✝¹ : S₂.HasHomology
      inst✝ : S₃.HasHomology
      φ : Quiver.Hom S₁ S₂
      φ' : Quiver.Hom S₂ S₃
      hφ' : CategoryTheory.ShortComplex.QuasiIso φ'
      a✝ : CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.comp  …
      ⊢ CategoryTheory.ShortComplex.QuasiIso φ
    -/
    exact quasiIso_of_comp_right φ φ'
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝² : S₁.HasHomology
      inst✝¹ : S₂.HasHomology
      inst✝ : S₃.HasHomology
      φ : Quiver.Hom S₁ S₂
      φ' : Quiver.Hom S₂ S₃
      hφ' : CategoryTheory.ShortComplex.QuasiIso φ'
      ⊢ CategoryTheory.ShortComplex.QuasiIso φ → CategoryTheory.ShortComplex.QuasiIs …
    -/
  · intro
    /-
      case mpr
      C : Type u_2
      inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      inst✝² : S₁.HasHomology
      inst✝¹ : S₂.HasHomology
      inst✝ : S₃.HasHomology
      φ : Quiver.Hom S₁ S₂
      φ' : Quiver.Hom S₂ S₃
      hφ' : CategoryTheory.ShortComplex.QuasiIso φ'
      a✝ : CategoryTheory.ShortComplex.QuasiIso φ
      ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.CategoryStruct.comp φ φ')
    -/
    exact quasiIso_comp φ φ'
    /-
      🎉 no goals
    -/


lemma quasiIso_of_arrow_mk_iso (φ : S₁ ⟶ S₂) (φ' : S₃ ⟶ S₄) (e : Arrow.mk φ ≅ Arrow.mk φ')
    [hφ : QuasiIso φ] : QuasiIso φ' := by
  /-
    C : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_1, u_2} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    inst✝³ : S₁.HasHomology
    inst✝² : S₂.HasHomology
    inst✝¹ : S₃.HasHomology
    inst✝ : S₄.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₃ S₄
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk φ) (CategoryTheory.Arrow.mk φ')
    hφ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ CategoryTheory.ShortComplex.QuasiIso φ'
  -/
  let α : S₃ ⟶ S₁ := e.inv.left
  /-
    C : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_1, u_2} C
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
    inst✝³ : S₁.HasHomology
    inst✝² : S₂.HasHomology
    inst✝¹ : S₃.HasHomology
    inst✝ : S₄.HasHomology
    φ : Quiver.Hom S₁ S₂
    φ' : Quiver.Hom S₃ S₄
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk φ) (CategoryTheory.Arrow.mk φ')
    hφ : CategoryTheory.ShortComplex.QuasiIso φ
    α : Quiver.Hom S₃ S₁ := e.inv.left
    ⊢ CategoryTheory.ShortComplex.QuasiIso φ'
  -/
  let β : S₂ ⟶ S₄ := e.hom.right
  suffices φ' = α ≫ φ ≫ β by
    rw [this]
    infer_instance
  simp only [α, β, Arrow.w_mk_right_assoc, Arrow.mk_left, Arrow.mk_right, Arrow.mk_hom,
    ← Arrow.comp_right, e.inv_hom_id, Arrow.id_right, comp_id]


lemma quasiIso_iff_of_arrow_mk_iso (φ : S₁ ⟶ S₂) (φ' : S₃ ⟶ S₄) (e : Arrow.mk φ ≅ Arrow.mk φ') :
    QuasiIso φ ↔ QuasiIso φ' :=
  ⟨fun _ => quasiIso_of_arrow_mk_iso φ φ' e, fun _ => quasiIso_of_arrow_mk_iso φ' φ e.symm⟩


lemma LeftHomologyMapData.quasiIso_iff {φ : S₁ ⟶ S₂} {h₁ : S₁.LeftHomologyData}
    {h₂ : S₂.LeftHomologyData} (γ : LeftHomologyMapData φ h₁ h₂) :
    QuasiIso φ ↔ IsIso γ.φH := by
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso γ.φH)
  -/
  rw [ShortComplex.quasiIso_iff, γ.homologyMap_eq]
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Iff (CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp h₁.homologyIso …
  -/
  constructor
    /-
      case mp
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.LeftHomologyData
      h₂ : S₂.LeftHomologyData
      γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp h₁.homologyIso.hom  …
    -/
  · intro h
    haveI : IsIso (γ.φH ≫ (LeftHomologyData.homologyIso h₂).inv) :=
      IsIso.of_isIso_comp_left (LeftHomologyData.homologyIso h₁).hom _
    /-
      case mp
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.LeftHomologyData
      h₂ : S₂.LeftHomologyData
      γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
      h : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp h₁.homologyIso.ho …
      this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp γ.φH h₂.homolo …
      ⊢ CategoryTheory.IsIso γ.φH
    -/
    exact IsIso.of_isIso_comp_right _ (LeftHomologyData.homologyIso h₂).inv
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.LeftHomologyData
      h₂ : S₂.LeftHomologyData
      γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
      ⊢ CategoryTheory.IsIso γ.φH → CategoryTheory.IsIso (CategoryTheory.CategoryStr …
    -/
  · intro h
    /-
      case mpr
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.LeftHomologyData
      h₂ : S₂.LeftHomologyData
      γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
      h : CategoryTheory.IsIso γ.φH
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp h₁.homologyIso.hom  …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma RightHomologyMapData.quasiIso_iff {φ : S₁ ⟶ S₂} {h₁ : S₁.RightHomologyData}
    {h₂ : S₂.RightHomologyData} (γ : RightHomologyMapData φ h₁ h₂) :
    QuasiIso φ ↔ IsIso γ.φH := by
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso γ.φH)
  -/
  rw [ShortComplex.quasiIso_iff, γ.homologyMap_eq]
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Iff (CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp h₁.homologyIso …
  -/
  constructor
    /-
      case mp
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp h₁.homologyIso.hom  …
    -/
  · intro h
    haveI : IsIso (γ.φH ≫ (RightHomologyData.homologyIso h₂).inv) :=
      IsIso.of_isIso_comp_left (RightHomologyData.homologyIso h₁).hom _
    /-
      case mp
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
      h : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp h₁.homologyIso.ho …
      this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp γ.φH h₂.homolo …
      ⊢ CategoryTheory.IsIso γ.φH
    -/
    exact IsIso.of_isIso_comp_right _ (RightHomologyData.homologyIso h₂).inv
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
      ⊢ CategoryTheory.IsIso γ.φH → CategoryTheory.IsIso (CategoryTheory.CategoryStr …
    -/
  · intro h
    /-
      case mpr
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
      h : CategoryTheory.IsIso γ.φH
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp h₁.homologyIso.hom  …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma quasiIso_iff_isIso_leftHomologyMap' (φ : S₁ ⟶ S₂)
    (h₁ : S₁.LeftHomologyData) (h₂ : S₂.LeftHomologyData) :
    QuasiIso φ ↔ IsIso (leftHomologyMap' φ h₁ h₂) := by
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso (Category …
  -/
  have γ : LeftHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso (Category …
  -/
  rw [γ.quasiIso_iff, γ.leftHomologyMap'_eq]
  /-
    🎉 no goals
  -/


lemma quasiIso_iff_isIso_rightHomologyMap' (φ : S₁ ⟶ S₂)
    (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    QuasiIso φ ↔ IsIso (rightHomologyMap' φ h₁ h₂) := by
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso (Category …
  -/
  have γ : RightHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso (Category …
  -/
  rw [γ.quasiIso_iff, γ.rightHomologyMap'_eq]
  /-
    🎉 no goals
  -/


lemma quasiIso_iff_isIso_homologyMap' (φ : S₁ ⟶ S₂)
    (h₁ : S₁.HomologyData) (h₂ : S₂.HomologyData) :
    QuasiIso φ ↔ IsIso (homologyMap' φ h₁ h₂) :=
  quasiIso_iff_isIso_leftHomologyMap' _ _ _


lemma quasiIso_of_epi_of_isIso_of_mono (φ : S₁ ⟶ S₂) [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    QuasiIso φ := by
  /-
    C : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_1, u_2} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝⁴ : S₁.HasHomology
    inst✝³ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.ShortComplex.QuasiIso φ
  -/
  rw [((LeftHomologyMapData.ofEpiOfIsIsoOfMono φ) S₁.leftHomologyData).quasiIso_iff]
  /-
    C : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_1, u_2} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝⁴ : S₁.HasHomology
    inst✝³ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.LeftHomologyMapData.ofEpiO …
  -/
  dsimp
  /-
    C : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_1, u_2} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝⁴ : S₁.HasHomology
    inst✝³ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.id S₁.leftHomologyData.H)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIso_opMap_iff (φ : S₁ ⟶ S₂) :
    QuasiIso (opMap φ) ↔ QuasiIso φ := by
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.ShortComplex.opMap …
  -/
  have γ : HomologyMapData φ S₁.homologyData S₂.homologyData := default
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    γ : CategoryTheory.ShortComplex.HomologyMapData φ S₁.homologyData S₂.homologyD …
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.ShortComplex.opMap …
  -/
  rw [γ.left.quasiIso_iff, γ.op.right.quasiIso_iff]
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    γ : CategoryTheory.ShortComplex.HomologyMapData φ S₁.homologyData S₂.homologyD …
    ⊢ Iff (CategoryTheory.IsIso γ.op.right.φH) (CategoryTheory.IsIso γ.left.φH)
  -/
  dsimp
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    γ : CategoryTheory.ShortComplex.HomologyMapData φ S₁.homologyData S₂.homologyD …
    ⊢ Iff (CategoryTheory.IsIso γ.left.φH.op) (CategoryTheory.IsIso γ.left.φH)
  -/
  constructor
    /-
      case mp
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      γ : CategoryTheory.ShortComplex.HomologyMapData φ S₁.homologyData S₂.homologyD …
      ⊢ CategoryTheory.IsIso γ.left.φH.op → CategoryTheory.IsIso γ.left.φH
    -/
  · intro h
    /-
      case mp
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      γ : CategoryTheory.ShortComplex.HomologyMapData φ S₁.homologyData S₂.homologyD …
      h : CategoryTheory.IsIso γ.left.φH.op
      ⊢ CategoryTheory.IsIso γ.left.φH
    -/
    apply isIso_of_op
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      γ : CategoryTheory.ShortComplex.HomologyMapData φ S₁.homologyData S₂.homologyD …
      ⊢ CategoryTheory.IsIso γ.left.φH → CategoryTheory.IsIso γ.left.φH.op
    -/
  · intro h
    /-
      case mpr
      C : Type u_2
      inst✝³ : CategoryTheory.Category.{u_1, u_2} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      inst✝¹ : S₁.HasHomology
      inst✝ : S₂.HasHomology
      φ : Quiver.Hom S₁ S₂
      γ : CategoryTheory.ShortComplex.HomologyMapData φ S₁.homologyData S₂.homologyD …
      h : CategoryTheory.IsIso γ.left.φH
      ⊢ CategoryTheory.IsIso γ.left.φH.op
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma quasiIso_opMap (φ : S₁ ⟶ S₂) [QuasiIso φ] :
    QuasiIso (opMap φ) := by
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.ShortComplex.opMap φ)
  -/
  rw [quasiIso_opMap_iff]
  /-
    C : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_1, u_2} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ CategoryTheory.ShortComplex.QuasiIso φ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIso_unopMap {S₁ S₂ : ShortComplex Cᵒᵖ} [S₁.HasHomology] [S₂.HasHomology]
    [S₁.unop.HasHomology] [S₂.unop.HasHomology]
    (φ : S₁ ⟶ S₂) [QuasiIso φ] : QuasiIso (unopMap φ) := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
    inst✝⁴ : S₁.HasHomology
    inst✝³ : S₂.HasHomology
    inst✝² : S₁.unop.HasHomology
    inst✝¹ : S₂.unop.HasHomology
    φ : Quiver.Hom S₁ S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.ShortComplex.unopMap φ)
  -/
  rw [← quasiIso_opMap_iff]
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
    inst✝⁴ : S₁.HasHomology
    inst✝³ : S₂.HasHomology
    inst✝² : S₁.unop.HasHomology
    inst✝¹ : S₂.unop.HasHomology
    φ : Quiver.Hom S₁ S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ CategoryTheory.ShortComplex.QuasiIso (CategoryTheory.ShortComplex.opMap (Cat …
  -/
  change QuasiIso φ
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
    inst✝⁴ : S₁.HasHomology
    inst✝³ : S₂.HasHomology
    inst✝² : S₁.unop.HasHomology
    inst✝¹ : S₂.unop.HasHomology
    φ : Quiver.Hom S₁ S₂
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ CategoryTheory.ShortComplex.QuasiIso φ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma quasiIso_iff_isIso_liftCycles (φ : S₁ ⟶ S₂)
    (hf₁ : S₁.f = 0) (hg₁ : S₁.g = 0) (hf₂ : S₂.f = 0) :
                                               /-
                                                 C : Type ?u.28344
                                                 inst✝⁵ : CategoryTheory.Category.{?u.28348, ?u.28344} C
                                                 inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                                 S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                                 inst✝³ : S₁.HasHomology
                                                 inst✝² : S₂.HasHomology
                                                 inst✝¹ : S₃.HasHomology
                                                 inst✝ : S₄.HasHomology
                                                 φ : Quiver.Hom S₁ S₂
                                                 hf₁ : Eq S₁.f 0
                                                 hg₁ : Eq S₁.g 0
                                                 hf₂ : Eq S₂.f 0
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
                                               -/
    QuasiIso φ ↔ IsIso (S₂.liftCycles φ.τ₂ (by rw [φ.comm₂₃, hg₁, zero_comp])) := by
                                               /-
                                                 🎉 no goals
                                               -/
  let H : LeftHomologyMapData φ (LeftHomologyData.ofZeros S₁ hf₁ hg₁)
      (LeftHomologyData.ofIsLimitKernelFork S₂ hf₂ _ S₂.cyclesIsKernel) :=
    { φK := S₂.liftCycles φ.τ₂ (by rw [φ.comm₂₃, hg₁, zero_comp])
      φH := S₂.liftCycles φ.τ₂ (by rw [φ.comm₂₃, hg₁, zero_comp]) }
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    hf₁ : Eq S₁.f 0
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    H : CategoryTheory.ShortComplex.LeftHomologyMapData φ (CategoryTheory.ShortCom …
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso (S₂.liftC …
  -/
  exact H.quasiIso_iff
  /-
    🎉 no goals
  -/


lemma quasiIso_iff_isIso_descOpcycles (φ : S₁ ⟶ S₂)
    (hg₁ : S₁.g = 0) (hf₂ : S₂.f = 0) (hg₂ : S₂.g = 0) :
                                                 /-
                                                   C : Type ?u.31679
                                                   inst✝⁵ : CategoryTheory.Category.{?u.31683, ?u.31679} C
                                                   inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                                   S₁ S₂ S₃ S₄ : CategoryTheory.ShortComplex C
                                                   inst✝³ : S₁.HasHomology
                                                   inst✝² : S₂.HasHomology
                                                   inst✝¹ : S₃.HasHomology
                                                   inst✝ : S₄.HasHomology
                                                   φ : Quiver.Hom S₁ S₂
                                                   hg₁ : Eq S₁.g 0
                                                   hf₂ : Eq S₂.f 0
                                                   hg₂ : Eq S₂.g 0
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp S₁.f φ.τ₂) 0
                                                 -/
    QuasiIso φ ↔ IsIso (S₁.descOpcycles φ.τ₂ (by rw [← φ.comm₁₂, hf₂, comp_zero])) := by
                                                 /-
                                                   🎉 no goals
                                                 -/
  let H : RightHomologyMapData φ
      (RightHomologyData.ofIsColimitCokernelCofork S₁ hg₁ _ S₁.opcyclesIsCokernel)
        (RightHomologyData.ofZeros S₂ hf₂ hg₂) :=
    { φQ := S₁.descOpcycles φ.τ₂ (by rw [← φ.comm₁₂, hf₂, comp_zero])
      φH := S₁.descOpcycles φ.τ₂ (by rw [← φ.comm₁₂, hf₂, comp_zero]) }
  /-
    C : Type u_2
    inst✝³ : CategoryTheory.Category.{u_1, u_2} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    φ : Quiver.Hom S₁ S₂
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    hg₂ : Eq S₂.g 0
    H : CategoryTheory.ShortComplex.RightHomologyMapData φ (CategoryTheory.ShortCo …
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (CategoryTheory.IsIso (S₁.descO …
  -/
  exact H.quasiIso_iff
  /-
    🎉 no goals
  -/


