/-- A functor `F : J ⥤ C` is eventually constant to `j : J` if
for any map `f : i ⟶ j`, the induced morphism `F.map f` is an isomorphism.
If `J` is cofiltered, this implies `F` has a limit. -/
def IsEventuallyConstantTo (j : J) : Prop :=
  ∀ ⦃i : J⦄ (f : i ⟶ j), IsIso (F.map f)


/-- A functor `F : J ⥤ C` is eventually constant from `i : J` if
for any map `f : i ⟶ j`, the induced morphism `F.map f` is an isomorphism.
If `J` is filtered, this implies `F` has a colimit. -/
def IsEventuallyConstantFrom (i : J) : Prop :=
  ∀ ⦃j : J⦄ (f : i ⟶ j), IsIso (F.map f)


lemma isIso_map {i j : J} (φ : i ⟶ j) (π : j ⟶ i₀) : IsIso (F.map φ) := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
    inst✝ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantTo i₀
    i j : J
    φ : Quiver.Hom i j
    π : Quiver.Hom j i₀
    ⊢ CategoryTheory.IsIso (F.map φ)
  -/
  have := h π
  /-
    J : Type u_1
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
    inst✝ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantTo i₀
    i j : J
    φ : Quiver.Hom i j
    π : Quiver.Hom j i₀
    this : CategoryTheory.IsIso (F.map π)
    ⊢ CategoryTheory.IsIso (F.map φ)
  -/
  have := h (φ ≫ π)
  /-
    J : Type u_1
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
    inst✝ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantTo i₀
    i j : J
    φ : Quiver.Hom i j
    π : Quiver.Hom j i₀
    this✝ : CategoryTheory.IsIso (F.map π)
    this : CategoryTheory.IsIso (F.map (CategoryTheory.CategoryStruct.comp φ π))
    ⊢ CategoryTheory.IsIso (F.map φ)
  -/
  exact IsIso.of_isIso_fac_right (F.map_comp φ π).symm
  /-
    🎉 no goals
  -/


lemma precomp {j : J} (f : j ⟶ i₀) : F.IsEventuallyConstantTo j :=
  fun _ φ ↦ h.isIso_map φ f


/-- The isomorphism `F.obj i ≅ F.obj j` induced by `φ : i ⟶ j`,
when `h : F.IsEventuallyConstantTo i₀` and there exists a map `j ⟶ i₀`. -/
@[simps! hom]
noncomputable def isoMap : F.obj i ≅ F.obj j :=
  have := h.isIso_map φ hφ.some
  asIso (F.map φ)


@[reassoc (attr := simp)]
lemma isoMap_hom_inv_id : F.map φ ≫ (h.isoMap φ hφ).inv = 𝟙 _ :=
  (h.isoMap φ hφ).hom_inv_id


@[reassoc (attr := simp)]
lemma isoMap_inv_hom_id : (h.isoMap φ hφ).inv ≫ F.map φ = 𝟙 _ :=
  (h.isoMap φ hφ).inv_hom_id


/-- Auxiliary definition for `IsEventuallyConstantTo.cone`. -/
noncomputable def coneπApp (j : J) : F.obj i₀ ⟶ F.obj j :=
  (h.isoMap (minToLeft i₀ j) ⟨𝟙 _⟩).inv ≫ F.map (minToRight i₀ j)


lemma coneπApp_eq (j j' : J) (α : j' ⟶ i₀) (β : j' ⟶ j) :
    h.coneπApp j = (h.isoMap α ⟨𝟙 _⟩).inv ≫ F.map β := by
  obtain ⟨s, γ, δ, h₁, h₂⟩ := IsCofiltered.bowtie
    (IsCofiltered.minToRight i₀ j) β (IsCofiltered.minToLeft i₀ j) α
  /-
    case intro.intro.intro.intro
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantTo i₀
    inst✝ : CategoryTheory.IsCofiltered J
    j j' : J
    α : Quiver.Hom j' i₀
    β : Quiver.Hom j' j
    s : J
    γ : Quiver.Hom s (CategoryTheory.IsCofiltered.min i₀ j)
    δ : Quiver.Hom s j'
    h₁ : Eq (CategoryTheory.CategoryStruct.comp γ (CategoryTheory.IsCofiltered.min …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp γ (CategoryTheory.IsCofiltered.min …
    ⊢ Eq (h.coneπApp j) (CategoryTheory.CategoryStruct.comp (h.isoMap α ⋯).inv (F. …
  -/
  dsimp [coneπApp]
  rw [← cancel_epi ((h.isoMap α ⟨𝟙 _⟩).hom), isoMap_hom, isoMap_hom_inv_id_assoc,
    ← cancel_epi (h.isoMap δ ⟨α⟩).hom, isoMap_hom,
    ← F.map_comp δ β, ← h₁, F.map_comp, ← F.map_comp_assoc, ← h₂, F.map_comp_assoc,
    isoMap_hom_inv_id_assoc]


@[simp]
lemma coneπApp_eq_id : h.coneπApp i₀ = 𝟙 _ := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantTo i₀
    inst✝ : CategoryTheory.IsCofiltered J
    ⊢ Eq (h.coneπApp i₀) (CategoryTheory.CategoryStruct.id (F.obj i₀))
  -/
  rw [h.coneπApp_eq i₀ i₀ (𝟙 _) (𝟙 _), h.isoMap_inv_hom_id]
  /-
    🎉 no goals
  -/


/-- Given `h : F.IsEventuallyConstantTo i₀`, this is the (limit) cone for `F` whose
point is `F.obj i₀`. -/
@[simps]
noncomputable def cone : Cone F where
  pt := F.obj i₀
  π :=
    { app := h.coneπApp
      naturality := fun j j' φ ↦ by
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.10303, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.10307, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantTo i₀
          inst✝ : CategoryTheory.IsCofiltered J
          j j' : J
          φ : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.10303, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.10307, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantTo i₀
          inst✝ : CategoryTheory.IsCofiltered J
          j j' : J
          φ : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (F. …
        -/
        rw [id_comp]
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.10303, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.10307, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantTo i₀
          inst✝ : CategoryTheory.IsCofiltered J
          j j' : J
          φ : Quiver.Hom j j'
          ⊢ Eq (h.coneπApp j') (CategoryTheory.CategoryStruct.comp (h.coneπApp j) (F.map …
        -/
        let i := IsCofiltered.min i₀ j
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.10303, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.10307, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantTo i₀
          inst✝ : CategoryTheory.IsCofiltered J
          j j' : J
          φ : Quiver.Hom j j'
          i : J := CategoryTheory.IsCofiltered.min i₀ j
          ⊢ Eq (h.coneπApp j') (CategoryTheory.CategoryStruct.comp (h.coneπApp j) (F.map …
        -/
        let α : i ⟶ i₀ := IsCofiltered.minToLeft _ _
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.10303, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.10307, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantTo i₀
          inst✝ : CategoryTheory.IsCofiltered J
          j j' : J
          φ : Quiver.Hom j j'
          i : J := CategoryTheory.IsCofiltered.min i₀ j
          α : Quiver.Hom i i₀ := CategoryTheory.IsCofiltered.minToLeft i₀ j
          ⊢ Eq (h.coneπApp j') (CategoryTheory.CategoryStruct.comp (h.coneπApp j) (F.map …
        -/
        let β : i ⟶ j := IsCofiltered.minToRight _ _
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.10303, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.10307, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantTo i₀
          inst✝ : CategoryTheory.IsCofiltered J
          j j' : J
          φ : Quiver.Hom j j'
          i : J := CategoryTheory.IsCofiltered.min i₀ j
          α : Quiver.Hom i i₀ := CategoryTheory.IsCofiltered.minToLeft i₀ j
          β : Quiver.Hom i j := CategoryTheory.IsCofiltered.minToRight i₀ j
          ⊢ Eq (h.coneπApp j') (CategoryTheory.CategoryStruct.comp (h.coneπApp j) (F.map …
        -/
        rw [h.coneπApp_eq j _ α β, assoc, h.coneπApp_eq j' _ α (β ≫ φ), map_comp] }
        /-
          🎉 no goals
        -/


/-- When `h : F.IsEventuallyConstantTo i₀`, the limit of `F` exists and is `F.obj i₀`. -/
def isLimitCone : IsLimit h.cone where
  lift s := s.π.app i₀
  fac s j := by
    /-
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.12691, u_1} J
      inst✝¹ : CategoryTheory.Category.{?u.12695, u_2} C
      F : CategoryTheory.Functor J C
      i₀ : J
      h : F.IsEventuallyConstantTo i₀
      inst✝ : CategoryTheory.IsCofiltered J
      s : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => s.π.app i₀) s) (h.cone.π.a …
    -/
    dsimp [coneπApp]
    rw [← s.w (IsCofiltered.minToLeft i₀ j), ← s.w (IsCofiltered.minToRight i₀ j), assoc,
      isoMap_hom_inv_id_assoc]
                    /-
                      J : Type u_1
                      C : Type u_2
                      inst✝² : CategoryTheory.Category.{?u.12691, u_1} J
                      inst✝¹ : CategoryTheory.Category.{?u.12695, u_2} C
                      F : CategoryTheory.Functor J C
                      i₀ : J
                      h : F.IsEventuallyConstantTo i₀
                      inst✝ : CategoryTheory.IsCofiltered J
                      s : CategoryTheory.Limits.Cone F
                      m : Quiver.Hom s.pt h.cone.pt
                      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (h.cone.π.app j)) (s. …
                      ⊢ Eq m ((fun s => s.π.app i₀) s)
                    -/
  uniq s m hm := by simp only [← hm i₀, cone_π_app, coneπApp_eq_id, cone_pt, comp_id]
                    /-
                      🎉 no goals
                    -/


lemma hasLimit : HasLimit F := ⟨_, h.isLimitCone⟩


lemma isIso_π_of_isLimit {c : Cone F} (hc : IsLimit c) :
    IsIso (c.π.app i₀) := by
  simp only [← IsLimit.conePointUniqueUpToIso_hom_comp hc h.isLimitCone i₀,
    cone_π_app, coneπApp_eq_id, cone_pt, comp_id]
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantTo i₀
    inst✝ : CategoryTheory.IsCofiltered J
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    ⊢ CategoryTheory.IsIso (hc.conePointUniqueUpToIso h.isLimitCone).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- More general version of `isIso_π_of_isLimit`. -/
lemma isIso_π_of_isLimit' {c : Cone F} (hc : IsLimit c) (j : J) (π : j ⟶ i₀) :
    IsIso (c.π.app j) :=
  (h.precomp π).isIso_π_of_isLimit hc


lemma isIso_map {i j : J} (φ : i ⟶ j) (ι : i₀ ⟶ i) : IsIso (F.map φ) := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
    inst✝ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantFrom i₀
    i j : J
    φ : Quiver.Hom i j
    ι : Quiver.Hom i₀ i
    ⊢ CategoryTheory.IsIso (F.map φ)
  -/
  have := h ι
  /-
    J : Type u_1
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
    inst✝ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantFrom i₀
    i j : J
    φ : Quiver.Hom i j
    ι : Quiver.Hom i₀ i
    this : CategoryTheory.IsIso (F.map ι)
    ⊢ CategoryTheory.IsIso (F.map φ)
  -/
  have := h (ι ≫ φ)
  /-
    J : Type u_1
    C : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} J
    inst✝ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantFrom i₀
    i j : J
    φ : Quiver.Hom i j
    ι : Quiver.Hom i₀ i
    this✝ : CategoryTheory.IsIso (F.map ι)
    this : CategoryTheory.IsIso (F.map (CategoryTheory.CategoryStruct.comp ι φ))
    ⊢ CategoryTheory.IsIso (F.map φ)
  -/
  exact IsIso.of_isIso_fac_left (F.map_comp ι φ).symm
  /-
    🎉 no goals
  -/


lemma postcomp {j : J} (f : i₀ ⟶ j) : F.IsEventuallyConstantFrom j :=
  fun _ φ ↦ h.isIso_map φ f


/-- The isomorphism `F.obj i ≅ F.obj j` induced by `φ : i ⟶ j`,
when `h : F.IsEventuallyConstantFrom i₀` and there exists a map `i₀ ⟶ i`. -/
@[simps! hom]
noncomputable def isoMap : F.obj i ≅ F.obj j :=
  have := h.isIso_map φ hφ.some
  asIso (F.map φ)


/-- Auxiliary definition for `IsEventuallyConstantFrom.cocone`. -/
noncomputable def coconeιApp (j : J) : F.obj j ⟶ F.obj i₀ :=
  F.map (rightToMax i₀ j) ≫ (h.isoMap (leftToMax i₀ j) ⟨𝟙 _⟩).inv


lemma coconeιApp_eq (j j' : J) (α : j ⟶ j') (β : i₀ ⟶ j') :
    h.coconeιApp j = F.map α ≫ (h.isoMap β ⟨𝟙 _⟩).inv  := by
  obtain ⟨s, γ, δ, h₁, h₂⟩ := IsFiltered.bowtie
    (IsFiltered.leftToMax i₀ j) β (IsFiltered.rightToMax i₀ j) α
  /-
    case intro.intro.intro.intro
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantFrom i₀
    inst✝ : CategoryTheory.IsFiltered J
    j j' : J
    α : Quiver.Hom j j'
    β : Quiver.Hom i₀ j'
    s : J
    γ : Quiver.Hom (CategoryTheory.IsFiltered.max i₀ j) s
    δ : Quiver.Hom j' s
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.leftToM …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.IsFiltered.rightTo …
    ⊢ Eq (h.coconeιApp j) (CategoryTheory.CategoryStruct.comp (F.map α) (h.isoMap  …
  -/
  dsimp [coconeιApp]
  rw [← cancel_mono ((h.isoMap β ⟨𝟙 _⟩).hom), assoc, assoc, isoMap_hom, isoMap_inv_hom_id,
    comp_id, ← cancel_mono (h.isoMap δ ⟨β⟩).hom, isoMap_hom, assoc, assoc, ← F.map_comp α δ,
    ← h₂, F.map_comp, ← F.map_comp β δ, ← h₁, F.map_comp, isoMap_inv_hom_id_assoc]


@[simp]
lemma coconeιApp_eq_id : h.coconeιApp i₀ = 𝟙 _ := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_4, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_3, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantFrom i₀
    inst✝ : CategoryTheory.IsFiltered J
    ⊢ Eq (h.coconeιApp i₀) (CategoryTheory.CategoryStruct.id (F.obj i₀))
  -/
  rw [h.coconeιApp_eq i₀ i₀ (𝟙 _) (𝟙 _), h.isoMap_hom_inv_id]
  /-
    🎉 no goals
  -/


/-- Given `h : F.IsEventuallyConstantFrom i₀`, this is the (limit) cocone for `F` whose
point is `F.obj i₀`. -/
@[simps]
noncomputable def cocone : Cocone F where
  pt := F.obj i₀
  ι :=
    { app := h.coconeιApp
      naturality := fun j j' φ ↦ by
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.29867, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.29871, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantFrom i₀
          inst✝ : CategoryTheory.IsFiltered J
          j j' : J
          φ : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (h.coconeιApp j')) (Categor …
        -/
        dsimp
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.29867, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.29871, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantFrom i₀
          inst✝ : CategoryTheory.IsFiltered J
          j j' : J
          φ : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (h.coconeιApp j')) (Categor …
        -/
        rw [comp_id]
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.29867, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.29871, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantFrom i₀
          inst✝ : CategoryTheory.IsFiltered J
          j j' : J
          φ : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (h.coconeιApp j')) (h.cocon …
        -/
        let i := IsFiltered.max i₀ j'
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.29867, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.29871, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantFrom i₀
          inst✝ : CategoryTheory.IsFiltered J
          j j' : J
          φ : Quiver.Hom j j'
          i : J := CategoryTheory.IsFiltered.max i₀ j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (h.coconeιApp j')) (h.cocon …
        -/
        let α : i₀ ⟶ i := IsFiltered.leftToMax _ _
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.29867, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.29871, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantFrom i₀
          inst✝ : CategoryTheory.IsFiltered J
          j j' : J
          φ : Quiver.Hom j j'
          i : J := CategoryTheory.IsFiltered.max i₀ j'
          α : Quiver.Hom i₀ i := CategoryTheory.IsFiltered.leftToMax i₀ j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (h.coconeιApp j')) (h.cocon …
        -/
        let β : j' ⟶ i := IsFiltered.rightToMax _ _
        /-
          J : Type u_1
          C : Type u_2
          inst✝² : CategoryTheory.Category.{?u.29867, u_1} J
          inst✝¹ : CategoryTheory.Category.{?u.29871, u_2} C
          F : CategoryTheory.Functor J C
          i₀ : J
          h : F.IsEventuallyConstantFrom i₀
          inst✝ : CategoryTheory.IsFiltered J
          j j' : J
          φ : Quiver.Hom j j'
          i : J := CategoryTheory.IsFiltered.max i₀ j'
          α : Quiver.Hom i₀ i := CategoryTheory.IsFiltered.leftToMax i₀ j'
          β : Quiver.Hom j' i := CategoryTheory.IsFiltered.rightToMax i₀ j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (h.coconeιApp j')) (h.cocon …
        -/
        rw [h.coconeιApp_eq j' _ β α, h.coconeιApp_eq j _ (φ ≫ β) α, map_comp, assoc] }
        /-
          🎉 no goals
        -/


/-- When `h : F.IsEventuallyConstantFrom i₀`, the colimit of `F` exists and is `F.obj i₀`. -/
def isColimitCocone : IsColimit h.cocone where
  desc s := s.ι.app i₀
  fac s j := by
    /-
      J : Type u_1
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.32916, u_1} J
      inst✝¹ : CategoryTheory.Category.{?u.32920, u_2} C
      F : CategoryTheory.Functor J C
      i₀ : J
      h : F.IsEventuallyConstantFrom i₀
      inst✝ : CategoryTheory.IsFiltered J
      s : CategoryTheory.Limits.Cocone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.cocone.ι.app j) ((fun s => s.ι.app …
    -/
    dsimp [coconeιApp]
    rw [← s.w (IsFiltered.rightToMax i₀ j), ← s.w (IsFiltered.leftToMax i₀ j), assoc,
      isoMap_inv_hom_id_assoc]
                    /-
                      J : Type u_1
                      C : Type u_2
                      inst✝² : CategoryTheory.Category.{?u.32916, u_1} J
                      inst✝¹ : CategoryTheory.Category.{?u.32920, u_2} C
                      F : CategoryTheory.Functor J C
                      i₀ : J
                      h : F.IsEventuallyConstantFrom i₀
                      inst✝ : CategoryTheory.IsFiltered J
                      s : CategoryTheory.Limits.Cocone F
                      m : Quiver.Hom h.cocone.pt s.pt
                      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (h.cocone.ι.app j) m) ( …
                      ⊢ Eq m ((fun s => s.ι.app i₀) s)
                    -/
  uniq s m hm := by simp only [← hm i₀, cocone_ι_app, coconeιApp_eq_id, id_comp]
                    /-
                      🎉 no goals
                    -/


lemma hasColimit : HasColimit F := ⟨_, h.isColimitCocone⟩


lemma isIso_ι_of_isColimit {c : Cocone F} (hc : IsColimit c) :
    IsIso (c.ι.app i₀) := by
  simp only [← IsColimit.comp_coconePointUniqueUpToIso_inv hc h.isColimitCocone i₀,
    cocone_ι_app, coconeιApp_eq_id, id_comp]
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    i₀ : J
    h : F.IsEventuallyConstantFrom i₀
    inst✝ : CategoryTheory.IsFiltered J
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.IsIso (hc.coconePointUniqueUpToIso h.isColimitCocone).inv
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- More general version of `isIso_ι_of_isColimit`. -/
lemma isIso_ι_of_isColimit' {c : Cocone F} (hc : IsColimit c) (j : J) (ι : i₀ ⟶ j) :
    IsIso (c.ι.app j) :=
  (h.postcomp ι).isIso_ι_of_isColimit hc


/-- A functor `F : J ⥤ C` from a cofiltered category is eventually constant if there
exists `j : J`, such that for any `f : i ⟶ j`, the induced map `F.map f` is an isomorphism. -/
class IsEventuallyConstant : Prop where
  exists_isEventuallyConstantTo : ∃ (j : J), F.IsEventuallyConstantTo j


instance [hF : IsEventuallyConstant F] [IsCofiltered J] : HasLimit F := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    hF : CategoryTheory.IsCofiltered.IsEventuallyConstant F
    inst✝ : CategoryTheory.IsCofiltered J
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  obtain ⟨j, h⟩ := hF.exists_isEventuallyConstantTo
  /-
    case intro
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    hF : CategoryTheory.IsCofiltered.IsEventuallyConstant F
    inst✝ : CategoryTheory.IsCofiltered J
    j : J
    h : F.IsEventuallyConstantTo j
    ⊢ CategoryTheory.Limits.HasLimit F
  -/
  exact h.hasLimit
  /-
    🎉 no goals
  -/


/-- A functor `F : J ⥤ C` from a filtered category is eventually constant if there
exists `i : J`, such that for any `f : i ⟶ j`, the induced map `F.map f` is an isomorphism. -/
class IsEventuallyConstant : Prop where
  exists_isEventuallyConstantFrom : ∃ (i : J), F.IsEventuallyConstantFrom i


instance [hF : IsEventuallyConstant F] [IsFiltered J] : HasColimit F := by
  /-
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    hF : CategoryTheory.IsFiltered.IsEventuallyConstant F
    inst✝ : CategoryTheory.IsFiltered J
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  obtain ⟨j, h⟩ := hF.exists_isEventuallyConstantFrom
  /-
    case intro
    J : Type u_1
    C : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} J
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} C
    F : CategoryTheory.Functor J C
    hF : CategoryTheory.IsFiltered.IsEventuallyConstant F
    inst✝ : CategoryTheory.IsFiltered J
    j : J
    h : F.IsEventuallyConstantFrom j
    ⊢ CategoryTheory.Limits.HasColimit F
  -/
  exact h.hasColimit
  /-
    🎉 no goals
  -/


