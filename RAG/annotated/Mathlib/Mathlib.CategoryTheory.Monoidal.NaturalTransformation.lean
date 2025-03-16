/-- A natural transformation between (lax) monoidal functors is monoidal if it satisfies
`ε F ≫ τ.app (𝟙_ C) = ε G` and `μ F X Y ≫ app (X ⊗ Y) = (app X ⊗ app Y) ≫ μ G X Y`. -/
class IsMonoidal : Prop where
  unit : ε F₁ ≫ τ.app (𝟙_ C) = ε F₂ := by aesop_cat
  tensor (X Y : C) : μ F₁ _ _ ≫ τ.app (X ⊗ Y) = (τ.app X ⊗ τ.app Y) ≫ μ F₂ _ _ := by aesop_cat


attribute [reassoc (attr := simp)] unit tensor


instance id : IsMonoidal (𝟙 F₁) where


instance comp (τ' : F₂ ⟶ F₃) [IsMonoidal τ] [IsMonoidal τ'] :
    IsMonoidal (τ ≫ τ') where


instance hcomp {G₁ G₂ : D ⥤ E} [G₁.LaxMonoidal] [G₂.LaxMonoidal] (τ' : G₁ ⟶ G₂)
    [IsMonoidal τ] [IsMonoidal τ'] : IsMonoidal (τ ◫ τ') where
  unit := by
    /-
      C : Type u₁
      inst✝¹⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹¹ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝¹⁰ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁹ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁸ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁷ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁶ : F₁.LaxMonoidal
      inst✝⁵ : F₂.LaxMonoidal
      inst✝⁴ : F₃.LaxMonoidal
      G₁ G₂ : CategoryTheory.Functor D E
      inst✝³ : G₁.LaxMonoidal
      inst✝² : G₂.LaxMonoidal
      τ' : Quiver.Hom G₁ G₂
      inst✝¹ : CategoryTheory.NatTrans.IsMonoidal τ
      inst✝ : CategoryTheory.NatTrans.IsMonoidal τ'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
    -/
    simp only [comp_obj, comp_ε, hcomp_app, assoc, naturality_assoc, unit_assoc, ← map_comp, unit]
    /-
      🎉 no goals
    -/
  tensor X Y := by
    simp only [comp_obj, comp_μ, hcomp_app, assoc, naturality_assoc,
      tensor_assoc, tensor_comp, μ_natural_assoc]
    /-
      C : Type u₁
      inst✝¹⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹³ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹¹ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝¹⁰ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁹ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁸ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁷ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁶ : F₁.LaxMonoidal
      inst✝⁵ : F₂.LaxMonoidal
      inst✝⁴ : F₃.LaxMonoidal
      G₁ G₂ : CategoryTheory.Functor D E
      inst✝³ : G₁.LaxMonoidal
      inst✝² : G₂.LaxMonoidal
      τ' : Quiver.Hom G₁ G₂
      inst✝¹ : CategoryTheory.NatTrans.IsMonoidal τ
      inst✝ : CategoryTheory.NatTrans.IsMonoidal τ'
      X Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
    -/
    simp only [← map_comp, tensor]
    /-
      🎉 no goals
    -/


instance (F : C ⥤ D) [F.LaxMonoidal] : NatTrans.IsMonoidal F.leftUnitor.hom where


instance (F : C ⥤ D) [F.LaxMonoidal] : NatTrans.IsMonoidal F.rightUnitor.hom where


instance (F : C ⥤ D) (G : D ⥤ E) (H : E ⥤ E') [F.LaxMonoidal] [G.LaxMonoidal] [H.LaxMonoidal] :
    NatTrans.IsMonoidal (Functor.associator F G H).hom where
  unit := by
    simp only [comp_obj, comp_ε, assoc, Functor.map_comp, associator_hom_app, comp_id,
      Functor.comp_map]
  tensor X Y := by
    simp only [comp_obj, comp_μ, associator_hom_app, Functor.comp_map, map_comp,
      comp_id, tensorHom_id, id_whiskerRight, assoc, id_comp]


instance {F G : C ⥤ D} {H K : C ⥤ E} (α : F ⟶ G) (β : H ⟶ K)
    [F.LaxMonoidal] [G.LaxMonoidal] [IsMonoidal α]
    [H.LaxMonoidal] [K.LaxMonoidal] [IsMonoidal β] :
    IsMonoidal (NatTrans.prod' α β) where
  unit := by
    /-
      C : Type u₁
      inst✝¹⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝¹² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹¹ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝¹⁰ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁹ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁸ : F₁.LaxMonoidal
      inst✝⁷ : F₂.LaxMonoidal
      inst✝⁶ : F₃.LaxMonoidal
      F G : CategoryTheory.Functor C D
      H K : CategoryTheory.Functor C E
      α : Quiver.Hom F G
      β : Quiver.Hom H K
      inst✝⁵ : F.LaxMonoidal
      inst✝⁴ : G.LaxMonoidal
      inst✝³ : CategoryTheory.NatTrans.IsMonoidal α
      inst✝² : H.LaxMonoidal
      inst✝¹ : K.LaxMonoidal
      inst✝ : CategoryTheory.NatTrans.IsMonoidal β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
    -/
    ext
      /-
        case h₁
        C : Type u₁
        inst✝¹⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹⁵ : CategoryTheory.MonoidalCategory C
        D : Type u₂
        inst✝¹⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹³ : CategoryTheory.MonoidalCategory D
        E : Type u₃
        inst✝¹² : CategoryTheory.Category.{v₃, u₃} E
        inst✝¹¹ : CategoryTheory.MonoidalCategory E
        E' : Type u₄
        inst✝¹⁰ : CategoryTheory.Category.{v₄, u₄} E'
        inst✝⁹ : CategoryTheory.MonoidalCategory E'
        F₁ F₂ F₃ : CategoryTheory.Functor C D
        τ : Quiver.Hom F₁ F₂
        inst✝⁸ : F₁.LaxMonoidal
        inst✝⁷ : F₂.LaxMonoidal
        inst✝⁶ : F₃.LaxMonoidal
        F G : CategoryTheory.Functor C D
        H K : CategoryTheory.Functor C E
        α : Quiver.Hom F G
        β : Quiver.Hom H K
        inst✝⁵ : F.LaxMonoidal
        inst✝⁴ : G.LaxMonoidal
        inst✝³ : CategoryTheory.NatTrans.IsMonoidal α
        inst✝² : H.LaxMonoidal
        inst✝¹ : K.LaxMonoidal
        inst✝ : CategoryTheory.NatTrans.IsMonoidal β
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
      -/
    · rw [prod_comp_fst, prod'_ε_fst, prod'_ε_fst, prod'_app_fst, IsMonoidal.unit]
      /-
        🎉 no goals
      -/
      /-
        case h₂
        C : Type u₁
        inst✝¹⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝¹⁵ : CategoryTheory.MonoidalCategory C
        D : Type u₂
        inst✝¹⁴ : CategoryTheory.Category.{v₂, u₂} D
        inst✝¹³ : CategoryTheory.MonoidalCategory D
        E : Type u₃
        inst✝¹² : CategoryTheory.Category.{v₃, u₃} E
        inst✝¹¹ : CategoryTheory.MonoidalCategory E
        E' : Type u₄
        inst✝¹⁰ : CategoryTheory.Category.{v₄, u₄} E'
        inst✝⁹ : CategoryTheory.MonoidalCategory E'
        F₁ F₂ F₃ : CategoryTheory.Functor C D
        τ : Quiver.Hom F₁ F₂
        inst✝⁸ : F₁.LaxMonoidal
        inst✝⁷ : F₂.LaxMonoidal
        inst✝⁶ : F₃.LaxMonoidal
        F G : CategoryTheory.Functor C D
        H K : CategoryTheory.Functor C E
        α : Quiver.Hom F G
        β : Quiver.Hom H K
        inst✝⁵ : F.LaxMonoidal
        inst✝⁴ : G.LaxMonoidal
        inst✝³ : CategoryTheory.NatTrans.IsMonoidal α
        inst✝² : H.LaxMonoidal
        inst✝¹ : K.LaxMonoidal
        inst✝ : CategoryTheory.NatTrans.IsMonoidal β
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
      -/
    · rw [prod_comp_snd, prod'_ε_snd, prod'_ε_snd, prod'_app_snd, IsMonoidal.unit]
      /-
        🎉 no goals
      -/
  tensor X Y := by
    /-
      C : Type u₁
      inst✝¹⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹⁵ : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹³ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝¹² : CategoryTheory.Category.{v₃, u₃} E
      inst✝¹¹ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝¹⁰ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁹ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁸ : F₁.LaxMonoidal
      inst✝⁷ : F₂.LaxMonoidal
      inst✝⁶ : F₃.LaxMonoidal
      F G : CategoryTheory.Functor C D
      H K : CategoryTheory.Functor C E
      α : Quiver.Hom F G
      β : Quiver.Hom H K
      inst✝⁵ : F.LaxMonoidal
      inst✝⁴ : G.LaxMonoidal
      inst✝³ : CategoryTheory.NatTrans.IsMonoidal α
      inst✝² : H.LaxMonoidal
      inst✝¹ : K.LaxMonoidal
      inst✝ : CategoryTheory.NatTrans.IsMonoidal β
      X Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    ext
    · simp only [prod_comp_fst, prod'_μ_fst, prod'_app_fst,
        prodMonoidal_tensorHom, IsMonoidal.tensor]
    · simp only [prod_comp_snd, prod'_μ_snd, prod'_app_snd,
        prodMonoidal_tensorHom, IsMonoidal.tensor]


instance : NatTrans.IsMonoidal e.inv where
             /-
               C : Type u₁
               inst✝¹¹ : CategoryTheory.Category.{v₁, u₁} C
               inst✝¹⁰ : CategoryTheory.MonoidalCategory C
               D : Type u₂
               inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
               inst✝⁸ : CategoryTheory.MonoidalCategory D
               E : Type u₃
               inst✝⁷ : CategoryTheory.Category.{v₃, u₃} E
               inst✝⁶ : CategoryTheory.MonoidalCategory E
               E' : Type u₄
               inst✝⁵ : CategoryTheory.Category.{v₄, u₄} E'
               inst✝⁴ : CategoryTheory.MonoidalCategory E'
               F₁ F₂ F₃ : CategoryTheory.Functor C D
               τ : Quiver.Hom F₁ F₂
               inst✝³ : F₁.LaxMonoidal
               inst✝² : F₂.LaxMonoidal
               inst✝¹ : F₃.LaxMonoidal
               e : CategoryTheory.Iso F₁ F₂
               inst✝ : CategoryTheory.NatTrans.IsMonoidal e.hom
               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
             -/
  unit := by rw [← NatTrans.IsMonoidal.unit (τ := e.hom), assoc, hom_inv_id_app, comp_id]
             /-
               🎉 no goals
             -/
  tensor X Y := by
    rw [← cancel_mono (e.hom.app (X ⊗ Y)), assoc, assoc, inv_hom_id_app, comp_id,
      NatTrans.IsMonoidal.tensor, ← MonoidalCategory.tensor_comp_assoc,
      inv_hom_id_app, inv_hom_id_app, tensorHom_id, id_whiskerRight, id_comp]


instance : NatTrans.IsMonoidal adj.unit where
  unit := by
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Cat …
    -/
    rw [id_comp, ← unit_app_unit_comp_map_η adj, assoc, Monoidal.map_η_ε]
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      ⊢ Eq (adj.unit.app CategoryTheory.MonoidalCategoryStruct.tensorUnit) (Category …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      ⊢ Eq (adj.unit.app CategoryTheory.MonoidalCategoryStruct.tensorUnit) (Category …
    -/
    rw [comp_id]
    /-
      🎉 no goals
    -/
  tensor X Y := by
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      X Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      X Y : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
    -/
    rw [← unit_app_tensor_comp_map_δ_assoc, id_comp, Monoidal.map_δ_μ, comp_id]
    /-
      🎉 no goals
    -/


instance : NatTrans.IsMonoidal adj.counit where
  unit := by
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.ε …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, map_ε_comp_counit_app_unit adj, ε_η]
    /-
      🎉 no goals
    -/
  tensor X Y := by
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      X Y : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.LaxMonoidal.μ …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹² : CategoryTheory.MonoidalCategory C
      D : Type u₂
      inst✝¹¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹⁰ : CategoryTheory.MonoidalCategory D
      E : Type u₃
      inst✝⁹ : CategoryTheory.Category.{v₃, u₃} E
      inst✝⁸ : CategoryTheory.MonoidalCategory E
      E' : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} E'
      inst✝⁶ : CategoryTheory.MonoidalCategory E'
      F₁ F₂ F₃ : CategoryTheory.Functor C D
      τ : Quiver.Hom F₁ F₂
      inst✝⁵ : F₁.LaxMonoidal
      inst✝⁴ : F₂.LaxMonoidal
      inst✝³ : F₃.LaxMonoidal
      F : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction F G
      inst✝² : F.Monoidal
      inst✝¹ : G.LaxMonoidal
      inst✝ : adj.IsMonoidal
      X Y : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, map_μ_comp_counit_app_tensor, μ_δ_assoc, comp_id]
    /-
      🎉 no goals
    -/


instance : NatTrans.IsMonoidal e.unit :=
  inferInstanceAs (NatTrans.IsMonoidal e.toAdjunction.unit)


instance : NatTrans.IsMonoidal e.counit :=
  inferInstanceAs (NatTrans.IsMonoidal e.toAdjunction.counit)


/-- The type of monoidal natural transformations between (bundled) lax monoidal functors. -/
structure Hom (F G : LaxMonoidalFunctor C D) where
  /-- the natural transformation between the underlying functors -/
  hom : F.toFunctor ⟶ G.toFunctor
  isMonoidal : NatTrans.IsMonoidal hom := by infer_instance


instance : Category (LaxMonoidalFunctor C D) where
  Hom := Hom
                             /-
                               C : Type u₁
                               inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
                               inst✝⁹ : CategoryTheory.MonoidalCategory C
                               D : Type u₂
                               inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
                               inst✝⁷ : CategoryTheory.MonoidalCategory D
                               E : Type u₃
                               inst✝⁶ : CategoryTheory.Category.{v₃, u₃} E
                               inst✝⁵ : CategoryTheory.MonoidalCategory E
                               E' : Type u₄
                               inst✝⁴ : CategoryTheory.Category.{v₄, u₄} E'
                               inst✝³ : CategoryTheory.MonoidalCategory E'
                               F₁ F₂ F₃ : CategoryTheory.Functor C D
                               τ : Quiver.Hom F₁ F₂
                               inst✝² : F₁.LaxMonoidal
                               inst✝¹ : F₂.LaxMonoidal
                               inst✝ : F₃.LaxMonoidal
                               X✝ Y✝ Z✝ : CategoryTheory.LaxMonoidalFunctor C D
                               α : Quiver.Hom X✝ Y✝
                               β : Quiver.Hom Y✝ Z✝
                               ⊢ CategoryTheory.NatTrans.IsMonoidal (CategoryTheory.CategoryStruct.comp α.hom …
                             -/
  comp α β := ⟨α.1 ≫ β.1, by have := α.2; have := β.2; infer_instance⟩
                                                       /-
                                                         🎉 no goals
                                                       -/
  id _ := ⟨𝟙 _, inferInstance⟩


@[simp]
lemma id_hom (F : LaxMonoidalFunctor C D) : Hom.hom (𝟙 F) = 𝟙 _ := rfl


@[reassoc, simp]
lemma comp_hom {F G H : LaxMonoidalFunctor C D} (α : F ⟶ G) (β : G ⟶ H) :
    (α ≫ β).hom = α.hom ≫ β.hom := rfl


@[ext]
lemma hom_ext {F G : LaxMonoidalFunctor C D} {α β : F ⟶ G} (h : α.hom = β.hom) : α = β := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    inst✝² : CategoryTheory.MonoidalCategory C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : CategoryTheory.MonoidalCategory D
    F G : CategoryTheory.LaxMonoidalFunctor C D
    α β : Quiver.Hom F G
    h : Eq α.hom β.hom
    ⊢ Eq α β
  -/
  cases α; cases β; subst h; rfl
                             /-
                               🎉 no goals
                             -/


/-- Constructor for morphisms in the category `LaxMonoidalFunctor C D`. -/
@[simps]
def homMk {F G : LaxMonoidalFunctor C D} (f : F.toFunctor ⟶ G.toFunctor) [NatTrans.IsMonoidal f] :
    F ⟶ G := ⟨f, inferInstance⟩


/-- Constructor for isomorphisms in the category `LaxMonoidalFunctor C D`. -/
@[simps]
def isoMk {F G : LaxMonoidalFunctor C D} (e : F.toFunctor ≅ G.toFunctor)
    [NatTrans.IsMonoidal e.hom] :
    F ≅ G where
  hom := homMk e.hom
  inv := homMk e.inv


/-- Constructor for isomorphisms between lax monoidal functors. -/
@[simps!]
def isoOfComponents {F G : LaxMonoidalFunctor C D} (e : ∀ X, F.obj X ≅ G.obj X)
    (naturality : ∀ {X Y : C} (f : X ⟶ Y), F.map f ≫ (e Y).hom = (e X).hom ≫ G.map f := by
      aesop_cat)
    (unit : ε F.toFunctor ≫ (e (𝟙_ C)).hom = ε G.toFunctor := by aesop_cat)
    (tensor : ∀ X Y, μ F.toFunctor X Y ≫ (e (X ⊗ Y)).hom =
      ((e X).hom ⊗ (e Y).hom) ≫ μ G.toFunctor X Y := by aesop_cat) :
    F ≅ G :=
                                                                /-
                                                                  C : Type u₁
                                                                  inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
                                                                  inst✝⁹ : CategoryTheory.MonoidalCategory C
                                                                  D : Type u₂
                                                                  inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
                                                                  inst✝⁷ : CategoryTheory.MonoidalCategory D
                                                                  E : Type u₃
                                                                  inst✝⁶ : CategoryTheory.Category.{v₃, u₃} E
                                                                  inst✝⁵ : CategoryTheory.MonoidalCategory E
                                                                  E' : Type u₄
                                                                  inst✝⁴ : CategoryTheory.Category.{v₄, u₄} E'
                                                                  inst✝³ : CategoryTheory.MonoidalCategory E'
                                                                  F₁ F₂ F₃ : CategoryTheory.Functor C D
                                                                  τ : Quiver.Hom F₁ F₂
                                                                  inst✝² : F₁.LaxMonoidal
                                                                  inst✝¹ : F₂.LaxMonoidal
                                                                  inst✝ : F₃.LaxMonoidal
                                                                  F G : CategoryTheory.LaxMonoidalFunctor C D
                                                                  e : (X : C) → CategoryTheory.Iso (F.obj X) (G.obj X)
                                                                  naturality : autoParam (∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.C …
                                                                  unit : autoParam (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Funct …
                                                                  tensor : autoParam (∀ (X Y : C), Eq (CategoryTheory.CategoryStruct.comp (Categ …
                                                                  ⊢ CategoryTheory.NatTrans.IsMonoidal (CategoryTheory.NatIso.ofComponents e ⋯). …
                                                                -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  @isoMk _ _ _ _ _ _ _ _ (NatIso.ofComponents e naturality) (by constructor <;> assumption)
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


