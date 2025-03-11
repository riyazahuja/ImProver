/-- The functor `Karoubi (SimplicialObject C) ⥤ Karoubi (ChainComplex C ℕ)` of
the Dold-Kan equivalence for additive categories. -/
@[simp]
def N : Karoubi (SimplicialObject C) ⥤ Karoubi (ChainComplex C ℕ) :=
  N₂


/-- The inverse functor `Karoubi (ChainComplex C ℕ) ⥤ Karoubi (SimplicialObject C)` of
the Dold-Kan equivalence for additive categories. -/
@[simp]
def Γ : Karoubi (ChainComplex C ℕ) ⥤ Karoubi (SimplicialObject C) :=
  Γ₂


/-- The Dold-Kan equivalence `Karoubi (SimplicialObject C) ≌ Karoubi (ChainComplex C ℕ)`
for additive categories. -/
@[simps]
def equivalence : Karoubi (SimplicialObject C) ≌ Karoubi (ChainComplex C ℕ) where
  functor := N
  inverse := Γ
  unitIso := Γ₂N₂
  counitIso := N₂Γ₂
  functor_unitIso_comp P := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.1407, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Preadditive.DoldKan.N …
    -/
    let α := N.mapIso (Γ₂N₂.app P)
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.1407, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      α : CategoryTheory.Iso (CategoryTheory.Preadditive.DoldKan.N.obj ((CategoryThe …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Preadditive.DoldKan.N …
    -/
    let β := N₂Γ₂.app (N.obj P)
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.1407, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      α : CategoryTheory.Iso (CategoryTheory.Preadditive.DoldKan.N.obj ((CategoryThe …
      β : CategoryTheory.Iso ((AlgebraicTopology.DoldKan.Γ₂.comp AlgebraicTopology.D …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Preadditive.DoldKan.N …
    -/
    symm
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.1407, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      α : CategoryTheory.Iso (CategoryTheory.Preadditive.DoldKan.N.obj ((CategoryThe …
      β : CategoryTheory.Iso ((AlgebraicTopology.DoldKan.Γ₂.comp AlgebraicTopology.D …
      ⊢ Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.Preadditive.DoldKan.N.o …
    -/
    change 𝟙 _ = α.hom ≫ β.hom
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.1407, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      α : CategoryTheory.Iso (CategoryTheory.Preadditive.DoldKan.N.obj ((CategoryThe …
      β : CategoryTheory.Iso ((AlgebraicTopology.DoldKan.Γ₂.comp AlgebraicTopology.D …
      ⊢ Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.Preadditive.DoldKan.N.o …
    -/
    rw [← Iso.inv_comp_eq, comp_id, ← comp_id β.hom, ← Iso.inv_comp_eq]
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.1407, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
      α : CategoryTheory.Iso (CategoryTheory.Preadditive.DoldKan.N.obj ((CategoryThe …
      β : CategoryTheory.Iso ((AlgebraicTopology.DoldKan.Γ₂.comp AlgebraicTopology.D …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp β.inv α.inv) (CategoryTheory.Category …
    -/
    exact AlgebraicTopology.DoldKan.identity_N₂_objectwise P
    /-
      🎉 no goals
    -/


