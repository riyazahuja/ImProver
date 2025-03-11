/-- The functor `N` for the equivalence is obtained by composing
`N' : SimplicialObject C ⥤ Karoubi (ChainComplex C ℕ)` and the inverse
of the equivalence `ChainComplex C ℕ ≌ Karoubi (ChainComplex C ℕ)`. -/
@[simps!, nolint unusedArguments]
def N [IsIdempotentComplete C] [HasFiniteCoproducts C] : SimplicialObject C ⥤ ChainComplex C ℕ :=
  N₁ ⋙ (toKaroubiEquivalence _).inverse


/-- The functor `Γ` for the equivalence is `Γ'`. -/
@[simps!, nolint unusedArguments]
def Γ [IsIdempotentComplete C] [HasFiniteCoproducts C] : ChainComplex C ℕ ⥤ SimplicialObject C :=
  Γ₀


/-- A reformulation of the isomorphism `toKaroubi (SimplicialObject C) ⋙ N₂ ≅ N₁` -/
def isoN₁ :
    (toKaroubiEquivalence (SimplicialObject C)).functor ⋙
      Preadditive.DoldKan.equivalence.functor ≅ N₁ := toKaroubiCompN₂IsoN₁


@[simp]
lemma isoN₁_hom_app_f (X : SimplicialObject C) :
    (isoN₁.hom.app X).f = PInfty := rfl


/-- A reformulation of the canonical isomorphism
`toKaroubi (ChainComplex C ℕ) ⋙ Γ₂ ≅ Γ ⋙ toKaroubi (SimplicialObject C)`. -/
def isoΓ₀ :
    (toKaroubiEquivalence (ChainComplex C ℕ)).functor ⋙ Preadditive.DoldKan.equivalence.inverse ≅
      Γ ⋙ (toKaroubiEquivalence _).functor :=
  (functorExtension₂CompWhiskeringLeftToKaroubiIso _ _).app Γ₀


@[simp]
lemma N₂_map_isoΓ₀_hom_app_f (X : ChainComplex C ℕ) :
    (N₂.map (isoΓ₀.hom.app X)).f = PInfty := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₂.map (CategoryTheory.Idempotents.DoldKan.iso …
  -/
  ext
  /-
    case h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    i✝ : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.N₂.map (CategoryTheory.Idempotents.DoldKan.is …
  -/
  apply comp_id
  /-
    🎉 no goals
  -/


/-- The Dold-Kan equivalence for pseudoabelian categories given
by the functors `N` and `Γ`. It is obtained by applying the results in
`Compatibility.lean` to the equivalence `Preadditive.DoldKan.Equivalence`. -/
def equivalence : SimplicialObject C ≌ ChainComplex C ℕ :=
  Compatibility.equivalence isoN₁ isoΓ₀


theorem equivalence_functor : (equivalence : SimplicialObject C ≌ _).functor = N :=
  rfl


theorem equivalence_inverse : (equivalence : SimplicialObject C ≌ _).inverse = Γ :=
  rfl


/-- The natural isomorphism `NΓ' satisfies the compatibility that is needed
for the construction of our counit isomorphism `η` -/
theorem hη :
    Compatibility.τ₀ =
      Compatibility.τ₁ isoN₁ isoΓ₀
        (N₁Γ₀ : Γ ⋙ N₁ ≅ (toKaroubiEquivalence (ChainComplex C ℕ)).functor) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ Eq AlgebraicTopology.DoldKan.Compatibility.τ₀ (AlgebraicTopology.DoldKan.Com …
  -/
  ext K : 3
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.τ₀.hom.app K) ((AlgebraicTopolog …
  -/
  simp only [Compatibility.τ₀_hom_app, Compatibility.τ₁_hom_app]
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (CategoryTheory.Preadditive.DoldKan.equivalence.counitIso.hom.app ((Categ …
  -/
  exact (N₂Γ₂_compatible_with_N₁Γ₀ K).trans (by simp )
  /-
    🎉 no goals
  -/


/-- The counit isomorphism induced by `N₁Γ₀` -/
@[simps!]
def η : Γ ⋙ N ≅ 𝟭 (ChainComplex C ℕ) :=
  Compatibility.equivalenceCounitIso
    (N₁Γ₀ : (Γ : ChainComplex C ℕ ⥤ _) ⋙ N₁ ≅ (toKaroubiEquivalence _).functor)


theorem equivalence_counitIso :
    DoldKan.equivalence.counitIso = (η : Γ ⋙ N ≅ 𝟭 (ChainComplex C ℕ)) :=
  Compatibility.equivalenceCounitIso_eq hη


theorem hε :
    Compatibility.υ (isoN₁) =
      (Γ₂N₁ : (toKaroubiEquivalence _).functor ≅
          (N₁ : SimplicialObject C ⥤ _) ⋙ Preadditive.DoldKan.equivalence.inverse) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.υ CategoryTheory.Idempotents.Dol …
  -/
  dsimp only [isoN₁]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.υ AlgebraicTopology.DoldKan.toKa …
  -/
  ext1
  /-
    case w
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ Eq (AlgebraicTopology.DoldKan.Compatibility.υ AlgebraicTopology.DoldKan.toKa …
  -/
  rw [← cancel_epi Γ₂N₁.inv, Iso.inv_hom_id]
  /-
    case w
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.Γ₂N₁.inv (A …
  -/
  ext X : 2
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.Γ₂N₁.inv ( …
  -/
  rw [NatTrans.comp_app]
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₂N₁.inv.a …
  -/
  erw [compatibility_Γ₂N₁_Γ₂N₂_natTrans X]
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Compatibility.υ_hom_app, Preadditive.DoldKan.equivalence_unitIso, Iso.app_inv, assoc]
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₂N₂ToKaro …
  -/
  erw [← NatTrans.comp_app_assoc, IsIso.hom_inv_id]
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₂N₂ToKaro …
  -/
  rw [NatTrans.id_app, id_comp, NatTrans.id_app, Γ₂N₂ToKaroubiIso_inv_app]
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₂.map (Al …
  -/
  dsimp only [Preadditive.DoldKan.equivalence_inverse, Preadditive.DoldKan.Γ]
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₂.map (Al …
  -/
  rw [← Γ₂.map_comp, Iso.inv_hom_id_app, Γ₂.map_id]
  /-
    case w.w.h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.IsIdempotentComplete C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.id (AlgebraicTopology.DoldKan.Γ₂.obj (Alge …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The unit isomorphism induced by `Γ₂N₁`. -/
def ε : 𝟭 (SimplicialObject C) ≅ N ⋙ Γ :=
  Compatibility.equivalenceUnitIso isoΓ₀ Γ₂N₁


theorem equivalence_unitIso :
    DoldKan.equivalence.unitIso = (ε : 𝟭 (SimplicialObject C) ≅ N ⋙ Γ) :=
  Compatibility.equivalenceUnitIso_eq hε


