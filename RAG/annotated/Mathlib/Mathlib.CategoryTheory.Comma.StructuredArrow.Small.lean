instance small_proj_preimage_of_locallySmall {𝒢 : Set C} [Small.{v₁} 𝒢] [LocallySmall.{v₁} D] :
    Small.{v₁} ((proj S T).obj ⁻¹' 𝒢) := by
  suffices (proj S T).obj ⁻¹' 𝒢 = Set.range fun f : ΣG : 𝒢, S ⟶ T.obj G => mk f.2 by
    rw [this]
    infer_instance
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    𝒢 : Set C
    inst✝¹ : Small.{v₁, u₁} ↑𝒢
    inst✝ : CategoryTheory.LocallySmall.{v₁, v₂, u₂} D
    ⊢ Eq (Set.preimage (CategoryTheory.StructuredArrow.proj S T).obj 𝒢) (Set.range …
  -/
  exact Set.ext fun X => ⟨fun h => ⟨⟨⟨_, h⟩, X.hom⟩, (eq_mk _).symm⟩, by aesop_cat⟩
  /-
    🎉 no goals
  -/


instance small_proj_preimage_of_locallySmall {𝒢 : Set C} [Small.{v₁} 𝒢] [LocallySmall.{v₁} D] :
    Small.{v₁} ((proj S T).obj ⁻¹' 𝒢) := by
  suffices (proj S T).obj ⁻¹' 𝒢 = Set.range fun f : ΣG : 𝒢, S.obj G ⟶ T => mk f.2 by
    rw [this]
    infer_instance
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    𝒢 : Set C
    inst✝¹ : Small.{v₁, u₁} ↑𝒢
    inst✝ : CategoryTheory.LocallySmall.{v₁, v₂, u₂} D
    ⊢ Eq (Set.preimage (CategoryTheory.CostructuredArrow.proj S T).obj 𝒢) (Set.ran …
  -/
  exact Set.ext fun X => ⟨fun h => ⟨⟨⟨_, h⟩, X.hom⟩, (eq_mk _).symm⟩, by aesop_cat⟩
  /-
    🎉 no goals
  -/


