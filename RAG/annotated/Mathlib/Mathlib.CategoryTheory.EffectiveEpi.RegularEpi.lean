/-- The data of an `EffectiveEpi` structure on a `RegularEpi`. -/
def effectiveEpiStructOfRegularEpi {B X : C} (f : X ⟶ B) [RegularEpi f] :
    EffectiveEpiStruct f where
  desc _ h := Cofork.IsColimit.desc isColimit _ (h _ _ w)
  fac _ _ := Cofork.IsColimit.π_desc' isColimit _ _
  uniq _ _ _ hg := Cofork.IsColimit.hom_ext isColimit (hg.trans
    (Cofork.IsColimit.π_desc' _ _ _).symm)


instance {B X : C} (f : X ⟶ B) [RegularEpi f] : EffectiveEpi f :=
  ⟨⟨effectiveEpiStructOfRegularEpi f⟩⟩


/-- A morphism which is a coequalizer for its kernel pair is an effective epi. -/
theorem effectiveEpiOfKernelPair {B X : C} (f : X ⟶ B) [HasPullback f f]
    (hc : IsColimit (Cofork.ofπ f pullback.condition)) : EffectiveEpi f :=
  let _ := regularEpiOfKernelPair f hc
  inferInstance


/-- An effective epi which has a kernel pair is a regular epi. -/
noncomputable instance regularEpiOfEffectiveEpi {B X : C} (f : X ⟶ B) [HasPullback f f]
    [EffectiveEpi f] : RegularEpi f where
  W := pullback f f
  left := pullback.fst f f
  right := pullback.snd f f
  w := pullback.condition
  isColimit := {
    desc := fun s ↦ EffectiveEpi.desc f (s.ι.app WalkingParallelPair.one) fun g₁ g₂ hg ↦ (by
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.3751, u_1} C
        B X : C
        f : Quiver.Hom X B
        inst✝¹ : CategoryTheory.Limits.HasPullback f f
        inst✝ : CategoryTheory.EffectiveEpi f
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair (Category …
        Z✝ : C
        g₁ g₂ : Quiver.Hom Z✝ X
        hg : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStru …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (s.ι.app CategoryTheory.Limits.Wal …
      -/
      simp only [Cofork.app_one_eq_π]
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.3751, u_1} C
        B X : C
        f : Quiver.Hom X B
        inst✝¹ : CategoryTheory.Limits.HasPullback f f
        inst✝ : CategoryTheory.EffectiveEpi f
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair (Category …
        Z✝ : C
        g₁ g₂ : Quiver.Hom Z✝ X
        hg : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStru …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Cofork.π s) …
      -/
      rw [← pullback.lift_snd g₁ g₂ hg, Category.assoc, ← Cofork.app_zero_eq_comp_π_right]
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.3751, u_1} C
        B X : C
        f : Quiver.Hom X B
        inst✝¹ : CategoryTheory.Limits.HasPullback f f
        inst✝ : CategoryTheory.EffectiveEpi f
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair (Category …
        Z✝ : C
        g₁ g₂ : Quiver.Hom Z✝ X
        hg : Eq (CategoryTheory.CategoryStruct.comp g₁ f) (CategoryTheory.CategoryStru …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Cofork.π s) …
      -/
      simp)
      /-
        🎉 no goals
      -/
    fac := by
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.3751, u_1} C
        B X : C
        f : Quiver.Hom X B
        inst✝¹ : CategoryTheory.Limits.HasPullback f f
        inst✝ : CategoryTheory.EffectiveEpi f
        ⊢ ∀ (s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair (Cat …
      -/
      intro s j
      have := EffectiveEpi.fac f (s.ι.app WalkingParallelPair.one) fun g₁ g₂ hg ↦ (by
          simp only [Cofork.app_one_eq_π]
          rw [← pullback.lift_snd g₁ g₂ hg, Category.assoc, ← Cofork.app_zero_eq_comp_π_right]
          simp)
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.3751, u_1} C
        B X : C
        f : Quiver.Hom X B
        inst✝¹ : CategoryTheory.Limits.HasPullback f f
        inst✝ : CategoryTheory.EffectiveEpi f
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair (Category …
        j : CategoryTheory.Limits.WalkingParallelPair
        this : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.EffectiveEpi.d …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Cofork.ofπ f  …
      -/
      simp only [Functor.const_obj_obj, Cofork.app_one_eq_π] at this
      cases j with
      | zero => simp [this]
      | one => simp [this]
    uniq := fun _ _ h ↦ EffectiveEpi.uniq f _ _ _ (h WalkingParallelPair.one) }


