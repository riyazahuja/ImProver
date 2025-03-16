/--
If `π` is a surjective morphism in `CompHausLike P`, then it is an effective epi.
-/
noncomputable
def effectiveEpiStruct {B X : CompHausLike P} (π : X ⟶ B) (hπ : Function.Surjective π) :
    EffectiveEpiStruct π where
  desc e h := (IsQuotientMap.of_surjective_continuous hπ π.continuous).lift e fun a b hab ↦
    DFunLike.congr_fun (h ⟨fun _ ↦ a, continuous_const⟩ ⟨fun _ ↦ b, continuous_const⟩
        /-
          P : TopCat → Prop
          B X : CompHausLike P
          π : Quiver.Hom X B
          hπ : Function.Surjective ⇑π
          W✝ : CompHausLike P
          e : Quiver.Hom X W✝
          h : ∀ {Z : CompHausLike P} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.Catego …
          a b : (CategoryTheory.forget (CompHausLike P)).obj X
          hab : Eq (π a) (π b)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => a, continuous_toF …
        -/
    (by ext; exact hab)) a
             /-
               🎉 no goals
             -/
  fac e h := ((IsQuotientMap.of_surjective_continuous hπ π.continuous).lift_comp e
    fun a b hab ↦ DFunLike.congr_fun (h ⟨fun _ ↦ a, continuous_const⟩ ⟨fun _ ↦ b, continuous_const⟩
        /-
          P : TopCat → Prop
          B X : CompHausLike P
          π : Quiver.Hom X B
          hπ : Function.Surjective ⇑π
          W✝ : CompHausLike P
          e : Quiver.Hom X W✝
          h : ∀ {Z : CompHausLike P} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.Catego …
          a b : (CategoryTheory.forget (CompHausLike P)).obj X
          hab : Eq (π a) (π b)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => a, continuous_toF …
        -/
    (by ext; exact hab)) a)
             /-
               🎉 no goals
             -/
  uniq e h g hm := by
    suffices g = (IsQuotientMap.of_surjective_continuous hπ π.continuous).liftEquiv ⟨e,
      fun a b hab ↦ DFunLike.congr_fun
        (h ⟨fun _ ↦ a, continuous_const⟩ ⟨fun _ ↦ b, continuous_const⟩ (by ext; exact hab))
        a⟩ by assumption
    /-
      P : TopCat → Prop
      B X : CompHausLike P
      π : Quiver.Hom X B
      hπ : Function.Surjective ⇑π
      W✝ : CompHausLike P
      e : Quiver.Hom X W✝
      h : ∀ {Z : CompHausLike P} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.Catego …
      g : Quiver.Hom B W✝
      hm : Eq (CategoryTheory.CategoryStruct.comp π g) e
      ⊢ Eq g (⋯.liftEquiv ⟨e, ⋯⟩)
    -/
    rw [← Equiv.symm_apply_eq (IsQuotientMap.of_surjective_continuous hπ π.continuous).liftEquiv]
    /-
      P : TopCat → Prop
      B X : CompHausLike P
      π : Quiver.Hom X B
      hπ : Function.Surjective ⇑π
      W✝ : CompHausLike P
      e : Quiver.Hom X W✝
      h : ∀ {Z : CompHausLike P} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.Catego …
      g : Quiver.Hom B W✝
      hm : Eq (CategoryTheory.CategoryStruct.comp π g) e
      ⊢ Eq (⋯.liftEquiv.symm g) ⟨e, ⋯⟩
    -/
    ext
    /-
      case a.h
      P : TopCat → Prop
      B X : CompHausLike P
      π : Quiver.Hom X B
      hπ : Function.Surjective ⇑π
      W✝ : CompHausLike P
      e : Quiver.Hom X W✝
      h : ∀ {Z : CompHausLike P} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.Catego …
      g : Quiver.Hom B W✝
      hm : Eq (CategoryTheory.CategoryStruct.comp π g) e
      a✝ : (CategoryTheory.forget (CompHausLike P)).obj X
      ⊢ Eq (↑(⋯.liftEquiv.symm g) a✝) (↑⟨e, ⋯⟩ a✝)
    -/
    simp only [IsQuotientMap.liftEquiv_symm_apply_coe, ContinuousMap.comp_apply, ← hm]
    /-
      case a.h
      P : TopCat → Prop
      B X : CompHausLike P
      π : Quiver.Hom X B
      hπ : Function.Surjective ⇑π
      W✝ : CompHausLike P
      e : Quiver.Hom X W✝
      h : ∀ {Z : CompHausLike P} (g₁ g₂ : Quiver.Hom Z X), Eq (CategoryTheory.Catego …
      g : Quiver.Hom B W✝
      hm : Eq (CategoryTheory.CategoryStruct.comp π g) e
      a✝ : (CategoryTheory.forget (CompHausLike P)).obj X
      ⊢ Eq (g (π a✝)) ((CategoryTheory.CategoryStruct.comp π g) a✝)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem preregular [HasExplicitPullbacks P]
    (hs : ∀ ⦃X Y : CompHausLike P⦄ (f : X ⟶ Y), EffectiveEpi f → Function.Surjective f) :
    Preregular (CompHausLike P) where
  exists_fac := by
    /-
      P : TopCat → Prop
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      ⊢ ∀ {X Y Z : CompHausLike P} (f : Quiver.Hom X Y) (g : Quiver.Hom Z Y) [inst : …
    -/
    intro X Y Z f π hπ
    refine ⟨pullback f π, pullback.fst f π, ⟨⟨effectiveEpiStruct _ ?_⟩⟩, pullback.snd f π,
      (pullback.condition _ _).symm⟩
    /-
      P : TopCat → Prop
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      X Y Z : CompHausLike P
      f : Quiver.Hom X Y
      π : Quiver.Hom Z Y
      hπ : CategoryTheory.EffectiveEpi π
      ⊢ Function.Surjective ⇑(CompHausLike.pullback.fst f π)
    -/
    intro y
    /-
      P : TopCat → Prop
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      X Y Z : CompHausLike P
      f : Quiver.Hom X Y
      π : Quiver.Hom Z Y
      hπ : CategoryTheory.EffectiveEpi π
      y : (CategoryTheory.forget (CompHausLike P)).obj X
      ⊢ Exists fun a => Eq ((CompHausLike.pullback.fst f π) a) y
    -/
    obtain ⟨z, hz⟩ := hs π hπ (f y)
    /-
      case intro
      P : TopCat → Prop
      inst✝ : CompHausLike.HasExplicitPullbacks P
      hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
      X Y Z : CompHausLike P
      f : Quiver.Hom X Y
      π : Quiver.Hom Z Y
      hπ : CategoryTheory.EffectiveEpi π
      y : (CategoryTheory.forget (CompHausLike P)).obj X
      z : (CategoryTheory.forget (CompHausLike P)).obj Z
      hz : Eq (π z) (f y)
      ⊢ Exists fun a => Eq ((CompHausLike.pullback.fst f π) a) y
    -/
    exact ⟨⟨(y, z), hz.symm⟩, rfl⟩
    /-
      🎉 no goals
    -/


theorem precoherent [HasExplicitPullbacks P] [HasExplicitFiniteCoproducts.{0} P]
    (hs : ∀ ⦃X Y : CompHausLike P⦄ (f : X ⟶ Y), EffectiveEpi f → Function.Surjective f) :
    Precoherent (CompHausLike P) := by
  /-
    P : TopCat → Prop
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    ⊢ CategoryTheory.Precoherent (CompHausLike P)
  -/
  have : Preregular (CompHausLike P) := preregular hs
  /-
    P : TopCat → Prop
    inst✝¹ : CompHausLike.HasExplicitPullbacks P
    inst✝ : CompHausLike.HasExplicitFiniteCoproducts P
    hs : ∀ ⦃X Y : CompHausLike P⦄ (f : Quiver.Hom X Y), CategoryTheory.EffectiveEp …
    this : CategoryTheory.Preregular (CompHausLike P)
    ⊢ CategoryTheory.Precoherent (CompHausLike P)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


