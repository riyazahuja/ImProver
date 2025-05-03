theorem Preadditive.isSeparating_iff (𝒢 : Set C) :
    IsSeparating 𝒢 ↔ ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ G ∈ 𝒢, ∀ (h : G ⟶ X), h ≫ f = 0) → f = 0 :=
                                 /-
                                   C : Type u
                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                   inst✝ : CategoryTheory.Preadditive C
                                   𝒢 : Set C
                                   h𝒢 : CategoryTheory.IsSeparating 𝒢
                                   X Y : C
                                   f : Quiver.Hom X Y
                                   hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G X), Eq (CategoryTheor …
                                   ⊢ ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.C …
                                 -/
  ⟨fun h𝒢 X Y f hf => h𝒢 _ _ (by simpa only [Limits.comp_zero] using hf), fun h𝒢 X Y f g hfg =>
                                 /-
                                   🎉 no goals
                                 -/
                              /-
                                C : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                inst✝ : CategoryTheory.Preadditive C
                                𝒢 : Set C
                                h𝒢 : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), (∀ (G : C), Membership.mem 𝒢 G → ∀ (h : …
                                X Y : C
                                f g : Quiver.Hom X Y
                                hfg : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G X), Eq (CategoryTheo …
                                ⊢ ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.C …
                              -/
    sub_eq_zero.1 <| h𝒢 _ (by simpa only [Preadditive.comp_sub, sub_eq_zero] using hfg)⟩
                              /-
                                🎉 no goals
                              -/


theorem Preadditive.isCoseparating_iff (𝒢 : Set C) :
    IsCoseparating 𝒢 ↔ ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ G ∈ 𝒢, ∀ (h : Y ⟶ G), f ≫ h = 0) → f = 0 :=
                                 /-
                                   C : Type u
                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                   inst✝ : CategoryTheory.Preadditive C
                                   𝒢 : Set C
                                   h𝒢 : CategoryTheory.IsCoseparating 𝒢
                                   X Y : C
                                   f : Quiver.Hom X Y
                                   hf : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheor …
                                   ⊢ ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.C …
                                 -/
  ⟨fun h𝒢 X Y f hf => h𝒢 _ _ (by simpa only [Limits.zero_comp] using hf), fun h𝒢 X Y f g hfg =>
                                 /-
                                   🎉 no goals
                                 -/
                              /-
                                C : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                inst✝ : CategoryTheory.Preadditive C
                                𝒢 : Set C
                                h𝒢 : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), (∀ (G : C), Membership.mem 𝒢 G → ∀ (h : …
                                X Y : C
                                f g : Quiver.Hom X Y
                                hfg : ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheo …
                                ⊢ ∀ (G : C), Membership.mem 𝒢 G → ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.C …
                              -/
    sub_eq_zero.1 <| h𝒢 _ (by simpa only [Preadditive.sub_comp, sub_eq_zero] using hfg)⟩
                              /-
                                🎉 no goals
                              -/


theorem Preadditive.isSeparator_iff (G : C) :
    IsSeparator G ↔ ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ h : G ⟶ X, h ≫ f = 0) → f = 0 :=
                                     /-
                                       C : Type u
                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                       inst✝ : CategoryTheory.Preadditive C
                                       G : C
                                       hG : CategoryTheory.IsSeparator G
                                       X Y : C
                                       f : Quiver.Hom X Y
                                       hf : ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.CategoryStruct.comp h f) 0
                                       ⊢ ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.CategoryStruct.comp h f) (Categor …
                                     -/
  ⟨fun hG X Y f hf => hG.def _ _ (by simpa only [Limits.comp_zero] using hf), fun hG =>
                                     /-
                                       🎉 no goals
                                     -/
    (isSeparator_def _).2 fun X Y f g hfg =>
                                /-
                                  C : Type u
                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                  inst✝ : CategoryTheory.Preadditive C
                                  G : C
                                  hG : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), (∀ (h : Quiver.Hom G X), Eq (CategoryTh …
                                  X Y : C
                                  f g : Quiver.Hom X Y
                                  hfg : ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.CategoryStruct.comp h f) (Cat …
                                  ⊢ ∀ (h : Quiver.Hom G X), Eq (CategoryTheory.CategoryStruct.comp h (HSub.hSub  …
                                -/
      sub_eq_zero.1 <| hG _ (by simpa only [Preadditive.comp_sub, sub_eq_zero] using hfg)⟩
                                /-
                                  🎉 no goals
                                -/


theorem Preadditive.isCoseparator_iff (G : C) :
    IsCoseparator G ↔ ∀ ⦃X Y : C⦄ (f : X ⟶ Y), (∀ h : Y ⟶ G, f ≫ h = 0) → f = 0 :=
                                     /-
                                       C : Type u
                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                       inst✝ : CategoryTheory.Preadditive C
                                       G : C
                                       hG : CategoryTheory.IsCoseparator G
                                       X Y : C
                                       f : Quiver.Hom X Y
                                       hf : ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.CategoryStruct.comp f h) 0
                                       ⊢ ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.CategoryStruct.comp f h) (Categor …
                                     -/
  ⟨fun hG X Y f hf => hG.def _ _ (by simpa only [Limits.zero_comp] using hf), fun hG =>
                                     /-
                                       🎉 no goals
                                     -/
    (isCoseparator_def _).2 fun X Y f g hfg =>
                                /-
                                  C : Type u
                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                  inst✝ : CategoryTheory.Preadditive C
                                  G : C
                                  hG : ∀ ⦃X Y : C⦄ (f : Quiver.Hom X Y), (∀ (h : Quiver.Hom Y G), Eq (CategoryTh …
                                  X Y : C
                                  f g : Quiver.Hom X Y
                                  hfg : ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.CategoryStruct.comp f h) (Cat …
                                  ⊢ ∀ (h : Quiver.Hom Y G), Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f  …
                                -/
      sub_eq_zero.1 <| hG _ (by simpa only [Preadditive.sub_comp, sub_eq_zero] using hfg)⟩
                                /-
                                  🎉 no goals
                                -/


theorem isSeparator_iff_faithful_preadditiveCoyoneda (G : C) :
    IsSeparator G ↔ (preadditiveCoyoneda.obj (op G)).Faithful := by
  rw [isSeparator_iff_faithful_coyoneda_obj, ← whiskering_preadditiveCoyoneda, Functor.comp_obj,
    whiskeringRight_obj_obj]
  exact ⟨fun h => Functor.Faithful.of_comp _ (forget AddCommGrp),
    fun h => Functor.Faithful.comp _ _⟩


theorem isSeparator_iff_faithful_preadditiveCoyonedaObj (G : C) :
    IsSeparator G ↔ (preadditiveCoyonedaObj (op G)).Faithful := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    G : C
    ⊢ Iff (CategoryTheory.IsSeparator G) (CategoryTheory.preadditiveCoyonedaObj {  …
  -/
  rw [isSeparator_iff_faithful_preadditiveCoyoneda, preadditiveCoyoneda_obj]
  exact ⟨fun h => Functor.Faithful.of_comp _ (forget₂ _ AddCommGrp.{v}),
    fun h => Functor.Faithful.comp _ _⟩


theorem isCoseparator_iff_faithful_preadditiveYoneda (G : C) :
    IsCoseparator G ↔ (preadditiveYoneda.obj G).Faithful := by
  rw [isCoseparator_iff_faithful_yoneda_obj, ← whiskering_preadditiveYoneda, Functor.comp_obj,
    whiskeringRight_obj_obj]
  exact ⟨fun h => Functor.Faithful.of_comp _ (forget AddCommGrp),
    fun h => Functor.Faithful.comp _ _⟩


theorem isCoseparator_iff_faithful_preadditiveYonedaObj (G : C) :
    IsCoseparator G ↔ (preadditiveYonedaObj G).Faithful := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    G : C
    ⊢ Iff (CategoryTheory.IsCoseparator G) (CategoryTheory.preadditiveYonedaObj G) …
  -/
  rw [isCoseparator_iff_faithful_preadditiveYoneda, preadditiveYoneda_obj]
  exact ⟨fun h => Functor.Faithful.of_comp _ (forget₂ _ AddCommGrp.{v}),
    fun h => Functor.Faithful.comp _ _⟩


