/--
Given an `EffectiveEpiFamily X π` and a corresponding coproduct cocone, the family descends to an
`EffectiveEpi` from the coproduct.
-/
noncomputable
def effectiveEpiStructIsColimitDescOfEffectiveEpiFamily {B : C} {α : Type*} (X : α → C)
    (c : Cofan X) (hc : IsColimit c) (π : (a : α) → (X a ⟶ B)) [EffectiveEpiFamily X π] :
    EffectiveEpiStruct (hc.desc (Cofan.mk B π)) where
  desc e h := EffectiveEpiFamily.desc X π (fun a ↦ c.ι.app ⟨a⟩ ≫ e) (fun a₁ a₂ g₁ g₂ hg ↦ by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      B : C
      α : Type u_2
      X : α → C
      c : CategoryTheory.Limits.Cofan X
      hc : CategoryTheory.Limits.IsColimit c
      π : (a : α) → Quiver.Hom (X a) B
      inst✝ : CategoryTheory.EffectiveEpiFamily X π
      W✝ : C
      e : Quiver.Hom c.pt W✝
      h : ∀ {Z : C} (g₁ g₂ : Quiver.Hom Z c.pt), Eq (CategoryTheory.CategoryStruct.c …
      Z✝ : C
      a₁ a₂ : α
      g₁ : Quiver.Hom Z✝ (X a₁)
      g₂ : Quiver.Hom Z✝ (X a₂)
      hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (π a₁)) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ ((fun a => CategoryTheory.Category …
    -/
    simp only [← Category.assoc]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      B : C
      α : Type u_2
      X : α → C
      c : CategoryTheory.Limits.Cofan X
      hc : CategoryTheory.Limits.IsColimit c
      π : (a : α) → Quiver.Hom (X a) B
      inst✝ : CategoryTheory.EffectiveEpiFamily X π
      W✝ : C
      e : Quiver.Hom c.pt W✝
      h : ∀ {Z : C} (g₁ g₂ : Quiver.Hom Z c.pt), Eq (CategoryTheory.CategoryStruct.c …
      Z✝ : C
      a₁ a₂ : α
      g₁ : Quiver.Hom Z✝ (X a₁)
      g₂ : Quiver.Hom Z✝ (X a₂)
      hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (π a₁)) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
    -/
    exact h (g₁ ≫ c.ι.app ⟨a₁⟩) (g₂ ≫ c.ι.app ⟨a₂⟩) (by simpa))
    /-
      🎉 no goals
    -/
                                       /-
                                         C : Type u_1
                                         inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
                                         B : C
                                         α : Type u_2
                                         X : α → C
                                         c : CategoryTheory.Limits.Cofan X
                                         hc : CategoryTheory.Limits.IsColimit c
                                         π : (a : α) → Quiver.Hom (X a) B
                                         inst✝ : CategoryTheory.EffectiveEpiFamily X π
                                         W✝ : C
                                         e : Quiver.Hom c.pt W✝
                                         h : ∀ {Z : C} (g₁ g₂ : Quiver.Hom Z c.pt), Eq (CategoryTheory.CategoryStruct.c …
                                         x✝ : CategoryTheory.Discrete α
                                         j : α
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { as := j }) (CategoryTheory …
                                       -/
  fac e h := hc.hom_ext (fun ⟨j⟩ ↦ (by simp))
                                       /-
                                         🎉 no goals
                                       -/
  uniq e _ m hm := EffectiveEpiFamily.uniq X π (fun a ↦ c.ι.app ⟨a⟩ ≫ e)
                            /-
                              C : Type u_1
                              inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
                              B : C
                              α : Type u_2
                              X : α → C
                              c : CategoryTheory.Limits.Cofan X
                              hc : CategoryTheory.Limits.IsColimit c
                              π : (a : α) → Quiver.Hom (X a) B
                              inst✝ : CategoryTheory.EffectiveEpiFamily X π
                              W✝ : C
                              e : Quiver.Hom c.pt W✝
                              x✝⁴ : ∀ {Z : C} (g₁ g₂ : Quiver.Hom Z c.pt), Eq (CategoryTheory.CategoryStruct …
                              m : Quiver.Hom (CategoryTheory.Limits.Cofan.mk B π).pt W✝
                              hm : Eq (CategoryTheory.CategoryStruct.comp (hc.desc (CategoryTheory.Limits.Co …
                              Z✝ : C
                              x✝³ x✝² : α
                              x✝¹ : Quiver.Hom Z✝ (X x✝³)
                              x✝ : Quiver.Hom Z✝ (X x✝²)
                              hg : Eq (CategoryTheory.CategoryStruct.comp x✝¹ (π x✝³)) (CategoryTheory.Categ …
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝¹ ((fun a => CategoryTheory.Categor …
                            -/
                            /-
                              🎉 no goals
                            -/
      (fun _ _ _ _ hg ↦ (by simp [← hm, reassoc_of% hg])) m (fun _ ↦ (by simp [← hm]))
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/--
Given an `EffectiveEpiFamily X π` such that the coproduct of `X` exists, `Sigma.desc π` is an
`EffectiveEpi`.
-/
noncomputable
def effectiveEpiStructDescOfEffectiveEpiFamily {B : C} {α : Type*} (X : α → C)
    (π : (a : α) → (X a ⟶ B)) [HasCoproduct X] [EffectiveEpiFamily X π] :
    EffectiveEpiStruct (Sigma.desc π) := by
  simpa [coproductIsCoproduct] using
    effectiveEpiStructIsColimitDescOfEffectiveEpiFamily X _ (coproductIsCoproduct _) π


instance {B : C} {α : Type*} (X : α → C) (π : (a : α) → (X a ⟶ B)) [HasCoproduct X]
    [EffectiveEpiFamily X π] : EffectiveEpi (Sigma.desc π) :=
  ⟨⟨effectiveEpiStructDescOfEffectiveEpiFamily X π⟩⟩


/--
This is an auxiliary lemma used twice in the definition of  `EffectiveEpiFamilyOfEffectiveEpiDesc`.
It is the `h` hypothesis of `EffectiveEpi.desc` and `EffectiveEpi.fac`.
-/
theorem effectiveEpiFamilyStructOfEffectiveEpiDesc_aux {B : C} {α : Type*} {X : α → C}
    {π : (a : α) → X a ⟶ B} [HasCoproduct X]
    [∀ {Z : C} (g : Z ⟶ ∐ X) (a : α), HasPullback g (Sigma.ι X a)]
    [∀ {Z : C} (g : Z ⟶ ∐ X), HasCoproduct fun a ↦ pullback g (Sigma.ι X a)]
    [∀ {Z : C} (g : Z ⟶ ∐ X), Epi (Sigma.desc fun a ↦ pullback.fst g (Sigma.ι X a))]
    {W : C} {e : (a : α) → X a ⟶ W} (h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂),
      g₁ ≫ π a₁ = g₂ ≫ π a₂ → g₁ ≫ e a₁ = g₂ ≫ e a₂) {Z : C}
    {g₁ g₂ : Z ⟶ ∐ fun b ↦ X b} (hg : g₁ ≫ Sigma.desc π = g₂ ≫ Sigma.desc π) :
    g₁ ≫ Sigma.desc e = g₂ ≫ Sigma.desc e := by
  apply_fun ((Sigma.desc fun a ↦ pullback.fst g₁ (Sigma.ι X a)) ≫ ·) using
    (fun a b ↦ (cancel_epi _).mp)
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    ⊢ Eq ((fun x => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigm …
  -/
  ext a
  /-
    case h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun a …
  -/
  simp only [colimit.ι_desc_assoc, Discrete.functor_obj, Cofan.mk_pt, Cofan.mk_ι_app]
  /-
    case h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst g …
  -/
  rw [← Category.assoc, pullback.condition]
  /-
    case h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, colimit.ι_desc, Cofan.mk_pt, Cofan.mk_ι_app]
  apply_fun ((Sigma.desc fun a ↦ pullback.fst (pullback.fst _ _ ≫ g₂) (Sigma.ι X a)) ≫ ·)
    using (fun a b ↦ (cancel_epi _).mp)
  /-
    case h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a : α
    ⊢ Eq ((fun x => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigm …
  -/
  ext b
  /-
    case h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a b : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun a …
  -/
  simp only [colimit.ι_desc_assoc, Discrete.functor_obj, Cofan.mk_pt, Cofan.mk_ι_app]
  /-
    case h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a b : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  simp only [← Category.assoc]
  /-
    case h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a b : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [(Category.assoc _ _ g₂), pullback.condition]
  /-
    case h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a b : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, colimit.ι_desc, Cofan.mk_pt, Cofan.mk_ι_app]
  /-
    case h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a b : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  rw [← Category.assoc]
  /-
    case h.h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a b : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  apply h
  /-
    case h.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.Limits.Sigma.de …
    a b : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  apply_fun (pullback.fst g₁ (Sigma.ι X a) ≫ ·) at hg
  /-
    case h.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    a b : α
    hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fs …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [← Category.assoc, pullback.condition] at hg
  /-
    case h.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    a b : α
    hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, colimit.ι_desc, Cofan.mk_pt, Cofan.mk_ι_app] at hg
  apply_fun ((Sigma.ι (fun a ↦ pullback _ _) b) ≫ (Sigma.desc fun a ↦
    pullback.fst (pullback.fst _ _ ≫ g₂) (Sigma.ι X a)) ≫ ·) at hg
  /-
    case h.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    a b : α
    hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fu …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [colimit.ι_desc_assoc, Discrete.functor_obj, Cofan.mk_pt, Cofan.mk_ι_app] at hg
  /-
    case h.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    a b : α
    hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fs …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [← Category.assoc] at hg
  /-
    case h.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    a b : α
    hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [(Category.assoc _ _ g₂), pullback.condition] at hg
  /-
    case h.h.a
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    B : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    inst✝³ : CategoryTheory.Limits.HasCoproduct X
    inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
    inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
    inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    g₁ g₂ : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj fun b => X b)
    a b : α
    hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simpa using hg
  /-
    🎉 no goals
  -/


/--
If a coproduct interacts well enough with pullbacks, then a family whose domains are the terms of
the coproduct is effective epimorphic whenever `Sigma.desc` induces an effective epimorphism from
the coproduct itself.
-/
noncomputable
def effectiveEpiFamilyStructOfEffectiveEpiDesc {B : C} {α : Type*} (X : α → C)
    (π : (a : α) → (X a ⟶ B)) [HasCoproduct X] [EffectiveEpi (Sigma.desc π)]
    [∀ {Z : C} (g : Z ⟶ ∐ X) (a : α), HasPullback g (Sigma.ι X a)]
    [∀ {Z : C} (g : Z ⟶ ∐ X), HasCoproduct (fun a ↦ pullback g (Sigma.ι X a))]
    [∀ {Z : C} (g : Z ⟶ ∐ X),
      Epi (Sigma.desc (fun a ↦ pullback.fst g (Sigma.ι X a)))] :
    EffectiveEpiFamilyStruct X π where
  desc e h := EffectiveEpi.desc (Sigma.desc π) (Sigma.desc e) fun _ _ hg ↦
    effectiveEpiFamilyStructOfEffectiveEpiDesc_aux h hg
  fac e h a := by
    rw [(by simp : π a = Sigma.ι X a ≫ Sigma.desc π), (by simp : e a = Sigma.ι X a ≫ Sigma.desc e),
      Category.assoc, EffectiveEpi.fac (Sigma.desc π) (Sigma.desc e) (fun g₁ g₂ hg ↦
      effectiveEpiFamilyStructOfEffectiveEpiDesc_aux h hg)]
  uniq _ _ _ hm := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.31508, u_1} C
      B : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝⁴ : CategoryTheory.Limits.HasCoproduct X
      inst✝³ : CategoryTheory.EffectiveEpi (CategoryTheory.Limits.Sigma.desc π)
      inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
      inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
      inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
      W✝ : C
      x✝² : (a : α) → Quiver.Hom (X a) W✝
      x✝¹ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a …
      x✝ : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) x✝) (x✝² a)
      ⊢ Eq x✝ ((fun {W} e h => CategoryTheory.EffectiveEpi.desc (CategoryTheory.Limi …
    -/
    apply EffectiveEpi.uniq (Sigma.desc π)
    /-
      case hm
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.31508, u_1} C
      B : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝⁴ : CategoryTheory.Limits.HasCoproduct X
      inst✝³ : CategoryTheory.EffectiveEpi (CategoryTheory.Limits.Sigma.desc π)
      inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
      inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
      inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
      W✝ : C
      x✝² : (a : α) → Quiver.Hom (X a) W✝
      x✝¹ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a …
      x✝ : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) x✝) (x✝² a)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc π)  …
    -/
    ext
    /-
      case hm.h
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.31508, u_1} C
      B : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      inst✝⁴ : CategoryTheory.Limits.HasCoproduct X
      inst✝³ : CategoryTheory.EffectiveEpi (CategoryTheory.Limits.Sigma.desc π)
      inst✝² : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)) (a :  …
      inst✝¹ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Cate …
      inst✝ : ∀ {Z : C} (g : Quiver.Hom Z (CategoryTheory.Limits.sigmaObj X)), Categ …
      W✝ : C
      x✝² : (a : α) → Quiver.Hom (X a) W✝
      x✝¹ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a …
      x✝ : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (π a) x✝) (x✝² a)
      b✝ : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι X b✝)  …
    -/
    simpa using hm _
    /-
      🎉 no goals
    -/


