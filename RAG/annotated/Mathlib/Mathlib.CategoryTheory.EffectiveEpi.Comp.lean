/--
An effective epi family precomposed by a family of split epis is effective epimorphic.
This version takes an explicit section to the split epis, and is mainly used to define
`effectiveEpiStructCompOfEffectiveEpiSplitEpi`,
which takes a `IsSplitEpi` instance instead.
-/
noncomputable
def effectiveEpiFamilyStructCompOfEffectiveEpiSplitEpi' {α : Type*} {B : C} {X Y : α → C}
    (f : (a : α) → X a ⟶ B) (g : (a : α) → Y a ⟶ X a) (i : (a : α) → X a ⟶ Y a)
    (hi : ∀ a, i a ≫ g a = 𝟙 _) [EffectiveEpiFamily _ f] :
    EffectiveEpiFamilyStruct _ (fun a ↦ g a ≫ f a) where
  desc e w := EffectiveEpiFamily.desc _ f (fun a ↦ i a ≫ e a) fun a₁ a₂ g₁ g₂ _ ↦ (by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      Z✝ : C
      a₁ a₂ : α
      g₁ : Quiver.Hom Z✝ (X a₁)
      g₂ : Quiver.Hom Z✝ (X a₂)
      x✝ : Eq (CategoryTheory.CategoryStruct.comp g₁ (f a₁)) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ ((fun a => CategoryTheory.Category …
    -/
    simp only [← Category.assoc]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      Z✝ : C
      a₁ a₂ : α
      g₁ : Quiver.Hom Z✝ (X a₁)
      g₂ : Quiver.Hom Z✝ (X a₂)
      x✝ : Eq (CategoryTheory.CategoryStruct.comp g₁ (f a₁)) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
    -/
    apply w _ _ (g₁ ≫ i a₁) (g₂ ≫ i a₂)
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      Z✝ : C
      a₁ a₂ : α
      g₁ : Quiver.Hom Z✝ (X a₁)
      g₂ : Quiver.Hom Z✝ (X a₂)
      x✝ : Eq (CategoryTheory.CategoryStruct.comp g₁ (f a₁)) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
    -/
    simp only [Category.assoc, hi]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      Z✝ : C
      a₁ a₂ : α
      g₁ : Quiver.Hom Z✝ (X a₁)
      g₂ : Quiver.Hom Z✝ (X a₂)
      x✝ : Eq (CategoryTheory.CategoryStruct.comp g₁ (f a₁)) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.CategoryStruct.com …
    -/
    simp only [← Category.assoc, hi]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      Z✝ : C
      a₁ a₂ : α
      g₁ : Quiver.Hom Z✝ (X a₁)
      g₂ : Quiver.Hom Z✝ (X a₂)
      x✝ : Eq (CategoryTheory.CategoryStruct.comp g₁ (f a₁)) (CategoryTheory.Categor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
    -/
    simpa)
    /-
      🎉 no goals
    -/
  fac e w a := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      a : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.assoc, EffectiveEpiFamily.fac]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      a : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (g a) (CategoryTheory.CategoryStruct. …
    -/
    rw [← Category.id_comp (e a), ← Category.assoc, ← Category.assoc]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      a : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    apply w
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      a : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Category.comp_id, Category.id_comp, ← Category.assoc]
    /-
      case a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      e : (a : α) → Quiver.Hom (Y a) W✝
      w : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      a : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    aesop
    /-
      🎉 no goals
    -/
  uniq _ _ _ hm := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      x✝² : (a : α) → Quiver.Hom (Y a) W✝
      x✝¹ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a …
      x✝ : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Categor …
      ⊢ Eq x✝ ((fun {W} e w => CategoryTheory.EffectiveEpiFamily.desc X f (fun a =>  …
    -/
    apply EffectiveEpiFamily.uniq _ f
    /-
      case hm
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      x✝² : (a : α) → Quiver.Hom (Y a) W✝
      x✝¹ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a …
      x✝ : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Categor …
      ⊢ ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (f a) x✝) (CategoryTheory. …
    -/
    intro a
    /-
      case hm
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13, u_1} C
      α : Type u_2
      B : C
      X Y : α → C
      f : (a : α) → Quiver.Hom (X a) B
      g : (a : α) → Quiver.Hom (Y a) (X a)
      i : (a : α) → Quiver.Hom (X a) (Y a)
      hi : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (i a) (g a)) (CategoryT …
      inst✝ : CategoryTheory.EffectiveEpiFamily X f
      W✝ : C
      x✝² : (a : α) → Quiver.Hom (Y a) W✝
      x✝¹ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a …
      x✝ : Quiver.Hom B W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Categor …
      a : α
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f a) x✝) (CategoryTheory.CategoryStr …
    -/
    rw [← hm a, ← Category.assoc, ← Category.assoc, hi, Category.id_comp]
    /-
      🎉 no goals
    -/


/--
An effective epi family precomposed with a family of split epis is effective epimorphic.
-/
noncomputable
def effectiveEpiFamilyStructCompOfEffectiveEpiSplitEpi {α : Type*} {B : C} {X Y : α → C}
    (f : (a : α) → X a ⟶ B) (g : (a : α) → Y a ⟶ X a) [∀ a, IsSplitEpi (g a)]
    [EffectiveEpiFamily _ f] : EffectiveEpiFamilyStruct _ (fun a ↦ g a ≫ f a) :=
  effectiveEpiFamilyStructCompOfEffectiveEpiSplitEpi' f g
    (fun a ↦ section_ (g a))
    (fun a ↦ IsSplitEpi.id (g a))


instance {α : Type*} {B : C} {X Y : α → C}
    (f : (a : α) → X a ⟶ B) (g : (a : α) → Y a ⟶ X a) [∀ a, IsSplitEpi (g a)]
    [EffectiveEpiFamily _ f] : EffectiveEpiFamily _ (fun a ↦ g a ≫ f a) :=
  ⟨⟨effectiveEpiFamilyStructCompOfEffectiveEpiSplitEpi f g⟩⟩


instance IsSplitEpi.EffectiveEpi {B X : C} (f : X ⟶ B) [IsSplitEpi f] : EffectiveEpi f := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    B X : C
    f : Quiver.Hom X B
    inst✝ : CategoryTheory.IsSplitEpi f
    ⊢ CategoryTheory.EffectiveEpi f
  -/
  rw [← Category.comp_id f]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    B X : C
    f : Quiver.Hom X B
    inst✝ : CategoryTheory.IsSplitEpi f
    ⊢ CategoryTheory.EffectiveEpi (CategoryTheory.CategoryStruct.comp f (CategoryT …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/--
If a family of morphisms with fixed target, precomposed by a family of epis is
effective epimorphic, then the original family is as well.
-/
noncomputable def effectiveEpiFamilyStructOfComp {C : Type*} [Category C]
    {I : Type*} {Z Y : I → C} {X : C} (g : ∀ i, Z i ⟶ Y i) (f : ∀ i, Y i ⟶ X)
    [EffectiveEpiFamily _ (fun i => g i ≫ f i)] [∀ i, Epi (g i)] :
    EffectiveEpiFamilyStruct _ f where
  desc {W} φ h := EffectiveEpiFamily.desc _ (fun i => g i ≫ f i)
    (fun i => g i ≫ φ i) (fun {T} i₁ i₂ g₁ g₂ eq =>
         /-
           C✝ : Type u_1
           inst✝³ : CategoryTheory.Category.{?u.10244, u_1} C✝
           C : Type u_2
           inst✝² : CategoryTheory.Category.{?u.10251, u_2} C
           I : Type u_3
           Z Y : I → C
           X : C
           g : (i : I) → Quiver.Hom (Z i) (Y i)
           f : (i : I) → Quiver.Hom (Y i) X
           inst✝¹ : CategoryTheory.EffectiveEpiFamily Z fun i => CategoryTheory.CategoryS …
           inst✝ : ∀ (i : I), CategoryTheory.Epi (g i)
           W : C
           φ : (a : I) → Quiver.Hom (Y a) W
           h : ∀ {Z : C} (a₁ a₂ : I) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
           T : C
           i₁ i₂ : I
           g₁ : Quiver.Hom T (Z i₁)
           g₂ : Quiver.Hom T (Z i₂)
           eq : Eq (CategoryTheory.CategoryStruct.comp g₁ ((fun i => CategoryTheory.Categ …
           ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ ((fun i => CategoryTheory.Category …
         -/
      by simpa [assoc] using h i₁ i₂ (g₁ ≫ g i₁) (g₂ ≫ g i₂) (by simpa [assoc] using eq))
         /-
           🎉 no goals
         -/
  fac {W} φ h i := by
    /-
      C✝ : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.10244, u_1} C✝
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.10251, u_2} C
      I : Type u_3
      Z Y : I → C
      X : C
      g : (i : I) → Quiver.Hom (Z i) (Y i)
      f : (i : I) → Quiver.Hom (Y i) X
      inst✝¹ : CategoryTheory.EffectiveEpiFamily Z fun i => CategoryTheory.CategoryS …
      inst✝ : ∀ (i : I), CategoryTheory.Epi (g i)
      W : C
      φ : (a : I) → Quiver.Hom (Y a) W
      h : ∀ {Z : C} (a₁ a₂ : I) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      i : I
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f i) ((fun {W} φ h => CategoryTheory …
    -/
    dsimp
    /-
      C✝ : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.10244, u_1} C✝
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.10251, u_2} C
      I : Type u_3
      Z Y : I → C
      X : C
      g : (i : I) → Quiver.Hom (Z i) (Y i)
      f : (i : I) → Quiver.Hom (Y i) X
      inst✝¹ : CategoryTheory.EffectiveEpiFamily Z fun i => CategoryTheory.CategoryS …
      inst✝ : ∀ (i : I), CategoryTheory.Epi (g i)
      W : C
      φ : (a : I) → Quiver.Hom (Y a) W
      h : ∀ {Z : C} (a₁ a₂ : I) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂) …
      i : I
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f i) (CategoryTheory.EffectiveEpiFam …
    -/
    rw [← cancel_epi (g i), ← assoc, EffectiveEpiFamily.fac _ (fun i => g i ≫ f i)]
    /-
      🎉 no goals
    -/
  uniq {W} φ _ m hm := EffectiveEpiFamily.uniq _ (fun i => g i ≫ f i) _ _ _
                 /-
                   C✝ : Type u_1
                   inst✝³ : CategoryTheory.Category.{?u.10244, u_1} C✝
                   C : Type u_2
                   inst✝² : CategoryTheory.Category.{?u.10251, u_2} C
                   I : Type u_3
                   Z Y : I → C
                   X : C
                   g : (i : I) → Quiver.Hom (Z i) (Y i)
                   f : (i : I) → Quiver.Hom (Y i) X
                   inst✝¹ : CategoryTheory.EffectiveEpiFamily Z fun i => CategoryTheory.CategoryS …
                   inst✝ : ∀ (i : I), CategoryTheory.Epi (g i)
                   W : C
                   φ : (a : I) → Quiver.Hom (Y a) W
                   x✝ : ∀ {Z : C} (a₁ a₂ : I) (g₁ : Quiver.Hom Z (Y a₁)) (g₂ : Quiver.Hom Z (Y a₂ …
                   m : Quiver.Hom X W
                   hm : ∀ (a : I), Eq (CategoryTheory.CategoryStruct.comp (f a) m) (φ a)
                   i : I
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => CategoryTheory.CategoryStr …
                 -/
    (fun i => by rw [assoc, hm])
                 /-
                   🎉 no goals
                 -/


lemma effectiveEpiFamily_of_effectiveEpi_epi_comp {α : Type*} {B : C} {X Y : α → C}
    (f : (a : α) → X a ⟶ B) (g : (a : α) → Y a ⟶ X a) [∀ a, Epi (g a)]
    [EffectiveEpiFamily _ (fun a ↦ g a ≫ f a)] : EffectiveEpiFamily _ f :=
  ⟨⟨effectiveEpiFamilyStructOfComp g f⟩⟩


lemma effectiveEpi_of_effectiveEpi_epi_comp {B X Y : C} (f : X ⟶ B) (g : Y ⟶ X)
    [Epi g] [EffectiveEpi (g ≫ f)] : EffectiveEpi f :=
  have := (effectiveEpi_iff_effectiveEpiFamily (g ≫ f)).mp inferInstance
  have := effectiveEpiFamily_of_effectiveEpi_epi_comp
    (X := fun () ↦ X) (Y := fun () ↦ Y) (fun () ↦ f) (fun () ↦ g)
  inferInstance


theorem effectiveEpiFamilyStructCompIso_aux
    {W : C} (e : (a : α) → X a ⟶ W)
    (h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂),
      g₁ ≫ π a₁ ≫ i = g₂ ≫ π a₂ ≫ i → g₁ ≫ e a₁ = g₂ ≫ e a₂)
    {Z : C} (a₁ a₂ : α) (g₁ : Z ⟶ X a₁) (g₂ : Z ⟶ X a₂) (hg : g₁ ≫ π a₁ = g₂ ≫ π a₂) :
    g₁ ≫ e a₁ = g₂ ≫ e a₂ := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    B B' : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    i : Quiver.Hom B B'
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    a₁ a₂ : α
    g₁ : Quiver.Hom Z (X a₁)
    g₂ : Quiver.Hom Z (X a₂)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (π a₁)) (CategoryTheory.Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (e a₁)) (CategoryTheory.CategorySt …
  -/
  apply h
  /-
    case a
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    B B' : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    i : Quiver.Hom B B'
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    a₁ a₂ : α
    g₁ : Quiver.Hom Z (X a₁)
    g₂ : Quiver.Hom Z (X a₂)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (π a₁)) (CategoryTheory.Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.CategoryStruct.com …
  -/
  rw [← Category.assoc, hg]
  /-
    case a
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_3, u_1} C
    B B' : C
    α : Type u_2
    X : α → C
    π : (a : α) → Quiver.Hom (X a) B
    i : Quiver.Hom B B'
    W : C
    e : (a : α) → Quiver.Hom (X a) W
    h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
    Z : C
    a₁ a₂ : α
    g₁ : Quiver.Hom Z (X a₁)
    g₂ : Quiver.Hom Z (X a₂)
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ (π a₁)) (CategoryTheory.Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An effective epi family followed by an iso is an effective epi family. -/
noncomputable
def effectiveEpiFamilyStructCompIso : EffectiveEpiFamilyStruct X (fun a ↦ π a ≫ i) where
  desc e h := inv i ≫ EffectiveEpiFamily.desc X π e (effectiveEpiFamilyStructCompIso_aux X π i e h)
                  /-
                    C : Type u_1
                    inst✝² : CategoryTheory.Category.{?u.17359, u_1} C
                    B B' : C
                    α : Type u_2
                    X : α → C
                    π : (a : α) → Quiver.Hom (X a) B
                    i : Quiver.Hom B B'
                    inst✝¹ : CategoryTheory.EffectiveEpiFamily X π
                    inst✝ : CategoryTheory.IsIso i
                    W✝ : C
                    x✝² : (a : α) → Quiver.Hom (X a) W✝
                    x✝¹ : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a …
                    x✝ : α
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                  -/
  fac _ _ _ := by simp
                  /-
                    🎉 no goals
                  -/
  uniq e h m hm := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.17359, u_1} C
      B B' : C
      α : Type u_2
      X : α → C
      π : (a : α) → Quiver.Hom (X a) B
      i : Quiver.Hom B B'
      inst✝¹ : CategoryTheory.EffectiveEpiFamily X π
      inst✝ : CategoryTheory.IsIso i
      W✝ : C
      e : (a : α) → Quiver.Hom (X a) W✝
      h : ∀ {Z : C} (a₁ a₂ : α) (g₁ : Quiver.Hom Z (X a₁)) (g₂ : Quiver.Hom Z (X a₂) …
      m : Quiver.Hom B' W✝
      hm : ∀ (a : α), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Categor …
      ⊢ Eq m ((fun {W} e h => CategoryTheory.CategoryStruct.comp (CategoryTheory.inv …
    -/
    simp only [Category.assoc] at hm
    simp [← EffectiveEpiFamily.uniq X π e
      (effectiveEpiFamilyStructCompIso_aux X π i e h) (i ≫ m) hm]


instance : EffectiveEpiFamily X (fun a ↦ π a ≫ i) := ⟨⟨effectiveEpiFamilyStructCompIso X π i⟩⟩


