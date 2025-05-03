/-- A bicategory is called `Strict` if the left unitors, the right unitors, and the associators are
isomorphisms given by equalities.
-/
class Bicategory.Strict : Prop where
  /-- Identity morphisms are left identities for composition. -/
  id_comp : ∀ {a b : B} (f : a ⟶ b), 𝟙 a ≫ f = f := by aesop_cat
  /-- Identity morphisms are right identities for composition. -/
  comp_id : ∀ {a b : B} (f : a ⟶ b), f ≫ 𝟙 b = f := by aesop_cat
  /-- Composition in a bicategory is associative. -/
  assoc : ∀ {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d), (f ≫ g) ≫ h = f ≫ g ≫ h := by
    aesop_cat
  /-- The left unitors are given by equalities -/
  leftUnitor_eqToIso : ∀ {a b : B} (f : a ⟶ b), λ_ f = eqToIso (id_comp f) := by aesop_cat
  /-- The right unitors are given by equalities -/
  rightUnitor_eqToIso : ∀ {a b : B} (f : a ⟶ b), ρ_ f = eqToIso (comp_id f) := by aesop_cat
  /-- The associators are given by equalities -/
  associator_eqToIso :
    ∀ {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d), α_ f g h = eqToIso (assoc f g h) := by
    aesop_cat

-- see Note [lower instance priority]

/-- Category structure on a strict bicategory -/
instance (priority := 100) StrictBicategory.category [Bicategory.Strict B] : Category B where
  id_comp := Bicategory.Strict.id_comp
  comp_id := Bicategory.Strict.comp_id
  assoc := Bicategory.Strict.assoc


@[simp]
theorem whiskerLeft_eqToHom {a b c : B} (f : a ⟶ b) {g h : b ⟶ c} (η : g = h) :
    f ◁ eqToHom η = eqToHom (congr_arg₂ (· ≫ ·) rfl η) := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    f : Quiver.Hom a b
    g h : Quiver.Hom b c
    η : Eq g h
    ⊢ Eq (CategoryTheory.Bicategory.whiskerLeft f (CategoryTheory.eqToHom η)) (Cat …
  -/
  cases η
  /-
    case refl
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    ⊢ Eq (CategoryTheory.Bicategory.whiskerLeft f (CategoryTheory.eqToHom ⋯)) (Cat …
  -/
  simp only [whiskerLeft_id, eqToHom_refl]
  /-
    🎉 no goals
  -/


@[simp]
theorem eqToHom_whiskerRight {a b c : B} {f g : a ⟶ b} (η : f = g) (h : b ⟶ c) :
    eqToHom η ▷ h = eqToHom (congr_arg₂ (· ≫ ·) η rfl) := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    f g : Quiver.Hom a b
    η : Eq f g
    h : Quiver.Hom b c
    ⊢ Eq (CategoryTheory.Bicategory.whiskerRight (CategoryTheory.eqToHom η) h) (Ca …
  -/
  cases η
  /-
    case refl
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    f : Quiver.Hom a b
    h : Quiver.Hom b c
    ⊢ Eq (CategoryTheory.Bicategory.whiskerRight (CategoryTheory.eqToHom ⋯) h) (Ca …
  -/
  simp only [id_whiskerRight, eqToHom_refl]
  /-
    🎉 no goals
  -/


