/-- In a bicategory, we can compose the 1-morphisms `f : a ⟶ b` and `g : b ⟶ c` to obtain
a 1-morphism `f ≫ g : a ⟶ c`. This composition does not need to be strictly associative,
but there is a specified associator, `α_ f g h : (f ≫ g) ≫ h ≅ f ≫ (g ≫ h)`.
There is an identity 1-morphism `𝟙 a : a ⟶ a`, with specified left and right unitor
isomorphisms `λ_ f : 𝟙 a ≫ f ≅ f` and `ρ_ f : f ≫ 𝟙 a ≅ f`.
These associators and unitors satisfy the pentagon and triangle equations.

See https://ncatlab.org/nlab/show/bicategory.
-/
@[nolint checkUnivs]
class Bicategory (B : Type u) extends CategoryStruct.{v} B where
  -- category structure on the collection of 1-morphisms:
  homCategory : ∀ a b : B, Category.{w} (a ⟶ b) := by infer_instance
  -- left whiskering:
  whiskerLeft {a b c : B} (f : a ⟶ b) {g h : b ⟶ c} (η : g ⟶ h) : f ≫ g ⟶ f ≫ h
  -- right whiskering:
  whiskerRight {a b c : B} {f g : a ⟶ b} (η : f ⟶ g) (h : b ⟶ c) : f ≫ h ⟶ g ≫ h
  -- associator:
  associator {a b c d : B} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) : (f ≫ g) ≫ h ≅ f ≫ g ≫ h
  -- left unitor:
  leftUnitor {a b : B} (f : a ⟶ b) : 𝟙 a ≫ f ≅ f
  -- right unitor:
  rightUnitor {a b : B} (f : a ⟶ b) : f ≫ 𝟙 b ≅ f
  -- axioms for left whiskering:
  whiskerLeft_id : ∀ {a b c} (f : a ⟶ b) (g : b ⟶ c), whiskerLeft f (𝟙 g) = 𝟙 (f ≫ g) := by
    aesop_cat
  whiskerLeft_comp :
    ∀ {a b c} (f : a ⟶ b) {g h i : b ⟶ c} (η : g ⟶ h) (θ : h ⟶ i),
      whiskerLeft f (η ≫ θ) = whiskerLeft f η ≫ whiskerLeft f θ := by
    aesop_cat
  id_whiskerLeft :
    ∀ {a b} {f g : a ⟶ b} (η : f ⟶ g),
      whiskerLeft (𝟙 a) η = (leftUnitor f).hom ≫ η ≫ (leftUnitor g).inv := by
    aesop_cat
  comp_whiskerLeft :
    ∀ {a b c d} (f : a ⟶ b) (g : b ⟶ c) {h h' : c ⟶ d} (η : h ⟶ h'),
      whiskerLeft (f ≫ g) η =
        (associator f g h).hom ≫ whiskerLeft f (whiskerLeft g η) ≫ (associator f g h').inv := by
    aesop_cat
  -- axioms for right whiskering:
  id_whiskerRight : ∀ {a b c} (f : a ⟶ b) (g : b ⟶ c), whiskerRight (𝟙 f) g = 𝟙 (f ≫ g) := by
    aesop_cat
  comp_whiskerRight :
    ∀ {a b c} {f g h : a ⟶ b} (η : f ⟶ g) (θ : g ⟶ h) (i : b ⟶ c),
      whiskerRight (η ≫ θ) i = whiskerRight η i ≫ whiskerRight θ i := by
    aesop_cat
  whiskerRight_id :
    ∀ {a b} {f g : a ⟶ b} (η : f ⟶ g),
      whiskerRight η (𝟙 b) = (rightUnitor f).hom ≫ η ≫ (rightUnitor g).inv := by
    aesop_cat
  whiskerRight_comp :
    ∀ {a b c d} {f f' : a ⟶ b} (η : f ⟶ f') (g : b ⟶ c) (h : c ⟶ d),
      whiskerRight η (g ≫ h) =
        (associator f g h).inv ≫ whiskerRight (whiskerRight η g) h ≫ (associator f' g h).hom := by
    aesop_cat
  -- associativity of whiskerings:
  whisker_assoc :
    ∀ {a b c d} (f : a ⟶ b) {g g' : b ⟶ c} (η : g ⟶ g') (h : c ⟶ d),
      whiskerRight (whiskerLeft f η) h =
        (associator f g h).hom ≫ whiskerLeft f (whiskerRight η h) ≫ (associator f g' h).inv := by
    aesop_cat
  -- exchange law of left and right whiskerings:
  whisker_exchange :
    ∀ {a b c} {f g : a ⟶ b} {h i : b ⟶ c} (η : f ⟶ g) (θ : h ⟶ i),
      whiskerLeft f θ ≫ whiskerRight η i = whiskerRight η h ≫ whiskerLeft g θ := by
    aesop_cat
  -- pentagon identity:
  pentagon :
    ∀ {a b c d e} (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e),
      whiskerRight (associator f g h).hom i ≫
          (associator f (g ≫ h) i).hom ≫ whiskerLeft f (associator g h i).hom =
        (associator (f ≫ g) h i).hom ≫ (associator f g (h ≫ i)).hom := by
    aesop_cat
  -- triangle identity:
  triangle :
    ∀ {a b c} (f : a ⟶ b) (g : b ⟶ c),
      (associator f (𝟙 b) g).hom ≫ whiskerLeft f (leftUnitor g).hom
      = whiskerRight (rightUnitor f).hom g := by
    aesop_cat


scoped infixr:81 " ◁ " => Bicategory.whiskerLeft

scoped infixl:81 " ▷ " => Bicategory.whiskerRight

scoped notation "α_" => Bicategory.associator

scoped notation "λ_" => Bicategory.leftUnitor

scoped notation "ρ_" => Bicategory.rightUnitor


attribute [reassoc]
  whiskerLeft_comp id_whiskerLeft comp_whiskerLeft comp_whiskerRight whiskerRight_id
  whiskerRight_comp whisker_assoc whisker_exchange


attribute [reassoc (attr := simp)] pentagon triangle
/-
The following simp attributes are put in order to rewrite any 2-morphisms into normal forms. There
are associators and unitors in the RHS in the several simp lemmas here (e.g. `id_whiskerLeft`),
which at first glance look more complicated than the LHS, but they will be eventually reduced by
the pentagon or the triangle identities, and more generally, (forthcoming) `coherence` tactic.
-/

@[reassoc (attr := simp)]
theorem whiskerLeft_hom_inv (f : a ⟶ b) {g h : b ⟶ c} (η : g ≅ h) :
                                            /-
                                              B : Type u
                                              inst✝ : CategoryTheory.Bicategory B
                                              a b c : B
                                              f : Quiver.Hom a b
                                              g h : Quiver.Hom b c
                                              η : CategoryTheory.Iso g h
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
                                            -/
    f ◁ η.hom ≫ f ◁ η.inv = 𝟙 (f ≫ g) := by rw [← whiskerLeft_comp, hom_inv_id, whiskerLeft_id]
                                            /-
                                              🎉 no goals
                                            -/


@[reassoc (attr := simp)]
theorem hom_inv_whiskerRight {f g : a ⟶ b} (η : f ≅ g) (h : b ⟶ c) :
                                            /-
                                              B : Type u
                                              inst✝ : CategoryTheory.Bicategory B
                                              a b c : B
                                              f g : Quiver.Hom a b
                                              η : CategoryTheory.Iso f g
                                              h : Quiver.Hom b c
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
                                            -/
    η.hom ▷ h ≫ η.inv ▷ h = 𝟙 (f ≫ h) := by rw [← comp_whiskerRight, hom_inv_id, id_whiskerRight]
                                            /-
                                              🎉 no goals
                                            -/


@[reassoc (attr := simp)]
theorem whiskerLeft_inv_hom (f : a ⟶ b) {g h : b ⟶ c} (η : g ≅ h) :
                                            /-
                                              B : Type u
                                              inst✝ : CategoryTheory.Bicategory B
                                              a b c : B
                                              f : Quiver.Hom a b
                                              g h : Quiver.Hom b c
                                              η : CategoryTheory.Iso g h
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
                                            -/
    f ◁ η.inv ≫ f ◁ η.hom = 𝟙 (f ≫ h) := by rw [← whiskerLeft_comp, inv_hom_id, whiskerLeft_id]
                                            /-
                                              🎉 no goals
                                            -/


@[reassoc (attr := simp)]
theorem inv_hom_whiskerRight {f g : a ⟶ b} (η : f ≅ g) (h : b ⟶ c) :
                                            /-
                                              B : Type u
                                              inst✝ : CategoryTheory.Bicategory B
                                              a b c : B
                                              f g : Quiver.Hom a b
                                              η : CategoryTheory.Iso f g
                                              h : Quiver.Hom b c
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
                                            -/
    η.inv ▷ h ≫ η.hom ▷ h = 𝟙 (g ≫ h) := by rw [← comp_whiskerRight, inv_hom_id, id_whiskerRight]
                                            /-
                                              🎉 no goals
                                            -/


/-- The left whiskering of a 2-isomorphism is a 2-isomorphism. -/
@[simps]
def whiskerLeftIso (f : a ⟶ b) {g h : b ⟶ c} (η : g ≅ h) : f ≫ g ≅ f ≫ h where
  hom := f ◁ η.hom
  inv := f ◁ η.inv


instance whiskerLeft_isIso (f : a ⟶ b) {g h : b ⟶ c} (η : g ⟶ h) [IsIso η] : IsIso (f ◁ η) :=
  (whiskerLeftIso f (asIso η)).isIso_hom


@[simp]
theorem inv_whiskerLeft (f : a ⟶ b) {g h : b ⟶ c} (η : g ⟶ h) [IsIso η] :
    inv (f ◁ η) = f ◁ inv η := by
  /-
    B : Type u
    inst✝¹ : CategoryTheory.Bicategory B
    a b c : B
    f : Quiver.Hom a b
    g h : Quiver.Hom b c
    η : Quiver.Hom g h
    inst✝ : CategoryTheory.IsIso η
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.Bicategory.whiskerLeft f η)) (Categor …
  -/
  apply IsIso.inv_eq_of_hom_inv_id
  /-
    case hom_inv_id
    B : Type u
    inst✝¹ : CategoryTheory.Bicategory B
    a b c : B
    f : Quiver.Hom a b
    g h : Quiver.Hom b c
    η : Quiver.Hom g h
    inst✝ : CategoryTheory.IsIso η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp only [← whiskerLeft_comp, whiskerLeft_id, IsIso.hom_inv_id]
  /-
    🎉 no goals
  -/


/-- The right whiskering of a 2-isomorphism is a 2-isomorphism. -/
@[simps!]
def whiskerRightIso {f g : a ⟶ b} (η : f ≅ g) (h : b ⟶ c) : f ≫ h ≅ g ≫ h where
  hom := η.hom ▷ h
  inv := η.inv ▷ h


instance whiskerRight_isIso {f g : a ⟶ b} (η : f ⟶ g) (h : b ⟶ c) [IsIso η] : IsIso (η ▷ h) :=
  (whiskerRightIso (asIso η) h).isIso_hom


@[simp]
theorem inv_whiskerRight {f g : a ⟶ b} (η : f ⟶ g) (h : b ⟶ c) [IsIso η] :
    inv (η ▷ h) = inv η ▷ h := by
  /-
    B : Type u
    inst✝¹ : CategoryTheory.Bicategory B
    a b c : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    h : Quiver.Hom b c
    inst✝ : CategoryTheory.IsIso η
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.Bicategory.whiskerRight η h)) (Catego …
  -/
  apply IsIso.inv_eq_of_hom_inv_id
  /-
    case hom_inv_id
    B : Type u
    inst✝¹ : CategoryTheory.Bicategory B
    a b c : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    h : Quiver.Hom b c
    inst✝ : CategoryTheory.IsIso η
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  simp only [← comp_whiskerRight, id_whiskerRight, IsIso.hom_inv_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_inv (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    f ◁ (α_ g h i).inv ≫ (α_ f (g ≫ h) i).inv ≫ (α_ f g h).inv ▷ i =
      (α_ f g (h ≫ i)).inv ≫ (α_ (f ≫ g) h i).inv :=
                       /-
                         B : Type u
                         inst✝ : CategoryTheory.Bicategory B
                         a b c d e : B
                         f : Quiver.Hom a b
                         g : Quiver.Hom b c
                         h : Quiver.Hom c d
                         i : Quiver.Hom d e
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.B …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem pentagon_inv_inv_hom_hom_inv (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    (α_ f (g ≫ h) i).inv ≫ (α_ f g h).inv ▷ i ≫ (α_ (f ≫ g) h i).hom =
    f ◁ (α_ g h i).hom ≫ (α_ f g (h ≫ i)).inv := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d e : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    i : Quiver.Hom d e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.associator …
  -/
  rw [← cancel_epi (f ◁ (α_ g h i).inv), ← cancel_mono (α_ (f ≫ g) h i).inv]
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d e : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    i : Quiver.Hom d e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_inv_hom_hom_hom_inv (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    (α_ (f ≫ g) h i).inv ≫ (α_ f g h).hom ▷ i ≫ (α_ f (g ≫ h) i).hom =
      (α_ f g (h ≫ i)).hom ≫ f ◁ (α_ g h i).inv :=
                       /-
                         B : Type u
                         inst✝ : CategoryTheory.Bicategory B
                         a b c d e : B
                         f : Quiver.Hom a b
                         g : Quiver.Hom b c
                         h : Quiver.Hom c d
                         i : Quiver.Hom d e
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.B …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem pentagon_hom_inv_inv_inv_inv (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    f ◁ (α_ g h i).hom ≫ (α_ f g (h ≫ i)).inv ≫ (α_ (f ≫ g) h i).inv =
      (α_ f (g ≫ h) i).inv ≫ (α_ f g h).inv ▷ i := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d e : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    i : Quiver.Hom d e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp [← cancel_epi (f ◁ (α_ g h i).inv)]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_hom_hom_inv_hom_hom (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    (α_ (f ≫ g) h i).hom ≫ (α_ f g (h ≫ i)).hom ≫ f ◁ (α_ g h i).inv =
      (α_ f g h).hom ▷ i ≫ (α_ f (g ≫ h) i).hom :=
                       /-
                         B : Type u
                         inst✝ : CategoryTheory.Bicategory B
                         a b c d e : B
                         f : Quiver.Hom a b
                         g : Quiver.Hom b c
                         h : Quiver.Hom c d
                         i : Quiver.Hom d e
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.B …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem pentagon_hom_inv_inv_inv_hom (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    (α_ f g (h ≫ i)).hom ≫ f ◁ (α_ g h i).inv ≫ (α_ f (g ≫ h) i).inv =
    (α_ (f ≫ g) h i).inv ≫ (α_ f g h).hom ▷ i := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d e : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    i : Quiver.Hom d e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.associator …
  -/
  rw [← cancel_epi (α_ f g (h ≫ i)).inv, ← cancel_mono ((α_ f g h).inv ▷ i)]
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d e : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    i : Quiver.Hom d e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_hom_hom_inv_inv_hom (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    (α_ f (g ≫ h) i).hom ≫ f ◁ (α_ g h i).hom ≫ (α_ f g (h ≫ i)).inv =
      (α_ f g h).inv ▷ i ≫ (α_ (f ≫ g) h i).hom :=
                       /-
                         B : Type u
                         inst✝ : CategoryTheory.Bicategory B
                         a b c d e : B
                         f : Quiver.Hom a b
                         g : Quiver.Hom b c
                         h : Quiver.Hom c d
                         i : Quiver.Hom d e
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.B …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc (attr := simp)]
theorem pentagon_inv_hom_hom_hom_hom (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    (α_ f g h).inv ▷ i ≫ (α_ (f ≫ g) h i).hom ≫ (α_ f g (h ≫ i)).hom =
      (α_ f (g ≫ h) i).hom ≫ f ◁ (α_ g h i).hom := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d e : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    i : Quiver.Hom d e
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  simp [← cancel_epi ((α_ f g h).hom ▷ i)]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pentagon_inv_inv_hom_inv_inv (f : a ⟶ b) (g : b ⟶ c) (h : c ⟶ d) (i : d ⟶ e) :
    (α_ f g (h ≫ i)).inv ≫ (α_ (f ≫ g) h i).inv ≫ (α_ f g h).hom ▷ i =
      f ◁ (α_ g h i).inv ≫ (α_ f (g ≫ h) i).inv :=
                       /-
                         B : Type u
                         inst✝ : CategoryTheory.Bicategory B
                         a b c d e : B
                         f : Quiver.Hom a b
                         g : Quiver.Hom b c
                         h : Quiver.Hom c d
                         i : Quiver.Hom d e
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.CategoryStruct.comp (CategoryTheory.B …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


theorem triangle_assoc_comp_left (f : a ⟶ b) (g : b ⟶ c) :
    (α_ f (𝟙 b) g).hom ≫ f ◁ (λ_ g).hom = (ρ_ f).hom ▷ g :=
  triangle f g


@[reassoc (attr := simp)]
theorem triangle_assoc_comp_right (f : a ⟶ b) (g : b ⟶ c) :
                                                               /-
                                                                 B : Type u
                                                                 inst✝ : CategoryTheory.Bicategory B
                                                                 a b c : B
                                                                 f : Quiver.Hom a b
                                                                 g : Quiver.Hom b c
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.associator …
                                                               -/
    (α_ f (𝟙 b) g).inv ≫ (ρ_ f).hom ▷ g = f ◁ (λ_ g).hom := by rw [← triangle, inv_hom_id_assoc]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[reassoc (attr := simp)]
theorem triangle_assoc_comp_right_inv (f : a ⟶ b) (g : b ⟶ c) :
    (ρ_ f).inv ▷ g ≫ (α_ f (𝟙 b) g).hom = f ◁ (λ_ g).inv := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
  -/
  simp [← cancel_mono (f ◁ (λ_ g).hom)]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem triangle_assoc_comp_left_inv (f : a ⟶ b) (g : b ⟶ c) :
    f ◁ (λ_ g).inv ≫ (α_ f (𝟙 b) g).inv = (ρ_ f).inv ▷ g := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp [← cancel_mono ((ρ_ f).hom ▷ g)]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem associator_naturality_left {f f' : a ⟶ b} (η : f ⟶ f') (g : b ⟶ c) (h : c ⟶ d) :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f f' : Quiver.Hom a b
                                                                       η : Quiver.Hom f f'
                                                                       g : Quiver.Hom b c
                                                                       h : Quiver.Hom c d
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
                                                                     -/
    η ▷ g ▷ h ≫ (α_ f' g h).hom = (α_ f g h).hom ≫ η ▷ (g ≫ h) := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_inv_naturality_left {f f' : a ⟶ b} (η : f ⟶ f') (g : b ⟶ c) (h : c ⟶ d) :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f f' : Quiver.Hom a b
                                                                       η : Quiver.Hom f f'
                                                                       g : Quiver.Hom b c
                                                                       h : Quiver.Hom c d
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
                                                                     -/
    η ▷ (g ≫ h) ≫ (α_ f' g h).inv = (α_ f g h).inv ≫ η ▷ g ▷ h := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem whiskerRight_comp_symm {f f' : a ⟶ b} (η : f ⟶ f') (g : b ⟶ c) (h : c ⟶ d) :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f f' : Quiver.Hom a b
                                                                       η : Quiver.Hom f f'
                                                                       g : Quiver.Hom b c
                                                                       h : Quiver.Hom c d
                                                                       ⊢ Eq (CategoryTheory.Bicategory.whiskerRight (CategoryTheory.Bicategory.whiske …
                                                                     -/
    η ▷ g ▷ h = (α_ f g h).hom ≫ η ▷ (g ≫ h) ≫ (α_ f' g h).inv := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_naturality_middle (f : a ⟶ b) {g g' : b ⟶ c} (η : g ⟶ g') (h : c ⟶ d) :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f : Quiver.Hom a b
                                                                       g g' : Quiver.Hom b c
                                                                       η : Quiver.Hom g g'
                                                                       h : Quiver.Hom c d
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
                                                                     -/
    (f ◁ η) ▷ h ≫ (α_ f g' h).hom = (α_ f g h).hom ≫ f ◁ η ▷ h := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_inv_naturality_middle (f : a ⟶ b) {g g' : b ⟶ c} (η : g ⟶ g') (h : c ⟶ d) :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f : Quiver.Hom a b
                                                                       g g' : Quiver.Hom b c
                                                                       η : Quiver.Hom g g'
                                                                       h : Quiver.Hom c d
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
                                                                     -/
    f ◁ η ▷ h ≫ (α_ f g' h).inv = (α_ f g h).inv ≫ (f ◁ η) ▷ h := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem whisker_assoc_symm (f : a ⟶ b) {g g' : b ⟶ c} (η : g ⟶ g') (h : c ⟶ d) :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f : Quiver.Hom a b
                                                                       g g' : Quiver.Hom b c
                                                                       η : Quiver.Hom g g'
                                                                       h : Quiver.Hom c d
                                                                       ⊢ Eq (CategoryTheory.Bicategory.whiskerLeft f (CategoryTheory.Bicategory.whisk …
                                                                     -/
    f ◁ η ▷ h = (α_ f g h).inv ≫ (f ◁ η) ▷ h ≫ (α_ f g' h).hom := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_naturality_right (f : a ⟶ b) (g : b ⟶ c) {h h' : c ⟶ d} (η : h ⟶ h') :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f : Quiver.Hom a b
                                                                       g : Quiver.Hom b c
                                                                       h h' : Quiver.Hom c d
                                                                       η : Quiver.Hom h h'
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
                                                                     -/
    (f ≫ g) ◁ η ≫ (α_ f g h').hom = (α_ f g h).hom ≫ f ◁ g ◁ η := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem associator_inv_naturality_right (f : a ⟶ b) (g : b ⟶ c) {h h' : c ⟶ d} (η : h ⟶ h') :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f : Quiver.Hom a b
                                                                       g : Quiver.Hom b c
                                                                       h h' : Quiver.Hom c d
                                                                       η : Quiver.Hom h h'
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
                                                                     -/
    f ◁ g ◁ η ≫ (α_ f g h').inv = (α_ f g h).inv ≫ (f ≫ g) ◁ η := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem comp_whiskerLeft_symm (f : a ⟶ b) (g : b ⟶ c) {h h' : c ⟶ d} (η : h ⟶ h') :
                                                                     /-
                                                                       B : Type u
                                                                       inst✝ : CategoryTheory.Bicategory B
                                                                       a b c d : B
                                                                       f : Quiver.Hom a b
                                                                       g : Quiver.Hom b c
                                                                       h h' : Quiver.Hom c d
                                                                       η : Quiver.Hom h h'
                                                                       ⊢ Eq (CategoryTheory.Bicategory.whiskerLeft f (CategoryTheory.Bicategory.whisk …
                                                                     -/
    f ◁ g ◁ η = (α_ f g h).inv ≫ (f ≫ g) ◁ η ≫ (α_ f g h').hom := by simp
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[reassoc]
theorem leftUnitor_naturality {f g : a ⟶ b} (η : f ⟶ g) :
    𝟙 a ◁ η ≫ (λ_ g).hom = (λ_ f).hom ≫ η := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem leftUnitor_inv_naturality {f g : a ⟶ b} (η : f ⟶ g) :
                                                /-
                                                  B : Type u
                                                  inst✝ : CategoryTheory.Bicategory B
                                                  a b : B
                                                  f g : Quiver.Hom a b
                                                  η : Quiver.Hom f g
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp η (CategoryTheory.Bicategory.leftUnit …
                                                -/
    η ≫ (λ_ g).inv = (λ_ f).inv ≫ 𝟙 a ◁ η := by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem id_whiskerLeft_symm {f g : a ⟶ b} (η : f ⟶ g) : η = (λ_ f).inv ≫ 𝟙 a ◁ η ≫ (λ_ g).hom := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    ⊢ Eq η (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.leftUnit …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem rightUnitor_naturality {f g : a ⟶ b} (η : f ⟶ g) :
                                                /-
                                                  B : Type u
                                                  inst✝ : CategoryTheory.Bicategory B
                                                  a b : B
                                                  f g : Quiver.Hom a b
                                                  η : Quiver.Hom f g
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerRig …
                                                -/
    η ▷ 𝟙 b ≫ (ρ_ g).hom = (ρ_ f).hom ≫ η := by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[reassoc]
theorem rightUnitor_inv_naturality {f g : a ⟶ b} (η : f ⟶ g) :
                                                /-
                                                  B : Type u
                                                  inst✝ : CategoryTheory.Bicategory B
                                                  a b : B
                                                  f g : Quiver.Hom a b
                                                  η : Quiver.Hom f g
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp η (CategoryTheory.Bicategory.rightUni …
                                                -/
    η ≫ (ρ_ g).inv = (ρ_ f).inv ≫ η ▷ 𝟙 b := by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem whiskerRight_id_symm {f g : a ⟶ b} (η : f ⟶ g) : η = (ρ_ f).inv ≫ η ▷ 𝟙 b ≫ (ρ_ g).hom := by
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b : B
    f g : Quiver.Hom a b
    η : Quiver.Hom f g
    ⊢ Eq η (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.rightUni …
  -/
  simp
  /-
    🎉 no goals
  -/


                                                                                      /-
                                                                                        B : Type u
                                                                                        inst✝ : CategoryTheory.Bicategory B
                                                                                        a b : B
                                                                                        f g : Quiver.Hom a b
                                                                                        η θ : Quiver.Hom f g
                                                                                        ⊢ Iff (Eq (CategoryTheory.Bicategory.whiskerLeft (CategoryTheory.CategoryStruc …
                                                                                      -/
theorem whiskerLeft_iff {f g : a ⟶ b} (η θ : f ⟶ g) : 𝟙 a ◁ η = 𝟙 a ◁ θ ↔ η = θ := by simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


                                                                                       /-
                                                                                         B : Type u
                                                                                         inst✝ : CategoryTheory.Bicategory B
                                                                                         a b : B
                                                                                         f g : Quiver.Hom a b
                                                                                         η θ : Quiver.Hom f g
                                                                                         ⊢ Iff (Eq (CategoryTheory.Bicategory.whiskerRight η (CategoryTheory.CategorySt …
                                                                                       -/
theorem whiskerRight_iff {f g : a ⟶ b} (η θ : f ⟶ g) : η ▷ 𝟙 b = θ ▷ 𝟙 b ↔ η = θ := by simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


/-- We state it as a simp lemma, which is regarded as an involved version of
`id_whiskerRight f g : 𝟙 f ▷ g = 𝟙 (f ≫ g)`.
-/
@[reassoc, simp]
theorem leftUnitor_whiskerRight (f : a ⟶ b) (g : b ⟶ c) :
    (λ_ f).hom ▷ g = (α_ (𝟙 a) f g).hom ≫ (λ_ (f ≫ g)).hom := by
  rw [← whiskerLeft_iff, whiskerLeft_comp, ← cancel_epi (α_ _ _ _).hom, ←
      cancel_epi ((α_ _ _ _).hom ▷ _), pentagon_assoc, triangle, ← associator_naturality_middle, ←
      comp_whiskerRight_assoc, triangle, associator_naturality_left]


@[reassoc, simp]
theorem leftUnitor_inv_whiskerRight (f : a ⟶ b) (g : b ⟶ c) :
    (λ_ f).inv ▷ g = (λ_ (f ≫ g)).inv ≫ (α_ (𝟙 a) f g).inv :=
                       /-
                         B : Type u
                         inst✝ : CategoryTheory.Bicategory B
                         a b c : B
                         f : Quiver.Hom a b
                         g : Quiver.Hom b c
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.Bicategory.whiskerRight (CategoryTheo …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/


@[reassoc, simp]
theorem whiskerLeft_rightUnitor (f : a ⟶ b) (g : b ⟶ c) :
    f ◁ (ρ_ g).hom = (α_ f g (𝟙 c)).inv ≫ (ρ_ (f ≫ g)).hom := by
  rw [← whiskerRight_iff, comp_whiskerRight, ← cancel_epi (α_ _ _ _).inv, ←
      cancel_epi (f ◁ (α_ _ _ _).inv), pentagon_inv_assoc, triangle_assoc_comp_right, ←
      associator_inv_naturality_middle, ← whiskerLeft_comp_assoc, triangle_assoc_comp_right,
      associator_inv_naturality_right]


@[reassoc, simp]
theorem whiskerLeft_rightUnitor_inv (f : a ⟶ b) (g : b ⟶ c) :
    f ◁ (ρ_ g).inv = (ρ_ (f ≫ g)).inv ≫ (α_ f g (𝟙 c)).hom :=
                       /-
                         B : Type u
                         inst✝ : CategoryTheory.Bicategory B
                         a b c : B
                         f : Quiver.Hom a b
                         g : Quiver.Hom b c
                         ⊢ Eq (CategoryTheory.inv (CategoryTheory.Bicategory.whiskerLeft f (CategoryThe …
                       -/
  eq_of_inv_eq_inv (by simp)
                       /-
                         🎉 no goals
                       -/

/-
It is not so obvious whether `leftUnitor_whiskerRight` or `leftUnitor_comp` should be a simp
lemma. Our choice is the former. One reason is that the latter yields the following loop:
[id_whiskerLeft]   : 𝟙 a ◁ (ρ_ f).hom ==> (λ_ (f ≫ 𝟙 b)).hom ≫ (ρ_ f).hom ≫ (λ_ f).inv
[leftUnitor_comp]  : (λ_ (f ≫ 𝟙 b)).hom ==> (α_ (𝟙 a) f (𝟙 b)).inv ≫ (λ_ f).hom ▷ 𝟙 b
[whiskerRight_id]  : (λ_ f).hom ▷ 𝟙 b ==> (ρ_ (𝟙 a ≫ f)).hom ≫ (λ_ f).hom ≫ (ρ_ f).inv
[rightUnitor_comp] : (ρ_ (𝟙 a ≫ f)).hom ==> (α_ (𝟙 a) f (𝟙 b)).hom ≫ 𝟙 a ◁ (ρ_ f).hom
-/

@[reassoc]
theorem leftUnitor_comp (f : a ⟶ b) (g : b ⟶ c) :
                                                                 /-
                                                                   B : Type u
                                                                   inst✝ : CategoryTheory.Bicategory B
                                                                   a b c : B
                                                                   f : Quiver.Hom a b
                                                                   g : Quiver.Hom b c
                                                                   ⊢ Eq (CategoryTheory.Bicategory.leftUnitor (CategoryTheory.CategoryStruct.comp …
                                                                 -/
    (λ_ (f ≫ g)).hom = (α_ (𝟙 a) f g).inv ≫ (λ_ f).hom ▷ g := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[reassoc]
theorem leftUnitor_comp_inv (f : a ⟶ b) (g : b ⟶ c) :
                                                                 /-
                                                                   B : Type u
                                                                   inst✝ : CategoryTheory.Bicategory B
                                                                   a b c : B
                                                                   f : Quiver.Hom a b
                                                                   g : Quiver.Hom b c
                                                                   ⊢ Eq (CategoryTheory.Bicategory.leftUnitor (CategoryTheory.CategoryStruct.comp …
                                                                 -/
    (λ_ (f ≫ g)).inv = (λ_ f).inv ▷ g ≫ (α_ (𝟙 a) f g).hom := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[reassoc]
theorem rightUnitor_comp (f : a ⟶ b) (g : b ⟶ c) :
                                                                 /-
                                                                   B : Type u
                                                                   inst✝ : CategoryTheory.Bicategory B
                                                                   a b c : B
                                                                   f : Quiver.Hom a b
                                                                   g : Quiver.Hom b c
                                                                   ⊢ Eq (CategoryTheory.Bicategory.rightUnitor (CategoryTheory.CategoryStruct.com …
                                                                 -/
    (ρ_ (f ≫ g)).hom = (α_ f g (𝟙 c)).hom ≫ f ◁ (ρ_ g).hom := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[reassoc]
theorem rightUnitor_comp_inv (f : a ⟶ b) (g : b ⟶ c) :
                                                                 /-
                                                                   B : Type u
                                                                   inst✝ : CategoryTheory.Bicategory B
                                                                   a b c : B
                                                                   f : Quiver.Hom a b
                                                                   g : Quiver.Hom b c
                                                                   ⊢ Eq (CategoryTheory.Bicategory.rightUnitor (CategoryTheory.CategoryStruct.com …
                                                                 -/
    (ρ_ (f ≫ g)).inv = f ◁ (ρ_ g).inv ≫ (α_ f g (𝟙 c)).inv := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem unitors_equal : (λ_ (𝟙 a)).hom = (ρ_ (𝟙 a)).hom := by
  rw [← whiskerLeft_iff, ← cancel_epi (α_ _ _ _).hom, ← cancel_mono (ρ_ _).hom, triangle, ←
      rightUnitor_comp, rightUnitor_naturality]


@[simp]
                                                                  /-
                                                                    B : Type u
                                                                    inst✝ : CategoryTheory.Bicategory B
                                                                    a : B
                                                                    ⊢ Eq (CategoryTheory.Bicategory.leftUnitor (CategoryTheory.CategoryStruct.id a …
                                                                  -/
theorem unitors_inv_equal : (λ_ (𝟙 a)).inv = (ρ_ (𝟙 a)).inv := by simp [Iso.inv_eq_inv]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Precomposition of a 1-morphism as a functor. -/
@[simps]
def precomp (c : B) (f : a ⟶ b) : (b ⟶ c) ⥤ (a ⟶ c) where
  obj := (f ≫ ·)
  map := (f ◁ ·)


/-- Precomposition of a 1-morphism as a functor from the category of 1-morphisms `a ⟶ b` into the
category of functors `(b ⟶ c) ⥤ (a ⟶ c)`. -/
@[simps]
def precomposing (a b c : B) : (a ⟶ b) ⥤ (b ⟶ c) ⥤ (a ⟶ c) where
  obj f := precomp c f
  map η := { app := (η ▷ ·) }


/-- Postcomposition of a 1-morphism as a functor. -/
@[simps]
def postcomp (a : B) (f : b ⟶ c) : (a ⟶ b) ⥤ (a ⟶ c) where
  obj := (· ≫ f)
  map := (· ▷ f)


/-- Postcomposition of a 1-morphism as a functor from the category of 1-morphisms `b ⟶ c` into the
category of functors `(a ⟶ b) ⥤ (a ⟶ c)`. -/
@[simps]
def postcomposing (a b c : B) : (b ⟶ c) ⥤ (a ⟶ b) ⥤ (a ⟶ c) where
  obj f := postcomp a f
  map η := { app := (· ◁ η) }


/-- Left component of the associator as a natural isomorphism. -/
@[simps!]
def associatorNatIsoLeft (a : B) (g : b ⟶ c) (h : c ⟶ d) :
    (postcomposing a ..).obj g ⋙ (postcomposing ..).obj h ≅ (postcomposing ..).obj (g ≫ h) :=
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a✝ b c d e a : B
    g : Quiver.Hom b c
    h : Quiver.Hom c d
    ⊢ ∀ {X Y : Quiver.Hom a b} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategorySt …
  -/
  NatIso.ofComponents (α_ · g h)
  /-
    🎉 no goals
  -/


/-- Middle component of the associator as a natural isomorphism. -/
@[simps!]
def associatorNatIsoMiddle (f : a ⟶ b) (h : c ⟶ d) :
    (precomposing ..).obj f ⋙ (postcomposing ..).obj h ≅
      (postcomposing ..).obj h ⋙ (precomposing ..).obj f :=
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d e : B
    f : Quiver.Hom a b
    h : Quiver.Hom c d
    ⊢ ∀ {X Y : Quiver.Hom b c} (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.Category …
  -/
  NatIso.ofComponents (α_ f · h)
  /-
    🎉 no goals
  -/


/-- Right component of the associator as a natural isomorphism. -/
@[simps!]
def associatorNatIsoRight (f : a ⟶ b) (g : b ⟶ c) (d : B) :
    (precomposing _ _ d).obj (f ≫ g) ≅ (precomposing ..).obj g ⋙ (precomposing ..).obj f :=
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a b c d✝ e : B
    f : Quiver.Hom a b
    g : Quiver.Hom b c
    d : B
    ⊢ ∀ {X Y : Quiver.Hom c d} (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.Category …
  -/
  NatIso.ofComponents (α_ f g ·)
  /-
    🎉 no goals
  -/


/-- Left unitor as a natural isomorphism. -/
@[simps!]
def leftUnitorNatIso (a b : B) : (precomposing _ _ b).obj (𝟙 a) ≅ 𝟭 (a ⟶ b) :=
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a✝ b✝ c d e a b : B
    ⊢ ∀ {X Y : Quiver.Hom a b} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategorySt …
  -/
  NatIso.ofComponents (λ_ ·)
  /-
    🎉 no goals
  -/


/-- Right unitor as a natural isomorphism. -/
@[simps!]
def rightUnitorNatIso (a b : B) : (postcomposing a _ _).obj (𝟙 b) ≅ 𝟭 (a ⟶ b) :=
  /-
    B : Type u
    inst✝ : CategoryTheory.Bicategory B
    a✝ b✝ c d e a b : B
    ⊢ ∀ {X Y : Quiver.Hom a b} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategorySt …
  -/
  NatIso.ofComponents (ρ_ ·)
  /-
    🎉 no goals
  -/


