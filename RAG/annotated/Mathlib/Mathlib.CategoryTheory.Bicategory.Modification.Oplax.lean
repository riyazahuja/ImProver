/-- A modification `Γ` between oplax natural transformations `η` and `θ` consists of a family of
2-morphisms `Γ.app a : η.app a ⟶ θ.app a`, which satisfies the equation
`(F.map f ◁ app b) ≫ θ.naturality f = η.naturality f ≫ (app a ▷ G.map f)`
for each 1-morphism `f : a ⟶ b`.
-/
@[ext]
structure Modification (η θ : F ⟶ G) where
  /-- The underlying family of 2-morphism. -/
  app (a : B) : η.app a ⟶ θ.app a
  /-- The naturality condition. -/
  naturality :
    ∀ {a b : B} (f : a ⟶ b),
      F.map f ◁ app b ≫ θ.naturality f = η.naturality f ≫ app a ▷ G.map f := by
    aesop_cat


attribute [reassoc (attr := simp)] Modification.naturality


/-- The identity modification. -/
@[simps]
def id : Modification η η where app a := 𝟙 (η.app a)


instance : Inhabited (Modification η η) :=
  ⟨Modification.id η⟩


@[reassoc (attr := simp)]
theorem whiskerLeft_naturality (f : a' ⟶ F.obj b) (g : b ⟶ c) :
    f ◁ F.map g ◁ Γ.app c ≫ f ◁ θ.naturality g = f ◁ η.naturality g ≫ f ◁ Γ.app b ▷ G.map g := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    η θ : Quiver.Hom F G
    Γ : CategoryTheory.Oplax.Modification η θ
    b c : B
    a' : C
    f : Quiver.Hom a' (F.obj b)
    g : Quiver.Hom b c
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp_rw [← Bicategory.whiskerLeft_comp, naturality]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerRight_naturality (f : a ⟶ b) (g : G.obj b ⟶ a') :
    F.map f ◁ Γ.app b ▷ g ≫ (α_ _ _ _).inv ≫ θ.naturality f ▷ g =
      (α_ _ _ _).inv ≫ η.naturality f ▷ g ≫ Γ.app a ▷ G.map f ▷ g := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    η θ : Quiver.Hom F G
    Γ : CategoryTheory.Oplax.Modification η θ
    a b : B
    a' : C
    f : Quiver.Hom a b
    g : Quiver.Hom (G.obj b) a'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
  -/
  simp_rw [associator_inv_naturality_middle_assoc, ← comp_whiskerRight, naturality]
  /-
    🎉 no goals
  -/


/-- Vertical composition of modifications. -/
@[simps]
def vcomp (Γ : Modification η θ) (Δ : Modification θ ι) : Modification η ι where
  app a := Γ.app a ≫ Δ.app a


/-- Category structure on the oplax natural transformations between OplaxFunctors. -/
@[simps]
instance category (F G : OplaxFunctor B C) : Category (F ⟶ G) where
  Hom := Modification
  id := Modification.id
  comp := Modification.vcomp


@[ext]
lemma ext {F G : OplaxFunctor B C} {α β : F ⟶ G} {m n : α ⟶ β} (w : ∀ b, m.app b = n.app b) :
    m = n := by
  /-
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    α β : Quiver.Hom F G
    m n : Quiver.Hom α β
    w : ∀ (b : B), Eq (m.app b) (n.app b)
    ⊢ Eq m n
  -/
  apply Modification.ext
  /-
    case app
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    α β : Quiver.Hom F G
    m n : Quiver.Hom α β
    w : ∀ (b : B), Eq (m.app b) (n.app b)
    ⊢ Eq m.app n.app
  -/
  ext
  /-
    case app.h
    B : Type u₁
    inst✝¹ : CategoryTheory.Bicategory B
    C : Type u₂
    inst✝ : CategoryTheory.Bicategory C
    F G : CategoryTheory.OplaxFunctor B C
    α β : Quiver.Hom F G
    m n : Quiver.Hom α β
    w : ∀ (b : B), Eq (m.app b) (n.app b)
    x✝ : B
    ⊢ Eq (m.app x✝) (n.app x✝)
  -/
  apply w
  /-
    🎉 no goals
  -/


/-- Version of `Modification.id_app` using category notation -/
@[simp]
lemma Modification.id_app' {X : B} {F G : OplaxFunctor B C} (α : F ⟶ G) :
    Modification.app (𝟙 α) X = 𝟙 (α.app X) := rfl


/-- Version of `Modification.comp_app` using category notation -/
@[simp]
lemma Modification.comp_app' {X : B} {F G : OplaxFunctor B C} {α β γ : F ⟶ G}
    (m : α ⟶ β) (n : β ⟶ γ) : (m ≫ n).app X = m.app X ≫ n.app X :=
  rfl


/-- Construct a modification isomorphism between oplax natural transformations
by giving object level isomorphisms, and checking naturality only in the forward direction.
-/
@[simps]
def ModificationIso.ofComponents (app : ∀ a, η.app a ≅ θ.app a)
    (naturality :
      ∀ {a b} (f : a ⟶ b),
        F.map f ◁ (app b).hom ≫ θ.naturality f = η.naturality f ≫ (app a).hom ▷ G.map f) :
    η ≅ θ where
  hom := { app := fun a => (app a).hom }
  inv :=
    { app := fun a => (app a).inv
      naturality := fun {a b} f => by
        /-
          B : Type u₁
          inst✝¹ : CategoryTheory.Bicategory B
          C : Type u₂
          inst✝ : CategoryTheory.Bicategory C
          F✝ G✝ : CategoryTheory.OplaxFunctor B C
          η✝ θ✝ : Quiver.Hom F✝ G✝
          F G : CategoryTheory.OplaxFunctor B C
          η θ ι : Quiver.Hom F G
          app : (a : B) → CategoryTheory.Iso (η.app a) (θ.app a)
          naturality : ∀ {a b : B} (f : Quiver.Hom a b), Eq (CategoryTheory.CategoryStru …
          a b : B
          f : Quiver.Hom a b
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Bicategory.whiskerLef …
        -/
        simpa using congr_arg (fun f => _ ◁ (app b).inv ≫ f ≫ (app a).inv ▷ _) (naturality f).symm }
        /-
          🎉 no goals
        -/


