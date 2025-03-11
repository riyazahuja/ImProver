/-- A wrapper for promoting any type to a category,
with the only morphisms being equalities.
-/
@[ext, aesop safe cases (rule_sets := [CategoryTheory])]
structure Discrete (α : Type u₁) where
  /-- A wrapper for promoting any type to a category,
  with the only morphisms being equalities. -/
  as : α


@[simp]
theorem Discrete.mk_as {α : Type u₁} (X : Discrete α) : Discrete.mk X.as = X := by
  /-
    α : Type u₁
    X : CategoryTheory.Discrete α
    ⊢ Eq { as := X.as } X
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `Discrete α` is equivalent to the original type `α`. -/
@[simps]
def discreteEquiv {α : Type u₁} : Discrete α ≃ α where
  toFun := Discrete.as
  invFun := Discrete.mk
                 /-
                   α : Type u₁
                   ⊢ Function.LeftInverse CategoryTheory.Discrete.mk CategoryTheory.Discrete.as
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    α : Type u₁
                    ⊢ Function.RightInverse CategoryTheory.Discrete.mk CategoryTheory.Discrete.as
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


instance {α : Type u₁} [DecidableEq α] : DecidableEq (Discrete α) :=
  discreteEquiv.decidableEq


/-- The "Discrete" category on a type, whose morphisms are equalities.

Because we do not allow morphisms in `Prop` (only in `Type`),
somewhat annoyingly we have to define `X ⟶ Y` as `ULift (PLift (X = Y))`.

See <https://stacks.math.columbia.edu/tag/001A>
-/
instance discreteCategory (α : Type u₁) : SmallCategory (Discrete α) where
  Hom X Y := ULift (PLift (X.as = Y.as))
  id _ := ULift.up (PLift.up rfl)
  comp {X Y Z} g f := by
    /-
      α : Type u₁
      X Y Z : CategoryTheory.Discrete α
      g : Quiver.Hom X Y
      f : Quiver.Hom Y Z
      ⊢ Quiver.Hom X Z
    -/
    cases X
    /-
      case mk
      α : Type u₁
      Y Z : CategoryTheory.Discrete α
      f : Quiver.Hom Y Z
      as✝ : α
      g : Quiver.Hom { as := as✝ } Y
      ⊢ Quiver.Hom { as := as✝ } Z
    -/
    cases Y
    /-
      case mk.mk
      α : Type u₁
      Z : CategoryTheory.Discrete α
      as✝¹ as✝ : α
      f : Quiver.Hom { as := as✝ } Z
      g : Quiver.Hom { as := as✝¹ } { as := as✝ }
      ⊢ Quiver.Hom { as := as✝¹ } Z
    -/
    cases Z
    /-
      case mk.mk.mk
      α : Type u₁
      as✝² as✝¹ : α
      g : Quiver.Hom { as := as✝² } { as := as✝¹ }
      as✝ : α
      f : Quiver.Hom { as := as✝¹ } { as := as✝ }
      ⊢ Quiver.Hom { as := as✝² } { as := as✝ }
    -/
    rcases f with ⟨⟨⟨⟩⟩⟩
    /-
      case mk.mk.mk.up.up.refl
      α : Type u₁
      as✝¹ as✝ : α
      g : Quiver.Hom { as := as✝¹ } { as := as✝ }
      ⊢ Quiver.Hom { as := as✝¹ } { as := as✝ }
    -/
    exact g
    /-
      🎉 no goals
    -/


instance [Inhabited α] : Inhabited (Discrete α) :=
  ⟨⟨default⟩⟩


instance [Subsingleton α] : Subsingleton (Discrete α) :=
      /-
        α : Type u₁
        inst✝ : Subsingleton α
        ⊢ ∀ (a b : CategoryTheory.Discrete α), Eq a b
      -/
  ⟨by aesop_cat⟩
      /-
        🎉 no goals
      -/


instance instSubsingletonDiscreteHom (X Y : Discrete α) : Subsingleton (X ⟶ Y) :=
  show Subsingleton (ULift (PLift _)) from inferInstance

/- Porting note: rewrote `discrete_cases` tactic -/

/-- A simple tactic to run `cases` on any `Discrete α` hypotheses. -/
macro "discrete_cases" : tactic =>
  `(tactic| fail_if_no_progress casesm* Discrete _, (_ : Discrete _) ⟶ (_ : Discrete _), PLift _)


open Lean Elab Tactic in
/--
Use:
```
attribute [local aesop safe tactic (rule_sets := [CategoryTheory])]
  CategoryTheory.Discrete.discreteCases
```
to locally gives `aesop_cat` the ability to call `cases` on
`Discrete` and `(_ : Discrete _) ⟶ (_ : Discrete _)` hypotheses.
-/
def discreteCases : TacticM Unit := do
  evalTactic (← `(tactic| discrete_cases))

-- Porting note:
-- investigate turning on either
-- `attribute [aesop safe cases (rule_sets := [CategoryTheory])] Discrete`
-- or
-- `attribute [aesop safe tactic (rule_sets := [CategoryTheory])] discreteCases`
-- globally.


instance [Unique α] : Unique (Discrete α) :=
  Unique.mk' (Discrete α)


/-- Extract the equation from a morphism in a discrete category. -/
theorem eq_of_hom {X Y : Discrete α} (i : X ⟶ Y) : X.as = Y.as :=
  i.down.down


/-- Promote an equation between the wrapped terms in `X Y : Discrete α` to a morphism `X ⟶ Y`
in the discrete category. -/
protected abbrev eqToHom {X Y : Discrete α} (h : X.as = Y.as) : X ⟶ Y :=
              /-
                α : Type u₁
                X Y : CategoryTheory.Discrete α
                h : Eq X.as Y.as
                ⊢ Eq X Y
              -/
  eqToHom (by aesop_cat)
              /-
                🎉 no goals
              -/


/-- Promote an equation between the wrapped terms in `X Y : Discrete α` to an isomorphism `X ≅ Y`
in the discrete category. -/
protected abbrev eqToIso {X Y : Discrete α} (h : X.as = Y.as) : X ≅ Y :=
              /-
                α : Type u₁
                X Y : CategoryTheory.Discrete α
                h : Eq X.as Y.as
                ⊢ Eq X Y
              -/
  eqToIso (by aesop_cat)
              /-
                🎉 no goals
              -/


/-- A variant of `eqToHom` that lifts terms to the discrete category. -/
abbrev eqToHom' {a b : α} (h : a = b) : Discrete.mk a ⟶ Discrete.mk b :=
  Discrete.eqToHom h


/-- A variant of `eqToIso` that lifts terms to the discrete category. -/
abbrev eqToIso' {a b : α} (h : a = b) : Discrete.mk a ≅ Discrete.mk b :=
  Discrete.eqToIso h


@[simp]
theorem id_def (X : Discrete α) : ULift.up (PLift.up (Eq.refl X.as)) = 𝟙 X :=
  rfl


instance {I : Type u₁} {i j : Discrete I} (f : i ⟶ j) : IsIso f :=
                                            /-
                                              α : Type u₁
                                              C : Type u₂
                                              inst✝ : CategoryTheory.Category.{v₂, u₂} C
                                              I : Type u₁
                                              i j : CategoryTheory.Discrete I
                                              f : Quiver.Hom i j
                                              ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Discrete.eqToH …
                                            -/
  ⟨⟨Discrete.eqToHom (eq_of_hom f).symm, by aesop_cat⟩⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- Any function `I → C` gives a functor `Discrete I ⥤ C`. -/
def functor {I : Type u₁} (F : I → C) : Discrete I ⥤ C where
  obj := F ∘ Discrete.as
  map {X Y} f := by
    /-
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F : I → C
      X Y : CategoryTheory.Discrete I
      f : Quiver.Hom X Y
      ⊢ Quiver.Hom (Function.comp F CategoryTheory.Discrete.as X) (Function.comp F C …
    -/
    dsimp
    /-
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F : I → C
      X Y : CategoryTheory.Discrete I
      f : Quiver.Hom X Y
      ⊢ Quiver.Hom (F X.as) (F Y.as)
    -/
    rcases f with ⟨⟨h⟩⟩
    /-
      case up.up
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F : I → C
      X Y : CategoryTheory.Discrete I
      h : Eq X.as Y.as
      ⊢ Quiver.Hom (F X.as) (F Y.as)
    -/
    exact eqToHom (congrArg _ h)
    /-
      🎉 no goals
    -/


@[simp]
theorem functor_obj {I : Type u₁} (F : I → C) (i : I) :
    (Discrete.functor F).obj (Discrete.mk i) = F i :=
  rfl


@[simp]
theorem functor_map {I : Type u₁} (F : I → C) {i : Discrete I} (f : i ⟶ i) :
                                                  /-
                                                    C : Type u₂
                                                    inst✝ : CategoryTheory.Category.{v₂, u₂} C
                                                    I : Type u₁
                                                    F : I → C
                                                    i : CategoryTheory.Discrete I
                                                    f : Quiver.Hom i i
                                                    ⊢ Eq ((CategoryTheory.Discrete.functor F).map f) (CategoryTheory.CategoryStruc …
                                                  -/
    (Discrete.functor F).map f = 𝟙 (F i.as) := by aesop_cat
                                                  /-
                                                    🎉 no goals
                                                  -/

@[deprecated (since := "2024-07-16")]
alias CategoryTheory.FreeMonoidalCategory.discrete_functor_map_eq_id := functor_map


@[simp]
theorem functor_obj_eq_as {I : Type u₁} (F : I → C) (X : Discrete I) :
    (Discrete.functor F).obj X = F X.as :=
  rfl

@[deprecated (since := "2024-07-16")]
alias CategoryTheory.FreeMonoidalCategory.discrete_functor_obj_eq_as := functor_obj_eq_as


/-- The discrete functor induced by a composition of maps can be written as a
composition of two discrete functors.
-/
@[simps!]
def functorComp {I : Type u₁} {J : Type u₁'} (f : J → C) (g : I → J) :
    Discrete.functor (f ∘ g) ≅ Discrete.functor (Discrete.mk ∘ g) ⋙ Discrete.functor f :=
  /-
    α : Type u₁
    C : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} C
    I : Type u₁
    J : Type u₁'
    f : J → C
    g : I → J
    ⊢ ∀ {X Y : CategoryTheory.Discrete I} (f_1 : Quiver.Hom X Y), Eq (CategoryTheo …
  -/
  NatIso.ofComponents fun _ => Iso.refl _
  /-
    🎉 no goals
  -/


/-- For functors out of a discrete category,
a natural transformation is just a collection of maps,
as the naturality squares are trivial.
-/
@[simps]
def natTrans {I : Type u₁} {F G : Discrete I ⥤ C} (f : ∀ i : Discrete I, F.obj i ⟶ G.obj i) :
    F ⟶ G where
  app := f
  naturality := fun {X Y} ⟨⟨g⟩⟩ => by
    /-
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
      f : (i : CategoryTheory.Discrete I) → Quiver.Hom (F.obj i) (G.obj i)
      X Y : CategoryTheory.Discrete I
      x✝ : Quiver.Hom X Y
      g : Eq X.as Y.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := g } }) (f  …
    -/
    discrete_cases
    /-
      case mk.mk.up.up
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
      f : (i : CategoryTheory.Discrete I) → Quiver.Hom (F.obj i) (G.obj i)
      as✝¹ as✝ : I
      g down✝ : Eq { as := as✝¹ }.as { as := as✝ }.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := g } }) (f  …
    -/
    rcases g
    /-
      case mk.mk.up.up.refl
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
      f : (i : CategoryTheory.Discrete I) → Quiver.Hom (F.obj i) (G.obj i)
      as✝ : I
      down✝ : Eq { as := as✝ }.as { as := as✝ }.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := ⋯ } }) (f  …
    -/
    change F.map (𝟙 _) ≫ _ = _ ≫ G.map (𝟙 _)
    /-
      case mk.mk.up.up.refl
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
      f : (i : CategoryTheory.Discrete I) → Quiver.Hom (F.obj i) (G.obj i)
      as✝ : I
      down✝ : Eq { as := as✝ }.as { as := as✝ }.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- For functors out of a discrete category,
a natural isomorphism is just a collection of isomorphisms,
as the naturality squares are trivial.
-/
@[simps!]
def natIso {I : Type u₁} {F G : Discrete I ⥤ C} (f : ∀ i : Discrete I, F.obj i ≅ G.obj i) :
    F ≅ G :=
  NatIso.ofComponents f fun ⟨⟨g⟩⟩ => by
    /-
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
      f : (i : CategoryTheory.Discrete I) → CategoryTheory.Iso (F.obj i) (G.obj i)
      X✝ Y✝ : CategoryTheory.Discrete I
      x✝ : Quiver.Hom X✝ Y✝
      g : Eq X✝.as Y✝.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := g } }) (f  …
    -/
    discrete_cases
    /-
      case mk.mk.up.up
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
      f : (i : CategoryTheory.Discrete I) → CategoryTheory.Iso (F.obj i) (G.obj i)
      as✝¹ as✝ : I
      g down✝ : Eq { as := as✝¹ }.as { as := as✝ }.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := g } }) (f  …
    -/
    rcases g
    /-
      case mk.mk.up.up.refl
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
      f : (i : CategoryTheory.Discrete I) → CategoryTheory.Iso (F.obj i) (G.obj i)
      as✝ : I
      down✝ : Eq { as := as✝ }.as { as := as✝ }.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := ⋯ } }) (f  …
    -/
    change F.map (𝟙 _) ≫ _ = _ ≫ G.map (𝟙 _)
    /-
      case mk.mk.up.up.refl
      α : Type u₁
      C : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} C
      I : Type u₁
      F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
      f : (i : CategoryTheory.Discrete I) → CategoryTheory.Iso (F.obj i) (G.obj i)
      as✝ : I
      down✝ : Eq { as := as✝ }.as { as := as✝ }.as
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
    -/
    simp
    /-
      🎉 no goals
    -/


instance {I : Type*} {F G : Discrete I ⥤ C} (f : ∀ i, F.obj i ⟶ G.obj i) [∀ i, IsIso (f i)] :
    IsIso (Discrete.natTrans f) := by
  /-
    α : Type u₁
    C : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
    I : Type u_1
    F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
    f : (i : CategoryTheory.Discrete I) → Quiver.Hom (F.obj i) (G.obj i)
    inst✝ : ∀ (i : CategoryTheory.Discrete I), CategoryTheory.IsIso (f i)
    ⊢ CategoryTheory.IsIso (CategoryTheory.Discrete.natTrans f)
  -/
  change IsIso (Discrete.natIso (fun i => asIso (f i))).hom
  /-
    α : Type u₁
    C : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} C
    I : Type u_1
    F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
    f : (i : CategoryTheory.Discrete I) → Quiver.Hom (F.obj i) (G.obj i)
    inst✝ : ∀ (i : CategoryTheory.Discrete I), CategoryTheory.IsIso (f i)
    ⊢ CategoryTheory.IsIso (CategoryTheory.Discrete.natIso fun i => CategoryTheory …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem natIso_app {I : Type u₁} {F G : Discrete I ⥤ C} (f : ∀ i : Discrete I, F.obj i ≅ G.obj i)
                                                             /-
                                                               C : Type u₂
                                                               inst✝ : CategoryTheory.Category.{v₂, u₂} C
                                                               I : Type u₁
                                                               F G : CategoryTheory.Functor (CategoryTheory.Discrete I) C
                                                               f : (i : CategoryTheory.Discrete I) → CategoryTheory.Iso (F.obj i) (G.obj i)
                                                               i : CategoryTheory.Discrete I
                                                               ⊢ Eq ((CategoryTheory.Discrete.natIso f).app i) (f i)
                                                             -/
    (i : Discrete I) : (Discrete.natIso f).app i = f i := by aesop_cat
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Every functor `F` from a discrete category is naturally isomorphic (actually, equal) to
  `Discrete.functor (F.obj)`. -/
@[simp]
def natIsoFunctor {I : Type u₁} {F : Discrete I ⥤ C} : F ≅ Discrete.functor (F.obj ∘ Discrete.mk) :=
  natIso fun _ => Iso.refl _


/-- Composing `Discrete.functor F` with another functor `G` amounts to composing `F` with `G.obj` -/
@[simp]
def compNatIsoDiscrete {I : Type u₁} {D : Type u₃} [Category.{v₃} D] (F : I → C) (G : C ⥤ D) :
    Discrete.functor F ⋙ G ≅ Discrete.functor (G.obj ∘ F) :=
  natIso fun _ => Iso.refl _


/-- We can promote a type-level `Equiv` to
an equivalence between the corresponding `discrete` categories.
-/
@[simps]
def equivalence {I : Type u₁} {J : Type u₂} (e : I ≃ J) : Discrete I ≌ Discrete J where
  functor := Discrete.functor (Discrete.mk ∘ (e : I → J))
  inverse := Discrete.functor (Discrete.mk ∘ (e.symm : J → I))
  unitIso :=
                                         /-
                                           α : Type u₁
                                           C : Type u₂
                                           inst✝ : CategoryTheory.Category.{v₂, u₂} C
                                           I : Type u₁
                                           J : Type u₂
                                           e : Equiv I J
                                           i : CategoryTheory.Discrete I
                                           ⊢ Eq ((CategoryTheory.Functor.id (CategoryTheory.Discrete I)).obj i) (((Catego …
                                         -/
    Discrete.natIso fun i => eqToIso (by aesop_cat)
                                         /-
                                           🎉 no goals
                                         -/
  counitIso :=
                                         /-
                                           α : Type u₁
                                           C : Type u₂
                                           inst✝ : CategoryTheory.Category.{v₂, u₂} C
                                           I : Type u₁
                                           J : Type u₂
                                           e : Equiv I J
                                           j : CategoryTheory.Discrete J
                                           ⊢ Eq (((CategoryTheory.Discrete.functor (Function.comp CategoryTheory.Discrete …
                                         -/
    Discrete.natIso fun j => eqToIso (by aesop_cat)
                                         /-
                                           🎉 no goals
                                         -/


/-- We can convert an equivalence of `discrete` categories to a type-level `Equiv`. -/
@[simps]
def equivOfEquivalence {α : Type u₁} {β : Type u₂} (h : Discrete α ≌ Discrete β) : α ≃ β where
  toFun := Discrete.as ∘ h.functor.obj ∘ Discrete.mk
  invFun := Discrete.as ∘ h.inverse.obj ∘ Discrete.mk
                   /-
                     α✝ : Type u₁
                     C : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} C
                     α : Type u₁
                     β : Type u₂
                     h : CategoryTheory.Equivalence (CategoryTheory.Discrete α) (CategoryTheory.Dis …
                     a : α
                     ⊢ Eq (Function.comp CategoryTheory.Discrete.as (Function.comp h.inverse.obj Ca …
                   -/
  left_inv a := by simpa using eq_of_hom (h.unitIso.app (Discrete.mk a)).2
                   /-
                     🎉 no goals
                   -/
                    /-
                      α✝ : Type u₁
                      C : Type u₂
                      inst✝ : CategoryTheory.Category.{v₂, u₂} C
                      α : Type u₁
                      β : Type u₂
                      h : CategoryTheory.Equivalence (CategoryTheory.Discrete α) (CategoryTheory.Dis …
                      a : β
                      ⊢ Eq (Function.comp CategoryTheory.Discrete.as (Function.comp h.functor.obj Ca …
                    -/
  right_inv a := by simpa using eq_of_hom (h.counitIso.app (Discrete.mk a)).1
                    /-
                      🎉 no goals
                    -/


/-- A discrete category is equivalent to its opposite category. -/
@[simps! functor_obj_as inverse_obj]
protected def opposite (α : Type u₁) : (Discrete α)ᵒᵖ ≌ Discrete α :=
  let F : Discrete α ⥤ (Discrete α)ᵒᵖ := Discrete.functor fun x => op (Discrete.mk x)
  { functor := F.leftOp
    inverse := F
               /-
                 J : Type v₁
                 α : Type u₁
                 F : CategoryTheory.Functor (CategoryTheory.Discrete α) (Opposite (CategoryTheo …
                 ⊢ ∀ {X Y : Opposite (CategoryTheory.Discrete α)} (f : Quiver.Hom X Y), Eq (Cat …
               -/
    unitIso := NatIso.ofComponents fun ⟨_⟩ => Iso.refl _
               /-
                 🎉 no goals
               -/
    counitIso := Discrete.natIso fun ⟨_⟩ => Iso.refl _ }


@[simp]
theorem functor_map_id (F : Discrete J ⥤ C) {j : Discrete J} (f : j ⟶ j) :
    F.map f = 𝟙 (F.obj j) := by
  /-
    J : Type v₁
    C : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} C
    F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    j : CategoryTheory.Discrete J
    f : Quiver.Hom j j
    ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.id (F.obj j))
  -/
  have h : f = 𝟙 j := by aesop_cat
  /-
    J : Type v₁
    C : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} C
    F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    j : CategoryTheory.Discrete J
    f : Quiver.Hom j j
    h : Eq f (CategoryTheory.CategoryStruct.id j)
    ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.id (F.obj j))
  -/
  rw [h]
  /-
    J : Type v₁
    C : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} C
    F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
    j : CategoryTheory.Discrete J
    f : Quiver.Hom j j
    h : Eq f (CategoryTheory.CategoryStruct.id j)
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id j)) (CategoryTheory.CategoryStru …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The equivalence of categories `(J → C) ≌ (Discrete J ⥤ C)`. -/
@[simps]
def piEquivalenceFunctorDiscrete (J : Type u₂) (C : Type u₁) [Category.{v₁} C] :
    (J → C) ≌ (Discrete J ⥤ C) where
  functor :=
    { obj := fun F => Discrete.functor F
      map := fun f => Discrete.natTrans (fun j => f j.as) }
  inverse :=
    { obj := fun F j => F.obj ⟨j⟩
      map := fun f j => f.app ⟨j⟩ }
  unitIso := Iso.refl _
  counitIso := NatIso.ofComponents (fun F => (NatIso.ofComponents (fun _ => Iso.refl _)
    (by
      /-
        J : Type u₂
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
        ⊢ ∀ {X Y : CategoryTheory.Discrete J} (f : Quiver.Hom X Y), Eq (CategoryTheory …
      -/
      rintro ⟨x⟩ ⟨y⟩ f
      /-
        case mk.mk
        J : Type u₂
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
        x y : J
        f : Quiver.Hom { as := x } { as := y }
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((({ obj := fun F j => F.obj { as :=  …
      -/
      obtain rfl : x = y := Discrete.eq_of_hom f
      /-
        case mk.mk
        J : Type u₂
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
        x : J
        f : Quiver.Hom { as := x } { as := x }
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((({ obj := fun F j => F.obj { as :=  …
      -/
      obtain rfl : f = 𝟙 _ := rfl
      /-
        case mk.mk
        J : Type u₂
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor (CategoryTheory.Discrete J) C
        x : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((({ obj := fun F j => F.obj { as :=  …
      -/
      /-
        🎉 no goals
      -/
      simp))) (by aesop_cat)
                  /-
                    🎉 no goals
                  -/


/-- A category is discrete when there is at most one morphism between two objects,
in which case they are equal. -/
class IsDiscrete (C : Type*) [Category C] : Prop where
  subsingleton (X Y : C) : Subsingleton (X ⟶ Y) := by infer_instance
  eq_of_hom {X Y : C} (f : X ⟶ Y) : X = Y


lemma obj_ext_of_isDiscrete {C : Type*} [Category C] [IsDiscrete C]
    {X Y : C} (f : X ⟶ Y) : X = Y := IsDiscrete.eq_of_hom f


instance Discrete.isDiscrete (C : Type*) : IsDiscrete (Discrete C) where
                  /-
                    C : Type u_1
                    ⊢ ∀ {X Y : CategoryTheory.Discrete C}, Quiver.Hom X Y → Eq X Y
                  -/
  eq_of_hom := by rintro ⟨_⟩ ⟨_⟩ ⟨⟨rfl⟩⟩; rfl
                                          /-
                                            🎉 no goals
                                          -/


instance (C : Type*) [Category C] [IsDiscrete C] : IsDiscrete Cᵒᵖ where
  eq_of_hom := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.IsDiscrete C
      ⊢ ∀ {X Y : Opposite C}, Quiver.Hom X Y → Eq X Y
    -/
    rintro ⟨_⟩ ⟨_⟩ ⟨f⟩
    /-
      case op.op.op
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.IsDiscrete C
      unop✝¹ unop✝ : C
      f : Quiver.Hom (Opposite.unop { unop := unop✝ }) (Opposite.unop { unop := unop …
      ⊢ Eq { unop := unop✝¹ } { unop := unop✝ }
    -/
    obtain rfl := obj_ext_of_isDiscrete f
    /-
      case op.op.op
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.IsDiscrete C
      unop✝ : C
      f : Quiver.Hom (Opposite.unop { unop := unop✝ }) (Opposite.unop { unop := Oppo …
      ⊢ Eq { unop := Opposite.unop { unop := unop✝ } } { unop := unop✝ }
    -/
    rfl
    /-
      🎉 no goals
    -/


