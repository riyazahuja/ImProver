/-- Construct a cone for the empty diagram given an object. -/
@[simps]
def asEmptyCone (X : C) : Cone (Functor.empty.{0} C) :=
  { pt := X
    π :=
                /-
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  X : C
                  ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (((CategoryTheory.Fu …
                -/
    { app := by aesop_cat } }
                /-
                  🎉 no goals
                -/


/-- Construct a cocone for the empty diagram given an object. -/
@[simps]
def asEmptyCocone (X : C) : Cocone (Functor.empty.{0} C) :=
  { pt := X
    ι :=
                /-
                  C : Type u₁
                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                  X : C
                  ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom ((CategoryTheory.Fun …
                -/
    { app := by aesop_cat } }
                /-
                  🎉 no goals
                -/


/-- `X` is terminal if the cone it induces on the empty diagram is limiting. -/
abbrev IsTerminal (X : C) :=
  IsLimit (asEmptyCone X)


/-- `X` is initial if the cocone it induces on the empty diagram is colimiting. -/
abbrev IsInitial (X : C) :=
  IsColimit (asEmptyCocone X)


/-- An object `Y` is terminal iff for every `X` there is a unique morphism `X ⟶ Y`. -/
def isTerminalEquivUnique (F : Discrete.{0} PEmpty.{1} ⥤ C) (Y : C) :
                    /-
                      C : Type u₁
                      inst✝ : CategoryTheory.Category.{v₁, u₁} C
                      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                      Y : C
                      ⊢ (X : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (((CategoryTheory.Func …
                    -/
                    /-
                      🎉 no goals
                    -/
    IsLimit (⟨Y, by aesop_cat, by aesop_cat⟩ : Cone F) ≃ ∀ X : C, Unique (X ⟶ Y) where
                                  /-
                                    🎉 no goals
                                  -/
  toFun t X :=
                                /-
                                  C : Type u₁
                                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                  F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                                  Y : C
                                  t : CategoryTheory.Limits.IsLimit { pt := Y, π := { app := fun X => id (Catego …
                                  X : C
                                  ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (((CategoryTheory.Fu …
                                -/
                                /-
                                  🎉 no goals
                                -/
    { default := t.lift ⟨X, ⟨by aesop_cat, by aesop_cat⟩⟩
                                              /-
                                                🎉 no goals
                                              -/
      uniq := fun f =>
                       /-
                         C : Type u₁
                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                         F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                         Y : C
                         t : CategoryTheory.Limits.IsLimit { pt := Y, π := { app := fun X => id (Catego …
                         X : C
                         f : Quiver.Hom X Y
                         ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (((CategoryTheory.Fu …
                       -/
                       /-
                         🎉 no goals
                       -/
                                     /-
                                       🎉 no goals
                                     -/
        t.uniq ⟨X, ⟨by aesop_cat, by aesop_cat⟩⟩ f (by aesop_cat) }
                                                       /-
                                                         🎉 no goals
                                                       -/
  invFun u :=
    { lift := fun s => (u s.pt).default
      uniq := fun s _ _ => (u s.pt).2 _ }
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                   Y : C
                   ⊢ Function.LeftInverse (fun u => { lift := fun s => Inhabited.default, fac :=  …
                 -/
  left_inv := by dsimp [Function.LeftInverse]; intro x; simp only [eq_iff_true_of_subsingleton]
                                                        /-
                                                          🎉 no goals
                                                        -/
  right_inv := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
      Y : C
      ⊢ Function.RightInverse (fun u => { lift := fun s => Inhabited.default, fac := …
    -/
    dsimp [Function.RightInverse,Function.LeftInverse]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
      Y : C
      ⊢ ∀ (x : (X : C) → Unique (Quiver.Hom X Y)), Eq (fun X => { default := Inhabit …
    -/
    intro u; funext X; simp only
                       /-
                         🎉 no goals
                       -/


/-- An object `Y` is terminal if for every `X` there is a unique morphism `X ⟶ Y`
    (as an instance). -/
def IsTerminal.ofUnique (Y : C) [h : ∀ X : C, Unique (X ⟶ Y)] : IsTerminal Y where
  lift s := (h s.pt).default
  fac := fun _ ⟨j⟩ => j.elim


/-- An object `Y` is terminal if for every `X` there is a unique morphism `X ⟶ Y`
    (as explicit arguments). -/
def IsTerminal.ofUniqueHom {Y : C} (h : ∀ X : C, X ⟶ Y) (uniq : ∀ (X : C) (m : X ⟶ Y), m = h X) :
    IsTerminal Y :=
  have : ∀ X : C, Unique (X ⟶ Y) := fun X ↦ ⟨⟨h X⟩, uniq X⟩
  IsTerminal.ofUnique Y


/-- If `α` is a preorder with top, then `⊤` is a terminal object. -/
def isTerminalTop {α : Type*} [Preorder α] [OrderTop α] : IsTerminal (⊤ : α) :=
  IsTerminal.ofUnique _


/-- Transport a term of type `IsTerminal` across an isomorphism. -/
def IsTerminal.ofIso {Y Z : C} (hY : IsTerminal Y) (i : Y ≅ Z) : IsTerminal Z :=
  IsLimit.ofIsoLimit hY
    { hom := { hom := i.hom }
      inv := { hom := i.inv } }


/-- If `X` and `Y` are isomorphic, then `X` is terminal iff `Y` is. -/
def IsTerminal.equivOfIso {X Y : C} (e : X ≅ Y) :
    IsTerminal X ≃ IsTerminal Y where
  toFun h := IsTerminal.ofIso h e
  invFun h := IsTerminal.ofIso h e.symm
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- An object `X` is initial iff for every `Y` there is a unique morphism `X ⟶ Y`. -/
def isInitialEquivUnique (F : Discrete.{0} PEmpty.{1} ⥤ C) (X : C) :
                       /-
                         C : Type u₁
                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                         F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                         X : C
                         ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (F.obj X_1) (((Categ …
                       -/
                       /-
                         🎉 no goals
                       -/
    IsColimit (⟨X, ⟨by aesop_cat, by aesop_cat⟩⟩ : Cocone F) ≃ ∀ Y : C, Unique (X ⟶ Y) where
                                     /-
                                       🎉 no goals
                                     -/
  toFun t X :=
                                /-
                                  C : Type u₁
                                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                  F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                                  X✝ : C
                                  t : CategoryTheory.Limits.IsColimit { pt := X✝, ι := { app := fun X_1 => id (C …
                                  X : C
                                  ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (F.obj X_1) (((Categ …
                                -/
                                /-
                                  🎉 no goals
                                -/
    { default := t.desc ⟨X, ⟨by aesop_cat, by aesop_cat⟩⟩
                                              /-
                                                🎉 no goals
                                              -/
                                      /-
                                        C : Type u₁
                                        inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                        F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                                        X✝ : C
                                        t : CategoryTheory.Limits.IsColimit { pt := X✝, ι := { app := fun X_1 => id (C …
                                        X : C
                                        f : Quiver.Hom X✝ X
                                        ⊢ (X_1 : CategoryTheory.Discrete PEmpty.{1}) → Quiver.Hom (F.obj X_1) (((Categ …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
                                                    /-
                                                      🎉 no goals
                                                    -/
      uniq := fun f => t.uniq ⟨X, ⟨by aesop_cat, by aesop_cat⟩⟩ f (by aesop_cat) }
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  invFun u :=
    { desc := fun s => (u s.pt).default
      uniq := fun s _ _ => (u s.pt).2 _ }
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
                   X : C
                   ⊢ Function.LeftInverse (fun u => { desc := fun s => Inhabited.default, fac :=  …
                 -/
  left_inv := by dsimp [Function.LeftInverse]; intro; simp only [eq_iff_true_of_subsingleton]
                                                      /-
                                                        🎉 no goals
                                                      -/
  right_inv := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
      X : C
      ⊢ Function.RightInverse (fun u => { desc := fun s => Inhabited.default, fac := …
    -/
    dsimp [Function.RightInverse,Function.LeftInverse]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{1}) C
      X : C
      ⊢ ∀ (x : (Y : C) → Unique (Quiver.Hom X Y)), Eq (fun X_1 => { default := Inhab …
    -/
    intro; funext; simp only
                   /-
                     🎉 no goals
                   -/


/-- An object `X` is initial if for every `Y` there is a unique morphism `X ⟶ Y`
    (as an instance). -/
def IsInitial.ofUnique (X : C) [h : ∀ Y : C, Unique (X ⟶ Y)] : IsInitial X where
  desc s := (h s.pt).default
  fac := fun _ ⟨j⟩ => j.elim


/-- An object `X` is initial if for every `Y` there is a unique morphism `X ⟶ Y`
    (as explicit arguments). -/
def IsInitial.ofUniqueHom {X : C} (h : ∀ Y : C, X ⟶ Y) (uniq : ∀ (Y : C) (m : X ⟶ Y), m = h Y) :
    IsInitial X :=
  have : ∀ Y : C, Unique (X ⟶ Y) := fun Y ↦ ⟨⟨h Y⟩, uniq Y⟩
  IsInitial.ofUnique X


/-- If `α` is a preorder with bot, then `⊥` is an initial object. -/
def isInitialBot {α : Type*} [Preorder α] [OrderBot α] : IsInitial (⊥ : α) :=
  IsInitial.ofUnique _


/-- Transport a term of type `is_initial` across an isomorphism. -/
def IsInitial.ofIso {X Y : C} (hX : IsInitial X) (i : X ≅ Y) : IsInitial Y :=
  IsColimit.ofIsoColimit hX
    { hom := { hom := i.hom }
      inv := { hom := i.inv } }


/-- If `X` and `Y` are isomorphic, then `X` is initial iff `Y` is. -/
def IsInitial.equivOfIso {X Y : C} (e : X ≅ Y) :
    IsInitial X ≃ IsInitial Y where
  toFun h := IsInitial.ofIso h e
  invFun h := IsInitial.ofIso h e.symm
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


/-- Give the morphism to a terminal object from any other. -/
def IsTerminal.from {X : C} (t : IsTerminal X) (Y : C) : Y ⟶ X :=
  t.lift (asEmptyCone Y)


/-- Any two morphisms to a terminal object are equal. -/
theorem IsTerminal.hom_ext {X Y : C} (t : IsTerminal X) (f g : Y ⟶ X) : f = g :=
                        /-
                          C : Type u₁
                          inst✝ : CategoryTheory.Category.{v₁, u₁} C
                          X Y : C
                          t : CategoryTheory.Limits.IsTerminal X
                          f g : Quiver.Hom Y X
                          ⊢ ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategoryStruc …
                        -/
  IsLimit.hom_ext t (by aesop_cat)
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem IsTerminal.comp_from {Z : C} (t : IsTerminal Z) {X Y : C} (f : X ⟶ Y) :
    f ≫ t.from Y = t.from X :=
  t.hom_ext _ _


@[simp]
theorem IsTerminal.from_self {X : C} (t : IsTerminal X) : t.from X = 𝟙 X :=
  t.hom_ext _ _


/-- Give the morphism from an initial object to any other. -/
def IsInitial.to {X : C} (t : IsInitial X) (Y : C) : X ⟶ Y :=
  t.desc (asEmptyCocone Y)


/-- Any two morphisms from an initial object are equal. -/
theorem IsInitial.hom_ext {X Y : C} (t : IsInitial X) (f g : X ⟶ Y) : f = g :=
                          /-
                            C : Type u₁
                            inst✝ : CategoryTheory.Category.{v₁, u₁} C
                            X Y : C
                            t : CategoryTheory.Limits.IsInitial X
                            f g : Quiver.Hom X Y
                            ⊢ ∀ (j : CategoryTheory.Discrete PEmpty.{1}), Eq (CategoryTheory.CategoryStruc …
                          -/
  IsColimit.hom_ext t (by aesop_cat)
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem IsInitial.to_comp {X : C} (t : IsInitial X) {Y Z : C} (f : Y ⟶ Z) : t.to Y ≫ f = t.to Z :=
  t.hom_ext _ _


@[simp]
theorem IsInitial.to_self {X : C} (t : IsInitial X) : t.to X = 𝟙 X :=
  t.hom_ext _ _


/-- Any morphism from a terminal object is split mono. -/
theorem IsTerminal.isSplitMono_from {X Y : C} (t : IsTerminal X) (f : X ⟶ Y) : IsSplitMono f :=
  IsSplitMono.mk' ⟨t.from _, t.hom_ext _ _⟩


/-- Any morphism to an initial object is split epi. -/
theorem IsInitial.isSplitEpi_to {X Y : C} (t : IsInitial X) (f : Y ⟶ X) : IsSplitEpi f :=
  IsSplitEpi.mk' ⟨t.to _, t.hom_ext _ _⟩


/-- Any morphism from a terminal object is mono. -/
theorem IsTerminal.mono_from {X Y : C} (t : IsTerminal X) (f : X ⟶ Y) : Mono f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    t : CategoryTheory.Limits.IsTerminal X
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Mono f
  -/
  haveI := t.isSplitMono_from f; infer_instance
                                 /-
                                   🎉 no goals
                                 -/


/-- Any morphism to an initial object is epi. -/
theorem IsInitial.epi_to {X Y : C} (t : IsInitial X) (f : Y ⟶ X) : Epi f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    t : CategoryTheory.Limits.IsInitial X
    f : Quiver.Hom Y X
    ⊢ CategoryTheory.Epi f
  -/
  haveI := t.isSplitEpi_to f; infer_instance
                              /-
                                🎉 no goals
                              -/


/-- If `T` and `T'` are terminal, they are isomorphic. -/
@[simps]
def IsTerminal.uniqueUpToIso {T T' : C} (hT : IsTerminal T) (hT' : IsTerminal T') : T ≅ T' where
  hom := hT'.from _
  inv := hT.from _


/-- If `I` and `I'` are initial, they are isomorphic. -/
@[simps]
def IsInitial.uniqueUpToIso {I I' : C} (hI : IsInitial I) (hI' : IsInitial I') : I ≅ I' where
  hom := hI.to _
  inv := hI'.to _


/-- Being terminal is independent of the empty diagram, its universe, and the cone over it,
    as long as the cone points are isomorphic. -/
def isLimitChangeEmptyCone {c₁ : Cone F₁} (hl : IsLimit c₁) (c₂ : Cone F₂) (hi : c₁.pt ≅ c₂.pt) :
    IsLimit c₂ where
                              /-
                                C : Type u₁
                                inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                X : C
                                F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
                                F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
                                c₁ : CategoryTheory.Limits.Cone F₁
                                hl : CategoryTheory.Limits.IsLimit c₁
                                c₂ : CategoryTheory.Limits.Cone F₂
                                hi : CategoryTheory.Iso c₁.pt c₂.pt
                                c : CategoryTheory.Limits.Cone F₂
                                ⊢ (X : CategoryTheory.Discrete PEmpty.{w + 1}) → Quiver.Hom (((CategoryTheory. …
                              -/
                              /-
                                🎉 no goals
                              -/
  lift c := hl.lift ⟨c.pt, by aesop_cat, by aesop_cat⟩ ≫ hi.hom
                                            /-
                                              🎉 no goals
                                            -/
  uniq c f _ := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
      F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
      c₁ : CategoryTheory.Limits.Cone F₁
      hl : CategoryTheory.Limits.IsLimit c₁
      c₂ : CategoryTheory.Limits.Cone F₂
      hi : CategoryTheory.Iso c₁.pt c₂.pt
      c : CategoryTheory.Limits.Cone F₂
      f : Quiver.Hom c.pt c₂.pt
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{w' + 1}), Eq (CategoryTheory.Categ …
      ⊢ Eq f ((fun c => CategoryTheory.CategoryStruct.comp (hl.lift { pt := c.pt, π  …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
      F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
      c₁ : CategoryTheory.Limits.Cone F₁
      hl : CategoryTheory.Limits.IsLimit c₁
      c₂ : CategoryTheory.Limits.Cone F₂
      hi : CategoryTheory.Iso c₁.pt c₂.pt
      c : CategoryTheory.Limits.Cone F₂
      f : Quiver.Hom c.pt c₂.pt
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{w' + 1}), Eq (CategoryTheory.Categ …
      ⊢ Eq f (CategoryTheory.CategoryStruct.comp (hl.lift { pt := c.pt, π := { app : …
    -/
    rw [← hl.uniq _ (f ≫ hi.inv) _]
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X : C
        F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
        F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
        c₁ : CategoryTheory.Limits.Cone F₁
        hl : CategoryTheory.Limits.IsLimit c₁
        c₂ : CategoryTheory.Limits.Cone F₂
        hi : CategoryTheory.Iso c₁.pt c₂.pt
        c : CategoryTheory.Limits.Cone F₂
        f : Quiver.Hom c.pt c₂.pt
        x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{w' + 1}), Eq (CategoryTheory.Categ …
        ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      -/
    · simp only [Category.assoc, Iso.inv_hom_id, Category.comp_id]
      /-
        🎉 no goals
      -/
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X : C
        F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
        F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
        c₁ : CategoryTheory.Limits.Cone F₁
        hl : CategoryTheory.Limits.IsLimit c₁
        c₂ : CategoryTheory.Limits.Cone F₂
        hi : CategoryTheory.Iso c₁.pt c₂.pt
        c : CategoryTheory.Limits.Cone F₂
        f : Quiver.Hom c.pt c₂.pt
        x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{w' + 1}), Eq (CategoryTheory.Categ …
        ⊢ ∀ (j : CategoryTheory.Discrete PEmpty.{w + 1}), Eq (CategoryTheory.CategoryS …
      -/
    · aesop_cat
      /-
        🎉 no goals
      -/


/-- Replacing an empty cone in `IsLimit` by another with the same cone point
    is an equivalence. -/
def isLimitEmptyConeEquiv (c₁ : Cone F₁) (c₂ : Cone F₂) (h : c₁.pt ≅ c₂.pt) :
    IsLimit c₁ ≃ IsLimit c₂ where
  toFun hl := isLimitChangeEmptyCone C hl c₂ h
  invFun hl := isLimitChangeEmptyCone C hl c₁ h.symm
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   X : C
                   F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
                   F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
                   c₁ : CategoryTheory.Limits.Cone F₁
                   c₂ : CategoryTheory.Limits.Cone F₂
                   h : CategoryTheory.Iso c₁.pt c₂.pt
                   ⊢ Function.LeftInverse (fun hl => CategoryTheory.Limits.isLimitChangeEmptyCone …
                 -/
  left_inv := by dsimp [Function.LeftInverse]; intro; simp only [eq_iff_true_of_subsingleton]
                                                      /-
                                                        🎉 no goals
                                                      -/
  right_inv := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
      F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
      c₁ : CategoryTheory.Limits.Cone F₁
      c₂ : CategoryTheory.Limits.Cone F₂
      h : CategoryTheory.Iso c₁.pt c₂.pt
      ⊢ Function.RightInverse (fun hl => CategoryTheory.Limits.isLimitChangeEmptyCon …
    -/
    dsimp [Function.LeftInverse,Function.RightInverse]; intro
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
      F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
      c₁ : CategoryTheory.Limits.Cone F₁
      c₂ : CategoryTheory.Limits.Cone F₂
      h : CategoryTheory.Iso c₁.pt c₂.pt
      x✝ : CategoryTheory.Limits.IsLimit c₂
      ⊢ Eq (CategoryTheory.Limits.isLimitChangeEmptyCone C (CategoryTheory.Limits.is …
    -/
    simp only [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/


/-- Being initial is independent of the empty diagram, its universe, and the cocone over it,
    as long as the cocone points are isomorphic. -/
def isColimitChangeEmptyCocone {c₁ : Cocone F₁} (hl : IsColimit c₁) (c₂ : Cocone F₂)
    (hi : c₁.pt ≅ c₂.pt) : IsColimit c₂ where
                                       /-
                                         C : Type u₁
                                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                         X : C
                                         F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
                                         F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
                                         c₁ : CategoryTheory.Limits.Cocone F₁
                                         hl : CategoryTheory.Limits.IsColimit c₁
                                         c₂ : CategoryTheory.Limits.Cocone F₂
                                         hi : CategoryTheory.Iso c₁.pt c₂.pt
                                         c : CategoryTheory.Limits.Cocone F₂
                                         ⊢ (X : CategoryTheory.Discrete PEmpty.{w + 1}) → Quiver.Hom (F₁.obj X) (((Cate …
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
  desc c := hi.inv ≫ hl.desc ⟨c.pt, by aesop_cat, by aesop_cat⟩
                                                     /-
                                                       🎉 no goals
                                                     -/
  uniq c f _ := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
      F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
      c₁ : CategoryTheory.Limits.Cocone F₁
      hl : CategoryTheory.Limits.IsColimit c₁
      c₂ : CategoryTheory.Limits.Cocone F₂
      hi : CategoryTheory.Iso c₁.pt c₂.pt
      c : CategoryTheory.Limits.Cocone F₂
      f : Quiver.Hom c₂.pt c.pt
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{w' + 1}), Eq (CategoryTheory.Categ …
      ⊢ Eq f ((fun c => CategoryTheory.CategoryStruct.comp hi.inv (hl.desc { pt := c …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
      F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
      c₁ : CategoryTheory.Limits.Cocone F₁
      hl : CategoryTheory.Limits.IsColimit c₁
      c₂ : CategoryTheory.Limits.Cocone F₂
      hi : CategoryTheory.Iso c₁.pt c₂.pt
      c : CategoryTheory.Limits.Cocone F₂
      f : Quiver.Hom c₂.pt c.pt
      x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{w' + 1}), Eq (CategoryTheory.Categ …
      ⊢ Eq f (CategoryTheory.CategoryStruct.comp hi.inv (hl.desc { pt := c.pt, ι :=  …
    -/
    rw [← hl.uniq _ (hi.hom ≫ f) _]
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X : C
        F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
        F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
        c₁ : CategoryTheory.Limits.Cocone F₁
        hl : CategoryTheory.Limits.IsColimit c₁
        c₂ : CategoryTheory.Limits.Cocone F₂
        hi : CategoryTheory.Iso c₁.pt c₂.pt
        c : CategoryTheory.Limits.Cocone F₂
        f : Quiver.Hom c₂.pt c.pt
        x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{w' + 1}), Eq (CategoryTheory.Categ …
        ⊢ Eq f (CategoryTheory.CategoryStruct.comp hi.inv (CategoryTheory.CategoryStru …
      -/
    · simp only [Iso.inv_hom_id_assoc]
      /-
        🎉 no goals
      -/
      /-
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        X : C
        F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
        F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
        c₁ : CategoryTheory.Limits.Cocone F₁
        hl : CategoryTheory.Limits.IsColimit c₁
        c₂ : CategoryTheory.Limits.Cocone F₂
        hi : CategoryTheory.Iso c₁.pt c₂.pt
        c : CategoryTheory.Limits.Cocone F₂
        f : Quiver.Hom c₂.pt c.pt
        x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{w' + 1}), Eq (CategoryTheory.Categ …
        ⊢ ∀ (j : CategoryTheory.Discrete PEmpty.{w + 1}), Eq (CategoryTheory.CategoryS …
      -/
    · aesop_cat
      /-
        🎉 no goals
      -/


/-- Replacing an empty cocone in `IsColimit` by another with the same cocone point
    is an equivalence. -/
def isColimitEmptyCoconeEquiv (c₁ : Cocone F₁) (c₂ : Cocone F₂) (h : c₁.pt ≅ c₂.pt) :
    IsColimit c₁ ≃ IsColimit c₂ where
  toFun hl := isColimitChangeEmptyCocone C hl c₂ h
  invFun hl := isColimitChangeEmptyCocone C hl c₁ h.symm
                 /-
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   X : C
                   F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
                   F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
                   c₁ : CategoryTheory.Limits.Cocone F₁
                   c₂ : CategoryTheory.Limits.Cocone F₂
                   h : CategoryTheory.Iso c₁.pt c₂.pt
                   ⊢ Function.LeftInverse (fun hl => CategoryTheory.Limits.isColimitChangeEmptyCo …
                 -/
  left_inv := by dsimp [Function.LeftInverse]; intro; simp only [eq_iff_true_of_subsingleton]
                                                      /-
                                                        🎉 no goals
                                                      -/
  right_inv := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
      F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
      c₁ : CategoryTheory.Limits.Cocone F₁
      c₂ : CategoryTheory.Limits.Cocone F₂
      h : CategoryTheory.Iso c₁.pt c₂.pt
      ⊢ Function.RightInverse (fun hl => CategoryTheory.Limits.isColimitChangeEmptyC …
    -/
    dsimp [Function.LeftInverse,Function.RightInverse]; intro
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      X : C
      F₁ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w + 1}) C
      F₂ : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{w' + 1}) C
      c₁ : CategoryTheory.Limits.Cocone F₁
      c₂ : CategoryTheory.Limits.Cocone F₂
      h : CategoryTheory.Iso c₁.pt c₂.pt
      x✝ : CategoryTheory.Limits.IsColimit c₂
      ⊢ Eq (CategoryTheory.Limits.isColimitChangeEmptyCocone C (CategoryTheory.Limit …
    -/
    simp only [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/


/-- An initial object is terminal in the opposite category. -/
def terminalOpOfInitial {X : C} (t : IsInitial X) : IsTerminal (Opposite.op X) where
  lift s := (t.to s.pt.unop).op
  uniq _ _ _ := Quiver.Hom.unop_inj (t.hom_ext _ _)


/-- An initial object in the opposite category is terminal in the original category. -/
def terminalUnopOfInitial {X : Cᵒᵖ} (t : IsInitial X) : IsTerminal X.unop where
  lift s := (t.to (Opposite.op s.pt)).unop
  uniq _ _ _ := Quiver.Hom.op_inj (t.hom_ext _ _)


/-- A terminal object is initial in the opposite category. -/
def initialOpOfTerminal {X : C} (t : IsTerminal X) : IsInitial (Opposite.op X) where
  desc s := (t.from s.pt.unop).op
  uniq _ _ _ := Quiver.Hom.unop_inj (t.hom_ext _ _)


/-- A terminal object in the opposite category is initial in the original category. -/
def initialUnopOfTerminal {X : Cᵒᵖ} (t : IsTerminal X) : IsInitial X.unop where
  desc s := (t.from (Opposite.op s.pt)).unop
  uniq _ _ _ := Quiver.Hom.op_inj (t.hom_ext _ _)


/-- A category is an `InitialMonoClass` if the canonical morphism of an initial object is a
monomorphism.  In practice, this is most useful when given an arbitrary morphism out of the chosen
initial object, see `initial.mono_from`.
Given a terminal object, this is equivalent to the assumption that the unique morphism from initial
to terminal is a monomorphism, which is the second of Freyd's axioms for an AT category.

TODO: This is a condition satisfied by categories with zero objects and morphisms.
-/
class InitialMonoClass (C : Type u₁) [Category.{v₁} C] : Prop where
  /-- The map from the (any as stated) initial object to any other object is a
    monomorphism -/
  isInitial_mono_from : ∀ {I} (X : C) (hI : IsInitial I), Mono (hI.to X)


theorem IsInitial.mono_from [InitialMonoClass C] {I} {X : C} (hI : IsInitial I) (f : I ⟶ X) :
    Mono f := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.InitialMonoClass C
    I X : C
    hI : CategoryTheory.Limits.IsInitial I
    f : Quiver.Hom I X
    ⊢ CategoryTheory.Mono f
  -/
  rw [hI.hom_ext f (hI.to X)]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Limits.InitialMonoClass C
    I X : C
    hI : CategoryTheory.Limits.IsInitial I
    f : Quiver.Hom I X
    ⊢ CategoryTheory.Mono (hI.to X)
  -/
  apply InitialMonoClass.isInitial_mono_from
  /-
    🎉 no goals
  -/


/-- To show a category is an `InitialMonoClass` it suffices to give an initial object such that
every morphism out of it is a monomorphism. -/
theorem InitialMonoClass.of_isInitial {I : C} (hI : IsInitial I) (h : ∀ X, Mono (hI.to X)) :
    InitialMonoClass C where
  isInitial_mono_from {I'} X hI' := by
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      I : C
      hI : CategoryTheory.Limits.IsInitial I
      h : ∀ (X : C), CategoryTheory.Mono (hI.to X)
      I' X : C
      hI' : CategoryTheory.Limits.IsInitial I'
      ⊢ CategoryTheory.Mono (hI'.to X)
    -/
    rw [hI'.hom_ext (hI'.to X) ((hI'.uniqueUpToIso hI).hom ≫ hI.to X)]
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      I : C
      hI : CategoryTheory.Limits.IsInitial I
      h : ∀ (X : C), CategoryTheory.Mono (hI.to X)
      I' X : C
      hI' : CategoryTheory.Limits.IsInitial I'
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (hI'.uniqueUpToIso h …
    -/
    apply mono_comp
    /-
      🎉 no goals
    -/


/-- To show a category is an `InitialMonoClass` it suffices to show the unique morphism from an
initial object to a terminal object is a monomorphism. -/
theorem InitialMonoClass.of_isTerminal {I T : C} (hI : IsInitial I) (hT : IsTerminal T)
    (_ : Mono (hI.to T)) : InitialMonoClass C :=
  InitialMonoClass.of_isInitial hI fun X => mono_of_mono_fac (hI.hom_ext (_ ≫ hT.from X) (hI.to T))


/-- From a functor `F : J ⥤ C`, given an initial object of `J`, construct a cone for `J`.
In `limitOfDiagramInitial` we show it is a limit cone. -/
@[simps]
def coneOfDiagramInitial {X : J} (tX : IsInitial X) (F : J ⥤ C) : Cone F where
  pt := F.obj X
  π :=
    { app := fun j => F.map (tX.to j)
      naturality := fun j j' k => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          X : J
          tX : CategoryTheory.Limits.IsInitial X
          F : CategoryTheory.Functor J C
          j j' : J
          k : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          X : J
          tX : CategoryTheory.Limits.IsInitial X
          F : CategoryTheory.Functor J C
          j j' : J
          k : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (F. …
        -/
        rw [← F.map_comp, Category.id_comp, tX.hom_ext (tX.to j ≫ k) (tX.to j')] }
        /-
          🎉 no goals
        -/


/-- From a functor `F : J ⥤ C`, given an initial object of `J`, show the cone
`coneOfDiagramInitial` is a limit. -/
def limitOfDiagramInitial {X : J} (tX : IsInitial X) (F : J ⥤ C) :
    IsLimit (coneOfDiagramInitial tX F) where
  lift s := s.π.app X
  uniq s m w := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      X : J
      tX : CategoryTheory.Limits.IsInitial X
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Limits.coneOfDiagramInitial tX F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      ⊢ Eq m ((fun s => s.π.app X) s)
    -/
    conv_lhs => dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      X : J
      tX : CategoryTheory.Limits.IsInitial X
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Limits.coneOfDiagramInitial tX F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      ⊢ Eq m ((fun s => s.π.app X) s)
    -/
    simp_rw [← w X, coneOfDiagramInitial_π_app, tX.hom_ext (tX.to X) (𝟙 _)]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      X : J
      tX : CategoryTheory.Limits.IsInitial X
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Limits.coneOfDiagramInitial tX F).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      ⊢ Eq m (CategoryTheory.CategoryStruct.comp m (F.map (CategoryTheory.CategorySt …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- From a functor `F : J ⥤ C`, given a terminal object of `J`, construct a cone for `J`,
provided that the morphisms in the diagram are isomorphisms.
In `limitOfDiagramTerminal` we show it is a limit cone. -/
@[simps]
def coneOfDiagramTerminal {X : J} (hX : IsTerminal X) (F : J ⥤ C)
    [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] : Cone F where
  pt := F.obj X
  π :=
    { app := fun _ => inv (F.map (hX.from _))
      naturality := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          X : J
          hX : CategoryTheory.Limits.IsTerminal X
          F : CategoryTheory.Functor J C
          inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
          ⊢ ∀ ⦃X_1 Y : J⦄ (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
        -/
        intro i j f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          X : J
          hX : CategoryTheory.Limits.IsTerminal X
          F : CategoryTheory.Functor J C
          inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
          i j : J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        simp only [IsIso.eq_inv_comp, IsIso.comp_inv_eq, Category.id_comp, ← F.map_comp,
          hX.hom_ext (hX.from i) (f ≫ hX.from j)] }


/-- From a functor `F : J ⥤ C`, given a terminal object of `J` and that the morphisms in the
diagram are isomorphisms, show the cone `coneOfDiagramTerminal` is a limit. -/
def limitOfDiagramTerminal {X : J} (hX : IsTerminal X) (F : J ⥤ C)
    [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] : IsLimit (coneOfDiagramTerminal hX F) where
  lift S := S.π.app _


/-- From a functor `F : J ⥤ C`, given a terminal object of `J`, construct a cocone for `J`.
In `colimitOfDiagramTerminal` we show it is a colimit cocone. -/
@[simps]
def coconeOfDiagramTerminal {X : J} (tX : IsTerminal X) (F : J ⥤ C) : Cocone F where
  pt := F.obj X
  ι :=
    { app := fun j => F.map (tX.from j)
      naturality := fun j j' k => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          X : J
          tX : CategoryTheory.Limits.IsTerminal X
          F : CategoryTheory.Functor J C
          j j' : J
          k : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map k) ((fun j => F.map (tX.from j …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          X : J
          tX : CategoryTheory.Limits.IsTerminal X
          F : CategoryTheory.Functor J C
          j j' : J
          k : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map k) (F.map (tX.from j'))) (Cate …
        -/
        rw [← F.map_comp, Category.comp_id, tX.hom_ext (k ≫ tX.from j') (tX.from j)] }
        /-
          🎉 no goals
        -/


/-- From a functor `F : J ⥤ C`, given a terminal object of `J`, show the cocone
`coconeOfDiagramTerminal` is a colimit. -/
def colimitOfDiagramTerminal {X : J} (tX : IsTerminal X) (F : J ⥤ C) :
    IsColimit (coconeOfDiagramTerminal tX F) where
  desc s := s.ι.app X
  uniq s m w := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      X : J
      tX : CategoryTheory.Limits.IsTerminal X
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfDiagramTerminal tX F).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq m ((fun s => s.ι.app X) s)
    -/
    conv_rhs => dsimp -- Porting note: why do I need this much firepower?
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      X : J
      tX : CategoryTheory.Limits.IsTerminal X
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfDiagramTerminal tX F).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq m (s.ι.app X)
    -/
    rw [← w X, coconeOfDiagramTerminal_ι_app, tX.hom_ext (tX.from X) (𝟙 _)]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      X : J
      tX : CategoryTheory.Limits.IsTerminal X
      F : CategoryTheory.Functor J C
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfDiagramTerminal tX F).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq m (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStru …
    -/
    simp
    /-
      🎉 no goals
    -/


lemma IsColimit.isIso_ι_app_of_isTerminal {F : J ⥤ C} {c : Cocone F} (hc : IsColimit c)
    (X : J) (hX : IsTerminal X) :
    IsIso (c.ι.app X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u
    inst✝ : CategoryTheory.Category.{v, u} J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X : J
    hX : CategoryTheory.Limits.IsTerminal X
    ⊢ CategoryTheory.IsIso (c.ι.app X)
  -/
  change IsIso (coconePointUniqueUpToIso (colimitOfDiagramTerminal hX F) hc).hom
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u
    inst✝ : CategoryTheory.Category.{v, u} J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cocone F
    hc : CategoryTheory.Limits.IsColimit c
    X : J
    hX : CategoryTheory.Limits.IsTerminal X
    ⊢ CategoryTheory.IsIso ((CategoryTheory.Limits.colimitOfDiagramTerminal hX F). …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- From a functor `F : J ⥤ C`, given an initial object of `J`, construct a cocone for `J`,
provided that the morphisms in the diagram are isomorphisms.
In `colimitOfDiagramInitial` we show it is a colimit cocone. -/
@[simps]
def coconeOfDiagramInitial {X : J} (hX : IsInitial X) (F : J ⥤ C)
    [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] : Cocone F where
  pt := F.obj X
  ι :=
    { app := fun _ => inv (F.map (hX.to _))
      naturality := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          X : J
          hX : CategoryTheory.Limits.IsInitial X
          F : CategoryTheory.Functor J C
          inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
          ⊢ ∀ ⦃X_1 Y : J⦄ (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategoryStruct.comp …
        -/
        intro i j f
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          J : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} J
          X : J
          hX : CategoryTheory.Limits.IsInitial X
          F : CategoryTheory.Functor J C
          inst✝ : ∀ (i j : J) (f : Quiver.Hom i j), CategoryTheory.IsIso (F.map f)
          i j : J
          f : Quiver.Hom i j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun x => CategoryTheory.i …
        -/
        dsimp
        simp only [IsIso.eq_inv_comp, IsIso.comp_inv_eq, Category.comp_id, ← F.map_comp,
          hX.hom_ext (hX.to i ≫ f) (hX.to j)] }


/-- From a functor `F : J ⥤ C`, given an initial object of `J` and that the morphisms in the
diagram are isomorphisms, show the cone `coconeOfDiagramInitial` is a colimit. -/
def colimitOfDiagramInitial {X : J} (hX : IsInitial X) (F : J ⥤ C)
    [∀ (i j : J) (f : i ⟶ j), IsIso (F.map f)] : IsColimit (coconeOfDiagramInitial hX F) where
  desc S := S.ι.app _


lemma IsLimit.isIso_π_app_of_isInitial {F : J ⥤ C} {c : Cone F} (hc : IsLimit c)
    (X : J) (hX : IsInitial X) :
    IsIso (c.π.app X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u
    inst✝ : CategoryTheory.Category.{v, u} J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    X : J
    hX : CategoryTheory.Limits.IsInitial X
    ⊢ CategoryTheory.IsIso (c.π.app X)
  -/
  change IsIso (conePointUniqueUpToIso hc (limitOfDiagramInitial hX F)).hom
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    J : Type u
    inst✝ : CategoryTheory.Category.{v, u} J
    F : CategoryTheory.Functor J C
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    X : J
    hX : CategoryTheory.Limits.IsInitial X
    ⊢ CategoryTheory.IsIso (hc.conePointUniqueUpToIso (CategoryTheory.Limits.limit …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Any morphism between terminal objects is an isomorphism. -/
lemma isIso_of_isTerminal {X Y : C} (hX : IsTerminal X) (hY : IsTerminal Y) (f : X ⟶ Y) :
    IsIso f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    hX : CategoryTheory.Limits.IsTerminal X
    hY : CategoryTheory.Limits.IsTerminal Y
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.IsIso f
  -/
  refine ⟨⟨IsTerminal.from hX Y, ?_⟩⟩
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    hX : CategoryTheory.Limits.IsTerminal X
    hY : CategoryTheory.Limits.IsTerminal Y
    f : Quiver.Hom X Y
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f (hX.from Y)) (CategoryTheory.C …
  -/
  simp only [IsTerminal.comp_from, IsTerminal.from_self, true_and]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    hX : CategoryTheory.Limits.IsTerminal X
    hY : CategoryTheory.Limits.IsTerminal Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (hX.from Y) f) (CategoryTheory.Catego …
  -/
  apply IsTerminal.hom_ext hY
  /-
    🎉 no goals
  -/


/-- Any morphism between initial objects is an isomorphism. -/
lemma isIso_of_isInitial {X Y : C} (hX : IsInitial X) (hY : IsInitial Y) (f : X ⟶ Y) :
    IsIso f := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    hX : CategoryTheory.Limits.IsInitial X
    hY : CategoryTheory.Limits.IsInitial Y
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.IsIso f
  -/
  refine ⟨⟨IsInitial.to hY X, ?_⟩⟩
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    hX : CategoryTheory.Limits.IsInitial X
    hY : CategoryTheory.Limits.IsInitial Y
    f : Quiver.Hom X Y
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f (hY.to X)) (CategoryTheory.Cat …
  -/
  simp only [IsInitial.to_comp, IsInitial.to_self, and_true]
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    hX : CategoryTheory.Limits.IsInitial X
    hY : CategoryTheory.Limits.IsInitial Y
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (hY.to X)) (CategoryTheory.Category …
  -/
  apply IsInitial.hom_ext hX
  /-
    🎉 no goals
  -/


