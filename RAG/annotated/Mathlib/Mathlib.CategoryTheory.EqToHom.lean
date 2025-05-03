/-- An equality `X = Y` gives us a morphism `X ⟶ Y`.

It is typically better to use this, rather than rewriting by the equality then using `𝟙 _`
which usually leads to dependent type theory hell.
-/
                                                /-
                                                  C : Type u₁
                                                  inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                  X Y : C
                                                  p : Eq X Y
                                                  ⊢ Quiver.Hom X Y
                                                -/
def eqToHom {X Y : C} (p : X = Y) : X ⟶ Y := by rw [p]; exact 𝟙 _
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem eqToHom_refl (X : C) (p : X = X) : eqToHom p = 𝟙 X :=
  rfl


@[reassoc (attr := simp)]
theorem eqToHom_trans {X Y Z : C} (p : X = Y) (q : Y = Z) :
    eqToHom p ≫ eqToHom q = eqToHom (p.trans q) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    p : Eq X Y
    q : Eq Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom p) (CategoryT …
  -/
  cases p
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Z : C
    q : Eq X Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  cases q
  /-
    case refl.refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (CategoryT …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Two morphisms are conjugate via eqToHom if and only if they are heterogeneously equal.
Note this used to be in the Functor namespace, where it doesn't belong. -/
theorem conj_eqToHom_iff_heq {W X Y Z : C} (f : W ⟶ X) (g : Y ⟶ Z) (h : W = Y) (h' : X = Z) :
    f = eqToHom h ≫ g ≫ eqToHom h'.symm ↔ HEq f g := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W X Y Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom Y Z
    h : Eq W Y
    h' : Eq X Z
    ⊢ Iff (Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom h) (Ca …
  -/
  cases h
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W X Z : C
    f : Quiver.Hom W X
    h' : Eq X Z
    g : Quiver.Hom W Z
    ⊢ Iff (Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Ca …
  -/
  cases h'
  /-
    case refl.refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    W X : C
    f g : Quiver.Hom W X
    ⊢ Iff (Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem conj_eqToHom_iff_heq' {C} [Category C] {W X Y Z : C}
    (f : W ⟶ X) (g : Y ⟶ Z) (h : W = Y) (h' : Z = X) :
    f = eqToHom h ≫ g ≫ eqToHom h' ↔ HEq f g := conj_eqToHom_iff_heq _ _ _ h'.symm


theorem comp_eqToHom_iff {X Y Y' : C} (p : Y = Y') (f : X ⟶ Y) (g : X ⟶ Y') :
    f ≫ eqToHom p = g ↔ f = g ≫ eqToHom p.symm :=
                          /-
                            C : Type u₁
                            inst✝ : CategoryTheory.Category.{v₁, u₁} C
                            X Y Y' : C
                            p : Eq Y Y'
                            f : Quiver.Hom X Y
                            g : Quiver.Hom X Y'
                            h : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.eqToHom p)) g
                            ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
                          -/
  { mp := fun h => h ▸ by simp
                          /-
                            🎉 no goals
                          -/
                       /-
                         C : Type u₁
                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                         X Y Y' : C
                         p : Eq Y Y'
                         f : Quiver.Hom X Y
                         g : Quiver.Hom X Y'
                         h : Eq f (CategoryTheory.CategoryStruct.comp g (CategoryTheory.eqToHom ⋯))
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.eqToHom p)) g
                       -/
    mpr := fun h => by simp [eq_whisker h (eqToHom p)] }
                       /-
                         🎉 no goals
                       -/


theorem eqToHom_comp_iff {X X' Y : C} (p : X = X') (f : X ⟶ Y) (g : X' ⟶ Y) :
    eqToHom p ≫ g = f ↔ g = eqToHom p.symm ≫ f :=
                          /-
                            C : Type u₁
                            inst✝ : CategoryTheory.Category.{v₁, u₁} C
                            X X' Y : C
                            p : Eq X X'
                            f : Quiver.Hom X Y
                            g : Quiver.Hom X' Y
                            h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom p) g) f
                            ⊢ Eq g (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Categor …
                          -/
  { mp := fun h => h ▸ by simp
                          /-
                            🎉 no goals
                          -/
                           /-
                             C : Type u₁
                             inst✝ : CategoryTheory.Category.{v₁, u₁} C
                             X X' Y : C
                             p : Eq X X'
                             f : Quiver.Hom X Y
                             g : Quiver.Hom X' Y
                             h : Eq g (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) f)
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom p) (CategoryT …
                           -/
    mpr := fun h => h ▸ by simp [whisker_eq _ h] }
                           /-
                             🎉 no goals
                           -/


theorem eqToHom_comp_heq {C} [Category C] {W X Y : C}
    (f : Y ⟶ X) (h : W = Y) : HEq (eqToHom h ≫ f) f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    W X Y : C
    f : Quiver.Hom Y X
    h : Eq W Y
    ⊢ HEq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom h) f) f
  -/
  rw [← conj_eqToHom_iff_heq _ _ h rfl, eqToHom_refl, Category.comp_id]
  /-
    🎉 no goals
  -/


@[simp] theorem eqToHom_comp_heq_iff {C} [Category C] {W X Y Z Z' : C}
    (f : Y ⟶ X) (g : Z ⟶ Z') (h : W = Y) :
    HEq (eqToHom h ≫ f) g ↔ HEq f g :=
  ⟨(eqToHom_comp_heq ..).symm.trans, (eqToHom_comp_heq ..).trans⟩


@[simp] theorem heq_eqToHom_comp_iff {C} [Category C] {W X Y Z Z' : C}
    (f : Y ⟶ X) (g : Z ⟶ Z') (h : W = Y) :
    HEq g (eqToHom h ≫ f) ↔ HEq g f :=
  ⟨(·.trans (eqToHom_comp_heq ..)), (·.trans (eqToHom_comp_heq ..).symm)⟩


theorem comp_eqToHom_heq {C} [Category C] {X Y Z : C}
    (f : X ⟶ Y) (h : Y = Z) : HEq (f ≫ eqToHom h) f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y Z : C
    f : Quiver.Hom X Y
    h : Eq Y Z
    ⊢ HEq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.eqToHom h)) f
  -/
  rw [← conj_eqToHom_iff_heq' _ _ rfl h, eqToHom_refl, Category.id_comp]
  /-
    🎉 no goals
  -/


@[simp] theorem comp_eqToHom_heq_iff {C} [Category C] {W X Y Z Z' : C}
    (f : X ⟶ Y) (g : Z ⟶ Z') (h : Y = W) :
    HEq (f ≫ eqToHom h) g ↔ HEq f g :=
  ⟨(comp_eqToHom_heq ..).symm.trans, (comp_eqToHom_heq ..).trans⟩


@[simp] theorem heq_comp_eqToHom_iff {C} [Category C] {W X Y Z Z' : C}
    (f : X ⟶ Y) (g : Z ⟶ Z') (h : Y = W) :
    HEq g (f ≫ eqToHom h) ↔ HEq g f :=
  ⟨(·.trans (comp_eqToHom_heq ..)), (·.trans (comp_eqToHom_heq ..).symm)⟩


theorem heq_comp {C} [Category C] {X Y Z X' Y' Z' : C}
    {f : X ⟶ Y} {g : Y ⟶ Z} {f' : X' ⟶ Y'} {g' : Y' ⟶ Z'}
    (eq1 : X = X') (eq2 : Y = Y') (eq3 : Z = Z')
    (H1 : HEq f f') (H2 : HEq g g') :
    HEq (f ≫ g) (f' ≫ g') := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X Y Z X' Y' Z' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom Y' Z'
    eq1 : Eq X X'
    eq2 : Eq Y Y'
    eq3 : Eq Z Z'
    H1 : HEq f f'
    H2 : HEq g g'
    ⊢ HEq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct. …
  -/
  cases eq1; cases eq2; cases eq3; cases H1; cases H2; rfl
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- We can push `eqToHom` to the left through families of morphisms. -/
-- The simpNF linter incorrectly claims that this will never apply.
-- https://github.com/leanprover-community/mathlib4/issues/5049
@[reassoc (attr := simp, nolint simpNF)]
theorem eqToHom_naturality {f g : β → C} (z : ∀ b, f b ⟶ g b) {j j' : β} (w : j = j') :
                      /-
                        C : Type u₁
                        inst✝ : CategoryTheory.Category.{v₁, u₁} C
                        β : Sort u_1
                        f g : β → C
                        z : (b : β) → Quiver.Hom (f b) (g b)
                        j j' : β
                        w : Eq j j'
                        ⊢ Eq (g j) (g j')
                      -/
                      /-
                        🎉 no goals
                      -/
    z j ≫ eqToHom (by simp [w]) = eqToHom (by simp [w]) ≫ z j' := by
                                              /-
                                                🎉 no goals
                                              -/
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    β : Sort u_1
    f g : β → C
    z : (b : β) → Quiver.Hom (f b) (g b)
    j j' : β
    w : Eq j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z j) (CategoryTheory.eqToHom ⋯)) (Ca …
  -/
  cases w
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    β : Sort u_1
    f g : β → C
    z : (b : β) → Quiver.Hom (f b) (g b)
    j : β
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z j) (CategoryTheory.eqToHom ⋯)) (Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A variant on `eqToHom_naturality` that helps Lean identify the families `f` and `g`. -/
-- The simpNF linter incorrectly claims that this will never apply.
-- https://github.com/leanprover-community/mathlib4/issues/5049
@[reassoc (attr := simp, nolint simpNF)]
theorem eqToHom_iso_hom_naturality {f g : β → C} (z : ∀ b, f b ≅ g b) {j j' : β} (w : j = j') :
                            /-
                              C : Type u₁
                              inst✝ : CategoryTheory.Category.{v₁, u₁} C
                              β : Sort u_1
                              f g : β → C
                              z : (b : β) → CategoryTheory.Iso (f b) (g b)
                              j j' : β
                              w : Eq j j'
                              ⊢ Eq (g j) (g j')
                            -/
                            /-
                              🎉 no goals
                            -/
    (z j).hom ≫ eqToHom (by simp [w]) = eqToHom (by simp [w]) ≫ (z j').hom := by
                                                    /-
                                                      🎉 no goals
                                                    -/
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    β : Sort u_1
    f g : β → C
    z : (b : β) → CategoryTheory.Iso (f b) (g b)
    j j' : β
    w : Eq j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z j).hom (CategoryTheory.eqToHom ⋯)) …
  -/
  cases w
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    β : Sort u_1
    f g : β → C
    z : (b : β) → CategoryTheory.Iso (f b) (g b)
    j : β
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z j).hom (CategoryTheory.eqToHom ⋯)) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A variant on `eqToHom_naturality` that helps Lean identify the families `f` and `g`. -/
-- The simpNF linter incorrectly claims that this will never apply.
-- https://github.com/leanprover-community/mathlib4/issues/5049
@[reassoc (attr := simp, nolint simpNF)]
theorem eqToHom_iso_inv_naturality {f g : β → C} (z : ∀ b, f b ≅ g b) {j j' : β} (w : j = j') :
                            /-
                              C : Type u₁
                              inst✝ : CategoryTheory.Category.{v₁, u₁} C
                              β : Sort u_1
                              f g : β → C
                              z : (b : β) → CategoryTheory.Iso (f b) (g b)
                              j j' : β
                              w : Eq j j'
                              ⊢ Eq (f j) (f j')
                            -/
                            /-
                              🎉 no goals
                            -/
    (z j).inv ≫ eqToHom (by simp [w]) = eqToHom (by simp [w]) ≫ (z j').inv := by
                                                    /-
                                                      🎉 no goals
                                                    -/
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    β : Sort u_1
    f g : β → C
    z : (b : β) → CategoryTheory.Iso (f b) (g b)
    j j' : β
    w : Eq j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z j).inv (CategoryTheory.eqToHom ⋯)) …
  -/
  cases w
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    β : Sort u_1
    f g : β → C
    z : (b : β) → CategoryTheory.Iso (f b) (g b)
    j : β
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (z j).inv (CategoryTheory.eqToHom ⋯)) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Reducible form of congrArg_mpr_hom_left -/
@[simp]
theorem congrArg_cast_hom_left {X Y Z : C} (p : X = Y) (q : Y ⟶ Z) :
    cast (congrArg (fun W : C => W ⟶ Z) p.symm) q = eqToHom p ≫ q := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    p : Eq X Y
    q : Quiver.Hom Y Z
    ⊢ Eq (cast ⋯ q) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom p) …
  -/
  cases p
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Z : C
    q : Quiver.Hom X Z
    ⊢ Eq (cast ⋯ q) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If we (perhaps unintentionally) perform equational rewriting on
the source object of a morphism,
we can replace the resulting `_.mpr f` term by a composition with an `eqToHom`.

It may be advisable to introduce any necessary `eqToHom` morphisms manually,
rather than relying on this lemma firing.
-/
theorem congrArg_mpr_hom_left {X Y Z : C} (p : X = Y) (q : Y ⟶ Z) :
    (congrArg (fun W : C => W ⟶ Z) p).mpr q = eqToHom p ≫ q := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    p : Eq X Y
    q : Quiver.Hom Y Z
    ⊢ Eq (⋯.mpr q) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom p) q)
  -/
  cases p
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Z : C
    q : Quiver.Hom X Z
    ⊢ Eq (⋯.mpr q) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) q)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Reducible form of `congrArg_mpr_hom_right` -/
@[simp]
theorem congrArg_cast_hom_right {X Y Z : C} (p : X ⟶ Y) (q : Z = Y) :
    cast (congrArg (fun W : C => X ⟶ W) q.symm) p = p ≫ eqToHom q.symm := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    p : Quiver.Hom X Y
    q : Eq Z Y
    ⊢ Eq (cast ⋯ p) (CategoryTheory.CategoryStruct.comp p (CategoryTheory.eqToHom  …
  -/
  cases q
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    p : Quiver.Hom X Y
    ⊢ Eq (cast ⋯ p) (CategoryTheory.CategoryStruct.comp p (CategoryTheory.eqToHom  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If we (perhaps unintentionally) perform equational rewriting on
the target object of a morphism,
we can replace the resulting `_.mpr f` term by a composition with an `eqToHom`.

It may be advisable to introduce any necessary `eqToHom` morphisms manually,
rather than relying on this lemma firing.
-/
theorem congrArg_mpr_hom_right {X Y Z : C} (p : X ⟶ Y) (q : Z = Y) :
    (congrArg (fun W : C => X ⟶ W) q).mpr p = p ≫ eqToHom q.symm := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y Z : C
    p : Quiver.Hom X Y
    q : Eq Z Y
    ⊢ Eq (⋯.mpr p) (CategoryTheory.CategoryStruct.comp p (CategoryTheory.eqToHom ⋯))
  -/
  cases q
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    p : Quiver.Hom X Y
    ⊢ Eq (⋯.mpr p) (CategoryTheory.CategoryStruct.comp p (CategoryTheory.eqToHom ⋯))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- An equality `X = Y` gives us an isomorphism `X ≅ Y`.

It is typically better to use this, rather than rewriting by the equality then using `Iso.refl _`
which usually leads to dependent type theory hell.
-/
def eqToIso {X Y : C} (p : X = Y) : X ≅ Y :=
                                 /-
                                   C : Type u₁
                                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                   β : Sort u_1
                                   X Y : C
                                   p : Eq X Y
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom p) (CategoryT …
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
  ⟨eqToHom p, eqToHom p.symm, by simp, by simp⟩
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem eqToIso.hom {X Y : C} (p : X = Y) : (eqToIso p).hom = eqToHom p :=
  rfl


@[simp]
theorem eqToIso.inv {X Y : C} (p : X = Y) : (eqToIso p).inv = eqToHom p.symm :=
  rfl


@[simp]
theorem eqToIso_refl {X : C} (p : X = X) : eqToIso p = Iso.refl X :=
  rfl


@[simp]
theorem eqToIso_trans {X Y Z : C} (p : X = Y) (q : Y = Z) :
                                                       /-
                                                         C : Type u₁
                                                         inst✝ : CategoryTheory.Category.{v₁, u₁} C
                                                         X Y Z : C
                                                         p : Eq X Y
                                                         q : Eq Y Z
                                                         ⊢ Eq ((CategoryTheory.eqToIso p).trans (CategoryTheory.eqToIso q)) (CategoryTh …
                                                       -/
    eqToIso p ≪≫ eqToIso q = eqToIso (p.trans q) := by ext; simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem eqToHom_op {X Y : C} (h : X = Y) : (eqToHom h).op = eqToHom (congr_arg op h.symm) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    h : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom h).op (CategoryTheory.eqToHom ⋯)
  -/
  cases h
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : C
    ⊢ Eq (CategoryTheory.eqToHom ⋯).op (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem eqToHom_unop {X Y : Cᵒᵖ} (h : X = Y) :
    (eqToHom h).unop = eqToHom (congr_arg unop h.symm) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : Opposite C
    h : Eq X Y
    ⊢ Eq (CategoryTheory.eqToHom h).unop (CategoryTheory.eqToHom ⋯)
  -/
  cases h
  /-
    case refl
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X : Opposite C
    ⊢ Eq (CategoryTheory.eqToHom ⋯).unop (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance {X Y : C} (h : X = Y) : IsIso (eqToHom h) :=
  (eqToIso h).isIso_hom


@[simp]
theorem inv_eqToHom {X Y : C} (h : X = Y) : inv (eqToHom h) = eqToHom h.symm := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    h : Eq X Y
    ⊢ Eq (CategoryTheory.inv (CategoryTheory.eqToHom h)) (CategoryTheory.eqToHom ⋯)
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- Proving equality between functors. This isn't an extensionality lemma,
  because usually you don't really want to do this. -/
theorem ext {F G : C ⥤ D} (h_obj : ∀ X, F.obj X = G.obj X)
    (h_map : ∀ X Y f,
      F.map f = eqToHom (h_obj X) ≫ G.map f ≫ eqToHom (h_obj Y).symm := by aesop_cat) :
    F = G := by
  match F, G with
  | mk F_pre _ _ , mk G_pre _ _ =>
    match F_pre, G_pre with  -- Porting note: did not unfold the Prefunctor unlike Lean3
    | Prefunctor.mk F_obj _ , Prefunctor.mk G_obj _ =>
    obtain rfl : F_obj = G_obj := by
      ext X
      apply h_obj
    congr
    funext X Y f
    simpa using h_map X Y f


lemma ext_of_iso {F G : C ⥤ D} (e : F ≅ G) (hobj : ∀ X, F.obj X = G.obj X)
    (happ : ∀ X, e.hom.app X = eqToHom (hobj X)) : F = G :=
  Functor.ext hobj (fun X Y f => by
    rw [← cancel_mono (e.hom.app Y), e.hom.naturality f, happ, happ, Category.assoc,
    Category.assoc, eqToHom_trans, eqToHom_refl, Category.comp_id])


/-- Proving equality between functors using heterogeneous equality. -/
theorem hext {F G : C ⥤ D} (h_obj : ∀ X, F.obj X = G.obj X)
    (h_map : ∀ (X Y) (f : X ⟶ Y), HEq (F.map f) (G.map f)) : F = G :=
  Functor.ext h_obj fun _ _ f => (conj_eqToHom_iff_heq _ _ (h_obj _) (h_obj _)).2 <| h_map _ _ f

-- Using equalities between functors.

                                                                          /-
                                                                            C : Type u₁
                                                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                            D : Type u₂
                                                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                            F G : CategoryTheory.Functor C D
                                                                            h : Eq F G
                                                                            X : C
                                                                            ⊢ Eq (F.obj X) (G.obj X)
                                                                          -/
theorem congr_obj {F G : C ⥤ D} (h : F = G) (X) : F.obj X = G.obj X := by rw [h]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem congr_hom {F G : C ⥤ D} (h : F = G) {X Y} (f : X ⟶ Y) :
    F.map f = eqToHom (congr_obj h X) ≫ G.map f ≫ eqToHom (congr_obj h Y).symm := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    h : Eq F G
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (F.map f) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯)  …
  -/
  subst h; simp
           /-
             🎉 no goals
           -/


theorem congr_inv_of_congr_hom (F G : C ⥤ D) {X Y : C} (e : X ≅ Y) (hX : F.obj X = G.obj X)
    (hY : F.obj Y = G.obj Y)
                                    /-
                                      C : Type u₁
                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                      β : Sort u_1
                                      D : Type u₂
                                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                      F G : CategoryTheory.Functor C D
                                      X Y : C
                                      e : CategoryTheory.Iso X Y
                                      hX : Eq (F.obj X) (G.obj X)
                                      hY : Eq (F.obj Y) (G.obj Y)
                                      ⊢ Eq (F.obj X) (G.obj X)
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
    (h₂ : F.map e.hom = eqToHom (by rw [hX]) ≫ G.map e.hom ≫ eqToHom (by rw [hY])) :
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
                              /-
                                C : Type u₁
                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                β : Sort u_1
                                D : Type u₂
                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                F G : CategoryTheory.Functor C D
                                X Y : C
                                e : CategoryTheory.Iso X Y
                                hX : Eq (F.obj X) (G.obj X)
                                hY : Eq (F.obj Y) (G.obj Y)
                                h₂ : Eq (F.map e.hom) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqTo …
                                ⊢ Eq (F.obj Y) (G.obj Y)
                              -/
                              /-
                                🎉 no goals
                              -/
    F.map e.inv = eqToHom (by rw [hY]) ≫ G.map e.inv ≫ eqToHom (by rw [hX]) := by
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  simp only [← IsIso.Iso.inv_hom e, Functor.map_inv, h₂, IsIso.inv_comp, inv_eqToHom,
    Category.assoc]


theorem map_comp_heq (hx : F.obj X = G.obj X) (hy : F.obj Y = G.obj Y) (hz : F.obj Z = G.obj Z)
    (hf : HEq (F.map f) (G.map f)) (hg : HEq (F.map g) (G.map g)) :
    HEq (F.map (f ≫ g)) (G.map (f ≫ g)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    hx : Eq (F.obj X) (G.obj X)
    hy : Eq (F.obj Y) (G.obj Y)
    hz : Eq (F.obj Z) (G.obj Z)
    hf : HEq (F.map f) (G.map f)
    hg : HEq (F.map g) (G.map g)
    ⊢ HEq (F.map (CategoryTheory.CategoryStruct.comp f g)) (G.map (CategoryTheory. …
  -/
  rw [F.map_comp, G.map_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    hx : Eq (F.obj X) (G.obj X)
    hy : Eq (F.obj Y) (G.obj Y)
    hz : Eq (F.obj Z) (G.obj Z)
    hf : HEq (F.map f) (G.map f)
    hg : HEq (F.map g) (G.map g)
    ⊢ HEq (CategoryTheory.CategoryStruct.comp (F.map f) (F.map g)) (CategoryTheory …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem map_comp_heq' (hobj : ∀ X : C, F.obj X = G.obj X)
    (hmap : ∀ {X Y} (f : X ⟶ Y), HEq (F.map f) (G.map f)) :
    HEq (F.map (f ≫ g)) (G.map (f ≫ g)) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    hobj : ∀ (X : C), Eq (F.obj X) (G.obj X)
    hmap : ∀ {X Y : C} (f : Quiver.Hom X Y), HEq (F.map f) (G.map f)
    ⊢ HEq (F.map (CategoryTheory.CategoryStruct.comp f g)) (G.map (CategoryTheory. …
  -/
  rw [Functor.hext hobj fun _ _ => hmap]
  /-
    🎉 no goals
  -/


theorem precomp_map_heq (H : E ⥤ C) (hmap : ∀ {X Y} (f : X ⟶ Y), HEq (F.map f) (G.map f)) {X Y : E}
    (f : X ⟶ Y) : HEq ((H ⋙ F).map f) ((H ⋙ G).map f) :=
  hmap _


theorem postcomp_map_heq (H : D ⥤ E) (hx : F.obj X = G.obj X) (hy : F.obj Y = G.obj Y)
    (hmap : HEq (F.map f) (G.map f)) : HEq ((F ⋙ H).map f) ((G ⋙ H).map f) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F G : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    H : CategoryTheory.Functor D E
    hx : Eq (F.obj X) (G.obj X)
    hy : Eq (F.obj Y) (G.obj Y)
    hmap : HEq (F.map f) (G.map f)
    ⊢ HEq ((F.comp H).map f) ((G.comp H).map f)
  -/
  dsimp
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    F G : CategoryTheory.Functor C D
    X Y : C
    f : Quiver.Hom X Y
    H : CategoryTheory.Functor D E
    hx : Eq (F.obj X) (G.obj X)
    hy : Eq (F.obj Y) (G.obj Y)
    hmap : HEq (F.map f) (G.map f)
    ⊢ HEq (H.map (F.map f)) (H.map (G.map f))
  -/
  congr
  /-
    🎉 no goals
  -/


theorem postcomp_map_heq' (H : D ⥤ E) (hobj : ∀ X : C, F.obj X = G.obj X)
    (hmap : ∀ {X Y} (f : X ⟶ Y), HEq (F.map f) (G.map f)) :
                                              /-
                                                C : Type u₁
                                                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                D : Type u₂
                                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                E : Type u₃
                                                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                                                F G : CategoryTheory.Functor C D
                                                X Y : C
                                                f : Quiver.Hom X Y
                                                H : CategoryTheory.Functor D E
                                                hobj : ∀ (X : C), Eq (F.obj X) (G.obj X)
                                                hmap : ∀ {X Y : C} (f : Quiver.Hom X Y), HEq (F.map f) (G.map f)
                                                ⊢ HEq ((F.comp H).map f) ((G.comp H).map f)
                                              -/
    HEq ((F ⋙ H).map f) ((G ⋙ H).map f) := by rw [Functor.hext hobj fun _ _ => hmap]
                                              /-
                                                🎉 no goals
                                              -/


theorem hcongr_hom {F G : C ⥤ D} (h : F = G) {X Y} (f : X ⟶ Y) : HEq (F.map f) (G.map f) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    h : Eq F G
    X Y : C
    f : Quiver.Hom X Y
    ⊢ HEq (F.map f) (G.map f)
  -/
  rw [h]
  /-
    🎉 no goals
  -/


/-- This is not always a good idea as a `@[simp]` lemma,
as we lose the ability to use results that interact with `F`,
e.g. the naturality of a natural transformation.

In some files it may be appropriate to use `attribute [local simp] eqToHom_map`, however.
-/
theorem eqToHom_map (F : C ⥤ D) {X Y : C} (p : X = Y) :
                                                          /-
                                                            C : Type u₁
                                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                            D : Type u₂
                                                            inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                            F : CategoryTheory.Functor C D
                                                            X Y : C
                                                            p : Eq X Y
                                                            ⊢ Eq (F.map (CategoryTheory.eqToHom p)) (CategoryTheory.eqToHom ⋯)
                                                          -/
    F.map (eqToHom p) = eqToHom (congr_arg F.obj p) := by cases p; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[reassoc (attr := simp)]
theorem eqToHom_map_comp (F : C ⥤ D) {X Y Z : C} (p : X = Y) (q : Y = Z) :
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 D : Type u₂
                                                                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                 F : CategoryTheory.Functor C D
                                                                                 X Y Z : C
                                                                                 p : Eq X Y
                                                                                 q : Eq Y Z
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.eqToHom p)) (F …
                                                                               -/
    F.map (eqToHom p) ≫ F.map (eqToHom q) = F.map (eqToHom <| p.trans q) := by aesop_cat
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- See the note on `eqToHom_map` regarding using this as a `simp` lemma.
-/
theorem eqToIso_map (F : C ⥤ D) {X Y : C} (p : X = Y) :
                                                             /-
                                                               C : Type u₁
                                                               inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                               D : Type u₂
                                                               inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                               F : CategoryTheory.Functor C D
                                                               X Y : C
                                                               p : Eq X Y
                                                               ⊢ Eq (F.mapIso (CategoryTheory.eqToIso p)) (CategoryTheory.eqToIso ⋯)
                                                             -/
    F.mapIso (eqToIso p) = eqToIso (congr_arg F.obj p) := by ext; cases p; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem eqToIso_map_trans (F : C ⥤ D) {X Y Z : C} (p : X = Y) (q : Y = Z) :
                                                                                         /-
                                                                                           C : Type u₁
                                                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                           D : Type u₂
                                                                                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                           F : CategoryTheory.Functor C D
                                                                                           X Y Z : C
                                                                                           p : Eq X Y
                                                                                           q : Eq Y Z
                                                                                           ⊢ Eq ((F.mapIso (CategoryTheory.eqToIso p)).trans (F.mapIso (CategoryTheory.eq …
                                                                                         -/
    F.mapIso (eqToIso p) ≪≫ F.mapIso (eqToIso q) = F.mapIso (eqToIso <| p.trans q) := by aesop_cat
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem eqToHom_app {F G : C ⥤ D} (h : F = G) (X : C) :
                                                                      /-
                                                                        C : Type u₁
                                                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                        D : Type u₂
                                                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                        F G : CategoryTheory.Functor C D
                                                                        h : Eq F G
                                                                        X : C
                                                                        ⊢ Eq ((CategoryTheory.eqToHom h).app X) (CategoryTheory.eqToHom ⋯)
                                                                      -/
    (eqToHom h : F ⟶ G).app X = eqToHom (Functor.congr_obj h X) := by subst h; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem NatTrans.congr {F G : C ⥤ D} (α : F ⟶ G) {X Y : C} (h : X = Y) :
    α.app X = F.map (eqToHom h) ≫ α.app Y ≫ G.map (eqToHom h.symm) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    X Y : C
    h : Eq X Y
    ⊢ Eq (α.app X) (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.eqTo …
  -/
  rw [α.naturality_assoc]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F G : CategoryTheory.Functor C D
    α : Quiver.Hom F G
    X Y : C
    h : Eq X Y
    ⊢ Eq (α.app X) (CategoryTheory.CategoryStruct.comp (α.app X) (CategoryTheory.C …
  -/
  simp [eqToHom_map]
  /-
    🎉 no goals
  -/


theorem eq_conj_eqToHom {X Y : C} (f : X ⟶ Y) : f = eqToHom rfl ≫ f ≫ eqToHom rfl := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Categor …
  -/
  simp only [Category.id_comp, eqToHom_refl, Category.comp_id]
  /-
    🎉 no goals
  -/


theorem dcongr_arg {ι : Type*} {F G : ι → C} (α : ∀ i, F i ⟶ G i) {i j : ι} (h : i = j) :
    α i = eqToHom (congr_arg F h) ≫ α j ≫ eqToHom (congr_arg G h.symm) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    ι : Type u_2
    F G : ι → C
    α : (i : ι) → Quiver.Hom (F i) (G i)
    i j : ι
    h : Eq i j
    ⊢ Eq (α i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cat …
  -/
  subst h
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    ι : Type u_2
    F G : ι → C
    α : (i : ι) → Quiver.Hom (F i) (G i)
    i : ι
    ⊢ Eq (α i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


