/--
Given a morphism `f : Y ⟶ X`, a presieve `S` on `Y` and presieve `T` on `X`,
we say that *`S` factors through `T` along `f`*, written `S.FactorsThruAlong T f`,
provided that for any morphism `g : Z ⟶ Y` in `S`, there exists some
morphism `e : W ⟶ X` in `T` and some morphism `i : Z ⟶ W` such that the obvious
square commutes: `i ≫ e = g ≫ f`.

This is used in the definition of a coverage.
-/
def FactorsThruAlong {X Y : C} (S : Presieve Y) (T : Presieve X) (f : Y ⟶ X) : Prop :=
  ∀ ⦃Z : C⦄ ⦃g : Z ⟶ Y⦄, S g →
  ∃ (W : C) (i : Z ⟶ W) (e : W ⟶ X), T e ∧ i ≫ e = g ≫ f


/--
Given `S T : Presieve X`, we say that `S` factors through `T` if any morphism in `S`
factors through some morphism in `T`.

The lemma `Presieve.isSheafFor_of_factorsThru` gives a *sufficient* condition for a
presheaf to be a sheaf for a presieve `T`, in terms of `S.FactorsThru T`, provided
that the presheaf is a sheaf for `S`.
-/
def FactorsThru {X : C} (S T : Presieve X) : Prop :=
  ∀ ⦃Z : C⦄ ⦃g : Z ⟶ X⦄, S g →
  ∃ (W : C) (i : Z ⟶ W) (e : W ⟶ X), T e ∧ i ≫ e = g


@[simp]
lemma factorsThruAlong_id {X : C} (S T : Presieve X) :
    S.FactorsThruAlong T (𝟙 X) ↔ S.FactorsThru T := by
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X : C
    S T : CategoryTheory.Presieve X
    ⊢ Iff (S.FactorsThruAlong T (CategoryTheory.CategoryStruct.id X)) (S.FactorsTh …
  -/
  simp [FactorsThruAlong, FactorsThru]
  /-
    🎉 no goals
  -/


lemma factorsThru_of_le {X : C} (S T : Presieve X) (h : S ≤ T) :
    S.FactorsThru T :=
                                       /-
                                         C : Type u_2
                                         inst✝ : CategoryTheory.Category.{u_1, u_2} C
                                         X : C
                                         S T : CategoryTheory.Presieve X
                                         h : LE.le S T
                                         Y : C
                                         g : Quiver.Hom Y X
                                         hg : S g
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Y)  …
                                       -/
  fun Y g hg => ⟨Y, 𝟙 _, g, h _ hg, by simp⟩
                                       /-
                                         🎉 no goals
                                       -/


lemma le_of_factorsThru_sieve {X : C} (S : Presieve X) (T : Sieve X) (h : S.FactorsThru T) :
    S ≤ T := by
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X : C
    S : CategoryTheory.Presieve X
    T : CategoryTheory.Sieve X
    h : S.FactorsThru T.arrows
    ⊢ LE.le S T.arrows
  -/
  rintro Y f hf
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X : C
    S : CategoryTheory.Presieve X
    T : CategoryTheory.Sieve X
    h : S.FactorsThru T.arrows
    Y : C
    f : Quiver.Hom Y X
    hf : Membership.mem S f
    ⊢ Membership.mem T.arrows f
  -/
  obtain ⟨W, i, e, h1, rfl⟩ := h hf
  /-
    case intro.intro.intro.intro
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X : C
    S : CategoryTheory.Presieve X
    T : CategoryTheory.Sieve X
    h : S.FactorsThru T.arrows
    Y W : C
    i : Quiver.Hom Y W
    e : Quiver.Hom W X
    h1 : T.arrows e
    hf : Membership.mem S (CategoryTheory.CategoryStruct.comp i e)
    ⊢ Membership.mem T.arrows (CategoryTheory.CategoryStruct.comp i e)
  -/
  exact T.downward_closed h1 _
  /-
    🎉 no goals
  -/


lemma factorsThru_top {X : C} (S : Presieve X) : S.FactorsThru ⊤ :=
  factorsThru_of_le _ _ le_top


lemma isSheafFor_of_factorsThru
    {X : C} {S T : Presieve X}
    (P : Cᵒᵖ ⥤ Type*)
    (H : S.FactorsThru T) (hS : S.IsSheafFor P)
    (h : ∀ ⦃Y : C⦄ ⦃f : Y ⟶ X⦄, T f → ∃ (R : Presieve Y),
      R.IsSeparatedFor P ∧ R.FactorsThruAlong S f) :
    T.IsSheafFor P := by
  /-
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    H : S.FactorsThru T
    hS : CategoryTheory.Presieve.IsSheafFor P S
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    ⊢ CategoryTheory.Presieve.IsSheafFor P T
  -/
  simp only [← Presieve.isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor] at *
  /-
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    H : S.FactorsThru T
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    ⊢ And (CategoryTheory.Presieve.IsSeparatedFor P T) (∀ (x : CategoryTheory.Pres …
  -/
  choose W i e h1 h2 using H
  /-
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    ⊢ And (CategoryTheory.Presieve.IsSeparatedFor P T) (∀ (x : CategoryTheory.Pres …
  -/
  refine ⟨?_, fun x hx => ?_⟩
    /-
      case refine_1
      C : Type u_3
      inst✝ : CategoryTheory.Category.{u_2, u_3} C
      X : C
      S T : CategoryTheory.Presieve X
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
      hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
      W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
      i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
      e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
      h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
      h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
      ⊢ CategoryTheory.Presieve.IsSeparatedFor P T
    -/
  · intro x y₁ y₂ h₁ h₂
    /-
      case refine_1
      C : Type u_3
      inst✝ : CategoryTheory.Category.{u_2, u_3} C
      X : C
      S T : CategoryTheory.Presieve X
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
      hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
      W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
      i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
      e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
      h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
      h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
      x : CategoryTheory.Presieve.FamilyOfElements P T
      y₁ y₂ : P.obj { unop := X }
      h₁ : x.IsAmalgamation y₁
      h₂ : x.IsAmalgamation y₂
      ⊢ Eq y₁ y₂
    -/
    refine hS.1.ext (fun Y g hg => ?_)
    /-
      case refine_1
      C : Type u_3
      inst✝ : CategoryTheory.Category.{u_2, u_3} C
      X : C
      S T : CategoryTheory.Presieve X
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
      hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
      W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
      i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
      e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
      h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
      h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
      x : CategoryTheory.Presieve.FamilyOfElements P T
      y₁ y₂ : P.obj { unop := X }
      h₁ : x.IsAmalgamation y₁
      h₂ : x.IsAmalgamation y₂
      Y : C
      g : Quiver.Hom Y X
      hg : S g
      ⊢ Eq (P.map g.op y₁) (P.map g.op y₂)
    -/
    simp only [← h2 hg, op_comp, P.map_comp, types_comp_apply, h₁ _ (h1 _ ), h₂ _ (h1 _)]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    ⊢ Exists fun t => x.IsAmalgamation t
  -/
  let y : S.FamilyOfElements P := fun Y g hg => P.map (i _).op (x (e hg) (h1 _))
  have hy : y.Compatible := by
    intro Y₁ Y₂ Z g₁ g₂ f₁ f₂ h₁ h₂ h
    rw [← types_comp_apply (P.map (i h₁).op) (P.map g₁.op),
      ← types_comp_apply (P.map (i h₂).op) (P.map g₂.op),
      ← P.map_comp, ← op_comp, ← P.map_comp, ← op_comp]
    apply hx
    simp only [h2, h, Category.assoc]
  /-
    case refine_2
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    y : CategoryTheory.Presieve.FamilyOfElements P S := fun Y g hg => P.map (i hg) …
    hy : y.Compatible
    ⊢ Exists fun t => x.IsAmalgamation t
  -/
  let ⟨_, h2'⟩ := hS
  /-
    case refine_2
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    y : CategoryTheory.Presieve.FamilyOfElements P S := fun Y g hg => P.map (i hg) …
    hy : y.Compatible
    left✝ : CategoryTheory.Presieve.IsSeparatedFor P S
    h2' : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P S), x.Compatible → Exi …
    ⊢ Exists fun t => x.IsAmalgamation t
  -/
  obtain ⟨z, hz⟩ := h2' y hy
  /-
    case refine_2.intro
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    y : CategoryTheory.Presieve.FamilyOfElements P S := fun Y g hg => P.map (i hg) …
    hy : y.Compatible
    left✝ : CategoryTheory.Presieve.IsSeparatedFor P S
    h2' : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P S), x.Compatible → Exi …
    z : P.obj { unop := X }
    hz : y.IsAmalgamation z
    ⊢ Exists fun t => x.IsAmalgamation t
  -/
  refine ⟨z, fun Y g hg => ?_⟩
  /-
    case refine_2.intro
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    y : CategoryTheory.Presieve.FamilyOfElements P S := fun Y g hg => P.map (i hg) …
    hy : y.Compatible
    left✝ : CategoryTheory.Presieve.IsSeparatedFor P S
    h2' : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P S), x.Compatible → Exi …
    z : P.obj { unop := X }
    hz : y.IsAmalgamation z
    Y : C
    g : Quiver.Hom Y X
    hg : T g
    ⊢ Eq (P.map g.op z) (x g hg)
  -/
  obtain ⟨R, hR1, hR2⟩ := h hg
  /-
    case refine_2.intro.intro.intro
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    y : CategoryTheory.Presieve.FamilyOfElements P S := fun Y g hg => P.map (i hg) …
    hy : y.Compatible
    left✝ : CategoryTheory.Presieve.IsSeparatedFor P S
    h2' : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P S), x.Compatible → Exi …
    z : P.obj { unop := X }
    hz : y.IsAmalgamation z
    Y : C
    g : Quiver.Hom Y X
    hg : T g
    R : CategoryTheory.Presieve Y
    hR1 : CategoryTheory.Presieve.IsSeparatedFor P R
    hR2 : R.FactorsThruAlong S g
    ⊢ Eq (P.map g.op z) (x g hg)
  -/
  choose WW ii ee hh1 hh2 using hR2
  /-
    case refine_2.intro.intro.intro
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    y : CategoryTheory.Presieve.FamilyOfElements P S := fun Y g hg => P.map (i hg) …
    hy : y.Compatible
    left✝ : CategoryTheory.Presieve.IsSeparatedFor P S
    h2' : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P S), x.Compatible → Exi …
    z : P.obj { unop := X }
    hz : y.IsAmalgamation z
    Y : C
    g : Quiver.Hom Y X
    hg : T g
    R : CategoryTheory.Presieve Y
    hR1 : CategoryTheory.Presieve.IsSeparatedFor P R
    WW : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → R g → C
    ii : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → (a : R g) → Quiver.Hom Z (WW a)
    ee : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → (a : R g) → Quiver.Hom (WW a) X
    hh1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z Y⦄ (a : R g), S (ee a)
    hh2 : ∀ ⦃Z : C⦄ ⦃g_1 : Quiver.Hom Z Y⦄ (a : R g_1), Eq (CategoryTheory.Categor …
    ⊢ Eq (P.map g.op z) (x g hg)
  -/
  refine hR1.ext (fun Q t ht => ?_)
  rw [← types_comp_apply (P.map g.op) (P.map t.op), ← P.map_comp, ← op_comp, ← hh2 ht,
    op_comp, P.map_comp, types_comp_apply, hz _ (hh1 _),
    ← types_comp_apply _ (P.map (ii ht).op), ← P.map_comp, ← op_comp]
  /-
    case refine_2.intro.intro.intro
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    y : CategoryTheory.Presieve.FamilyOfElements P S := fun Y g hg => P.map (i hg) …
    hy : y.Compatible
    left✝ : CategoryTheory.Presieve.IsSeparatedFor P S
    h2' : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P S), x.Compatible → Exi …
    z : P.obj { unop := X }
    hz : y.IsAmalgamation z
    Y : C
    g : Quiver.Hom Y X
    hg : T g
    R : CategoryTheory.Presieve Y
    hR1 : CategoryTheory.Presieve.IsSeparatedFor P R
    WW : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → R g → C
    ii : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → (a : R g) → Quiver.Hom Z (WW a)
    ee : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → (a : R g) → Quiver.Hom (WW a) X
    hh1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z Y⦄ (a : R g), S (ee a)
    hh2 : ∀ ⦃Z : C⦄ ⦃g_1 : Quiver.Hom Z Y⦄ (a : R g_1), Eq (CategoryTheory.Categor …
    Q : C
    t : Quiver.Hom Q Y
    ht : R t
    ⊢ Eq (P.map (CategoryTheory.CategoryStruct.comp (ii ht) (i ⋯)).op (x (e ⋯) ⋯)) …
  -/
  apply hx
  /-
    case refine_2.intro.intro.intro.a
    C : Type u_3
    inst✝ : CategoryTheory.Category.{u_2, u_3} C
    X : C
    S T : CategoryTheory.Presieve X
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, T f → Exists fun R => And (CategoryTheory. …
    hS : And (CategoryTheory.Presieve.IsSeparatedFor P S) (∀ (x : CategoryTheory.P …
    W : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → S g → C
    i : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom Z (W a)
    e : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z X⦄ → (a : S g) → Quiver.Hom (W a) X
    h1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), T (e a)
    h2 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z X⦄ (a : S g), Eq (CategoryTheory.CategoryStru …
    x : CategoryTheory.Presieve.FamilyOfElements P T
    hx : x.Compatible
    y : CategoryTheory.Presieve.FamilyOfElements P S := fun Y g hg => P.map (i hg) …
    hy : y.Compatible
    left✝ : CategoryTheory.Presieve.IsSeparatedFor P S
    h2' : ∀ (x : CategoryTheory.Presieve.FamilyOfElements P S), x.Compatible → Exi …
    z : P.obj { unop := X }
    hz : y.IsAmalgamation z
    Y : C
    g : Quiver.Hom Y X
    hg : T g
    R : CategoryTheory.Presieve Y
    hR1 : CategoryTheory.Presieve.IsSeparatedFor P R
    WW : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → R g → C
    ii : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → (a : R g) → Quiver.Hom Z (WW a)
    ee : ⦃Z : C⦄ → ⦃g : Quiver.Hom Z Y⦄ → (a : R g) → Quiver.Hom (WW a) X
    hh1 : ∀ ⦃Z : C⦄ ⦃g : Quiver.Hom Z Y⦄ (a : R g), S (ee a)
    hh2 : ∀ ⦃Z : C⦄ ⦃g_1 : Quiver.Hom Z Y⦄ (a : R g_1), Eq (CategoryTheory.Categor …
    Q : C
    t : Quiver.Hom Q Y
    ht : R t
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, h2, hh2]
  /-
    🎉 no goals
  -/



variable (C) in
/--
The type `Coverage C` of coverages on `C`.
A coverage is a collection of *covering* presieves on every object `X : C`,
which satisfies a *pullback compatibility* condition.
Explicitly, this condition says that whenever `S` is a covering presieve for `X` and
`f : Y ⟶ X` is a morphism, then there exists some covering presieve `T` for `Y`
such that `T` factors through `S` along `f`.
-/
@[ext]
structure Coverage where
  /-- The collection of covering presieves for an object `X`. -/
  covering : ∀ (X : C), Set (Presieve X)
  /-- Given any covering sieve `S` on `X` and a morphism `f : Y ⟶ X`, there exists
    some covering sieve `T` on `Y` such that `T` factors through `S` along `f`. -/
  pullback : ∀ ⦃X Y : C⦄ (f : Y ⟶ X) (S : Presieve X) (_ : S ∈ covering X),
    ∃ (T : Presieve Y), T ∈ covering Y ∧ T.FactorsThruAlong S f


instance : CoeFun (Coverage C) (fun _ => (X : C) → Set (Presieve X)) where
  coe := covering


variable (C) in
/--
Associate a coverage to any Grothendieck topology.
If `J` is a Grothendieck topology, and `K` is the associated coverage, then a presieve
`S` is a covering presieve for `K` if and only if the sieve that it generates is a
covering sieve for `J`.
-/
def ofGrothendieck (J : GrothendieckTopology C) : Coverage C where
  covering X := { S | Sieve.generate S ∈ J X }
  pullback := by
    /-
      C : Type ?u.30665
      D : Type ?u.30668
      inst✝¹ : CategoryTheory.Category.{?u.30672, ?u.30665} C
      inst✝ : CategoryTheory.Category.{?u.30676, ?u.30668} D
      J : CategoryTheory.GrothendieckTopology C
      ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Presieve X), Membership …
    -/
    intro X Y f S (hS : Sieve.generate S ∈ J X)
    /-
      C : Type ?u.30665
      D : Type ?u.30668
      inst✝¹ : CategoryTheory.Category.{?u.30672, ?u.30665} C
      inst✝ : CategoryTheory.Category.{?u.30676, ?u.30668} D
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem (J X) (CategoryTheory.Sieve.generate S)
      ⊢ Exists fun T => And (Membership.mem ((fun X => setOf fun S => Membership.mem …
    -/
    refine ⟨(Sieve.generate S).pullback f, ?_, fun Z g h => h⟩
    /-
      C : Type ?u.30665
      D : Type ?u.30668
      inst✝¹ : CategoryTheory.Category.{?u.30672, ?u.30665} C
      inst✝ : CategoryTheory.Category.{?u.30676, ?u.30668} D
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem (J X) (CategoryTheory.Sieve.generate S)
      ⊢ Membership.mem ((fun X => setOf fun S => Membership.mem (J X) (CategoryTheor …
    -/
    dsimp
    /-
      C : Type ?u.30665
      D : Type ?u.30668
      inst✝¹ : CategoryTheory.Category.{?u.30672, ?u.30665} C
      inst✝ : CategoryTheory.Category.{?u.30676, ?u.30668} D
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem (J X) (CategoryTheory.Sieve.generate S)
      ⊢ Membership.mem (J Y) (CategoryTheory.Sieve.generate (CategoryTheory.Sieve.pu …
    -/
    rw [Sieve.generate_sieve]
    /-
      C : Type ?u.30665
      D : Type ?u.30668
      inst✝¹ : CategoryTheory.Category.{?u.30672, ?u.30665} C
      inst✝ : CategoryTheory.Category.{?u.30676, ?u.30668} D
      J : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Presieve X
      hS : Membership.mem (J X) (CategoryTheory.Sieve.generate S)
      ⊢ Membership.mem (J Y) (CategoryTheory.Sieve.pullback f (CategoryTheory.Sieve. …
    -/
    exact J.pullback_stable _ hS
    /-
      🎉 no goals
    -/


lemma ofGrothendieck_iff {X : C} {S : Presieve X} (J : GrothendieckTopology C) :
    S ∈ ofGrothendieck _ J X ↔ Sieve.generate S ∈ J X := Iff.rfl


/--
An auxiliary definition used to define the Grothendieck topology associated to a
coverage. See `Coverage.toGrothendieck`.
-/
inductive Saturate (K : Coverage C) : (X : C) → Sieve X → Prop where
  | of (X : C) (S : Presieve X) (hS : S ∈ K X) : Saturate K X (Sieve.generate S)
  | top (X : C) : Saturate K X ⊤
  | transitive (X : C) (R S : Sieve X) :
    Saturate K X R →
    (∀ ⦃Y : C⦄ ⦃f : Y ⟶ X⦄, R f → Saturate K Y (S.pullback f)) →
    Saturate K X S


lemma eq_top_pullback {X Y : C} {S T : Sieve X} (h : S ≤ T) (f : Y ⟶ X) (hf : S f) :
    T.pullback f = ⊤ := by
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X Y : C
    S T : CategoryTheory.Sieve X
    h : LE.le S T
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ Eq (CategoryTheory.Sieve.pullback f T) Top.top
  -/
  ext Z g
  /-
    case h
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X Y : C
    S T : CategoryTheory.Sieve X
    h : LE.le S T
    f : Quiver.Hom Y X
    hf : S.arrows f
    Z : C
    g : Quiver.Hom Z Y
    ⊢ Iff ((CategoryTheory.Sieve.pullback f T).arrows g) (Top.top.arrows g)
  -/
  simp only [Sieve.pullback_apply, Sieve.top_apply, iff_true]
  /-
    case h
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X Y : C
    S T : CategoryTheory.Sieve X
    h : LE.le S T
    f : Quiver.Hom Y X
    hf : S.arrows f
    Z : C
    g : Quiver.Hom Z Y
    ⊢ T.arrows (CategoryTheory.CategoryStruct.comp g f)
  -/
  apply h
  /-
    case h.a
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X Y : C
    S T : CategoryTheory.Sieve X
    h : LE.le S T
    f : Quiver.Hom Y X
    hf : S.arrows f
    Z : C
    g : Quiver.Hom Z Y
    ⊢ S.arrows (CategoryTheory.CategoryStruct.comp g f)
  -/
  apply S.downward_closed
  /-
    case h.a.x
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_1, u_2} C
    X Y : C
    S T : CategoryTheory.Sieve X
    h : LE.le S T
    f : Quiver.Hom Y X
    hf : S.arrows f
    Z : C
    g : Quiver.Hom Z Y
    ⊢ S.arrows f
  -/
  exact hf
  /-
    🎉 no goals
  -/


lemma saturate_of_superset (K : Coverage C) {X : C} {S T : Sieve X} (h : S ≤ T)
    (hS : Saturate K X S) : Saturate K X T := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    K : CategoryTheory.Coverage C
    X : C
    S T : CategoryTheory.Sieve X
    h : LE.le S T
    hS : K.Saturate X S
    ⊢ K.Saturate X T
  -/
  apply Saturate.transitive _ _ _ hS
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    K : CategoryTheory.Coverage C
    X : C
    S T : CategoryTheory.Sieve X
    h : LE.le S T
    hS : K.Saturate X S
    ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → K.Saturate Y (CategoryTheory.Si …
  -/
  intro Y g hg
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    K : CategoryTheory.Coverage C
    X : C
    S T : CategoryTheory.Sieve X
    h : LE.le S T
    hS : K.Saturate X S
    Y : C
    g : Quiver.Hom Y X
    hg : S.arrows g
    ⊢ K.Saturate Y (CategoryTheory.Sieve.pullback g T)
  -/
  rw [eq_top_pullback (h := h)]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      K : CategoryTheory.Coverage C
      X : C
      S T : CategoryTheory.Sieve X
      h : LE.le S T
      hS : K.Saturate X S
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      ⊢ K.Saturate Y Top.top
    -/
  · apply Saturate.top
    /-
      🎉 no goals
    -/
    /-
      case hf
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      K : CategoryTheory.Coverage C
      X : C
      S T : CategoryTheory.Sieve X
      h : LE.le S T
      hS : K.Saturate X S
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      ⊢ S.arrows g
    -/
  · assumption
    /-
      🎉 no goals
    -/


variable (C) in
/--
The Grothendieck topology associated to a coverage `K`.
It is defined *inductively* as follows:
1. If `S` is a covering presieve for `K`, then the sieve generated by `S` is a covering
  sieve for the associated Grothendieck topology.
2. The top sieves are in the associated Grothendieck topology.
3. Add all sieves required by the *local character* axiom of a Grothendieck topology.

The pullback compatibility condition for a coverage ensures that the
associated Grothendieck topology is pullback stable, and so an additional constructor
in the inductive construction is not needed.
-/
def toGrothendieck (K : Coverage C) : GrothendieckTopology C where
  sieves := Saturate K
  top_mem' := .top
  pullback_stable' := by
    /-
      C : Type ?u.35380
      D : Type ?u.35383
      inst✝¹ : CategoryTheory.Category.{?u.35387, ?u.35380} C
      inst✝ : CategoryTheory.Category.{?u.35391, ?u.35383} D
      K : CategoryTheory.Coverage C
      ⊢ ∀ ⦃X Y : C⦄ ⦃S : CategoryTheory.Sieve X⦄ (f : Quiver.Hom Y X), Membership.me …
    -/
    intro X Y S f hS
    induction hS generalizing Y with
    | of X S hS =>
      obtain ⟨R,hR1,hR2⟩ := K.pullback f S hS
      suffices Sieve.generate R ≤ (Sieve.generate S).pullback f from
        saturate_of_superset _ this (Saturate.of _ _ hR1)
      rintro Z g ⟨W, i, e, h1, h2⟩
      obtain ⟨WW, ii, ee, hh1, hh2⟩ := hR2 h1
      refine ⟨WW, i ≫ ii, ee, hh1, ?_⟩
      simp only [hh2, reassoc_of% h2, Category.assoc]
    | top X => apply Saturate.top
    | transitive X R S _ hS H1 _ =>
      apply Saturate.transitive
      · apply H1 f
      intro Z g hg
      rw [← Sieve.pullback_comp]
      exact hS hg
  transitive' _ _ hS _ hR := .transitive _ _ _ hS hR


instance : PartialOrder (Coverage C) where
  le A B := A.covering ≤ B.covering
  le_refl _ _ := le_refl _
  le_trans _ _ _ h1 h2 X := le_trans (h1 X) (h2 X)
  le_antisymm _ _ h1 h2 := Coverage.ext <| funext <|
    fun X => le_antisymm (h1 X) (h2 X)


variable (C) in
/--
The two constructions `Coverage.toGrothendieck` and `Coverage.ofGrothendieck` form
a Galois insertion.
-/
def gi : GaloisInsertion (toGrothendieck C) (ofGrothendieck C) where
  choice K _ := toGrothendieck _ K
  choice_eq := fun _ _ => rfl
  le_l_u J X S hS := by
    /-
      C : Type ?u.38433
      D : Type ?u.38436
      inst✝¹ : CategoryTheory.Category.{?u.38440, ?u.38433} C
      inst✝ : CategoryTheory.Category.{?u.38444, ?u.38436} D
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      ⊢ Membership.mem ((CategoryTheory.Coverage.toGrothendieck C (CategoryTheory.Co …
    -/
    rw [← Sieve.generate_sieve S]
    /-
      C : Type ?u.38433
      D : Type ?u.38436
      inst✝¹ : CategoryTheory.Category.{?u.38440, ?u.38433} C
      inst✝ : CategoryTheory.Category.{?u.38444, ?u.38436} D
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      ⊢ Membership.mem ((CategoryTheory.Coverage.toGrothendieck C (CategoryTheory.Co …
    -/
    apply Saturate.of
    /-
      case hS
      C : Type ?u.38433
      D : Type ?u.38436
      inst✝¹ : CategoryTheory.Category.{?u.38440, ?u.38433} C
      inst✝ : CategoryTheory.Category.{?u.38444, ?u.38436} D
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      ⊢ Membership.mem ((CategoryTheory.Coverage.ofGrothendieck C J).covering X) S.a …
    -/
    /-
      C : Type ?u.38433
      D : Type ?u.38436
      inst✝¹ : CategoryTheory.Category.{?u.38440, ?u.38433} C
      inst✝ : CategoryTheory.Category.{?u.38444, ?u.38436} D
      K : CategoryTheory.Coverage C
      J : CategoryTheory.GrothendieckTopology C
      ⊢ Iff (LE.le (CategoryTheory.Coverage.toGrothendieck C K) J) (LE.le K (Categor …
    -/
    dsimp [ofGrothendieck]
      /-
        case mp
        C : Type ?u.38433
        D : Type ?u.38436
        inst✝¹ : CategoryTheory.Category.{?u.38440, ?u.38433} C
        inst✝ : CategoryTheory.Category.{?u.38444, ?u.38436} D
        K : CategoryTheory.Coverage C
        J : CategoryTheory.GrothendieckTopology C
        ⊢ LE.le (CategoryTheory.Coverage.toGrothendieck C K) J → LE.le K (CategoryTheo …
      -/
    /-
      case hS
      C : Type ?u.38433
      D : Type ?u.38436
      inst✝¹ : CategoryTheory.Category.{?u.38440, ?u.38433} C
      inst✝ : CategoryTheory.Category.{?u.38444, ?u.38436} D
      J : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J X) S
      ⊢ Membership.mem (J X) (CategoryTheory.Sieve.generate S.arrows)
    -/
      /-
        case mp
        C : Type ?u.38433
        D : Type ?u.38436
        inst✝¹ : CategoryTheory.Category.{?u.38440, ?u.38433} C
        inst✝ : CategoryTheory.Category.{?u.38444, ?u.38436} D
        K : CategoryTheory.Coverage C
        J : CategoryTheory.GrothendieckTopology C
        H : LE.le (CategoryTheory.Coverage.toGrothendieck C K) J
        X : C
        S : CategoryTheory.Presieve X
        hS : Membership.mem (K.covering X) S
        ⊢ Membership.mem ((CategoryTheory.Coverage.ofGrothendieck C J).covering X) S
      -/
    rwa [Sieve.generate_sieve S]
      /-
        🎉 no goals
      -/
      /-
        case mpr
        C : Type ?u.38433
        D : Type ?u.38436
        inst✝¹ : CategoryTheory.Category.{?u.38440, ?u.38433} C
        inst✝ : CategoryTheory.Category.{?u.38444, ?u.38436} D
        K : CategoryTheory.Coverage C
        J : CategoryTheory.GrothendieckTopology C
        ⊢ LE.le K (CategoryTheory.Coverage.ofGrothendieck C J) → LE.le (CategoryTheory …
      -/
    /-
      🎉 no goals
    -/
  gc K J := by
    constructor
    · intro H X S hS
      exact H _ <| Saturate.of _ _ hS
    · intro H X S hS
      induction hS with
      | of X S hS => exact H _ hS
      | top => apply J.top_mem
      | transitive X R S _ _ H1 H2 => exact J.transitive H1 _ H2


/--
An alternative characterization of the Grothendieck topology associated to a coverage `K`:
it is the infimum of all Grothendieck topologies whose associated coverage contains `K`.
-/
theorem toGrothendieck_eq_sInf (K : Coverage C) : toGrothendieck _ K =
    sInf {J | K ≤ ofGrothendieck _ J } := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    K : CategoryTheory.Coverage C
    ⊢ Eq (CategoryTheory.Coverage.toGrothendieck C K) (InfSet.sInf (setOf fun J => …
  -/
  apply le_antisymm
    /-
      case a
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      K : CategoryTheory.Coverage C
      ⊢ LE.le (CategoryTheory.Coverage.toGrothendieck C K) (InfSet.sInf (setOf fun J …
    -/
  · apply le_sInf; intro J hJ
    /-
      case a.a
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      K : CategoryTheory.Coverage C
      J : CategoryTheory.GrothendieckTopology C
      hJ : Membership.mem (setOf fun J => LE.le K (CategoryTheory.Coverage.ofGrothen …
      ⊢ LE.le (CategoryTheory.Coverage.toGrothendieck C K) J
    -/
    intro X S hS
    induction hS with
    | of X S hS => apply hJ; assumption
    | top => apply J.top_mem
    | transitive X R S _ _ H1 H2 => exact J.transitive H1 _ H2
    /-
      case a
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      K : CategoryTheory.Coverage C
      ⊢ LE.le (InfSet.sInf (setOf fun J => LE.le K (CategoryTheory.Coverage.ofGrothe …
    -/
  · apply sInf_le
    /-
      case a.a
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      K : CategoryTheory.Coverage C
      ⊢ Membership.mem (setOf fun J => LE.le K (CategoryTheory.Coverage.ofGrothendie …
    -/
    intro X S hS
    /-
      case a.a
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      K : CategoryTheory.Coverage C
      X : C
      S : CategoryTheory.Presieve X
      hS : Membership.mem (K.covering X) S
      ⊢ Membership.mem ((CategoryTheory.Coverage.ofGrothendieck C (CategoryTheory.Co …
    -/
    apply Saturate.of _ _ hS
    /-
      🎉 no goals
    -/


instance : SemilatticeSup (Coverage C) where
  sup x y :=
  { covering := fun B ↦ x.covering B ∪ y.covering B
    pullback := by
      /-
        C : Type ?u.40160
        D : Type ?u.40163
        inst✝¹ : CategoryTheory.Category.{?u.40167, ?u.40160} C
        inst✝ : CategoryTheory.Category.{?u.40171, ?u.40163} D
        x y : CategoryTheory.Coverage C
        ⊢ ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Presieve X), Membership …
      -/
      rintro X Y f S (hx | hy)
        /-
          case inl
          C : Type ?u.40160
          D : Type ?u.40163
          inst✝¹ : CategoryTheory.Category.{?u.40167, ?u.40160} C
          inst✝ : CategoryTheory.Category.{?u.40171, ?u.40163} D
          x y : CategoryTheory.Coverage C
          X Y : C
          f : Quiver.Hom Y X
          S : CategoryTheory.Presieve X
          hx : Membership.mem (x.covering X) S
          ⊢ Exists fun T => And (Membership.mem ((fun B => Union.union (x.covering B) (y …
        -/
      · obtain ⟨T, hT⟩ := x.pullback f S hx
        /-
          case inl.intro
          C : Type ?u.40160
          D : Type ?u.40163
          inst✝¹ : CategoryTheory.Category.{?u.40167, ?u.40160} C
          inst✝ : CategoryTheory.Category.{?u.40171, ?u.40163} D
          x y : CategoryTheory.Coverage C
          X Y : C
          f : Quiver.Hom Y X
          S : CategoryTheory.Presieve X
          hx : Membership.mem (x.covering X) S
          T : CategoryTheory.Presieve Y
          hT : And (Membership.mem (x.covering Y) T) (T.FactorsThruAlong S f)
          ⊢ Exists fun T => And (Membership.mem ((fun B => Union.union (x.covering B) (y …
        -/
        exact ⟨T, Or.inl hT.1, hT.2⟩
        /-
          🎉 no goals
        -/
        /-
          case inr
          C : Type ?u.40160
          D : Type ?u.40163
          inst✝¹ : CategoryTheory.Category.{?u.40167, ?u.40160} C
          inst✝ : CategoryTheory.Category.{?u.40171, ?u.40163} D
          x y : CategoryTheory.Coverage C
          X Y : C
          f : Quiver.Hom Y X
          S : CategoryTheory.Presieve X
          hy : Membership.mem (y.covering X) S
          ⊢ Exists fun T => And (Membership.mem ((fun B => Union.union (x.covering B) (y …
        -/
      · obtain ⟨T, hT⟩ := y.pullback f S hy
        /-
          case inr.intro
          C : Type ?u.40160
          D : Type ?u.40163
          inst✝¹ : CategoryTheory.Category.{?u.40167, ?u.40160} C
          inst✝ : CategoryTheory.Category.{?u.40171, ?u.40163} D
          x y : CategoryTheory.Coverage C
          X Y : C
          f : Quiver.Hom Y X
          S : CategoryTheory.Presieve X
          hy : Membership.mem (y.covering X) S
          T : CategoryTheory.Presieve Y
          hT : And (Membership.mem (y.covering Y) T) (T.FactorsThruAlong S f)
          ⊢ Exists fun T => And (Membership.mem ((fun B => Union.union (x.covering B) (y …
        -/
        exact ⟨T, Or.inr hT.1, hT.2⟩ }
        /-
          🎉 no goals
        -/
  toPartialOrder := inferInstance
  le_sup_left _ _ _ := Set.subset_union_left
  le_sup_right _ _ _ := Set.subset_union_right
  sup_le _ _ _ hx hy X := Set.union_subset_iff.mpr ⟨hx X, hy X⟩


@[simp]
lemma sup_covering (x y : Coverage C) (B : C) :
    (x ⊔ y).covering B = x.covering B ∪ y.covering B :=
  rfl


/--
Any sieve that contains a covering presieve for a coverage is a covering sieve for the associated
Grothendieck topology.
-/
theorem mem_toGrothendieck_sieves_of_superset (K : Coverage C) {X : C} {S : Sieve X}
    {R : Presieve X} (h : R ≤ S) (hR : R ∈ K.covering X) : S ∈ (K.toGrothendieck C) X :=
  K.saturate_of_superset ((Sieve.generate_le_iff _ _).mpr h) (Coverage.Saturate.of X _ hR)


/--
The main theorem of this file: Given a coverage `K` on `C`,
a `Type*`-valued presheaf on `C` is a sheaf for `K` if and only if it is a sheaf for
the associated Grothendieck topology.
-/
theorem isSheaf_coverage (K : Coverage C) (P : Cᵒᵖ ⥤ Type*) :
    Presieve.IsSheaf (toGrothendieck _ K) P ↔
    (∀ {X : C} (R : Presieve X), R ∈ K X → Presieve.IsSheafFor P R) := by
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} C
    K : CategoryTheory.Coverage C
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    ⊢ Iff (CategoryTheory.Presieve.IsSheaf (CategoryTheory.Coverage.toGrothendieck …
  -/
  constructor
    /-
      case mp
      C : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} C
      K : CategoryTheory.Coverage C
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.Coverage.toGrothendieck C K) …
    -/
  · intro H X R hR
    /-
      case mp
      C : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} C
      K : CategoryTheory.Coverage C
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      H : CategoryTheory.Presieve.IsSheaf (CategoryTheory.Coverage.toGrothendieck C  …
      X : C
      R : CategoryTheory.Presieve X
      hR : Membership.mem (K.covering X) R
      ⊢ CategoryTheory.Presieve.IsSheafFor P R
    -/
    rw [Presieve.isSheafFor_iff_generate]
    /-
      case mp
      C : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} C
      K : CategoryTheory.Coverage C
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      H : CategoryTheory.Presieve.IsSheaf (CategoryTheory.Coverage.toGrothendieck C  …
      X : C
      R : CategoryTheory.Presieve X
      hR : Membership.mem (K.covering X) R
      ⊢ CategoryTheory.Presieve.IsSheafFor P (CategoryTheory.Sieve.generate R).arrows
    -/
    apply H _ <| Saturate.of _ _ hR
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} C
      K : CategoryTheory.Coverage C
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      ⊢ (∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem (K.covering X) R  …
    -/
  · intro H X S hS
    -- This is the key point of the proof:
    -- We must generalize the induction in the correct way.
    suffices ∀ ⦃Y : C⦄ (f : Y ⟶ X), Presieve.IsSheafFor P (S.pullback f).arrows by
      simpa using this (f := 𝟙 _)
    induction hS with
    | of X S hS =>
      intro Y f
      obtain ⟨T, hT1, hT2⟩ := K.pullback f S hS
      apply Presieve.isSheafFor_of_factorsThru (S := T)
      · intro Z g hg
        obtain ⟨W, i, e, h1, h2⟩ := hT2 hg
        exact ⟨Z, 𝟙 _, g, ⟨W, i, e, h1, h2⟩, by simp⟩
      · apply H; assumption
      · intro Z g _
        obtain ⟨R, hR1, hR2⟩ := K.pullback g _ hT1
        exact ⟨R, (H _ hR1).isSeparatedFor, hR2⟩
    | top => intros; simpa using Presieve.isSheafFor_top_sieve _
    | transitive X R S _ _ H1 H2 =>
      intro Y f
      simp only [← Presieve.isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor] at *
      choose H1 H1' using H1
      choose H2 H2' using H2
      refine ⟨?_, fun x hx => ?_⟩
      · intro x t₁ t₂ h₁ h₂
        refine (H1 f).ext (fun Z g hg => ?_)
        refine (H2 hg (𝟙 _)).ext (fun ZZ gg hgg => ?_)
        simp only [Sieve.pullback_id, Sieve.pullback_apply] at hgg
        simp only [← types_comp_apply]
        rw [← P.map_comp, ← op_comp, h₁, h₂]
        simpa only [Sieve.pullback_apply, Category.assoc] using hgg
      let y : ∀ ⦃Z : C⦄ (g : Z ⟶ Y),
        ((S.pullback (g ≫ f)).pullback (𝟙 _)).arrows.FamilyOfElements P :=
        fun Z g ZZ gg hgg => x (gg ≫ g) (by simpa using hgg)
      have hy : ∀ ⦃Z : C⦄ (g : Z ⟶ Y), (y g).Compatible := by
        intro Z g Y₁ Y₂ ZZ g₁ g₂ f₁ f₂ h₁ h₂ h
        rw [hx]
        rw [reassoc_of% h]
      choose z hz using fun ⦃Z : C⦄ ⦃g : Z ⟶ Y⦄ (hg : R.pullback f g) =>
        H2' hg (𝟙 _) (y g) (hy g)
      let q : (R.pullback f).arrows.FamilyOfElements P := fun Z g hg => z hg
      have hq : q.Compatible := by
        intro Y₁ Y₂ Z g₁ g₂ f₁ f₂ h₁ h₂ h
        apply (H2 h₁ g₁).ext
        intro ZZ gg hgg
        simp only [← types_comp_apply]
        rw [← P.map_comp, ← P.map_comp, ← op_comp, ← op_comp, hz, hz]
        · dsimp [y]; congr 1; simp only [Category.assoc, h]
        · simpa [reassoc_of% h] using hgg
        · simpa using hgg
      obtain ⟨t, ht⟩ := H1' f q hq
      refine ⟨t, fun Z g hg => ?_⟩
      refine (H1 (g ≫ f)).ext (fun ZZ gg hgg => ?_)
      rw [← types_comp_apply _ (P.map gg.op), ← P.map_comp, ← op_comp, ht]
      on_goal 2 => simpa using hgg
      refine (H2 hgg (𝟙 _)).ext (fun ZZZ ggg hggg => ?_)
      rw [← types_comp_apply _ (P.map ggg.op), ← P.map_comp, ← op_comp, hz]
      on_goal 2 => simpa using hggg
      refine (H2 hgg ggg).ext (fun ZZZZ gggg _ => ?_)
      rw [← types_comp_apply _ (P.map gggg.op), ← P.map_comp, ← op_comp]
      apply hx
      simp


/--
A presheaf is a sheaf for the Grothendieck topology generated by a union of coverages iff it is a
sheaf for the Grothendieck topology generated by each coverage separately.
-/
theorem isSheaf_sup (K L : Coverage C) (P : Cᵒᵖ ⥤ Type*) :
    (Presieve.IsSheaf ((K ⊔ L).toGrothendieck C)) P ↔
    (Presieve.IsSheaf (K.toGrothendieck C)) P ∧ (Presieve.IsSheaf (L.toGrothendieck C)) P := by
  refine ⟨fun h ↦ ⟨Presieve.isSheaf_of_le _ ((gi C).gc.monotone_l le_sup_left) h,
      Presieve.isSheaf_of_le _ ((gi C).gc.monotone_l le_sup_right) h⟩, fun h ↦ ?_⟩
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} C
    K L : CategoryTheory.Coverage C
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : And (CategoryTheory.Presieve.IsSheaf (CategoryTheory.Coverage.toGrothendie …
    ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.Coverage.toGrothendieck C (M …
  -/
  rw [isSheaf_coverage, isSheaf_coverage] at h
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} C
    K L : CategoryTheory.Coverage C
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : And (∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem (K.covering …
    ⊢ CategoryTheory.Presieve.IsSheaf (CategoryTheory.Coverage.toGrothendieck C (M …
  -/
  rw [isSheaf_coverage]
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} C
    K L : CategoryTheory.Coverage C
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : And (∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem (K.covering …
    ⊢ ∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem ((Max.max K L).cov …
  -/
  intro X R hR
  /-
    C : Type u_2
    inst✝ : CategoryTheory.Category.{u_3, u_2} C
    K L : CategoryTheory.Coverage C
    P : CategoryTheory.Functor (Opposite C) (Type u_1)
    h : And (∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem (K.covering …
    X : C
    R : CategoryTheory.Presieve X
    hR : Membership.mem ((Max.max K L).covering X) R
    ⊢ CategoryTheory.Presieve.IsSheafFor P R
  -/
  cases' hR with hR hR
    /-
      case inl
      C : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} C
      K L : CategoryTheory.Coverage C
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      h : And (∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem (K.covering …
      X : C
      R : CategoryTheory.Presieve X
      hR : Membership.mem (K.covering X) R
      ⊢ CategoryTheory.Presieve.IsSheafFor P R
    -/
  · exact h.1 R hR
    /-
      🎉 no goals
    -/
    /-
      case inr
      C : Type u_2
      inst✝ : CategoryTheory.Category.{u_3, u_2} C
      K L : CategoryTheory.Coverage C
      P : CategoryTheory.Functor (Opposite C) (Type u_1)
      h : And (∀ {X : C} (R : CategoryTheory.Presieve X), Membership.mem (K.covering …
      X : C
      R : CategoryTheory.Presieve X
      hR : Membership.mem (L.covering X) R
      ⊢ CategoryTheory.Presieve.IsSheafFor P R
    -/
  · exact h.2 R hR
    /-
      🎉 no goals
    -/


theorem isSheaf_iff_isLimit_coverage (K : Coverage C) (P : Cᵒᵖ ⥤ D) :
    Presheaf.IsSheaf (toGrothendieck _ K) P ↔ ∀ ⦃X : C⦄ (R : Presieve X),
      R ∈ K.covering X →
        Nonempty (IsLimit (P.mapCone (Sieve.generate R).arrows.cocone.op)) := by
  simp only [Presheaf.IsSheaf, Presieve.isSheaf_coverage, isLimit_iff_isSheafFor,
    ← Presieve.isSheafFor_iff_generate]
  /-
    C : Type u_1
    D : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Category.{u_3, u_4} D
    K : CategoryTheory.Coverage C
    P : CategoryTheory.Functor (Opposite C) D
    ⊢ Iff (∀ (E : D) {X : C} (R : CategoryTheory.Presieve X), Membership.mem (K.co …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem isSheaf_sup (K L : Coverage C) (P : Cᵒᵖ ⥤ D) :
    (IsSheaf ((K ⊔ L).toGrothendieck C)) P ↔
    (IsSheaf (K.toGrothendieck C)) P ∧ (IsSheaf (L.toGrothendieck C)) P :=
  ⟨fun h ↦ ⟨fun E ↦ ((Presieve.isSheaf_sup K L _).mp (h E)).1, fun E ↦
    ((Presieve.isSheaf_sup K L _).mp (h E)).2⟩,
      fun ⟨h₁, h₂⟩ E ↦ (Presieve.isSheaf_sup K L _).mpr ⟨h₁ E, h₂ E⟩⟩


