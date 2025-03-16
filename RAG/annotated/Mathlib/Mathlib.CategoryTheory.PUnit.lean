/-- The constant functor sending everything to `PUnit.star`. -/
@[simps!]
def star : C ⥤ Discrete PUnit.{w + 1} :=
  (Functor.const _).obj ⟨⟨⟩⟩

/-- Any two functors to `Discrete PUnit` are isomorphic. -/
@[simps!]
def punitExt (F G : C ⥤ Discrete PUnit.{w + 1}) : F ≅ G :=
                                           /-
                                             C : Type u
                                             inst✝ : CategoryTheory.Category.{v, u} C
                                             F G : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{w + 1})
                                             X : C
                                             ⊢ Eq (F.obj X) (G.obj X)
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  NatIso.ofComponents fun X => eqToIso (by simp only [eq_iff_true_of_subsingleton])
  /-
    🎉 no goals
  -/
-- Porting note: simp does indeed fire for these despite the linter warning

/-- Any two functors to `Discrete PUnit` are *equal*.
You probably want to use `punitExt` instead of this. -/
theorem punit_ext' (F G : C ⥤ Discrete PUnit.{w + 1}) : F = G :=
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            F G : CategoryTheory.Functor C (CategoryTheory.Discrete PUnit.{w + 1})
                            X : C
                            ⊢ Eq (F.obj X) (G.obj X)
                          -/
                          /-
                            🎉 no goals
                          -/
  Functor.ext fun X => by simp only [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


/-- The functor from `Discrete PUnit` sending everything to the given object. -/
abbrev fromPUnit (X : C) : Discrete PUnit.{w + 1} ⥤ C :=
  (Functor.const _).obj X


/-- Functors from `Discrete PUnit` are equivalent to the category itself. -/
@[simps]
def equiv : Discrete PUnit.{w + 1} ⥤ C ≌ C where
  functor :=
    { obj := fun F => F.obj ⟨⟨⟩⟩
      map := fun θ => θ.app ⟨⟨⟩⟩ }
  inverse := Functor.const _
             /-
               C : Type u
               inst✝ : CategoryTheory.Category.{v, u} C
               ⊢ ∀ {X Y : CategoryTheory.Functor (CategoryTheory.Discrete PUnit.{w + 1}) C} ( …
             -/
  unitIso := NatIso.ofComponents fun _ => Discrete.natIso fun _ => Iso.refl _
             /-
               🎉 no goals
             -/
               /-
                 C : Type u
                 inst✝ : CategoryTheory.Category.{v, u} C
                 ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
               -/
  counitIso := NatIso.ofComponents Iso.refl
               /-
                 🎉 no goals
               -/


/-- A category being equivalent to `PUnit` is equivalent to it having a unique morphism between
  any two objects. (In fact, such a category is also a groupoid;
  see `CategoryTheory.Groupoid.ofHomUnique`) -/
theorem equiv_punit_iff_unique :
    Nonempty (C ≌ Discrete PUnit.{w + 1}) ↔ Nonempty C ∧ ∀ x y : C, Nonempty <| Unique (x ⟶ y) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    ⊢ Iff (Nonempty (CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{ …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ Nonempty (CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1 …
    -/
  · rintro ⟨h⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      ⊢ And (Nonempty C) (∀ (x y : C), Nonempty (Unique (Quiver.Hom x y)))
    -/
    refine ⟨⟨h.inverse.obj ⟨⟨⟩⟩⟩, fun x y => Nonempty.intro ?_⟩
    let f : x ⟶ y := by
      have hx : x ⟶ h.inverse.obj ⟨⟨⟩⟩ := by convert h.unit.app x
      have hy : h.inverse.obj ⟨⟨⟩⟩ ⟶ y := by convert h.unitInv.app y
      exact hx ≫ hy
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      x y : C
      f : Quiver.Hom x y := letFun (⋯.mpr (h.unit.app x)) fun hx => letFun (⋯.mpr (h …
      ⊢ Unique (Quiver.Hom x y)
    -/
    suffices sub : Subsingleton (x ⟶ y) from uniqueOfSubsingleton f
    have : ∀ z, z = h.unit.app x ≫ (h.functor ⋙ h.inverse).map z ≫ h.unitInv.app y := by
      intro z
      simp [congrArg (· ≫ h.unitInv.app y) (h.unit.naturality z)]
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      x y : C
      f : Quiver.Hom x y := letFun (⋯.mpr (h.unit.app x)) fun hx => letFun (⋯.mpr (h …
      this : ∀ (z : Quiver.Hom x y), Eq z (CategoryTheory.CategoryStruct.comp (h.uni …
      ⊢ Subsingleton (Quiver.Hom x y)
    -/
    apply Subsingleton.intro
    /-
      case mp.intro.allEq
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      x y : C
      f : Quiver.Hom x y := letFun (⋯.mpr (h.unit.app x)) fun hx => letFun (⋯.mpr (h …
      this : ∀ (z : Quiver.Hom x y), Eq z (CategoryTheory.CategoryStruct.comp (h.uni …
      ⊢ ∀ (a b : Quiver.Hom x y), Eq a b
    -/
    intro a b
    /-
      case mp.intro.allEq
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      x y : C
      f : Quiver.Hom x y := letFun (⋯.mpr (h.unit.app x)) fun hx => letFun (⋯.mpr (h …
      this : ∀ (z : Quiver.Hom x y), Eq z (CategoryTheory.CategoryStruct.comp (h.uni …
      a b : Quiver.Hom x y
      ⊢ Eq a b
    -/
    rw [this a, this b]
    /-
      case mp.intro.allEq
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      x y : C
      f : Quiver.Hom x y := letFun (⋯.mpr (h.unit.app x)) fun hx => letFun (⋯.mpr (h …
      this : ∀ (z : Quiver.Hom x y), Eq z (CategoryTheory.CategoryStruct.comp (h.uni …
      a b : Quiver.Hom x y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app x) (CategoryTheory.Catego …
    -/
    simp only [Functor.comp_map]
    /-
      case mp.intro.allEq
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      x y : C
      f : Quiver.Hom x y := letFun (⋯.mpr (h.unit.app x)) fun hx => letFun (⋯.mpr (h …
      this : ∀ (z : Quiver.Hom x y), Eq z (CategoryTheory.CategoryStruct.comp (h.uni …
      a b : Quiver.Hom x y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.unit.app x) (CategoryTheory.Catego …
    -/
    congr 3
    /-
      case mp.intro.allEq.e_a.e_a.e_a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      x y : C
      f : Quiver.Hom x y := letFun (⋯.mpr (h.unit.app x)) fun hx => letFun (⋯.mpr (h …
      this : ∀ (z : Quiver.Hom x y), Eq z (CategoryTheory.CategoryStruct.comp (h.uni …
      a b : Quiver.Hom x y
      ⊢ Eq (h.functor.map a) (h.functor.map b)
    -/
    apply ULift.ext
    /-
      case mp.intro.allEq.e_a.e_a.e_a.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1})
      x y : C
      f : Quiver.Hom x y := letFun (⋯.mpr (h.unit.app x)) fun hx => letFun (⋯.mpr (h …
      this : ∀ (z : Quiver.Hom x y), Eq z (CategoryTheory.CategoryStruct.comp (h.uni …
      a b : Quiver.Hom x y
      ⊢ Eq (h.functor.map a).down (h.functor.map b).down
    -/
    simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ And (Nonempty C) (∀ (x y : C), Nonempty (Unique (Quiver.Hom x y))) → Nonempt …
    -/
  · rintro ⟨⟨p⟩, h⟩
    /-
      case mpr.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      h : ∀ (x y : C), Nonempty (Unique (Quiver.Hom x y))
      p : C
      ⊢ Nonempty (CategoryTheory.Equivalence C (CategoryTheory.Discrete PUnit.{w + 1 …
    -/
    haveI := fun x y => (h x y).some
    refine
      Nonempty.intro
        (CategoryTheory.Equivalence.mk ((Functor.const _).obj ⟨⟨⟩⟩)
          ((@Functor.const <| Discrete PUnit).obj p) ?_ (by apply Functor.punitExt))
    exact
      NatIso.ofComponents fun _ =>
        { hom := default
          inv := default }


