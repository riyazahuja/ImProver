/-- A type synonym for `Type u`, which carries the category instance for which
    morphisms are binary relations. -/
def RelCat :=
  Type u


                                                   /-
                                                     ⊢ Inhabited CategoryTheory.RelCat
                                                   -/
instance RelCat.inhabited : Inhabited RelCat := by unfold RelCat; infer_instance
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- The category of types with binary relations as morphisms. -/
instance rel : LargeCategory RelCat where
  Hom X Y := X → Y → Prop
  id _ x y := x = y
  comp f g x z := ∃ y, f x y ∧ g y z




@[ext] theorem hom_ext {X Y : RelCat} (f g : X ⟶ Y) (h : ∀ a b, f a b ↔ g a b) : f = g :=
  funext₂ (fun a b => propext (h a b))


protected theorem rel_id (X : RelCat) : 𝟙 X = (· = ·) := rfl


protected theorem rel_comp {X Y Z : RelCat} (f : X ⟶ Y) (g : Y ⟶ Z) : f ≫ g = Rel.comp f g := rfl


theorem rel_id_apply₂ (X : RelCat) (x y : X) : (𝟙 X) x y ↔ x = y := by
  /-
    X : CategoryTheory.RelCat
    x y : X
    ⊢ Iff (CategoryTheory.CategoryStruct.id X x y) (Eq x y)
  -/
  rw [RelCat.Hom.rel_id]
  /-
    🎉 no goals
  -/


theorem rel_comp_apply₂ {X Y Z : RelCat} (f : X ⟶ Y) (g : Y ⟶ Z) (x : X) (z : Z) :
                                           /-
                                             X Y Z : CategoryTheory.RelCat
                                             f : Quiver.Hom X Y
                                             g : Quiver.Hom Y Z
                                             x : X
                                             z : Z
                                             ⊢ Iff (CategoryTheory.CategoryStruct.comp f g x z) (Exists fun y => And (f x y …
                                           -/
    (f ≫ g) x z ↔ ∃ y, f x y ∧ g y z := by rfl
                                           /-
                                             🎉 no goals
                                           -/


/-- The essentially surjective faithful embedding
from the category of types and functions into the category of types and relations. -/
def graphFunctor : Type u ⥤ RelCat.{u} where
  obj X := X
  map f := f.graph
  map_id X := by
    /-
      X : Type u
      ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => Function.graph f }.map (Categ …
    -/
    ext
    /-
      case h
      X : Type u
      a✝ b✝ : { obj := fun X => X, map := fun {X Y} f => Function.graph f }.obj X
      ⊢ Iff ({ obj := fun X => X, map := fun {X Y} f => Function.graph f }.map (Cate …
    -/
    simp [Hom.rel_id_apply₂]
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      X✝ Y✝ Z✝ : Type u
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => X, map := fun {X Y} f => Function.graph f }.map (Categ …
    -/
    ext
    /-
      case h
      X✝ Y✝ Z✝ : Type u
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      a✝ : { obj := fun X => X, map := fun {X Y} f => Function.graph f }.obj X✝
      b✝ : { obj := fun X => X, map := fun {X Y} f => Function.graph f }.obj Z✝
      ⊢ Iff ({ obj := fun X => X, map := fun {X Y} f => Function.graph f }.map (Cate …
    -/
    simp [Hom.rel_comp_apply₂]
    /-
      🎉 no goals
    -/


@[simp] theorem graphFunctor_map {X Y : Type u} (f : X ⟶ Y) (x : X) (y : Y) :
    graphFunctor.map f x y ↔ f x = y := f.graph_def x y


instance graphFunctor_faithful : graphFunctor.Faithful where
  map_injective h := Function.graph_injective h


instance graphFunctor_essSurj : graphFunctor.EssSurj :=
    graphFunctor.essSurj_of_surj Function.surjective_id


/-- A relation is an isomorphism in `RelCat` iff it is the image of an isomorphism in
`Type u`. -/
theorem rel_iso_iff {X Y : RelCat} (r : X ⟶ Y) :
    IsIso (C := RelCat) r ↔ ∃ f : (Iso (C := Type u) X Y), graphFunctor.map f.hom = r := by
  /-
    X Y : CategoryTheory.RelCat
    r : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.IsIso r) (Exists fun f => Eq (CategoryTheory.RelCat.grap …
  -/
  constructor
    /-
      case mp
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      ⊢ CategoryTheory.IsIso r → Exists fun f => Eq (CategoryTheory.RelCat.graphFunc …
    -/
  · intro h
    /-
      case mp
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      h : CategoryTheory.IsIso r
      ⊢ Exists fun f => Eq (CategoryTheory.RelCat.graphFunctor.map f.hom) r
    -/
    have h1 := congr_fun₂ h.hom_inv_id
    /-
      case mp
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      h : CategoryTheory.IsIso r
      h1 : ∀ (a b : X), Eq (CategoryTheory.CategoryStruct.comp r (CategoryTheory.inv …
      ⊢ Exists fun f => Eq (CategoryTheory.RelCat.graphFunctor.map f.hom) r
    -/
    have h2 := congr_fun₂ h.inv_hom_id
    /-
      case mp
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      h : CategoryTheory.IsIso r
      h1 : ∀ (a b : X), Eq (CategoryTheory.CategoryStruct.comp r (CategoryTheory.inv …
      h2 : ∀ (a b : Y), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv r …
      ⊢ Exists fun f => Eq (CategoryTheory.RelCat.graphFunctor.map f.hom) r
    -/
    simp only [RelCat.Hom.rel_comp_apply₂, RelCat.Hom.rel_id_apply₂, eq_iff_iff] at h1 h2
    /-
      case mp
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      h : CategoryTheory.IsIso r
      h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
      h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
      ⊢ Exists fun f => Eq (CategoryTheory.RelCat.graphFunctor.map f.hom) r
    -/
    obtain ⟨f, hf⟩ := Classical.axiomOfChoice (fun a => (h1 a a).mpr rfl)
    /-
      case mp.intro
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      h : CategoryTheory.IsIso r
      h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
      h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
      f : X → Y
      hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
      ⊢ Exists fun f => Eq (CategoryTheory.RelCat.graphFunctor.map f.hom) r
    -/
    obtain ⟨g, hg⟩ := Classical.axiomOfChoice (fun a => (h2 a a).mpr rfl)
    suffices hif : IsIso (C := Type u) f by
      use asIso f
      ext x y
      simp only [asIso_hom, graphFunctor_map]
      constructor
      · rintro rfl
        exact (hf x).1
      · intro hr
        specialize h2 (f x) y
        rw [← h2]
        use x, (hf x).2, hr
    /-
      case mp.intro.intro
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      h : CategoryTheory.IsIso r
      h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
      h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
      f : X → Y
      hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
      g : Y → X
      hg : ∀ (x : Y), And (CategoryTheory.inv r x (g x)) (r (g x) x)
      ⊢ CategoryTheory.IsIso f
    -/
    use g
    /-
      case h
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      h : CategoryTheory.IsIso r
      h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
      h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
      f : X → Y
      hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
      g : Y → X
      hg : ∀ (x : Y), And (CategoryTheory.inv r x (g x)) (r (g x) x)
      ⊢ And (Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStr …
    -/
    constructor
      /-
        case h.left
        X Y : CategoryTheory.RelCat
        r : Quiver.Hom X Y
        h : CategoryTheory.IsIso r
        h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
        h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
        f : X → Y
        hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
        g : Y → X
        hg : ∀ (x : Y), And (CategoryTheory.inv r x (g x)) (r (g x) x)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.i …
      -/
    · ext x
      /-
        case h.left.h
        X Y : CategoryTheory.RelCat
        r : Quiver.Hom X Y
        h : CategoryTheory.IsIso r
        h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
        h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
        f : X → Y
        hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
        g : Y → X
        hg : ∀ (x : Y), And (CategoryTheory.inv r x (g x)) (r (g x) x)
        x : X
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f g x) (CategoryTheory.CategoryStruct …
      -/
      apply (h1 _ _).mp
      /-
        case h.left.h
        X Y : CategoryTheory.RelCat
        r : Quiver.Hom X Y
        h : CategoryTheory.IsIso r
        h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
        h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
        f : X → Y
        hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
        g : Y → X
        hg : ∀ (x : Y), And (CategoryTheory.inv r x (g x)) (r (g x) x)
        x : X
        ⊢ Exists fun y => And (r (CategoryTheory.CategoryStruct.comp f g x) y) (Catego …
      -/
      use f x, (hg _).2, (hf _).2
      /-
        🎉 no goals
      -/
      /-
        case h.right
        X Y : CategoryTheory.RelCat
        r : Quiver.Hom X Y
        h : CategoryTheory.IsIso r
        h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
        h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
        f : X → Y
        hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
        g : Y → X
        hg : ∀ (x : Y), And (CategoryTheory.inv r x (g x)) (r (g x) x)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruct.i …
      -/
    · ext y
      /-
        case h.right.h
        X Y : CategoryTheory.RelCat
        r : Quiver.Hom X Y
        h : CategoryTheory.IsIso r
        h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
        h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
        f : X → Y
        hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
        g : Y → X
        hg : ∀ (x : Y), And (CategoryTheory.inv r x (g x)) (r (g x) x)
        y : Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g f y) (CategoryTheory.CategoryStruct …
      -/
      apply (h2 _ _).mp
      /-
        case h.right.h
        X Y : CategoryTheory.RelCat
        r : Quiver.Hom X Y
        h : CategoryTheory.IsIso r
        h1 : ∀ (a b : X), Iff (Exists fun y => And (r a y) (CategoryTheory.inv r y b)) …
        h2 : ∀ (a b : Y), Iff (Exists fun y => And (CategoryTheory.inv r a y) (r y b)) …
        f : X → Y
        hf : ∀ (x : X), And (r x (f x)) (CategoryTheory.inv r (f x) x)
        g : Y → X
        hg : ∀ (x : Y), And (CategoryTheory.inv r x (g x)) (r (g x) x)
        y : Y
        ⊢ Exists fun y_1 => And (CategoryTheory.inv r (CategoryTheory.CategoryStruct.c …
      -/
      use g y, (hf (g y)).2, (hg y).2
      /-
        🎉 no goals
      -/
    /-
      case mpr
      X Y : CategoryTheory.RelCat
      r : Quiver.Hom X Y
      ⊢ (Exists fun f => Eq (CategoryTheory.RelCat.graphFunctor.map f.hom) r) → Cate …
    -/
  · rintro ⟨f, rfl⟩
    /-
      case mpr.intro
      X Y : CategoryTheory.RelCat
      f : CategoryTheory.Iso X Y
      ⊢ CategoryTheory.IsIso (CategoryTheory.RelCat.graphFunctor.map f.hom)
    -/
    apply graphFunctor.map_isIso
    /-
      🎉 no goals
    -/


/-- The argument-swap isomorphism from `RelCat` to its opposite. -/
def opFunctor : RelCat ⥤ RelCatᵒᵖ where
  obj X := op X
  map {_ _} r := op (fun y x => r x y)
  map_id X := by
    /-
      X : CategoryTheory.RelCat
      ⊢ Eq ({ obj := fun X => { unop := X }, map := fun {x x_1} r => { unop := fun y …
    -/
    congr
    /-
      case e_a
      X : CategoryTheory.RelCat
      ⊢ Eq (CategoryTheory.CategoryStruct.id X) fun x y => CategoryTheory.CategorySt …
    -/
    simp only [unop_op, RelCat.Hom.rel_id]
    /-
      case e_a
      X : CategoryTheory.RelCat
      ⊢ Eq (fun x1 x2 => Eq x1 x2) fun x y => Eq y x
    -/
    ext x y
    /-
      case e_a.h
      X : CategoryTheory.RelCat
      x y : X
      ⊢ Iff (Eq x y) (Eq y x)
    -/
    exact Eq.comm
    /-
      🎉 no goals
    -/
  map_comp {X Y Z} f g := by
    /-
      X Y Z : CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => { unop := X }, map := fun {x x_1} r => { unop := fun y …
    -/
    unfold Category.opposite
    /-
      X Y Z : CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => { unop := X }, map := fun {x x_1} r => { unop := fun y …
    -/
    congr
    /-
      case e_a
      X Y Z : CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) fun x y => CategoryTheory.Catego …
    -/
    ext x y
    /-
      case e_a.h
      X Y Z : CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      x : X
      y : Z
      ⊢ Iff (CategoryTheory.CategoryStruct.comp f g x y) (CategoryTheory.CategoryStr …
    -/
    apply exists_congr
    /-
      case e_a.h.h
      X Y Z : CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      x : X
      y : Z
      ⊢ ∀ (a : Y), Iff (And (f x a) (g a y)) (And (({ obj := fun X => { unop := X }, …
    -/
    exact fun a => And.comm
    /-
      🎉 no goals
    -/


/-- The other direction of `opFunctor`. -/
def unopFunctor : RelCatᵒᵖ ⥤ RelCat where
  obj X := unop X
  map {_ _} r x y := unop r y x
  map_id X := by
    /-
      X : Opposite CategoryTheory.RelCat
      ⊢ Eq ({ obj := fun X => Opposite.unop X, map := fun {x x_1} r x_2 y => Opposit …
    -/
    dsimp
    /-
      X : Opposite CategoryTheory.RelCat
      ⊢ Eq (fun x y => Opposite.unop (CategoryTheory.CategoryStruct.id X) y x) (Cate …
    -/
    ext x y
    /-
      case h
      X : Opposite CategoryTheory.RelCat
      x y : Opposite.unop X
      ⊢ Iff (Opposite.unop (CategoryTheory.CategoryStruct.id X) y x) (CategoryTheory …
    -/
    exact Eq.comm
    /-
      🎉 no goals
    -/
  map_comp {X Y Z} f g := by
    /-
      X Y Z : Opposite CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => Opposite.unop X, map := fun {x x_1} r x_2 y => Opposit …
    -/
    unfold Category.opposite
    /-
      X Y Z : Opposite CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => Opposite.unop X, map := fun {x x_1} r x_2 y => Opposit …
    -/
    ext x y
    /-
      case h
      X Y Z : Opposite CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      x : { obj := fun X => Opposite.unop X, map := fun {x x_1} r x_2 y => Opposite. …
      y : { obj := fun X => Opposite.unop X, map := fun {x x_1} r x_2 y => Opposite. …
      ⊢ Iff ({ obj := fun X => Opposite.unop X, map := fun {x x_1} r x_2 y => Opposi …
    -/
    apply exists_congr
    /-
      case h.h
      X Y Z : Opposite CategoryTheory.RelCat
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      x : { obj := fun X => Opposite.unop X, map := fun {x x_1} r x_2 y => Opposite. …
      y : { obj := fun X => Opposite.unop X, map := fun {x x_1} r x_2 y => Opposite. …
      ⊢ ∀ (a : Opposite.unop Y), Iff (And (g.unop y a) (f.unop a x)) (And ({ obj :=  …
    -/
    exact fun a => And.comm
    /-
      🎉 no goals
    -/


@[simp] theorem opFunctor_comp_unopFunctor_eq :
    Functor.comp opFunctor unopFunctor = Functor.id _ := rfl


@[simp] theorem unopFunctor_comp_opFunctor_eq :
    Functor.comp unopFunctor opFunctor = Functor.id _ := rfl


/-- `RelCat` is self-dual: The map that swaps the argument order of a
    relation induces an equivalence between `RelCat` and its opposite. -/
@[simps]
def opEquivalence : Equivalence RelCat RelCatᵒᵖ where
  functor := opFunctor
  inverse := unopFunctor
  unitIso := Iso.refl _
  counitIso := Iso.refl _


instance : opFunctor.IsEquivalence := by
  /-
    ⊢ CategoryTheory.RelCat.opFunctor.IsEquivalence
  -/
  change opEquivalence.functor.IsEquivalence
  /-
    ⊢ CategoryTheory.RelCat.opEquivalence.functor.IsEquivalence
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : unopFunctor.IsEquivalence := by
  /-
    ⊢ CategoryTheory.RelCat.unopFunctor.IsEquivalence
  -/
  change opEquivalence.inverse.IsEquivalence
  /-
    ⊢ CategoryTheory.RelCat.opEquivalence.inverse.IsEquivalence
  -/
  infer_instance
  /-
    🎉 no goals
  -/


