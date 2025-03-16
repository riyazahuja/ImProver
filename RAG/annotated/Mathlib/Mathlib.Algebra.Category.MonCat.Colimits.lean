/-- An inductive type representing all monoid expressions (without relations)
on a collection of types indexed by the objects of `J`.
-/
inductive Prequotient
  -- There's always `of`
  | of : ∀ (j : J) (_ : F.obj j), Prequotient
  -- Then one generator for each operation
  | one : Prequotient
  | mul : Prequotient → Prequotient → Prequotient


instance : Inhabited (Prequotient F) :=
  ⟨Prequotient.one⟩


/-- The relation on `Prequotient` saying when two expressions are equal
because of the monoid laws, or
because one element is mapped to another by a morphism in the diagram.
-/
inductive Relation : Prequotient F → Prequotient F → Prop-- Make it an equivalence relation:
  | refl : ∀ x, Relation x x
  | symm : ∀ (x y) (_ : Relation x y), Relation y x
  | trans : ∀ (x y z) (_ : Relation x y) (_ : Relation y z),
      Relation x z-- There's always a `map` relation
  | map :
    ∀ (j j' : J) (f : j ⟶ j') (x : F.obj j),
      Relation (Prequotient.of j' ((F.map f) x))
        (Prequotient.of j x)-- Then one relation per operation, describing the interaction with `of`
  | mul : ∀ (j) (x y : F.obj j), Relation (Prequotient.of j (x * y))
      (mul (Prequotient.of j x) (Prequotient.of j y))
  | one : ∀ j, Relation (Prequotient.of j 1) one-- Then one relation per argument of each operation
  | mul_1 : ∀ (x x' y) (_ : Relation x x'), Relation (mul x y) (mul x' y)
  | mul_2 : ∀ (x y y') (_ : Relation y y'), Relation (mul x y) (mul x y')
    -- And one relation per axiom
  | mul_assoc : ∀ x y z, Relation (mul (mul x y) z) (mul x (mul y z))
  | one_mul : ∀ x, Relation (mul one x) x
  | mul_one : ∀ x, Relation (mul x one) x


/-- The setoid corresponding to monoid expressions modulo monoid relations and identifications.
-/
def colimitSetoid : Setoid (Prequotient F) where
  r := Relation F
  iseqv := ⟨Relation.refl, Relation.symm _ _, Relation.trans _ _ _⟩


/-- The underlying type of the colimit of a diagram in `MonCat`.
-/
def ColimitType : Type v :=
  Quotient (colimitSetoid F)


instance : Inhabited (ColimitType F) := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{u, v} J
    F : CategoryTheory.Functor J MonCat
    ⊢ Inhabited (MonCat.Colimits.ColimitType F)
  -/
  dsimp [ColimitType]
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{u, v} J
    F : CategoryTheory.Functor J MonCat
    ⊢ Inhabited (Quotient (MonCat.Colimits.colimitSetoid F))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance monoidColimitType : Monoid (ColimitType F) where
  one := Quotient.mk _ one
  mul := Quotient.map₂ mul fun _ x' rx y _ ry =>
    Setoid.trans (Relation.mul_1 _ _ y rx) (Relation.mul_2 x' _ _ ry)
  one_mul := Quotient.ind fun _ => Quotient.sound <| Relation.one_mul _
  mul_one := Quotient.ind fun _ => Quotient.sound <| Relation.mul_one _
  mul_assoc := Quotient.ind fun _ => Quotient.ind₂ fun _ _ =>
    Quotient.sound <| Relation.mul_assoc _ _ _


@[simp]
theorem quot_one : Quot.mk Setoid.r one = (1 : ColimitType F) :=
  rfl


@[simp]
theorem quot_mul (x y : Prequotient F) : Quot.mk Setoid.r (mul x y) =
    @HMul.hMul (ColimitType F) (ColimitType F) (ColimitType F) _
      (Quot.mk Setoid.r x) (Quot.mk Setoid.r y) :=
  rfl


/-- The bundled monoid giving the colimit of a diagram. -/
def colimit : MonCat :=
                     /-
                       J : Type v
                       inst✝ : CategoryTheory.Category.{u, v} J
                       F : CategoryTheory.Functor J MonCat
                       ⊢ Monoid (MonCat.Colimits.ColimitType F)
                     -/
  ⟨ColimitType F, by infer_instance⟩
                     /-
                       🎉 no goals
                     -/


/-- The function from a given monoid in the diagram to the colimit monoid. -/
def coconeFun (j : J) (x : F.obj j) : ColimitType F :=
  Quot.mk _ (Prequotient.of j x)


/-- The monoid homomorphism from a given monoid in the diagram to the colimit monoid. -/
def coconeMorphism (j : J) : F.obj j ⟶ colimit F where
  toFun := coconeFun F j
  map_one' := Quot.sound (Relation.one _)
  map_mul' _ _ := Quot.sound (Relation.mul _ _ _)


@[simp]
theorem cocone_naturality {j j' : J} (f : j ⟶ j') :
    F.map f ≫ coconeMorphism F j' = coconeMorphism F j := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{u, v} J
    F : CategoryTheory.Functor J MonCat
    j j' : J
    f : Quiver.Hom j j'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (MonCat.Colimits.coconeMorp …
  -/
  ext
  /-
    case w
    J : Type v
    inst✝ : CategoryTheory.Category.{u, v} J
    F : CategoryTheory.Functor J MonCat
    j j' : J
    f : Quiver.Hom j j'
    x✝ : ↑(F.obj j)
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (F.map f) (MonCat.Colimits.coconeMor …
  -/
  apply Quot.sound
  /-
    case w.a
    J : Type v
    inst✝ : CategoryTheory.Category.{u, v} J
    F : CategoryTheory.Functor J MonCat
    j j' : J
    f : Quiver.Hom j j'
    x✝ : ↑(F.obj j)
    ⊢ (MonCat.Colimits.colimitSetoid F) (MonCat.Colimits.Prequotient.of j' ((F.map …
  -/
  apply Relation.map
  /-
    🎉 no goals
  -/


@[simp]
theorem cocone_naturality_components (j j' : J) (f : j ⟶ j') (x : F.obj j) :
    (coconeMorphism F j') (F.map f x) = (coconeMorphism F j) x := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{u, v} J
    F : CategoryTheory.Functor J MonCat
    j j' : J
    f : Quiver.Hom j j'
    x : ↑(F.obj j)
    ⊢ Eq ((MonCat.Colimits.coconeMorphism F j') ((F.map f) x)) ((MonCat.Colimits.c …
  -/
  rw [← cocone_naturality F f]
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{u, v} J
    F : CategoryTheory.Functor J MonCat
    j j' : J
    f : Quiver.Hom j j'
    x : ↑(F.obj j)
    ⊢ Eq ((MonCat.Colimits.coconeMorphism F j') ((F.map f) x)) ((CategoryTheory.Ca …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The cocone over the proposed colimit monoid. -/
def colimitCocone : Cocone F where
  pt := colimit F
  ι := { app := coconeMorphism F }


/-- The function from the free monoid on the diagram to the cone point of any other cocone. -/
@[simp]
def descFunLift (s : Cocone F) : Prequotient F → s.pt
  | Prequotient.of j x => (s.ι.app j) x
  | one => 1
  | mul x y => descFunLift _ x * descFunLift _ y


/-- The function from the colimit monoid to the cone point of any other cocone. -/
def descFun (s : Cocone F) : ColimitType F → s.pt := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{u, v} J
    F : CategoryTheory.Functor J MonCat
    s : CategoryTheory.Limits.Cocone F
    ⊢ MonCat.Colimits.ColimitType F → ↑s.pt
  -/
  fapply Quot.lift
    /-
      case f
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      ⊢ MonCat.Colimits.Prequotient F → ↑s.pt
    -/
  · exact descFunLift F s
    /-
      🎉 no goals
    -/
    /-
      case a
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      ⊢ ∀ (a b : MonCat.Colimits.Prequotient F), (MonCat.Colimits.colimitSetoid F) a …
    -/
  · intro x y r
    induction r with
    | refl x => rfl
    | symm x y _ h => exact h.symm
    | trans x y z _ _ h₁ h₂ => exact h₁.trans h₂
    | map j j' f x => exact s.w_apply f x
    | mul j x y => exact map_mul (s.ι.app j) x y
    | one j => exact map_one (s.ι.app j)
    | mul_1 x x' y _ h => exact congr_arg (· * _) h
    | mul_2 x y y' _ h => exact congr_arg (_ * ·) h
    | mul_assoc x y z => exact mul_assoc _ _ _
    | one_mul x => exact one_mul _
    | mul_one x => exact mul_one _


/-- The monoid homomorphism from the colimit monoid to the cone point of any other cocone. -/
def descMorphism (s : Cocone F) : colimit F ⟶ s.pt where
  toFun := descFun F s
  map_one' := rfl
  map_mul' x y := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      x y : ↑(MonCat.Colimits.colimit F)
      ⊢ Eq ({ toFun := MonCat.Colimits.descFun F s, map_one' := ⋯ }.toFun (HMul.hMul …
    -/
    induction x using Quot.inductionOn
    /-
      case h
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      y : ↑(MonCat.Colimits.colimit F)
      a✝ : MonCat.Colimits.Prequotient F
      ⊢ Eq ({ toFun := MonCat.Colimits.descFun F s, map_one' := ⋯ }.toFun (HMul.hMul …
    -/
    induction y using Quot.inductionOn
    /-
      case h.h
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      a✝¹ a✝ : MonCat.Colimits.Prequotient F
      ⊢ Eq ({ toFun := MonCat.Colimits.descFun F s, map_one' := ⋯ }.toFun (HMul.hMul …
    -/
    dsimp [descFun]
    /-
      case h.h
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      a✝¹ a✝ : MonCat.Colimits.Prequotient F
      ⊢ Eq (Quot.lift (MonCat.Colimits.descFunLift F s) ⋯ (HMul.hMul (Quot.mk (⇑(Mon …
    -/
    rw [← quot_mul]
    /-
      case h.h
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      a✝¹ a✝ : MonCat.Colimits.Prequotient F
      ⊢ Eq (Quot.lift (MonCat.Colimits.descFunLift F s) ⋯ (Quot.mk (⇑(MonCat.Colimit …
    -/
    simp only [descFunLift]
    /-
      🎉 no goals
    -/


/-- Evidence that the proposed colimit is the colimit. -/
def colimitIsColimit : IsColimit (colimitCocone F) where
  desc s := descMorphism F s
  uniq s m w := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (MonCat.Colimits.colimitCocone F).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((MonCat.Colimits.colimi …
      ⊢ Eq m ((fun s => MonCat.Colimits.descMorphism F s) s)
    -/
    ext x
    /-
      case w
      J : Type v
      inst✝ : CategoryTheory.Category.{u, v} J
      F : CategoryTheory.Functor J MonCat
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (MonCat.Colimits.colimitCocone F).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((MonCat.Colimits.colimi …
      x : ↑(MonCat.Colimits.colimitCocone F).pt
      ⊢ Eq (m x) (((fun s => MonCat.Colimits.descMorphism F s) s) x)
    -/
    induction x using Quot.inductionOn with | h x => ?_
    induction x with
    | of j =>
      change _ = s.ι.app j _
      rw [← w j]
      rfl
    | one =>
      rw [quot_one, map_one]
      rfl
    | mul x y hx hy =>
      rw [quot_mul, map_mul, hx, hy]
      dsimp [descMorphism, DFunLike.coe, descFun]
      simp only [← quot_mul, descFunLift]


instance hasColimits_monCat : HasColimits MonCat where
  has_colimits_of_shape _ _ :=
    { has_colimit := fun F =>
        HasColimit.mk
          { cocone := colimitCocone F
            isColimit := colimitIsColimit F } }


