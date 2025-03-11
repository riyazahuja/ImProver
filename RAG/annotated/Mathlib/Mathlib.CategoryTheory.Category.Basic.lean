/-- A preliminary structure on the way to defining a category,
containing the data, but none of the axioms. -/
@[pp_with_univ]
class CategoryStruct (obj : Type u) extends Quiver.{v + 1} obj : Type max u (v + 1) where
  /-- The identity morphism on an object. -/
  id : ∀ X : obj, Hom X X
  /-- Composition of morphisms in a category, written `f ≫ g`. -/
  comp : ∀ {X Y Z : obj}, (X ⟶ Y) → (Y ⟶ Z) → (X ⟶ Z)


/-- Notation for the identity morphism in a category. -/
scoped notation "𝟙" => CategoryStruct.id  -- type as \b1


/-- Notation for composition of morphisms in a category. -/
scoped infixr:80 " ≫ " => CategoryStruct.comp -- type as \gg


/-- Close the main goal with `sorry` if its type contains `sorry`, and fail otherwise. -/
syntax (name := sorryIfSorry) "sorry_if_sorry" : tactic


open Lean Meta Elab.Tactic in
@[tactic sorryIfSorry, inherit_doc sorryIfSorry] def evalSorryIfSorry : Tactic := fun _ => do
  let goalType ← getMainTarget
  if goalType.hasSorry then
    closeMainGoal `sorry_if_sorry (← mkSorry goalType true)
  else
    throwError "The goal does not contain `sorry`"


/--
A thin wrapper for `aesop` which adds the `CategoryTheory` rule set and
allows `aesop` to look through semireducible definitions when calling `intros`.
This tactic fails when it is unable to solve the goal, making it suitable for
use in auto-params.
-/
macro (name := aesop_cat) "aesop_cat" c:Aesop.tactic_clause* : tactic =>
`(tactic|
  first | sorry_if_sorry |
  aesop $c* (config := { introsTransparency? := some .default, terminal := true })
            (rule_sets := [$(Lean.mkIdent `CategoryTheory):ident]))


/--
We also use `aesop_cat?` to pass along a `Try this` suggestion when using `aesop_cat`
-/
macro (name := aesop_cat?) "aesop_cat?" c:Aesop.tactic_clause* : tactic =>
`(tactic|
  first | sorry_if_sorry |
  aesop? $c* (config := { introsTransparency? := some .default, terminal := true })
             (rule_sets := [$(Lean.mkIdent `CategoryTheory):ident]))

/--
A variant of `aesop_cat` which does not fail when it is unable to solve the
goal. Use this only for exploration! Nonterminal `aesop` is even worse than
nonterminal `simp`.
-/
macro (name := aesop_cat_nonterminal) "aesop_cat_nonterminal" c:Aesop.tactic_clause* : tactic =>
  `(tactic|
    aesop $c* (config := { introsTransparency? := some .default, warnOnNonterminal := false })
              (rule_sets := [$(Lean.mkIdent `CategoryTheory):ident]))


/-- The typeclass `Category C` describes morphisms associated to objects of type `C`.
The universe levels of the objects and morphisms are unconstrained, and will often need to be
specified explicitly, as `Category.{v} C`. (See also `LargeCategory` and `SmallCategory`.)

See <https://stacks.math.columbia.edu/tag/0014>.
-/
@[pp_with_univ]
class Category (obj : Type u) extends CategoryStruct.{v} obj : Type max u (v + 1) where
  /-- Identity morphisms are left identities for composition. -/
  id_comp : ∀ {X Y : obj} (f : X ⟶ Y), 𝟙 X ≫ f = f := by aesop_cat
  /-- Identity morphisms are right identities for composition. -/
  comp_id : ∀ {X Y : obj} (f : X ⟶ Y), f ≫ 𝟙 Y = f := by aesop_cat
  /-- Composition in a category is associative. -/
  assoc : ∀ {W X Y Z : obj} (f : W ⟶ X) (g : X ⟶ Y) (h : Y ⟶ Z), (f ≫ g) ≫ h = f ≫ g ≫ h := by
    aesop_cat


/-- A `LargeCategory` has objects in one universe level higher than the universe level of
the morphisms. It is useful for examples such as the category of types, or the category
of groups, etc.
-/
abbrev LargeCategory (C : Type (u + 1)) : Type (u + 1) := Category.{u} C


/-- A `SmallCategory` has objects and morphisms in the same universe level.
-/
abbrev SmallCategory (C : Type u) : Type (u + 1) := Category.{u} C


/-- postcompose an equation between morphisms by another morphism -/
                                                                               /-
                                                                                 C : Type u
                                                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                                                 X Y Z : C
                                                                                 f g : Quiver.Hom X Y
                                                                                 w : Eq f g
                                                                                 h : Quiver.Hom Y Z
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct.c …
                                                                               -/
theorem eq_whisker {f g : X ⟶ Y} (w : f = g) (h : Y ⟶ Z) : f ≫ h = g ≫ h := by rw [w]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- precompose an equation between morphisms by another morphism -/
                                                                               /-
                                                                                 C : Type u
                                                                                 inst✝ : CategoryTheory.Category.{v, u} C
                                                                                 X Y Z : C
                                                                                 f : Quiver.Hom X Y
                                                                                 g h : Quiver.Hom Y Z
                                                                                 w : Eq g h
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
                                                                               -/
theorem whisker_eq (f : X ⟶ Y) {g h : Y ⟶ Z} (w : g = h) : f ≫ g = f ≫ h := by rw [w]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/--
Notation for whiskering an equation by a morphism (on the right).
If `f g : X ⟶ Y` and `w : f = g` and `h : Y ⟶ Z`, then `w =≫ h : f ≫ h = g ≫ h`.
-/
scoped infixr:80 " =≫ " => eq_whisker


/--
Notation for whiskering an equation by a morphism (on the left).
If `g h : Y ⟶ Z` and `w : g = h` and `h : X ⟶ Y`, then `f ≫= w : f ≫ g = f ≫ h`.
-/
scoped infixr:80 " ≫= " => whisker_eq


theorem eq_of_comp_left_eq {f g : X ⟶ Y} (w : ∀ {Z : C} (h : Y ⟶ Z), f ≫ h = g ≫ h) :
    f = g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    w : ∀ {Z : C} (h : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp f h …
    ⊢ Eq f g
  -/
                      /-
                        🎉 no goals
                      -/
  convert w (𝟙 Y) <;> simp
                      /-
                        🎉 no goals
                      -/


theorem eq_of_comp_right_eq {f g : Y ⟶ Z} (w : ∀ {X : C} (h : X ⟶ Y), h ≫ f = h ≫ g) :
    f = g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    Y Z : C
    f g : Quiver.Hom Y Z
    w : ∀ {X : C} (h : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp h f …
    ⊢ Eq f g
  -/
                      /-
                        🎉 no goals
                      -/
  convert w (𝟙 Y) <;> simp
                      /-
                        🎉 no goals
                      -/


theorem eq_of_comp_left_eq' (f g : X ⟶ Y)
    (w : (fun {Z} (h : Y ⟶ Z) => f ≫ h) = fun {Z} (h : Y ⟶ Z) => g ≫ h) : f = g :=
                                    /-
                                      C : Type u
                                      inst✝ : CategoryTheory.Category.{v, u} C
                                      X Y : C
                                      f g : Quiver.Hom X Y
                                      w : Eq (fun {Z} h => CategoryTheory.CategoryStruct.comp f h) fun {Z} h => Cate …
                                      Z : C
                                      h : Quiver.Hom Y Z
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStruct.c …
                                    -/
  eq_of_comp_left_eq @fun Z h => by convert congr_fun (congr_fun w Z) h
                                    /-
                                      🎉 no goals
                                    -/


theorem eq_of_comp_right_eq' (f g : Y ⟶ Z)
    (w : (fun {X} (h : X ⟶ Y) => h ≫ f) = fun {X} (h : X ⟶ Y) => h ≫ g) : f = g :=
                                     /-
                                       C : Type u
                                       inst✝ : CategoryTheory.Category.{v, u} C
                                       Y Z : C
                                       f g : Quiver.Hom Y Z
                                       w : Eq (fun {X} h => CategoryTheory.CategoryStruct.comp h f) fun {X} h => Cate …
                                       X : C
                                       h : Quiver.Hom X Y
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct.c …
                                     -/
  eq_of_comp_right_eq @fun X h => by convert congr_fun (congr_fun w X) h
                                     /-
                                       🎉 no goals
                                     -/


theorem id_of_comp_left_id (f : X ⟶ X) (w : ∀ {Y : C} (g : X ⟶ Y), f ≫ g = g) : f = 𝟙 X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    f : Quiver.Hom X X
    w : ∀ {Y : C} (g : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp f g …
    ⊢ Eq f (CategoryTheory.CategoryStruct.id X)
  -/
  convert w (𝟙 X)
  /-
    case h.e'_2
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    f : Quiver.Hom X X
    w : ∀ {Y : C} (g : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp f g …
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem id_of_comp_right_id (f : X ⟶ X) (w : ∀ {Y : C} (g : Y ⟶ X), g ≫ f = g) : f = 𝟙 X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    f : Quiver.Hom X X
    w : ∀ {Y : C} (g : Quiver.Hom Y X), Eq (CategoryTheory.CategoryStruct.comp g f …
    ⊢ Eq f (CategoryTheory.CategoryStruct.id X)
  -/
  convert w (𝟙 X)
  /-
    case h.e'_2
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    f : Quiver.Hom X X
    w : ∀ {Y : C} (g : Quiver.Hom Y X), Eq (CategoryTheory.CategoryStruct.comp g f …
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem comp_ite {P : Prop} [Decidable P] {X Y Z : C} (f : X ⟶ Y) (g g' : Y ⟶ Z) :
                                                                  /-
                                                                    C : Type u
                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                    P : Prop
                                                                    inst✝ : Decidable P
                                                                    X Y Z : C
                                                                    f : Quiver.Hom X Y
                                                                    g g' : Quiver.Hom Y Z
                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (ite P g g')) (ite P (CategoryTheor …
                                                                  -/
    (f ≫ if P then g else g') = if P then f ≫ g else f ≫ g' := by aesop
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem ite_comp {P : Prop} [Decidable P] {X Y Z : C} (f f' : X ⟶ Y) (g : Y ⟶ Z) :
                                                                  /-
                                                                    C : Type u
                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                    P : Prop
                                                                    inst✝ : Decidable P
                                                                    X Y Z : C
                                                                    f f' : Quiver.Hom X Y
                                                                    g : Quiver.Hom Y Z
                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ite P f f') g) (ite P (CategoryTheor …
                                                                  -/
    (if P then f else f') ≫ g = if P then f ≫ g else f' ≫ g := by aesop
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem comp_dite {P : Prop} [Decidable P]
    {X Y Z : C} (f : X ⟶ Y) (g : P → (Y ⟶ Z)) (g' : ¬P → (Y ⟶ Z)) :
                                                                                  /-
                                                                                    C : Type u
                                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                    P : Prop
                                                                                    inst✝ : Decidable P
                                                                                    X Y Z : C
                                                                                    f : Quiver.Hom X Y
                                                                                    g : P → Quiver.Hom Y Z
                                                                                    g' : Not P → Quiver.Hom Y Z
                                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (dite P (fun h => g h) fun h => g'  …
                                                                                  -/
    (f ≫ if h : P then g h else g' h) = if h : P then f ≫ g h else f ≫ g' h := by aesop
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem dite_comp {P : Prop} [Decidable P]
    {X Y Z : C} (f : P → (X ⟶ Y)) (f' : ¬P → (X ⟶ Y)) (g : Y ⟶ Z) :
                                                                                  /-
                                                                                    C : Type u
                                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                    P : Prop
                                                                                    inst✝ : Decidable P
                                                                                    X Y Z : C
                                                                                    f : P → Quiver.Hom X Y
                                                                                    f' : Not P → Quiver.Hom X Y
                                                                                    g : Quiver.Hom Y Z
                                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite P (fun h => f h) fun h => f' h) …
                                                                                  -/
    (if h : P then f h else f' h) ≫ g = if h : P then f h ≫ g else f' h ≫ g := by aesop
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- A morphism `f` is an epimorphism if it can be cancelled when precomposed:
`f ≫ g = f ≫ h` implies `g = h`.

See <https://stacks.math.columbia.edu/tag/003B>.
-/
class Epi (f : X ⟶ Y) : Prop where
  /-- A morphism `f` is an epimorphism if it can be cancelled when precomposed. -/
  left_cancellation : ∀ {Z : C} (g h : Y ⟶ Z), f ≫ g = f ≫ h → g = h


/-- A morphism `f` is a monomorphism if it can be cancelled when postcomposed:
`g ≫ f = h ≫ f` implies `g = h`.

See <https://stacks.math.columbia.edu/tag/003B>.
-/
class Mono (f : X ⟶ Y) : Prop where
  /-- A morphism `f` is a monomorphism if it can be cancelled when postcomposed. -/
  right_cancellation : ∀ {Z : C} (g h : Z ⟶ X), g ≫ f = h ≫ f → g = h


instance (X : C) : Epi (𝟙 X) :=
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     X✝ Y Z X Z✝ : C
                     g h : Quiver.Hom X Z✝
                     w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X …
                     ⊢ Eq g h
                   -/
  ⟨fun g h w => by aesop⟩
                   /-
                     🎉 no goals
                   -/


instance (X : C) : Mono (𝟙 X) :=
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     X✝ Y Z X Z✝ : C
                     g h : Quiver.Hom Z✝ X
                     w : Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.id …
                     ⊢ Eq g h
                   -/
  ⟨fun g h w => by aesop⟩
                   /-
                     🎉 no goals
                   -/


theorem cancel_epi (f : X ⟶ Y) [Epi f] {g h : Y ⟶ Z} : f ≫ g = f ≫ h ↔ g = h :=
  ⟨fun p => Epi.left_cancellation g h p, congr_arg _⟩


theorem cancel_epi_assoc_iff (f : X ⟶ Y) [Epi f] {g h : Y ⟶ Z} {W : C} {k l : Z ⟶ W} :
    (f ≫ g) ≫ k = (f ≫ h) ≫ l ↔ g ≫ k = h ≫ l :=
                                   /-
                                     C : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                     X Y Z : C
                                     f : Quiver.Hom X Y
                                     inst✝ : CategoryTheory.Epi f
                                     g h : Quiver.Hom Y Z
                                     W : C
                                     k l : Quiver.Hom Z W
                                     p : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
  ⟨fun p => (cancel_epi f).1 <| by simpa using p, fun p => by simp only [Category.assoc, p]⟩
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem cancel_mono (f : X ⟶ Y) [Mono f] {g h : Z ⟶ X} : g ≫ f = h ≫ f ↔ g = h :=
  -- Porting note: in Lean 3 we could just write `congr_arg _` here.
  ⟨fun p => Mono.right_cancellation g h p, congr_arg (fun k => k ≫ f)⟩


theorem cancel_mono_assoc_iff (f : X ⟶ Y) [Mono f] {g h : Z ⟶ X} {W : C} {k l : W ⟶ Z} :
    k ≫ (g ≫ f) = l ≫ (h ≫ f) ↔ k ≫ g = l ≫ h :=
                                    /-
                                      C : Type u
                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                      X Y Z : C
                                      f : Quiver.Hom X Y
                                      inst✝ : CategoryTheory.Mono f
                                      g h : Quiver.Hom Z X
                                      W : C
                                      k l : Quiver.Hom W Z
                                      p : Eq (CategoryTheory.CategoryStruct.comp k (CategoryTheory.CategoryStruct.co …
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp k …
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  ⟨fun p => (cancel_mono f).1 <| by simpa using p, fun p => by simp only [← Category.assoc, p]⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem cancel_epi_id (f : X ⟶ Y) [Epi f] {h : Y ⟶ Y} : f ≫ h = f ↔ h = 𝟙 Y := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    h : Quiver.Hom Y Y
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f h) f) (Eq h (CategoryTheory.Ca …
  -/
  convert cancel_epi f
  /-
    case h.e'_1.h.e'_3.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    h : Quiver.Hom Y Y
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem cancel_mono_id (f : X ⟶ Y) [Mono f] {g : X ⟶ X} : g ≫ f = f ↔ g = 𝟙 X := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    g : Quiver.Hom X X
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp g f) f) (Eq g (CategoryTheory.Ca …
  -/
  convert cancel_mono f
  /-
    case h.e'_1.h.e'_3.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    g : Quiver.Hom X X
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X …
  -/
  simp
  /-
    🎉 no goals
  -/


instance epi_comp {X Y Z : C} (f : X ⟶ Y) [Epi f] (g : Y ⟶ Z) [Epi g] : Epi (f ≫ g) :=
  ⟨fun _ _ w => (cancel_epi g).1 <| (cancel_epi_assoc_iff f).1 w⟩


instance mono_comp {X Y Z : C} (f : X ⟶ Y) [Mono f] (g : Y ⟶ Z) [Mono g] : Mono (f ≫ g) :=
  ⟨fun _ _ w => (cancel_mono f).1 <| (cancel_mono_assoc_iff g).1 w⟩


theorem mono_of_mono {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [Mono (f ≫ g)] : Mono f :=
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                X Y Z : C
                                                f : Quiver.Hom X Y
                                                g : Quiver.Hom Y Z
                                                inst✝ : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp f g)
                                                Z✝ : C
                                                x✝¹ x✝ : Quiver.Hom Z✝ X
                                                w : Eq (CategoryTheory.CategoryStruct.comp x✝¹ f) (CategoryTheory.CategoryStru …
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp x✝¹ (CategoryTheory.CategoryStruct.co …
                                              -/
  ⟨fun _ _ w => (cancel_mono (f ≫ g)).1 <| by simp only [← Category.assoc, w]⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem mono_of_mono_fac {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z} {h : X ⟶ Z} [Mono h]
    (w : f ≫ g = h) : Mono f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom X Z
    inst✝ : CategoryTheory.Mono h
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    ⊢ CategoryTheory.Mono f
  -/
  subst h; exact mono_of_mono f g
           /-
             🎉 no goals
           -/


theorem epi_of_epi {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [Epi (f ≫ g)] : Epi g :=
                                             /-
                                               C : Type u
                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                               X Y Z : C
                                               f : Quiver.Hom X Y
                                               g : Quiver.Hom Y Z
                                               inst✝ : CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp f g)
                                               Z✝ : C
                                               x✝¹ x✝ : Quiver.Hom Z Z✝
                                               w : Eq (CategoryTheory.CategoryStruct.comp g x✝¹) (CategoryTheory.CategoryStru …
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                             -/
  ⟨fun _ _ w => (cancel_epi (f ≫ g)).1 <| by simp only [Category.assoc, w]⟩
                                             /-
                                               🎉 no goals
                                             -/


theorem epi_of_epi_fac {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z} {h : X ⟶ Z} [Epi h]
    (w : f ≫ g = h) : Epi g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom X Z
    inst✝ : CategoryTheory.Epi h
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    ⊢ CategoryTheory.Epi g
  -/
  subst h; exact epi_of_epi f g
           /-
             🎉 no goals
           -/


instance : Mono f where
  right_cancellation _ _ _ := Subsingleton.elim _ _


instance : Epi f where
  left_cancellation _ _ _ := Subsingleton.elim _ _


instance uliftCategory : Category.{v} (ULift.{u'} C) where
  Hom X Y := X.down ⟶ Y.down
  id X := 𝟙 X.down
  comp f g := f ≫ g

-- We verify that this previous instance can lift small categories to large categories.

