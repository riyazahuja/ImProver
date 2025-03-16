/-- A split equalizer diagram consists of morphisms

```
      ι   f
    W → X ⇉ Y
          g
```

satisfying `ι ≫ f = ι ≫ g` together with morphisms

```
      r   t
    W ← X ← Y
```

satisfying `ι ≫ r = 𝟙 W`, `g ≫ t = 𝟙 X` and `f ≫ t = r ≫ ι`.

The name "equalizer" is appropriate, since any split equalizer is a equalizer, see
`CategoryTheory.IsSplitEqualizer.isEqualizer`.
Split equalizers are also absolute, since a functor preserves all the structure above.
-/
structure IsSplitEqualizer {W : C} (ι : W ⟶ X) where
  /-- A map from `X` to the equalizer -/
  leftRetraction : X ⟶ W
  /-- A map in the opposite direction to `f` and `g` -/
  rightRetraction : Y ⟶ X
  /-- Composition of `ι` with `f` and with `g` agree -/
  condition : ι ≫ f = ι ≫ g := by aesop_cat
  /-- `leftRetraction` splits `ι` -/
  ι_leftRetraction : ι ≫ leftRetraction = 𝟙 W := by aesop_cat
  /-- `rightRetraction` splits `g` -/
  bottom_rightRetraction : g ≫ rightRetraction = 𝟙 X := by aesop_cat
  /-- `f` composed with `rightRetraction` is `leftRetraction` composed with `ι` -/
  top_rightRetraction : f ≫ rightRetraction = leftRetraction ≫ ι := by aesop_cat


instance {X : C} : Inhabited (IsSplitEqualizer (𝟙 X) (𝟙 X) (𝟙 X)) where
  default := { leftRetraction := 𝟙 X, rightRetraction := 𝟙 X }


attribute [reassoc] condition


attribute [reassoc (attr := simp)] ι_leftRetraction bottom_rightRetraction top_rightRetraction


/-- Split equalizers are absolute: they are preserved by any functor. -/
@[simps]
def IsSplitEqualizer.map {W : C} {ι : W ⟶ X} (q : IsSplitEqualizer f g ι) (F : C ⥤ D) :
    IsSplitEqualizer (F.map f) (F.map g) (F.map ι) where
  leftRetraction := F.map q.leftRetraction
  rightRetraction := F.map q.rightRetraction
                  /-
                    C : Type u
                    inst✝¹ : CategoryTheory.Category.{v, u} C
                    D : Type u₂
                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                    G : CategoryTheory.Functor C D
                    X Y : C
                    f g : Quiver.Hom X Y
                    W : C
                    ι : Quiver.Hom W X
                    q : CategoryTheory.IsSplitEqualizer f g ι
                    F : CategoryTheory.Functor C D
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ι) (F.map f)) (CategoryTheory. …
                  -/
  condition := by rw [← F.map_comp, q.condition, F.map_comp]
                  /-
                    🎉 no goals
                  -/
                         /-
                           C : Type u
                           inst✝¹ : CategoryTheory.Category.{v, u} C
                           D : Type u₂
                           inst✝ : CategoryTheory.Category.{v₂, u₂} D
                           G : CategoryTheory.Functor C D
                           X Y : C
                           f g : Quiver.Hom X Y
                           W : C
                           ι : Quiver.Hom W X
                           q : CategoryTheory.IsSplitEqualizer f g ι
                           F : CategoryTheory.Functor C D
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ι) (F.map q.leftRetraction)) ( …
                         -/
  ι_leftRetraction := by rw [← F.map_comp, q.ι_leftRetraction, F.map_id]
                         /-
                           🎉 no goals
                         -/
                               /-
                                 C : Type u
                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                 D : Type u₂
                                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                 G : CategoryTheory.Functor C D
                                 X Y : C
                                 f g : Quiver.Hom X Y
                                 W : C
                                 ι : Quiver.Hom W X
                                 q : CategoryTheory.IsSplitEqualizer f g ι
                                 F : CategoryTheory.Functor C D
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map g) (F.map q.rightRetraction))  …
                               -/
  bottom_rightRetraction := by rw [← F.map_comp, q.bottom_rightRetraction, F.map_id]
                               /-
                                 🎉 no goals
                               -/
                            /-
                              C : Type u
                              inst✝¹ : CategoryTheory.Category.{v, u} C
                              D : Type u₂
                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                              G : CategoryTheory.Functor C D
                              X Y : C
                              f g : Quiver.Hom X Y
                              W : C
                              ι : Quiver.Hom W X
                              q : CategoryTheory.IsSplitEqualizer f g ι
                              F : CategoryTheory.Functor C D
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (F.map q.rightRetraction))  …
                            -/
  top_rightRetraction := by rw [← F.map_comp, q.top_rightRetraction, F.map_comp]
                            /-
                              🎉 no goals
                            -/


/-- A split equalizer clearly induces a fork. -/
@[simps! pt]
def IsSplitEqualizer.asFork {W : C} {h : W ⟶ X} (t : IsSplitEqualizer f g h) :
    Fork f g := Fork.ofι h t.condition


@[simp]
theorem IsSplitEqualizer.asFork_ι {W : C} {h : W ⟶ X} (t : IsSplitEqualizer f g h) :
    t.asFork.ι = h := rfl


/--
The fork induced by a split equalizer is an equalizer, justifying the name. In some cases it
is more convenient to show a given fork is an equalizer by showing it is split.
-/
def IsSplitEqualizer.isEqualizer {W : C} {h : W ⟶ X} (t : IsSplitEqualizer f g h) :
    IsLimit t.asFork :=
  Fork.IsLimit.mk' _ fun s =>
    ⟨ s.ι ≫ t.leftRetraction,
         /-
           C : Type u
           inst✝¹ : CategoryTheory.Category.{v, u} C
           D : Type u₂
           inst✝ : CategoryTheory.Category.{v₂, u₂} D
           G : CategoryTheory.Functor C D
           X Y : C
           f g : Quiver.Hom X Y
           W : C
           h : Quiver.Hom W X
           t : CategoryTheory.IsSplitEqualizer f g h
           s : CategoryTheory.Limits.Fork f g
           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
         -/
      by simp [- top_rightRetraction, ← t.top_rightRetraction, s.condition_assoc],
         /-
           🎉 no goals
         -/
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     G : CategoryTheory.Functor C D
                     X Y : C
                     f g : Quiver.Hom X Y
                     W : C
                     h : Quiver.Hom W X
                     t : CategoryTheory.IsSplitEqualizer f g h
                     s : CategoryTheory.Limits.Fork f g
                     m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
                     hm : Eq (CategoryTheory.CategoryStruct.comp m✝ t.asFork.ι) s.ι
                     ⊢ Eq m✝ (CategoryTheory.CategoryStruct.comp s.ι t.leftRetraction)
                   -/
      fun hm => by simp [← hm] ⟩
                   /-
                     🎉 no goals
                   -/


/--
The pair `f,g` is a cosplit pair if there is an `h : W ⟶ X` so that `f, g, h` forms a split
equalizer in `C`.
-/
class HasSplitEqualizer : Prop where
  /-- There is some split equalizer -/
  splittable : ∃ (W : C) (h : W ⟶ X), Nonempty (IsSplitEqualizer f g h)


/--
The pair `f,g` is a `G`-cosplit pair if there is an `h : W ⟶ G X` so that `G f, G g, h` forms a
split equalizer in `D`.
-/
abbrev Functor.IsCosplitPair : Prop :=
  HasSplitEqualizer (G.map f) (G.map g)


/-- Get the equalizer object from the typeclass `IsCosplitPair`. -/
noncomputable def HasSplitEqualizer.equalizerOfSplit [HasSplitEqualizer f g] : C :=
  (splittable (f := f) (g := g)).choose


/-- Get the equalizer morphism from the typeclass `IsCosplitPair`. -/
noncomputable def HasSplitEqualizer.equalizerι [HasSplitEqualizer f g] :
    HasSplitEqualizer.equalizerOfSplit f g ⟶ X :=
  (splittable (f := f) (g := g)).choose_spec.choose


/-- The equalizer morphism `equalizerι` gives a split equalizer on `f,g`. -/
noncomputable def HasSplitEqualizer.isSplitEqualizer [HasSplitEqualizer f g] :
    IsSplitEqualizer f g (HasSplitEqualizer.equalizerι f g) :=
  Classical.choice (splittable (f := f) (g := g)).choose_spec.choose_spec


/-- If `f, g` is cosplit, then `G f, G g` is cosplit. -/
instance map_is_cosplit_pair [HasSplitEqualizer f g] : HasSplitEqualizer (G.map f) (G.map g) where
  splittable :=
    ⟨_, _, ⟨IsSplitEqualizer.map (HasSplitEqualizer.isSplitEqualizer f g) _⟩⟩


/-- If a pair has a split equalizer, it has a equalizer. -/
instance (priority := 1) hasEqualizer_of_hasSplitEqualizer [HasSplitEqualizer f g] :
    HasEqualizer f g :=
  HasLimit.mk ⟨_, (HasSplitEqualizer.isSplitEqualizer f g).isEqualizer⟩


