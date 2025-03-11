/-- For a pair of functors `L : C ⥤ D`, `R : D ⥤ C`, an adjunction `h : L ⊣ R` induces a monad on
the category `C`.
-/
-- Porting note: Specifying simps projections manually to match mathlib3 behavior.
@[simps! coe η μ]
def toMonad (h : L ⊣ R) : Monad C where
  toFunctor := L ⋙ R
  η := h.unit
  μ := whiskerRight (whiskerLeft L h.counit) R
  assoc X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.comp R).map ((CategoryTheory.whis …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map (L.map (R.map (h.counit.app (L …
    -/
    rw [← R.map_comp]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : C
      ⊢ Eq (R.map (CategoryTheory.CategoryStruct.comp (L.map (R.map (h.counit.app (L …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_unit X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((L.comp R).map (h.unit.app X)) ((Cat …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map (L.map (h.unit.app X))) (R.map …
    -/
    rw [← R.map_comp]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : C
      ⊢ Eq (R.map (CategoryTheory.CategoryStruct.comp (L.map (h.unit.app X)) (h.coun …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- For a pair of functors `L : C ⥤ D`, `R : D ⥤ C`, an adjunction `h : L ⊣ R` induces a comonad on
the category `D`.
-/
-- Porting note: Specifying simps projections manually to match mathlib3 behavior.
@[simps coe ε δ]
def toComonad (h : L ⊣ R) : Comonad D where
  toFunctor := R ⋙ L
  ε := h.counit
  δ := whiskerRight (whiskerLeft R h.unit) L
  coassoc X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.whiskerRight (Catego …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (h.unit.app (R.obj X))) (L.map …
    -/
    rw [← L.map_comp]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : D
      ⊢ Eq (L.map (CategoryTheory.CategoryStruct.comp (h.unit.app (R.obj X)) (R.map  …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_counit X := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.whiskerRight (Catego …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (h.unit.app (R.obj X))) (L.map …
    -/
    rw [← L.map_comp]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      X : D
      ⊢ Eq (L.map (CategoryTheory.CategoryStruct.comp (h.unit.app (R.obj X)) (R.map  …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The monad induced by the Eilenberg-Moore adjunction is the original monad. -/
@[simps!]
def adjToMonadIso (T : Monad C) : T.adj.toMonad ≅ T :=
               /-
                 C : Type u₁
                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝ : CategoryTheory.Category.{v₂, u₂} D
                 L : CategoryTheory.Functor C D
                 R : CategoryTheory.Functor D C
                 T : CategoryTheory.Monad C
                 ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (T. …
               -/
               /-
                 🎉 no goals
               -/
  /-
    🎉 no goals
  -/
  MonadIso.mk (NatIso.ofComponents fun _ => Iso.refl _)
  /-
    🎉 no goals
  -/


/-- The comonad induced by the Eilenberg-Moore adjunction is the original comonad. -/
@[simps!]
def adjToComonadIso (G : Comonad C) : G.adj.toComonad ≅ G :=
                 /-
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝ : CategoryTheory.Category.{v₂, u₂} D
                   L : CategoryTheory.Functor C D
                   R : CategoryTheory.Functor D C
                   G : CategoryTheory.Comonad C
                   ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (G. …
                 -/
                 /-
                   🎉 no goals
                 -/
  /-
    🎉 no goals
  -/
  ComonadIso.mk (NatIso.ofComponents fun _ => Iso.refl _)
  /-
    🎉 no goals
  -/


/--
Given an adjunction `L ⊣ R`, if `L ⋙ R` is abstractly isomorphic to the identity functor, then the
unit is an isomorphism.
-/
def unitAsIsoOfIso (adj : L ⊣ R) (i : L ⋙ R ≅ 𝟭 C) : 𝟭 C ≅ L ⋙ R where
  hom := adj.unit
  inv :=  i.hom ≫ (adj.toMonad.transport i).μ
  hom_inv_id := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      i : CategoryTheory.Iso (L.comp R) (CategoryTheory.Functor.id C)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp adj.unit (CategoryTheory.CategoryStru …
    -/
    rw [← assoc]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      i : CategoryTheory.Iso (L.comp R) (CategoryTheory.Functor.id C)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp a …
    -/
    ext X
    /-
      case w.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      i : CategoryTheory.Iso (L.comp R) (CategoryTheory.Functor.id C)
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    exact (adj.toMonad.transport i).right_unit X
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    rw [assoc, ← Iso.eq_inv_comp, comp_id, ← id_comp i.inv, Iso.eq_comp_inv, assoc,
      NatTrans.id_comm]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      i : CategoryTheory.Iso (L.comp R) (CategoryTheory.Functor.id C)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp a …
    -/
    ext X
    /-
      case w.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      i : CategoryTheory.Iso (L.comp R) (CategoryTheory.Functor.id C)
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    exact (adj.toMonad.transport i).right_unit X
    /-
      🎉 no goals
    -/


lemma isIso_unit_of_iso (adj : L ⊣ R) (i : L ⋙ R ≅ 𝟭 C) : IsIso adj.unit :=
  (inferInstanceAs (IsIso (unitAsIsoOfIso adj i).hom))


/--
Given an adjunction `L ⊣ R`, if `L ⋙ R` is isomorphic to the identity functor, then `L` is
fully faithful.
-/
noncomputable def fullyFaithfulLOfCompIsoId (adj : L ⊣ R) (i : L ⋙ R ≅ 𝟭 C) : L.FullyFaithful :=
  haveI := adj.isIso_unit_of_iso i
  adj.fullyFaithfulLOfIsIsoUnit


/--
Given an adjunction `L ⊣ R`, if `R ⋙ L` is abstractly isomorphic to the identity functor, then the
counit is an isomorphism.
-/
def counitAsIsoOfIso (adj : L ⊣ R) (j : R ⋙ L ≅ 𝟭 D) : R ⋙ L ≅ 𝟭 D where
  hom := adj.counit
  inv := (adj.toComonad.transport j).δ ≫ j.inv
  hom_inv_id := by
    rw [← assoc, Iso.comp_inv_eq, id_comp, ← comp_id j.hom, ← Iso.inv_comp_eq, ← assoc,
      NatTrans.id_comm]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      j : CategoryTheory.Iso (R.comp L) (CategoryTheory.Functor.id D)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.toComonad.transport j).δ (Catego …
    -/
    ext X
    /-
      case w.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      j : CategoryTheory.Iso (R.comp L) (CategoryTheory.Functor.id D)
      X : D
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (adj.toComonad.transport j).δ (Categ …
    -/
    exact (adj.toComonad.transport j).right_counit X
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      j : CategoryTheory.Iso (R.comp L) (CategoryTheory.Functor.id D)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      j : CategoryTheory.Iso (R.comp L) (CategoryTheory.Functor.id D)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj.toComonad.transport j).δ (Catego …
    -/
    ext X
    /-
      case w.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      adj : CategoryTheory.Adjunction L R
      j : CategoryTheory.Iso (R.comp L) (CategoryTheory.Functor.id D)
      X : D
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (adj.toComonad.transport j).δ (Categ …
    -/
    exact (adj.toComonad.transport j).right_counit X
    /-
      🎉 no goals
    -/


lemma isIso_counit_of_iso (adj : L ⊣ R) (j : R ⋙ L ≅ 𝟭 D) : IsIso adj.counit :=
  inferInstanceAs (IsIso (counitAsIsoOfIso adj j).hom)


/--
Given an adjunction `L ⊣ R`, if `R ⋙ L` is isomorphic to the identity functor, then `R` is
fully faithful.
-/
noncomputable def fullyFaithfulROfCompIsoId (adj : L ⊣ R) (j : R ⋙ L ≅ 𝟭 D) : R.FullyFaithful :=
  haveI := adj.isIso_counit_of_iso j
  adj.fullyFaithfulROfIsIsoCounit


/-- Given any adjunction `L ⊣ R`, there is a comparison functor `CategoryTheory.Monad.comparison R`
sending objects `Y : D` to Eilenberg-Moore algebras for `L ⋙ R` with underlying object `R.obj X`.

We later show that this is full when `R` is full, faithful when `R` is faithful,
and essentially surjective when `R` is reflective.
-/
@[simps]
def Monad.comparison (h : L ⊣ R) : D ⥤ h.toMonad.Algebra where
  obj X :=
    { A := R.obj X
      a := R.map (h.counit.app X)
      assoc := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X : D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.toMonad.μ.app (R.obj X)) (R.map (h …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X : D
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map (h.counit.app (L.obj (R.obj X) …
        -/
        rw [← R.map_comp, ← Adjunction.counit_naturality, R.map_comp] }
        /-
          🎉 no goals
        -/
  map f :=
    { f := R.map f
      h := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X✝ Y✝ : D
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.toMonad.map (R.map f)) ((fun X =>  …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X✝ Y✝ : D
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map (L.map (R.map f))) (R.map (h.c …
        -/
        rw [← R.map_comp, Adjunction.counit_naturality, R.map_comp] }
        /-
          🎉 no goals
        -/


/-- The underlying object of `(Monad.comparison R).obj X` is just `R.obj X`.
-/
@[simps]
def Monad.comparisonForget (h : L ⊣ R) : Monad.comparison h ⋙ h.toMonad.forget ≅ R where
  hom := { app := fun _ => 𝟙 _ }
  inv := { app := fun _ => 𝟙 _ }


theorem Monad.left_comparison (h : L ⊣ R) : L ⋙ Monad.comparison h = h.toMonad.free :=
  rfl


instance [R.Faithful] (h : L ⊣ R) : (Monad.comparison h).Faithful where
  map_injective {_ _} _ _ w := R.map_injective (congr_arg Monad.Algebra.Hom.f w : _)


instance (T : Monad C) : (Monad.comparison T.adj).Full where
                                      /-
                                        C : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                        D : Type u₂
                                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                        L : CategoryTheory.Functor C D
                                        R : CategoryTheory.Functor D C
                                        T : CategoryTheory.Monad C
                                        x✝¹ x✝ : T.Algebra
                                        f : Quiver.Hom ((CategoryTheory.Monad.comparison T.adj).obj x✝¹) ((CategoryThe …
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map f.f) x✝.a) (CategoryTheory.Cat …
                                      -/
  map_surjective {_ _} f := ⟨⟨f.f, by simpa using f.h⟩, rfl⟩
                                      /-
                                        🎉 no goals
                                      -/


instance (T : Monad C) : (Monad.comparison T.adj).EssSurj where
  mem_essImage X :=
    ⟨{  A := X.A
        a := X.a
                   /-
                     C : Type u₁
                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                     D : Type u₂
                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                     L : CategoryTheory.Functor C D
                     R : CategoryTheory.Functor D C
                     T : CategoryTheory.Monad C
                     X : T.adj.toMonad.Algebra
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.η.app X.A) X.a) (CategoryTheory.Ca …
                   -/
        unit := by simpa using X.unit
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u₁
                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                      D : Type u₂
                      inst✝ : CategoryTheory.Category.{v₂, u₂} D
                      L : CategoryTheory.Functor C D
                      R : CategoryTheory.Functor D C
                      T : CategoryTheory.Monad C
                      X : T.adj.toMonad.Algebra
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.μ.app X.A) X.a) (CategoryTheory.Ca …
                    -/
        assoc := by simpa using X.assoc },
                    /-
                      🎉 no goals
                    -/
     /-
       C : Type u₁
       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
       D : Type u₂
       inst✝ : CategoryTheory.Category.{v₂, u₂} D
       L : CategoryTheory.Functor C D
       R : CategoryTheory.Functor D C
       T : CategoryTheory.Monad C
       X : T.adj.toMonad.Algebra
       ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.adj.toMonad.map (CategoryTheory.Is …
     -/
    ⟨Monad.Algebra.isoMk (Iso.refl _)⟩⟩
     /-
       🎉 no goals
     -/


/--
Given any adjunction `L ⊣ R`, there is a comparison functor `CategoryTheory.Comonad.comparison L`
sending objects `X : C` to Eilenberg-Moore coalgebras for `L ⋙ R` with underlying object
`L.obj X`.
-/
@[simps]
def Comonad.comparison (h : L ⊣ R) : C ⥤ h.toComonad.Coalgebra where
  obj X :=
    { A := L.obj X
      a := L.map (h.unit.app X)
      coassoc := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (h.unit.app X)) (h.toComonad.δ …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X : C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (h.unit.app X)) (L.map (h.unit …
        -/
        rw [← L.map_comp, ← Adjunction.unit_naturality, L.map_comp] }
        /-
          🎉 no goals
        -/
  map f :=
    { f := L.map f
      h := by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun X => { A := L.obj X, a := L.map …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map (h.unit.app X✝)) (L.map (R.map …
        -/
        rw [← L.map_comp]
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          h : CategoryTheory.Adjunction L R
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (L.map (CategoryTheory.CategoryStruct.comp (h.unit.app X✝) (R.map (L.map  …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- The underlying object of `(Comonad.comparison L).obj X` is just `L.obj X`.
-/
@[simps]
def Comonad.comparisonForget {L : C ⥤ D} {R : D ⥤ C} (h : L ⊣ R) :
    Comonad.comparison h ⋙ h.toComonad.forget ≅ L where
  hom := { app := fun _ => 𝟙 _ }
  inv := { app := fun _ => 𝟙 _ }


theorem Comonad.left_comparison (h : L ⊣ R) : R ⋙ Comonad.comparison h = h.toComonad.cofree :=
  rfl


instance Comonad.comparison_faithful_of_faithful [L.Faithful] (h : L ⊣ R) :
    (Comonad.comparison h).Faithful where
  map_injective {_ _} _ _ w := L.map_injective (congr_arg Comonad.Coalgebra.Hom.f w : _)


instance (G : Comonad C) : (Comonad.comparison G.adj).Full where
                                /-
                                  C : Type u₁
                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                  D : Type u₂
                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                  L : CategoryTheory.Functor C D
                                  R : CategoryTheory.Functor D C
                                  G : CategoryTheory.Comonad C
                                  X✝ Y✝ : G.Coalgebra
                                  f : Quiver.Hom ((CategoryTheory.Comonad.comparison G.adj).obj X✝) ((CategoryTh …
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp X✝.a (G.map f.f)) (CategoryTheory.Cat …
                                -/
  map_surjective f := ⟨⟨f.f, by simpa using f.h⟩, rfl⟩
                                /-
                                  🎉 no goals
                                -/


instance (G : Comonad C) : (Comonad.comparison G.adj).EssSurj where
  mem_essImage X :=
    ⟨{  A := X.A
        a := X.a
                     /-
                       C : Type u₁
                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                       D : Type u₂
                       inst✝ : CategoryTheory.Category.{v₂, u₂} D
                       L : CategoryTheory.Functor C D
                       R : CategoryTheory.Functor D C
                       G : CategoryTheory.Comonad C
                       X : G.adj.toComonad.Coalgebra
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp X.a (G.ε.app X.A)) (CategoryTheory.Ca …
                     -/
        counit := by simpa using X.counit
                     /-
                       🎉 no goals
                     -/
                      /-
                        C : Type u₁
                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                        D : Type u₂
                        inst✝ : CategoryTheory.Category.{v₂, u₂} D
                        L : CategoryTheory.Functor C D
                        R : CategoryTheory.Functor D C
                        G : CategoryTheory.Comonad C
                        X : G.adj.toComonad.Coalgebra
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp X.a (G.δ.app X.A)) (CategoryTheory.Ca …
                      -/
        coassoc := by simpa using X.coassoc },
                      /-
                        🎉 no goals
                      -/
       /-
         C : Type u₁
         inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
         D : Type u₂
         inst✝ : CategoryTheory.Category.{v₂, u₂} D
         L : CategoryTheory.Functor C D
         R : CategoryTheory.Functor D C
         G : CategoryTheory.Comonad C
         X : G.adj.toComonad.Coalgebra
         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comonad.comparison G …
       -/
      ⟨Comonad.Coalgebra.isoMk (Iso.refl _)⟩⟩
       /-
         🎉 no goals
       -/


/-- A right adjoint functor `R : D ⥤ C` is *monadic* if the comparison functor `Monad.comparison R`
from `D` to the category of Eilenberg-Moore algebras for the adjunction is an equivalence.
-/
class MonadicRightAdjoint (R : D ⥤ C) where
  /-- a choice of left adjoint for `R` -/
  L : C ⥤ D
  /-- `R` is a right adjoint -/
  adj : L ⊣ R
  eqv : (Monad.comparison adj).IsEquivalence


/-- The left adjoint functor to `R` given by `[MonadicRightAdjoint R]`. -/
def monadicLeftAdjoint (R : D ⥤ C) [MonadicRightAdjoint R] : C ⥤ D :=
  MonadicRightAdjoint.L (R := R)


/-- The adjunction `monadicLeftAdjoint R ⊣ R` given by `[MonadicRightAdjoint R]`. -/
def monadicAdjunction (R : D ⥤ C) [MonadicRightAdjoint R] :
    monadicLeftAdjoint R ⊣ R :=
  MonadicRightAdjoint.adj


instance (R : D ⥤ C) [MonadicRightAdjoint R] :
    (Monad.comparison (monadicAdjunction R)).IsEquivalence :=
  MonadicRightAdjoint.eqv


instance (R : D ⥤ C) [MonadicRightAdjoint R] : R.IsRightAdjoint :=
  (monadicAdjunction R).isRightAdjoint


noncomputable instance (T : Monad C) : MonadicRightAdjoint T.forget where
  adj := T.adj
  eqv := { }


/--
A left adjoint functor `L : C ⥤ D` is *comonadic* if the comparison functor `Comonad.comparison L`
from `C` to the category of Eilenberg-Moore algebras for the adjunction is an equivalence.
-/
class ComonadicLeftAdjoint (L : C ⥤ D) where
  /-- a choice of right adjoint for `L` -/
  R : D ⥤ C
  /-- `L` is a left adjoint -/
  adj : L ⊣ R
  eqv : (Comonad.comparison adj).IsEquivalence


/-- The right adjoint functor to `L` given by `[ComonadicLeftAdjoint L]`. -/
def comonadicRightAdjoint (L : C ⥤ D) [ComonadicLeftAdjoint L] : D ⥤ C :=
  ComonadicLeftAdjoint.R (L := L)


/-- The adjunction `L ⊣ comonadicRightAdjoint L` given by `[ComonadicLeftAdjoint L]`. -/
def comonadicAdjunction (L : C ⥤ D) [ComonadicLeftAdjoint L] :
    L ⊣ comonadicRightAdjoint L :=
  ComonadicLeftAdjoint.adj


instance (L : C ⥤ D) [ComonadicLeftAdjoint L] :
    (Comonad.comparison (comonadicAdjunction L)).IsEquivalence :=
  ComonadicLeftAdjoint.eqv


instance (L : C ⥤ D) [ComonadicLeftAdjoint L] : L.IsLeftAdjoint :=
  (comonadicAdjunction L).isLeftAdjoint


noncomputable instance (G : Comonad C) : ComonadicLeftAdjoint G.forget where
  adj := G.adj
  eqv := { }

-- TODO: This holds more generally for idempotent adjunctions, not just reflective adjunctions.

instance μ_iso_of_reflective [Reflective R] : IsIso (reflectorAdjunction R).toMonad.μ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Reflective R
    ⊢ CategoryTheory.IsIso (CategoryTheory.reflectorAdjunction R).toMonad.μ
  -/
  dsimp
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Reflective R
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerRight (CategoryTheory.whiskerLef …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance δ_iso_of_coreflective [Coreflective R] : IsIso (coreflectorAdjunction R).toComonad.δ := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Coreflective R
    ⊢ CategoryTheory.IsIso (CategoryTheory.coreflectorAdjunction R).toComonad.δ
  -/
  dsimp
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Coreflective R
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerRight (CategoryTheory.whiskerLef …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Reflective R] (X : (reflectorAdjunction R).toMonad.Algebra) :
    IsIso ((reflectorAdjunction R).unit.app X.A) :=
  ⟨⟨X.a,
      ⟨X.unit, by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Reflective R
          X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
          ⊢ Eq (CategoryTheory.CategoryStruct.comp X.a ((CategoryTheory.reflectorAdjunct …
        -/
        dsimp only [Functor.id_obj]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Reflective R
          X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
          ⊢ Eq (CategoryTheory.CategoryStruct.comp X.a ((CategoryTheory.reflectorAdjunct …
        -/
        rw [← (reflectorAdjunction R).unit_naturality]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Reflective R
          X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.reflectorAdjunction  …
        -/
        dsimp only [Functor.comp_obj, Adjunction.toMonad_coe]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Reflective R
          X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.reflectorAdjunction  …
        -/
        rw [unit_obj_eq_map_unit, ← Functor.map_comp, ← Functor.map_comp]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Reflective R
          X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
          ⊢ Eq (R.map ((CategoryTheory.reflector R).map (CategoryTheory.CategoryStruct.c …
        -/
        erw [X.unit]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Reflective R
          X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
          ⊢ Eq (R.map ((CategoryTheory.reflector R).map (CategoryTheory.CategoryStruct.i …
        -/
        simp⟩⟩⟩
        /-
          🎉 no goals
        -/


instance comparison_essSurj [Reflective R] :
    (Monad.comparison (reflectorAdjunction R)).EssSurj := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Reflective R
    ⊢ (CategoryTheory.Monad.comparison (CategoryTheory.reflectorAdjunction R)).Ess …
  -/
  refine ⟨fun X => ⟨(reflector R).obj X.A, ⟨?_⟩⟩⟩
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Reflective R
    X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
    ⊢ CategoryTheory.Iso ((CategoryTheory.Monad.comparison (CategoryTheory.reflect …
  -/
  symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Reflective R
    X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
    ⊢ CategoryTheory.Iso X ((CategoryTheory.Monad.comparison (CategoryTheory.refle …
  -/
  refine Monad.Algebra.isoMk ?_ ?_
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      inst✝ : CategoryTheory.Reflective R
      X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
      ⊢ CategoryTheory.Iso X.A ((CategoryTheory.Monad.comparison (CategoryTheory.ref …
    -/
  · exact asIso ((reflectorAdjunction R).unit.app X.A)
    /-
      🎉 no goals
    -/
  dsimp only [Functor.comp_map, Monad.comparison_obj_a, asIso_hom, Functor.comp_obj,
    Monad.comparison_obj_A, Adjunction.toMonad_coe]
  /-
    case refine_2
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Reflective R
    X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map ((CategoryTheory.reflector R). …
  -/
  rw [← cancel_epi ((reflectorAdjunction R).unit.app X.A)]
  /-
    case refine_2
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Reflective R
    X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.reflectorAdjunction  …
  -/
  dsimp only [Functor.id_obj, Functor.comp_obj]
  rw [Adjunction.unit_naturality_assoc,
    Adjunction.right_triangle_components, comp_id]
  /-
    case refine_2
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Reflective R
    X : (CategoryTheory.reflectorAdjunction R).toMonad.Algebra
    ⊢ Eq ((CategoryTheory.reflectorAdjunction R).unit.app X.A) (CategoryTheory.Cat …
  -/
  apply (X.unit_assoc _).symm
  /-
    🎉 no goals
  -/


lemma comparison_full [R.Full] {L : C ⥤ D} (adj : L ⊣ R) :
    (Monad.comparison adj).Full where
                                          /-
                                            C : Type u₁
                                            inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                            D : Type u₂
                                            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                            R : CategoryTheory.Functor D C
                                            inst✝ : R.Full
                                            L : CategoryTheory.Functor C D
                                            adj : CategoryTheory.Adjunction L R
                                            X✝ Y✝ : D
                                            f : Quiver.Hom ((CategoryTheory.Monad.comparison adj).obj X✝) ((CategoryTheory …
                                            ⊢ Eq ((CategoryTheory.Monad.comparison adj).map (R.preimage f.f)) f
                                          -/
  map_surjective f := ⟨R.preimage f.f, by aesop_cat⟩
                                          /-
                                            🎉 no goals
                                          -/


instance [Coreflective R] (X : (coreflectorAdjunction R).toComonad.Coalgebra) :
    IsIso ((coreflectorAdjunction R).counit.app X.A) :=
  ⟨⟨X.a,
      ⟨by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Coreflective R
          X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coreflectorAdjunctio …
        -/
        dsimp only [Functor.id_obj]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Coreflective R
          X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.coreflectorAdjunctio …
        -/
        rw [← (coreflectorAdjunction R).counit_naturality]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Coreflective R
          X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map ((CategoryTheory.coreflector R …
        -/
        dsimp only [Functor.comp_obj, Adjunction.toMonad_coe]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Coreflective R
          X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map ((CategoryTheory.coreflector R …
        -/
        rw [counit_obj_eq_map_counit, ← Functor.map_comp, ← Functor.map_comp]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Coreflective R
          X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
          ⊢ Eq (R.map ((CategoryTheory.coreflector R).map (CategoryTheory.CategoryStruct …
        -/
        erw [X.counit]
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          L : CategoryTheory.Functor C D
          R : CategoryTheory.Functor D C
          inst✝ : CategoryTheory.Coreflective R
          X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
          ⊢ Eq (R.map ((CategoryTheory.coreflector R).map (CategoryTheory.CategoryStruct …
        -/
        simp, X.counit⟩⟩⟩
        /-
          🎉 no goals
        -/


instance comparison_essSurj [Coreflective R] :
    (Comonad.comparison (coreflectorAdjunction R)).EssSurj := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Coreflective R
    ⊢ (CategoryTheory.Comonad.comparison (CategoryTheory.coreflectorAdjunction R)) …
  -/
  refine ⟨fun X => ⟨(coreflector R).obj X.A, ⟨?_⟩⟩⟩
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Coreflective R
    X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
    ⊢ CategoryTheory.Iso ((CategoryTheory.Comonad.comparison (CategoryTheory.coref …
  -/
  refine Comonad.Coalgebra.isoMk ?_ ?_
    /-
      case refine_1
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      inst✝ : CategoryTheory.Coreflective R
      X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
      ⊢ CategoryTheory.Iso ((CategoryTheory.Comonad.comparison (CategoryTheory.coref …
    -/
  · exact (asIso ((coreflectorAdjunction R).counit.app X.A))
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Coreflective R
    X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comonad.comparison ( …
  -/
  rw [← cancel_mono ((coreflectorAdjunction R).counit.app X.A)]
  simp only [Adjunction.counit_naturality, Functor.comp_obj, Functor.id_obj,
    Adjunction.left_triangle_components_assoc, assoc]
  /-
    case refine_2
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Coreflective R
    X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comonad.comparison ( …
  -/
  erw [X.counit]
  /-
    case refine_2
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    inst✝ : CategoryTheory.Coreflective R
    X : (CategoryTheory.coreflectorAdjunction R).toComonad.Coalgebra
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comonad.comparison ( …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma comparison_full [R.Full] {L : C ⥤ D} (adj : R ⊣ L) :
    (Comonad.comparison adj).Full where
                                          /-
                                            C : Type u₁
                                            inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                            D : Type u₂
                                            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                            R : CategoryTheory.Functor D C
                                            inst✝ : R.Full
                                            L : CategoryTheory.Functor C D
                                            adj : CategoryTheory.Adjunction R L
                                            X✝ Y✝ : D
                                            f : Quiver.Hom ((CategoryTheory.Comonad.comparison adj).obj X✝) ((CategoryTheo …
                                            ⊢ Eq ((CategoryTheory.Comonad.comparison adj).map (R.preimage f.f)) f
                                          -/
  map_surjective f := ⟨R.preimage f.f, by aesop_cat⟩
                                          /-
                                            🎉 no goals
                                          -/


/-- Any reflective inclusion has a monadic right adjoint.
    cf Prop 5.3.3 of [Riehl][riehl2017] -/
instance (priority := 100) monadicOfReflective [Reflective R] :
    MonadicRightAdjoint R where
  adj := reflectorAdjunction R
  eqv := { full := Reflective.comparison_full _ }


/-- Any coreflective inclusion has a comonadic left adjoint.
    cf Dual statement of Prop 5.3.3 of [Riehl][riehl2017] -/
instance (priority := 100) comonadicOfCoreflective [Coreflective R] :
    ComonadicLeftAdjoint R where
  adj := coreflectorAdjunction R
  eqv := { full := Coreflective.comparison_full _ }


