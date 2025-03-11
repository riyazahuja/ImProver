/-- The top map in the equalizer diagram we will construct. -/
@[simps!]
def CofreeEqualizer.topMap :  (Comonad.cofree T).obj X.A ⟶ (Comonad.cofree T).obj (T.obj X.A) :=
  (Comonad.cofree T).map X.a


/-- The bottom map in the equalizer diagram we will construct. -/
@[simps]
def CofreeEqualizer.bottomMap :
    (Comonad.cofree T).obj X.A ⟶ (Comonad.cofree T).obj (T.obj X.A) where
  f := T.δ.app X.A
  h := T.coassoc X.A


/-- The fork map in the equalizer diagram we will construct. -/
@[simps]
def CofreeEqualizer.ι : X ⟶ (Comonad.cofree T).obj X.A where
  f := X.a
  h := X.coassoc.symm


theorem CofreeEqualizer.condition :
    CofreeEqualizer.ι X ≫ CofreeEqualizer.topMap X =
      CofreeEqualizer.ι X ≫ CofreeEqualizer.bottomMap X :=
  Coalgebra.Hom.ext X.coassoc.symm


instance : IsCoreflexivePair (CofreeEqualizer.topMap X) (CofreeEqualizer.bottomMap X) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Comonad C
    X : T.Coalgebra
    ⊢ CategoryTheory.IsCoreflexivePair (CategoryTheory.Comonad.CofreeEqualizer.top …
  -/
  apply IsCoreflexivePair.mk' _ _ _
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Comonad C
      X : T.Coalgebra
      ⊢ Quiver.Hom (T.cofree.obj (T.obj X.A)) (T.cofree.obj X.A)
    -/
  · apply (cofree T).map (T.ε.app X.A)
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Comonad C
      X : T.Coalgebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Comonad.CofreeEqualiz …
    -/
  · ext
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Comonad C
      X : T.Coalgebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Comonad.CofreeEqualiz …
    -/
    dsimp
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Comonad C
      X : T.Coalgebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) (T.map (T.ε.app X.A))) (C …
    -/
    rw [← Functor.map_comp, X.counit, Functor.map_id]
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Comonad C
      X : T.Coalgebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Comonad.CofreeEqualiz …
    -/
  · ext
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Comonad C
      X : T.Coalgebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Comonad.CofreeEqualiz …
    -/
    apply Comonad.right_counit
    /-
      🎉 no goals
    -/


/-- Construct the Beck fork in the category of coalgebras. This fork is coreflexive as well as an
equalizer.
-/
@[simps!]
def beckCoalgebraFork : Fork (CofreeEqualizer.topMap X) (CofreeEqualizer.bottomMap X) :=
  Fork.ofι _ (CofreeEqualizer.condition X)


/-- The fork constructed is a limit. This shows that any coalgebra is a (coreflexive) equalizer of
cofree coalgebras.
-/
def beckCoalgebraEqualizer : IsLimit (beckCoalgebraFork X) :=
  Fork.IsLimit.mk' _ fun s => by
    have h₁ :  s.ι.f  ≫ (T : C ⥤ C).map X.a = s.ι.f ≫ T.δ.app X.A :=
      congr_arg Comonad.Coalgebra.Hom.f s.condition
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Comonad C
      X : T.Coalgebra
      s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
    -/
    have h₂ :  s.pt.a ≫ (T : C ⥤ C).map s.ι.f = s.ι.f ≫ T.δ.app X.A := s.ι.h
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Comonad C
      X : T.Coalgebra
      s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
    -/
    refine ⟨⟨s.ι.f ≫ T.ε.app _, ?_⟩, ?_, ?_⟩
      /-
        case refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const Categ …
      -/
    · dsimp
      /-
        case refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map (CategoryTheory.Categor …
      -/
      rw [Functor.map_comp, reassoc_of% h₂, Comonad.right_counit]
      /-
        case refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι.f (CategoryTheory.CategoryStruct. …
      -/
      dsimp
      /-
        case refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι.f (CategoryTheory.CategoryStruct. …
      -/
      rw [Category.comp_id, Category.assoc]
      /-
        case refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ Eq s.ι.f (CategoryTheory.CategoryStruct.comp s.ι.f (CategoryTheory.CategoryS …
      -/
      erw [← T.ε.naturality, reassoc_of% h₁, Comonad.left_counit] -- TODO: missing simp lemmas
      /-
        case refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ Eq s.ι.f (CategoryTheory.CategoryStruct.comp s.ι.f (CategoryTheory.CategoryS …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := CategoryTheory.CategoryStruct. …
      -/
    · ext
      /-
        case refine_2.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp { f := CategoryTheory.CategoryStruct. …
      -/
      simpa [← T.ε.naturality_assoc, T.left_counit_assoc] using h₁ =≫ T.ε.app ((T : C ⥤ C).obj X.A)
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        ⊢ ∀ {m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.Walk …
      -/
    · intro m hm
      /-
        case refine_3
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Comonad.beckCoal …
        ⊢ Eq m { f := CategoryTheory.CategoryStruct.comp s.ι.f (T.ε.app X.A), h := ⋯ }
      -/
      ext
      /-
        case refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Comonad.beckCoal …
        ⊢ Eq m.f { f := CategoryTheory.CategoryStruct.comp s.ι.f (T.ε.app X.A), h := ⋯ …
      -/
      dsimp only
      /-
        case refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Comonad.beckCoal …
        ⊢ Eq m.f (CategoryTheory.CategoryStruct.comp s.ι.f (T.ε.app X.A))
      -/
      rw [← hm]
      /-
        case refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Comonad C
        X : T.Coalgebra
        s : CategoryTheory.Limits.Fork (CategoryTheory.Comonad.CofreeEqualizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp s.ι.f (T.map X.a)) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map s.ι.f)) (CategoryThe …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Comonad.beckCoal …
        ⊢ Eq m.f (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
      -/
      simp [beckCoalgebraFork, X.counit]
      /-
        🎉 no goals
      -/


/-- The Beck fork is a split equalizer. -/
def beckSplitEqualizer : IsSplitEqualizer (T.map X.a) (T.δ.app _) X.a :=
  ⟨T.ε.app _, T.ε.app _, X.coassoc.symm, X.counit, T.left_counit _, (T.ε.naturality _)⟩


/-- This is the Beck fork. It is a split equalizer, in particular a equalizer. -/
@[simps! pt]
def beckFork : Fork (T.map X.a) (T.δ.app _) :=
  (beckSplitEqualizer X).asFork


@[simp]
theorem beckFork_ι : (beckFork X).ι = X.a :=
  rfl


/-- The Beck fork is a equalizer. -/
def beckEqualizer : IsLimit (beckFork X) :=
  (beckSplitEqualizer X).isEqualizer


@[simp]
theorem beckEqualizer_lift (s : Fork (T.toFunctor.map X.a) (T.δ.app X.A)) :
    (beckEqualizer X).lift s = s.ι ≫ T.ε.app _ :=
  rfl


