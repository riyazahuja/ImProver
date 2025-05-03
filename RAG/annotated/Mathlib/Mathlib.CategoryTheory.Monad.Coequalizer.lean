/-- The top map in the coequalizer diagram we will construct. -/
@[simps!]
def FreeCoequalizer.topMap : (Monad.free T).obj (T.obj X.A) ⟶ (Monad.free T).obj X.A :=
  (Monad.free T).map X.a


/-- The bottom map in the coequalizer diagram we will construct. -/
@[simps]
def FreeCoequalizer.bottomMap : (Monad.free T).obj (T.obj X.A) ⟶ (Monad.free T).obj X.A where
  f := T.μ.app X.A
  h := T.assoc X.A


/-- The cofork map in the coequalizer diagram we will construct. -/
@[simps]
def FreeCoequalizer.π : (Monad.free T).obj X.A ⟶ X where
  f := X.a
  h := X.assoc.symm


theorem FreeCoequalizer.condition :
    FreeCoequalizer.topMap X ≫ FreeCoequalizer.π X =
      FreeCoequalizer.bottomMap X ≫ FreeCoequalizer.π X :=
  Algebra.Hom.ext X.assoc.symm


instance : IsReflexivePair (FreeCoequalizer.topMap X) (FreeCoequalizer.bottomMap X) := by
  /-
    C : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} C
    T : CategoryTheory.Monad C
    X : T.Algebra
    ⊢ CategoryTheory.IsReflexivePair (CategoryTheory.Monad.FreeCoequalizer.topMap  …
  -/
  apply IsReflexivePair.mk' _ _ _
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      X : T.Algebra
      ⊢ Quiver.Hom (T.free.obj X.A) (T.free.obj (T.obj X.A))
    -/
  · apply (free T).map (T.η.app X.A)
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      X : T.Algebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.free.map (T.η.app X.A)) (CategoryT …
    -/
  · ext
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      X : T.Algebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.free.map (T.η.app X.A)) (CategoryT …
    -/
    dsimp
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      X : T.Algebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (T.η.app X.A)) (T.map X.a)) (C …
    -/
    rw [← Functor.map_comp, X.unit, Functor.map_id]
    /-
      🎉 no goals
    -/
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      X : T.Algebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.free.map (T.η.app X.A)) (CategoryT …
    -/
  · ext
    /-
      case h
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      X : T.Algebra
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.free.map (T.η.app X.A)) (CategoryT …
    -/
    apply Monad.right_unit
    /-
      🎉 no goals
    -/


/-- Construct the Beck cofork in the category of algebras. This cofork is reflexive as well as a
coequalizer.
-/
@[simps!]
def beckAlgebraCofork : Cofork (FreeCoequalizer.topMap X) (FreeCoequalizer.bottomMap X) :=
  Cofork.ofπ _ (FreeCoequalizer.condition X)


/-- The cofork constructed is a colimit. This shows that any algebra is a (reflexive) coequalizer of
free algebras.
-/
def beckAlgebraCoequalizer : IsColimit (beckAlgebraCofork X) :=
  Cofork.IsColimit.mk' _ fun s => by
    have h₁ : (T : C ⥤ C).map X.a ≫ s.π.f = T.μ.app X.A ≫ s.π.f :=
      congr_arg Monad.Algebra.Hom.f s.condition
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      X : T.Algebra
      s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    have h₂ : (T : C ⥤ C).map s.π.f ≫ s.pt.a = T.μ.app X.A ≫ s.π.f := s.π.h
    /-
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      X : T.Algebra
      s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    refine ⟨⟨T.η.app _ ≫ s.π.f, ?_⟩, ?_, ?_⟩
      /-
        case refine_1
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        X : T.Algebra
        s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (CategoryTheory.CategoryStruct …
      -/
    · dsimp
      rw [Functor.map_comp, Category.assoc, h₂, Monad.right_unit_assoc,
        show X.a ≫ _ ≫ _ = _ from T.η.naturality_assoc _ _, h₁, Monad.left_unit_assoc]
      /-
        case refine_2
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        X : T.Algebra
        s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monad.beckAlgebraCofo …
      -/
    · ext
      /-
        case refine_2.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        X : T.Algebra
        s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monad.beckAlgebraCofo …
      -/
      simpa [← T.η.naturality_assoc, T.left_unit_assoc] using T.η.app ((T : C ⥤ C).obj X.A) ≫= h₁
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        X : T.Algebra
        s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
        ⊢ ∀ {m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.Walk …
      -/
    · intro m hm
      /-
        case refine_3
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        X : T.Algebra
        s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monad.beckAlgebraC …
        ⊢ Eq m { f := CategoryTheory.CategoryStruct.comp (T.η.app (CategoryTheory.Mona …
      -/
      ext
      /-
        case refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        X : T.Algebra
        s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monad.beckAlgebraC …
        ⊢ Eq m.f { f := CategoryTheory.CategoryStruct.comp (T.η.app (CategoryTheory.Mo …
      -/
      dsimp only
      /-
        case refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        X : T.Algebra
        s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monad.beckAlgebraC …
        ⊢ Eq m.f (CategoryTheory.CategoryStruct.comp (T.η.app (CategoryTheory.Monad.be …
      -/
      rw [← hm]
      /-
        case refine_3.h
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        X : T.Algebra
        s : CategoryTheory.Limits.Cofork (CategoryTheory.Monad.FreeCoequalizer.topMap  …
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (T.map X.a) s.π.f) (CategoryTheory …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (T.map s.π.f) s.pt.a) (CategoryThe …
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Monad.beckAlgebraC …
        ⊢ Eq m.f (CategoryTheory.CategoryStruct.comp (T.η.app (CategoryTheory.Monad.be …
      -/
      apply (X.unit_assoc _).symm
      /-
        🎉 no goals
      -/


/-- The Beck cofork is a split coequalizer. -/
def beckSplitCoequalizer : IsSplitCoequalizer (T.map X.a) (T.μ.app _) X.a :=
  ⟨T.η.app _, T.η.app _, X.assoc.symm, X.unit, T.left_unit _, (T.η.naturality _).symm⟩


/-- This is the Beck cofork. It is a split coequalizer, in particular a coequalizer. -/
@[simps! pt]
def beckCofork : Cofork (T.map X.a) (T.μ.app _) :=
  (beckSplitCoequalizer X).asCofork


@[simp]
theorem beckCofork_π : (beckCofork X).π = X.a :=
  rfl


/-- The Beck cofork is a coequalizer. -/
def beckCoequalizer : IsColimit (beckCofork X) :=
  (beckSplitCoequalizer X).isCoequalizer


@[simp]
theorem beckCoequalizer_desc (s : Cofork (T.toFunctor.map X.a) (T.μ.app X.A)) :
    (beckCoequalizer X).desc s = T.η.app _ ≫ s.π :=
  rfl


