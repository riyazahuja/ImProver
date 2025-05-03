/-- (Impl) The natural transformation used to define the new cone -/
@[simps]
def γ : D ⋙ T.forget ⋙ ↑T ⟶ D ⋙ T.forget where app j := (D.obj j).a


/-- (Impl) This new cone is used to construct the algebra structure -/
@[simps! π_app]
def newCone : Cone (D ⋙ forget T) where
  pt := T.obj c.pt
  π := (Functor.constComp _ _ (T : C ⥤ C)).inv ≫ whiskerRight c.π (T : C ⥤ C) ≫ γ D


/-- The algebra structure which will be the apex of the new limit cone for `D`. -/
@[simps]
def conePoint : Algebra T where
  A := c.pt
  a := t.lift (newCone D c)
  unit :=
    t.hom_ext fun j => by
      rw [Category.assoc, t.fac, newCone_π_app, ← T.η.naturality_assoc, Functor.id_map,
        (D.obj j).unit]
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        J : Type u
        inst✝ : CategoryTheory.Category.{v, u} J
        D : CategoryTheory.Functor J T.Algebra
        c : CategoryTheory.Limits.Cone (D.comp T.forget)
        t : CategoryTheory.Limits.IsLimit c
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app j) (CategoryTheory.CategoryS …
      -/
      dsimp; simp
             /-
               🎉 no goals
             -/
  -- See library note [dsimp, simp]
  assoc :=
    t.hom_ext fun j => by
      rw [Category.assoc, Category.assoc, t.fac (newCone D c), newCone_π_app, ←
        Functor.map_comp_assoc, t.fac (newCone D c), newCone_π_app, ← T.μ.naturality_assoc,
        (D.obj j).assoc, Functor.map_comp, Category.assoc]
      /-
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        T : CategoryTheory.Monad C
        J : Type u
        inst✝ : CategoryTheory.Category.{v, u} J
        D : CategoryTheory.Functor J T.Algebra
        c : CategoryTheory.Limits.Cone (D.comp T.forget)
        t : CategoryTheory.Limits.IsLimit c
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((T.comp T.toFunctor).map (c.π.app j) …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- (Impl) Construct the lifted cone in `Algebra T` which will be limiting. -/
@[simps]
def liftedCone : Cone D where
  pt := conePoint D c t
  π :=
    { app := fun j => { f := c.π.app j }
      naturality := fun X Y f => by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          D : CategoryTheory.Functor J T.Algebra
          c : CategoryTheory.Limits.Cone (D.comp T.forget)
          t : CategoryTheory.Limits.IsLimit c
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        ext1
        /-
          case h
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          D : CategoryTheory.Functor J T.Algebra
          c : CategoryTheory.Limits.Cone (D.comp T.forget)
          t : CategoryTheory.Limits.IsLimit c
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          case h
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          D : CategoryTheory.Functor J T.Algebra
          c : CategoryTheory.Limits.Cone (D.comp T.forget)
          t : CategoryTheory.Limits.IsLimit c
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
        -/
        erw [c.w f]
        /-
          case h
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          D : CategoryTheory.Functor J T.Algebra
          c : CategoryTheory.Limits.Cone (D.comp T.forget)
          t : CategoryTheory.Limits.IsLimit c
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- (Impl) Prove that the lifted cone is limiting. -/
@[simps]
def liftedConeIsLimit : IsLimit (liftedCone D c t) where
  lift s :=
    { f := t.lift ((forget T).mapCone s)
      h :=
        t.hom_ext fun j => by
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            T : CategoryTheory.Monad C
            J : Type u
            inst✝ : CategoryTheory.Category.{v, u} J
            D : CategoryTheory.Functor J T.Algebra
            c : CategoryTheory.Limits.Cone (D.comp T.forget)
            t : CategoryTheory.Limits.IsLimit c
            s : CategoryTheory.Limits.Cone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          dsimp
          rw [Category.assoc, Category.assoc, t.fac, newCone_π_app, ← Functor.map_comp_assoc,
            t.fac, Functor.mapCone_π_app]
          /-
            C : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
            T : CategoryTheory.Monad C
            J : Type u
            inst✝ : CategoryTheory.Category.{v, u} J
            D : CategoryTheory.Functor J T.Algebra
            c : CategoryTheory.Limits.Cone (D.comp T.forget)
            t : CategoryTheory.Limits.IsLimit c
            s : CategoryTheory.Limits.Cone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (T.forget.map (s.π.app j))) (D …
          -/
          apply (s.π.app j).h }
          /-
            🎉 no goals
          -/
  uniq s m J := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} J✝
      D : CategoryTheory.Functor J✝ T.Algebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D
      m : Quiver.Hom s.pt (CategoryTheory.Monad.ForgetCreatesLimits.liftedCone D c t …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Mona …
      ⊢ Eq m ((fun s => { f := t.lift (T.forget.mapCone s), h := ⋯ }) s)
    -/
    ext1
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} J✝
      D : CategoryTheory.Functor J✝ T.Algebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D
      m : Quiver.Hom s.pt (CategoryTheory.Monad.ForgetCreatesLimits.liftedCone D c t …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Mona …
      ⊢ Eq m.f ((fun s => { f := t.lift (T.forget.mapCone s), h := ⋯ }) s).f
    -/
    apply t.hom_ext
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} J✝
      D : CategoryTheory.Functor J✝ T.Algebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D
      m : Quiver.Hom s.pt (CategoryTheory.Monad.ForgetCreatesLimits.liftedCone D c t …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Mona …
      ⊢ ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m.f (c.π.app j)) (Categor …
    -/
    intro j
    /-
      case h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} J✝
      D : CategoryTheory.Functor J✝ T.Algebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D
      m : Quiver.Hom s.pt (CategoryTheory.Monad.ForgetCreatesLimits.liftedCone D c t …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Mona …
      j : J✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m.f (c.π.app j)) (CategoryTheory.Cate …
    -/
    simpa [t.fac ((forget T).mapCone s) j] using congr_arg Algebra.Hom.f (J j)
    /-
      🎉 no goals
    -/


/-- The forgetful functor from the Eilenberg-Moore category creates limits. -/
noncomputable instance forgetCreatesLimits : CreatesLimitsOfSize (forget T) where
  CreatesLimitsOfShape := {
    CreatesLimit := fun {D} =>
      createsLimitOfReflectsIso fun c t =>
        { liftedCone := ForgetCreatesLimits.liftedCone D c t
          validLift := Cones.ext (Iso.refl _) fun _ => (id_comp _).symm
          makesLimit := ForgetCreatesLimits.liftedConeIsLimit _ _ _ } }


/-- `D ⋙ forget T` has a limit, then `D` has a limit. -/
theorem hasLimit_of_comp_forget_hasLimit (D : J ⥤ Algebra T) [HasLimit (D ⋙ forget T)] :
    HasLimit D :=
  hasLimit_of_created D (forget T)


/-- (Impl)
The natural transformation given by the algebra structure maps, used to construct a cocone `c` with
point `colimit (D ⋙ forget T)`.
 -/
@[simps]
def γ : (D ⋙ forget T) ⋙ ↑T ⟶ D ⋙ forget T where app j := (D.obj j).a


/-- (Impl)
A cocone for the diagram `(D ⋙ forget T) ⋙ T` found by composing the natural transformation `γ`
with the colimiting cocone for `D ⋙ forget T`.
-/
@[simps]
def newCocone : Cocone ((D ⋙ forget T) ⋙ (T : C ⥤ C)) where
  pt := c.pt
  ι := γ ≫ c.ι


/-- (Impl)
Define the map `λ : TL ⟶ L`, which will serve as the structure of the coalgebra on `L`, and
we will show is the colimiting object. We use the cocone constructed by `c` and the fact that
`T` preserves colimits to produce this morphism.
-/
noncomputable abbrev lambda : ((T : C ⥤ C).mapCocone c).pt ⟶ c.pt :=
  (isColimitOfPreserves _ t).desc (newCocone c)


/-- (Impl) The key property defining the map `λ : TL ⟶ L`. -/
theorem commuting (j : J) : (T : C ⥤ C).map (c.ι.app j) ≫ lambda c t = (D.obj j).a ≫ c.ι.app j :=
  (isColimitOfPreserves _ t).fac (newCocone c) j


/-- (Impl)
Construct the colimiting algebra from the map `λ : TL ⟶ L` given by `lambda`. We are required to
show it satisfies the two algebra laws, which follow from the algebra laws for the image of `D` and
our `commuting` lemma.
-/
@[simps]
noncomputable def coconePoint : Algebra T where
  A := c.pt
  a := lambda c t
  unit := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      D : CategoryTheory.Functor J T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.η.app c.pt) (CategoryTheory.Monad. …
    -/
    apply t.hom_ext
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      D : CategoryTheory.Functor J T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) (CategoryTheor …
    -/
    intro j
    rw [show c.ι.app j ≫ T.η.app c.pt ≫ _ = T.η.app (D.obj j).A ≫ _ ≫ _ from
        T.η.naturality_assoc _ _,
      commuting, Algebra.unit_assoc (D.obj j)]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      D : CategoryTheory.Functor J T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      j : J
      ⊢ Eq (c.ι.app j) (CategoryTheory.CategoryStruct.comp (c.ι.app j) (CategoryTheo …
    -/
    dsimp; simp
           /-
             🎉 no goals
           -/
  -- See library note [dsimp, simp]
  assoc := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      D : CategoryTheory.Functor J T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.μ.app c.pt) (CategoryTheory.Monad. …
    -/
    refine (isColimitOfPreserves _ (isColimitOfPreserves _ t)).hom_ext fun j => ?_
    rw [Functor.mapCocone_ι_app, Functor.mapCocone_ι_app,
      show (T : C ⥤ C).map ((T : C ⥤ C).map _) ≫ _ ≫ _ = _ from T.μ.naturality_assoc _ _, ←
      Functor.map_comp_assoc, commuting, Functor.map_comp, Category.assoc, commuting]
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      D : CategoryTheory.Functor J T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.μ.app ((D.comp T.forget).obj j)) ( …
    -/
    apply (D.obj j).assoc_assoc _
    /-
      🎉 no goals
    -/


/-- (Impl) Construct the lifted cocone in `Algebra T` which will be colimiting. -/
@[simps]
noncomputable def liftedCocone : Cocone D where
  pt := coconePoint c t
  ι :=
    { app := fun j =>
        { f := c.ι.app j
          h := commuting _ _ _ }
      naturality := fun A B f => by
        /-
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          J : Type u
          inst✝² : CategoryTheory.Category.{v, u} J
          D : CategoryTheory.Functor J T.Algebra
          c : CategoryTheory.Limits.Cocone (D.comp T.forget)
          t : CategoryTheory.Limits.IsColimit c
          inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
          inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
          A B : J
          f : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f) ((fun j => { f := c.ι.app j …
        -/
        ext1
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          J : Type u
          inst✝² : CategoryTheory.Category.{v, u} J
          D : CategoryTheory.Functor J T.Algebra
          c : CategoryTheory.Limits.Cocone (D.comp T.forget)
          t : CategoryTheory.Limits.IsColimit c
          inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
          inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
          A B : J
          f : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f) ((fun j => { f := c.ι.app j …
        -/
        dsimp
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          J : Type u
          inst✝² : CategoryTheory.Category.{v, u} J
          D : CategoryTheory.Functor J T.Algebra
          c : CategoryTheory.Limits.Cocone (D.comp T.forget)
          t : CategoryTheory.Limits.IsColimit c
          inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
          inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
          A B : J
          f : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f).f (c.ι.app B)) (CategoryThe …
        -/
        rw [comp_id]
        /-
          case h
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          T : CategoryTheory.Monad C
          J : Type u
          inst✝² : CategoryTheory.Category.{v, u} J
          D : CategoryTheory.Functor J T.Algebra
          c : CategoryTheory.Limits.Cocone (D.comp T.forget)
          t : CategoryTheory.Limits.IsColimit c
          inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
          inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
          A B : J
          f : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f).f (c.ι.app B)) (c.ι.app A)
        -/
        apply c.w }
        /-
          🎉 no goals
        -/


/-- (Impl) Prove that the lifted cocone is colimiting. -/
@[simps]
noncomputable def liftedCoconeIsColimit : IsColimit (liftedCocone c t) where
  desc s :=
    { f := t.desc ((forget T).mapCocone s)
      h :=
        (isColimitOfPreserves (T : C ⥤ C) t).hom_ext fun j => by
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            T : CategoryTheory.Monad C
            J : Type u
            inst✝² : CategoryTheory.Category.{v, u} J
            D : CategoryTheory.Functor J T.Algebra
            c : CategoryTheory.Limits.Cocone (D.comp T.forget)
            t : CategoryTheory.Limits.IsColimit c
            inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
            inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
            s : CategoryTheory.Limits.Cocone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((T.mapCocone c).ι.app j) (CategoryTh …
          -/
          dsimp
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            T : CategoryTheory.Monad C
            J : Type u
            inst✝² : CategoryTheory.Category.{v, u} J
            D : CategoryTheory.Functor J T.Algebra
            c : CategoryTheory.Limits.Cocone (D.comp T.forget)
            t : CategoryTheory.Limits.IsColimit c
            inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
            inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
            s : CategoryTheory.Limits.Cocone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map (c.ι.app j)) (CategoryTheory.C …
          -/
          rw [← Functor.map_comp_assoc, ← Category.assoc, t.fac, commuting, Category.assoc, t.fac]
          /-
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            T : CategoryTheory.Monad C
            J : Type u
            inst✝² : CategoryTheory.Category.{v, u} J
            D : CategoryTheory.Functor J T.Algebra
            c : CategoryTheory.Limits.Cocone (D.comp T.forget)
            t : CategoryTheory.Limits.IsColimit c
            inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
            inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
            s : CategoryTheory.Limits.Cocone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (T.map ((T.forget.mapCocone s).ι.app  …
          -/
          apply Algebra.Hom.h }
          /-
            🎉 no goals
          -/
  uniq s m J := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J✝ : Type u
      inst✝² : CategoryTheory.Category.{v, u} J✝
      D : CategoryTheory.Functor J✝ T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      s : CategoryTheory.Limits.Cocone D
      m : Quiver.Hom (CategoryTheory.Monad.ForgetCreatesColimits.liftedCocone c t).p …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monad. …
      ⊢ Eq m ((fun s => { f := t.desc (T.forget.mapCocone s), h := ⋯ }) s)
    -/
    ext1
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J✝ : Type u
      inst✝² : CategoryTheory.Category.{v, u} J✝
      D : CategoryTheory.Functor J✝ T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      s : CategoryTheory.Limits.Cocone D
      m : Quiver.Hom (CategoryTheory.Monad.ForgetCreatesColimits.liftedCocone c t).p …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monad. …
      ⊢ Eq m.f ((fun s => { f := t.desc (T.forget.mapCocone s), h := ⋯ }) s).f
    -/
    apply t.hom_ext
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J✝ : Type u
      inst✝² : CategoryTheory.Category.{v, u} J✝
      D : CategoryTheory.Functor J✝ T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      s : CategoryTheory.Limits.Cocone D
      m : Quiver.Hom (CategoryTheory.Monad.ForgetCreatesColimits.liftedCocone c t).p …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monad. …
      ⊢ ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m.f) (Categor …
    -/
    intro j
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      T : CategoryTheory.Monad C
      J✝ : Type u
      inst✝² : CategoryTheory.Category.{v, u} J✝
      D : CategoryTheory.Functor J✝ T.Algebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      s : CategoryTheory.Limits.Cocone D
      m : Quiver.Hom (CategoryTheory.Monad.ForgetCreatesColimits.liftedCocone c t).p …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Monad. …
      j : J✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m.f) (CategoryTheory.Cate …
    -/
    simpa using congr_arg Algebra.Hom.f (J j)
    /-
      🎉 no goals
    -/


/-- The forgetful functor from the Eilenberg-Moore category for a monad creates any colimit
which the monad itself preserves.
-/
noncomputable instance forgetCreatesColimit (D : J ⥤ Algebra T)
    [PreservesColimit (D ⋙ forget T) (T : C ⥤ C)]
    [PreservesColimit ((D ⋙ forget T) ⋙ ↑T) (T : C ⥤ C)] : CreatesColimit D (forget T) :=
  createsColimitOfReflectsIso fun c t =>
    { liftedCocone :=
        { pt := coconePoint c t
          ι :=
            { app := fun j =>
                { f := c.ι.app j
                  h := commuting _ _ _ }
              naturality := fun A B f => by
                /-
                  C : Type u₁
                  inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                  T : CategoryTheory.Monad C
                  J : Type u
                  inst✝² : CategoryTheory.Category.{v, u} J
                  D : CategoryTheory.Functor J T.Algebra
                  inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
                  inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
                  c : CategoryTheory.Limits.Cocone (D.comp T.forget)
                  t : CategoryTheory.Limits.IsColimit c
                  A B : J
                  f : Quiver.Hom A B
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f) ((fun j => { f := c.ι.app j …
                -/
                ext1
                /-
                  case h
                  C : Type u₁
                  inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                  T : CategoryTheory.Monad C
                  J : Type u
                  inst✝² : CategoryTheory.Category.{v, u} J
                  D : CategoryTheory.Functor J T.Algebra
                  inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
                  inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
                  c : CategoryTheory.Limits.Cocone (D.comp T.forget)
                  t : CategoryTheory.Limits.IsColimit c
                  A B : J
                  f : Quiver.Hom A B
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f) ((fun j => { f := c.ι.app j …
                -/
                dsimp
                /-
                  case h
                  C : Type u₁
                  inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                  T : CategoryTheory.Monad C
                  J : Type u
                  inst✝² : CategoryTheory.Category.{v, u} J
                  D : CategoryTheory.Functor J T.Algebra
                  inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
                  inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
                  c : CategoryTheory.Limits.Cocone (D.comp T.forget)
                  t : CategoryTheory.Limits.IsColimit c
                  A B : J
                  f : Quiver.Hom A B
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f).f (c.ι.app B)) (CategoryThe …
                -/
                erw [comp_id, c.w] } }
                /-
                  🎉 no goals
                -/
                   /-
                     C : Type u₁
                     inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                     T : CategoryTheory.Monad C
                     J : Type u
                     inst✝² : CategoryTheory.Category.{v, u} J
                     D : CategoryTheory.Functor J T.Algebra
                     inst✝¹ : CategoryTheory.Limits.PreservesColimit (D.comp T.forget) T.toFunctor
                     inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
                     c : CategoryTheory.Limits.Cocone (D.comp T.forget)
                     t : CategoryTheory.Limits.IsColimit c
                     ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((T.forget.mapCocone { pt  …
                   -/
      validLift := Cocones.ext (Iso.refl _)
                   /-
                     🎉 no goals
                   -/
      makesColimit := liftedCoconeIsColimit _ _ }


noncomputable instance forgetCreatesColimitsOfShape [PreservesColimitsOfShape J (T : C ⥤ C)] :
                                                                   /-
                                                                     C : Type u₁
                                                                     inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                     T : CategoryTheory.Monad C
                                                                     J : Type u
                                                                     inst✝¹ : CategoryTheory.Category.{v, u} J
                                                                     inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J T.toFunctor
                                                                     ⊢ {K : CategoryTheory.Functor J T.Algebra} → CategoryTheory.CreatesColimit K T …
                                                                   -/
    CreatesColimitsOfShape J (forget T) where CreatesColimit := by infer_instance
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


noncomputable instance forgetCreatesColimits [PreservesColimitsOfSize.{v, u} (T : C ⥤ C)] :
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                                 T : CategoryTheory.Monad C
                                                                                 J : Type u
                                                                                 inst✝¹ : CategoryTheory.Category.{v, u} J
                                                                                 inst✝ : CategoryTheory.Limits.PreservesColimitsOfSize.{v, u, v₁, v₁, u₁, u₁} T …
                                                                                 ⊢ {J : Type u} → [inst : CategoryTheory.Category.{v, u} J] → CategoryTheory.Cr …
                                                                               -/
    CreatesColimitsOfSize.{v, u} (forget T) where CreatesColimitsOfShape := by infer_instance
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- For `D : J ⥤ Algebra T`, `D ⋙ forget T` has a colimit, then `D` has a colimit provided colimits
of shape `J` are preserved by `T`.
-/
theorem forget_creates_colimits_of_monad_preserves [PreservesColimitsOfShape J (T : C ⥤ C)]
    (D : J ⥤ Algebra T) [HasColimit (D ⋙ forget T)] : HasColimit D :=
  hasColimit_of_created D (forget T)


instance comp_comparison_forget_hasLimit (F : J ⥤ D) (R : D ⥤ C) [MonadicRightAdjoint R]
    [HasLimit (F ⋙ R)] :
    HasLimit ((F ⋙ Monad.comparison (monadicAdjunction R)) ⋙ Monad.forget _) :=
  @hasLimitOfIso _ _ _ _ (F ⋙ R) _ _
    (isoWhiskerLeft F (Monad.comparisonForget (monadicAdjunction R)).symm)


instance comp_comparison_hasLimit (F : J ⥤ D) (R : D ⥤ C) [MonadicRightAdjoint R]
    [HasLimit (F ⋙ R)] : HasLimit (F ⋙ Monad.comparison (monadicAdjunction R)) :=
  Monad.hasLimit_of_comp_forget_hasLimit (F ⋙ Monad.comparison (monadicAdjunction R))


/-- Any monadic functor creates limits. -/
noncomputable def monadicCreatesLimits (R : D ⥤ C) [MonadicRightAdjoint R] :
    CreatesLimitsOfSize.{v, u} R :=
  createsLimitsOfNatIso (Monad.comparisonForget (monadicAdjunction R))


/-- The forgetful functor from the Eilenberg-Moore category for a monad creates any colimit
which the monad itself preserves.
-/
noncomputable def monadicCreatesColimitOfPreservesColimit (R : D ⥤ C) (K : J ⥤ D)
    [MonadicRightAdjoint R] [PreservesColimit (K ⋙ R) (monadicLeftAdjoint R ⋙ R)]
    [PreservesColimit ((K ⋙ R) ⋙ monadicLeftAdjoint R ⋙ R) (monadicLeftAdjoint R ⋙ R)] :
      CreatesColimit K R := by
  -- Porting note: It would be nice to have a variant of apply which introduces goals for missing
  -- instances.
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.MonadicRightAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (K.comp R) ((CategoryTheory.mo …
    inst✝ : CategoryTheory.Limits.PreservesColimit ((K.comp R).comp ((CategoryTheo …
    ⊢ CategoryTheory.CreatesColimit K R
  -/
  letI A := Monad.comparison (monadicAdjunction R)
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.MonadicRightAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (K.comp R) ((CategoryTheory.mo …
    inst✝ : CategoryTheory.Limits.PreservesColimit ((K.comp R).comp ((CategoryTheo …
    A : CategoryTheory.Functor D (CategoryTheory.monadicAdjunction R).toMonad.Alge …
    ⊢ CategoryTheory.CreatesColimit K R
  -/
  letI B := Monad.forget (Adjunction.toMonad (monadicAdjunction R))
  let i : (K ⋙ Monad.comparison (monadicAdjunction R)) ⋙ Monad.forget _ ≅ K ⋙ R :=
    Functor.associator _ _ _ ≪≫
      isoWhiskerLeft K (Monad.comparisonForget (monadicAdjunction R))
  letI : PreservesColimit ((K ⋙ A) ⋙ Monad.forget
    (Adjunction.toMonad (monadicAdjunction R)))
      (Adjunction.toMonad (monadicAdjunction R)).toFunctor := by
    dsimp
    exact preservesColimit_of_iso_diagram _ i.symm
  letI : PreservesColimit
    (((K ⋙ A) ⋙ Monad.forget (Adjunction.toMonad (monadicAdjunction R))) ⋙
      (Adjunction.toMonad (monadicAdjunction R)).toFunctor)
      (Adjunction.toMonad (monadicAdjunction R)).toFunctor := by
    dsimp
    exact preservesColimit_of_iso_diagram _ (isoWhiskerRight i (monadicLeftAdjoint R ⋙ R)).symm
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.MonadicRightAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (K.comp R) ((CategoryTheory.mo …
    inst✝ : CategoryTheory.Limits.PreservesColimit ((K.comp R).comp ((CategoryTheo …
    A : CategoryTheory.Functor D (CategoryTheory.monadicAdjunction R).toMonad.Alge …
    B : CategoryTheory.Functor (CategoryTheory.monadicAdjunction R).toMonad.Algebr …
    i : CategoryTheory.Iso ((K.comp (CategoryTheory.Monad.comparison (CategoryTheo …
    this✝ : CategoryTheory.Limits.PreservesColimit ((K.comp A).comp (CategoryTheor …
    this : CategoryTheory.Limits.PreservesColimit (((K.comp A).comp (CategoryTheor …
    ⊢ CategoryTheory.CreatesColimit K R
  -/
  letI : CreatesColimit (K ⋙ A) B := CategoryTheory.Monad.forgetCreatesColimit _
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.MonadicRightAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (K.comp R) ((CategoryTheory.mo …
    inst✝ : CategoryTheory.Limits.PreservesColimit ((K.comp R).comp ((CategoryTheo …
    A : CategoryTheory.Functor D (CategoryTheory.monadicAdjunction R).toMonad.Alge …
    B : CategoryTheory.Functor (CategoryTheory.monadicAdjunction R).toMonad.Algebr …
    i : CategoryTheory.Iso ((K.comp (CategoryTheory.Monad.comparison (CategoryTheo …
    this✝¹ : CategoryTheory.Limits.PreservesColimit ((K.comp A).comp (CategoryTheo …
    this✝ : CategoryTheory.Limits.PreservesColimit (((K.comp A).comp (CategoryTheo …
    this : CategoryTheory.CreatesColimit (K.comp A) B := CategoryTheory.Monad.forg …
    ⊢ CategoryTheory.CreatesColimit K R
  -/
  letI : CreatesColimit K (A ⋙ B) := CategoryTheory.compCreatesColimit _ _
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.MonadicRightAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (K.comp R) ((CategoryTheory.mo …
    inst✝ : CategoryTheory.Limits.PreservesColimit ((K.comp R).comp ((CategoryTheo …
    A : CategoryTheory.Functor D (CategoryTheory.monadicAdjunction R).toMonad.Alge …
    B : CategoryTheory.Functor (CategoryTheory.monadicAdjunction R).toMonad.Algebr …
    i : CategoryTheory.Iso ((K.comp (CategoryTheory.Monad.comparison (CategoryTheo …
    this✝² : CategoryTheory.Limits.PreservesColimit ((K.comp A).comp (CategoryTheo …
    this✝¹ : CategoryTheory.Limits.PreservesColimit (((K.comp A).comp (CategoryThe …
    this✝ : CategoryTheory.CreatesColimit (K.comp A) B := CategoryTheory.Monad.for …
    this : CategoryTheory.CreatesColimit K (A.comp B) := CategoryTheory.compCreate …
    ⊢ CategoryTheory.CreatesColimit K R
  -/
  let e := Monad.comparisonForget (monadicAdjunction R)
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.MonadicRightAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesColimit (K.comp R) ((CategoryTheory.mo …
    inst✝ : CategoryTheory.Limits.PreservesColimit ((K.comp R).comp ((CategoryTheo …
    A : CategoryTheory.Functor D (CategoryTheory.monadicAdjunction R).toMonad.Alge …
    B : CategoryTheory.Functor (CategoryTheory.monadicAdjunction R).toMonad.Algebr …
    i : CategoryTheory.Iso ((K.comp (CategoryTheory.Monad.comparison (CategoryTheo …
    this✝² : CategoryTheory.Limits.PreservesColimit ((K.comp A).comp (CategoryTheo …
    this✝¹ : CategoryTheory.Limits.PreservesColimit (((K.comp A).comp (CategoryThe …
    this✝ : CategoryTheory.CreatesColimit (K.comp A) B := CategoryTheory.Monad.for …
    this : CategoryTheory.CreatesColimit K (A.comp B) := CategoryTheory.compCreate …
    e : CategoryTheory.Iso ((CategoryTheory.Monad.comparison (CategoryTheory.monad …
    ⊢ CategoryTheory.CreatesColimit K R
  -/
  apply createsColimitOfNatIso e
  /-
    🎉 no goals
  -/


/-- A monadic functor creates any colimits of shapes it preserves. -/
noncomputable def monadicCreatesColimitsOfShapeOfPreservesColimitsOfShape (R : D ⥤ C)
    [MonadicRightAdjoint R] [PreservesColimitsOfShape J R] : CreatesColimitsOfShape J R :=
  letI : PreservesColimitsOfShape J (monadicLeftAdjoint R) := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      R : CategoryTheory.Functor D C
      inst✝¹ : CategoryTheory.MonadicRightAdjoint R
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J R
      ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J (CategoryTheory.monadicLeft …
    -/
    apply (Adjunction.leftAdjoint_preservesColimits (monadicAdjunction R)).1
    /-
      🎉 no goals
    -/
  letI : PreservesColimitsOfShape J (monadicLeftAdjoint R ⋙ R) := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      R : CategoryTheory.Functor D C
      inst✝¹ : CategoryTheory.MonadicRightAdjoint R
      inst✝ : CategoryTheory.Limits.PreservesColimitsOfShape J R
      this : CategoryTheory.Limits.PreservesColimitsOfShape J (CategoryTheory.monadi …
      ⊢ CategoryTheory.Limits.PreservesColimitsOfShape J ((CategoryTheory.monadicLef …
    -/
    apply CategoryTheory.Limits.comp_preservesColimitsOfShape _ _
    /-
      🎉 no goals
    -/
  ⟨monadicCreatesColimitOfPreservesColimit _ _⟩


/-- A monadic functor creates colimits if it preserves colimits. -/
noncomputable def monadicCreatesColimitsOfPreservesColimits (R : D ⥤ C) [MonadicRightAdjoint R]
    [PreservesColimitsOfSize.{v, u} R] : CreatesColimitsOfSize.{v, u} R where
  CreatesColimitsOfShape :=
    monadicCreatesColimitsOfShapeOfPreservesColimitsOfShape _


theorem hasLimit_of_reflective (F : J ⥤ D) (R : D ⥤ C) [HasLimit (F ⋙ R)] [Reflective R] :
    HasLimit F :=
  haveI := monadicCreatesLimits.{v, u} R
  hasLimit_of_created F R


/-- If `C` has limits of shape `J` then any reflective subcategory has limits of shape `J`. -/
theorem hasLimitsOfShape_of_reflective [HasLimitsOfShape J C] (R : D ⥤ C) [Reflective R] :
    HasLimitsOfShape J D :=
  ⟨fun F => hasLimit_of_reflective F R⟩


/-- If `C` has limits then any reflective subcategory has limits. -/
theorem hasLimits_of_reflective (R : D ⥤ C) [HasLimitsOfSize.{v, u} C] [Reflective R] :
    HasLimitsOfSize.{v, u} D :=
  ⟨fun _ => hasLimitsOfShape_of_reflective R⟩


/-- If `C` has colimits of shape `J` then any reflective subcategory has colimits of shape `J`. -/
theorem hasColimitsOfShape_of_reflective (R : D ⥤ C) [Reflective R] [HasColimitsOfShape J C] :
    HasColimitsOfShape J D where
  has_colimit := fun F => by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        J : Type u
        inst✝² : CategoryTheory.Category.{v, u} J
        R : CategoryTheory.Functor D C
        inst✝¹ : CategoryTheory.Reflective R
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        F : CategoryTheory.Functor J D
        ⊢ CategoryTheory.Limits.HasColimit F
      -/
      let c := (monadicLeftAdjoint R).mapCocone (colimit.cocone (F ⋙ R))
      letI : PreservesColimitsOfShape J _ :=
        (monadicAdjunction R).leftAdjoint_preservesColimits.1
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        J : Type u
        inst✝² : CategoryTheory.Category.{v, u} J
        R : CategoryTheory.Functor D C
        inst✝¹ : CategoryTheory.Reflective R
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone ((F.comp R).comp (CategoryTheory.monadicLeftA …
        this : CategoryTheory.Limits.PreservesColimitsOfShape J (CategoryTheory.monadi …
        ⊢ CategoryTheory.Limits.HasColimit F
      -/
      let t : IsColimit c := isColimitOfPreserves (monadicLeftAdjoint R) (colimit.isColimit _)
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        J : Type u
        inst✝² : CategoryTheory.Category.{v, u} J
        R : CategoryTheory.Functor D C
        inst✝¹ : CategoryTheory.Reflective R
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cocone ((F.comp R).comp (CategoryTheory.monadicLeftA …
        this : CategoryTheory.Limits.PreservesColimitsOfShape J (CategoryTheory.monadi …
        t : CategoryTheory.Limits.IsColimit c := CategoryTheory.Limits.isColimitOfPres …
        ⊢ CategoryTheory.Limits.HasColimit F
      -/
      apply HasColimit.mk ⟨_, (IsColimit.precomposeInvEquiv _ _).symm t⟩
      apply
        (isoWhiskerLeft F (asIso (monadicAdjunction R).counit) : _) ≪≫ F.rightUnitor


/-- If `C` has colimits then any reflective subcategory has colimits. -/
theorem hasColimits_of_reflective (R : D ⥤ C) [Reflective R] [HasColimitsOfSize.{v, u} C] :
    HasColimitsOfSize.{v, u} D :=
  ⟨fun _ => hasColimitsOfShape_of_reflective R⟩


/-- The reflector always preserves terminal objects. Note this in general doesn't apply to any other
limit.
-/
lemma leftAdjoint_preservesTerminal_of_reflective (R : D ⥤ C) [Reflective R] :
    PreservesLimitsOfShape (Discrete.{v} PEmpty) (monadicLeftAdjoint R) where
  preservesLimit {K} := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      R : CategoryTheory.Functor D C
      inst✝ : CategoryTheory.Reflective R
      K : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{v + 1}) C
      ⊢ CategoryTheory.Limits.PreservesLimit K (CategoryTheory.monadicLeftAdjoint R)
    -/
    let F := Functor.empty.{v} D
    letI : PreservesLimit (F ⋙ R) (monadicLeftAdjoint R) := by
      constructor
      intro c h
      haveI : HasLimit (F ⋙ R) := ⟨⟨⟨c, h⟩⟩⟩
      haveI : HasLimit F := hasLimit_of_reflective F R
      constructor
      apply isLimitChangeEmptyCone D (limit.isLimit F)
      apply (asIso ((monadicAdjunction R).counit.app _)).symm.trans
      apply (monadicLeftAdjoint R).mapIso
      letI := monadicCreatesLimits.{v, v} R
      let A := CategoryTheory.preservesLimit_of_createsLimit_and_hasLimit F R
      apply (isLimitOfPreserves _ (limit.isLimit F)).conePointUniqueUpToIso h
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      R : CategoryTheory.Functor D C
      inst✝ : CategoryTheory.Reflective R
      K : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{v + 1}) C
      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{v + 1}) D := Categ …
      this : CategoryTheory.Limits.PreservesLimit (F.comp R) (CategoryTheory.monadic …
        {
          preserves := fun {c} h =>
            Nonempty.intro
              (CategoryTheory.Limits.isLimitChangeEmptyCone D (CategoryTheory.Limits …
                ((CategoryTheory.asIso ((CategoryTheory.monadicAdjunction R).counit. …
                  ((CategoryTheory.monadicLeftAdjoint R).mapIso
                    (let A := CategoryTheory.preservesLimit_of_createsLimit_and_hasL …
                    (CategoryTheory.Limits.isLimitOfPreserves R (CategoryTheory.Limi …
      ⊢ CategoryTheory.Limits.PreservesLimit K (CategoryTheory.monadicLeftAdjoint R)
    -/
    apply preservesLimit_of_iso_diagram _ (Functor.emptyExt (F ⋙ R) _)
    /-
      🎉 no goals
    -/


/-- (Impl) The natural transformation used to define the new cocone -/
@[simps]
def γ : D ⋙ T.forget ⟶ D ⋙ T.forget ⋙ ↑T  where app j := (D.obj j).a


/-- (Impl) This new cocone is used to construct the coalgebra structure -/
@[simps! ι_app]
def newCocone : Cocone (D ⋙ forget T) where
  pt := T.obj c.pt
  ι := γ D ≫ whiskerRight c.ι (T : C ⥤ C) ≫ (Functor.constComp J _ (T : C ⥤ C)).hom


/-- The coalgebra structure which will be the point of the new colimit cone for `D`. -/
@[simps]
def coconePoint : Coalgebra T where
  A := c.pt
  a := t.desc (newCocone D c)
  counit := t.hom_ext fun j ↦ by
    simp only [Functor.comp_obj, forget_obj, Functor.id_obj, Functor.const_obj_obj,
      IsColimit.fac_assoc, newCocone_ι_app, assoc, NatTrans.naturality, Functor.id_map, comp_id]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
      J : Type u
      inst✝ : CategoryTheory.Category.{v, u} J
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J T.Coalgebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.obj j).a (CategoryTheory.CategoryS …
    -/
    rw [← Category.assoc, (D.obj j).counit, Category.id_comp]
    /-
      🎉 no goals
    -/
  coassoc := t.hom_ext fun j ↦ by
    simp only [Functor.comp_obj, forget_obj, Functor.const_obj_obj, IsColimit.fac_assoc,
      newCocone_ι_app, assoc, NatTrans.naturality, Functor.comp_map]
    rw [← Category.assoc, (D.obj j).coassoc, ← Functor.map_comp, t.fac (newCocone D c) j,
      newCocone_ι_app, Functor.map_comp, assoc]


/-- (Impl) Construct the lifted cocone in `Coalgebra T` which will be colimiting. -/
@[simps]
def liftedCocone : Cocone D where
  pt := coconePoint D c t
  ι :=
    { app := fun j => { f := c.ι.app j }
      naturality := fun X Y f => by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D✝ : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          T : CategoryTheory.Comonad C
          D : CategoryTheory.Functor J T.Coalgebra
          c : CategoryTheory.Limits.Cocone (D.comp T.forget)
          t : CategoryTheory.Limits.IsColimit c
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f) ((fun j => { f := c.ι.app j …
        -/
        ext1
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D✝ : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          T : CategoryTheory.Comonad C
          D : CategoryTheory.Functor J T.Coalgebra
          c : CategoryTheory.Limits.Cocone (D.comp T.forget)
          t : CategoryTheory.Limits.IsColimit c
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f) ((fun j => { f := c.ι.app j …
        -/
        dsimp
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D✝ : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          T : CategoryTheory.Comonad C
          D : CategoryTheory.Functor J T.Coalgebra
          c : CategoryTheory.Limits.Cocone (D.comp T.forget)
          t : CategoryTheory.Limits.IsColimit c
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.map f).f (c.ι.app Y)) (CategoryThe …
        -/
        erw [c.w f]
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D✝ : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
          J : Type u
          inst✝ : CategoryTheory.Category.{v, u} J
          T : CategoryTheory.Comonad C
          D : CategoryTheory.Functor J T.Coalgebra
          c : CategoryTheory.Limits.Cocone (D.comp T.forget)
          t : CategoryTheory.Limits.IsColimit c
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (c.ι.app X) (CategoryTheory.CategoryStruct.comp (c.ι.app X) (CategoryTheo …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- (Impl) Prove that the lifted cocone is colimiting. -/
@[simps]
def liftedCoconeIsColimit : IsColimit (liftedCocone D c t) where
  desc s :=
    { f := t.desc ((forget T).mapCocone s)
      h :=
        t.hom_ext fun j => by
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            D✝ : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
            J : Type u
            inst✝ : CategoryTheory.Category.{v, u} J
            T : CategoryTheory.Comonad C
            D : CategoryTheory.Functor J T.Coalgebra
            c : CategoryTheory.Limits.Cocone (D.comp T.forget)
            t : CategoryTheory.Limits.IsColimit c
            s : CategoryTheory.Limits.Cocone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) (CategoryTheory.CategoryS …
          -/
          dsimp
          rw [← Category.assoc, ← Category.assoc, t.fac, newCocone_ι_app, t.fac,
            Functor.mapCocone_ι_app, Category.assoc, ← Functor.map_comp, t.fac]
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            D✝ : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
            J : Type u
            inst✝ : CategoryTheory.Category.{v, u} J
            T : CategoryTheory.Comonad C
            D : CategoryTheory.Functor J T.Coalgebra
            c : CategoryTheory.Limits.Cocone (D.comp T.forget)
            t : CategoryTheory.Limits.IsColimit c
            s : CategoryTheory.Limits.Cocone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.obj j).a (T.map ((T.forget.mapCoco …
          -/
          apply (s.ι.app j).h }
          /-
            🎉 no goals
          -/
  uniq s m J := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
      J✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} J✝
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J✝ T.Coalgebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D
      m : Quiver.Hom (CategoryTheory.Comonad.ForgetCreatesColimits'.liftedCocone D c …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comona …
      ⊢ Eq m ((fun s => { f := t.desc (T.forget.mapCocone s), h := ⋯ }) s)
    -/
    ext1
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
      J✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} J✝
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J✝ T.Coalgebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D
      m : Quiver.Hom (CategoryTheory.Comonad.ForgetCreatesColimits'.liftedCocone D c …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comona …
      ⊢ Eq m.f ((fun s => { f := t.desc (T.forget.mapCocone s), h := ⋯ }) s).f
    -/
    apply t.hom_ext
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
      J✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} J✝
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J✝ T.Coalgebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D
      m : Quiver.Hom (CategoryTheory.Comonad.ForgetCreatesColimits'.liftedCocone D c …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comona …
      ⊢ ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m.f) (Categor …
    -/
    intro j
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D✝
      J✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} J✝
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J✝ T.Coalgebra
      c : CategoryTheory.Limits.Cocone (D.comp T.forget)
      t : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D
      m : Quiver.Hom (CategoryTheory.Comonad.ForgetCreatesColimits'.liftedCocone D c …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Comona …
      j : J✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m.f) (CategoryTheory.Cate …
    -/
    simpa [t.fac ((forget T).mapCocone s) j] using congr_arg Coalgebra.Hom.f (J j)
    /-
      🎉 no goals
    -/


/-- The forgetful functor from the Eilenberg-Moore category creates colimits. -/
noncomputable instance forgetCreatesColimit : CreatesColimitsOfSize (forget T) where
  CreatesColimitsOfShape := {
    CreatesColimit := fun {D} =>
      createsColimitOfReflectsIso fun c t =>
        { liftedCocone := ForgetCreatesColimits'.liftedCocone D c t
          validLift := Cocones.ext (Iso.refl _) fun _ => (comp_id _)
          makesColimit := ForgetCreatesColimits'.liftedCoconeIsColimit _ _ _ } }


/-- If `D ⋙ forget T` has a colimit, then `D` has a colimit. -/
theorem hasColimit_of_comp_forget_hasColimit (D : J ⥤ Coalgebra T) [HasColimit (D ⋙ forget T)] :
    HasColimit D :=
  hasColimit_of_created D (forget T)


/-- (Impl)
The natural transformation given by the coalgebra structure maps, used to construct a cone `c` with
point `limit (D ⋙ forget T)`.
 -/
@[simps]
def γ : D ⋙ forget T ⟶ (D ⋙ forget T) ⋙ ↑T where app j := (D.obj j).a


/-- (Impl)
A cone for the diagram `(D ⋙ forget T) ⋙ T` found by composing the natural transformation `γ`
with the limiting cone for `D ⋙ forget T`.
-/
@[simps]
def newCone : Cone ((D ⋙ forget T) ⋙ (T : C ⥤ C)) where
  pt := c.pt
  π := c.π ≫ γ


/-- (Impl)
Define the map `λ : L ⟶ TL`, which will serve as the structure of the algebra on `L`, and
we will show is the limiting object. We use the cone constructed by `c` and the fact that
`T` preserves limits to produce this morphism.
-/
noncomputable abbrev lambda : c.pt ⟶ ((T : C ⥤ C).mapCone c).pt :=
  (isLimitOfPreserves _ t).lift (newCone c)


/-- (Impl) The key property defining the map `λ : L ⟶ TL`. -/
theorem commuting (j : J) : lambda c t ≫ (T : C ⥤ C).map (c.π.app j) = c.π.app j ≫ (D.obj j).a :=
  (isLimitOfPreserves _ t).fac (newCone c) j


/-- (Impl)
Construct the limiting coalgebra from the map `λ : L ⟶ TL` given by `lambda`. We are required to
show it satisfies the two coalgebra laws, which follow from the coalgebra laws for the image of `D`
and our `commuting` lemma.
-/
@[simps]
noncomputable def conePoint : Coalgebra T where
  A := c.pt
  a := lambda c t
  counit := t.hom_ext fun j ↦ by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [assoc, ← show _ = _ ≫ c.π.app j from T.ε.naturality _, ← assoc, commuting, assoc]
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app j) (CategoryTheory.CategoryS …
    -/
    simp [Coalgebra.counit (D.obj j)]
    /-
      🎉 no goals
    -/
  coassoc := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Comonad.ForgetCreates …
    -/
    refine (isLimitOfPreserves _ (isLimitOfPreserves _ t)).hom_ext fun j => ?_
    rw [Functor.mapCone_π_app, Functor.mapCone_π_app, assoc,
      ← show _ = _ ≫ T.map (T.map _) from T.δ.naturality _, assoc, ← Functor.map_comp, commuting,
      Functor.map_comp, ← assoc, commuting]
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Functor.comp_obj, forget_obj, Functor.const_obj_obj, assoc]
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app j) (CategoryTheory.CategoryS …
    -/
    rw [(D.obj j).coassoc,  ← assoc, ← assoc, commuting]
    /-
      🎉 no goals
    -/


/-- (Impl) Construct the lifted cone in `Coalgebra T` which will be limiting. -/
@[simps]
noncomputable def liftedCone : Cone D where
  pt := conePoint c t
  π :=
    { app := fun j =>
        { f := c.π.app j
          h := commuting _ _ _ }
      naturality := fun A B f => by
        /-
          C : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
          D✝ : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
          J : Type u
          inst✝³ : CategoryTheory.Category.{v, u} J
          T : CategoryTheory.Comonad C
          D : CategoryTheory.Functor J T.Coalgebra
          c : CategoryTheory.Limits.Cone (D.comp T.forget)
          t : CategoryTheory.Limits.IsLimit c
          inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
          inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
          inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
          A B : J
          f : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        ext1
        /-
          case h
          C : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
          D✝ : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
          J : Type u
          inst✝³ : CategoryTheory.Category.{v, u} J
          T : CategoryTheory.Comonad C
          D : CategoryTheory.Functor J T.Coalgebra
          c : CategoryTheory.Limits.Cone (D.comp T.forget)
          t : CategoryTheory.Limits.IsLimit c
          inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
          inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
          inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
          A B : J
          f : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        dsimp
        /-
          case h
          C : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
          D✝ : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
          J : Type u
          inst✝³ : CategoryTheory.Category.{v, u} J
          T : CategoryTheory.Comonad C
          D : CategoryTheory.Functor J T.Coalgebra
          c : CategoryTheory.Limits.Cone (D.comp T.forget)
          t : CategoryTheory.Limits.IsLimit c
          inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
          inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
          inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
          A B : J
          f : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
        -/
        rw [id_comp, ← c.w]
        /-
          case h
          C : Type u₁
          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
          D✝ : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
          J : Type u
          inst✝³ : CategoryTheory.Category.{v, u} J
          T : CategoryTheory.Comonad C
          D : CategoryTheory.Functor J T.Coalgebra
          c : CategoryTheory.Limits.Cone (D.comp T.forget)
          t : CategoryTheory.Limits.IsLimit c
          inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
          inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
          inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
          A B : J
          f : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.π.app ?m.211487) ((D.comp T.forget …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- (Impl) Prove that the lifted cone is limiting. -/
@[simps]
noncomputable def liftedConeIsLimit : IsLimit (liftedCone c t) where
  lift s :=
    { f := t.lift ((forget T).mapCone s)
      h :=
        (isLimitOfPreserves (T : C ⥤ C) t).hom_ext fun j => by
          /-
            C : Type u₁
            inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
            D✝ : Type u₂
            inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
            J : Type u
            inst✝³ : CategoryTheory.Category.{v, u} J
            T : CategoryTheory.Comonad C
            D : CategoryTheory.Functor J T.Coalgebra
            c : CategoryTheory.Limits.Cone (D.comp T.forget)
            t : CategoryTheory.Limits.IsLimit c
            inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
            inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
            inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
            s : CategoryTheory.Limits.Cone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
          -/
          dsimp
          rw [Category.assoc, ← t.fac, Category.assoc, t.fac, commuting, ← assoc, ← assoc, t.fac,
            assoc, ← Functor.map_comp, t.fac]
          /-
            C : Type u₁
            inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
            D✝ : Type u₂
            inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
            J : Type u
            inst✝³ : CategoryTheory.Category.{v, u} J
            T : CategoryTheory.Comonad C
            D : CategoryTheory.Functor J T.Coalgebra
            c : CategoryTheory.Limits.Cone (D.comp T.forget)
            t : CategoryTheory.Limits.IsLimit c
            inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
            inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
            inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
            s : CategoryTheory.Limits.Cone D
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp s.pt.a (T.map ((T.forget.mapCone s).π …
          -/
          exact (s.π.app j).h }
          /-
            🎉 no goals
          -/
  uniq s m J := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J✝ : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J✝
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J✝ T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      s : CategoryTheory.Limits.Cone D
      m : Quiver.Hom s.pt (CategoryTheory.Comonad.ForgetCreatesLimits'.liftedCone c  …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Como …
      ⊢ Eq m ((fun s => { f := t.lift (T.forget.mapCone s), h := ⋯ }) s)
    -/
    ext1
    /-
      case h
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J✝ : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J✝
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J✝ T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      s : CategoryTheory.Limits.Cone D
      m : Quiver.Hom s.pt (CategoryTheory.Comonad.ForgetCreatesLimits'.liftedCone c  …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Como …
      ⊢ Eq m.f ((fun s => { f := t.lift (T.forget.mapCone s), h := ⋯ }) s).f
    -/
    apply t.hom_ext
    /-
      case h
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J✝ : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J✝
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J✝ T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      s : CategoryTheory.Limits.Cone D
      m : Quiver.Hom s.pt (CategoryTheory.Comonad.ForgetCreatesLimits'.liftedCone c  …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Como …
      ⊢ ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m.f (c.π.app j)) (Categor …
    -/
    intro j
    /-
      case h
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      D✝ : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D✝
      J✝ : Type u
      inst✝³ : CategoryTheory.Category.{v, u} J✝
      T : CategoryTheory.Comonad C
      D : CategoryTheory.Functor J✝ T.Coalgebra
      c : CategoryTheory.Limits.Cone (D.comp T.forget)
      t : CategoryTheory.Limits.IsLimit c
      inst✝² : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
      inst✝¹ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunc …
      inst✝ : CategoryTheory.Limits.PreservesColimit ((D.comp T.forget).comp T.toFun …
      s : CategoryTheory.Limits.Cone D
      m : Quiver.Hom s.pt (CategoryTheory.Comonad.ForgetCreatesLimits'.liftedCone c  …
      J : ∀ (j : J✝), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Como …
      j : J✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m.f (c.π.app j)) (CategoryTheory.Cate …
    -/
    simpa using congr_arg Coalgebra.Hom.f (J j)
    /-
      🎉 no goals
    -/


/-- The forgetful functor from the Eilenberg-Moore category for a comonad creates any limit
which the comonad itself preserves.
-/
noncomputable instance forgetCreatesLimit (D : J ⥤ Coalgebra T)
    [PreservesLimit (D ⋙ forget T) (T : C ⥤ C)]
    [PreservesLimit ((D ⋙ forget T) ⋙ ↑T) (T : C ⥤ C)] : CreatesLimit D (forget T) :=
  createsLimitOfReflectsIso fun c t =>
    { liftedCone :=
        { pt := conePoint c t
          π :=
            { app := fun j =>
                { f := c.π.app j
                  h := commuting _ _ _ }
              naturality := fun A B f => by
                /-
                  C : Type u₁
                  inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                  D✝ : Type u₂
                  inst✝³ : CategoryTheory.Category.{v₂, u₂} D✝
                  J : Type u
                  inst✝² : CategoryTheory.Category.{v, u} J
                  T : CategoryTheory.Comonad C
                  D : CategoryTheory.Functor J T.Coalgebra
                  inst✝¹ : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
                  inst✝ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunct …
                  c : CategoryTheory.Limits.Cone (D.comp T.forget)
                  t : CategoryTheory.Limits.IsLimit c
                  A B : J
                  f : Quiver.Hom A B
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                -/
                ext1
                /-
                  case h
                  C : Type u₁
                  inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                  D✝ : Type u₂
                  inst✝³ : CategoryTheory.Category.{v₂, u₂} D✝
                  J : Type u
                  inst✝² : CategoryTheory.Category.{v, u} J
                  T : CategoryTheory.Comonad C
                  D : CategoryTheory.Functor J T.Coalgebra
                  inst✝¹ : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
                  inst✝ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunct …
                  c : CategoryTheory.Limits.Cone (D.comp T.forget)
                  t : CategoryTheory.Limits.IsLimit c
                  A B : J
                  f : Quiver.Hom A B
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                -/
                dsimp
                /-
                  case h
                  C : Type u₁
                  inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                  D✝ : Type u₂
                  inst✝³ : CategoryTheory.Category.{v₂, u₂} D✝
                  J : Type u
                  inst✝² : CategoryTheory.Category.{v, u} J
                  T : CategoryTheory.Comonad C
                  D : CategoryTheory.Functor J T.Coalgebra
                  inst✝¹ : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
                  inst✝ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunct …
                  c : CategoryTheory.Limits.Cone (D.comp T.forget)
                  t : CategoryTheory.Limits.IsLimit c
                  A B : J
                  f : Quiver.Hom A B
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id c.p …
                -/
                erw [id_comp, c.w] } }
                /-
                  🎉 no goals
                -/
                   /-
                     C : Type u₁
                     inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                     D✝ : Type u₂
                     inst✝³ : CategoryTheory.Category.{v₂, u₂} D✝
                     J : Type u
                     inst✝² : CategoryTheory.Category.{v, u} J
                     T : CategoryTheory.Comonad C
                     D : CategoryTheory.Functor J T.Coalgebra
                     inst✝¹ : CategoryTheory.Limits.PreservesLimit (D.comp T.forget) T.toFunctor
                     inst✝ : CategoryTheory.Limits.PreservesLimit ((D.comp T.forget).comp T.toFunct …
                     c : CategoryTheory.Limits.Cone (D.comp T.forget)
                     t : CategoryTheory.Limits.IsLimit c
                     ⊢ ∀ (j : J), Eq ((T.forget.mapCone { pt := CategoryTheory.Comonad.ForgetCreate …
                   -/
      validLift := Cones.ext (Iso.refl _)
                   /-
                     🎉 no goals
                   -/
      makesLimit := liftedConeIsLimit _ _ }


noncomputable instance forgetCreatesLimitsOfShape [PreservesLimitsOfShape J (T : C ⥤ C)] :
                                                               /-
                                                                 C : Type u₁
                                                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                                 D : Type u₂
                                                                 inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                 J : Type u
                                                                 inst✝¹ : CategoryTheory.Category.{v, u} J
                                                                 T : CategoryTheory.Comonad C
                                                                 inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J T.toFunctor
                                                                 ⊢ {K : CategoryTheory.Functor J T.Coalgebra} → CategoryTheory.CreatesLimit K T …
                                                               -/
    CreatesLimitsOfShape J (forget T) where CreatesLimit := by infer_instance
                                                               /-
                                                                 🎉 no goals
                                                               -/


noncomputable instance forgetCreatesLimits [PreservesLimitsOfSize.{v, u} (T : C ⥤ C)] :
                                                                           /-
                                                                             C : Type u₁
                                                                             inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                                             D : Type u₂
                                                                             inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                             J : Type u
                                                                             inst✝¹ : CategoryTheory.Category.{v, u} J
                                                                             T : CategoryTheory.Comonad C
                                                                             inst✝ : CategoryTheory.Limits.PreservesLimitsOfSize.{v, u, v₁, v₁, u₁, u₁} T.t …
                                                                             ⊢ {J : Type u} → [inst : CategoryTheory.Category.{v, u} J] → CategoryTheory.Cr …
                                                                           -/
    CreatesLimitsOfSize.{v, u} (forget T) where CreatesLimitsOfShape := by infer_instance
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- For `D : J ⥤ Coalgebra T`, `D ⋙ forget T` has a limit, then `D` has a limit provided limits
of shape `J` are preserved by `T`.
-/
theorem forget_creates_limits_of_comonad_preserves [PreservesLimitsOfShape J (T : C ⥤ C)]
    (D : J ⥤ Coalgebra T) [HasLimit (D ⋙ forget T)] : HasLimit D :=
  hasLimit_of_created D (forget T)


instance comp_comparison_forget_hasColimit (F : J ⥤ D) (R : D ⥤ C) [ComonadicLeftAdjoint R]
    [HasColimit (F ⋙ R)] :
    HasColimit ((F ⋙ Comonad.comparison (comonadicAdjunction R)) ⋙ Comonad.forget _) :=
  @hasColimitOfIso _ _ _ _ (F ⋙ R) _ _
    (isoWhiskerLeft F (Comonad.comparisonForget (comonadicAdjunction R)).symm)


instance comp_comparison_hasColimit (F : J ⥤ D) (R : D ⥤ C) [ComonadicLeftAdjoint R]
    [HasColimit (F ⋙ R)] : HasColimit (F ⋙ Comonad.comparison (comonadicAdjunction R)) :=
  Comonad.hasColimit_of_comp_forget_hasColimit (F ⋙ Comonad.comparison (comonadicAdjunction R))


/-- Any comonadic functor creates colimits. -/
noncomputable def comonadicCreatesColimits (R : D ⥤ C) [ComonadicLeftAdjoint R] :
    CreatesColimitsOfSize.{v, u} R :=
  createsColimitsOfNatIso (Comonad.comparisonForget (comonadicAdjunction R))


/-- The forgetful functor from the Eilenberg-Moore category for a comonad creates any limit
which the comonad itself preserves.
-/
noncomputable def comonadicCreatesLimitOfPreservesLimit (R : D ⥤ C) (K : J ⥤ D)
    [ComonadicLeftAdjoint R] [PreservesLimit (K ⋙ R) (comonadicRightAdjoint R ⋙ R)]
    [PreservesLimit ((K ⋙ R) ⋙ comonadicRightAdjoint R ⋙ R) (comonadicRightAdjoint R ⋙ R)] :
      CreatesLimit K R := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.ComonadicLeftAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (K.comp R) ((CategoryTheory.como …
    inst✝ : CategoryTheory.Limits.PreservesLimit ((K.comp R).comp ((CategoryTheory …
    ⊢ CategoryTheory.CreatesLimit K R
  -/
  letI A := Comonad.comparison (comonadicAdjunction R)
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.ComonadicLeftAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (K.comp R) ((CategoryTheory.como …
    inst✝ : CategoryTheory.Limits.PreservesLimit ((K.comp R).comp ((CategoryTheory …
    A : CategoryTheory.Functor D (CategoryTheory.comonadicAdjunction R).toComonad. …
    ⊢ CategoryTheory.CreatesLimit K R
  -/
  letI B := Comonad.forget (Adjunction.toComonad (comonadicAdjunction R))
  let i : (K ⋙ Comonad.comparison (comonadicAdjunction R)) ⋙ Comonad.forget _ ≅ K ⋙ R :=
    Functor.associator _ _ _ ≪≫
      isoWhiskerLeft K (Comonad.comparisonForget (comonadicAdjunction R))
  letI : PreservesLimit ((K ⋙ A) ⋙ Comonad.forget
    (Adjunction.toComonad (comonadicAdjunction R)))
      (Adjunction.toComonad (comonadicAdjunction R)).toFunctor := by
    dsimp
    exact preservesLimit_of_iso_diagram _ i.symm
  letI : PreservesLimit
    (((K ⋙ A) ⋙ Comonad.forget (Adjunction.toComonad (comonadicAdjunction R))) ⋙
      (Adjunction.toComonad (comonadicAdjunction R)).toFunctor)
      (Adjunction.toComonad (comonadicAdjunction R)).toFunctor := by
    dsimp
    exact preservesLimit_of_iso_diagram _ (isoWhiskerRight i (comonadicRightAdjoint R ⋙ R)).symm
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.ComonadicLeftAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (K.comp R) ((CategoryTheory.como …
    inst✝ : CategoryTheory.Limits.PreservesLimit ((K.comp R).comp ((CategoryTheory …
    A : CategoryTheory.Functor D (CategoryTheory.comonadicAdjunction R).toComonad. …
    B : CategoryTheory.Functor (CategoryTheory.comonadicAdjunction R).toComonad.Co …
    i : CategoryTheory.Iso ((K.comp (CategoryTheory.Comonad.comparison (CategoryTh …
    this✝ : CategoryTheory.Limits.PreservesLimit ((K.comp A).comp (CategoryTheory. …
    this : CategoryTheory.Limits.PreservesLimit (((K.comp A).comp (CategoryTheory. …
    ⊢ CategoryTheory.CreatesLimit K R
  -/
  letI : CreatesLimit (K ⋙ A) B := CategoryTheory.Comonad.forgetCreatesLimit _
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.ComonadicLeftAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (K.comp R) ((CategoryTheory.como …
    inst✝ : CategoryTheory.Limits.PreservesLimit ((K.comp R).comp ((CategoryTheory …
    A : CategoryTheory.Functor D (CategoryTheory.comonadicAdjunction R).toComonad. …
    B : CategoryTheory.Functor (CategoryTheory.comonadicAdjunction R).toComonad.Co …
    i : CategoryTheory.Iso ((K.comp (CategoryTheory.Comonad.comparison (CategoryTh …
    this✝¹ : CategoryTheory.Limits.PreservesLimit ((K.comp A).comp (CategoryTheory …
    this✝ : CategoryTheory.Limits.PreservesLimit (((K.comp A).comp (CategoryTheory …
    this : CategoryTheory.CreatesLimit (K.comp A) B := CategoryTheory.Comonad.forg …
    ⊢ CategoryTheory.CreatesLimit K R
  -/
  letI : CreatesLimit K (A ⋙ B) := CategoryTheory.compCreatesLimit _ _
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.ComonadicLeftAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (K.comp R) ((CategoryTheory.como …
    inst✝ : CategoryTheory.Limits.PreservesLimit ((K.comp R).comp ((CategoryTheory …
    A : CategoryTheory.Functor D (CategoryTheory.comonadicAdjunction R).toComonad. …
    B : CategoryTheory.Functor (CategoryTheory.comonadicAdjunction R).toComonad.Co …
    i : CategoryTheory.Iso ((K.comp (CategoryTheory.Comonad.comparison (CategoryTh …
    this✝² : CategoryTheory.Limits.PreservesLimit ((K.comp A).comp (CategoryTheory …
    this✝¹ : CategoryTheory.Limits.PreservesLimit (((K.comp A).comp (CategoryTheor …
    this✝ : CategoryTheory.CreatesLimit (K.comp A) B := CategoryTheory.Comonad.for …
    this : CategoryTheory.CreatesLimit K (A.comp B) := CategoryTheory.compCreatesL …
    ⊢ CategoryTheory.CreatesLimit K R
  -/
  let e := Comonad.comparisonForget (comonadicAdjunction R)
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    J : Type u
    inst✝³ : CategoryTheory.Category.{v, u} J
    R : CategoryTheory.Functor D C
    K : CategoryTheory.Functor J D
    inst✝² : CategoryTheory.ComonadicLeftAdjoint R
    inst✝¹ : CategoryTheory.Limits.PreservesLimit (K.comp R) ((CategoryTheory.como …
    inst✝ : CategoryTheory.Limits.PreservesLimit ((K.comp R).comp ((CategoryTheory …
    A : CategoryTheory.Functor D (CategoryTheory.comonadicAdjunction R).toComonad. …
    B : CategoryTheory.Functor (CategoryTheory.comonadicAdjunction R).toComonad.Co …
    i : CategoryTheory.Iso ((K.comp (CategoryTheory.Comonad.comparison (CategoryTh …
    this✝² : CategoryTheory.Limits.PreservesLimit ((K.comp A).comp (CategoryTheory …
    this✝¹ : CategoryTheory.Limits.PreservesLimit (((K.comp A).comp (CategoryTheor …
    this✝ : CategoryTheory.CreatesLimit (K.comp A) B := CategoryTheory.Comonad.for …
    this : CategoryTheory.CreatesLimit K (A.comp B) := CategoryTheory.compCreatesL …
    e : CategoryTheory.Iso ((CategoryTheory.Comonad.comparison (CategoryTheory.com …
    ⊢ CategoryTheory.CreatesLimit K R
  -/
  apply createsLimitOfNatIso e
  /-
    🎉 no goals
  -/


/-- A comonadic functor creates any limits of shapes it preserves. -/
noncomputable def comonadicCreatesLimitsOfShapeOfPreservesLimitsOfShape (R : D ⥤ C)
    [ComonadicLeftAdjoint R] [PreservesLimitsOfShape J R] : CreatesLimitsOfShape J R :=
  letI : PreservesLimitsOfShape J (comonadicRightAdjoint R) := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      R : CategoryTheory.Functor D C
      inst✝¹ : CategoryTheory.ComonadicLeftAdjoint R
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J R
      ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J (CategoryTheory.comonadicRigh …
    -/
    apply (Adjunction.rightAdjoint_preservesLimits (comonadicAdjunction R)).1
    /-
      🎉 no goals
    -/
  letI : PreservesLimitsOfShape J (comonadicRightAdjoint R ⋙ R) := by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
      J : Type u
      inst✝² : CategoryTheory.Category.{v, u} J
      R : CategoryTheory.Functor D C
      inst✝¹ : CategoryTheory.ComonadicLeftAdjoint R
      inst✝ : CategoryTheory.Limits.PreservesLimitsOfShape J R
      this : CategoryTheory.Limits.PreservesLimitsOfShape J (CategoryTheory.comonadi …
      ⊢ CategoryTheory.Limits.PreservesLimitsOfShape J ((CategoryTheory.comonadicRig …
    -/
    apply CategoryTheory.Limits.comp_preservesLimitsOfShape _ _
    /-
      🎉 no goals
    -/
  ⟨comonadicCreatesLimitOfPreservesLimit _ _⟩


/-- A comonadic functor creates limits if it preserves limits. -/
noncomputable def comonadicCreatesLimitsOfPreservesLimits (R : D ⥤ C) [ComonadicLeftAdjoint R]
    [PreservesLimitsOfSize.{v, u} R] : CreatesLimitsOfSize.{v, u} R where
  CreatesLimitsOfShape :=
    comonadicCreatesLimitsOfShapeOfPreservesLimitsOfShape _


theorem hasColimit_of_coreflective (F : J ⥤ D) (R : D ⥤ C) [HasColimit (F ⋙ R)] [Coreflective R] :
    HasColimit F :=
  haveI := comonadicCreatesColimits.{v, u} R
  hasColimit_of_created F R


/-- If `C` has colimits of shape `J` then any coreflective subcategory has colimits of shape `J`. -/
theorem hasColimitsOfShape_of_coreflective [HasColimitsOfShape J C] (R : D ⥤ C) [Coreflective R] :
    HasColimitsOfShape J D :=
  ⟨fun F => hasColimit_of_coreflective F R⟩


/-- If `C` has colimits then any coreflective subcategory has colimits. -/
theorem hasColimits_of_coreflective (R : D ⥤ C) [HasColimitsOfSize.{v, u} C] [Coreflective R] :
    HasColimitsOfSize.{v, u} D :=
  ⟨fun _ => hasColimitsOfShape_of_coreflective R⟩


/-- If `C` has limits of shape `J` then any coreflective subcategory has limits of shape `J`. -/
theorem hasLimitsOfShape_of_coreflective (R : D ⥤ C) [Coreflective R] [HasLimitsOfShape J C] :
    HasLimitsOfShape J D where
  has_limit := fun F => by
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        J : Type u
        inst✝² : CategoryTheory.Category.{v, u} J
        R : CategoryTheory.Functor D C
        inst✝¹ : CategoryTheory.Coreflective R
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
        F : CategoryTheory.Functor J D
        ⊢ CategoryTheory.Limits.HasLimit F
      -/
      let c := (comonadicRightAdjoint R).mapCone (limit.cone (F ⋙ R))
      letI : PreservesLimitsOfShape J _ :=
        (comonadicAdjunction R).rightAdjoint_preservesLimits.1
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        J : Type u
        inst✝² : CategoryTheory.Category.{v, u} J
        R : CategoryTheory.Functor D C
        inst✝¹ : CategoryTheory.Coreflective R
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cone ((F.comp R).comp (CategoryTheory.comonadicRight …
        this : CategoryTheory.Limits.PreservesLimitsOfShape J (CategoryTheory.comonadi …
        ⊢ CategoryTheory.Limits.HasLimit F
      -/
      let t : IsLimit c := isLimitOfPreserves (comonadicRightAdjoint R) (limit.isLimit _)
      /-
        C : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} D
        J : Type u
        inst✝² : CategoryTheory.Category.{v, u} J
        R : CategoryTheory.Functor D C
        inst✝¹ : CategoryTheory.Coreflective R
        inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
        F : CategoryTheory.Functor J D
        c : CategoryTheory.Limits.Cone ((F.comp R).comp (CategoryTheory.comonadicRight …
        this : CategoryTheory.Limits.PreservesLimitsOfShape J (CategoryTheory.comonadi …
        t : CategoryTheory.Limits.IsLimit c := CategoryTheory.Limits.isLimitOfPreserve …
        ⊢ CategoryTheory.Limits.HasLimit F
      -/
      apply HasLimit.mk ⟨_, (IsLimit.postcomposeHomEquiv _ _).symm t⟩
      apply
        (F.rightUnitor ≪≫ (isoWhiskerLeft F ((asIso (comonadicAdjunction R).unit) : _) )).symm


/-- If `C` has limits then any coreflective subcategory has limits. -/
theorem hasLimits_of_coreflective (R : D ⥤ C) [Coreflective R] [HasLimitsOfSize.{v, u} C] :
    HasLimitsOfSize.{v, u} D :=
  ⟨fun _ => hasLimitsOfShape_of_coreflective R⟩


/-- The coreflector always preserves initial objects. Note this in general doesn't apply to any
other colimit.
-/
lemma rightAdjoint_preservesInitial_of_coreflective (R : D ⥤ C) [Coreflective R] :
    PreservesColimitsOfShape (Discrete.{v} PEmpty) (comonadicRightAdjoint R) where
  preservesColimit {K} := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      R : CategoryTheory.Functor D C
      inst✝ : CategoryTheory.Coreflective R
      K : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{v + 1}) C
      ⊢ CategoryTheory.Limits.PreservesColimit K (CategoryTheory.comonadicRightAdjoi …
    -/
    let F := Functor.empty.{v} D
    letI : PreservesColimit (F ⋙ R) (comonadicRightAdjoint R) := by
      constructor
      intro c h
      haveI : HasColimit (F ⋙ R) := ⟨⟨⟨c, h⟩⟩⟩
      haveI : HasColimit F := hasColimit_of_coreflective F R
      constructor
      apply isColimitChangeEmptyCocone D (colimit.isColimit F)
      apply (asIso ((comonadicAdjunction R).unit.app _)).trans
      apply (comonadicRightAdjoint R).mapIso
      letI := comonadicCreatesColimits.{v, v} R
      let A := CategoryTheory.preservesColimit_of_createsColimit_and_hasColimit F R
      apply (isColimitOfPreserves _ (colimit.isColimit F)).coconePointUniqueUpToIso h
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      R : CategoryTheory.Functor D C
      inst✝ : CategoryTheory.Coreflective R
      K : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{v + 1}) C
      F : CategoryTheory.Functor (CategoryTheory.Discrete PEmpty.{v + 1}) D := Categ …
      this : CategoryTheory.Limits.PreservesColimit (F.comp R) (CategoryTheory.comon …
        {
          preserves := fun {c} h =>
            Nonempty.intro
              (CategoryTheory.Limits.isColimitChangeEmptyCocone D (CategoryTheory.Li …
                ((CategoryTheory.asIso ((CategoryTheory.comonadicAdjunction R).unit. …
                  ((CategoryTheory.comonadicRightAdjoint R).mapIso
                    (let A := CategoryTheory.preservesColimit_of_createsColimit_and_ …
                    (CategoryTheory.Limits.isColimitOfPreserves R (CategoryTheory.Li …
      ⊢ CategoryTheory.Limits.PreservesColimit K (CategoryTheory.comonadicRightAdjoi …
    -/
    apply preservesColimit_of_iso_diagram _ (Functor.emptyExt (F ⋙ R) _)
    /-
      🎉 no goals
    -/


