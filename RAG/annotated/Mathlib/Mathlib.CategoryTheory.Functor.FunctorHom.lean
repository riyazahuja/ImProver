/-- Given functors `F G : C ⥤ D`, `HomObj F G A` is a proxy for the type
of "morphisms" `F ⊗ A ⟶ G`, where `A : C ⥤ Type w` (`w` an arbitrary universe). -/
@[ext]
structure HomObj (A : C ⥤ Type w) where
  /-- The morphism `F.obj c ⟶ G.obj c` associated with `a : A.obj c`. -/
  app (c : C) (a : A.obj c) : F.obj c ⟶ G.obj c
  naturality {c d : C} (f : c ⟶ d) (a : A.obj c) :
    F.map f ≫ app d (A.map f a) = app c a ≫ G.map f := by aesop_cat


/-- When `F`, `G`, and `A` are all functors `C ⥤ Type w`, then `HomObj F G A` is in
bijection with `F ⊗ A ⟶ G`. -/
@[simps]
def homObjEquiv (F G A : C ⥤ Type w) : (HomObj F G A) ≃ (F ⊗ A ⟶ G) where
  toFun a := ⟨fun X ⟨x, y⟩ ↦ a.app X y x, fun X Y f ↦ by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ G✝ : CategoryTheory.Functor C D
      F G A : CategoryTheory.Functor C (Type w)
      a : F.HomObj G A
      X Y : C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
    -/
    ext ⟨x, y⟩
    /-
      case h.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ G✝ : CategoryTheory.Functor C D
      F G A : CategoryTheory.Functor C (Type w)
      a : F.HomObj G A
      X Y : C
      f : Quiver.Hom X Y
      x : F.obj X
      y : A.obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
    -/
    erw [congr_fun (a.naturality f y) x]
    /-
      case h.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ G✝ : CategoryTheory.Functor C D
      F G A : CategoryTheory.Functor C (Type w)
      a : F.HomObj G A
      X Y : C
      f : Quiver.Hom X Y
      x : F.obj X
      y : A.obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (a.app X y) (G.map f) x) (CategoryThe …
    -/
    rfl ⟩
    /-
      🎉 no goals
    -/
  invFun a := ⟨fun X y x ↦ a.app X (x, y), fun φ y ↦ by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ G✝ : CategoryTheory.Functor C D
      F G A : CategoryTheory.Functor C (Type w)
      a : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj F A) G
      c✝ d✝ : C
      φ : Quiver.Hom c✝ d✝
      y : A.obj c✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) ((fun X y x => a.app X { fs …
    -/
    ext x
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ G✝ : CategoryTheory.Functor C D
      F G A : CategoryTheory.Functor C (Type w)
      a : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj F A) G
      c✝ d✝ : C
      φ : Quiver.Hom c✝ d✝
      y : A.obj c✝
      x : F.obj c✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) ((fun X y x => a.app X { fs …
    -/
    erw [congr_fun (a.naturality φ) (x, y)]
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F✝ G✝ : CategoryTheory.Functor C D
      F G A : CategoryTheory.Functor C (Type w)
      a : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj F A) G
      c✝ d✝ : C
      φ : Quiver.Hom c✝ d✝
      y : A.obj c✝
      x : F.obj c✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (a.app c✝) (G.map φ) { fst := x, snd  …
    -/
    rfl ⟩
    /-
      🎉 no goals
    -/
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     D : Type u'
                     inst✝ : CategoryTheory.Category.{v', u'} D
                     F✝ G✝ : CategoryTheory.Functor C D
                     F G A : CategoryTheory.Functor C (Type w)
                     x✝ : F.HomObj G A
                     ⊢ Eq ((fun a => { app := fun X y x => a.app X { fst := x, snd := y }, naturali …
                   -/
  left_inv _ := by aesop
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u
                      inst✝¹ : CategoryTheory.Category.{v, u} C
                      D : Type u'
                      inst✝ : CategoryTheory.Category.{v', u'} D
                      F✝ G✝ : CategoryTheory.Functor C D
                      F G A : CategoryTheory.Functor C (Type w)
                      x✝ : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj F A) G
                      ⊢ Eq ((fun a => { app := fun X x => CategoryTheory.Functor.homObjEquiv.match_1 …
                    -/
  right_inv _ := by aesop
                    /-
                      🎉 no goals
                    -/


attribute [reassoc (attr := simp)] naturality


lemma congr_app {f g : HomObj F G A} (h : f = g) (X : C)
                                                /-
                                                  C : Type u
                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                  D : Type u'
                                                  inst✝ : CategoryTheory.Category.{v', u'} D
                                                  F G : CategoryTheory.Functor C D
                                                  A : CategoryTheory.Functor C (Type w)
                                                  f g : F.HomObj G A
                                                  h : Eq f g
                                                  X : C
                                                  a : A.obj X
                                                  ⊢ Eq (f.app X a) (g.app X a)
                                                -/
    (a : A.obj X) : f.app X a = g.app X a := by subst h; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Given a natural transformation `F ⟶ G`, get a term of `HomObj F G A` by "ignoring" `A`. -/
@[simps]
def ofNatTrans (f : F ⟶ G) : HomObj F G A where
  app X _ := f.app X


/-- The identity `HomObj F F A`. -/
@[simps!]
def id (A : C ⥤ Type w) : HomObj F F A := ofNatTrans (𝟙 F)


/-- Composition of `f : HomObj F G A` with `g : HomObj G M A`. -/
@[simps]
def comp {M : C ⥤ D} (f : HomObj F G A) (g : HomObj G M A) : HomObj F M A where
  app X a := f.app X a ≫ g.app X a


/-- Given a morphism `A' ⟶ A`, send a term of `HomObj F G A` to a term of `HomObj F G A'`. -/
@[simps]
def map {A' : C ⥤ Type w} (f : A' ⟶ A) (x : HomObj F G A) : HomObj F G A' where
  app Δ a := x.app Δ (f.app Δ a)
  naturality {Δ Δ'} φ a := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      A A' : CategoryTheory.Functor C (Type w)
      f : Quiver.Hom A' A
      x : F.HomObj G A
      Δ Δ' : C
      φ : Quiver.Hom Δ Δ'
      a : A'.obj Δ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) ((fun Δ a => x.app Δ (f.app …
    -/
    dsimp
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      A A' : CategoryTheory.Functor C (Type w)
      f : Quiver.Hom A' A
      x : F.HomObj G A
      Δ Δ' : C
      φ : Quiver.Hom Δ Δ'
      a : A'.obj Δ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (x.app Δ' (f.app Δ' (A'.map …
    -/
    rw [← x.naturality φ (f.app Δ a), FunctorToTypes.naturality _ _ f φ a]
    /-
      🎉 no goals
    -/


/-- The contravariant functor taking `A : C ⥤ Type w` to `HomObj F G A`, i.e. Hom(F ⊗ -, G). -/
@[simps]
def homObjFunctor : (C ⥤ Type w)ᵒᵖ ⥤ Type max w v' u where
  obj A := HomObj F G A.unop
  map {A A'} f x :=
    { app := fun X a ↦ x.app X (f.unop.app _ a)
      naturality := fun {X Y} φ a ↦ by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A A' : Opposite (CategoryTheory.Functor C (Type w))
          f : Quiver.Hom A A'
          x : (fun A => F.HomObj G (Opposite.unop A)) A
          X Y : C
          φ : Quiver.Hom X Y
          a : (Opposite.unop A').obj X
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) ((fun X a => x.app X (f.uno …
        -/
        dsimp
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A A' : Opposite (CategoryTheory.Functor C (Type w))
          f : Quiver.Hom A A'
          x : (fun A => F.HomObj G (Opposite.unop A)) A
          X Y : C
          φ : Quiver.Hom X Y
          a : (Opposite.unop A').obj X
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (x.app Y (f.unop.app Y ((Op …
        -/
        rw [← HomObj.naturality]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A A' : Opposite (CategoryTheory.Functor C (Type w))
          f : Quiver.Hom A A'
          x : (fun A => F.HomObj G (Opposite.unop A)) A
          X Y : C
          φ : Quiver.Hom X Y
          a : (Opposite.unop A').obj X
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (x.app Y (f.unop.app Y ((Op …
        -/
        congr 2
        /-
          case e_a.e_a
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A A' : Opposite (CategoryTheory.Functor C (Type w))
          f : Quiver.Hom A A'
          x : (fun A => F.HomObj G (Opposite.unop A)) A
          X Y : C
          φ : Quiver.Hom X Y
          a : (Opposite.unop A').obj X
          ⊢ Eq (f.unop.app Y ((Opposite.unop A').map φ a)) ((Opposite.unop A).map φ (f.u …
        -/
        exact congr_fun (f.unop.naturality φ) a }
        /-
          🎉 no goals
        -/


/-- Composition of `homObjFunctor` with the co-Yoneda embedding, i.e. Hom(F ⊗ coyoneda(-), G).
When `F G : C ⥤ Type max v' v u`, this is the internal hom of `F` and `G`: see
`Mathlib.CategoryTheory.Closed.FunctorToTypes`. -/
def functorHom (F G : C ⥤ D) : C ⥤ Type max v' v u := coyoneda.rightOp ⋙ homObjFunctor.{v} F G


variable {F G} in
@[ext]
lemma functorHom_ext {X : C} {x y : (F.functorHom G).obj X}
    (h : ∀ (Y : C) (f : X ⟶ Y), x.app Y f = y.app Y f) : x = y :=
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   D : Type u'
                   inst✝ : CategoryTheory.Category.{v', u'} D
                   F G : CategoryTheory.Functor C D
                   X : C
                   x y : (F.functorHom G).obj X
                   h : ∀ (Y : C) (f : Quiver.Hom X Y), Eq (x.app Y f) (y.app Y f)
                   ⊢ Eq x.app y.app
                 -/
  HomObj.ext (by ext; apply h)
                      /-
                        🎉 no goals
                      -/


/-- The equivalence `(A ⟶ F.functorHom G) ≃ HomObj F G A`. -/
@[simps]
def functorHomEquiv (A : C ⥤ Type max u v v') : (A ⟶ F.functorHom G) ≃ HomObj F G A where
  toFun φ :=
    { app := fun X a ↦ (φ.app X a).app X (𝟙 _)
      naturality := fun {X Y} f a => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A : CategoryTheory.Functor C (Type (max u v v'))
          φ : Quiver.Hom A (F.functorHom G)
          X Y : C
          f : Quiver.Hom X Y
          a : A.obj X
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X a => (φ.app X a).ap …
        -/
        rw [← (φ.app X a).naturality f (𝟙 _)]
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A : CategoryTheory.Functor C (Type (max u v v'))
          φ : Quiver.Hom A (F.functorHom G)
          X Y : C
          f : Quiver.Hom X Y
          a : A.obj X
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X a => (φ.app X a).ap …
        -/
        have := HomObj.congr_app (congr_fun (φ.naturality f) a) Y (𝟙 _)
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A : CategoryTheory.Functor C (Type (max u v v'))
          φ : Quiver.Hom A (F.functorHom G)
          X Y : C
          f : Quiver.Hom X Y
          a : A.obj X
          this : Eq ((CategoryTheory.CategoryStruct.comp (A.map f) (φ.app Y) a).app Y (C …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X a => (φ.app X a).ap …
        -/
        dsimp [functorHom, homObjFunctor] at this
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A : CategoryTheory.Functor C (Type (max u v v'))
          φ : Quiver.Hom A (F.functorHom G)
          X Y : C
          f : Quiver.Hom X Y
          a : A.obj X
          this : Eq ((φ.app Y (A.map f a)).app Y (CategoryTheory.CategoryStruct.id Y)) ( …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X a => (φ.app X a).ap …
        -/
        aesop }
        /-
          🎉 no goals
        -/
  invFun x :=
    { app := fun X a ↦ { app := fun Y f => x.app Y (A.map f a) }
      naturality := fun X Y f => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A : CategoryTheory.Functor C (Type (max u v v'))
          x : F.HomObj G A
          X Y : C
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (A.map f) ((fun X a => { app := fun Y …
        -/
        ext
        /-
          case h.h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A : CategoryTheory.Functor C (Type (max u v v'))
          x : F.HomObj G A
          X Y : C
          f : Quiver.Hom X Y
          a✝ : A.obj X
          Y✝ : C
          f✝ : Quiver.Hom Y Y✝
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (A.map f) ((fun X a => { app := fun  …
        -/
        dsimp only [types_comp_apply]
        /-
          case h.h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A : CategoryTheory.Functor C (Type (max u v v'))
          x : F.HomObj G A
          X Y : C
          f : Quiver.Hom X Y
          a✝ : A.obj X
          Y✝ : C
          f✝ : Quiver.Hom Y Y✝
          ⊢ Eq (x.app Y✝ (A.map f✝ (A.map f a✝))) (((F.functorHom G).map f { app := fun  …
        -/
        rw [← FunctorToTypes.map_comp_apply]
        /-
          case h.h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} D
          F G : CategoryTheory.Functor C D
          A : CategoryTheory.Functor C (Type (max u v v'))
          x : F.HomObj G A
          X Y : C
          f : Quiver.Hom X Y
          a✝ : A.obj X
          Y✝ : C
          f✝ : Quiver.Hom Y Y✝
          ⊢ Eq (x.app Y✝ (A.map (CategoryTheory.CategoryStruct.comp f f✝) a✝)) (((F.func …
        -/
        rfl }
        /-
          🎉 no goals
        -/
  left_inv φ := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      A : CategoryTheory.Functor C (Type (max u v v'))
      φ : Quiver.Hom A (F.functorHom G)
      ⊢ Eq ((fun x => { app := fun X a => { app := fun Y f => x.app Y (A.map f a), n …
    -/
    ext X a Y f
    exact (HomObj.congr_app (congr_fun (φ.naturality f) a) Y (𝟙 _)).trans
      (congr_arg ((φ.app X a).app Y) (by simp))
                    /-
                      C : Type u
                      inst✝¹ : CategoryTheory.Category.{v, u} C
                      D : Type u'
                      inst✝ : CategoryTheory.Category.{v', u'} D
                      F G : CategoryTheory.Functor C D
                      A : CategoryTheory.Functor C (Type (max u v v'))
                      x : F.HomObj G A
                      ⊢ Eq ((fun φ => { app := fun X a => (φ.app X a).app X (CategoryTheory.Category …
                    -/
  right_inv x := by aesop
                    /-
                      🎉 no goals
                    -/


variable {F G} in
/-- Morphisms `(𝟙_ (C ⥤ Type max v' v u) ⟶ F.functorHom G)` are in bijection with
morphisms `F ⟶ G`. -/
@[simps]
def natTransEquiv : (𝟙_ (C ⥤ Type max v' v u) ⟶ F.functorHom G) ≃ (F ⟶ G) where
  toFun f := ⟨fun X ↦ (f.app X (PUnit.unit)).app X (𝟙 _), by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      ⊢ ∀ ⦃X Y : C⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
    -/
    intro X Y φ
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      X Y : C
      φ : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) ((fun X => (f.app X PUnit.u …
    -/
    rw [← (f.app X (PUnit.unit)).naturality φ]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      X Y : C
      φ : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) ((fun X => (f.app X PUnit.u …
    -/
    congr 1
    /-
      case e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      X Y : C
      φ : Quiver.Hom X Y
      ⊢ Eq ((fun X => (f.app X PUnit.unit).app X (CategoryTheory.CategoryStruct.id ( …
    -/
    have := HomObj.congr_app (congr_fun (f.naturality φ) PUnit.unit) Y (𝟙 Y)
    /-
      case e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      X Y : C
      φ : Quiver.Hom X Y
      this : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategor …
      ⊢ Eq ((fun X => (f.app X PUnit.unit).app X (CategoryTheory.CategoryStruct.id ( …
    -/
    dsimp [functorHom, homObjFunctor] at this
    /-
      case e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      X Y : C
      φ : Quiver.Hom X Y
      this : Eq ((f.app Y PUnit.unit).app Y (CategoryTheory.CategoryStruct.id Y)) (( …
      ⊢ Eq ((fun X => (f.app X PUnit.unit).app X (CategoryTheory.CategoryStruct.id ( …
    -/
    aesop ⟩
    /-
      🎉 no goals
    -/
  invFun f := ⟨fun _ _ ↦ HomObj.ofNatTrans f, _⟩
  left_inv f := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      ⊢ Eq ((fun f => { app := fun x x_1 => CategoryTheory.Functor.HomObj.ofNatTrans …
    -/
    ext X a Y φ
    /-
      case w.h.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      X : C
      a : CategoryTheory.MonoidalCategoryStruct.tensorUnit.obj X
      Y : C
      φ : Quiver.Hom X Y
      ⊢ Eq ((((fun f => { app := fun x x_1 => CategoryTheory.Functor.HomObj.ofNatTra …
    -/
    have := HomObj.congr_app (congr_fun (f.naturality φ) PUnit.unit) Y (𝟙 Y)
    /-
      case w.h.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      X : C
      a : CategoryTheory.MonoidalCategoryStruct.tensorUnit.obj X
      Y : C
      φ : Quiver.Hom X Y
      this : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategor …
      ⊢ Eq ((((fun f => { app := fun x x_1 => CategoryTheory.Functor.HomObj.ofNatTra …
    -/
    dsimp [functorHom, homObjFunctor] at this
    /-
      case w.h.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} D
      F G : CategoryTheory.Functor C D
      f : Quiver.Hom CategoryTheory.MonoidalCategoryStruct.tensorUnit (F.functorHom G)
      X : C
      a : CategoryTheory.MonoidalCategoryStruct.tensorUnit.obj X
      Y : C
      φ : Quiver.Hom X Y
      this : Eq ((f.app Y PUnit.unit).app Y (CategoryTheory.CategoryStruct.id Y)) (( …
      ⊢ Eq ((((fun f => { app := fun x x_1 => CategoryTheory.Functor.HomObj.ofNatTra …
    -/
    aesop
    /-
      🎉 no goals
    -/
  right_inv _ := rfl


@[simp]
lemma natTransEquiv_symm_app_app_apply (F G : C ⥤ D) (f : F ⟶ G)
    {X : C} {a : (𝟙_ (C ⥤ Type (max v' v u))).obj X} (Y : C) {φ : X ⟶ Y} :
    ((natTransEquiv.symm f).app X a).app Y φ = f.app Y := rfl


@[simp]
lemma natTransEquiv_symm_whiskerRight_functorHom_app (K L : C ⥤ D) (X : C) (f : K ⟶ K)
    (x : 𝟙_ _ ⊗ (K.functorHom L).obj X) :
    ((natTransEquiv.symm f ▷ K.functorHom L).app X x) =
    (HomObj.ofNatTrans f, x.2) := rfl


@[simp]
lemma functorHom_whiskerLeft_natTransEquiv_symm_app (K L : C ⥤ D) (X : C) (f : L ⟶ L)
    (x : (K.functorHom L).obj X ⊗ 𝟙_ _) :
    ((K.functorHom L ◁ natTransEquiv.symm f).app X x) =
    (x.1, HomObj.ofNatTrans f) := rfl


@[simp]
lemma whiskerLeft_app_apply (K L M N : C ⥤ D)
    (g : L.functorHom M ⊗ M.functorHom N ⟶ L.functorHom N)
    {X : C} (a : (K.functorHom L ⊗ L.functorHom M ⊗ M.functorHom N).obj X) :
    (K.functorHom L ◁ g).app X a = ⟨a.1, g.app X a.2⟩ := rfl


@[simp]
lemma whiskerRight_app_apply (K L M N : C ⥤ D)
    (f : K.functorHom L ⊗ L.functorHom M ⟶ K.functorHom M)
    {X : C} (a : ((K.functorHom L ⊗ L.functorHom M) ⊗ M.functorHom N).obj X) :
    (f ▷  M.functorHom N).app X a = ⟨f.app X a.1, a.2⟩ := rfl


@[simp]
lemma associator_inv_apply (K L M N : C ⥤ D) {X : C}
    (x : ((K.functorHom L) ⊗ (L.functorHom M) ⊗ (M.functorHom N)).obj X) :
    (α_ ((K.functorHom L).obj X) ((L.functorHom M).obj X) ((M.functorHom N).obj X)).inv x =
    ⟨⟨x.1, x.2.1⟩, x.2.2⟩ := rfl


@[simp]
lemma associator_hom_apply (K L M N : C ⥤ D) {X : C}
    (x : (((K.functorHom L) ⊗ (L.functorHom M)) ⊗ (M.functorHom N)).obj X) :
    (α_ ((K.functorHom L).obj X) ((L.functorHom M).obj X) ((M.functorHom N).obj X)).hom x =
    ⟨x.1.1, x.1.2, x.2⟩ := rfl


noncomputable instance : EnrichedCategory (C ⥤ Type max v' v u) (C ⥤ D) where
  Hom := functorHom
  id F := natTransEquiv.symm (𝟙 F)
  comp F G H := { app := fun _ ⟨f, g⟩ => f.comp g }


