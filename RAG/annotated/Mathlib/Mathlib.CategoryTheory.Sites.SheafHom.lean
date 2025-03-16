/-- Given two presheaves `F` and `G` on a category `C` with values in a category `A`,
this `presheafHom F G` is the presheaf of types which sends an object `X : C`
to the type of morphisms between the "restrictions" of `F` and `G` to the category `Over X`. -/
@[simps! obj]
def presheafHom : Cᵒᵖ ⥤ Type _ where
  obj X := (Over.forget X.unop).op ⋙ F ⟶ (Over.forget X.unop).op ⋙ G
  map f := whiskerLeft (Over.map f.unop).op
  map_id := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      ⊢ ∀ (X : Opposite C), Eq ({ obj := fun X => Quiver.Hom ((CategoryTheory.Over.f …
    -/
    rintro ⟨X⟩
    /-
      case op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      ⊢ Eq ({ obj := fun X => Quiver.Hom ((CategoryTheory.Over.forget (Opposite.unop …
    -/
    ext φ ⟨Y⟩
    /-
      case op.h.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      φ : { obj := fun X => Quiver.Hom ((CategoryTheory.Over.forget (Opposite.unop X …
      Y : CategoryTheory.Over (Opposite.unop { unop := X })
      ⊢ Eq (({ obj := fun X => Quiver.Hom ((CategoryTheory.Over.forget (Opposite.uno …
    -/
    simpa [Over.mapId] using φ.naturality ((Over.mapId X).hom.app Y).op
    /-
      🎉 no goals
    -/
  map_comp := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      ⊢ ∀ {X Y Z : Opposite C} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ obj  …
    -/
    rintro ⟨X⟩ ⟨Y⟩ ⟨Z⟩ ⟨f : Y ⟶ X⟩ ⟨g : Z ⟶ Y⟩
    /-
      case op.op.op.op.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z Y
      ⊢ Eq ({ obj := fun X => Quiver.Hom ((CategoryTheory.Over.forget (Opposite.unop …
    -/
    ext φ ⟨W⟩
    /-
      case op.op.op.op.op.h.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X Y Z : C
      f : Quiver.Hom Y X
      g : Quiver.Hom Z Y
      φ : { obj := fun X => Quiver.Hom ((CategoryTheory.Over.forget (Opposite.unop X …
      W : CategoryTheory.Over (Opposite.unop { unop := Z })
      ⊢ Eq (({ obj := fun X => Quiver.Hom ((CategoryTheory.Over.forget (Opposite.uno …
    -/
    simpa [Over.mapComp] using φ.naturality ((Over.mapComp g f).hom.app W).op
    /-
      🎉 no goals
    -/


/-- Equational lemma for the presheaf structure on `presheafHom`.
It is advisable to use this lemma rather than `dsimp [presheafHom]` which may result
in the need to prove equalities of objects in an `Over` category. -/
lemma presheafHom_map_app {X Y Z : C} (f : Z ⟶ Y) (g : Y ⟶ X) (h : Z ⟶ X) (w : f ≫ g = h)
    (α : (presheafHom F G).obj (op X)) :
    ((presheafHom F G).map g.op α).app (op (Over.mk f)) =
      α.app (op (Over.mk h)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    X Y Z : C
    f : Quiver.Hom Z Y
    g : Quiver.Hom Y X
    h : Quiver.Hom Z X
    w : Eq (CategoryTheory.CategoryStruct.comp f g) h
    α : (CategoryTheory.presheafHom F G).obj { unop := X }
    ⊢ Eq (((CategoryTheory.presheafHom F G).map g.op α).app { unop := CategoryTheo …
  -/
  subst w
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    X Y Z : C
    f : Quiver.Hom Z Y
    g : Quiver.Hom Y X
    α : (CategoryTheory.presheafHom F G).obj { unop := X }
    ⊢ Eq (((CategoryTheory.presheafHom F G).map g.op α).app { unop := CategoryTheo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma presheafHom_map_app_op_mk_id {X Y : C} (g : Y ⟶ X)
    (α : (presheafHom F G).obj (op X)) :
    ((presheafHom F G).map g.op α).app (op (Over.mk (𝟙 Y))) =
      α.app (op (Over.mk g)) :=
                                    /-
                                      C : Type u
                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                      A : Type u'
                                      inst✝ : CategoryTheory.Category.{v', u'} A
                                      F G : CategoryTheory.Functor (Opposite C) A
                                      X Y : C
                                      g : Quiver.Hom Y X
                                      α : (CategoryTheory.presheafHom F G).obj { unop := X }
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Y)  …
                                    -/
  presheafHom_map_app (𝟙 Y) g g (by simp) α
                                    /-
                                      🎉 no goals
                                    -/


/-- The sections of the presheaf `presheafHom F G` identify to morphisms `F ⟶ G`. -/
def presheafHomSectionsEquiv : (presheafHom F G).sections ≃ (F ⟶ G) where
  toFun s :=
    { app := fun X => (s.1 X).app ⟨Over.mk (𝟙 _)⟩
      naturality := by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          A : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} A
          F G : CategoryTheory.Functor (Opposite C) A
          s : ↑(CategoryTheory.presheafHom F G).sections
          ⊢ ∀ ⦃X Y : Opposite C⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
        -/
        rintro ⟨X₁⟩ ⟨X₂⟩ ⟨f : X₂ ⟶ X₁⟩
        /-
          case op.op.op
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          A : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} A
          F G : CategoryTheory.Functor (Opposite C) A
          s : ↑(CategoryTheory.presheafHom F G).sections
          X₁ X₂ : C
          f : Quiver.Hom X₂ X₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { unop := f }) ((fun X => (↑s  …
        -/
        dsimp
        refine Eq.trans ?_ ((s.1 ⟨X₁⟩).naturality
          (Over.homMk f : Over.mk f ⟶ Over.mk (𝟙 X₁)).op)
        /-
          case op.op.op
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          A : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} A
          F G : CategoryTheory.Functor (Opposite C) A
          s : ↑(CategoryTheory.presheafHom F G).sections
          X₁ X₂ : C
          f : Quiver.Hom X₂ X₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { unop := f }) ((↑s { unop :=  …
        -/
        rw [← s.2 f.op, presheafHom_map_app_op_mk_id]
        /-
          case op.op.op
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          J : CategoryTheory.GrothendieckTopology C
          A : Type u'
          inst✝ : CategoryTheory.Category.{v', u'} A
          F G : CategoryTheory.Functor (Opposite C) A
          s : ↑(CategoryTheory.presheafHom F G).sections
          X₁ X₂ : C
          f : Quiver.Hom X₂ X₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { unop := f }) ((↑s { unop :=  …
        -/
        rfl }
        /-
          🎉 no goals
        -/
  invFun f := ⟨fun _ => whiskerLeft _ f, fun _ => rfl⟩
  left_inv s := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      s : ↑(CategoryTheory.presheafHom F G).sections
      ⊢ Eq ((fun f => ⟨fun x => CategoryTheory.whiskerLeft (CategoryTheory.Over.forg …
    -/
    dsimp
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      s : ↑(CategoryTheory.presheafHom F G).sections
      ⊢ Eq ⟨fun x => CategoryTheory.whiskerLeft (CategoryTheory.Over.forget (Opposit …
    -/
    ext ⟨X⟩ ⟨Y : Over X⟩
    /-
      case a.h.op.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      s : ↑(CategoryTheory.presheafHom F G).sections
      X : C
      Y : CategoryTheory.Over X
      ⊢ Eq ((↑⟨fun x => CategoryTheory.whiskerLeft (CategoryTheory.Over.forget (Oppo …
    -/
    have H := s.2 Y.hom.op
    /-
      case a.h.op.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      s : ↑(CategoryTheory.presheafHom F G).sections
      X : C
      Y : CategoryTheory.Over X
      H : Eq ((CategoryTheory.presheafHom F G).map Y.hom.op (↑s { unop := (CategoryT …
      ⊢ Eq ((↑⟨fun x => CategoryTheory.whiskerLeft (CategoryTheory.Over.forget (Oppo …
    -/
    dsimp at H ⊢
    /-
      case a.h.op.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      s : ↑(CategoryTheory.presheafHom F G).sections
      X : C
      Y : CategoryTheory.Over X
      H : Eq ((CategoryTheory.presheafHom F G).map Y.hom.op (↑s { unop := X })) (↑s  …
      ⊢ Eq ((↑s { unop := Y.left }).app { unop := CategoryTheory.Over.mk (CategoryTh …
    -/
    rw [← H]
    /-
      case a.h.op.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      s : ↑(CategoryTheory.presheafHom F G).sections
      X : C
      Y : CategoryTheory.Over X
      H : Eq ((CategoryTheory.presheafHom F G).map Y.hom.op (↑s { unop := X })) (↑s  …
      ⊢ Eq (((CategoryTheory.presheafHom F G).map Y.hom.op (↑s { unop := X })).app { …
    -/
    apply presheafHom_map_app_op_mk_id
    /-
      🎉 no goals
    -/
  right_inv _ := rfl


lemma PresheafHom.isAmalgamation_iff {X : C} (S : Sieve X)
    (x : Presieve.FamilyOfElements (presheafHom F G) S.arrows)
    (hx : x.Compatible) (y : (presheafHom F G).obj (op X)) :
    x.IsAmalgamation y ↔ ∀ (Y : C) (g : Y ⟶ X) (hg : S g),
      y.app (op (Over.mk g)) = (x g hg).app (op (Over.mk (𝟙 Y))) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
    hx : x.Compatible
    y : (CategoryTheory.presheafHom F G).obj { unop := X }
    ⊢ Iff (x.IsAmalgamation y) (∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g),  …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      ⊢ x.IsAmalgamation y → ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y …
    -/
  · intro h Y g hg
    /-
      case mp
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : x.IsAmalgamation y
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      ⊢ Eq (y.app { unop := CategoryTheory.Over.mk g }) ((x g hg).app { unop := Cate …
    -/
    rw [← h g hg, presheafHom_map_app_op_mk_id]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      ⊢ (∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Categ …
    -/
  · intro h Y g hg
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Cate …
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      ⊢ Eq ((CategoryTheory.presheafHom F G).map g.op y) (x g hg)
    -/
    dsimp
    /-
      case mpr
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Cate …
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      ⊢ Eq ((CategoryTheory.presheafHom F G).map g.op y) (x g hg)
    -/
    ext ⟨W : Over Y⟩
    /-
      case mpr.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Cate …
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      W : CategoryTheory.Over Y
      ⊢ Eq (((CategoryTheory.presheafHom F G).map g.op y).app { unop := W }) ((x g h …
    -/
    refine (h W.left (W.hom ≫ g) (S.downward_closed hg _)).trans ?_
    /-
      case mpr.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Cate …
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      W : CategoryTheory.Over Y
      ⊢ Eq ((x (CategoryTheory.CategoryStruct.comp W.hom g) ⋯).app { unop := Categor …
    -/
    have H := hx (𝟙 _) W.hom (S.downward_closed hg W.hom) hg (by simp)
    /-
      case mpr.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Cate …
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      W : CategoryTheory.Over Y
      H : Eq ((CategoryTheory.presheafHom F G).map (CategoryTheory.CategoryStruct.id …
      ⊢ Eq ((x (CategoryTheory.CategoryStruct.comp W.hom g) ⋯).app { unop := Categor …
    -/
    dsimp at H
    /-
      case mpr.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Cate …
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      W : CategoryTheory.Over Y
      H : Eq ((CategoryTheory.presheafHom F G).map (CategoryTheory.CategoryStruct.id …
      ⊢ Eq ((x (CategoryTheory.CategoryStruct.comp W.hom g) ⋯).app { unop := Categor …
    -/
    simp only [Functor.map_id, FunctorToTypes.map_id_apply] at H
    /-
      case mpr.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Cate …
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      W : CategoryTheory.Over Y
      H : Eq (x (CategoryTheory.CategoryStruct.comp W.hom g) ⋯) ((CategoryTheory.pre …
      ⊢ Eq ((x (CategoryTheory.CategoryStruct.comp W.hom g) ⋯).app { unop := Categor …
    -/
    rw [H, presheafHom_map_app_op_mk_id]
    /-
      case mpr.w.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y : (CategoryTheory.presheafHom F G).obj { unop := X }
      h : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y.app { unop := Cate …
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      W : CategoryTheory.Over Y
      H : Eq (x (CategoryTheory.CategoryStruct.comp W.hom g) ⋯) ((CategoryTheory.pre …
      ⊢ Eq ((x g hg).app { unop := CategoryTheory.Over.mk W.hom }) ((x g hg).app { u …
    -/
    rfl
    /-
      🎉 no goals
    -/


include hG in
lemma exists_app (hx : x.Compatible) (g : Y ⟶ X) :
    ∃ (φ : F.obj (op Y) ⟶ G.obj (op Y)),
      ∀ {Z : C} (p : Z ⟶ Y) (hp : S (p ≫ g)), φ ≫ G.map p.op =
        F.map p.op ≫ (x (p ≫ g) hp).app ⟨Over.mk (𝟙 Z)⟩ := by
  let c : Cone ((Presieve.diagram (Sieve.pullback g S).arrows).op ⋙ G) :=
    { pt := F.obj (op Y)
      π :=
        { app := fun ⟨Z, hZ⟩ => F.map Z.hom.op ≫ (x _ hZ).app (op (Over.mk (𝟙 _)))
          naturality := by
            rintro ⟨Z₁, hZ₁⟩ ⟨Z₂, hZ₂⟩ ⟨f : Z₂ ⟶ Z₁⟩
            dsimp
            rw [id_comp, assoc]
            have H := hx f.left (𝟙 _) hZ₁ hZ₂ (by simp)
            simp only [presheafHom_obj, unop_op, Functor.id_obj, op_id,
              FunctorToTypes.map_id_apply] at H
            let φ : Over.mk f.left ⟶ Over.mk (𝟙 Z₁.left) := Over.homMk f.left
            have H' := (x (Z₁.hom ≫ g) hZ₁).naturality φ.op
            dsimp at H H' ⊢
            erw [← H, ← H', presheafHom_map_app_op_mk_id, ← F.map_comp_assoc,
              ← op_comp, Over.w f] } }
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
    Y : C
    hx : x.Compatible
    g : Quiver.Hom Y X
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Sieve.pullback g S).arrows.dia …
    ⊢ Exists fun φ => ∀ {Z : C} (p : Quiver.Hom Z Y) (hp : S.arrows (CategoryTheor …
  -/
  use (hG g).lift c
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
    Y : C
    hx : x.Compatible
    g : Quiver.Hom Y X
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Sieve.pullback g S).arrows.dia …
    ⊢ ∀ {Z : C} (p : Quiver.Hom Z Y) (hp : S.arrows (CategoryTheory.CategoryStruct …
  -/
  intro Z p hp
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
    Y : C
    hx : x.Compatible
    g : Quiver.Hom Y X
    c : CategoryTheory.Limits.Cone ((CategoryTheory.Sieve.pullback g S).arrows.dia …
    Z : C
    p : Quiver.Hom Z Y
    hp : S.arrows (CategoryTheory.CategoryStruct.comp p g)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((hG g).lift c) (G.map p.op)) (Catego …
  -/
  exact ((hG g).fac c ⟨Over.mk p, hp⟩)
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `presheafHom_isSheafFor`. -/
noncomputable def app (hx : x.Compatible) (g : Y ⟶ X) : F.obj (op Y) ⟶ G.obj (op Y) :=
  (exists_app hG x hx g).choose


lemma app_cond (hx : x.Compatible) (g : Y ⟶ X) {Z : C} (p : Z ⟶ Y) (hp : S (p ≫ g)) :
    app hG x hx g ≫ G.map p.op = F.map p.op ≫ (x (p ≫ g) hp).app ⟨Over.mk (𝟙 Z)⟩ :=
  (exists_app hG x hx g).choose_spec p hp


include hG in
open PresheafHom.IsSheafFor in
lemma presheafHom_isSheafFor  :
    Presieve.IsSheafFor (presheafHom F G) S.arrows := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.presheafHom F G) S.arrows
  -/
  intro x hx
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    X : C
    S : CategoryTheory.Sieve X
    hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
    x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
    hx : x.Compatible
    ⊢ ExistsUnique fun t => x.IsAmalgamation t
  -/
  apply existsUnique_of_exists_of_unique
  · refine ⟨
      { app := fun Y => app hG x hx Y.unop.hom
        naturality := by
          rintro ⟨Y₁ : Over X⟩ ⟨Y₂ : Over X⟩ ⟨φ : Y₂ ⟶ Y₁⟩
          apply (hG Y₂.hom).hom_ext
          rintro ⟨Z : Over Y₂.left, hZ⟩
          dsimp
          rw [assoc, assoc, app_cond hG x hx Y₂.hom Z.hom hZ, ← G.map_comp, ← op_comp]
          rw [app_cond hG x hx Y₁.hom (Z.hom ≫ φ.left) (by simpa using hZ),
            ← F.map_comp_assoc, op_comp]
          congr 3
          simp }, ?_⟩
    /-
      case hex
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      ⊢ x.IsAmalgamation { app := fun Y => CategoryTheory.PresheafHom.IsSheafFor.app …
    -/
    rw [PresheafHom.isAmalgamation_iff _ _ hx]
    /-
      case hex
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      ⊢ ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq ({ app := fun Y => Cate …
    -/
    intro Y g hg
    /-
      case hex
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      ⊢ Eq ({ app := fun Y => CategoryTheory.PresheafHom.IsSheafFor.app hG x hx (Opp …
    -/
    dsimp
    /-
      case hex
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      ⊢ Eq (CategoryTheory.PresheafHom.IsSheafFor.app hG x hx g) ((x g hg).app { uno …
    -/
    have H := app_cond hG x hx g (𝟙 _) (by simpa using hg)
    /-
      case hex
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.PresheafHom.IsSheaf …
      ⊢ Eq (CategoryTheory.PresheafHom.IsSheafFor.app hG x hx g) ((x g hg).app { uno …
    -/
    rw [op_id, G.map_id, comp_id, F.map_id, id_comp] at H
    /-
      case hex
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      Y : C
      g : Quiver.Hom Y X
      hg : S.arrows g
      H : Eq (CategoryTheory.PresheafHom.IsSheafFor.app hG x hx g) ((x (CategoryTheo …
      ⊢ Eq (CategoryTheory.PresheafHom.IsSheafFor.app hG x hx g) ((x g hg).app { uno …
    -/
    exact H.trans (by congr; simp)
    /-
      🎉 no goals
    -/
    /-
      case hunique
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      ⊢ ∀ (y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }), x.IsAmalgama …
    -/
  · intro y₁ y₂ hy₁ hy₂
    /-
      case hunique
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : x.IsAmalgamation y₁
      hy₂ : x.IsAmalgamation y₂
      ⊢ Eq y₁ y₂
    -/
    rw [PresheafHom.isAmalgamation_iff _ _ hx] at hy₁ hy₂
    /-
      case hunique
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₁.app { unop := C …
      hy₂ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₂.app { unop := C …
      ⊢ Eq y₁ y₂
    -/
    apply NatTrans.ext
    /-
      case hunique.app
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₁.app { unop := C …
      hy₂ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₂.app { unop := C …
      ⊢ Eq y₁.app y₂.app
    -/
    ext ⟨Y : Over X⟩
    /-
      case hunique.app.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₁.app { unop := C …
      hy₂ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₂.app { unop := C …
      Y : CategoryTheory.Over X
      ⊢ Eq (y₁.app { unop := Y }) (y₂.app { unop := Y })
    -/
    apply (hG Y.hom).hom_ext
    /-
      case hunique.app.h.op
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₁.app { unop := C …
      hy₂ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₂.app { unop := C …
      Y : CategoryTheory.Over X
      ⊢ ∀ (j : Opposite (CategoryTheory.Sieve.pullback Y.hom S).arrows.category), Eq …
    -/
    rintro ⟨Z : Over Y.left, hZ⟩
    /-
      case hunique.app.h.op.op.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₁.app { unop := C …
      hy₂ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₂.app { unop := C …
      Y : CategoryTheory.Over X
      Z : CategoryTheory.Over Y.left
      hZ : (CategoryTheory.Sieve.pullback Y.hom S).arrows Z.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (y₁.app { unop := Y }) ((G.mapCone (C …
    -/
    dsimp
    /-
      case hunique.app.h.op.op.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₁.app { unop := C …
      hy₂ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₂.app { unop := C …
      Y : CategoryTheory.Over X
      Z : CategoryTheory.Over Y.left
      hZ : (CategoryTheory.Sieve.pullback Y.hom S).arrows Z.hom
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (y₁.app { unop := Y }) (G.map Z.hom.o …
    -/
    let φ : Over.mk (Z.hom ≫ Y.hom) ⟶ Y := Over.homMk Z.hom
    /-
      case hunique.app.h.op.op.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₁.app { unop := C …
      hy₂ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₂.app { unop := C …
      Y : CategoryTheory.Over X
      Z : CategoryTheory.Over Y.left
      hZ : (CategoryTheory.Sieve.pullback Y.hom S).arrows Z.hom
      φ : Quiver.Hom (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp Z.h …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (y₁.app { unop := Y }) (G.map Z.hom.o …
    -/
    refine (y₁.naturality φ.op).symm.trans (Eq.trans ?_ (y₂.naturality φ.op))
    /-
      case hunique.app.h.op.op.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F G : CategoryTheory.Functor (Opposite C) A
      X : C
      S : CategoryTheory.Sieve X
      hG : ⦃Y : C⦄ → (f : Quiver.Hom Y X) → CategoryTheory.Limits.IsLimit (G.mapCone …
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.presheafHom F G)  …
      hx : x.Compatible
      y₁ y₂ : (CategoryTheory.presheafHom F G).obj { unop := X }
      hy₁ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₁.app { unop := C …
      hy₂ : ∀ (Y : C) (g : Quiver.Hom Y X) (hg : S.arrows g), Eq (y₂.app { unop := C …
      Y : CategoryTheory.Over X
      Z : CategoryTheory.Over Y.left
      hZ : (CategoryTheory.Sieve.pullback Y.hom S).arrows Z.hom
      φ : Quiver.Hom (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp Z.h …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Over.forget (Opposi …
    -/
    rw [(hy₁ _ _ hZ), ← ((hy₂ _ _ hZ))]
    /-
      🎉 no goals
    -/


lemma Presheaf.IsSheaf.hom (hG : Presheaf.IsSheaf J G) :
    Presheaf.IsSheaf J (presheafHom F G) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    hG : CategoryTheory.Presheaf.IsSheaf J G
    ⊢ CategoryTheory.Presheaf.IsSheaf J (CategoryTheory.presheafHom F G)
  -/
  rw [isSheaf_iff_isSheaf_of_type]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    J : CategoryTheory.GrothendieckTopology C
    A : Type u'
    inst✝ : CategoryTheory.Category.{v', u'} A
    F G : CategoryTheory.Functor (Opposite C) A
    hG : CategoryTheory.Presheaf.IsSheaf J G
    ⊢ CategoryTheory.Presieve.IsSheaf J (CategoryTheory.presheafHom F G)
  -/
  intro X S hS
  exact presheafHom_isSheafFor F G S
    (fun _ _ => ((Presheaf.isSheaf_iff_isLimit J G).1 hG _ (J.pullback_stable _ hS)).some)



/-- The underlying presheaf of `sheafHom F G`. It is isomorphic to `presheafHom F.1 G.1`
(see `sheafHom'Iso`), but has better definitional properties. -/
def sheafHom' (F G : Sheaf J A) : Cᵒᵖ ⥤ Type _ where
  obj X := (J.overPullback A X.unop).obj F ⟶ (J.overPullback A X.unop).obj G
  map f := fun φ => (J.overMapPullback A f.unop).map φ
  map_id X := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F✝ G✝ : CategoryTheory.Functor (Opposite C) A
      F G : CategoryTheory.Sheaf J A
      X : Opposite C
      ⊢ Eq ({ obj := fun X => Quiver.Hom ((J.overPullback A (Opposite.unop X)).obj F …
    -/
    ext φ : 2
    /-
      case h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F✝ G✝ : CategoryTheory.Functor (Opposite C) A
      F G : CategoryTheory.Sheaf J A
      X : Opposite C
      φ : { obj := fun X => Quiver.Hom ((J.overPullback A (Opposite.unop X)).obj F)  …
      ⊢ Eq ({ obj := fun X => Quiver.Hom ((J.overPullback A (Opposite.unop X)).obj F …
    -/
    exact congr_fun ((presheafHom F.1 G.1).map_id X) φ.1
    /-
      🎉 no goals
    -/
  map_comp f g := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F✝ G✝ : CategoryTheory.Functor (Opposite C) A
      F G : CategoryTheory.Sheaf J A
      X✝ Y✝ Z✝ : Opposite C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => Quiver.Hom ((J.overPullback A (Opposite.unop X)).obj F …
    -/
    ext φ : 2
    /-
      case h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      J : CategoryTheory.GrothendieckTopology C
      A : Type u'
      inst✝ : CategoryTheory.Category.{v', u'} A
      F✝ G✝ : CategoryTheory.Functor (Opposite C) A
      F G : CategoryTheory.Sheaf J A
      X✝ Y✝ Z✝ : Opposite C
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      φ : { obj := fun X => Quiver.Hom ((J.overPullback A (Opposite.unop X)).obj F)  …
      ⊢ Eq ({ obj := fun X => Quiver.Hom ((J.overPullback A (Opposite.unop X)).obj F …
    -/
    exact congr_fun ((presheafHom F.1 G.1).map_comp f g) φ.1
    /-
      🎉 no goals
    -/


/-- The canonical isomorphism `sheafHom' F G ≅ presheafHom F.1 G.1`. -/
def sheafHom'Iso (F G : Sheaf J A) :
    sheafHom' F G ≅ presheafHom F.1 G.1 :=
  NatIso.ofComponents
    (fun _ => Sheaf.homEquiv.toIso) (fun _ => rfl)


/-- Given two sheaves `F` and `G` on a site `(C, J)` with values in a category `A`,
this `sheafHom F G` is the sheaf of types which sends an object `X : C`
to the type of morphisms between the "restrictions" of `F` and `G` to the category `Over X`. -/
def sheafHom (F G : Sheaf J A) : Sheaf J (Type _) where
  val := sheafHom' F G
  cond := (Presheaf.isSheaf_of_iso_iff (sheafHom'Iso F G)).2 (G.2.hom F.1)


/-- The sections of the sheaf `sheafHom F G` identify to morphisms `F ⟶ G`. -/
def sheafHomSectionsEquiv (F G : Sheaf J A) :
    (sheafHom F G).1.sections ≃ (F ⟶ G) :=
  ((Functor.sectionsFunctor Cᵒᵖ).mapIso (sheafHom'Iso F G)).toEquiv.trans
    ((presheafHomSectionsEquiv F.1 G.1).trans Sheaf.homEquiv.symm)


@[simp]
lemma sheafHomSectionsEquiv_symm_apply_coe_apply {F G : Sheaf J A} (φ : F ⟶ G) (X : Cᵒᵖ) :
    ((sheafHomSectionsEquiv F G).symm φ).1 X = (J.overPullback A X.unop).map φ := rfl


