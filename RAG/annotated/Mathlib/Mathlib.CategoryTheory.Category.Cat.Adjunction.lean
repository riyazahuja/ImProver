private def typeToCatObjectsAdjHomEquiv : (typeToCat.obj X ⟶ C) ≃ (X ⟶ Cat.objects.obj C) where
  toFun f x := f.obj ⟨x⟩
  invFun := Discrete.functor
  left_inv F := Functor.ext (fun _ ↦ rfl) (fun ⟨_⟩ ⟨_⟩ f => by
    /-
      X : Type u
      C : CategoryTheory.Cat
      F : Quiver.Hom (CategoryTheory.typeToCat.obj X) C
      x✝¹ x✝ : CategoryTheory.Discrete X
      as✝¹ as✝ : X
      f : Quiver.Hom { as := as✝¹ } { as := as✝ }
      ⊢ Eq ((CategoryTheory.Discrete.functor ((fun f x => f.obj { as := x }) F)).map …
    -/
    obtain rfl := Discrete.eq_of_hom f
    /-
      X : Type u
      C : CategoryTheory.Cat
      F : Quiver.Hom (CategoryTheory.typeToCat.obj X) C
      x✝¹ x✝ : CategoryTheory.Discrete X
      as✝ : X
      f : Quiver.Hom { as := as✝ } { as := { as := as✝ }.as }
      ⊢ Eq ((CategoryTheory.Discrete.functor ((fun f x => f.obj { as := x }) F)).map …
    -/
    simp)
    /-
      🎉 no goals
    -/
  right_inv _ := rfl


private def typeToCatObjectsAdjCounitApp : (Cat.objects ⋙ typeToCat).obj C ⥤ C where
  obj := Discrete.as
  map := eqToHom ∘ Discrete.eq_of_hom


/-- `typeToCat : Type ⥤ Cat` is left adjoint to `Cat.objects : Cat ⥤ Type` -/
def typeToCatObjectsAdj : typeToCat ⊣ Cat.objects :=
  Adjunction.mk' {
    homEquiv := typeToCatObjectsAdjHomEquiv
    unit := { app:= fun _  ↦ Discrete.mk }
    counit := {
      app := typeToCatObjectsAdjCounitApp
      naturality := fun _ _ _  ↦  Functor.hext (fun _ ↦ rfl)
            /-
              X : Type u
              C : CategoryTheory.Cat
              x✝² x✝¹ : CategoryTheory.Cat
              x✝ : Quiver.Hom x✝² x✝¹
              ⊢ ∀ (X Y : ↑((CategoryTheory.Cat.objects.comp CategoryTheory.typeToCat).obj x✝ …
            -/
        (by intro ⟨_⟩ ⟨_⟩ f
            /-
              X : Type u
              C : CategoryTheory.Cat
              x✝² x✝¹ : CategoryTheory.Cat
              x✝ : Quiver.Hom x✝² x✝¹
              as✝¹ as✝ : CategoryTheory.Cat.objects.obj x✝²
              f : Quiver.Hom { as := as✝¹ } { as := as✝ }
              ⊢ HEq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Cat.objects.comp C …
            -/
            obtain rfl := Discrete.eq_of_hom f
            /-
              X : Type u
              C : CategoryTheory.Cat
              x✝² x✝¹ : CategoryTheory.Cat
              x✝ : Quiver.Hom x✝² x✝¹
              as✝ : CategoryTheory.Cat.objects.obj x✝²
              f : Quiver.Hom { as := as✝ } { as := { as := as✝ }.as }
              ⊢ HEq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Cat.objects.comp C …
            -/
            aesop_cat ) } }
            /-
              🎉 no goals
            -/


/-- The connected components functor -/
def connectedComponents : Cat.{v, u} ⥤ Type u where
  obj C := ConnectedComponents C
  map F := Functor.mapConnectedComponents F
                                                                        /-
                                                                          X : Type u
                                                                          C : CategoryTheory.Cat
                                                                          x✝¹ : CategoryTheory.Cat
                                                                          x : { obj := fun C => CategoryTheory.ConnectedComponents ↑C, map := fun {X Y}  …
                                                                          x✝ : ↑x✝¹
                                                                          h : Eq (Quotient.mk (CategoryTheory.Zigzag.setoid ↑x✝¹) x✝) x
                                                                          ⊢ Eq ({ obj := fun C => CategoryTheory.ConnectedComponents ↑C, map := fun {X Y …
                                                                        -/
  map_id _ := funext fun x ↦ (Quotient.exists_rep x).elim (fun _ h ↦ by subst h; rfl)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
                                                                             /-
                                                                               X : Type u
                                                                               C : CategoryTheory.Cat
                                                                               X✝ Y✝ Z✝ : CategoryTheory.Cat
                                                                               x✝² : Quiver.Hom X✝ Y✝
                                                                               x✝¹ : Quiver.Hom Y✝ Z✝
                                                                               x : { obj := fun C => CategoryTheory.ConnectedComponents ↑C, map := fun {X Y}  …
                                                                               x✝ : ↑X✝
                                                                               h : Eq (Quotient.mk (CategoryTheory.Zigzag.setoid ↑X✝) x✝) x
                                                                               ⊢ Eq ({ obj := fun C => CategoryTheory.ConnectedComponents ↑C, map := fun {X Y …
                                                                             -/
  map_comp _ _ := funext fun x ↦ (Quotient.exists_rep x).elim (fun _ h => by subst h; rfl)
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


/-- `typeToCat : Type ⥤ Cat` is right adjoint to `connectedComponents : Cat ⥤ Type` -/
def connectedComponentsTypeToCatAdj : connectedComponents ⊣ typeToCat :=
  Adjunction.mk' {
    homEquiv := fun C X ↦ ConnectedComponents.typeToCatHomEquiv C X
    unit :=
      { app:= fun C  ↦ ConnectedComponents.functorToDiscrete _ (𝟙 (connectedComponents.obj C)) }
    counit := {
        app := fun X => ConnectedComponents.liftFunctor _ (𝟙 typeToCat.obj X)
        naturality := fun _ _ _ =>
          funext (fun xcc => by
            /-
              X : Type u
              C : CategoryTheory.Cat
              x✝² x✝¹ : Type ?u.9762
              x✝ : Quiver.Hom x✝² x✝¹
              xcc : (CategoryTheory.typeToCat.comp CategoryTheory.Cat.connectedComponents).o …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.typeToCat.comp Categ …
            -/
            obtain ⟨x,h⟩ := Quotient.exists_rep xcc
            /-
              case intro
              X : Type u
              C : CategoryTheory.Cat
              x✝² x✝¹ : Type ?u.9762
              x✝ : Quiver.Hom x✝² x✝¹
              xcc : (CategoryTheory.typeToCat.comp CategoryTheory.Cat.connectedComponents).o …
              x : ↑(CategoryTheory.typeToCat.obj x✝²)
              h : Eq (Quotient.mk (CategoryTheory.Zigzag.setoid ↑(CategoryTheory.typeToCat.o …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.typeToCat.comp Categ …
            -/
            aesop_cat) }
            /-
              🎉 no goals
            -/
    homEquiv_counit := fun {C X G} => by
      /-
        X✝ : Type u
        C✝ : CategoryTheory.Cat
        C : CategoryTheory.Cat
        X : Type ?u.9762
        G : Quiver.Hom C (CategoryTheory.typeToCat.obj X)
        ⊢ Eq (((fun C X => CategoryTheory.ConnectedComponents.typeToCatHomEquiv (↑C) X …
      -/
      funext cc
      /-
        case h
        X✝ : Type u
        C✝ : CategoryTheory.Cat
        C : CategoryTheory.Cat
        X : Type ?u.9762
        G : Quiver.Hom C (CategoryTheory.typeToCat.obj X)
        cc : CategoryTheory.Cat.connectedComponents.obj C
        ⊢ Eq (((fun C X => CategoryTheory.ConnectedComponents.typeToCatHomEquiv (↑C) X …
      -/
      obtain ⟨_, _⟩ := Quotient.exists_rep cc
      /-
        case h.intro
        X✝ : Type u
        C✝ : CategoryTheory.Cat
        C : CategoryTheory.Cat
        X : Type ?u.9762
        G : Quiver.Hom C (CategoryTheory.typeToCat.obj X)
        cc : CategoryTheory.Cat.connectedComponents.obj C
        w✝ : ↑C
        h✝ : Eq (Quotient.mk (CategoryTheory.Zigzag.setoid ↑C) w✝) cc
        ⊢ Eq (((fun C X => CategoryTheory.ConnectedComponents.typeToCatHomEquiv (↑C) X …
      -/
      aesop_cat }
      /-
        🎉 no goals
      -/


