/-- The category of pointed types. -/
structure Pointed : Type (u + 1) where
  /-- the underlying type -/
  protected X : Type u
  /-- the distinguished element -/
  point : X


instance : CoeSort Pointed Type* :=
  ⟨Pointed.X⟩


/-- Turns a point into a pointed type. -/
def of {X : Type*} (point : X) : Pointed :=
  ⟨X, point⟩


@[simp]
theorem coe_of {X : Type*} (point : X) : ↥(of point) = X :=
  rfl


alias _root_.Prod.Pointed := of


instance : Inhabited Pointed :=
  ⟨of ((), ())⟩


/-- Morphisms in `Pointed`. -/
@[ext]
protected structure Hom (X Y : Pointed.{u}) : Type u where
  /-- the underlying map -/
  toFun : X → Y
  /-- compatibility with the distinguished points -/
  map_point : toFun X.point = Y.point


/-- The identity morphism of `X : Pointed`. -/
@[simps]
def id (X : Pointed) : Pointed.Hom X X :=
  ⟨_root_.id, rfl⟩


instance (X : Pointed) : Inhabited (Pointed.Hom X X) :=
  ⟨id X⟩


/-- Composition of morphisms of `Pointed`. -/
@[simps]
def comp {X Y Z : Pointed.{u}} (f : Pointed.Hom X Y) (g : Pointed.Hom Y Z) : Pointed.Hom X Z :=
                         /-
                           X Y Z : Pointed
                           f : X.Hom Y
                           g : Y.Hom Z
                           ⊢ Eq (Function.comp g.toFun f.toFun X.point) Z.point
                         -/
  ⟨g.toFun ∘ f.toFun, by rw [Function.comp_apply, f.map_point, g.map_point]⟩
                         /-
                           🎉 no goals
                         -/


instance largeCategory : LargeCategory Pointed where
  Hom := Pointed.Hom
  id := Hom.id
  comp := @Hom.comp


@[simp] lemma Hom.id_toFun' (X : Pointed.{u}) : (𝟙 X : X ⟶ X).toFun = _root_.id := rfl


@[simp] lemma Hom.comp_toFun' {X Y Z : Pointed.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).toFun = g.toFun ∘ f.toFun := rfl


instance concreteCategory : ConcreteCategory Pointed where
  forget :=
    { obj := Pointed.X
      map := @Hom.toFun }
  forget_faithful := ⟨@Hom.ext⟩


/-- Constructs an isomorphism between pointed types from an equivalence that preserves the point
between them. -/
@[simps]
def Iso.mk {α β : Pointed} (e : α ≃ β) (he : e α.point = β.point) : α ≅ β where
  hom := ⟨e, he⟩
  inv := ⟨e.symm, e.symm_apply_eq.2 he.symm⟩
  hom_inv_id := Pointed.Hom.ext e.symm_comp_self
  inv_hom_id := Pointed.Hom.ext e.self_comp_symm


/-- `Option` as a functor from types to pointed types. This is the free functor. -/
@[simps]
def typeToPointed : Type u ⥤ Pointed.{u} where
  obj X := ⟨Option X, none⟩
  map f := ⟨Option.map f, rfl⟩
  map_id _ := Pointed.Hom.ext Option.map_id
  map_comp _ _ := Pointed.Hom.ext (Option.map_comp_map _ _).symm


/-- `typeToPointed` is the free functor. -/
def typeToPointedForgetAdjunction : typeToPointed ⊣ forget Pointed :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun X Y =>
        { toFun := fun f => f.toFun ∘ Option.some
          invFun := fun f => ⟨fun o => o.elim Y.point f, rfl⟩
          left_inv := fun f => by
            /-
              X : Type ?u.5013
              Y : Pointed
              f : Quiver.Hom (typeToPointed.obj X) Y
              ⊢ Eq ((fun f => { toFun := fun o => Option.elim o Y.point f, map_point := ⋯ }) …
            -/
            apply Pointed.Hom.ext
            /-
              case toFun
              X : Type ?u.5013
              Y : Pointed
              f : Quiver.Hom (typeToPointed.obj X) Y
              ⊢ Eq ((fun f => { toFun := fun o => Option.elim o Y.point f, map_point := ⋯ }) …
            -/
            funext x
            /-
              case toFun.h
              X : Type ?u.5013
              Y : Pointed
              f : Quiver.Hom (typeToPointed.obj X) Y
              x : (typeToPointed.obj X).X
              ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.point f, map_point := ⋯ } …
            -/
            cases x
              /-
                case toFun.h.none
                X : Type ?u.5013
                Y : Pointed
                f : Quiver.Hom (typeToPointed.obj X) Y
                ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.point f, map_point := ⋯ } …
              -/
            · exact f.map_point.symm
              /-
                🎉 no goals
              -/
              /-
                case toFun.h.some
                X : Type ?u.5013
                Y : Pointed
                f : Quiver.Hom (typeToPointed.obj X) Y
                val✝ : X
                ⊢ Eq (((fun f => { toFun := fun o => Option.elim o Y.point f, map_point := ⋯ } …
              -/
            · rfl
              /-
                🎉 no goals
              -/
          right_inv := fun _ => funext fun _ => rfl }
      homEquiv_naturality_left_symm := fun f g => by
        /-
          X'✝ X✝ : Type ?u.5013
          Y✝ : Pointed
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ ((CategoryTheory.forget Pointed).obj Y✝)
          ⊢ Eq (((fun X Y => { toFun := fun f => Function.comp f.toFun Option.some, invF …
        -/
        apply Pointed.Hom.ext
        /-
          case toFun
          X'✝ X✝ : Type ?u.5013
          Y✝ : Pointed
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ ((CategoryTheory.forget Pointed).obj Y✝)
          ⊢ Eq (((fun X Y => { toFun := fun f => Function.comp f.toFun Option.some, invF …
        -/
        funext x
        /-
          case toFun.h
          X'✝ X✝ : Type ?u.5013
          Y✝ : Pointed
          f : Quiver.Hom X'✝ X✝
          g : Quiver.Hom X✝ ((CategoryTheory.forget Pointed).obj Y✝)
          x : (typeToPointed.obj X'✝).X
          ⊢ Eq ((((fun X Y => { toFun := fun f => Function.comp f.toFun Option.some, inv …
        -/
                    /-
                      🎉 no goals
                    -/
        cases x <;> rfl }
                    /-
                      🎉 no goals
                    -/

