/-- Given a category `J`, construct a walking `Bicone J` by adjoining two elements. -/
inductive Bicone (J : Type u₁)
  | left : Bicone J
  | right : Bicone J
  | diagram (val : J) : Bicone J
  deriving DecidableEq


instance : Inhabited (Bicone J) :=
  ⟨Bicone.left⟩


open scoped Classical in
instance finBicone [Fintype J] : Fintype (Bicone J) where
  elems := [Bicone.left, Bicone.right].toFinset ∪ Finset.image Bicone.diagram Fintype.elems
  complete j := by
    /-
      J : Type u₁
      inst✝ : Fintype J
      j : CategoryTheory.Bicone J
      ⊢ Membership.mem (Union.union (List.cons CategoryTheory.Bicone.left (List.cons …
    -/
                /-
                  🎉 no goals
                -/
                /-
                  🎉 no goals
                -/
    cases j <;> simp [Fintype.complete]
                /-
                  🎉 no goals
                -/


/-- The homs for a walking `Bicone J`. -/
inductive BiconeHom : Bicone J → Bicone J → Type max u₁ v₁
  | left_id : BiconeHom Bicone.left Bicone.left
  | right_id : BiconeHom Bicone.right Bicone.right
  | left (j : J) : BiconeHom Bicone.left (Bicone.diagram j)
  | right (j : J) : BiconeHom Bicone.right (Bicone.diagram j)
  | diagram {j k : J} (f : j ⟶ k) : BiconeHom (Bicone.diagram j) (Bicone.diagram k)


instance : Inhabited (BiconeHom J Bicone.left Bicone.left) :=
  ⟨BiconeHom.left_id⟩


instance BiconeHom.decidableEq {j k : Bicone J} : DecidableEq (BiconeHom J j k) := fun f g => by
  /-
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    j k : CategoryTheory.Bicone J
    f g : CategoryTheory.BiconeHom J j k
    ⊢ Decidable (Eq f g)
  -/
  classical cases f <;> cases g <;> simp only [diagram.injEq] <;> infer_instance
  /-
    🎉 no goals
  -/


@[simps]
instance biconeCategoryStruct : CategoryStruct (Bicone J) where
  Hom := BiconeHom J
  id j := Bicone.casesOn j BiconeHom.left_id BiconeHom.right_id fun k => BiconeHom.diagram (𝟙 k)
  comp f g := by
    /-
      J : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} J
      X✝ Y✝ Z✝ : CategoryTheory.Bicone J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Quiver.Hom X✝ Z✝
    -/
    rcases f with (_ | _ | _ | _ | f)
      /-
        case left_id
        J : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} J
        Z✝ : CategoryTheory.Bicone J
        g : Quiver.Hom CategoryTheory.Bicone.left Z✝
        ⊢ Quiver.Hom CategoryTheory.Bicone.left Z✝
      -/
    · exact g
      /-
        🎉 no goals
      -/
      /-
        case right_id
        J : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} J
        Z✝ : CategoryTheory.Bicone J
        g : Quiver.Hom CategoryTheory.Bicone.right Z✝
        ⊢ Quiver.Hom CategoryTheory.Bicone.right Z✝
      -/
    · exact g
      /-
        🎉 no goals
      -/
      /-
        case left
        J : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} J
        Z✝ : CategoryTheory.Bicone J
        j✝ : J
        g : Quiver.Hom (CategoryTheory.Bicone.diagram j✝) Z✝
        ⊢ Quiver.Hom CategoryTheory.Bicone.left Z✝
      -/
    · cases g
      /-
        case left.diagram
        J : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} J
        j✝ k✝ : J
        f✝ : Quiver.Hom j✝ k✝
        ⊢ Quiver.Hom CategoryTheory.Bicone.left (CategoryTheory.Bicone.diagram k✝)
      -/
      apply BiconeHom.left
      /-
        🎉 no goals
      -/
      /-
        case right
        J : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} J
        Z✝ : CategoryTheory.Bicone J
        j✝ : J
        g : Quiver.Hom (CategoryTheory.Bicone.diagram j✝) Z✝
        ⊢ Quiver.Hom CategoryTheory.Bicone.right Z✝
      -/
    · cases g
      /-
        case right.diagram
        J : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} J
        j✝ k✝ : J
        f✝ : Quiver.Hom j✝ k✝
        ⊢ Quiver.Hom CategoryTheory.Bicone.right (CategoryTheory.Bicone.diagram k✝)
      -/
      apply BiconeHom.right
      /-
        🎉 no goals
      -/
      /-
        case diagram
        J : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} J
        Z✝ : CategoryTheory.Bicone J
        j✝ k✝ : J
        f : Quiver.Hom j✝ k✝
        g : Quiver.Hom (CategoryTheory.Bicone.diagram k✝) Z✝
        ⊢ Quiver.Hom (CategoryTheory.Bicone.diagram j✝) Z✝
      -/
    · rcases g with (_|_|_|_|g)
      /-
        case diagram.diagram
        J : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} J
        j✝ k✝¹ : J
        f : Quiver.Hom j✝ k✝¹
        k✝ : J
        g : Quiver.Hom k✝¹ k✝
        ⊢ Quiver.Hom (CategoryTheory.Bicone.diagram j✝) (CategoryTheory.Bicone.diagram …
      -/
      exact BiconeHom.diagram (f ≫ g)
      /-
        🎉 no goals
      -/


instance biconeCategory : Category (Bicone J) where
                  /-
                    J : Type u₁
                    inst✝ : CategoryTheory.Category.{v₁, u₁} J
                    X✝ Y✝ : CategoryTheory.Bicone J
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
                  -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
  id_comp f := by cases f <;> simp
                              /-
                                🎉 no goals
                              -/
                  /-
                    J : Type u₁
                    inst✝ : CategoryTheory.Category.{v₁, u₁} J
                    X✝ Y✝ : CategoryTheory.Bicone J
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                  -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
  comp_id f := by cases f <;> simp
                              /-
                                🎉 no goals
                              -/
                    /-
                      J : Type u₁
                      inst✝ : CategoryTheory.Category.{v₁, u₁} J
                      W✝ X✝ Y✝ Z✝ : CategoryTheory.Bicone J
                      f : Quiver.Hom W✝ X✝
                      g : Quiver.Hom X✝ Y✝
                      h : Quiver.Hom Y✝ Z✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                    -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  assoc f g h := by cases f <;> cases g <;> cases h <;> simp
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Given a diagram `F : J ⥤ C` and two `Cone F`s, we can join them into a diagram `Bicone J ⥤ C`.
-/
@[simps]
def biconeMk {C : Type u₁} [Category.{v₁} C] {F : J ⥤ C} (c₁ c₂ : Cone F) : Bicone J ⥤ C where
  obj X := Bicone.casesOn X c₁.pt c₂.pt fun j => F.obj j
  map f := by
    /-
      J : Type v₁
      inst✝¹ : CategoryTheory.SmallCategory J
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor J C
      c₁ c₂ : CategoryTheory.Limits.Cone F
      X✝ Y✝ : CategoryTheory.Bicone J
      f : Quiver.Hom X✝ Y✝
      ⊢ Quiver.Hom ((fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
    -/
    rcases f with (_|_|_|_|f)
      /-
        case left_id
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        ⊢ Quiver.Hom ((fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · exact 𝟙 _
      /-
        🎉 no goals
      -/
      /-
        case right_id
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        ⊢ Quiver.Hom ((fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · exact 𝟙 _
      /-
        🎉 no goals
      -/
      /-
        case left
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        j✝ : J
        ⊢ Quiver.Hom ((fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · exact c₁.π.app _
      /-
        🎉 no goals
      -/
      /-
        case right
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        j✝ : J
        ⊢ Quiver.Hom ((fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · exact c₂.π.app _
      /-
        🎉 no goals
      -/
      /-
        case diagram
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        j✝ k✝ : J
        f : Quiver.Hom j✝ k✝
        ⊢ Quiver.Hom ((fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · exact F.map f
      /-
        🎉 no goals
      -/
                 /-
                   J : Type v₁
                   inst✝¹ : CategoryTheory.SmallCategory J
                   C : Type u₁
                   inst✝ : CategoryTheory.Category.{v₁, u₁} C
                   F : CategoryTheory.Functor J C
                   c₁ c₂ : CategoryTheory.Limits.Cone F
                   X : CategoryTheory.Bicone J
                   ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
                 -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
  map_id X := by cases X <;> simp
                             /-
                               🎉 no goals
                             -/
  map_comp f g := by
    /-
      J : Type v₁
      inst✝¹ : CategoryTheory.SmallCategory J
      C : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor J C
      c₁ c₂ : CategoryTheory.Limits.Cone F
      X✝ Y✝ Z✝ : CategoryTheory.Bicone J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
    -/
    rcases f with (_|_|_|_|_)
      /-
        case left_id
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        Z✝ : CategoryTheory.Bicone J
        g : Quiver.Hom CategoryTheory.Bicone.left Z✝
        ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · exact (Category.id_comp _).symm
      /-
        🎉 no goals
      -/
      /-
        case right_id
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        Z✝ : CategoryTheory.Bicone J
        g : Quiver.Hom CategoryTheory.Bicone.right Z✝
        ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · exact (Category.id_comp _).symm
      /-
        🎉 no goals
      -/
      /-
        case left
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        Z✝ : CategoryTheory.Bicone J
        j✝ : J
        g : Quiver.Hom (CategoryTheory.Bicone.diagram j✝) Z✝
        ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · cases g
      /-
        case left.diagram
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        j✝ k✝ : J
        f✝ : Quiver.Hom j✝ k✝
        ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
      exact (Category.id_comp _).symm.trans (c₁.π.naturality _)
      /-
        🎉 no goals
      -/
      /-
        case right
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        Z✝ : CategoryTheory.Bicone J
        j✝ : J
        g : Quiver.Hom (CategoryTheory.Bicone.diagram j✝) Z✝
        ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · cases g
      /-
        case right.diagram
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        j✝ k✝ : J
        f✝ : Quiver.Hom j✝ k✝
        ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
      exact (Category.id_comp _).symm.trans (c₂.π.naturality _)
      /-
        🎉 no goals
      -/
      /-
        case diagram
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        Z✝ : CategoryTheory.Bicone J
        j✝ k✝ : J
        f✝ : Quiver.Hom j✝ k✝
        g : Quiver.Hom (CategoryTheory.Bicone.diagram k✝) Z✝
        ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
    · cases g
      /-
        case diagram.diagram
        J : Type v₁
        inst✝¹ : CategoryTheory.SmallCategory J
        C : Type u₁
        inst✝ : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor J C
        c₁ c₂ : CategoryTheory.Limits.Cone F
        j✝ k✝¹ : J
        f✝¹ : Quiver.Hom j✝ k✝¹
        k✝ : J
        f✝ : Quiver.Hom k✝¹ k✝
        ⊢ Eq ({ obj := fun X => CategoryTheory.Bicone.casesOn X c₁.pt c₂.pt fun j => F …
      -/
      apply F.map_comp
      /-
        🎉 no goals
      -/


open scoped Classical in
instance finBiconeHom [FinCategory J] (j k : Bicone J) : Fintype (j ⟶ k) := by
  /-
    J : Type v₁
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.FinCategory J
    j k : CategoryTheory.Bicone J
    ⊢ Fintype (Quiver.Hom j k)
  -/
  cases j <;> cases k
  · exact
      { elems := {BiconeHom.left_id}
        complete := fun f => by cases f; simp }
  · exact
    { elems := ∅
      complete := fun f => by cases f }
  · exact
    { elems := {BiconeHom.left _}
      complete := fun f => by cases f; simp }
  · exact
    { elems := ∅
      complete := fun f => by cases f }
  · exact
      { elems := {BiconeHom.right_id}
        complete := fun f => by cases f; simp }
  · exact
    { elems := {BiconeHom.right _}
      complete := fun f => by cases f; simp }
  · exact
    { elems := ∅
      complete := fun f => by cases f }
  · exact
    { elems := ∅
      complete := fun f => by cases f }
  · exact
    { elems := Finset.image BiconeHom.diagram Fintype.elems
      complete := fun f => by
        rcases f with (_|_|_|_|f)
        simp only [Finset.mem_image]
        use f
        simpa using Fintype.complete _ }


instance biconeSmallCategory : SmallCategory (Bicone J) :=
  CategoryTheory.biconeCategory J


instance biconeFinCategory [FinCategory J] : FinCategory (Bicone J) where


