/-- A structure carrying a diagram of cones over the functors `F.obj j`.
-/
structure DiagramOfCones where
  /-- For each object, a cone. -/
  obj : ∀ j : J, Cone (F.obj j)
  /-- For each map, a map of cones. -/
  map : ∀ {j j' : J} (f : j ⟶ j'), (Cones.postcompose (F.map f)).obj (obj j) ⟶ obj j'
  id : ∀ j : J, (map (𝟙 j)).hom = 𝟙 _ := by aesop_cat
  comp : ∀ {j₁ j₂ j₃ : J} (f : j₁ ⟶ j₂) (g : j₂ ⟶ j₃),
    (map (f ≫ g)).hom = (map f).hom ≫ (map g).hom := by aesop_cat


/-- A structure carrying a diagram of cocones over the functors `F.obj j`.
-/
structure DiagramOfCocones where
  /-- For each object, a cocone. -/
  obj : ∀ j : J, Cocone (F.obj j)
  /-- For each map, a map of cocones. -/
  map : ∀ {j j' : J} (f : j ⟶ j'), (obj j) ⟶ (Cocones.precompose (F.map f)).obj (obj j')
  id : ∀ j : J, (map (𝟙 j)).hom = 𝟙 _ := by aesop_cat
  comp : ∀ {j₁ j₂ j₃ : J} (f : j₁ ⟶ j₂) (g : j₂ ⟶ j₃),
    (map (f ≫ g)).hom = (map f).hom ≫ (map g).hom := by aesop_cat


/-- Extract the functor `J ⥤ C` consisting of the cone points and the maps between them,
from a `DiagramOfCones`.
-/
@[simps]
def DiagramOfCones.conePoints (D : DiagramOfCones F) : J ⥤ C where
  obj j := (D.obj j).pt
  map f := (D.map f).hom
  map_id j := D.id j
  map_comp f g := D.comp f g


/-- Extract the functor `J ⥤ C` consisting of the cocone points and the maps between them,
from a `DiagramOfCocones`.
-/
@[simps]
def DiagramOfCocones.coconePoints (D : DiagramOfCocones F) : J ⥤ C where
  obj j := (D.obj j).pt
  map f := (D.map f).hom
  map_id j := D.id j
  map_comp f g := D.comp f g


/-- Given a diagram `D` of limit cones over the `F.obj j`, and a cone over `uncurry.obj F`,
we can construct a cone over the diagram consisting of the cone points from `D`.
-/
@[simps]
def coneOfConeUncurry {D : DiagramOfCones F} (Q : ∀ j, IsLimit (D.obj j))
    (c : Cone (uncurry.obj F)) : Cone D.conePoints where
  pt := c.pt
  π :=
    { app := fun j =>
        (Q j).lift
          { pt := c.pt
            π :=
              { app := fun k => c.π.app (j, k)
                naturality := fun k k' f => by
                  /-
                    J K : Type v
                    inst✝² : CategoryTheory.SmallCategory J
                    inst✝¹ : CategoryTheory.SmallCategory K
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                    D : CategoryTheory.Limits.DiagramOfCones F
                    Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                    j : J
                    k k' : K
                    f : Quiver.Hom k k'
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const K).ob …
                  -/
                  dsimp; simp only [Category.id_comp]
                  /-
                    J K : Type v
                    inst✝² : CategoryTheory.SmallCategory J
                    inst✝¹ : CategoryTheory.SmallCategory K
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                    D : CategoryTheory.Limits.DiagramOfCones F
                    Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                    j : J
                    k k' : K
                    f : Quiver.Hom k k'
                    ⊢ Eq (c.π.app { fst := j, snd := k' }) (CategoryTheory.CategoryStruct.comp (c. …
                  -/
                  have := @NatTrans.naturality _ _ _ _ _ _ c.π (j, k) (j, k') (𝟙 j, f)
                  /-
                    J K : Type v
                    inst✝² : CategoryTheory.SmallCategory J
                    inst✝¹ : CategoryTheory.SmallCategory K
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                    D : CategoryTheory.Limits.DiagramOfCones F
                    Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                    j : J
                    k k' : K
                    f : Quiver.Hom k k'
                    this : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const  …
                    ⊢ Eq (c.π.app { fst := j, snd := k' }) (CategoryTheory.CategoryStruct.comp (c. …
                  -/
                  dsimp at this
                  simp? at this says
                    simp only [Category.id_comp, Functor.map_id, NatTrans.id_app] at this
                  /-
                    J K : Type v
                    inst✝² : CategoryTheory.SmallCategory J
                    inst✝¹ : CategoryTheory.SmallCategory K
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                    D : CategoryTheory.Limits.DiagramOfCones F
                    Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                    j : J
                    k k' : K
                    f : Quiver.Hom k k'
                    this : Eq (c.π.app { fst := j, snd := k' }) (CategoryTheory.CategoryStruct.com …
                    ⊢ Eq (c.π.app { fst := j, snd := k' }) (CategoryTheory.CategoryStruct.comp (c. …
                  -/
                  exact this } }
                  /-
                    🎉 no goals
                  -/
      naturality := fun j j' f =>
        (Q j').hom_ext
          (by
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCones F
              Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
              c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              ⊢ ∀ (j_1 : K), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
            -/
            dsimp
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCones F
              Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
              c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              ⊢ ∀ (j_1 : K), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
            -/
            intro k
            simp only [Limits.ConeMorphism.w, Limits.Cones.postcompose_obj_π,
              Limits.IsLimit.fac_assoc, Limits.IsLimit.fac, NatTrans.comp_app, Category.id_comp,
              Category.assoc]
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCones F
              Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
              c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              k : K
              ⊢ Eq (c.π.app { fst := j', snd := k }) (CategoryTheory.CategoryStruct.comp (c. …
            -/
            have := @NatTrans.naturality _ _ _ _ _ _ c.π (j, k) (j', k) (f, 𝟙 k)
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCones F
              Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
              c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              k : K
              this : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const  …
              ⊢ Eq (c.π.app { fst := j', snd := k }) (CategoryTheory.CategoryStruct.comp (c. …
            -/
            dsimp at this
            simp only [Category.id_comp, Category.comp_id, CategoryTheory.Functor.map_id,
              NatTrans.id_app] at this
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCones F
              Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
              c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              k : K
              this : Eq (c.π.app { fst := j', snd := k }) (CategoryTheory.CategoryStruct.com …
              ⊢ Eq (c.π.app { fst := j', snd := k }) (CategoryTheory.CategoryStruct.comp (c. …
            -/
            exact this) }
            /-
              🎉 no goals
            -/


/-- Given a diagram `D` of colimit cocones over the `F.obj j`, and a cocone over `uncurry.obj F`,
we can construct a cocone over the diagram consisting of the cocone points from `D`.
-/
@[simps]
def coconeOfCoconeUncurry {D : DiagramOfCocones F} (Q : ∀ j, IsColimit (D.obj j))
    (c : Cocone (uncurry.obj F)) : Cocone D.coconePoints where
  pt := c.pt
  ι :=
    { app := fun j =>
        (Q j).desc
          { pt := c.pt
            ι :=
              { app := fun k => c.ι.app (j, k)
                naturality := fun k k' f => by
                  /-
                    J K : Type v
                    inst✝² : CategoryTheory.SmallCategory J
                    inst✝¹ : CategoryTheory.SmallCategory K
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                    D : CategoryTheory.Limits.DiagramOfCocones F
                    Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                    j : J
                    k k' : K
                    f : Quiver.Hom k k'
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) ((fun k => c.ι.app  …
                  -/
                  dsimp; simp only [Category.comp_id]
                  conv_lhs =>
                    arg 1; equals (F.map (𝟙 _)).app _ ≫  (F.obj j).map f =>
                      simp
                  /-
                    J K : Type v
                    inst✝² : CategoryTheory.SmallCategory J
                    inst✝¹ : CategoryTheory.SmallCategory K
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                    D : CategoryTheory.Limits.DiagramOfCocones F
                    Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                    j : J
                    k k' : K
                    f : Quiver.Hom k k'
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                  -/
                  conv_lhs => arg 1; rw [← uncurry_obj_map F ((𝟙 j,f) : (j,k) ⟶ (j,k'))]
                  /-
                    J K : Type v
                    inst✝² : CategoryTheory.SmallCategory J
                    inst✝¹ : CategoryTheory.SmallCategory K
                    C : Type u
                    inst✝ : CategoryTheory.Category.{v, u} C
                    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                    D : CategoryTheory.Limits.DiagramOfCocones F
                    Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                    j : J
                    k k' : K
                    f : Quiver.Hom k k'
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.uncurry.obj F).map { …
                  -/
                  rw [c.w] } }
                  /-
                    🎉 no goals
                  -/
      naturality := fun j j' f =>
        (Q j).hom_ext
          (by
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCocones F
              Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
              c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              ⊢ ∀ (j_1 : K), Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app j_1) (C …
            -/
            dsimp
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCocones F
              Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
              c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              ⊢ ∀ (j_1 : K), Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app j_1) (C …
            -/
            intro k
            simp only [Limits.CoconeMorphism.w_assoc, Limits.Cocones.precompose_obj_ι,
              Limits.IsColimit.fac_assoc, Limits.IsColimit.fac, NatTrans.comp_app, Category.comp_id,
              Category.assoc]
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCocones F
              Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
              c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              k : K
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f).app k) (c.ι.app { fst := j …
            -/
            have := @NatTrans.naturality _ _ _ _ _ _ c.ι (j, k) (j', k) (f, 𝟙 k)
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCocones F
              Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
              c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              k : K
              this : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.uncurry.obj F). …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f).app k) (c.ι.app { fst := j …
            -/
            dsimp at this
            simp only [Category.id_comp, Category.comp_id, CategoryTheory.Functor.map_id,
              NatTrans.id_app] at this
            /-
              J K : Type v
              inst✝² : CategoryTheory.SmallCategory J
              inst✝¹ : CategoryTheory.SmallCategory K
              C : Type u
              inst✝ : CategoryTheory.Category.{v, u} C
              F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
              D : CategoryTheory.Limits.DiagramOfCocones F
              Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
              c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
              j j' : J
              f : Quiver.Hom j j'
              k : K
              this : Eq (CategoryTheory.CategoryStruct.comp ((F.map f).app k) (c.ι.app { fst …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f).app k) (c.ι.app { fst := j …
            -/
            exact this) }
            /-
              🎉 no goals
            -/


/-- `coneOfConeUncurry Q c` is a limit cone when `c` is a limit cone.
-/
def coneOfConeUncurryIsLimit {D : DiagramOfCones F} (Q : ∀ j, IsLimit (D.obj j))
    {c : Cone (uncurry.obj F)} (P : IsLimit c) : IsLimit (coneOfConeUncurry Q c) where
  lift s :=
    P.lift
      { pt := s.pt
        π :=
          { app := fun p => s.π.app p.1 ≫ (D.obj p.1).π.app p.2
            naturality := fun p p' f => by
              /-
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                p p' : Prod J K
                f : Quiver.Hom p p'
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Prod …
              -/
              dsimp; simp only [Category.id_comp, Category.assoc]
              /-
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                p p' : Prod J K
                f : Quiver.Hom p p'
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app p'.1) ((D.obj p'.1).π.app p' …
              -/
              rcases p with ⟨j, k⟩
              /-
                case mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                p' : Prod J K
                j : J
                k : K
                f : Quiver.Hom { fst := j, snd := k } p'
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app p'.1) ((D.obj p'.1).π.app p' …
              -/
              rcases p' with ⟨j', k'⟩
              /-
                case mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                f : Quiver.Hom { fst := j, snd := k } { fst := j', snd := k' }
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app { fst := j', snd := k' }.1)  …
              -/
              rcases f with ⟨fj, fk⟩
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app { fst := j', snd := k' }.1)  …
              -/
              dsimp
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              slice_rhs 3 4 => rw [← NatTrans.naturality]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              slice_rhs 2 3 => rw [← (D.obj j).π.naturality]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              simp only [Functor.const_obj_map, Category.id_comp, Category.assoc]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              have w := (D.map fj).w k'
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom ((D.obj { fst := j', …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              dsimp at w
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom ((D.obj j').π.app k' …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              rw [← w]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom ((D.obj j').π.app k' …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              have n := s.π.naturality fj
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom ((D.obj j').π.app k' …
                n : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J). …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              dsimp at n
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom ((D.obj j').π.app k' …
                n : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id s …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              simp only [Category.id_comp] at n
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom ((D.obj j').π.app k' …
                n : Eq (s.π.app j') (CategoryTheory.CategoryStruct.comp (s.π.app j) (D.map fj) …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app j') ((D.obj j').π.app k')) ( …
              -/
              rw [n]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCones F
                Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
                c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsLimit c
                s : CategoryTheory.Limits.Cone D.conePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom ((D.obj j').π.app k' …
                n : Eq (s.π.app j') (CategoryTheory.CategoryStruct.comp (s.π.app j) (D.map fj) …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              simp } }
              /-
                🎉 no goals
              -/
  fac s j := by
    /-
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCones F
      Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
      c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D.conePoints
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => P.lift { pt := s.pt, π :=  …
    -/
    apply (Q j).hom_ext
    /-
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCones F
      Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
      c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D.conePoints
      j : J
      ⊢ ∀ (j_1 : K), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
    -/
    intro k
    /-
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCones F
      Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
      c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D.conePoints
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp
    /-
      🎉 no goals
    -/
  uniq s m w := by
    refine P.uniq
      { pt := s.pt
        π := _ } m ?_
    /-
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCones F
      Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
      c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D.conePoints
      m : Quiver.Hom s.pt (CategoryTheory.Limits.coneOfConeUncurry Q c).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      ⊢ ∀ (j : Prod J K), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) ({ p …
    -/
    rintro ⟨j, k⟩
    /-
      case mk
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCones F
      Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
      c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D.conePoints
      m : Quiver.Hom s.pt (CategoryTheory.Limits.coneOfConeUncurry Q c).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (c.π.app { fst := j, snd := k })) ( …
    -/
    dsimp
    /-
      case mk
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCones F
      Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
      c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D.conePoints
      m : Quiver.Hom s.pt (CategoryTheory.Limits.coneOfConeUncurry Q c).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (c.π.app { fst := j, snd := k })) ( …
    -/
    rw [← w j]
    /-
      case mk
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCones F
      Q : (j : J) → CategoryTheory.Limits.IsLimit (D.obj j)
      c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone D.conePoints
      m : Quiver.Hom s.pt (CategoryTheory.Limits.coneOfConeUncurry Q c).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m (c.π.app { fst := j, snd := k })) ( …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- `coconeOfCoconeUncurry Q c` is a colimit cocone when `c` is a colimit cocone.
-/
def coconeOfCoconeUncurryIsColimit {D : DiagramOfCocones F} (Q : ∀ j, IsColimit (D.obj j))
    {c : Cocone (uncurry.obj F)} (P : IsColimit c) : IsColimit (coconeOfCoconeUncurry Q c) where
  desc s :=
    P.desc
      { pt := s.pt
        ι :=
          { app := fun p => (D.obj p.1).ι.app p.2 ≫ s.ι.app p.1
            naturality := fun p p' f => by
              /-
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                p p' : Prod J K
                f : Quiver.Hom p p'
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.uncurry.obj F).map f …
              -/
              dsimp; simp only [Category.id_comp, Category.assoc]
              /-
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                p p' : Prod J K
                f : Quiver.Hom p p'
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f.1).app p.2) (CategoryTheory …
              -/
              rcases p with ⟨j, k⟩
              /-
                case mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                p' : Prod J K
                j : J
                k : K
                f : Quiver.Hom { fst := j, snd := k } p'
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f.1).app { fst := j, snd := k …
              -/
              rcases p' with ⟨j', k'⟩
              /-
                case mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                f : Quiver.Hom { fst := j, snd := k } { fst := j', snd := k' }
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f.1).app { fst := j, snd := k …
              -/
              rcases f with ⟨fj, fk⟩
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map { fst := fj, snd := fk }.1).a …
              -/
              dsimp
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map fj).app k) (CategoryTheory.Ca …
              -/
              slice_lhs 2 3 => rw [(D.obj j').ι.naturality]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map fj).app k) (CategoryTheory.Ca …
              -/
              simp only [Functor.const_obj_map, Category.id_comp, Category.assoc]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map fj).app k) (CategoryTheory.Ca …
              -/
              have w := (D.map fj).w k
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp ((D.obj { fst := j, snd := k }.1).ι …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map fj).app k) (CategoryTheory.Ca …
              -/
              dsimp at w
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app k) (D.map fj).hom) …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map fj).app k) (CategoryTheory.Ca …
              -/
              slice_lhs 1 2 => rw [← w]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app k) (D.map fj).hom) …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              have n := s.ι.naturality fj
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app k) (D.map fj).hom) …
                n : Eq (CategoryTheory.CategoryStruct.comp (D.coconePoints.map fj) (s.ι.app {  …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              dsimp at n
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app k) (D.map fj).hom) …
                n : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom (s.ι.app j')) (Categ …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              simp only [Category.comp_id] at n
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app k) (D.map fj).hom) …
                n : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom (s.ι.app j')) (s.ι.a …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              rw [← n]
              /-
                case mk.mk.mk
                J K : Type v
                inst✝² : CategoryTheory.SmallCategory J
                inst✝¹ : CategoryTheory.SmallCategory K
                C : Type u
                inst✝ : CategoryTheory.Category.{v, u} C
                F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                D : CategoryTheory.Limits.DiagramOfCocones F
                Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
                c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
                P : CategoryTheory.Limits.IsColimit c
                s : CategoryTheory.Limits.Cocone D.coconePoints
                j : J
                k : K
                j' : J
                k' : K
                fj : Quiver.Hom { fst := j, snd := k }.1 { fst := j', snd := k' }.1
                fk : Quiver.Hom { fst := j, snd := k }.2 { fst := j', snd := k' }.2
                w : Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app k) (D.map fj).hom) …
                n : Eq (CategoryTheory.CategoryStruct.comp (D.map fj).hom (s.ι.app j')) (s.ι.a …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              simp } }
              /-
                🎉 no goals
              -/
  fac s j := by
    /-
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCocones F
      Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
      c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D.coconePoints
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.coconeOfCocon …
    -/
    apply (Q j).hom_ext
    /-
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCocones F
      Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
      c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D.coconePoints
      j : J
      ⊢ ∀ (j_1 : K), Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app j_1) (C …
    -/
    intro k
    /-
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCocones F
      Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
      c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D.coconePoints
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.obj j).ι.app k) (CategoryTheory.C …
    -/
    simp
    /-
      🎉 no goals
    -/
  uniq s m w := by
    refine P.uniq
      { pt := s.pt
        ι := _ } m ?_
    /-
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCocones F
      Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
      c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D.coconePoints
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfCoconeUncurry Q c).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ ∀ (j : Prod J K), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) ({ p …
    -/
    rintro ⟨j, k⟩
    /-
      case mk
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCocones F
      Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
      c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D.coconePoints
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfCoconeUncurry Q c).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { fst := j, snd := k }) m) ( …
    -/
    dsimp
    /-
      case mk
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCocones F
      Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
      c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D.coconePoints
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfCoconeUncurry Q c).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { fst := j, snd := k }) m) ( …
    -/
    rw [← w j]
    /-
      case mk
      J K : Type v
      inst✝² : CategoryTheory.SmallCategory J
      inst✝¹ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      D : CategoryTheory.Limits.DiagramOfCocones F
      Q : (j : J) → CategoryTheory.Limits.IsColimit (D.obj j)
      c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F)
      P : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone D.coconePoints
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfCoconeUncurry Q c).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      j : J
      k : K
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app { fst := j, snd := k }) m) ( …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Given a functor `F : J ⥤ K ⥤ C`, with all needed limits,
we can construct a diagram consisting of the limit cone over each functor `F.obj j`,
and the universal cone morphisms between these.
-/
@[simps]
noncomputable def DiagramOfCones.mkOfHasLimits : DiagramOfCones F where
  obj j := limit.cone (F.obj j)
  map f := { hom := lim.map (F.map f) }

-- Satisfying the inhabited linter.

noncomputable instance diagramOfConesInhabited : Inhabited (DiagramOfCones F) :=
  ⟨DiagramOfCones.mkOfHasLimits F⟩


@[simp]
theorem DiagramOfCones.mkOfHasLimits_conePoints :
    (DiagramOfCones.mkOfHasLimits F).conePoints = F ⋙ lim :=
  rfl


/-- The Fubini theorem for a functor `F : J ⥤ K ⥤ C`,
showing that the limit of `uncurry.obj F` can be computed as
the limit of the limits of the functors `F.obj j`.
-/
noncomputable def limitUncurryIsoLimitCompLim : limit (uncurry.obj F) ≅ limit (F ⋙ lim) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (CategoryTheory.uncurry.obj  …
  -/
  let c := limit.cone (uncurry.obj F)
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F) := CategoryTheor …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (CategoryTheory.uncurry.obj  …
  -/
  let P : IsLimit c := limit.isLimit _
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F) := CategoryTheor …
    P : CategoryTheory.Limits.IsLimit c := CategoryTheory.Limits.limit.isLimit (Ca …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (CategoryTheory.uncurry.obj  …
  -/
  let G := DiagramOfCones.mkOfHasLimits F
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F) := CategoryTheor …
    P : CategoryTheory.Limits.IsLimit c := CategoryTheory.Limits.limit.isLimit (Ca …
    G : CategoryTheory.Limits.DiagramOfCones F := CategoryTheory.Limits.DiagramOfC …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (CategoryTheory.uncurry.obj  …
  -/
  let Q : ∀ j, IsLimit (G.obj j) := fun j => limit.isLimit _
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F) := CategoryTheor …
    P : CategoryTheory.Limits.IsLimit c := CategoryTheory.Limits.limit.isLimit (Ca …
    G : CategoryTheory.Limits.DiagramOfCones F := CategoryTheory.Limits.DiagramOfC …
    Q : (j : J) → CategoryTheory.Limits.IsLimit (G.obj j) := fun j => CategoryTheo …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (CategoryTheory.uncurry.obj  …
  -/
  have Q' := coneOfConeUncurryIsLimit Q P
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F) := CategoryTheor …
    P : CategoryTheory.Limits.IsLimit c := CategoryTheory.Limits.limit.isLimit (Ca …
    G : CategoryTheory.Limits.DiagramOfCones F := CategoryTheory.Limits.DiagramOfC …
    Q : (j : J) → CategoryTheory.Limits.IsLimit (G.obj j) := fun j => CategoryTheo …
    Q' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.coneOfConeUncurry Q c)
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (CategoryTheory.uncurry.obj  …
  -/
  have Q'' := limit.isLimit (F ⋙ lim)
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    c : CategoryTheory.Limits.Cone (CategoryTheory.uncurry.obj F) := CategoryTheor …
    P : CategoryTheory.Limits.IsLimit c := CategoryTheory.Limits.limit.isLimit (Ca …
    G : CategoryTheory.Limits.DiagramOfCones F := CategoryTheory.Limits.DiagramOfC …
    Q : (j : J) → CategoryTheory.Limits.IsLimit (G.obj j) := fun j => CategoryTheo …
    Q' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.coneOfConeUncurry Q c)
    Q'' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.limit.cone (F.comp  …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (CategoryTheory.uncurry.obj  …
  -/
  exact IsLimit.conePointUniqueUpToIso Q' Q''
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem limitUncurryIsoLimitCompLim_hom_π_π {j} {k} :
    (limitUncurryIsoLimitCompLim F).hom ≫ limit.π _ j ≫ limit.π _ k = limit.π _ (j, k) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitUncurryIs …
  -/
  dsimp [limitUncurryIsoLimitCompLim, IsLimit.conePointUniqueUpToIso, IsLimit.uniqueUpToIso]
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift (F. …
  -/
  simp
  /-
    🎉 no goals
  -/

-- Porting note: Added type annotation `limit (_ ⋙ lim) ⟶ _`

@[simp, reassoc]
theorem limitUncurryIsoLimitCompLim_inv_π {j} {k} :
    (limitUncurryIsoLimitCompLim F).inv ≫ limit.π _ (j, k) =
      (limit.π _ j ≫ limit.π _ k : limit (_ ⋙ lim) ⟶ _) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitUncurryIs …
  -/
  rw [← cancel_epi (limitUncurryIsoLimitCompLim F).hom]
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasLimit (F.comp CategoryTheory.Limits.lim)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitUncurryIs …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a functor `F : J ⥤ K ⥤ C`, with all needed colimits,
we can construct a diagram consisting of the colimit cocone over each functor `F.obj j`,
and the universal cocone morphisms between these.
-/
@[simps]
noncomputable def DiagramOfCocones.mkOfHasColimits : DiagramOfCocones F where
  obj j := colimit.cocone (F.obj j)
  map f := { hom := colim.map (F.map f) }

-- Satisfying the inhabited linter.

noncomputable instance diagramOfCoconesInhabited : Inhabited (DiagramOfCocones F) :=
  ⟨DiagramOfCocones.mkOfHasColimits F⟩


@[simp]
theorem DiagramOfCocones.mkOfHasColimits_coconePoints :
    (DiagramOfCocones.mkOfHasColimits F).coconePoints = F ⋙ colim :=
  rfl


/-- The Fubini theorem for a functor `F : J ⥤ K ⥤ C`,
showing that the colimit of `uncurry.obj F` can be computed as
the colimit of the colimits of the functors `F.obj j`.
-/
noncomputable def colimitUncurryIsoColimitCompColim :
    colimit (uncurry.obj F) ≅ colimit (F ⋙ colim) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.uncurry.ob …
  -/
  let c := colimit.cocone (uncurry.obj F)
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F) := CategoryThe …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.uncurry.ob …
  -/
  let P : IsColimit c := colimit.isColimit _
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F) := CategoryThe …
    P : CategoryTheory.Limits.IsColimit c := CategoryTheory.Limits.colimit.isColim …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.uncurry.ob …
  -/
  let G := DiagramOfCocones.mkOfHasColimits F
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F) := CategoryThe …
    P : CategoryTheory.Limits.IsColimit c := CategoryTheory.Limits.colimit.isColim …
    G : CategoryTheory.Limits.DiagramOfCocones F := CategoryTheory.Limits.DiagramO …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.uncurry.ob …
  -/
  let Q : ∀ j, IsColimit (G.obj j) := fun j => colimit.isColimit _
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F) := CategoryThe …
    P : CategoryTheory.Limits.IsColimit c := CategoryTheory.Limits.colimit.isColim …
    G : CategoryTheory.Limits.DiagramOfCocones F := CategoryTheory.Limits.DiagramO …
    Q : (j : J) → CategoryTheory.Limits.IsColimit (G.obj j) := fun j => CategoryTh …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.uncurry.ob …
  -/
  have Q' := coconeOfCoconeUncurryIsColimit Q P
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F) := CategoryThe …
    P : CategoryTheory.Limits.IsColimit c := CategoryTheory.Limits.colimit.isColim …
    G : CategoryTheory.Limits.DiagramOfCocones F := CategoryTheory.Limits.DiagramO …
    Q : (j : J) → CategoryTheory.Limits.IsColimit (G.obj j) := fun j => CategoryTh …
    Q' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.coconeOfCoconeUncu …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.uncurry.ob …
  -/
  have Q'' := colimit.isColimit (F ⋙ colim)
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    c : CategoryTheory.Limits.Cocone (CategoryTheory.uncurry.obj F) := CategoryThe …
    P : CategoryTheory.Limits.IsColimit c := CategoryTheory.Limits.colimit.isColim …
    G : CategoryTheory.Limits.DiagramOfCocones F := CategoryTheory.Limits.DiagramO …
    Q : (j : J) → CategoryTheory.Limits.IsColimit (G.obj j) := fun j => CategoryTh …
    Q' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.coconeOfCoconeUncu …
    Q'' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.colimit.cocone (F …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.uncurry.ob …
  -/
  exact IsColimit.coconePointUniqueUpToIso Q' Q''
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem colimitUncurryIsoColimitCompColim_ι_ι_inv {j} {k} :
    colimit.ι (F.obj j) k ≫ colimit.ι (F ⋙ colim) j ≫ (colimitUncurryIsoColimitCompColim F).inv =
      colimit.ι (uncurry.obj F) (j, k) := by
  dsimp [colimitUncurryIsoColimitCompColim, IsColimit.coconePointUniqueUpToIso,
    IsColimit.uniqueUpToIso]
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.o …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem colimitUncurryIsoColimitCompColim_ι_hom {j} {k} :
    colimit.ι _ (j, k) ≫ (colimitUncurryIsoColimitCompColim F).hom =
      (colimit.ι _ k ≫ colimit.ι (F ⋙ colim) j : _ ⟶ (colimit (F ⋙ colim))) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
  -/
  rw [← cancel_mono (colimitUncurryIsoColimitCompColim F).inv]
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj F)
    inst✝ : CategoryTheory.Limits.HasColimit (F.comp CategoryTheory.Limits.colim)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The limit of `F.flip ⋙ lim` is isomorphic to the limit of `F ⋙ lim`. -/
noncomputable def limitFlipCompLimIsoLimitCompLim : limit (F.flip ⋙ lim) ≅ limit (F ⋙ lim) :=
  (limitUncurryIsoLimitCompLim _).symm ≪≫
    HasLimit.isoOfNatIso (uncurryObjFlip _) ≪≫
      HasLimit.isoOfEquivalence (Prod.braiding _ _)
                                           /-
                                             J K : Type v
                                             inst✝⁶ : CategoryTheory.SmallCategory J
                                             inst✝⁵ : CategoryTheory.SmallCategory K
                                             C : Type u
                                             inst✝⁴ : CategoryTheory.Category.{v, u} C
                                             F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                                             inst✝³ : CategoryTheory.Limits.HasLimitsOfShape J C
                                             inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
                                             inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Prod J K) C
                                             inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Prod K J) C
                                             x✝ : Prod K J
                                             ⊢ CategoryTheory.Iso (((CategoryTheory.Prod.braiding K J).functor.comp (Catego …
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
          (NatIso.ofComponents fun _ => by rfl) ≪≫
           /-
             🎉 no goals
           -/
        limitUncurryIsoLimitCompLim _

-- Porting note: Added type annotation `limit (_ ⋙ lim) ⟶ _`

@[simp, reassoc]
theorem limitFlipCompLimIsoLimitCompLim_hom_π_π (j) (k) :
    (limitFlipCompLimIsoLimitCompLim F).hom ≫ limit.π _ j ≫ limit.π _ k =
      (limit.π _ k ≫ limit.π _ j : limit (_ ⋙ lim) ⟶ _) := by
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Prod K J) C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitFlipCompL …
  -/
  dsimp [limitFlipCompLimIsoLimitCompLim]
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Prod K J) C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [Equivalence.counit]
  /-
    🎉 no goals
  -/

-- Porting note: Added type annotation `limit (_ ⋙ lim) ⟶ _`
-- See note [dsimp, simp]

@[simp, reassoc]
theorem limitFlipCompLimIsoLimitCompLim_inv_π_π (k) (j) :
    (limitFlipCompLimIsoLimitCompLim F).inv ≫ limit.π _ k ≫ limit.π _ j =
      (limit.π _ j ≫ limit.π _ k : limit (_ ⋙ lim) ⟶ _) := by
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Prod K J) C
    k : K
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitFlipCompL …
  -/
  dsimp [limitFlipCompLimIsoLimitCompLim]
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasLimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (Prod K J) C
    k : K
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The colimit of `F.flip ⋙ colim` is isomorphic to the colimit of `F ⋙ colim`. -/
noncomputable def colimitFlipCompColimIsoColimitCompColim :
    colimit (F.flip ⋙ colim) ≅ colimit (F ⋙ colim) :=
  (colimitUncurryIsoColimitCompColim _).symm ≪≫
    HasColimit.isoOfNatIso (uncurryObjFlip _) ≪≫
      HasColimit.isoOfEquivalence (Prod.braiding _ _)
                                           /-
                                             J K : Type v
                                             inst✝⁶ : CategoryTheory.SmallCategory J
                                             inst✝⁵ : CategoryTheory.SmallCategory K
                                             C : Type u
                                             inst✝⁴ : CategoryTheory.Category.{v, u} C
                                             F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                                             inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J C
                                             inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
                                             inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Prod J K) C
                                             inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Prod K J) C
                                             x✝ : Prod K J
                                             ⊢ CategoryTheory.Iso (((CategoryTheory.Prod.braiding K J).functor.comp (Catego …
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
          (NatIso.ofComponents fun _ => by rfl) ≪≫
           /-
             🎉 no goals
           -/
        colimitUncurryIsoColimitCompColim _


@[simp, reassoc]
theorem colimitFlipCompColimIsoColimitCompColim_ι_ι_hom (j) (k) :
    colimit.ι (F.flip.obj k) j ≫ colimit.ι (F.flip ⋙ colim) k ≫
      (colimitFlipCompColimIsoColimitCompColim F).hom =
        (colimit.ι _ k ≫ colimit.ι (F ⋙ colim) j : _ ⟶ colimit (F⋙ colim)) := by
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Prod K J) C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.f …
  -/
  dsimp [colimitFlipCompColimIsoColimitCompColim]
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Prod K J) C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.f …
  -/
  slice_lhs 1 3 => simp only []
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Prod K J) C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [Equivalence.unit]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem colimitFlipCompColimIsoColimitCompColim_ι_ι_inv (k) (j) :
    colimit.ι (F.obj j) k ≫ colimit.ι (F ⋙ colim) j ≫
      (colimitFlipCompColimIsoColimitCompColim F).inv =
        (colimit.ι _ j ≫ colimit.ι (F.flip ⋙ colim) k : _ ⟶ colimit (F.flip ⋙ colim)) := by
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Prod K J) C
    k : K
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.o …
  -/
  dsimp [colimitFlipCompColimIsoColimitCompColim]
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Prod K J) C
    k : K
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.o …
  -/
  slice_lhs 1 3 => simp only []
  /-
    J K : Type v
    inst✝⁶ : CategoryTheory.SmallCategory J
    inst✝⁵ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝³ : CategoryTheory.Limits.HasColimitsOfShape J C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimitsOfShape (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape (Prod K J) C
    k : K
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [Equivalence.counitInv]
  /-
    🎉 no goals
  -/


/-- The Fubini theorem for a functor `G : J × K ⥤ C`,
showing that the limit of `G` can be computed as
the limit of the limits of the functors `G.obj (j, _)`.
-/
noncomputable def limitIsoLimitCurryCompLim : limit G ≅ limit (curry.obj G ⋙ lim) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit G
    inst✝ : CategoryTheory.Limits.HasLimit ((CategoryTheory.curry.obj G).comp Cate …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit G) (CategoryTheory.Limits.li …
  -/
  have i : G ≅ uncurry.obj ((@curry J _ K _ C _).obj G) := currying.symm.unitIso.app G
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit G
    inst✝ : CategoryTheory.Limits.HasLimit ((CategoryTheory.curry.obj G).comp Cate …
    i : CategoryTheory.Iso G (CategoryTheory.uncurry.obj (CategoryTheory.curry.obj …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit G) (CategoryTheory.Limits.li …
  -/
  haveI : Limits.HasLimit (uncurry.obj ((@curry J _ K _ C _).obj G)) := hasLimitOfIso i
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit G
    inst✝ : CategoryTheory.Limits.HasLimit ((CategoryTheory.curry.obj G).comp Cate …
    i : CategoryTheory.Iso G (CategoryTheory.uncurry.obj (CategoryTheory.curry.obj …
    this : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj (CategoryThe …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit G) (CategoryTheory.Limits.li …
  -/
  trans limit (uncurry.obj ((@curry J _ K _ C _).obj G))
    /-
      J K : Type v
      inst✝⁵ : CategoryTheory.SmallCategory J
      inst✝⁴ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor (Prod J K) C
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
      inst✝¹ : CategoryTheory.Limits.HasLimit G
      inst✝ : CategoryTheory.Limits.HasLimit ((CategoryTheory.curry.obj G).comp Cate …
      i : CategoryTheory.Iso G (CategoryTheory.uncurry.obj (CategoryTheory.curry.obj …
      this : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj (CategoryThe …
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit G) (CategoryTheory.Limits.li …
    -/
  · apply HasLimit.isoOfNatIso i
    /-
      🎉 no goals
    -/
    /-
      J K : Type v
      inst✝⁵ : CategoryTheory.SmallCategory J
      inst✝⁴ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor (Prod J K) C
      inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
      inst✝¹ : CategoryTheory.Limits.HasLimit G
      inst✝ : CategoryTheory.Limits.HasLimit ((CategoryTheory.curry.obj G).comp Cate …
      i : CategoryTheory.Iso G (CategoryTheory.uncurry.obj (CategoryTheory.curry.obj …
      this : CategoryTheory.Limits.HasLimit (CategoryTheory.uncurry.obj (CategoryThe …
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.limit (CategoryTheory.uncurry.obj  …
    -/
  · exact limitUncurryIsoLimitCompLim ((@curry J _ K _ C _).obj G)
    /-
      🎉 no goals
    -/


@[simp, reassoc]
theorem limitIsoLimitCurryCompLim_hom_π_π {j} {k} :
    (limitIsoLimitCurryCompLim G).hom ≫ limit.π _ j ≫ limit.π _ k = limit.π _ (j, k) := by
  set_option tactic.skipAssignedInstances false in
  simp [limitIsoLimitCurryCompLim, Trans.simple, HasLimit.isoOfNatIso, limitUncurryIsoLimitCompLim]

-- Porting note: Added type annotation `limit (_ ⋙ lim) ⟶ _`

@[simp, reassoc]
theorem limitIsoLimitCurryCompLim_inv_π {j} {k} :
    (limitIsoLimitCurryCompLim G).inv ≫ limit.π _ (j, k) =
      (limit.π _ j ≫ limit.π _ k : limit (_ ⋙ lim) ⟶ _) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit G
    inst✝ : CategoryTheory.Limits.HasLimit ((CategoryTheory.curry.obj G).comp Cate …
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitIsoLimitC …
  -/
  rw [← cancel_epi (limitIsoLimitCurryCompLim G).hom]
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasLimit G
    inst✝ : CategoryTheory.Limits.HasLimit ((CategoryTheory.curry.obj G).comp Cate …
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitIsoLimitC …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The Fubini theorem for a functor `G : J × K ⥤ C`,
showing that the colimit of `G` can be computed as
the colimit of the colimits of the functors `G.obj (j, _)`.
-/
noncomputable def colimitIsoColimitCurryCompColim : colimit G ≅ colimit (curry.obj G ⋙ colim) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit G
    inst✝ : CategoryTheory.Limits.HasColimit ((CategoryTheory.curry.obj G).comp Ca …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit G) (CategoryTheory.Limits. …
  -/
  have i : G ≅ uncurry.obj ((@curry J _ K _ C _).obj G) := currying.symm.unitIso.app G
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit G
    inst✝ : CategoryTheory.Limits.HasColimit ((CategoryTheory.curry.obj G).comp Ca …
    i : CategoryTheory.Iso G (CategoryTheory.uncurry.obj (CategoryTheory.curry.obj …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit G) (CategoryTheory.Limits. …
  -/
  haveI : Limits.HasColimit (uncurry.obj ((@curry J _ K _ C _).obj G)) := hasColimitOfIso i.symm
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit G
    inst✝ : CategoryTheory.Limits.HasColimit ((CategoryTheory.curry.obj G).comp Ca …
    i : CategoryTheory.Iso G (CategoryTheory.uncurry.obj (CategoryTheory.curry.obj …
    this : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj (CategoryT …
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit G) (CategoryTheory.Limits. …
  -/
  trans colimit (uncurry.obj ((@curry J _ K _ C _).obj G))
    /-
      J K : Type v
      inst✝⁵ : CategoryTheory.SmallCategory J
      inst✝⁴ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor (Prod J K) C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
      inst✝¹ : CategoryTheory.Limits.HasColimit G
      inst✝ : CategoryTheory.Limits.HasColimit ((CategoryTheory.curry.obj G).comp Ca …
      i : CategoryTheory.Iso G (CategoryTheory.uncurry.obj (CategoryTheory.curry.obj …
      this : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj (CategoryT …
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit G) (CategoryTheory.Limits. …
    -/
  · apply HasColimit.isoOfNatIso i
    /-
      🎉 no goals
    -/
    /-
      J K : Type v
      inst✝⁵ : CategoryTheory.SmallCategory J
      inst✝⁴ : CategoryTheory.SmallCategory K
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor (Prod J K) C
      inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
      inst✝¹ : CategoryTheory.Limits.HasColimit G
      inst✝ : CategoryTheory.Limits.HasColimit ((CategoryTheory.curry.obj G).comp Ca …
      i : CategoryTheory.Iso G (CategoryTheory.uncurry.obj (CategoryTheory.curry.obj …
      this : CategoryTheory.Limits.HasColimit (CategoryTheory.uncurry.obj (CategoryT …
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.colimit (CategoryTheory.uncurry.ob …
    -/
  · exact colimitUncurryIsoColimitCompColim ((@curry J _ K _ C _).obj G)
    /-
      🎉 no goals
    -/


@[simp, reassoc]
theorem colimitIsoColimitCurryCompColim_ι_ι_inv {j} {k} :
    colimit.ι ((curry.obj G).obj j) k ≫ colimit.ι (curry.obj G ⋙ colim) j ≫
      (colimitIsoColimitCurryCompColim G).inv  = colimit.ι _ (j, k) := by
  set_option tactic.skipAssignedInstances false in
  simp [colimitIsoColimitCurryCompColim, Trans.simple, HasColimit.isoOfNatIso,
    colimitUncurryIsoColimitCompColim]


@[simp, reassoc]
theorem colimitIsoColimitCurryCompColim_ι_hom {j} {k} :
    colimit.ι _ (j, k) ≫ (colimitIsoColimitCurryCompColim G).hom =
      (colimit.ι (_) k ≫ colimit.ι (curry.obj G ⋙ colim) j : _ ⟶ colimit (_ ⋙ colim)) := by
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit G
    inst✝ : CategoryTheory.Limits.HasColimit ((CategoryTheory.curry.obj G).comp Ca …
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι G {  …
  -/
  rw [← cancel_mono (colimitIsoColimitCurryCompColim G).inv]
  /-
    J K : Type v
    inst✝⁵ : CategoryTheory.SmallCategory J
    inst✝⁴ : CategoryTheory.SmallCategory K
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝² : CategoryTheory.Limits.HasColimitsOfShape K C
    inst✝¹ : CategoryTheory.Limits.HasColimit G
    inst✝ : CategoryTheory.Limits.HasColimit ((CategoryTheory.curry.obj G).comp Ca …
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A variant of the Fubini theorem for a functor `G : J × K ⥤ C`,
showing that $\lim_k \lim_j G(j,k) ≅ \lim_j \lim_k G(j,k)$.
-/
noncomputable def limitCurrySwapCompLimIsoLimitCurryCompLim :
    limit (curry.obj (Prod.swap K J ⋙ G) ⋙ lim) ≅ limit (curry.obj G ⋙ lim) :=
  calc
    limit (curry.obj (Prod.swap K J ⋙ G) ⋙ lim) ≅ limit (Prod.swap K J ⋙ G) :=
      (limitIsoLimitCurryCompLim _).symm
    _ ≅ limit G := HasLimit.isoOfEquivalence (Prod.braiding K J) (Iso.refl _)
    _ ≅ limit (curry.obj G ⋙ lim) := limitIsoLimitCurryCompLim _

-- Porting note: Added type annotation `limit (_ ⋙ lim) ⟶ _`

@[simp]
theorem limitCurrySwapCompLimIsoLimitCurryCompLim_hom_π_π {j} {k} :
    (limitCurrySwapCompLimIsoLimitCurryCompLim G).hom ≫ limit.π _ j ≫ limit.π _ k =
      (limit.π _ k ≫ limit.π _ j : limit (_ ⋙ lim) ⟶ _) := by
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitCurrySwap …
  -/
  dsimp [limitCurrySwapCompLimIsoLimitCurryCompLim, Equivalence.counit]
  rw [Category.assoc, Category.assoc, limitIsoLimitCurryCompLim_hom_π_π,
    HasLimit.isoOfEquivalence_hom_π]
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitIsoLimitC …
  -/
  dsimp [Equivalence.counit]
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitIsoLimitC …
  -/
  rw [← prod_id, G.map_id]
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitIsoLimitC …
  -/
  simp
  /-
    🎉 no goals
  -/

-- Porting note: Added type annotation `limit (_ ⋙ lim) ⟶ _`

@[simp]
theorem limitCurrySwapCompLimIsoLimitCurryCompLim_inv_π_π {j} {k} :
    (limitCurrySwapCompLimIsoLimitCurryCompLim G).inv ≫ limit.π _ k ≫ limit.π _ j =
      (limit.π _ j ≫ limit.π _ k : limit (_ ⋙ lim) ⟶ _) := by
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasLimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitCurrySwap …
  -/
  simp [limitCurrySwapCompLimIsoLimitCurryCompLim]
  /-
    🎉 no goals
  -/


/-- A variant of the Fubini theorem for a functor `G : J × K ⥤ C`,
showing that $\colim_k \colim_j G(j,k) ≅ \colim_j \colim_k G(j,k)$.
-/
noncomputable def colimitCurrySwapCompColimIsoColimitCurryCompColim :
    colimit (curry.obj (Prod.swap K J ⋙ G) ⋙ colim) ≅ colimit (curry.obj G ⋙ colim) :=
  calc
    colimit (curry.obj (Prod.swap K J ⋙ G) ⋙ colim) ≅ colimit (Prod.swap K J ⋙ G) :=
      (colimitIsoColimitCurryCompColim _).symm
    _ ≅ colimit G := HasColimit.isoOfEquivalence (Prod.braiding K J) (Iso.refl _)
    _ ≅ colimit (curry.obj G ⋙ colim) := colimitIsoColimitCurryCompColim _


@[simp]
theorem colimitCurrySwapCompColimIsoColimitCurryCompColim_ι_ι_hom {j} {k} :
    colimit.ι _ j ≫ colimit.ι (curry.obj (Prod.swap K J ⋙ G) ⋙ colim) k ≫
      (colimitCurrySwapCompColimIsoColimitCurryCompColim G).hom =
        (colimit.ι _ k ≫ colimit.ι (curry.obj G ⋙ colim) j : _ ⟶ colimit (curry.obj G⋙ colim)) := by
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  dsimp [colimitCurrySwapCompColimIsoColimitCurryCompColim]
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  slice_lhs 1 3 => simp only []
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem colimitCurrySwapCompColimIsoColimitCurryCompColim_ι_ι_inv {j} {k} :
    colimit.ι _ k ≫ colimit.ι (curry.obj G ⋙ colim) j ≫
      (colimitCurrySwapCompColimIsoColimitCurryCompColim G).inv =
        (colimit.ι _ j ≫
          colimit.ι (curry.obj _ ⋙ colim) k :
            _ ⟶ colimit (curry.obj (Prod.swap K J ⋙ G) ⋙ colim)) := by
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  dsimp [colimitCurrySwapCompColimIsoColimitCurryCompColim]
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  slice_lhs 1 3 => simp only []
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [colimitIsoColimitCurryCompColim_ι_ι_inv, HasColimit.isoOfEquivalence_inv_π]
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp [Equivalence.counitInv]
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [CategoryTheory.Bifunctor.map_id]
  /-
    J K : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.SmallCategory K
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    G : CategoryTheory.Functor (Prod J K) C
    inst✝ : CategoryTheory.Limits.HasColimits C
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


