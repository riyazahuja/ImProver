@[local instance]
lemma hasColimit_ι_comp : ∀ X, HasColimit (Grothendieck.ι F X ⋙ G) :=
  fun X => hasColimitOfIso (F := F.map (𝟙 _) ⋙ Grothendieck.ι F X ⋙ G) <|
    (Functor.leftUnitor (Grothendieck.ι F X ⋙ G)).symm ≪≫
    (isoWhiskerRight (eqToIso (F.map_id X).symm) (Grothendieck.ι F X ⋙ G))


/-- A functor taking a colimit on each fiber of a functor `G : Grothendieck F ⥤ H`. -/
@[simps]
def fiberwiseColimit : C ⥤ H where
  obj X := colimit (Grothendieck.ι F X ⋙ G)
  map {X Y} f := colimMap (whiskerRight (Grothendieck.ιNatTrans f) G ≫
    (Functor.associator _ _ _).hom) ≫ colimit.pre (Grothendieck.ι F Y ⋙ G) (F.map f)
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      X : C
      ⊢ Eq ({ obj := fun X => CategoryTheory.Limits.colimit ((CategoryTheory.Grothen …
    -/
    ext d
    simp only [Functor.comp_obj, Grothendieck.ιNatTrans, Grothendieck.ι_obj, ι_colimMap_assoc,
      NatTrans.comp_app, whiskerRight_app, Functor.associator_hom_app, Category.comp_id,
      colimit.ι_pre]
    conv_rhs => rw [← colimit.eqToHom_comp_ι (Grothendieck.ι F X ⋙ G)
      (j := (F.map (𝟙 X)).obj d) (by simp)]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      X : C
      d : ↑(F.obj X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { base := CategoryTheory.Categ …
    -/
    rw [← eqToHom_map G (by simp), Grothendieck.eqToHom_eq]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      X : C
      d : ↑(F.obj X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { base := CategoryTheory.Categ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp {X Y Z} f g := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      ⊢ Eq ({ obj := fun X => CategoryTheory.Limits.colimit ((CategoryTheory.Grothen …
    -/
    ext d
    simp only [Functor.comp_obj, Grothendieck.ιNatTrans, ι_colimMap_assoc, NatTrans.comp_app,
      whiskerRight_app, Functor.associator_hom_app, Category.comp_id, colimit.ι_pre, Category.assoc,
      colimit.ι_pre_assoc]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      d : ↑(F.obj X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { base := CategoryTheory.Categ …
    -/
    rw [← Category.assoc, ← G.map_comp]
    conv_rhs => rw [← colimit.eqToHom_comp_ι (Grothendieck.ι F Z ⋙ G)
      (j := (F.map (f ≫ g)).obj d) (by simp)]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      d : ↑(F.obj X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { base := CategoryTheory.Categ …
    -/
    rw [← Category.assoc, ← eqToHom_map G (by simp), ← G.map_comp, Grothendieck.eqToHom_eq]
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      d : ↑(F.obj X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { base := CategoryTheory.Categ …
    -/
    congr 2
    /-
      case w.e_a.e_a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      d : ↑(F.obj X)
      ⊢ Eq { base := CategoryTheory.CategoryStruct.comp f g, fiber := CategoryTheory …
    -/
    fapply Grothendieck.ext
    · simp only [Cat.comp_obj, eqToHom_refl, Category.assoc, Grothendieck.comp_base,
        Category.comp_id]
    · simp only [Grothendieck.ι_obj, Cat.comp_obj, eqToHom_refl, Cat.id_obj,
        Grothendieck.comp_base, Category.comp_id, Grothendieck.comp_fiber, Functor.map_id]
      /-
        case w.e_a.e_a.w_fiber
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor C CategoryTheory.Cat
        H : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
        G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
        inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
        X Y Z : C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        d : ↑(F.obj X)
        ⊢ Eq (CategoryTheory.eqToHom ⋯) (CategoryTheory.CategoryStruct.comp (CategoryT …
      -/
      conv_rhs => enter [2, 1]; rw [eqToHom_map (F.map (𝟙 Z))]
      /-
        case w.e_a.e_a.w_fiber
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        F : CategoryTheory.Functor C CategoryTheory.Cat
        H : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
        G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
        inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
        X Y Z : C
        f : Quiver.Hom X Y
        g : Quiver.Hom Y Z
        d : ↑(F.obj X)
        ⊢ Eq (CategoryTheory.eqToHom ⋯) (CategoryTheory.CategoryStruct.comp (CategoryT …
      -/
      conv_rhs => rw [eqToHom_trans, eqToHom_trans]
      /-
        🎉 no goals
      -/


/-- Every functor `G : Grothendieck F ⥤ H` induces a natural transformation from `G` to the
composition of the forgetful functor on `Grothendieck F` with the fiberwise colimit on `G`. -/
@[simps]
def natTransIntoForgetCompFiberwiseColimit :
    G ⟶ Grothendieck.forget F ⋙ fiberwiseColimit G where
  app X := colimit.ι (Grothendieck.ι F X.base ⋙ G) X.fiber
  naturality _ _ f := by
    simp only [Functor.comp_obj, Grothendieck.forget_obj, fiberwiseColimit_obj, Functor.comp_map,
      Grothendieck.forget_map, fiberwiseColimit_map, ι_colimMap_assoc, Grothendieck.ι_obj,
      NatTrans.comp_app, whiskerRight_app, Functor.associator_hom_app, Category.comp_id,
      colimit.ι_pre]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      x✝¹ x✝ : CategoryTheory.Grothendieck F
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.Limits.coli …
    -/
    rw [← colimit.w (Grothendieck.ι F _ ⋙ G) f.fiber]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      x✝¹ x✝ : CategoryTheory.Grothendieck F
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.Limits.coli …
    -/
    simp only [← Category.assoc, Functor.comp_obj, Functor.comp_map, ← G.map_comp]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      x✝¹ x✝ : CategoryTheory.Grothendieck F
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (CategoryTheory.Limits.coli …
    -/
    congr 2
    /-
      case e_a.e_a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      x✝¹ x✝ : CategoryTheory.Grothendieck F
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq f (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Grothendieck.ιNatT …
    -/
                               /-
                                 🎉 no goals
                               -/
    apply Grothendieck.ext <;> simp
                               /-
                                 🎉 no goals
                               -/


variable {G} in
/-- A cocone on a functor `G : Grothendieck F ⥤ H` induces a cocone on the fiberwise colimit
on `G`. -/
@[simps]
def coconeFiberwiseColimitOfCocone (c : Cocone G) : Cocone (fiberwiseColimit G) where
  pt := c.pt
  ι := { app := fun X => colimit.desc _ (c.whisker (Grothendieck.ι F X)),
                                       /-
                                         C : Type u₁
                                         inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                         F : CategoryTheory.Functor C CategoryTheory.Cat
                                         H : Type u₂
                                         inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
                                         G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
                                         inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
                                         c : CategoryTheory.Limits.Cocone G
                                         x✝¹ x✝ : C
                                         f : Quiver.Hom x✝¹ x✝
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.fiberwiseColi …
                                       -/
         naturality := fun _ _ f => by dsimp; ext; simp }
                                                   /-
                                                     🎉 no goals
                                                   -/


variable {G} in
/-- If `c` is a colimit cocone on `G : Grockendieck F ⥤ H`, then the induced cocone on the
fiberwise colimit on `G` is a colimit cocone, too. -/
def isColimitCoconeFiberwiseColimitOfCocone {c : Cocone G} (hc : IsColimit c) :
    IsColimit (coconeFiberwiseColimitOfCocone c) where
  desc s := hc.desc <| Cocone.mk s.pt <| natTransIntoForgetCompFiberwiseColimit G ≫
    whiskerLeft (Grothendieck.forget F) s.ι
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  F : CategoryTheory.Functor C CategoryTheory.Cat
                  H : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
                  G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
                  inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
                  c✝ : CategoryTheory.Limits.Cocone G
                  hc : CategoryTheory.Limits.IsColimit c✝
                  s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
                  c : C
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.coconeFiberwi …
                -/
  fac s c := by dsimp; ext; simp
                            /-
                              🎉 no goals
                            -/
  uniq s m hm := hc.hom_ext fun X => by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
      m : Quiver.Hom (CategoryTheory.Limits.coconeFiberwiseColimitOfCocone c).pt s.pt
      hm : ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits …
      X : CategoryTheory.Grothendieck F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app X) m) (CategoryTheory.Catego …
    -/
    have := hm X.base
    simp only [Functor.const_obj_obj, IsColimit.fac, NatTrans.comp_app, Functor.comp_obj,
      Grothendieck.forget_obj, fiberwiseColimit_obj, natTransIntoForgetCompFiberwiseColimit_app,
      whiskerLeft_app]
    simp only [fiberwiseColimit_obj, coconeFiberwiseColimitOfCocone_pt, Functor.const_obj_obj,
      coconeFiberwiseColimitOfCocone_ι_app] at this
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
      m : Quiver.Hom (CategoryTheory.Limits.coconeFiberwiseColimitOfCocone c).pt s.pt
      hm : ∀ (j : C), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits …
      X : CategoryTheory.Grothendieck F
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.d …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app X) m) (CategoryTheory.Catego …
    -/
    simp [← this]
    /-
      🎉 no goals
    -/


lemma hasColimit_fiberwiseColimit [HasColimit G] : HasColimit (fiberwiseColimit G) where
  exists_colimit := ⟨⟨_, isColimitCoconeFiberwiseColimitOfCocone (colimit.isColimit _)⟩⟩


/-- For a functor `G : Grothendieck F ⥤ H`, every cocone over `fiberwiseColimit G` induces a
cocone over `G` itself. -/
@[simps]
def coconeOfCoconeFiberwiseColimit (c : Cocone (fiberwiseColimit G)) : Cocone G where
  pt := c.pt
  ι := { app := fun X => colimit.ι (Grothendieck.ι F X.base ⋙ G) X.fiber ≫ c.ι.app X.base
         naturality := fun {X Y} ⟨f, g⟩ => by
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            F : CategoryTheory.Functor C CategoryTheory.Cat
            H : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
            G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
            inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
            c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
            X Y : CategoryTheory.Grothendieck F
            x✝ : Quiver.Hom X Y
            f : Quiver.Hom X.base Y.base
            g : Quiver.Hom ((F.map f).obj X.fiber) Y.fiber
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { base := f, fiber := g }) ((f …
          -/
          simp only [Functor.const_obj_obj, Functor.const_obj_map, Category.comp_id]
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            F : CategoryTheory.Functor C CategoryTheory.Cat
            H : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
            G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
            inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
            c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
            X Y : CategoryTheory.Grothendieck F
            x✝ : Quiver.Hom X Y
            f : Quiver.Hom X.base Y.base
            g : Quiver.Hom ((F.map f).obj X.fiber) Y.fiber
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map { base := f, fiber := g }) (Ca …
          -/
          rw [← Category.assoc, ← c.w f, ← Category.assoc]
          simp only [fiberwiseColimit_obj, fiberwiseColimit_map, ι_colimMap_assoc, Functor.comp_obj,
            Grothendieck.ι_obj, NatTrans.comp_app, whiskerRight_app, Functor.associator_hom_app,
            Category.comp_id, colimit.ι_pre]
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            F : CategoryTheory.Functor C CategoryTheory.Cat
            H : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
            G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
            inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
            c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
            X Y : CategoryTheory.Grothendieck F
            x✝ : Quiver.Hom X Y
            f : Quiver.Hom X.base Y.base
            g : Quiver.Hom ((F.map f).obj X.fiber) Y.fiber
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          rw [← colimit.w _ g, ← Category.assoc, Functor.comp_map, ← G.map_comp]
          /-
            C : Type u₁
            inst✝² : CategoryTheory.Category.{v₁, u₁} C
            F : CategoryTheory.Functor C CategoryTheory.Cat
            H : Type u₂
            inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
            G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
            inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
            c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
            X Y : CategoryTheory.Grothendieck F
            x✝ : Quiver.Hom X Y
            f : Quiver.Hom X.base Y.base
            g : Quiver.Hom ((F.map f).obj X.fiber) Y.fiber
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
                    /-
                      🎉 no goals
                    -/
          congr <;> simp }
                    /-
                      🎉 no goals
                    -/


/-- If a cocone `c` over a functor `G : Grothendieck F ⥤ H` is a colimit, than the induced cocone
`coconeOfFiberwiseCocone G c` -/
def isColimitCoconeOfFiberwiseCocone {c : Cocone (fiberwiseColimit G)} (hc : IsColimit c) :
    IsColimit (coconeOfCoconeFiberwiseColimit c) where
  desc s := hc.desc <| Cocone.mk s.pt <|
    { app := fun X => colimit.desc (Grothendieck.ι F X ⋙ G) (s.whisker _) }
  uniq s m hm := hc.hom_ext <| fun X => by
    simp only [fiberwiseColimit_obj, Functor.const_obj_obj, fiberwiseColimit_map,
      Functor.const_obj_map, Cocone.whisker_pt, id_eq, Functor.comp_obj, Cocone.whisker_ι,
      whiskerLeft_app, NatTrans.comp_app, whiskerRight_app, Functor.associator_hom_app,
      whiskerLeft_twice, eq_mpr_eq_cast, IsColimit.fac]
    simp only [coconeOfCoconeFiberwiseColimit_pt, Functor.const_obj_obj,
      coconeOfCoconeFiberwiseColimit_ι_app, Category.assoc] at hm
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone G
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfCoconeFiberwiseColimit c).pt s.pt
      X : C
      hm : ∀ (j : CategoryTheory.Grothendieck F), Eq (CategoryTheory.CategoryStruct. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app X) m) (CategoryTheory.Limits …
    -/
    ext d
    /-
      case w
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      F : CategoryTheory.Functor C CategoryTheory.Cat
      H : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} H
      G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (Ca …
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.fiberwiseColimit G)
      hc : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.Cocone G
      m : Quiver.Hom (CategoryTheory.Limits.coconeOfCoconeFiberwiseColimit c).pt s.pt
      X : C
      hm : ∀ (j : CategoryTheory.Grothendieck F), Eq (CategoryTheory.CategoryStruct. …
      d : ↑(F.obj X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
    -/
    simp [hm ⟨X, d⟩]
    /-
      🎉 no goals
    -/


/-- We can infer that a functor `G : Grothendieck F ⥤ H`, with `F : C ⥤ Cat`, has a colimit from
the fact that each of its fibers has a colimit and that these fiberwise colimits, as a functor
`C ⥤ H` have a colimit. -/
@[local instance]
lemma hasColimit_of_hasColimit_fiberwiseColimit_of_hasColimit : HasColimit G where
  exists_colimit := ⟨⟨_, isColimitCoconeOfFiberwiseCocone (colimit.isColimit _)⟩⟩


/-- For every functor `G` on the Grothendieck construction `Grothendieck F`, if `G` has a colimit
and every fiber of `G` has a colimit, then taking this colimit is isomorphic to first taking the
fiberwise colimit and then the colimit of the resulting functor. -/
def colimitFiberwiseColimitIso : colimit (fiberwiseColimit G) ≅ colimit G :=
  IsColimit.coconePointUniqueUpToIso (colimit.isColimit (fiberwiseColimit G))
    (isColimitCoconeFiberwiseColimitOfCocone (colimit.isColimit _))


@[reassoc (attr := simp)]
lemma ι_colimitFiberwiseColimitIso_hom (X : C) (d : F.obj X) :
    colimit.ι (Grothendieck.ι F X ⋙ G) d ≫ colimit.ι (fiberwiseColimit G) X ≫
      (colimitFiberwiseColimitIso G).hom = colimit.ι G ⟨X, d⟩ := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    H : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} H
    G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
    inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (C …
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.fiberwiseColim …
    X : C
    d : ↑(F.obj X)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  simp [colimitFiberwiseColimitIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_colimitFiberwiseColimitIso_inv (X : Grothendieck F) :
    colimit.ι G X ≫ (colimitFiberwiseColimitIso G).inv =
    colimit.ι (Grothendieck.ι F X.base ⋙ G) X.fiber ≫ colimit.ι (fiberwiseColimit G) X.base := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    H : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} H
    G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
    inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (C …
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.fiberwiseColim …
    X : CategoryTheory.Grothendieck F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι G X) …
  -/
  rw [Iso.comp_inv_eq]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    F : CategoryTheory.Functor C CategoryTheory.Cat
    H : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} H
    G : CategoryTheory.Functor (CategoryTheory.Grothendieck F) H
    inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.HasColimit (C …
    inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.fiberwiseColim …
    X : CategoryTheory.Grothendieck F
    ⊢ Eq (CategoryTheory.Limits.colimit.ι G X) (CategoryTheory.CategoryStruct.comp …
  -/
  simp
  /-
    🎉 no goals
  -/


@[instance]
theorem hasColimitsOfShape_grothendieck [∀ X, HasColimitsOfShape (F.obj X) H]
    [HasColimitsOfShape C H] : HasColimitsOfShape (Grothendieck F) H where
  has_colimit _ := hasColimit_of_hasColimit_fiberwiseColimit_of_hasColimit _


