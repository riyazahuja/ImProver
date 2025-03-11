@[reassoc (attr := simp)]
theorem limit.lift_π_app (H : J ⥤ K ⥤ C) [HasLimit H] (c : Cone H) (j : J) (k : K) :
    (limit.lift H c).app k ≫ (limit.π H j).app k = (c.π.app j).app k :=
  congr_app (limit.lift_π c j) k


@[reassoc (attr := simp)]
theorem colimit.ι_desc_app (H : J ⥤ K ⥤ C) [HasColimit H] (c : Cocone H) (j : J) (k : K) :
    (colimit.ι H j).app k ≫ (colimit.desc H c).app k = (c.ι.app j).app k :=
  congr_app (colimit.ι_desc c j) k


/-- The evaluation functors jointly reflect limits: that is, to show a cone is a limit of `F`
it suffices to show that each evaluation cone is a limit. In other words, to prove a cone is
limiting you can show it's pointwise limiting.
-/
def evaluationJointlyReflectsLimits {F : J ⥤ K ⥤ C} (c : Cone F)
    (t : ∀ k : K, IsLimit (((evaluation K C).obj k).mapCone c)) : IsLimit c where
  lift s :=
    { app := fun k => (t k).lift ⟨s.pt.obj k, whiskerRight s.π ((evaluation K C).obj k)⟩
      naturality := fun X Y f =>
        (t Y).hom_ext fun j => by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : CategoryTheory.Limits.Cone F
            t : (k : K) → CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation K C). …
            s : CategoryTheory.Limits.Cone F
            X Y : K
            f : Quiver.Hom X Y
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          rw [assoc, (t Y).fac _ j]
          simpa using
            ((t X).fac_assoc ⟨s.pt.obj X, whiskerRight s.π ((evaluation K C).obj X)⟩ j _).symm }
                /-
                  C : Type u
                  inst✝³ : CategoryTheory.Category.{v, u} C
                  D : Type u'
                  inst✝² : CategoryTheory.Category.{v', u'} D
                  J : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                  K : Type u₂
                  inst✝ : CategoryTheory.Category.{v₂, u₂} K
                  F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                  c : CategoryTheory.Limits.Cone F
                  t : (k : K) → CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation K C). …
                  s : CategoryTheory.Limits.Cone F
                  j : J
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => { app := fun k => (t k).li …
                -/
  fac s j := by ext k; exact (t k).fac _ j
                       /-
                         🎉 no goals
                       -/
  uniq s m w := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      c : CategoryTheory.Limits.Cone F
      t : (k : K) → CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation K C). …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt c.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (c.π.app j)) (s.π.app j)
      ⊢ Eq m ((fun s => { app := fun k => (t k).lift { pt := s.pt.obj k, π := Catego …
    -/
    ext x
    exact (t x).hom_ext fun j =>
      (congr_app (w j) x).trans
        ((t x).fac ⟨s.pt.obj _, whiskerRight s.π ((evaluation K C).obj _)⟩ j).symm


/-- Given a functor `F` and a collection of limit cones for each diagram `X ↦ F X k`, we can stitch
them together to give a cone for the diagram `F`.
`combinedIsLimit` shows that the new cone is limiting, and `evalCombined` shows it is
(essentially) made up of the original cones.
-/
@[simps]
def combineCones (F : J ⥤ K ⥤ C) (c : ∀ k : K, LimitCone (F.flip.obj k)) : Cone F where
  pt :=
    { obj := fun k => (c k).cone.pt
      map := fun {k₁} {k₂} f => (c k₂).isLimit.lift ⟨_, (c k₁).cone.π ≫ F.flip.map f⟩
      map_id := fun k =>
        (c k).isLimit.hom_ext fun j => by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : (k : K) → CategoryTheory.Limits.LimitCone (F.flip.obj k)
            k : K
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun k => (c k).cone.pt, map …
          -/
          dsimp
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : (k : K) → CategoryTheory.Limits.LimitCone (F.flip.obj k)
            k : K
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c k).isLimit.lift { pt := (c k).con …
          -/
          simp
          /-
            🎉 no goals
          -/
                                                                                 /-
                                                                                   C : Type u
                                                                                   inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                   D : Type u'
                                                                                   inst✝² : CategoryTheory.Category.{v', u'} D
                                                                                   J : Type u₁
                                                                                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                                   K : Type u₂
                                                                                   inst✝ : CategoryTheory.Category.{v₂, u₂} K
                                                                                   F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                                                                                   c : (k : K) → CategoryTheory.Limits.LimitCone (F.flip.obj k)
                                                                                   k₁ k₂ k₃ : K
                                                                                   f₁ : Quiver.Hom k₁ k₂
                                                                                   f₂ : Quiver.Hom k₂ k₃
                                                                                   j : J
                                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ({ obj := fun k => (c k).cone.pt, map …
                                                                                 -/
      map_comp := fun {k₁} {k₂} {k₃} f₁ f₂ => (c k₃).isLimit.hom_ext fun j => by simp }
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  π :=
    { app := fun j => { app := fun k => (c k).cone.π.app j }
                                      /-
                                        C : Type u
                                        inst✝³ : CategoryTheory.Category.{v, u} C
                                        D : Type u'
                                        inst✝² : CategoryTheory.Category.{v', u'} D
                                        J : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                        K : Type u₂
                                        inst✝ : CategoryTheory.Category.{v₂, u₂} K
                                        F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                                        c : (k : K) → CategoryTheory.Limits.LimitCone (F.flip.obj k)
                                        j₁ j₂ : J
                                        g : Quiver.Hom j₁ j₂
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                                      -/
      naturality := fun j₁ j₂ g => by ext k; exact (c k).cone.π.naturality g }
                                             /-
                                               🎉 no goals
                                             -/


/-- The stitched together cones each project down to the original given cones (up to iso). -/
def evaluateCombinedCones (F : J ⥤ K ⥤ C) (c : ∀ k : K, LimitCone (F.flip.obj k)) (k : K) :
    ((evaluation K C).obj k).mapCone (combineCones F c) ≅ (c k).cone :=
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    c : (k : K) → CategoryTheory.Limits.LimitCone (F.flip.obj k)
    k : K
    ⊢ ∀ (j : J), Eq ((((CategoryTheory.evaluation K C).obj k).mapCone (CategoryThe …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Stitching together limiting cones gives a limiting cone. -/
def combinedIsLimit (F : J ⥤ K ⥤ C) (c : ∀ k : K, LimitCone (F.flip.obj k)) :
    IsLimit (combineCones F c) :=
  evaluationJointlyReflectsLimits _ fun k =>
    (c k).isLimit.ofIsoLimit (evaluateCombinedCones F c k).symm


/-- The evaluation functors jointly reflect colimits: that is, to show a cocone is a colimit of `F`
it suffices to show that each evaluation cocone is a colimit. In other words, to prove a cocone is
colimiting you can show it's pointwise colimiting.
-/
def evaluationJointlyReflectsColimits {F : J ⥤ K ⥤ C} (c : Cocone F)
    (t : ∀ k : K, IsColimit (((evaluation K C).obj k).mapCocone c)) : IsColimit c where
  desc s :=
    { app := fun k => (t k).desc ⟨s.pt.obj k, whiskerRight s.ι ((evaluation K C).obj k)⟩
      naturality := fun X Y f =>
        (t X).hom_ext fun j => by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : CategoryTheory.Limits.Cocone F
            t : (k : K) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C …
            s : CategoryTheory.Limits.Cocone F
            X Y : K
            f : Quiver.Hom X Y
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.evaluation K C).ob …
          -/
          rw [(t X).fac_assoc _ j]
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : CategoryTheory.Limits.Cocone F
            t : (k : K) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C …
            s : CategoryTheory.Limits.Cocone F
            X Y : K
            f : Quiver.Hom X Y
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.evaluation K C).ob …
          -/
          erw [← (c.ι.app j).naturality_assoc f]
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : CategoryTheory.Limits.Cocone F
            t : (k : K) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C …
            s : CategoryTheory.Limits.Cocone F
            X Y : K
            f : Quiver.Hom X Y
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) (CategoryTheory.Cat …
          -/
          erw [(t Y).fac ⟨s.pt.obj _, whiskerRight s.ι _⟩ j]
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : CategoryTheory.Limits.Cocone F
            t : (k : K) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C …
            s : CategoryTheory.Limits.Cocone F
            X Y : K
            f : Quiver.Hom X Y
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) ({ pt := s.pt.obj Y …
          -/
          dsimp
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : CategoryTheory.Limits.Cocone F
            t : (k : K) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C …
            s : CategoryTheory.Limits.Cocone F
            X Y : K
            f : Quiver.Hom X Y
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj j).map f) ((s.ι.app j).app Y) …
          -/
          simp }
          /-
            🎉 no goals
          -/
                /-
                  C : Type u
                  inst✝³ : CategoryTheory.Category.{v, u} C
                  D : Type u'
                  inst✝² : CategoryTheory.Category.{v', u'} D
                  J : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                  K : Type u₂
                  inst✝ : CategoryTheory.Category.{v₂, u₂} K
                  F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                  c : CategoryTheory.Limits.Cocone F
                  t : (k : K) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C …
                  s : CategoryTheory.Limits.Cocone F
                  j : J
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) ((fun s => { app := fun k …
                -/
  fac s j := by ext k; exact (t k).fac _ j
                       /-
                         🎉 no goals
                       -/
  uniq s m w := by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
      c : CategoryTheory.Limits.Cocone F
      t : (k : K) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom c.pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) m) (s.ι.app j)
      ⊢ Eq m ((fun s => { app := fun k => (t k).desc { pt := s.pt.obj k, ι := Catego …
    -/
    ext x
    exact (t x).hom_ext fun j =>
      (congr_app (w j) x).trans
        ((t x).fac ⟨s.pt.obj _, whiskerRight s.ι ((evaluation K C).obj _)⟩ j).symm


/--
Given a functor `F` and a collection of colimit cocones for each diagram `X ↦ F X k`, we can stitch
them together to give a cocone for the diagram `F`.
`combinedIsColimit` shows that the new cocone is colimiting, and `evalCombined` shows it is
(essentially) made up of the original cocones.
-/
@[simps]
def combineCocones (F : J ⥤ K ⥤ C) (c : ∀ k : K, ColimitCocone (F.flip.obj k)) : Cocone F where
  pt :=
    { obj := fun k => (c k).cocone.pt
      map := fun {k₁} {k₂} f => (c k₁).isColimit.desc ⟨_, F.flip.map f ≫ (c k₂).cocone.ι⟩
      map_id := fun k =>
        (c k).isColimit.hom_ext fun j => by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : (k : K) → CategoryTheory.Limits.ColimitCocone (F.flip.obj k)
            k : K
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c k).cocone.ι.app j) ({ obj := fun  …
          -/
          dsimp
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            D : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} D
            J : Type u₁
            inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
            K : Type u₂
            inst✝ : CategoryTheory.Category.{v₂, u₂} K
            F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
            c : (k : K) → CategoryTheory.Limits.ColimitCocone (F.flip.obj k)
            k : K
            j : J
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c k).cocone.ι.app j) ((c k).isColim …
          -/
          simp
          /-
            🎉 no goals
          -/
                                                                                   /-
                                                                                     C : Type u
                                                                                     inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                     D : Type u'
                                                                                     inst✝² : CategoryTheory.Category.{v', u'} D
                                                                                     J : Type u₁
                                                                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                                     K : Type u₂
                                                                                     inst✝ : CategoryTheory.Category.{v₂, u₂} K
                                                                                     F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                                                                                     c : (k : K) → CategoryTheory.Limits.ColimitCocone (F.flip.obj k)
                                                                                     k₁ k₂ k₃ : K
                                                                                     f₁ : Quiver.Hom k₁ k₂
                                                                                     f₂ : Quiver.Hom k₂ k₃
                                                                                     j : J
                                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ((c k₁).cocone.ι.app j) ({ obj := fun …
                                                                                   -/
      map_comp := fun {k₁} {k₂} {k₃} f₁ f₂ => (c k₁).isColimit.hom_ext fun j => by simp }
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  ι :=
    { app := fun j => { app := fun k => (c k).cocone.ι.app j }
                                      /-
                                        C : Type u
                                        inst✝³ : CategoryTheory.Category.{v, u} C
                                        D : Type u'
                                        inst✝² : CategoryTheory.Category.{v', u'} D
                                        J : Type u₁
                                        inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                        K : Type u₂
                                        inst✝ : CategoryTheory.Category.{v₂, u₂} K
                                        F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
                                        c : (k : K) → CategoryTheory.Limits.ColimitCocone (F.flip.obj k)
                                        j₁ j₂ : J
                                        g : Quiver.Hom j₁ j₂
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map g) ((fun j => { app := fun k = …
                                      -/
      naturality := fun j₁ j₂ g => by ext k; exact (c k).cocone.ι.naturality g }
                                             /-
                                               🎉 no goals
                                             -/


/-- The stitched together cocones each project down to the original given cocones (up to iso). -/
def evaluateCombinedCocones (F : J ⥤ K ⥤ C) (c : ∀ k : K, ColimitCocone (F.flip.obj k)) (k : K) :
    ((evaluation K C).obj k).mapCocone (combineCocones F c) ≅ (c k).cocone :=
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    c : (k : K) → CategoryTheory.Limits.ColimitCocone (F.flip.obj k)
    k : K
    ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.evaluat …
  -/
  Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- Stitching together colimiting cocones gives a colimiting cocone. -/
def combinedIsColimit (F : J ⥤ K ⥤ C) (c : ∀ k : K, ColimitCocone (F.flip.obj k)) :
    IsColimit (combineCocones F c) :=
  evaluationJointlyReflectsColimits _ fun k =>
    (c k).isColimit.ofIsoColimit (evaluateCombinedCocones F c k).symm


/--
An alternative colimit cocone in the functor category `K ⥤ C` in the case where `C` has
`J`-shaped colimits, with cocone point `F.flip ⋙ colim`.
-/
@[simps]
noncomputable def pointwiseCocone [HasColimitsOfShape J C] (F : J ⥤ K ⥤ C) : Cocone F where
  pt := F.flip ⋙ colim
  ι := {
    app X := { app Y := (colimit.ι _ X : (F.flip.obj Y).obj X ⟶ _) }
    naturality X Y f := by
      /-
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
        X Y : J
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun X => { app := fun Y = …
      -/
      ext x
      simp only [Functor.const_obj_obj, Functor.comp_obj, colim_obj, NatTrans.comp_app,
        Functor.const_obj_map, Category.comp_id]
      /-
        case w.h
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
        X Y : J
        f : Quiver.Hom X Y
        x : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.map f).app x) (CategoryTheory.Lim …
      -/
      change (F.flip.obj x).map f ≫ _ = _
      /-
        case w.h
        C : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} C
        D : Type u'
        inst✝³ : CategoryTheory.Category.{v', u'} D
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
        F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
        X Y : J
        f : Quiver.Hom X Y
        x : K
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.flip.obj x).map f) (CategoryTheor …
      -/
      rw [colimit.w] }
      /-
        🎉 no goals
      -/


/-- `pointwiseCocone` is indeed a colimit cocone. -/
noncomputable def pointwiseIsColimit [HasColimitsOfShape J C] (F : J ⥤ K ⥤ C) :
    IsColimit (pointwiseCocone F) := by
  apply IsColimit.ofIsoColimit (combinedIsColimit _
    (fun k ↦ ⟨colimit.cocone _, colimit.isColimit _⟩))
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.combineCocones F fun k => { cocone …
  -/
  exact Cocones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


instance functorCategoryHasLimit (F : J ⥤ K ⥤ C) [∀ k, HasLimit (F.flip.obj k)] : HasLimit F :=
  HasLimit.mk
    { cone := combineCones F fun _ => getLimitCone _
      isLimit := combinedIsLimit _ _ }


instance functorCategoryHasLimitsOfShape [HasLimitsOfShape J C] : HasLimitsOfShape J (K ⥤ C) where
  has_limit _ := inferInstance


instance functorCategoryHasColimit (F : J ⥤ K ⥤ C) [∀ k, HasColimit (F.flip.obj k)] :
    HasColimit F :=
  HasColimit.mk
    { cocone := combineCocones F fun _ => getColimitCocone _
      isColimit := combinedIsColimit _ _ }


instance functorCategoryHasColimitsOfShape [HasColimitsOfShape J C] :
    HasColimitsOfShape J (K ⥤ C) where
  has_colimit _ := inferInstance

-- Porting note: previously Lean could see through the binders and infer_instance sufficed

instance functorCategoryHasLimitsOfSize [HasLimitsOfSize.{v₁, u₁} C] :
    HasLimitsOfSize.{v₁, u₁} (K ⥤ C) where
  has_limits_of_shape := fun _ _ => inferInstance

-- Porting note: previously Lean could see through the binders and infer_instance sufficed

instance functorCategoryHasColimitsOfSize [HasColimitsOfSize.{v₁, u₁} C] :
    HasColimitsOfSize.{v₁, u₁} (K ⥤ C) where
  has_colimits_of_shape := fun _ _ => inferInstance


instance hasLimitCompEvaluation (F : J ⥤ K ⥤ C) (k : K) [HasLimit (F.flip.obj k)] :
    HasLimit (F ⋙ (evaluation _ _).obj k) :=
  hasLimitOfIso (F := F.flip.obj k) (Iso.refl _)


instance evaluation_preservesLimit (F : J ⥤ K ⥤ C) [∀ k, HasLimit (F.flip.obj k)] (k : K) :
    PreservesLimit F ((evaluation K C).obj k) :=
    -- Porting note: added a let because X was not inferred
  let X : (k : K) → LimitCone (F.flip.obj k) := fun k => getLimitCone (F.flip.obj k)
  preservesLimit_of_preserves_limit_cone (combinedIsLimit _ X) <|
    IsLimit.ofIsoLimit (limit.isLimit _) (evaluateCombinedCones F X k).symm


instance evaluation_preservesLimitsOfShape [HasLimitsOfShape J C] (k : K) :
    PreservesLimitsOfShape J ((evaluation K C).obj k) where
  preservesLimit := inferInstance


/-- If `F : J ⥤ K ⥤ C` is a functor into a functor category which has a limit,
then the evaluation of that limit at `k` is the limit of the evaluations of `F.obj j` at `k`.
-/
def limitObjIsoLimitCompEvaluation [HasLimitsOfShape J C] (F : J ⥤ K ⥤ C) (k : K) :
    (limit F).obj k ≅ limit (F ⋙ (evaluation K C).obj k) :=
  preservesLimitIso ((evaluation K C).obj k) F


@[reassoc (attr := simp)]
theorem limitObjIsoLimitCompEvaluation_hom_π [HasLimitsOfShape J C] (F : J ⥤ K ⥤ C) (j : J)
    (k : K) :
    (limitObjIsoLimitCompEvaluation F k).hom ≫ limit.π (F ⋙ (evaluation K C).obj k) j =
      (limit.π F j).app k := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitObjIsoLim …
  -/
  dsimp [limitObjIsoLimitCompEvaluation]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIso ((C …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem limitObjIsoLimitCompEvaluation_inv_π_app [HasLimitsOfShape J C] (F : J ⥤ K ⥤ C) (j : J)
    (k : K) :
    (limitObjIsoLimitCompEvaluation F k).inv ≫ (limit.π F j).app k =
      limit.π (F ⋙ (evaluation K C).obj k) j := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitObjIsoLim …
  -/
  dsimp [limitObjIsoLimitCompEvaluation]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.preservesLimitIso ((C …
  -/
  rw [Iso.inv_comp_eq]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq ((CategoryTheory.Limits.limit.π F j).app k) (CategoryTheory.CategoryStruc …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem limit_map_limitObjIsoLimitCompEvaluation_hom [HasLimitsOfShape J C] {i j : K}
    (F : J ⥤ K ⥤ C) (f : i ⟶ j) : (limit F).map f ≫ (limitObjIsoLimitCompEvaluation _ _).hom =
    (limitObjIsoLimitCompEvaluation _ _).hom ≫ limMap (whiskerLeft _ ((evaluation _ _).map f)) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    i j : K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    f : Quiver.Hom i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.limit F).map  …
  -/
  ext
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    i j : K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    f : Quiver.Hom i j
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    i j : K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    f : Quiver.Hom i j
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem limitObjIsoLimitCompEvaluation_inv_limit_map [HasLimitsOfShape J C] {i j : K}
    (F : J ⥤ K ⥤ C) (f : i ⟶ j) : (limitObjIsoLimitCompEvaluation _ _).inv ≫ (limit F).map f =
    limMap (whiskerLeft _ ((evaluation _ _).map f)) ≫ (limitObjIsoLimitCompEvaluation _ _).inv := by
  rw [Iso.inv_comp_eq, ← Category.assoc, Iso.eq_comp_inv,
    limit_map_limitObjIsoLimitCompEvaluation_hom]


@[ext]
theorem limit_obj_ext {H : J ⥤ K ⥤ C} [HasLimitsOfShape J C] {k : K} {W : C}
    {f g : W ⟶ (limit H).obj k}
    (w : ∀ j, f ≫ (Limits.limit.π H j).app k = g ≫ (Limits.limit.π H j).app k) : f = g := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    H : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    k : K
    W : C
    f g : Quiver.Hom W ((CategoryTheory.Limits.limit H).obj k)
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Limit …
    ⊢ Eq f g
  -/
  apply (cancel_mono (limitObjIsoLimitCompEvaluation H k).hom).1
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    H : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    k : K
    W : C
    f g : Quiver.Hom W ((CategoryTheory.Limits.limit H).obj k)
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Limit …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.limitObjIsoL …
  -/
  ext j
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    H : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    k : K
    W : C
    f g : Quiver.Hom W ((CategoryTheory.Limits.limit H).obj k)
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.Limit …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
  -/
  simpa using w j
  /-
    🎉 no goals
  -/


/-- Taking a limit after whiskering by `G` is the same as using `G` and then taking a limit. -/
def limitCompWhiskeringLeftIsoCompLimit (F : J ⥤ K ⥤ C) (G : D ⥤ K) [HasLimitsOfShape J C] :
    limit (F ⋙ (whiskeringLeft _ _ _).obj G) ≅ G ⋙ limit F :=
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor D K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    ⊢ ∀ {X Y : D} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
  -/
  NatIso.ofComponents (fun j =>
  /-
    🎉 no goals
  -/
    limitObjIsoLimitCompEvaluation (F ⋙ (whiskeringLeft _ _ _).obj G) j ≪≫
      HasLimit.isoOfNatIso (isoWhiskerLeft F (whiskeringLeftCompEvaluation G j)) ≪≫
      (limitObjIsoLimitCompEvaluation F (G.obj j)).symm)


@[reassoc (attr := simp)]
theorem limitCompWhiskeringLeftIsoCompLimit_hom_whiskerLeft_π (F : J ⥤ K ⥤ C) (G : D ⥤ K)
    [HasLimitsOfShape J C] (j : J) :
    (limitCompWhiskeringLeftIsoCompLimit F G).hom ≫ whiskerLeft G (limit.π F j) =
      limit.π (F ⋙ (whiskeringLeft _ _ _).obj G) j := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor D K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitCompWhisk …
  -/
  ext d
  /-
    case w.h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor D K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    j : J
    d : D
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitCompWhis …
  -/
  simp [limitCompWhiskeringLeftIsoCompLimit]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem limitCompWhiskeringLeftIsoCompLimit_inv_π (F : J ⥤ K ⥤ C) (G : D ⥤ K)
    [HasLimitsOfShape J C] (j : J) :
    (limitCompWhiskeringLeftIsoCompLimit F G).inv ≫ limit.π (F ⋙ (whiskeringLeft _ _ _).obj G) j =
      whiskerLeft G (limit.π F j) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor D K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limitCompWhisk …
  -/
  simp [Iso.inv_comp_eq]
  /-
    🎉 no goals
  -/


instance hasColimitCompEvaluation (F : J ⥤ K ⥤ C) (k : K) [HasColimit (F.flip.obj k)] :
    HasColimit (F ⋙ (evaluation _ _).obj k) :=
  hasColimitOfIso (F := F.flip.obj k) (Iso.refl _)


instance evaluation_preservesColimit (F : J ⥤ K ⥤ C) [∀ k, HasColimit (F.flip.obj k)] (k : K) :
    PreservesColimit F ((evaluation K C).obj k) :=
  -- Porting note: added a let because X was not inferred
  let X : (k : K) → ColimitCocone (F.flip.obj k) := fun k => getColimitCocone (F.flip.obj k)
  preservesColimit_of_preserves_colimit_cocone (combinedIsColimit _ X) <|
    IsColimit.ofIsoColimit (colimit.isColimit _) (evaluateCombinedCocones F X k).symm


instance evaluation_preservesColimitsOfShape [HasColimitsOfShape J C] (k : K) :
    PreservesColimitsOfShape J ((evaluation K C).obj k) where
  preservesColimit := inferInstance


/-- If `F : J ⥤ K ⥤ C` is a functor into a functor category which has a colimit,
then the evaluation of that colimit at `k` is the colimit of the evaluations of `F.obj j` at `k`.
-/
def colimitObjIsoColimitCompEvaluation [HasColimitsOfShape J C] (F : J ⥤ K ⥤ C) (k : K) :
    (colimit F).obj k ≅ colimit (F ⋙ (evaluation K C).obj k) :=
  preservesColimitIso ((evaluation K C).obj k) F


@[reassoc (attr := simp)]
theorem colimitObjIsoColimitCompEvaluation_ι_inv [HasColimitsOfShape J C] (F : J ⥤ K ⥤ C) (j : J)
    (k : K) :
    colimit.ι (F ⋙ (evaluation K C).obj k) j ≫ (colimitObjIsoColimitCompEvaluation F k).inv =
      (colimit.ι F j).app k := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  dsimp [colimitObjIsoColimitCompEvaluation]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem colimitObjIsoColimitCompEvaluation_ι_app_hom [HasColimitsOfShape J C] (F : J ⥤ K ⥤ C)
    (j : J) (k : K) :
    (colimit.ι F j).app k ≫ (colimitObjIsoColimitCompEvaluation F k).hom =
      colimit.ι (F ⋙ (evaluation K C).obj k) j := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.ι F j …
  -/
  dsimp [colimitObjIsoColimitCompEvaluation]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.ι F j …
  -/
  rw [← Iso.eq_comp_inv]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    j : J
    k : K
    ⊢ Eq ((CategoryTheory.Limits.colimit.ι F j).app k) (CategoryTheory.CategoryStr …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem colimitObjIsoColimitCompEvaluation_inv_colimit_map [HasColimitsOfShape J C] (F : J ⥤ K ⥤ C)
    {i j : K} (f : i ⟶ j) :
    (colimitObjIsoColimitCompEvaluation _ _).inv ≫ (colimit F).map f =
      colimMap (whiskerLeft _ ((evaluation _ _).map f)) ≫
        (colimitObjIsoColimitCompEvaluation _ _).inv := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    i j : K
    f : Quiver.Hom i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitObjIsoC …
  -/
  ext
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    i j : K
    f : Quiver.Hom i j
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  dsimp
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    i j : K
    f : Quiver.Hom i j
    j✝ : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem colimit_map_colimitObjIsoColimitCompEvaluation_hom [HasColimitsOfShape J C] (F : J ⥤ K ⥤ C)
    {i j : K} (f : i ⟶ j) :
    (colimit F).map f ≫ (colimitObjIsoColimitCompEvaluation _ _).hom =
      (colimitObjIsoColimitCompEvaluation _ _).hom ≫
        colimMap (whiskerLeft _ ((evaluation _ _).map f)) := by
  rw [← Iso.inv_comp_eq, ← Category.assoc, ← Iso.eq_comp_inv,
    colimitObjIsoColimitCompEvaluation_inv_colimit_map]


@[ext]
theorem colimit_obj_ext {H : J ⥤ K ⥤ C} [HasColimitsOfShape J C] {k : K} {W : C}
    {f g : (colimit H).obj k ⟶ W} (w : ∀ j, (colimit.ι H j).app k ≫ f = (colimit.ι H j).app k ≫ g) :
    f = g := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    H : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    k : K
    W : C
    f g : Quiver.Hom ((CategoryTheory.Limits.colimit H).obj k) W
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
    ⊢ Eq f g
  -/
  apply (cancel_epi (colimitObjIsoColimitCompEvaluation H k).inv).1
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    H : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    k : K
    W : C
    f g : Quiver.Hom ((CategoryTheory.Limits.colimit H).obj k) W
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitObjIsoC …
  -/
  ext j
  /-
    case w
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    H : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    k : K
    W : C
    f g : Quiver.Hom ((CategoryTheory.Limits.colimit H).obj k) W
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (H.c …
  -/
  simpa using w j
  /-
    🎉 no goals
  -/


/-- Taking a colimit after whiskering by `G` is the same as using `G` and then taking a colimit. -/
def colimitCompWhiskeringLeftIsoCompColimit (F : J ⥤ K ⥤ C) (G : D ⥤ K) [HasColimitsOfShape J C] :
    colimit (F ⋙ (whiskeringLeft _ _ _).obj G) ≅ G ⋙ colimit F :=
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor D K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    ⊢ ∀ {X Y : D} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
  -/
  NatIso.ofComponents (fun j =>
  /-
    🎉 no goals
  -/
    colimitObjIsoColimitCompEvaluation (F ⋙ (whiskeringLeft _ _ _).obj G) j ≪≫
      HasColimit.isoOfNatIso (isoWhiskerLeft F (whiskeringLeftCompEvaluation G j)) ≪≫
      (colimitObjIsoColimitCompEvaluation F (G.obj j)).symm)


@[reassoc (attr := simp)]
theorem ι_colimitCompWhiskeringLeftIsoCompColimit_hom (F : J ⥤ K ⥤ C) (G : D ⥤ K)
    [HasColimitsOfShape J C] (j : J) :
    colimit.ι (F ⋙ (whiskeringLeft _ _ _).obj G) j ≫
      (colimitCompWhiskeringLeftIsoCompColimit F G).hom = whiskerLeft G (colimit.ι F j) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor D K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F.c …
  -/
  ext d
  /-
    case w.h
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor D K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    j : J
    d : D
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (F. …
  -/
  simp [colimitCompWhiskeringLeftIsoCompColimit]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem whiskerLeft_ι_colimitCompWhiskeringLeftIsoCompColimit_inv (F : J ⥤ K ⥤ C) (G : D ⥤ K)
    [HasColimitsOfShape J C] (j : J) :
    whiskerLeft G (colimit.ι F j) ≫ (colimitCompWhiskeringLeftIsoCompColimit F G).inv =
      colimit.ι (F ⋙ (whiskeringLeft _ _ _).obj G) j := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    G : CategoryTheory.Functor D K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft G (Catego …
  -/
  simp [Iso.comp_inv_eq]
  /-
    🎉 no goals
  -/


instance evaluationPreservesLimits [HasLimits C] (k : K) :
    PreservesLimits ((evaluation K C).obj k) where
  preservesLimitsOfShape {_} _𝒥 := inferInstance


/-- `F : D ⥤ K ⥤ C` preserves the limit of some `G : J ⥤ D` if it does for each `k : K`. -/
lemma preservesLimit_of_evaluation (F : D ⥤ K ⥤ C) (G : J ⥤ D)
    (H : ∀ k : K, PreservesLimit G (F ⋙ (evaluation K C).obj k : D ⥤ C)) : PreservesLimit G F :=
  ⟨fun {c} hc => ⟨by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesLimit G (F.comp ((CategoryTheory …
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (F.mapCone c)
    -/
    apply evaluationJointlyReflectsLimits
    /-
      case t
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesLimit G (F.comp ((CategoryTheory …
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ (k : K) → CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation K C).ob …
    -/
    intro X
    /-
      case t
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesLimit G (F.comp ((CategoryTheory …
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit c
      X : K
      ⊢ CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation K C).obj X).mapCo …
    -/
    haveI := H X
    /-
      case t
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesLimit G (F.comp ((CategoryTheory …
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit c
      X : K
      this : CategoryTheory.Limits.PreservesLimit G (F.comp ((CategoryTheory.evaluat …
      ⊢ CategoryTheory.Limits.IsLimit (((CategoryTheory.evaluation K C).obj X).mapCo …
    -/
    change IsLimit ((F ⋙ (evaluation K C).obj X).mapCone c)
    /-
      case t
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesLimit G (F.comp ((CategoryTheory …
      c : CategoryTheory.Limits.Cone G
      hc : CategoryTheory.Limits.IsLimit c
      X : K
      this : CategoryTheory.Limits.PreservesLimit G (F.comp ((CategoryTheory.evaluat …
      ⊢ CategoryTheory.Limits.IsLimit ((F.comp ((CategoryTheory.evaluation K C).obj  …
    -/
    exact isLimitOfPreserves _ hc⟩⟩
    /-
      🎉 no goals
    -/


@[deprecated "No deprecation message was provided." (since := "2024-11-19")]
lemma preservesLimitOfEvaluation (F : D ⥤ K ⥤ C) (G : J ⥤ D)
    (H : ∀ k : K, PreservesLimit G (F ⋙ (evaluation K C).obj k : D ⥤ C)) :
    PreservesLimit G F :=
  preservesLimit_of_evaluation _ _ H


/-- `F : D ⥤ K ⥤ C` preserves limits of shape `J` if it does for each `k : K`. -/
lemma preservesLimitsOfShape_of_evaluation (F : D ⥤ K ⥤ C) (J : Type*) [Category J]
    (_ : ∀ k : K, PreservesLimitsOfShape J (F ⋙ (evaluation K C).obj k)) :
    PreservesLimitsOfShape J F :=
  ⟨fun {G} => preservesLimit_of_evaluation F G fun _ => PreservesLimitsOfShape.preservesLimit⟩


@[deprecated "No deprecation message was provided." (since := "2024-11-19")]
lemma preservesLimitsOfShapeOfEvaluation (F : D ⥤ K ⥤ C) (J : Type*) [Category J]
    (H : ∀ k : K, PreservesLimitsOfShape J (F ⋙ (evaluation K C).obj k)) :
    PreservesLimitsOfShape J F :=
  preservesLimitsOfShape_of_evaluation _ _ H


/-- `F : D ⥤ K ⥤ C` preserves all limits if it does for each `k : K`. -/
lemma preservesLimits_of_evaluation (F : D ⥤ K ⥤ C)
    (_ : ∀ k : K, PreservesLimitsOfSize.{w', w} (F ⋙ (evaluation K C).obj k)) :
    PreservesLimitsOfSize.{w', w} F :=
  ⟨fun {L} _ =>
    preservesLimitsOfShape_of_evaluation F L fun _ => PreservesLimitsOfSize.preservesLimitsOfShape⟩


@[deprecated "No deprecation message was provided." (since := "2024-11-19")]
lemma preservesLimitsOfEvaluation (F : D ⥤ K ⥤ C)
    (H : ∀ k : K, PreservesLimitsOfSize.{w', w} (F ⋙ (evaluation K C).obj k)) :
    PreservesLimitsOfSize.{w', w} F :=
  preservesLimits_of_evaluation _ H


/-- The constant functor `C ⥤ (D ⥤ C)` preserves limits. -/
instance preservesLimits_const : PreservesLimitsOfSize.{w', w} (const D : C ⥤ _) :=
  preservesLimits_of_evaluation _ fun _ =>
    preservesLimits_of_natIso <| Iso.symm <| constCompEvaluationObj _ _


instance evaluation_preservesColimits [HasColimits C] (k : K) :
    PreservesColimits ((evaluation K C).obj k) where
  preservesColimitsOfShape := inferInstance


/-- `F : D ⥤ K ⥤ C` preserves the colimit of some `G : J ⥤ D` if it does for each `k : K`. -/
lemma preservesColimit_of_evaluation (F : D ⥤ K ⥤ C) (G : J ⥤ D)
    (H : ∀ k, PreservesColimit G (F ⋙ (evaluation K C).obj k)) : PreservesColimit G F :=
  ⟨fun {c} hc => ⟨by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesColimit G (F.comp ((CategoryTheo …
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Limits.IsColimit (F.mapCocone c)
    -/
    apply evaluationJointlyReflectsColimits
    /-
      case t
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesColimit G (F.comp ((CategoryTheo …
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ (k : K) → CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C). …
    -/
    intro X
    /-
      case t
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesColimit G (F.comp ((CategoryTheo …
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      X : K
      ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C).obj X).map …
    -/
    haveI := H X
    /-
      case t
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesColimit G (F.comp ((CategoryTheo …
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      X : K
      this : CategoryTheory.Limits.PreservesColimit G (F.comp ((CategoryTheory.evalu …
      ⊢ CategoryTheory.Limits.IsColimit (((CategoryTheory.evaluation K C).obj X).map …
    -/
    change IsColimit ((F ⋙ (evaluation K C).obj X).mapCocone c)
    /-
      case t
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} D
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      F : CategoryTheory.Functor D (CategoryTheory.Functor K C)
      G : CategoryTheory.Functor J D
      H : ∀ (k : K), CategoryTheory.Limits.PreservesColimit G (F.comp ((CategoryTheo …
      c : CategoryTheory.Limits.Cocone G
      hc : CategoryTheory.Limits.IsColimit c
      X : K
      this : CategoryTheory.Limits.PreservesColimit G (F.comp ((CategoryTheory.evalu …
      ⊢ CategoryTheory.Limits.IsColimit ((F.comp ((CategoryTheory.evaluation K C).ob …
    -/
    exact isColimitOfPreserves _ hc⟩⟩
    /-
      🎉 no goals
    -/


@[deprecated "No deprecation message was provided."  (since := "2024-11-19")]
lemma preservesColimitOfEvaluation (F : D ⥤ K ⥤ C) (G : J ⥤ D)
    (H : ∀ k, PreservesColimit G (F ⋙ (evaluation K C).obj k)) : PreservesColimit G F :=
  preservesColimit_of_evaluation _ _ H


/-- `F : D ⥤ K ⥤ C` preserves all colimits of shape `J` if it does for each `k : K`. -/
lemma preservesColimitsOfShape_of_evaluation (F : D ⥤ K ⥤ C) (J : Type*) [Category J]
    (_ : ∀ k : K, PreservesColimitsOfShape J (F ⋙ (evaluation K C).obj k)) :
    PreservesColimitsOfShape J F :=
  ⟨fun {G} => preservesColimit_of_evaluation F G fun _ => PreservesColimitsOfShape.preservesColimit⟩


/-- `F : D ⥤ K ⥤ C` preserves all colimits if it does for each `k : K`. -/
lemma preservesColimits_of_evaluation (F : D ⥤ K ⥤ C)
    (_ : ∀ k : K, PreservesColimitsOfSize.{w', w} (F ⋙ (evaluation K C).obj k)) :
    PreservesColimitsOfSize.{w', w} F :=
  ⟨fun {L} _ =>
    preservesColimitsOfShape_of_evaluation F L fun _ =>
      PreservesColimitsOfSize.preservesColimitsOfShape⟩


/-- The constant functor `C ⥤ (D ⥤ C)` preserves colimits. -/
instance preservesColimits_const : PreservesColimitsOfSize.{w', w} (const D : C ⥤ _) :=
  preservesColimits_of_evaluation _ fun _ =>
    preservesColimits_of_natIso <| Iso.symm <| constCompEvaluationObj _ _


/-- The limit of a diagram `F : J ⥤ K ⥤ C` is isomorphic to the functor given by
the individual limits on objects. -/
@[simps!]
def limitIsoFlipCompLim [HasLimitsOfShape J C] (F : J ⥤ K ⥤ C) : limit F ≅ F.flip ⋙ lim :=
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    ⊢ ∀ {X Y : K} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
  -/
  NatIso.ofComponents (limitObjIsoLimitCompEvaluation F)
  /-
    🎉 no goals
  -/


/-- A variant of `limitIsoFlipCompLim` where the arguments of `F` are flipped. -/
@[simps!]
def limitFlipIsoCompLim [HasLimitsOfShape J C] (F : K ⥤ J ⥤ C) : limit F.flip ≅ F ⋙ lim :=
  let f := fun k =>
    limitObjIsoLimitCompEvaluation F.flip k ≪≫ HasLimit.isoOfNatIso (flipCompEvaluation _ _)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasLimitsOfShape J C
    F : CategoryTheory.Functor K (CategoryTheory.Functor J C)
    f : (k : K) → CategoryTheory.Iso ((CategoryTheory.Limits.limit F.flip).obj k)  …
    ⊢ ∀ {X Y : K} (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
  -/
  NatIso.ofComponents f
  /-
    🎉 no goals
  -/


/-- For a functor `G : J ⥤ K ⥤ C`, its limit `K ⥤ C` is given by `(G' : K ⥤ J ⥤ C) ⋙ lim`.
Note that this does not require `K` to be small.
-/
@[simps!]
def limitIsoSwapCompLim [HasLimitsOfShape J C] (G : J ⥤ K ⥤ C) :
    limit G ≅ curry.obj (Prod.swap K J ⋙ uncurry.obj G) ⋙ lim :=
  limitIsoFlipCompLim G ≪≫ isoWhiskerRight (flipIsoCurrySwapUncurry _) _


/-- The colimit of a diagram `F : J ⥤ K ⥤ C` is isomorphic to the functor given by
the individual colimits on objects. -/
@[simps!]
def colimitIsoFlipCompColim [HasColimitsOfShape J C] (F : J ⥤ K ⥤ C) : colimit F ≅ F.flip ⋙ colim :=
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor J (CategoryTheory.Functor K C)
    ⊢ ∀ {X Y : K} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((C …
  -/
  NatIso.ofComponents (colimitObjIsoColimitCompEvaluation F)
  /-
    🎉 no goals
  -/


/-- A variant of `colimit_iso_flip_comp_colim` where the arguments of `F` are flipped. -/
@[simps!]
def colimitFlipIsoCompColim [HasColimitsOfShape J C] (F : K ⥤ J ⥤ C) : colimit F.flip ≅ F ⋙ colim :=
  let f := fun _ =>
      colimitObjIsoColimitCompEvaluation _ _ ≪≫ HasColimit.isoOfNatIso (flipCompEvaluation _ _)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} D
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape J C
    F : CategoryTheory.Functor K (CategoryTheory.Functor J C)
    f : (x : K) → CategoryTheory.Iso ((CategoryTheory.Limits.colimit F.flip).obj x …
    ⊢ ∀ {X Y : K} (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
  -/
  NatIso.ofComponents f
  /-
    🎉 no goals
  -/


/-- For a functor `G : J ⥤ K ⥤ C`, its colimit `K ⥤ C` is given by `(G' : K ⥤ J ⥤ C) ⋙ colim`.
Note that this does not require `K` to be small.
-/
@[simps!]
def colimitIsoSwapCompColim [HasColimitsOfShape J C] (G : J ⥤ K ⥤ C) :
    colimit G ≅ curry.obj (Prod.swap K J ⋙ uncurry.obj G) ⋙ colim :=
  colimitIsoFlipCompColim G ≪≫ isoWhiskerRight (flipIsoCurrySwapUncurry _) _


