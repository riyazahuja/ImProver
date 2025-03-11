/-- (Impl) Given a diagram in the over category, produce a natural transformation from the
diagram legs to the specific object.
-/
def natTransInOver {B : C} (F : J ⥤ Over B) :
    F ⋙ forget B ⟶ (CategoryTheory.Functor.const J).obj B where
  app j := (F.obj j).hom


/-- (Impl) Given a cone in the base category, raise it to a cone in the over category. Note this is
where the connected assumption is used.
-/
@[simps]
def raiseCone [IsConnected J] {B : C} {F : J ⥤ Over B} (c : Cone (F ⋙ forget B)) :
    Cone F where
  pt := Over.mk (c.π.app (Classical.arbitrary J) ≫ (F.obj (Classical.arbitrary J)).hom)
  π :=
    { app := fun j =>
        Over.homMk (c.π.app j) (nat_trans_from_is_connected (c.π ≫ natTransInOver F) j _)
      naturality := by
        /-
          J : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} J
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X : C
          inst✝ : CategoryTheory.IsConnected J
          B : C
          F : CategoryTheory.Functor J (CategoryTheory.Over B)
          c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
          ⊢ ∀ ⦃X Y : J⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
        -/
        intro X Y f
        /-
          J : Type u'
          inst✝² : CategoryTheory.Category.{v', u'} J
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X✝ : C
          inst✝ : CategoryTheory.IsConnected J
          B : C
          F : CategoryTheory.Functor J (CategoryTheory.Over B)
          c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
          X Y : J
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        apply CommaMorphism.ext
          /-
            case left
            J : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X✝ : C
            inst✝ : CategoryTheory.IsConnected J
            B : C
            F : CategoryTheory.Functor J (CategoryTheory.Over B)
            c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
            X Y : J
            f : Quiver.Hom X Y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
          -/
        · simpa using (c.w f).symm
          /-
            🎉 no goals
          -/
          /-
            case right
            J : Type u'
            inst✝² : CategoryTheory.Category.{v', u'} J
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            X✝ : C
            inst✝ : CategoryTheory.IsConnected J
            B : C
            F : CategoryTheory.Functor J (CategoryTheory.Over B)
            c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
            X Y : J
            f : Quiver.Hom X Y
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
          -/
        · simp }
          /-
            🎉 no goals
          -/


theorem raised_cone_lowers_to_original [IsConnected J] {B : C} {F : J ⥤ Over B}
    (c : Cone (F ⋙ forget B)) :
                                               /-
                                                 J : Type u'
                                                 inst✝² : CategoryTheory.Category.{v', u'} J
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 inst✝ : CategoryTheory.IsConnected J
                                                 B : C
                                                 F : CategoryTheory.Functor J (CategoryTheory.Over B)
                                                 c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
                                                 ⊢ Eq ((CategoryTheory.Over.forget B).mapCone (CategoryTheory.Over.CreatesConne …
                                               -/
    (forget B).mapCone (raiseCone c) = c := by aesop_cat
                                               /-
                                                 🎉 no goals
                                               -/


/-- (Impl) Show that the raised cone is a limit. -/
def raisedConeIsLimit [IsConnected J] {B : C} {F : J ⥤ Over B} {c : Cone (F ⋙ forget B)}
    (t : IsLimit c) : IsLimit (raiseCone c) where
  lift s :=
    /-
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X : C
      inst✝ : CategoryTheory.IsConnected J
      B : C
      F : CategoryTheory.Functor J (CategoryTheory.Over B)
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.lift ((CategoryTheory.Over.forget  …
    -/
    Over.homMk (t.lift ((forget B).mapCone s))
    /-
      🎉 no goals
    -/
  uniq s m K := by
    /-
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X : C
      inst✝ : CategoryTheory.IsConnected J
      B : C
      F : CategoryTheory.Functor J (CategoryTheory.Over B)
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Over.CreatesConnected.raiseCone c).pt
      K : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Over. …
      ⊢ Eq m ((fun s => CategoryTheory.Over.homMk (t.lift ((CategoryTheory.Over.forg …
    -/
    ext1
    /-
      case h
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X : C
      inst✝ : CategoryTheory.IsConnected J
      B : C
      F : CategoryTheory.Functor J (CategoryTheory.Over B)
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Over.CreatesConnected.raiseCone c).pt
      K : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Over. …
      ⊢ Eq m.left ((fun s => CategoryTheory.Over.homMk (t.lift ((CategoryTheory.Over …
    -/
    apply t.hom_ext
    /-
      case h
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X : C
      inst✝ : CategoryTheory.IsConnected J
      B : C
      F : CategoryTheory.Functor J (CategoryTheory.Over B)
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Over.CreatesConnected.raiseCone c).pt
      K : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Over. …
      ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m.left (c.π.app j)) (Categ …
    -/
    intro j
    /-
      case h
      J : Type u'
      inst✝² : CategoryTheory.Category.{v', u'} J
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X : C
      inst✝ : CategoryTheory.IsConnected J
      B : C
      F : CategoryTheory.Functor J (CategoryTheory.Over B)
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Over.forget B))
      t : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Over.CreatesConnected.raiseCone c).pt
      K : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Over. …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp m.left (c.π.app j)) (CategoryTheory.C …
    -/
    simp [← K j]
    /-
      🎉 no goals
    -/


/-- The forgetful functor from the over category creates any connected limit. -/
instance forgetCreatesConnectedLimits [IsConnected J] {B : C} :
    CreatesLimitsOfShape J (forget B) where
  CreatesLimit :=
    createsLimitOfReflectsIso fun c t =>
      { liftedCone := CreatesConnected.raiseCone c
        validLift := eqToIso (CreatesConnected.raised_cone_lowers_to_original c)
        makesLimit := CreatesConnected.raisedConeIsLimit t }


/-- The over category has any connected limit which the original category has. -/
instance has_connected_limits {B : C} [IsConnected J] [HasLimitsOfShape J C] :
    HasLimitsOfShape J (Over B) where
  has_limit F := hasLimit_of_created F (forget B)


