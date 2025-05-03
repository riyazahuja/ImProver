/-- A cone over `F : J ⥤ Π i, C i` has as its components cones over each of the `F ⋙ Pi.eval C i`.
-/
def coneCompEval (c : Cone F) (i : I) : Cone (F ⋙ Pi.eval C i) where
  pt := c.pt i
  π :=
    { app := fun j => c.π.app j i
      naturality := fun _ _ f => congr_fun (c.π.naturality f) i }


/--
A cocone over `F : J ⥤ Π i, C i` has as its components cocones over each of the `F ⋙ Pi.eval C i`.
-/
def coconeCompEval (c : Cocone F) (i : I) : Cocone (F ⋙ Pi.eval C i) where
  pt := c.pt i
  ι :=
    { app := fun j => c.ι.app j i
      naturality := fun _ _ f => congr_fun (c.ι.naturality f) i }


/--
Given a family of cones over the `F ⋙ Pi.eval C i`, we can assemble these together as a `Cone F`.
-/
def coneOfConeCompEval (c : ∀ i, Cone (F ⋙ Pi.eval C i)) : Cone F where
  pt i := (c i).pt
  π :=
    { app := fun j i => (c i).π.app j
      naturality := fun j j' f => by
        /-
          I : Type v₁
          C : I → Type u₁
          inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
          J : Type v₁
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J ((i : I) → C i)
          c : (i : I) → CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Pi.eval C i))
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        funext i
        /-
          case h
          I : Type v₁
          C : I → Type u₁
          inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
          J : Type v₁
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J ((i : I) → C i)
          c : (i : I) → CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Pi.eval C i))
          j j' : J
          f : Quiver.Hom j j'
          i : I
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        exact (c i).π.naturality f }
        /-
          🎉 no goals
        -/


/-- Given a family of cocones over the `F ⋙ Pi.eval C i`,
we can assemble these together as a `Cocone F`.
-/
def coconeOfCoconeCompEval (c : ∀ i, Cocone (F ⋙ Pi.eval C i)) : Cocone F where
  pt i := (c i).pt
  ι :=
    { app := fun j i => (c i).ι.app j
      naturality := fun j j' f => by
        /-
          I : Type v₁
          C : I → Type u₁
          inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
          J : Type v₁
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J ((i : I) → C i)
          c : (i : I) → CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Pi.eval C i))
          j j' : J
          f : Quiver.Hom j j'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j i => (c i).ι.app j) …
        -/
        funext i
        /-
          case h
          I : Type v₁
          C : I → Type u₁
          inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
          J : Type v₁
          inst✝ : CategoryTheory.SmallCategory J
          F : CategoryTheory.Functor J ((i : I) → C i)
          c : (i : I) → CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Pi.eval C i))
          j j' : J
          f : Quiver.Hom j j'
          i : I
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j i => (c i).ι.app j) …
        -/
        exact (c i).ι.naturality f }
        /-
          🎉 no goals
        -/


/-- Given a family of limit cones over the `F ⋙ Pi.eval C i`,
assembling them together as a `Cone F` produces a limit cone.
-/
def coneOfConeEvalIsLimit {c : ∀ i, Cone (F ⋙ Pi.eval C i)} (P : ∀ i, IsLimit (c i)) :
    IsLimit (coneOfConeCompEval c) where
  lift s i := (P i).lift (coneCompEval s i)
  fac s j := by
    /-
      I : Type v₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type v₁
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ((i : I) → C i)
      c : (i : I) → CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Pi.eval C i))
      P : (i : I) → CategoryTheory.Limits.IsLimit (c i)
      s : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s i => (P i).lift (CategoryTheo …
    -/
    funext i
    /-
      case h
      I : Type v₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type v₁
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ((i : I) → C i)
      c : (i : I) → CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Pi.eval C i))
      P : (i : I) → CategoryTheory.Limits.IsLimit (c i)
      s : CategoryTheory.Limits.Cone F
      j : J
      i : I
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s i => (P i).lift (CategoryTheo …
    -/
    exact (P i).fac (coneCompEval s i) j
    /-
      🎉 no goals
    -/
  uniq s m w := by
    /-
      I : Type v₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type v₁
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ((i : I) → C i)
      c : (i : I) → CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Pi.eval C i))
      P : (i : I) → CategoryTheory.Limits.IsLimit (c i)
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.pi.coneOfConeCompEval c).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.pi.co …
      ⊢ Eq m ((fun s i => (P i).lift (CategoryTheory.pi.coneCompEval s i)) s)
    -/
    funext i
    /-
      case h
      I : Type v₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type v₁
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ((i : I) → C i)
      c : (i : I) → CategoryTheory.Limits.Cone (F.comp (CategoryTheory.Pi.eval C i))
      P : (i : I) → CategoryTheory.Limits.IsLimit (c i)
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.pi.coneOfConeCompEval c).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.pi.co …
      i : I
      ⊢ Eq (m i) ((fun s i => (P i).lift (CategoryTheory.pi.coneCompEval s i)) s i)
    -/
    exact (P i).uniq (coneCompEval s i) (m i) fun j => congr_fun (w j) i
    /-
      🎉 no goals
    -/


/-- Given a family of colimit cocones over the `F ⋙ Pi.eval C i`,
assembling them together as a `Cocone F` produces a colimit cocone.
-/
def coconeOfCoconeEvalIsColimit {c : ∀ i, Cocone (F ⋙ Pi.eval C i)} (P : ∀ i, IsColimit (c i)) :
    IsColimit (coconeOfCoconeCompEval c) where
  desc s i := (P i).desc (coconeCompEval s i)
  fac s j := by
    /-
      I : Type v₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type v₁
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ((i : I) → C i)
      c : (i : I) → CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Pi.eval C i))
      P : (i : I) → CategoryTheory.Limits.IsColimit (c i)
      s : CategoryTheory.Limits.Cocone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pi.coconeOfCoconeCom …
    -/
    funext i
    /-
      case h
      I : Type v₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type v₁
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ((i : I) → C i)
      c : (i : I) → CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Pi.eval C i))
      P : (i : I) → CategoryTheory.Limits.IsColimit (c i)
      s : CategoryTheory.Limits.Cocone F
      j : J
      i : I
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pi.coconeOfCoconeCom …
    -/
    exact (P i).fac (coconeCompEval s i) j
    /-
      🎉 no goals
    -/
  uniq s m w := by
    /-
      I : Type v₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type v₁
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ((i : I) → C i)
      c : (i : I) → CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Pi.eval C i))
      P : (i : I) → CategoryTheory.Limits.IsColimit (c i)
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.pi.coconeOfCoconeCompEval c).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pi.coco …
      ⊢ Eq m ((fun s i => (P i).desc (CategoryTheory.pi.coconeCompEval s i)) s)
    -/
    funext i
    /-
      case h
      I : Type v₁
      C : I → Type u₁
      inst✝¹ : (i : I) → CategoryTheory.Category.{v₁, u₁} (C i)
      J : Type v₁
      inst✝ : CategoryTheory.SmallCategory J
      F : CategoryTheory.Functor J ((i : I) → C i)
      c : (i : I) → CategoryTheory.Limits.Cocone (F.comp (CategoryTheory.Pi.eval C i))
      P : (i : I) → CategoryTheory.Limits.IsColimit (c i)
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.pi.coconeOfCoconeCompEval c).pt s.pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pi.coco …
      i : I
      ⊢ Eq (m i) ((fun s i => (P i).desc (CategoryTheory.pi.coconeCompEval s i)) s i)
    -/
    exact (P i).uniq (coconeCompEval s i) (m i) fun j => congr_fun (w j) i
    /-
      🎉 no goals
    -/


/-- If we have a functor `F : J ⥤ Π i, C i` into a category of indexed families,
and we have limits for each of the `F ⋙ Pi.eval C i`,
then `F` has a limit.
-/
theorem hasLimit_of_hasLimit_comp_eval : HasLimit F :=
  HasLimit.mk
    { cone := coneOfConeCompEval fun _ => limit.cone _
      isLimit := coneOfConeEvalIsLimit fun _ => limit.isLimit _ }


/-- If we have a functor `F : J ⥤ Π i, C i` into a category of indexed families,
and colimits exist for each of the `F ⋙ Pi.eval C i`,
there is a colimit for `F`.
-/
theorem hasColimit_of_hasColimit_comp_eval : HasColimit F :=
  HasColimit.mk
    { cocone := coconeOfCoconeCompEval fun _ => colimit.cocone _
      isColimit := coconeOfCoconeEvalIsColimit fun _ => colimit.isColimit _ }


