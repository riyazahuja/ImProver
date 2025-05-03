/-- A trivial cone for a functor into `PUnit`. `punitConeIsLimit` shows it is a limit. -/
def punitCone : Cone F :=
  ⟨⟨⟨⟩⟩, (Functor.punitExt _ _).hom⟩


/-- A trivial cocone for a functor into `PUnit`. `punitCoconeIsLimit` shows it is a colimit. -/
def punitCocone : Cocone F :=
  ⟨⟨⟨⟩⟩, (Functor.punitExt _ _).hom⟩


/-- Any cone over a functor into `PUnit` is a limit cone.
-/
def punitConeIsLimit {c : Cone F} : IsLimit c where
                               /-
                                 J : Type v
                                 inst✝ : CategoryTheory.Category.{v', v} J
                                 F : CategoryTheory.Functor J (CategoryTheory.Discrete PUnit.{?u.1306 + 1})
                                 c s : CategoryTheory.Limits.Cone F
                                 ⊢ Eq s.pt c.pt
                               -/
  lift := fun s => eqToHom (by simp [eq_iff_true_of_subsingleton])
                               /-
                                 🎉 no goals
                               -/


/-- Any cocone over a functor into `PUnit` is a colimit cocone.
-/
def punitCoconeIsColimit {c : Cocone F} : IsColimit c where
                               /-
                                 J : Type v
                                 inst✝ : CategoryTheory.Category.{v', v} J
                                 F : CategoryTheory.Functor J (CategoryTheory.Discrete PUnit.{?u.3110 + 1})
                                 c s : CategoryTheory.Limits.Cocone F
                                 ⊢ Eq c.pt s.pt
                               -/
  desc := fun s => eqToHom (by simp [eq_iff_true_of_subsingleton])
                               /-
                                 🎉 no goals
                               -/


instance : HasLimitsOfSize.{v', v} (Discrete PUnit) :=
  ⟨fun _ _ => ⟨fun _ => ⟨punitCone, punitConeIsLimit⟩⟩⟩


instance : HasColimitsOfSize.{v', v} (Discrete PUnit) :=
  ⟨fun _ _ => ⟨fun _ => ⟨punitCocone, punitCoconeIsColimit⟩⟩⟩


