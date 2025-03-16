/-- If `L : C ⥤ D` satisfies the universal property of the localisation
for `W : MorphismProperty C`, then `L.op` also does. -/
def StrictUniversalPropertyFixedTarget.op {E : Type*} [Category E]
    (h : StrictUniversalPropertyFixedTarget L W Eᵒᵖ) :
    StrictUniversalPropertyFixedTarget L.op W.op E where
  inverts := h.inverts.op
  lift F hF := (h.lift F.rightOp hF.rightOp).leftOp
  fac F hF := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.64, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.68, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.116, u_3} E
      h : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L W (Opposi …
      F : CategoryTheory.Functor (Opposite C) E
      hF : W.op.IsInvertedBy F
      ⊢ Eq (L.op.comp ((fun F hF => (h.lift F.rightOp ⋯).leftOp) F hF)) F
    -/
    convert congr_arg Functor.leftOp (h.fac F.rightOp hF.rightOp)
    /-
      🎉 no goals
    -/
  uniq F₁ F₂ eq := by
    suffices F₁.rightOp = F₂.rightOp by
      rw [← F₁.rightOp_leftOp_eq, ← F₂.rightOp_leftOp_eq, this]
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.64, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.68, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.116, u_3} E
      h : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L W (Opposi …
      F₁ F₂ : CategoryTheory.Functor (Opposite D) E
      eq : Eq (L.op.comp F₁) (L.op.comp F₂)
      ⊢ Eq F₁.rightOp F₂.rightOp
    -/
    have eq' := congr_arg Functor.rightOp eq
    /-
      C : Type u_1
      D : Type u_2
      inst✝² : CategoryTheory.Category.{?u.64, u_1} C
      inst✝¹ : CategoryTheory.Category.{?u.68, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      E : Type u_3
      inst✝ : CategoryTheory.Category.{?u.116, u_3} E
      h : CategoryTheory.Localization.StrictUniversalPropertyFixedTarget L W (Opposi …
      F₁ F₂ : CategoryTheory.Functor (Opposite D) E
      eq : Eq (L.op.comp F₁) (L.op.comp F₂)
      eq' : Eq (L.op.comp F₁).rightOp (L.op.comp F₂).rightOp
      ⊢ Eq F₁.rightOp F₂.rightOp
    -/
    exact h.uniq _ _ eq'
    /-
      🎉 no goals
    -/


instance isLocalization_op : W.Q.op.IsLocalization W.op :=
  Functor.IsLocalization.mk' W.Q.op W.op (strictUniversalPropertyFixedTargetQ W _).op
    (strictUniversalPropertyFixedTargetQ W _).op


instance IsLocalization.op : L.op.IsLocalization W.op :=
  IsLocalization.of_equivalence_target W.Q.op W.op L.op (Localization.equivalenceFromModel L W).op
    (NatIso.op (Localization.qCompEquivalenceFromModelFunctorIso L W).symm)


lemma isoOfHom_unop  {X Y : Cᵒᵖ} (w : X ⟶ Y) (hw : W.op w) :
                                                                    /-
                                                                      C : Type u_1
                                                                      D : Type u_2
                                                                      inst✝² : CategoryTheory.Category.{u_3, u_1} C
                                                                      inst✝¹ : CategoryTheory.Category.{u_4, u_2} D
                                                                      L : CategoryTheory.Functor C D
                                                                      W : CategoryTheory.MorphismProperty C
                                                                      inst✝ : L.IsLocalization W
                                                                      X Y : Opposite C
                                                                      w : Quiver.Hom X Y
                                                                      hw : W.op w
                                                                      ⊢ Eq (CategoryTheory.Localization.isoOfHom L.op W.op w hw).unop (CategoryTheor …
                                                                    -/
    (isoOfHom L.op W.op w hw).unop = (isoOfHom L W w.unop hw) := by ext; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma isoOfHom_op_inv {X Y : Cᵒᵖ} (w : X ⟶ Y) (hw : W.op w) :
    (isoOfHom L.op W.op w hw).inv = (isoOfHom L W w.unop hw).inv.op :=
  congr_arg Quiver.Hom.op (congr_arg Iso.inv (isoOfHom_unop L W w hw))


