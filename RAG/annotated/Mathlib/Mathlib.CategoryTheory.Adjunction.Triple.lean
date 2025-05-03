lemma isIso_unit_iff_isIso_counit : IsIso adj₁.unit ↔ IsIso adj₂.counit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F H : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction F G
    adj₂ : CategoryTheory.Adjunction G H
    ⊢ Iff (CategoryTheory.IsIso adj₁.unit) (CategoryTheory.IsIso adj₂.counit)
  -/
  let adj : F ⋙ G ⊣ H ⋙ G := adj₁.comp adj₂
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F H : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj₁ : CategoryTheory.Adjunction F G
    adj₂ : CategoryTheory.Adjunction G H
    adj : CategoryTheory.Adjunction (F.comp G) (H.comp G) := adj₁.comp adj₂
    ⊢ Iff (CategoryTheory.IsIso adj₁.unit) (CategoryTheory.IsIso adj₂.counit)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      F H : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction F G
      adj₂ : CategoryTheory.Adjunction G H
      adj : CategoryTheory.Adjunction (F.comp G) (H.comp G) := adj₁.comp adj₂
      ⊢ CategoryTheory.IsIso adj₁.unit → CategoryTheory.IsIso adj₂.counit
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      F H : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction F G
      adj₂ : CategoryTheory.Adjunction G H
      adj : CategoryTheory.Adjunction (F.comp G) (H.comp G) := adj₁.comp adj₂
      h : CategoryTheory.IsIso adj₁.unit
      ⊢ CategoryTheory.IsIso adj₂.counit
    -/
    let idAdj : 𝟭 C ⊣ H ⋙ G := adj.ofNatIsoLeft (asIso adj₁.unit).symm
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      F H : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction F G
      adj₂ : CategoryTheory.Adjunction G H
      adj : CategoryTheory.Adjunction (F.comp G) (H.comp G) := adj₁.comp adj₂
      h : CategoryTheory.IsIso adj₁.unit
      idAdj : CategoryTheory.Adjunction (CategoryTheory.Functor.id C) (H.comp G) :=  …
      ⊢ CategoryTheory.IsIso adj₂.counit
    -/
    exact adj₂.isIso_counit_of_iso (idAdj.rightAdjointUniq id)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      F H : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction F G
      adj₂ : CategoryTheory.Adjunction G H
      adj : CategoryTheory.Adjunction (F.comp G) (H.comp G) := adj₁.comp adj₂
      ⊢ CategoryTheory.IsIso adj₂.counit → CategoryTheory.IsIso adj₁.unit
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      F H : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction F G
      adj₂ : CategoryTheory.Adjunction G H
      adj : CategoryTheory.Adjunction (F.comp G) (H.comp G) := adj₁.comp adj₂
      h : CategoryTheory.IsIso adj₂.counit
      ⊢ CategoryTheory.IsIso adj₁.unit
    -/
    let adjId : F ⋙ G ⊣ 𝟭 C := adj.ofNatIsoRight (asIso adj₂.counit)
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Category.{u_4, u_2} D
      F H : CategoryTheory.Functor C D
      G : CategoryTheory.Functor D C
      adj₁ : CategoryTheory.Adjunction F G
      adj₂ : CategoryTheory.Adjunction G H
      adj : CategoryTheory.Adjunction (F.comp G) (H.comp G) := adj₁.comp adj₂
      h : CategoryTheory.IsIso adj₂.counit
      adjId : CategoryTheory.Adjunction (F.comp G) (CategoryTheory.Functor.id C) :=  …
      ⊢ CategoryTheory.IsIso adj₁.unit
    -/
    exact adj₁.isIso_unit_of_iso (adjId.leftAdjointUniq id)
    /-
      🎉 no goals
    -/


/--
Given an adjoint triple `F ⊣ G ⊣ H`, the left adjoint `F` is fully faithful if and only if the
right adjoint `H` is fully faithful.
-/
noncomputable def fullyFaithfulEquiv : F.FullyFaithful ≃ H.FullyFaithful where
  toFun h :=
    haveI := h.full
    haveI := h.faithful
    haveI : IsIso adj₂.counit := by
      /-
        C : Type u_1
        D : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.2856, u_1} C
        inst✝ : CategoryTheory.Category.{?u.2860, u_2} D
        F H : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj₁ : CategoryTheory.Adjunction F G
        adj₂ : CategoryTheory.Adjunction G H
        h : F.FullyFaithful
        this✝ : F.Full
        this : F.Faithful
        ⊢ CategoryTheory.IsIso adj₂.counit
      -/
      rw [← adj₁.isIso_unit_iff_isIso_counit adj₂]
      /-
        C : Type u_1
        D : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.2856, u_1} C
        inst✝ : CategoryTheory.Category.{?u.2860, u_2} D
        F H : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj₁ : CategoryTheory.Adjunction F G
        adj₂ : CategoryTheory.Adjunction G H
        h : F.FullyFaithful
        this✝ : F.Full
        this : F.Faithful
        ⊢ CategoryTheory.IsIso adj₁.unit
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    adj₂.fullyFaithfulROfIsIsoCounit
  invFun h :=
    haveI := h.full
    haveI := h.faithful
    haveI : IsIso adj₁.unit := by
      /-
        C : Type u_1
        D : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.2856, u_1} C
        inst✝ : CategoryTheory.Category.{?u.2860, u_2} D
        F H : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj₁ : CategoryTheory.Adjunction F G
        adj₂ : CategoryTheory.Adjunction G H
        h : H.FullyFaithful
        this✝ : H.Full
        this : H.Faithful
        ⊢ CategoryTheory.IsIso adj₁.unit
      -/
      rw [adj₁.isIso_unit_iff_isIso_counit adj₂]
      /-
        C : Type u_1
        D : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.2856, u_1} C
        inst✝ : CategoryTheory.Category.{?u.2860, u_2} D
        F H : CategoryTheory.Functor C D
        G : CategoryTheory.Functor D C
        adj₁ : CategoryTheory.Adjunction F G
        adj₂ : CategoryTheory.Adjunction G H
        h : H.FullyFaithful
        this✝ : H.Full
        this : H.Faithful
        ⊢ CategoryTheory.IsIso adj₂.counit
      -/
      infer_instance
      /-
        🎉 no goals
      -/
    adj₁.fullyFaithfulLOfIsIsoUnit
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := Subsingleton.elim _ _


