/-- A short complex `S` is short exact if it is exact, `S.f` is a mono and `S.g` is an epi. -/
structure ShortExact : Prop where
  exact : S.Exact
  [mono_f : Mono S.f]
  [epi_g : Epi S.g]


lemma ShortExact.mk' (h : S.Exact) (_ : Mono S.f) (_ : Epi S.g) : S.ShortExact where
  exact := h


lemma shortExact_of_iso (e : S₁ ≅ S₂) (h : S₁.ShortExact) : S₂.ShortExact where
  exact := exact_of_iso e h.exact
  mono_f := by
    suffices Mono (S₂.f ≫ e.inv.τ₂) by
      exact mono_of_mono _ e.inv.τ₂
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      e : CategoryTheory.Iso S₁ S₂
      h : S₁.ShortExact
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp S₂.f e.inv.τ₂)
    -/
    have := h.mono_f
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      e : CategoryTheory.Iso S₁ S₂
      h : S₁.ShortExact
      this : CategoryTheory.Mono S₁.f
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp S₂.f e.inv.τ₂)
    -/
    rw [← e.inv.comm₁₂]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      e : CategoryTheory.Iso S₁ S₂
      h : S₁.ShortExact
      this : CategoryTheory.Mono S₁.f
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp e.inv.τ₁ S₁.f)
    -/
    apply mono_comp
    /-
      🎉 no goals
    -/
  epi_g := by
    suffices Epi (e.hom.τ₂ ≫ S₂.g) by
      exact epi_of_epi e.hom.τ₂ _
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      e : CategoryTheory.Iso S₁ S₂
      h : S₁.ShortExact
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp e.hom.τ₂ S₂.g)
    -/
    have := h.epi_g
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      e : CategoryTheory.Iso S₁ S₂
      h : S₁.ShortExact
      this : CategoryTheory.Epi S₁.g
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp e.hom.τ₂ S₂.g)
    -/
    rw [e.hom.comm₂₃]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      e : CategoryTheory.Iso S₁ S₂
      h : S₁.ShortExact
      this : CategoryTheory.Epi S₁.g
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp S₁.g e.hom.τ₃)
    -/
    apply epi_comp
    /-
      🎉 no goals
    -/


lemma shortExact_iff_of_iso (e : S₁ ≅ S₂) : S₁.ShortExact ↔ S₂.ShortExact := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    e : CategoryTheory.Iso S₁ S₂
    ⊢ Iff S₁.ShortExact S₂.ShortExact
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      e : CategoryTheory.Iso S₁ S₂
      ⊢ S₁.ShortExact → S₂.ShortExact
    -/
  · exact shortExact_of_iso e
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      e : CategoryTheory.Iso S₁ S₂
      ⊢ S₂.ShortExact → S₁.ShortExact
    -/
  · exact shortExact_of_iso e.symm
    /-
      🎉 no goals
    -/


lemma ShortExact.op (h : S.ShortExact) : S.op.ShortExact where
  exact := h.exact.op
  mono_f := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.ShortExact
      ⊢ CategoryTheory.Mono S.op.f
    -/
    have := h.epi_g
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.ShortExact
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.Mono S.op.f
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.ShortExact
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.Mono S.g.op
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  epi_g := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.ShortExact
      ⊢ CategoryTheory.Epi S.op.g
    -/
    have := h.mono_f
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.ShortExact
      this : CategoryTheory.Mono S.f
      ⊢ CategoryTheory.Epi S.op.g
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.ShortExact
      this : CategoryTheory.Mono S.f
      ⊢ CategoryTheory.Epi S.f.op
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma ShortExact.unop {S : ShortComplex Cᵒᵖ} (h : S.ShortExact) : S.unop.ShortExact where
  exact := h.exact.unop
  mono_f := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex (Opposite C)
      h : S.ShortExact
      ⊢ CategoryTheory.Mono S.unop.f
    -/
    have := h.epi_g
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex (Opposite C)
      h : S.ShortExact
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.Mono S.unop.f
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex (Opposite C)
      h : S.ShortExact
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.Mono S.g.unop
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  epi_g := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex (Opposite C)
      h : S.ShortExact
      ⊢ CategoryTheory.Epi S.unop.g
    -/
    have := h.mono_f
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex (Opposite C)
      h : S.ShortExact
      this : CategoryTheory.Mono S.f
      ⊢ CategoryTheory.Epi S.unop.g
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex (Opposite C)
      h : S.ShortExact
      this : CategoryTheory.Mono S.f
      ⊢ CategoryTheory.Epi S.f.unop
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma shortExact_iff_op : S.ShortExact ↔ S.op.ShortExact :=
  ⟨ShortExact.op, ShortExact.unop⟩


lemma shortExact_iff_unop (S : ShortComplex Cᵒᵖ) : S.ShortExact ↔ S.unop.ShortExact :=
  S.unop.shortExact_iff_op.symm


lemma ShortExact.map (h : S.ShortExact) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [F.PreservesLeftHomologyOf S]
    [F.PreservesRightHomologyOf S] [Mono (F.map S.f)] [Epi (F.map S.g)] :
    (S.map F).ShortExact where
  exact := h.exact.map F
  mono_f := (inferInstance : Mono (F.map S.f))
  epi_g := (inferInstance : Epi (F.map S.g))


lemma ShortExact.map_of_exact (hS : S.ShortExact)
    (F : C ⥤ D) [F.PreservesZeroMorphisms] [PreservesFiniteLimits F]
    [PreservesFiniteColimits F] : (S.map F).ShortExact := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits F
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    ⊢ (S.map F).ShortExact
  -/
  have := hS.mono_f
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits F
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    this : CategoryTheory.Mono S.f
    ⊢ (S.map F).ShortExact
  -/
  have := hS.epi_g
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits F
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    this✝ : CategoryTheory.Mono S.f
    this : CategoryTheory.Epi S.g
    ⊢ (S.map F).ShortExact
  -/
  have := preserves_mono_of_preservesLimit F S.f
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits F
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    this✝¹ : CategoryTheory.Mono S.f
    this✝ : CategoryTheory.Epi S.g
    this : CategoryTheory.Mono (F.map S.f)
    ⊢ (S.map F).ShortExact
  -/
  have := preserves_epi_of_preservesColimit F S.g
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.PreservesFiniteLimits F
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits F
    this✝² : CategoryTheory.Mono S.f
    this✝¹ : CategoryTheory.Epi S.g
    this✝ : CategoryTheory.Mono (F.map S.f)
    this : CategoryTheory.Epi (F.map S.g)
    ⊢ (S.map F).ShortExact
  -/
  exact hS.map F
  /-
    🎉 no goals
  -/


lemma ShortExact.isIso_f_iff {S : ShortComplex C} (hS : S.ShortExact) [Balanced C] :
    IsIso S.f ↔ IsZero S.X₃ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : CategoryTheory.Balanced C
    ⊢ Iff (CategoryTheory.IsIso S.f) (CategoryTheory.Limits.IsZero S.X₃)
  -/
  have := hS.exact.hasZeroObject
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : CategoryTheory.Balanced C
    this : CategoryTheory.Limits.HasZeroObject C
    ⊢ Iff (CategoryTheory.IsIso S.f) (CategoryTheory.Limits.IsZero S.X₃)
  -/
  have := hS.mono_f
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : CategoryTheory.Balanced C
    this✝ : CategoryTheory.Limits.HasZeroObject C
    this : CategoryTheory.Mono S.f
    ⊢ Iff (CategoryTheory.IsIso S.f) (CategoryTheory.Limits.IsZero S.X₃)
  -/
  have := hS.epi_g
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : CategoryTheory.Balanced C
    this✝¹ : CategoryTheory.Limits.HasZeroObject C
    this✝ : CategoryTheory.Mono S.f
    this : CategoryTheory.Epi S.g
    ⊢ Iff (CategoryTheory.IsIso S.f) (CategoryTheory.Limits.IsZero S.X₃)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      inst✝ : CategoryTheory.Balanced C
      this✝¹ : CategoryTheory.Limits.HasZeroObject C
      this✝ : CategoryTheory.Mono S.f
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.IsIso S.f → CategoryTheory.Limits.IsZero S.X₃
    -/
  · intro hf
    simp only [IsZero.iff_id_eq_zero, ← cancel_epi S.g, ← cancel_epi S.f,
      S.zero_assoc, zero_comp]
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      inst✝ : CategoryTheory.Balanced C
      this✝¹ : CategoryTheory.Limits.HasZeroObject C
      this✝ : CategoryTheory.Mono S.f
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.Limits.IsZero S.X₃ → CategoryTheory.IsIso S.f
    -/
  · intro hX₃
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      inst✝ : CategoryTheory.Balanced C
      this✝¹ : CategoryTheory.Limits.HasZeroObject C
      this✝ : CategoryTheory.Mono S.f
      this : CategoryTheory.Epi S.g
      hX₃ : CategoryTheory.Limits.IsZero S.X₃
      ⊢ CategoryTheory.IsIso S.f
    -/
    have : Epi S.f := (S.exact_iff_epi (hX₃.eq_of_tgt _ _)).1 hS.exact
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      inst✝ : CategoryTheory.Balanced C
      this✝² : CategoryTheory.Limits.HasZeroObject C
      this✝¹ : CategoryTheory.Mono S.f
      this✝ : CategoryTheory.Epi S.g
      hX₃ : CategoryTheory.Limits.IsZero S.X₃
      this : CategoryTheory.Epi S.f
      ⊢ CategoryTheory.IsIso S.f
    -/
    apply isIso_of_mono_of_epi
    /-
      🎉 no goals
    -/


lemma ShortExact.isIso_g_iff  {S : ShortComplex C} (hS : S.ShortExact) [Balanced C] :
    IsIso S.g ↔ IsZero S.X₁ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : CategoryTheory.Balanced C
    ⊢ Iff (CategoryTheory.IsIso S.g) (CategoryTheory.Limits.IsZero S.X₁)
  -/
  have := hS.exact.hasZeroObject
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : CategoryTheory.Balanced C
    this : CategoryTheory.Limits.HasZeroObject C
    ⊢ Iff (CategoryTheory.IsIso S.g) (CategoryTheory.Limits.IsZero S.X₁)
  -/
  have := hS.mono_f
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : CategoryTheory.Balanced C
    this✝ : CategoryTheory.Limits.HasZeroObject C
    this : CategoryTheory.Mono S.f
    ⊢ Iff (CategoryTheory.IsIso S.g) (CategoryTheory.Limits.IsZero S.X₁)
  -/
  have := hS.epi_g
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : CategoryTheory.Balanced C
    this✝¹ : CategoryTheory.Limits.HasZeroObject C
    this✝ : CategoryTheory.Mono S.f
    this : CategoryTheory.Epi S.g
    ⊢ Iff (CategoryTheory.IsIso S.g) (CategoryTheory.Limits.IsZero S.X₁)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      inst✝ : CategoryTheory.Balanced C
      this✝¹ : CategoryTheory.Limits.HasZeroObject C
      this✝ : CategoryTheory.Mono S.f
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.IsIso S.g → CategoryTheory.Limits.IsZero S.X₁
    -/
  · intro hf
    simp only [IsZero.iff_id_eq_zero, ← cancel_mono S.f, ← cancel_mono S.g,
      S.zero, zero_comp, assoc, comp_zero]
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      inst✝ : CategoryTheory.Balanced C
      this✝¹ : CategoryTheory.Limits.HasZeroObject C
      this✝ : CategoryTheory.Mono S.f
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.Limits.IsZero S.X₁ → CategoryTheory.IsIso S.g
    -/
  · intro hX₁
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      inst✝ : CategoryTheory.Balanced C
      this✝¹ : CategoryTheory.Limits.HasZeroObject C
      this✝ : CategoryTheory.Mono S.f
      this : CategoryTheory.Epi S.g
      hX₁ : CategoryTheory.Limits.IsZero S.X₁
      ⊢ CategoryTheory.IsIso S.g
    -/
    have : Mono S.g := (S.exact_iff_mono (hX₁.eq_of_src _ _)).1 hS.exact
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      inst✝ : CategoryTheory.Balanced C
      this✝² : CategoryTheory.Limits.HasZeroObject C
      this✝¹ : CategoryTheory.Mono S.f
      this✝ : CategoryTheory.Epi S.g
      hX₁ : CategoryTheory.Limits.IsZero S.X₁
      this : CategoryTheory.Mono S.g
      ⊢ CategoryTheory.IsIso S.g
    -/
    apply isIso_of_mono_of_epi
    /-
      🎉 no goals
    -/


lemma isIso₂_of_shortExact_of_isIso₁₃ [Balanced C] {S₁ S₂ : ShortComplex C} (φ : S₁ ⟶ S₂)
    (h₁ : S₁.ShortExact) (h₂ : S₂.ShortExact) [IsIso φ.τ₁] [IsIso φ.τ₃] : IsIso φ.τ₂ := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.ShortExact
    h₂ : S₂.ShortExact
    inst✝¹ : CategoryTheory.IsIso φ.τ₁
    inst✝ : CategoryTheory.IsIso φ.τ₃
    ⊢ CategoryTheory.IsIso φ.τ₂
  -/
  have := h₁.mono_f
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.ShortExact
    h₂ : S₂.ShortExact
    inst✝¹ : CategoryTheory.IsIso φ.τ₁
    inst✝ : CategoryTheory.IsIso φ.τ₃
    this : CategoryTheory.Mono S₁.f
    ⊢ CategoryTheory.IsIso φ.τ₂
  -/
  have := h₂.mono_f
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.ShortExact
    h₂ : S₂.ShortExact
    inst✝¹ : CategoryTheory.IsIso φ.τ₁
    inst✝ : CategoryTheory.IsIso φ.τ₃
    this✝ : CategoryTheory.Mono S₁.f
    this : CategoryTheory.Mono S₂.f
    ⊢ CategoryTheory.IsIso φ.τ₂
  -/
  have := h₁.epi_g
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.ShortExact
    h₂ : S₂.ShortExact
    inst✝¹ : CategoryTheory.IsIso φ.τ₁
    inst✝ : CategoryTheory.IsIso φ.τ₃
    this✝¹ : CategoryTheory.Mono S₁.f
    this✝ : CategoryTheory.Mono S₂.f
    this : CategoryTheory.Epi S₁.g
    ⊢ CategoryTheory.IsIso φ.τ₂
  -/
  have := h₂.epi_g
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.ShortExact
    h₂ : S₂.ShortExact
    inst✝¹ : CategoryTheory.IsIso φ.τ₁
    inst✝ : CategoryTheory.IsIso φ.τ₃
    this✝² : CategoryTheory.Mono S₁.f
    this✝¹ : CategoryTheory.Mono S₂.f
    this✝ : CategoryTheory.Epi S₁.g
    this : CategoryTheory.Epi S₂.g
    ⊢ CategoryTheory.IsIso φ.τ₂
  -/
  have := mono_τ₂_of_exact_of_mono φ h₁.exact
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.ShortExact
    h₂ : S₂.ShortExact
    inst✝¹ : CategoryTheory.IsIso φ.τ₁
    inst✝ : CategoryTheory.IsIso φ.τ₃
    this✝³ : CategoryTheory.Mono S₁.f
    this✝² : CategoryTheory.Mono S₂.f
    this✝¹ : CategoryTheory.Epi S₁.g
    this✝ : CategoryTheory.Epi S₂.g
    this : CategoryTheory.Mono φ.τ₂
    ⊢ CategoryTheory.IsIso φ.τ₂
  -/
  have := epi_τ₂_of_exact_of_epi φ h₂.exact
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.ShortExact
    h₂ : S₂.ShortExact
    inst✝¹ : CategoryTheory.IsIso φ.τ₁
    inst✝ : CategoryTheory.IsIso φ.τ₃
    this✝⁴ : CategoryTheory.Mono S₁.f
    this✝³ : CategoryTheory.Mono S₂.f
    this✝² : CategoryTheory.Epi S₁.g
    this✝¹ : CategoryTheory.Epi S₂.g
    this✝ : CategoryTheory.Mono φ.τ₂
    this : CategoryTheory.Epi φ.τ₂
    ⊢ CategoryTheory.IsIso φ.τ₂
  -/
  apply isIso_of_mono_of_epi
  /-
    🎉 no goals
  -/


lemma isIso₂_of_shortExact_of_isIso₁₃' [Balanced C] {S₁ S₂ : ShortComplex C} (φ : S₁ ⟶ S₂)
    (h₁ : S₁.ShortExact) (h₂ : S₂.ShortExact) (_ : IsIso φ.τ₁) (_ : IsIso φ.τ₃) : IsIso φ.τ₂ :=
  isIso₂_of_shortExact_of_isIso₁₃ φ h₁ h₂


/-- If `S` is a short exact short complex in a balanced category,
then `S.X₁` is the kernel of `S.g`. -/
noncomputable def ShortExact.fIsKernel [Balanced C] {S : ShortComplex C} (hS : S.ShortExact) :
    IsLimit (KernelFork.ofι S.f S.zero) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.15921, u_1} C
    inst✝² : CategoryTheory.Category.{?u.15925, u_2} D
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Balanced C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
  -/
  have := hS.mono_f
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.15921, u_1} C
    inst✝² : CategoryTheory.Category.{?u.15925, u_2} D
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Balanced C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this : CategoryTheory.Mono S.f
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
  -/
  exact hS.exact.fIsKernel
  /-
    🎉 no goals
  -/


/-- If `S` is a short exact short complex in a balanced category,
then `S.X₃` is the cokernel of `S.f`. -/
noncomputable def ShortExact.gIsCokernel [Balanced C] {S : ShortComplex C} (hS : S.ShortExact) :
    IsColimit (CokernelCofork.ofπ S.g S.zero) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.16693, u_1} C
    inst✝² : CategoryTheory.Category.{?u.16697, u_2} D
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Balanced C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ S. …
  -/
  have := hS.epi_g
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.16693, u_1} C
    inst✝² : CategoryTheory.Category.{?u.16697, u_2} D
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Balanced C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    this : CategoryTheory.Epi S.g
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ S. …
  -/
  exact hS.exact.gIsCokernel
  /-
    🎉 no goals
  -/


/-- A split short complex is short exact. -/
lemma Splitting.shortExact {S : ShortComplex C} [HasZeroObject C] (s : S.Splitting) :
    S.ShortExact where
  exact := s.exact
  mono_f := s.mono_f
  epi_g := s.epi_g


/-- A choice of splitting for a short exact short complex `S` in a balanced category
such that `S.X₁` is injective. -/
noncomputable def splittingOfInjective {S : ShortComplex C} (hS : S.ShortExact)
    [Injective S.X₁] [Balanced C] :
    S.Splitting :=
  have := hS.mono_f
                                                                                   /-
                                                                                     C : Type u_1
                                                                                     D : Type u_2
                                                                                     inst✝⁴ : CategoryTheory.Category.{?u.17736, u_1} C
                                                                                     inst✝³ : CategoryTheory.Category.{?u.17740, u_2} D
                                                                                     inst✝² : CategoryTheory.Preadditive C
                                                                                     S : CategoryTheory.ShortComplex C
                                                                                     hS : S.ShortExact
                                                                                     inst✝¹ : CategoryTheory.Injective S.X₁
                                                                                     inst✝ : CategoryTheory.Balanced C
                                                                                     this : CategoryTheory.Mono S.f
                                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f (CategoryTheory.Injective.factorT …
                                                                                   -/
  Splitting.ofExactOfRetraction S hS.exact (Injective.factorThru (𝟙 S.X₁) S.f) (by simp) hS.epi_g
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- A choice of splitting for a short exact short complex `S` in a balanced category
such that `S.X₃` is projective. -/
noncomputable def splittingOfProjective {S : ShortComplex C} (hS : S.ShortExact)
    [Projective S.X₃] [Balanced C] :
    S.Splitting :=
  have := hS.epi_g
                                                                                 /-
                                                                                   C : Type u_1
                                                                                   D : Type u_2
                                                                                   inst✝⁴ : CategoryTheory.Category.{?u.18578, u_1} C
                                                                                   inst✝³ : CategoryTheory.Category.{?u.18582, u_2} D
                                                                                   inst✝² : CategoryTheory.Preadditive C
                                                                                   S : CategoryTheory.ShortComplex C
                                                                                   hS : S.ShortExact
                                                                                   inst✝¹ : CategoryTheory.Projective S.X₃
                                                                                   inst✝ : CategoryTheory.Balanced C
                                                                                   this : CategoryTheory.Epi S.g
                                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Projective.factorThru …
                                                                                 -/
  Splitting.ofExactOfSection S hS.exact (Projective.factorThru (𝟙 S.X₃) S.g) (by simp) hS.mono_f
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


