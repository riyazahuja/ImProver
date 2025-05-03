/-- The assertion that the short complex `S : ShortComplex C` is exact. -/
structure Exact : Prop where
  /-- the condition that there exists an homology data whose `left.H` field is zero -/
  condition : ∃ (h : S.HomologyData), IsZero h.left.H


lemma Exact.hasHomology (h : S.Exact) : S.HasHomology :=
  HasHomology.mk' h.condition.choose


lemma Exact.hasZeroObject (h : S.Exact) : HasZeroObject C :=
  ⟨h.condition.choose.left.H, h.condition.choose_spec⟩


lemma exact_iff_isZero_homology [S.HasHomology] :
    S.Exact ↔ IsZero S.homology := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    ⊢ Iff S.Exact (CategoryTheory.Limits.IsZero S.homology)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      ⊢ S.Exact → CategoryTheory.Limits.IsZero S.homology
    -/
  · rintro ⟨⟨h', z⟩⟩
    /-
      case mp.mk.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      h' : S.HomologyData
      z : CategoryTheory.Limits.IsZero h'.left.H
      ⊢ CategoryTheory.Limits.IsZero S.homology
    -/
    exact IsZero.of_iso z h'.left.homologyIso
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      ⊢ CategoryTheory.Limits.IsZero S.homology → S.Exact
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      ⊢ S.Exact
    -/
    exact ⟨⟨_, h⟩⟩
    /-
      🎉 no goals
    -/


lemma LeftHomologyData.exact_iff [S.HasHomology]
    (h : S.LeftHomologyData) :
    S.Exact ↔ IsZero h.H := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Iff S.Exact (CategoryTheory.Limits.IsZero h.H)
  -/
  rw [S.exact_iff_isZero_homology]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Iff (CategoryTheory.Limits.IsZero S.homology) (CategoryTheory.Limits.IsZero  …
  -/
  exact Iso.isZero_iff h.homologyIso
  /-
    🎉 no goals
  -/


lemma RightHomologyData.exact_iff [S.HasHomology]
    (h : S.RightHomologyData) :
    S.Exact ↔ IsZero h.H := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Iff S.Exact (CategoryTheory.Limits.IsZero h.H)
  -/
  rw [S.exact_iff_isZero_homology]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Iff (CategoryTheory.Limits.IsZero S.homology) (CategoryTheory.Limits.IsZero  …
  -/
  exact Iso.isZero_iff h.homologyIso
  /-
    🎉 no goals
  -/


lemma exact_iff_isZero_leftHomology [S.HasHomology] :
    S.Exact ↔ IsZero S.leftHomology :=
  LeftHomologyData.exact_iff _


lemma exact_iff_isZero_rightHomology [S.HasHomology] :
    S.Exact ↔ IsZero S.rightHomology :=
  RightHomologyData.exact_iff _


lemma HomologyData.exact_iff (h : S.HomologyData) :
    S.Exact ↔ IsZero h.left.H := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    ⊢ Iff S.Exact (CategoryTheory.Limits.IsZero h.left.H)
  -/
  haveI := HasHomology.mk' h
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    this : S.HasHomology
    ⊢ Iff S.Exact (CategoryTheory.Limits.IsZero h.left.H)
  -/
  exact LeftHomologyData.exact_iff h.left
  /-
    🎉 no goals
  -/


lemma HomologyData.exact_iff' (h : S.HomologyData) :
    S.Exact ↔ IsZero h.right.H := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    ⊢ Iff S.Exact (CategoryTheory.Limits.IsZero h.right.H)
  -/
  haveI := HasHomology.mk' h
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    this : S.HasHomology
    ⊢ Iff S.Exact (CategoryTheory.Limits.IsZero h.right.H)
  -/
  exact RightHomologyData.exact_iff h.right
  /-
    🎉 no goals
  -/


lemma exact_iff_homology_iso_zero [S.HasHomology] [HasZeroObject C] :
    S.Exact ↔ Nonempty (S.homology ≅ 0) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Iff S.Exact (Nonempty (CategoryTheory.Iso S.homology 0))
  -/
  rw [exact_iff_isZero_homology]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasHomology
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Iff (CategoryTheory.Limits.IsZero S.homology) (Nonempty (CategoryTheory.Iso  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : S.HasHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      ⊢ CategoryTheory.Limits.IsZero S.homology → Nonempty (CategoryTheory.Iso S.hom …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : S.HasHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      h : CategoryTheory.Limits.IsZero S.homology
      ⊢ Nonempty (CategoryTheory.Iso S.homology 0)
    -/
    exact ⟨h.isoZero⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : S.HasHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      ⊢ Nonempty (CategoryTheory.Iso S.homology 0) → CategoryTheory.Limits.IsZero S. …
    -/
  · rintro ⟨e⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : S.HasHomology
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      e : CategoryTheory.Iso S.homology 0
      ⊢ CategoryTheory.Limits.IsZero S.homology
    -/
    exact IsZero.of_iso (isZero_zero C) e
    /-
      🎉 no goals
    -/


lemma exact_of_iso (e : S₁ ≅ S₂) (h : S₁.Exact) : S₂.Exact := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    e : CategoryTheory.Iso S₁ S₂
    h : S₁.Exact
    ⊢ S₂.Exact
  -/
  obtain ⟨⟨h, z⟩⟩ := h
  /-
    case mk.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    e : CategoryTheory.Iso S₁ S₂
    h : S₁.HomologyData
    z : CategoryTheory.Limits.IsZero h.left.H
    ⊢ S₂.Exact
  -/
  exact ⟨⟨HomologyData.ofIso e h, z⟩⟩
  /-
    🎉 no goals
  -/


lemma exact_iff_of_iso (e : S₁ ≅ S₂) : S₁.Exact ↔ S₂.Exact :=
  ⟨exact_of_iso e, exact_of_iso e.symm⟩


lemma exact_and_mono_f_iff_of_iso (e : S₁ ≅ S₂) :
    S₁.Exact ∧ Mono S₁.f ↔ S₂.Exact ∧ Mono S₂.f := by
  have : Mono S₁.f ↔ Mono S₂.f :=
    (MorphismProperty.monomorphisms C).arrow_mk_iso_iff
      (Arrow.isoMk (ShortComplex.π₁.mapIso e) (ShortComplex.π₂.mapIso e) e.hom.comm₁₂)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    e : CategoryTheory.Iso S₁ S₂
    this : Iff (CategoryTheory.Mono S₁.f) (CategoryTheory.Mono S₂.f)
    ⊢ Iff (And S₁.Exact (CategoryTheory.Mono S₁.f)) (And S₂.Exact (CategoryTheory. …
  -/
  rw [exact_iff_of_iso e, this]
  /-
    🎉 no goals
  -/


lemma exact_and_epi_g_iff_of_iso (e : S₁ ≅ S₂) :
    S₁.Exact ∧ Epi S₁.g ↔ S₂.Exact ∧ Epi S₂.g := by
  have : Epi S₁.g ↔ Epi S₂.g :=
    (MorphismProperty.epimorphisms C).arrow_mk_iso_iff
      (Arrow.isoMk (ShortComplex.π₂.mapIso e) (ShortComplex.π₃.mapIso e) e.hom.comm₂₃)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    e : CategoryTheory.Iso S₁ S₂
    this : Iff (CategoryTheory.Epi S₁.g) (CategoryTheory.Epi S₂.g)
    ⊢ Iff (And S₁.Exact (CategoryTheory.Epi S₁.g)) (And S₂.Exact (CategoryTheory.E …
  -/
  rw [exact_iff_of_iso e, this]
  /-
    🎉 no goals
  -/


lemma exact_of_isZero_X₂ (h : IsZero S.X₂) : S.Exact := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : CategoryTheory.Limits.IsZero S.X₂
    ⊢ S.Exact
  -/
  rw [(HomologyData.ofZeros S (IsZero.eq_of_tgt h _ _) (IsZero.eq_of_src h _ _)).exact_iff]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : CategoryTheory.Limits.IsZero S.X₂
    ⊢ CategoryTheory.Limits.IsZero (CategoryTheory.ShortComplex.HomologyData.ofZer …
  -/
  exact h
  /-
    🎉 no goals
  -/


lemma exact_iff_of_epi_of_isIso_of_mono (φ : S₁ ⟶ S₂) [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    S₁.Exact ↔ S₂.Exact := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ Iff S₁.Exact S₂.Exact
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝² : CategoryTheory.Epi φ.τ₁
      inst✝¹ : CategoryTheory.IsIso φ.τ₂
      inst✝ : CategoryTheory.Mono φ.τ₃
      ⊢ S₁.Exact → S₂.Exact
    -/
  · rintro ⟨h₁, z₁⟩
    /-
      case mp.mk.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝² : CategoryTheory.Epi φ.τ₁
      inst✝¹ : CategoryTheory.IsIso φ.τ₂
      inst✝ : CategoryTheory.Mono φ.τ₃
      h₁ : S₁.HomologyData
      z₁ : CategoryTheory.Limits.IsZero h₁.left.H
      ⊢ S₂.Exact
    -/
    exact ⟨HomologyData.ofEpiOfIsIsoOfMono φ h₁, z₁⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝² : CategoryTheory.Epi φ.τ₁
      inst✝¹ : CategoryTheory.IsIso φ.τ₂
      inst✝ : CategoryTheory.Mono φ.τ₃
      ⊢ S₂.Exact → S₁.Exact
    -/
  · rintro ⟨h₂, z₂⟩
    /-
      case mpr.mk.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      inst✝² : CategoryTheory.Epi φ.τ₁
      inst✝¹ : CategoryTheory.IsIso φ.τ₂
      inst✝ : CategoryTheory.Mono φ.τ₃
      h₂ : S₂.HomologyData
      z₂ : CategoryTheory.Limits.IsZero h₂.left.H
      ⊢ S₁.Exact
    -/
    exact ⟨HomologyData.ofEpiOfIsIsoOfMono' φ h₂, z₂⟩
    /-
      🎉 no goals
    -/


lemma HomologyData.exact_iff_i_p_zero (h : S.HomologyData) :
    S.Exact ↔ h.left.i ≫ h.right.p = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    ⊢ Iff S.Exact (Eq (CategoryTheory.CategoryStruct.comp h.left.i h.right.p) 0)
  -/
  haveI := HasHomology.mk' h
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    this : S.HasHomology
    ⊢ Iff S.Exact (Eq (CategoryTheory.CategoryStruct.comp h.left.i h.right.p) 0)
  -/
  rw [h.left.exact_iff, ← h.comm]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    this : S.HasHomology
    ⊢ Iff (CategoryTheory.Limits.IsZero h.left.H) (Eq (CategoryTheory.CategoryStru …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.HomologyData
      this : S.HasHomology
      ⊢ CategoryTheory.Limits.IsZero h.left.H → Eq (CategoryTheory.CategoryStruct.co …
    -/
  · intro z
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.HomologyData
      this : S.HasHomology
      z : CategoryTheory.Limits.IsZero h.left.H
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h.left.π (CategoryTheory.CategoryStru …
    -/
    rw [IsZero.eq_of_src z h.iso.hom 0, zero_comp, comp_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.HomologyData
      this : S.HasHomology
      ⊢ Eq (CategoryTheory.CategoryStruct.comp h.left.π (CategoryTheory.CategoryStru …
    -/
  · intro eq
    simp only [IsZero.iff_id_eq_zero, ← cancel_mono h.iso.hom, id_comp, ← cancel_mono h.right.ι,
      ← cancel_epi h.left.π, eq, zero_comp, comp_zero]


lemma exact_iff_i_p_zero [S.HasHomology] (h₁ : S.LeftHomologyData)
    (h₂ : S.RightHomologyData) :
    S.Exact ↔ h₁.i ≫ h₂.p = 0 :=
  (HomologyData.ofIsIsoLeftRightHomologyComparison' h₁ h₂).exact_iff_i_p_zero


lemma exact_iff_iCycles_pOpcycles_zero [S.HasHomology] :
    S.Exact ↔ S.iCycles ≫ S.pOpcycles = 0 :=
  S.exact_iff_i_p_zero _ _


lemma exact_iff_kernel_ι_comp_cokernel_π_zero [S.HasHomology]
    [HasKernel S.g] [HasCokernel S.f] :
    S.Exact ↔ kernel.ι S.g ≫ cokernel.π S.f = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝² : S.HasHomology
    inst✝¹ : CategoryTheory.Limits.HasKernel S.g
    inst✝ : CategoryTheory.Limits.HasCokernel S.f
    ⊢ Iff S.Exact (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.k …
  -/
  haveI := HasLeftHomology.hasCokernel S
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝² : S.HasHomology
    inst✝¹ : CategoryTheory.Limits.HasKernel S.g
    inst✝ : CategoryTheory.Limits.HasCokernel S.f
    this : CategoryTheory.Limits.HasCokernel (CategoryTheory.Limits.kernel.lift S. …
    ⊢ Iff S.Exact (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.k …
  -/
  haveI := HasRightHomology.hasKernel S
  exact S.exact_iff_i_p_zero (LeftHomologyData.ofHasKernelOfHasCokernel S)
    (RightHomologyData.ofHasCokernelOfHasKernel S)


lemma Exact.op (h : S.Exact) : S.op.Exact := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    ⊢ S.op.Exact
  -/
  obtain ⟨h, z⟩ := h
  /-
    case mk.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.HomologyData
    z : CategoryTheory.Limits.IsZero h.left.H
    ⊢ S.op.Exact
  -/
  exact ⟨⟨h.op, (IsZero.of_iso z h.iso.symm).op⟩⟩
  /-
    🎉 no goals
  -/


lemma Exact.unop {S : ShortComplex Cᵒᵖ} (h : S.Exact) : S.unop.Exact := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex (Opposite C)
    h : S.Exact
    ⊢ S.unop.Exact
  -/
  obtain ⟨h, z⟩ := h
  /-
    case mk.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex (Opposite C)
    h : S.HomologyData
    z : CategoryTheory.Limits.IsZero h.left.H
    ⊢ S.unop.Exact
  -/
  exact ⟨⟨h.unop, (IsZero.of_iso z h.iso.symm).unop⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma exact_op_iff : S.op.Exact ↔ S.Exact :=
  ⟨Exact.unop, Exact.op⟩


@[simp]
lemma exact_unop_iff (S : ShortComplex Cᵒᵖ) : S.unop.Exact ↔ S.Exact :=
  S.unop.exact_op_iff.symm


lemma LeftHomologyData.exact_map_iff (h : S.LeftHomologyData) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [h.IsPreservedBy F] [(S.map F).HasHomology] :
    (S.map F).Exact ↔ IsZero (F.obj h.H) :=
  (h.map F).exact_iff


lemma RightHomologyData.exact_map_iff (h : S.RightHomologyData) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [h.IsPreservedBy F] [(S.map F).HasHomology] :
    (S.map F).Exact ↔ IsZero (F.obj h.H) :=
  (h.map F).exact_iff


lemma Exact.map_of_preservesLeftHomologyOf (h : S.Exact) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [F.PreservesLeftHomologyOf S]
    [(S.map F).HasHomology] : (S.map F).Exact := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : (S.map F).HasHomology
    ⊢ (S.map F).Exact
  -/
  have := h.hasHomology
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : (S.map F).HasHomology
    this : S.HasHomology
    ⊢ (S.map F).Exact
  -/
  rw [S.leftHomologyData.exact_iff, IsZero.iff_id_eq_zero] at h
  rw [S.leftHomologyData.exact_map_iff F, IsZero.iff_id_eq_zero,
    ← F.map_id, h, F.map_zero]


lemma Exact.map_of_preservesRightHomologyOf (h : S.Exact) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [F.PreservesRightHomologyOf S]
    [(S.map F).HasHomology] : (S.map F).Exact := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : F.PreservesRightHomologyOf S
    inst✝ : (S.map F).HasHomology
    ⊢ (S.map F).Exact
  -/
  have : S.HasHomology := h.hasHomology
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : F.PreservesRightHomologyOf S
    inst✝ : (S.map F).HasHomology
    this : S.HasHomology
    ⊢ (S.map F).Exact
  -/
  rw [S.rightHomologyData.exact_iff, IsZero.iff_id_eq_zero] at h
  rw [S.rightHomologyData.exact_map_iff F, IsZero.iff_id_eq_zero,
    ← F.map_id, h, F.map_zero]


lemma Exact.map (h : S.Exact) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [F.PreservesLeftHomologyOf S]
    [F.PreservesRightHomologyOf S] : (S.map F).Exact := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    ⊢ (S.map F).Exact
  -/
  have := h.hasHomology
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : F.PreservesLeftHomologyOf S
    inst✝ : F.PreservesRightHomologyOf S
    this : S.HasHomology
    ⊢ (S.map F).Exact
  -/
  exact h.map_of_preservesLeftHomologyOf F
  /-
    🎉 no goals
  -/


lemma exact_map_iff_of_faithful [S.HasHomology]
    (F : C ⥤ D) [F.PreservesZeroMorphisms] [F.PreservesLeftHomologyOf S]
    [F.PreservesRightHomologyOf S] [F.Faithful] :
    (S.map F).Exact ↔ S.Exact := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
    S : CategoryTheory.ShortComplex C
    inst✝⁴ : S.HasHomology
    F : CategoryTheory.Functor C D
    inst✝³ : F.PreservesZeroMorphisms
    inst✝² : F.PreservesLeftHomologyOf S
    inst✝¹ : F.PreservesRightHomologyOf S
    inst✝ : F.Faithful
    ⊢ Iff (S.map F).Exact S.Exact
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
      S : CategoryTheory.ShortComplex C
      inst✝⁴ : S.HasHomology
      F : CategoryTheory.Functor C D
      inst✝³ : F.PreservesZeroMorphisms
      inst✝² : F.PreservesLeftHomologyOf S
      inst✝¹ : F.PreservesRightHomologyOf S
      inst✝ : F.Faithful
      ⊢ (S.map F).Exact → S.Exact
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
      S : CategoryTheory.ShortComplex C
      inst✝⁴ : S.HasHomology
      F : CategoryTheory.Functor C D
      inst✝³ : F.PreservesZeroMorphisms
      inst✝² : F.PreservesLeftHomologyOf S
      inst✝¹ : F.PreservesRightHomologyOf S
      inst✝ : F.Faithful
      h : (S.map F).Exact
      ⊢ S.Exact
    -/
    rw [S.leftHomologyData.exact_iff, IsZero.iff_id_eq_zero]
    rw [(S.leftHomologyData.map F).exact_iff, IsZero.iff_id_eq_zero,
      LeftHomologyData.map_H] at h
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
      S : CategoryTheory.ShortComplex C
      inst✝⁴ : S.HasHomology
      F : CategoryTheory.Functor C D
      inst✝³ : F.PreservesZeroMorphisms
      inst✝² : F.PreservesLeftHomologyOf S
      inst✝¹ : F.PreservesRightHomologyOf S
      inst✝ : F.Faithful
      h : Eq (CategoryTheory.CategoryStruct.id (F.obj S.leftHomologyData.H)) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.id S.leftHomologyData.H) 0
    -/
    apply F.map_injective
    /-
      case mp.a
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
      S : CategoryTheory.ShortComplex C
      inst✝⁴ : S.HasHomology
      F : CategoryTheory.Functor C D
      inst✝³ : F.PreservesZeroMorphisms
      inst✝² : F.PreservesLeftHomologyOf S
      inst✝¹ : F.PreservesRightHomologyOf S
      inst✝ : F.Faithful
      h : Eq (CategoryTheory.CategoryStruct.id (F.obj S.leftHomologyData.H)) 0
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id S.leftHomologyData.H)) (F.map 0)
    -/
    rw [F.map_id, F.map_zero, h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
      S : CategoryTheory.ShortComplex C
      inst✝⁴ : S.HasHomology
      F : CategoryTheory.Functor C D
      inst✝³ : F.PreservesZeroMorphisms
      inst✝² : F.PreservesLeftHomologyOf S
      inst✝¹ : F.PreservesRightHomologyOf S
      inst✝ : F.Faithful
      ⊢ S.Exact → (S.map F).Exact
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
      inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
      S : CategoryTheory.ShortComplex C
      inst✝⁴ : S.HasHomology
      F : CategoryTheory.Functor C D
      inst✝³ : F.PreservesZeroMorphisms
      inst✝² : F.PreservesLeftHomologyOf S
      inst✝¹ : F.PreservesRightHomologyOf S
      inst✝ : F.Faithful
      h : S.Exact
      ⊢ (S.map F).Exact
    -/
    exact h.map F
    /-
      🎉 no goals
    -/


@[reassoc]
lemma Exact.comp_eq_zero (h : S.Exact) {X Y : C} {a : X ⟶ S.X₂} (ha : a ≫ S.g = 0)
    {b : S.X₂ ⟶ Y} (hb : S.f ≫ b = 0) : a ≫ b = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    X Y : C
    a : Quiver.Hom X S.X₂
    ha : Eq (CategoryTheory.CategoryStruct.comp a S.g) 0
    b : Quiver.Hom S.X₂ Y
    hb : Eq (CategoryTheory.CategoryStruct.comp S.f b) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp a b) 0
  -/
  have := h.hasHomology
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    X Y : C
    a : Quiver.Hom X S.X₂
    ha : Eq (CategoryTheory.CategoryStruct.comp a S.g) 0
    b : Quiver.Hom S.X₂ Y
    hb : Eq (CategoryTheory.CategoryStruct.comp S.f b) 0
    this : S.HasHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp a b) 0
  -/
  have eq := h
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    X Y : C
    a : Quiver.Hom X S.X₂
    ha : Eq (CategoryTheory.CategoryStruct.comp a S.g) 0
    b : Quiver.Hom S.X₂ Y
    hb : Eq (CategoryTheory.CategoryStruct.comp S.f b) 0
    this : S.HasHomology
    eq : S.Exact
    ⊢ Eq (CategoryTheory.CategoryStruct.comp a b) 0
  -/
  rw [exact_iff_iCycles_pOpcycles_zero] at eq
  rw [← S.liftCycles_i a ha, ← S.p_descOpcycles b hb, assoc, reassoc_of% eq,
    zero_comp, comp_zero]


lemma Exact.isZero_of_both_zeros (ex : S.Exact) (hf : S.f = 0) (hg : S.g = 0) :
    IsZero S.X₂ :=
  (ShortComplex.HomologyData.ofZeros S hf hg).exact_iff.1 ex


lemma exact_iff_mono [HasZeroObject C] (hf : S.f = 0) :
    S.Exact ↔ Mono S.g := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    hf : Eq S.f 0
    ⊢ Iff S.Exact (CategoryTheory.Mono S.g)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      ⊢ S.Exact → CategoryTheory.Mono S.g
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      h : S.Exact
      ⊢ CategoryTheory.Mono S.g
    -/
    have := h.hasHomology
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      h : S.Exact
      this : S.HasHomology
      ⊢ CategoryTheory.Mono S.g
    -/
    simp only [exact_iff_isZero_homology] at h
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      this : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      ⊢ CategoryTheory.Mono S.g
    -/
    have := S.isIso_pOpcycles hf
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      this✝ : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      this : CategoryTheory.IsIso S.pOpcycles
      ⊢ CategoryTheory.Mono S.g
    -/
    have := mono_of_isZero_kernel' _ S.homologyIsKernel h
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      this✝¹ : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      this✝ : CategoryTheory.IsIso S.pOpcycles
      this : CategoryTheory.Mono S.fromOpcycles
      ⊢ CategoryTheory.Mono S.g
    -/
    rw [← S.p_fromOpcycles]
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      this✝¹ : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      this✝ : CategoryTheory.IsIso S.pOpcycles
      this : CategoryTheory.Mono S.fromOpcycles
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp S.pOpcycles S.fromOp …
    -/
    apply mono_comp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      ⊢ CategoryTheory.Mono S.g → S.Exact
    -/
  · intro
    rw [(HomologyData.ofIsLimitKernelFork S hf _
      (KernelFork.IsLimit.ofMonoOfIsZero (KernelFork.ofι (0 : 0 ⟶ S.X₂) zero_comp)
        inferInstance (isZero_zero C))).exact_iff]
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hf : Eq S.f 0
      a✝ : CategoryTheory.Mono S.g
      ⊢ CategoryTheory.Limits.IsZero (CategoryTheory.ShortComplex.HomologyData.ofIsL …
    -/
    exact isZero_zero C
    /-
      🎉 no goals
    -/


lemma exact_iff_epi [HasZeroObject C] (hg : S.g = 0) :
    S.Exact ↔ Epi S.f := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    hg : Eq S.g 0
    ⊢ Iff S.Exact (CategoryTheory.Epi S.f)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      ⊢ S.Exact → CategoryTheory.Epi S.f
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      h : S.Exact
      ⊢ CategoryTheory.Epi S.f
    -/
    have := h.hasHomology
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      h : S.Exact
      this : S.HasHomology
      ⊢ CategoryTheory.Epi S.f
    -/
    simp only [exact_iff_isZero_homology] at h
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      this : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      ⊢ CategoryTheory.Epi S.f
    -/
    haveI := S.isIso_iCycles hg
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      this✝ : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      this : CategoryTheory.IsIso S.iCycles
      ⊢ CategoryTheory.Epi S.f
    -/
    haveI : Epi S.toCycles := epi_of_isZero_cokernel' _ S.homologyIsCokernel h
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      this✝¹ : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      this✝ : CategoryTheory.IsIso S.iCycles
      this : CategoryTheory.Epi S.toCycles
      ⊢ CategoryTheory.Epi S.f
    -/
    rw [← S.toCycles_i]
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      this✝¹ : S.HasHomology
      h : CategoryTheory.Limits.IsZero S.homology
      this✝ : CategoryTheory.IsIso S.iCycles
      this : CategoryTheory.Epi S.toCycles
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp S.toCycles S.iCycles)
    -/
    apply epi_comp
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      ⊢ CategoryTheory.Epi S.f → S.Exact
    -/
  · intro
    rw [(HomologyData.ofIsColimitCokernelCofork S hg _
      (CokernelCofork.IsColimit.ofEpiOfIsZero (CokernelCofork.ofπ (0 : S.X₂ ⟶ 0) comp_zero)
        inferInstance (isZero_zero C))).exact_iff]
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      hg : Eq S.g 0
      a✝ : CategoryTheory.Epi S.f
      ⊢ CategoryTheory.Limits.IsZero (CategoryTheory.ShortComplex.HomologyData.ofIsC …
    -/
    exact isZero_zero C
    /-
      🎉 no goals
    -/


lemma Exact.epi_f' (hS : S.Exact) (h : LeftHomologyData S) : Epi h.f' :=
  epi_of_isZero_cokernel' _ h.hπ (by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      h : S.LeftHomologyData
      ⊢ CategoryTheory.Limits.IsZero (CategoryTheory.Limits.CokernelCofork.ofπ h.π ⋯ …
    -/
    haveI := hS.hasHomology
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      h : S.LeftHomologyData
      this : S.HasHomology
      ⊢ CategoryTheory.Limits.IsZero (CategoryTheory.Limits.CokernelCofork.ofπ h.π ⋯ …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      h : S.LeftHomologyData
      this : S.HasHomology
      ⊢ CategoryTheory.Limits.IsZero h.H
    -/
    simpa only [← h.exact_iff] using hS)
    /-
      🎉 no goals
    -/


lemma Exact.mono_g' (hS : S.Exact) (h : RightHomologyData S) : Mono h.g' :=
  mono_of_isZero_kernel' _ h.hι (by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      h : S.RightHomologyData
      ⊢ CategoryTheory.Limits.IsZero (CategoryTheory.Limits.KernelFork.ofι h.ι ⋯).pt
    -/
    haveI := hS.hasHomology
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      h : S.RightHomologyData
      this : S.HasHomology
      ⊢ CategoryTheory.Limits.IsZero (CategoryTheory.Limits.KernelFork.ofι h.ι ⋯).pt
    -/
    dsimp
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      h : S.RightHomologyData
      this : S.HasHomology
      ⊢ CategoryTheory.Limits.IsZero h.H
    -/
    simpa only [← h.exact_iff] using hS)
    /-
      🎉 no goals
    -/


lemma Exact.epi_toCycles (hS : S.Exact) [S.HasLeftHomology] : Epi S.toCycles :=
  hS.epi_f' _


lemma Exact.mono_fromOpcycles (hS : S.Exact) [S.HasRightHomology] : Mono S.fromOpcycles :=
  hS.mono_g' _


lemma LeftHomologyData.exact_iff_epi_f' [S.HasHomology] (h : LeftHomologyData S) :
    S.Exact ↔ Epi h.f' := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.LeftHomologyData
    ⊢ Iff S.Exact (CategoryTheory.Epi h.f')
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      h : S.LeftHomologyData
      ⊢ S.Exact → CategoryTheory.Epi h.f'
    -/
  · intro hS
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      h : S.LeftHomologyData
      hS : S.Exact
      ⊢ CategoryTheory.Epi h.f'
    -/
    exact hS.epi_f' h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      h : S.LeftHomologyData
      ⊢ CategoryTheory.Epi h.f' → S.Exact
    -/
  · intro
    simp only [h.exact_iff, IsZero.iff_id_eq_zero, ← cancel_epi h.π, ← cancel_epi h.f',
      comp_id, h.f'_π, comp_zero]


lemma RightHomologyData.exact_iff_mono_g' [S.HasHomology] (h : RightHomologyData S) :
    S.Exact ↔ Mono h.g' := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasHomology
    h : S.RightHomologyData
    ⊢ Iff S.Exact (CategoryTheory.Mono h.g')
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      h : S.RightHomologyData
      ⊢ S.Exact → CategoryTheory.Mono h.g'
    -/
  · intro hS
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      h : S.RightHomologyData
      hS : S.Exact
      ⊢ CategoryTheory.Mono h.g'
    -/
    exact hS.mono_g' h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝ : S.HasHomology
      h : S.RightHomologyData
      ⊢ CategoryTheory.Mono h.g' → S.Exact
    -/
  · intro
    simp only [h.exact_iff, IsZero.iff_id_eq_zero, ← cancel_mono h.ι, ← cancel_mono h.g',
      id_comp, h.ι_g', zero_comp]


/-- Given an exact short complex `S` and a limit kernel fork `kf` for `S.g`, this is the
left homology data for `S` with `K := kf.pt` and `H := 0`. -/
@[simps]
noncomputable def Exact.leftHomologyDataOfIsLimitKernelFork
    (hS : S.Exact) [HasZeroObject C] (kf : KernelFork S.g) (hkf : IsLimit kf) :
    S.LeftHomologyData where
  K := kf.pt
  H := 0
  i := kf.ι
  π := 0
  wi := kf.condition
                                                          /-
                                                            C : Type u_1
                                                            D : Type u_2
                                                            inst✝⁴ : CategoryTheory.Category.{?u.41048, u_1} C
                                                            inst✝³ : CategoryTheory.Category.{?u.41052, u_2} D
                                                            inst✝² : CategoryTheory.Preadditive C
                                                            inst✝¹ : CategoryTheory.Preadditive D
                                                            S : CategoryTheory.ShortComplex C
                                                            hS : S.Exact
                                                            inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                            kf : CategoryTheory.Limits.KernelFork S.g
                                                            hkf : CategoryTheory.Limits.IsLimit kf
                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl kf.pt).hom ( …
                                                          -/
  hi := IsLimit.ofIsoLimit hkf (Fork.ext (Iso.refl _) (by simp))
                                                          /-
                                                            🎉 no goals
                                                          -/
  wπ := comp_zero
  hπ := CokernelCofork.IsColimit.ofEpiOfIsZero _ (by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.41048, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.41052, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      kf : CategoryTheory.Limits.KernelFork S.g
      hkf : CategoryTheory.Limits.IsLimit kf
      ⊢ CategoryTheory.Epi ((hkf.ofIsoLimit (CategoryTheory.Limits.Fork.ext (Categor …
    -/
    have := hS.hasHomology
    refine ((MorphismProperty.epimorphisms C).arrow_mk_iso_iff ?_).1
      hS.epi_toCycles
    refine Arrow.isoMk (Iso.refl _)
      (IsLimit.conePointUniqueUpToIso S.cyclesIsKernel hkf) ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.41048, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.41052, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      kf : CategoryTheory.Limits.KernelFork S.g
      hkf : CategoryTheory.Limits.IsLimit kf
      this : S.HasHomology
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
    -/
    apply Fork.IsLimit.hom_ext hkf
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.41048, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.41052, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      kf : CategoryTheory.Limits.KernelFork S.g
      hkf : CategoryTheory.Limits.IsLimit kf
      this : S.HasHomology
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp [IsLimit.conePointUniqueUpToIso]) (isZero_zero C)
    /-
      🎉 no goals
    -/


/-- Given an exact short complex `S` and a colimit cokernel cofork `cc` for `S.f`, this is the
right homology data for `S` with `Q := cc.pt` and `H := 0`. -/
@[simps]
noncomputable def Exact.rightHomologyDataOfIsColimitCokernelCofork
    (hS : S.Exact) [HasZeroObject C] (cc : CokernelCofork S.f) (hcc : IsColimit cc) :
    S.RightHomologyData where
  Q := cc.pt
  H := 0
  p := cc.π
  ι := 0
  wp := cc.condition
                                                                /-
                                                                  C : Type u_1
                                                                  D : Type u_2
                                                                  inst✝⁴ : CategoryTheory.Category.{?u.46580, u_1} C
                                                                  inst✝³ : CategoryTheory.Category.{?u.46584, u_2} D
                                                                  inst✝² : CategoryTheory.Preadditive C
                                                                  inst✝¹ : CategoryTheory.Preadditive D
                                                                  S : CategoryTheory.ShortComplex C
                                                                  hS : S.Exact
                                                                  inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                                  cc : CategoryTheory.Limits.CokernelCofork S.f
                                                                  hcc : CategoryTheory.Limits.IsColimit cc
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π cc) ( …
                                                                -/
  hp := IsColimit.ofIsoColimit hcc (Cofork.ext (Iso.refl _) (by simp))
                                                                /-
                                                                  🎉 no goals
                                                                -/
  wι := zero_comp
  hι := KernelFork.IsLimit.ofMonoOfIsZero _ (by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.46580, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.46584, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      cc : CategoryTheory.Limits.CokernelCofork S.f
      hcc : CategoryTheory.Limits.IsColimit cc
      ⊢ CategoryTheory.Mono ((hcc.ofIsoColimit (CategoryTheory.Limits.Cofork.ext (Ca …
    -/
    have := hS.hasHomology
    refine ((MorphismProperty.monomorphisms C).arrow_mk_iso_iff ?_).2
      hS.mono_fromOpcycles
    refine Arrow.isoMk (IsColimit.coconePointUniqueUpToIso hcc S.opcyclesIsCokernel)
      (Iso.refl _) ?_
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.46580, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.46584, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      cc : CategoryTheory.Limits.CokernelCofork S.f
      hcc : CategoryTheory.Limits.IsColimit cc
      this : S.HasHomology
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (hcc.coconePointUniqueUpToIso S.opcyc …
    -/
    apply Cofork.IsColimit.hom_ext hcc
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.46580, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.46584, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      cc : CategoryTheory.Limits.CokernelCofork S.f
      hcc : CategoryTheory.Limits.IsColimit cc
      this : S.HasHomology
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π cc) ( …
    -/
    simp [IsColimit.coconePointUniqueUpToIso]) (isZero_zero C)
    /-
      🎉 no goals
    -/


lemma exact_iff_epi_toCycles [S.HasHomology] : S.Exact ↔ Epi S.toCycles :=
  S.leftHomologyData.exact_iff_epi_f'


lemma exact_iff_mono_fromOpcycles [S.HasHomology] : S.Exact ↔ Mono S.fromOpcycles :=
  S.rightHomologyData.exact_iff_mono_g'


lemma exact_iff_epi_kernel_lift [S.HasHomology] [HasKernel S.g] :
    S.Exact ↔ Epi (kernel.lift S.g S.f S.zero) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasHomology
    inst✝ : CategoryTheory.Limits.HasKernel S.g
    ⊢ Iff S.Exact (CategoryTheory.Epi (CategoryTheory.Limits.kernel.lift S.g S.f ⋯))
  -/
  rw [exact_iff_epi_toCycles]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasHomology
    inst✝ : CategoryTheory.Limits.HasKernel S.g
    ⊢ Iff (CategoryTheory.Epi S.toCycles) (CategoryTheory.Epi (CategoryTheory.Limi …
  -/
  apply (MorphismProperty.epimorphisms C).arrow_mk_iso_iff
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasHomology
    inst✝ : CategoryTheory.Limits.HasKernel S.g
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk S.toCycles) (CategoryTheory.Arro …
  -/
  exact Arrow.isoMk (Iso.refl _) S.cyclesIsoKernel (by aesop_cat)
  /-
    🎉 no goals
  -/


lemma exact_iff_mono_cokernel_desc [S.HasHomology] [HasCokernel S.f] :
    S.Exact ↔ Mono (cokernel.desc S.f S.g S.zero) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasHomology
    inst✝ : CategoryTheory.Limits.HasCokernel S.f
    ⊢ Iff S.Exact (CategoryTheory.Mono (CategoryTheory.Limits.cokernel.desc S.f S. …
  -/
  rw [exact_iff_mono_fromOpcycles]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasHomology
    inst✝ : CategoryTheory.Limits.HasCokernel S.f
    ⊢ Iff (CategoryTheory.Mono S.fromOpcycles) (CategoryTheory.Mono (CategoryTheor …
  -/
  refine (MorphismProperty.monomorphisms C).arrow_mk_iso_iff (Iso.symm ?_)
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasHomology
    inst✝ : CategoryTheory.Limits.HasCokernel S.f
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (CategoryTheory.Limits.cokernel. …
  -/
  exact Arrow.isoMk S.opcyclesIsoCokernel.symm (Iso.refl _) (by aesop_cat)
  /-
    🎉 no goals
  -/


lemma QuasiIso.exact_iff {S₁ S₂ : ShortComplex C} (φ : S₁ ⟶ S₂)
    [S₁.HasHomology] [S₂.HasHomology] [QuasiIso φ] : S₁.Exact ↔ S₂.Exact := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ Iff S₁.Exact S₂.Exact
  -/
  simp only [exact_iff_isZero_homology]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝² : S₁.HasHomology
    inst✝¹ : S₂.HasHomology
    inst✝ : CategoryTheory.ShortComplex.QuasiIso φ
    ⊢ Iff (CategoryTheory.Limits.IsZero S₁.homology) (CategoryTheory.Limits.IsZero …
  -/
  exact Iso.isZero_iff (asIso (homologyMap φ))
  /-
    🎉 no goals
  -/


lemma exact_of_f_is_kernel (hS : IsLimit (KernelFork.ofι S.f S.zero))
    [S.HasHomology] : S.Exact := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
    inst✝ : S.HasHomology
    ⊢ S.Exact
  -/
  rw [exact_iff_epi_toCycles]
  have : IsSplitEpi S.toCycles :=
    ⟨⟨{ section_ := hS.lift (KernelFork.ofι S.iCycles S.iCycles_g)
        id := by
          rw [← cancel_mono S.iCycles, assoc, toCycles_i, id_comp]
          exact Fork.IsLimit.lift_ι hS }⟩⟩
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
    inst✝ : S.HasHomology
    this : CategoryTheory.IsSplitEpi S.toCycles
    ⊢ CategoryTheory.Epi S.toCycles
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma exact_of_g_is_cokernel (hS : IsColimit (CokernelCofork.ofπ S.g S.zero))
    [S.HasHomology] : S.Exact := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    inst✝ : S.HasHomology
    ⊢ S.Exact
  -/
  rw [exact_iff_mono_fromOpcycles]
  have : IsSplitMono S.fromOpcycles :=
    ⟨⟨{ retraction := hS.desc (CokernelCofork.ofπ S.pOpcycles S.f_pOpcycles)
        id := by
          rw [← cancel_epi S.pOpcycles, p_fromOpcycles_assoc, comp_id]
          exact Cofork.IsColimit.π_desc hS }⟩⟩
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    inst✝ : S.HasHomology
    this : CategoryTheory.IsSplitMono S.fromOpcycles
    ⊢ CategoryTheory.Mono S.fromOpcycles
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma Exact.mono_g (hS : S.Exact) (hf : S.f = 0) : Mono S.g := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hf : Eq S.f 0
    ⊢ CategoryTheory.Mono S.g
  -/
  have := hS.hasHomology
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hf : Eq S.f 0
    this : S.HasHomology
    ⊢ CategoryTheory.Mono S.g
  -/
  have := hS.epi_toCycles
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hf : Eq S.f 0
    this✝ : S.HasHomology
    this : CategoryTheory.Epi S.toCycles
    ⊢ CategoryTheory.Mono S.g
  -/
  have : S.iCycles = 0 := by rw [← cancel_epi S.toCycles, comp_zero, toCycles_i, hf]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hf : Eq S.f 0
    this✝¹ : S.HasHomology
    this✝ : CategoryTheory.Epi S.toCycles
    this : Eq S.iCycles 0
    ⊢ CategoryTheory.Mono S.g
  -/
  apply Preadditive.mono_of_cancel_zero
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hf : Eq S.f 0
    this✝¹ : S.HasHomology
    this✝ : CategoryTheory.Epi S.toCycles
    this : Eq S.iCycles 0
    ⊢ ∀ {P : C} (g : Quiver.Hom P S.X₂), Eq (CategoryTheory.CategoryStruct.comp g  …
  -/
  intro A x₂ hx₂
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hf : Eq S.f 0
    this✝¹ : S.HasHomology
    this✝ : CategoryTheory.Epi S.toCycles
    this : Eq S.iCycles 0
    A : C
    x₂ : Quiver.Hom A S.X₂
    hx₂ : Eq (CategoryTheory.CategoryStruct.comp x₂ S.g) 0
    ⊢ Eq x₂ 0
  -/
  rw [← S.liftCycles_i x₂ hx₂, this, comp_zero]
  /-
    🎉 no goals
  -/


lemma Exact.epi_f (hS : S.Exact) (hg : S.g = 0) : Epi S.f := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hg : Eq S.g 0
    ⊢ CategoryTheory.Epi S.f
  -/
  have := hS.hasHomology
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hg : Eq S.g 0
    this : S.HasHomology
    ⊢ CategoryTheory.Epi S.f
  -/
  have := hS.mono_fromOpcycles
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hg : Eq S.g 0
    this✝ : S.HasHomology
    this : CategoryTheory.Mono S.fromOpcycles
    ⊢ CategoryTheory.Epi S.f
  -/
  have : S.pOpcycles = 0 := by rw [← cancel_mono S.fromOpcycles, zero_comp, p_fromOpcycles, hg]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hg : Eq S.g 0
    this✝¹ : S.HasHomology
    this✝ : CategoryTheory.Mono S.fromOpcycles
    this : Eq S.pOpcycles 0
    ⊢ CategoryTheory.Epi S.f
  -/
  apply Preadditive.epi_of_cancel_zero
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hg : Eq S.g 0
    this✝¹ : S.HasHomology
    this✝ : CategoryTheory.Mono S.fromOpcycles
    this : Eq S.pOpcycles 0
    ⊢ ∀ {R : C} (g : Quiver.Hom S.X₂ R), Eq (CategoryTheory.CategoryStruct.comp S. …
  -/
  intro A x₂ hx₂
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hg : Eq S.g 0
    this✝¹ : S.HasHomology
    this✝ : CategoryTheory.Mono S.fromOpcycles
    this : Eq S.pOpcycles 0
    A : C
    x₂ : Quiver.Hom S.X₂ A
    hx₂ : Eq (CategoryTheory.CategoryStruct.comp S.f x₂) 0
    ⊢ Eq x₂ 0
  -/
  rw [← S.p_descOpcycles x₂ hx₂, this, zero_comp]
  /-
    🎉 no goals
  -/


lemma Exact.mono_g_iff (hS : S.Exact) : Mono S.g ↔ S.f = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    ⊢ Iff (CategoryTheory.Mono S.g) (Eq S.f 0)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      ⊢ CategoryTheory.Mono S.g → Eq S.f 0
    -/
  · intro
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      a✝ : CategoryTheory.Mono S.g
      ⊢ Eq S.f 0
    -/
    rw [← cancel_mono S.g, zero, zero_comp]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      ⊢ Eq S.f 0 → CategoryTheory.Mono S.g
    -/
  · exact hS.mono_g
    /-
      🎉 no goals
    -/


lemma Exact.epi_f_iff (hS : S.Exact) : Epi S.f ↔ S.g = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    ⊢ Iff (CategoryTheory.Epi S.f) (Eq S.g 0)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      ⊢ CategoryTheory.Epi S.f → Eq S.g 0
    -/
  · intro
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      a✝ : CategoryTheory.Epi S.f
      ⊢ Eq S.g 0
    -/
    rw [← cancel_epi S.f, zero, comp_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      ⊢ Eq S.g 0 → CategoryTheory.Epi S.f
    -/
  · exact hS.epi_f
    /-
      🎉 no goals
    -/


lemma Exact.isZero_X₂ (hS : S.Exact) (hf : S.f = 0) (hg : S.g = 0) : IsZero S.X₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hf : Eq S.f 0
    hg : Eq S.g 0
    ⊢ CategoryTheory.Limits.IsZero S.X₂
  -/
  have := hS.mono_g hf
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    hf : Eq S.f 0
    hg : Eq S.g 0
    this : CategoryTheory.Mono S.g
    ⊢ CategoryTheory.Limits.IsZero S.X₂
  -/
  rw [IsZero.iff_id_eq_zero, ← cancel_mono S.g, hg, comp_zero, comp_zero]
  /-
    🎉 no goals
  -/


lemma Exact.isZero_X₂_iff (hS : S.Exact) : IsZero S.X₂ ↔ S.f = 0 ∧ S.g = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    ⊢ Iff (CategoryTheory.Limits.IsZero S.X₂) (And (Eq S.f 0) (Eq S.g 0))
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      ⊢ CategoryTheory.Limits.IsZero S.X₂ → And (Eq S.f 0) (Eq S.g 0)
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      h : CategoryTheory.Limits.IsZero S.X₂
      ⊢ And (Eq S.f 0) (Eq S.g 0)
    -/
    exact ⟨h.eq_of_tgt _ _, h.eq_of_src _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      ⊢ And (Eq S.f 0) (Eq S.g 0) → CategoryTheory.Limits.IsZero S.X₂
    -/
  · rintro ⟨hf, hg⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      hS : S.Exact
      hf : Eq S.f 0
      hg : Eq S.g 0
      ⊢ CategoryTheory.Limits.IsZero S.X₂
    -/
    exact hS.isZero_X₂ hf hg
    /-
      🎉 no goals
    -/


/-- A splitting for a short complex `S` consists of the data of a retraction `r : X₂ ⟶ X₁`
of `S.f` and section `s : X₃ ⟶ X₂` of `S.g` which satisfy `r ≫ S.f + S.g ≫ s = 𝟙 _` -/
structure Splitting (S : ShortComplex C) where
  /-- a retraction of `S.f` -/
  r : S.X₂ ⟶ S.X₁
  /-- a section of `S.g` -/
  s : S.X₃ ⟶ S.X₂
  /-- the condition that `r` is a retraction of `S.f` -/
  f_r : S.f ≫ r = 𝟙 _ := by aesop_cat
  /-- the condition that `s` is a section of `S.g` -/
  s_g : s ≫ S.g = 𝟙 _ := by aesop_cat
  /-- the compatibility between the given section and retraction -/
  id : r ≫ S.f + S.g ≫ s = 𝟙 _ := by aesop_cat


attribute [reassoc (attr := simp)] f_r s_g


@[reassoc]
                                                                /-
                                                                  C : Type u_1
                                                                  inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                                                                  inst✝ : CategoryTheory.Preadditive C
                                                                  S : CategoryTheory.ShortComplex C
                                                                  s : S.Splitting
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp s.r S.f) (HSub.hSub (CategoryTheory.C …
                                                                -/
lemma r_f (s : S.Splitting) : s.r ≫ S.f = 𝟙 _ - S.g ≫ s.s := by rw [← s.id, add_sub_cancel_right]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[reassoc]
                                                                /-
                                                                  C : Type u_1
                                                                  inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
                                                                  inst✝ : CategoryTheory.Preadditive C
                                                                  S : CategoryTheory.ShortComplex C
                                                                  s : S.Splitting
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp S.g s.s) (HSub.hSub (CategoryTheory.C …
                                                                -/
lemma g_s (s : S.Splitting) : S.g ≫ s.s = 𝟙 _ - s.r ≫ S.f := by rw [← s.id, add_sub_cancel_left]
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- Given a splitting of a short complex `S`, this shows that `S.f` is a split monomorphism. -/
@[simps] def splitMono_f (s : S.Splitting) : SplitMono S.f := ⟨s.r, s.f_r⟩


lemma isSplitMono_f (s : S.Splitting) : IsSplitMono S.f := ⟨⟨s.splitMono_f⟩⟩


lemma mono_f (s : S.Splitting) : Mono S.f := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s : S.Splitting
    ⊢ CategoryTheory.Mono S.f
  -/
  have := s.isSplitMono_f
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s : S.Splitting
    this : CategoryTheory.IsSplitMono S.f
    ⊢ CategoryTheory.Mono S.f
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Given a splitting of a short complex `S`, this shows that `S.g` is a split epimorphism. -/
@[simps] def splitEpi_g (s : S.Splitting) : SplitEpi S.g := ⟨s.s, s.s_g⟩


lemma isSplitEpi_g (s : S.Splitting) : IsSplitEpi S.g := ⟨⟨s.splitEpi_g⟩⟩


lemma epi_g (s : S.Splitting) : Epi S.g := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s : S.Splitting
    ⊢ CategoryTheory.Epi S.g
  -/
  have := s.isSplitEpi_g
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s : S.Splitting
    this : CategoryTheory.IsSplitEpi S.g
    ⊢ CategoryTheory.Epi S.g
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma s_r (s : S.Splitting) : s.s ≫ s.r = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s : S.Splitting
    ⊢ Eq (CategoryTheory.CategoryStruct.comp s.s s.r) 0
  -/
  have := s.epi_g
  simp only [← cancel_epi S.g, comp_zero, g_s_assoc, sub_comp, id_comp,
    assoc, f_r, comp_id, sub_self]


lemma ext_r (s s' : S.Splitting) (h : s.r = s'.r) : s = s' := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s s' : S.Splitting
    h : Eq s.r s'.r
    ⊢ Eq s s'
  -/
  have := s.epi_g
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s s' : S.Splitting
    h : Eq s.r s'.r
    this : CategoryTheory.Epi S.g
    ⊢ Eq s s'
  -/
  have eq := s.id
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s s' : S.Splitting
    h : Eq s.r s'.r
    this : CategoryTheory.Epi S.g
    eq : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp s.r S.f) (CategoryTheor …
    ⊢ Eq s s'
  -/
  rw [← s'.id, h, add_right_inj, cancel_epi S.g] at eq
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s s' : S.Splitting
    h : Eq s.r s'.r
    this : CategoryTheory.Epi S.g
    eq : Eq s.s s'.s
    ⊢ Eq s s'
  -/
  cases s
  /-
    case mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s' : S.Splitting
    this : CategoryTheory.Epi S.g
    r✝ : Quiver.Hom S.X₂ S.X₁
    s✝ : Quiver.Hom S.X₃ S.X₂
    f_r✝ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝) (CategoryTheory.Category …
    s_g✝ : Eq (CategoryTheory.CategoryStruct.comp s✝ S.g) (CategoryTheory.Category …
    id✝ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝ S.f) (CategoryTheor …
    h : Eq { r := r✝, s := s✝, f_r := f_r✝, s_g := s_g✝, id := id✝ }.r s'.r
    eq : Eq { r := r✝, s := s✝, f_r := f_r✝, s_g := s_g✝, id := id✝ }.s s'.s
    ⊢ Eq { r := r✝, s := s✝, f_r := f_r✝, s_g := s_g✝, id := id✝ } s'
  -/
  cases s'
  /-
    case mk.mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    this : CategoryTheory.Epi S.g
    r✝¹ : Quiver.Hom S.X₂ S.X₁
    s✝¹ : Quiver.Hom S.X₃ S.X₂
    f_r✝¹ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝¹) (CategoryTheory.Catego …
    s_g✝¹ : Eq (CategoryTheory.CategoryStruct.comp s✝¹ S.g) (CategoryTheory.Catego …
    id✝¹ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝¹ S.f) (CategoryThe …
    r✝ : Quiver.Hom S.X₂ S.X₁
    s✝ : Quiver.Hom S.X₃ S.X₂
    f_r✝ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝) (CategoryTheory.Category …
    s_g✝ : Eq (CategoryTheory.CategoryStruct.comp s✝ S.g) (CategoryTheory.Category …
    id✝ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝ S.f) (CategoryTheor …
    h : Eq { r := r✝¹, s := s✝¹, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ }.r { r := …
    eq : Eq { r := r✝¹, s := s✝¹, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ }.s { r : …
    ⊢ Eq { r := r✝¹, s := s✝¹, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ } { r := r✝, …
  -/
  obtain rfl := eq
  /-
    case mk.mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    this : CategoryTheory.Epi S.g
    r✝¹ : Quiver.Hom S.X₂ S.X₁
    s✝ : Quiver.Hom S.X₃ S.X₂
    f_r✝¹ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝¹) (CategoryTheory.Catego …
    s_g✝¹ : Eq (CategoryTheory.CategoryStruct.comp s✝ S.g) (CategoryTheory.Categor …
    id✝¹ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝¹ S.f) (CategoryThe …
    r✝ : Quiver.Hom S.X₂ S.X₁
    f_r✝ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝) (CategoryTheory.Category …
    s_g✝ : Eq (CategoryTheory.CategoryStruct.comp { r := r✝¹, s := s✝, f_r := f_r✝ …
    id✝ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝ S.f) (CategoryTheor …
    h : Eq { r := r✝¹, s := s✝, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ }.r { r :=  …
    ⊢ Eq { r := r✝¹, s := s✝, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ } { r := r✝,  …
  -/
  obtain rfl := h
  /-
    case mk.mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    this : CategoryTheory.Epi S.g
    r✝ : Quiver.Hom S.X₂ S.X₁
    s✝ : Quiver.Hom S.X₃ S.X₂
    f_r✝¹ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝) (CategoryTheory.Categor …
    s_g✝¹ : Eq (CategoryTheory.CategoryStruct.comp s✝ S.g) (CategoryTheory.Categor …
    id✝¹ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝ S.f) (CategoryTheo …
    s_g✝ : Eq (CategoryTheory.CategoryStruct.comp { r := r✝, s := s✝, f_r := f_r✝¹ …
    f_r✝ : Eq (CategoryTheory.CategoryStruct.comp S.f { r := r✝, s := s✝, f_r := f …
    id✝ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp { r := r✝, s := s✝, f_ …
    ⊢ Eq { r := r✝, s := s✝, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ } { r := { r : …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma ext_s (s s' : S.Splitting) (h : s.s = s'.s) : s = s' := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s s' : S.Splitting
    h : Eq s.s s'.s
    ⊢ Eq s s'
  -/
  have := s.mono_f
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s s' : S.Splitting
    h : Eq s.s s'.s
    this : CategoryTheory.Mono S.f
    ⊢ Eq s s'
  -/
  have eq := s.id
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s s' : S.Splitting
    h : Eq s.s s'.s
    this : CategoryTheory.Mono S.f
    eq : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp s.r S.f) (CategoryTheor …
    ⊢ Eq s s'
  -/
  rw [← s'.id, h, add_left_inj, cancel_mono S.f] at eq
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s s' : S.Splitting
    h : Eq s.s s'.s
    this : CategoryTheory.Mono S.f
    eq : Eq s.r s'.r
    ⊢ Eq s s'
  -/
  cases s
  /-
    case mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    s' : S.Splitting
    this : CategoryTheory.Mono S.f
    r✝ : Quiver.Hom S.X₂ S.X₁
    s✝ : Quiver.Hom S.X₃ S.X₂
    f_r✝ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝) (CategoryTheory.Category …
    s_g✝ : Eq (CategoryTheory.CategoryStruct.comp s✝ S.g) (CategoryTheory.Category …
    id✝ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝ S.f) (CategoryTheor …
    h : Eq { r := r✝, s := s✝, f_r := f_r✝, s_g := s_g✝, id := id✝ }.s s'.s
    eq : Eq { r := r✝, s := s✝, f_r := f_r✝, s_g := s_g✝, id := id✝ }.r s'.r
    ⊢ Eq { r := r✝, s := s✝, f_r := f_r✝, s_g := s_g✝, id := id✝ } s'
  -/
  cases s'
  /-
    case mk.mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    this : CategoryTheory.Mono S.f
    r✝¹ : Quiver.Hom S.X₂ S.X₁
    s✝¹ : Quiver.Hom S.X₃ S.X₂
    f_r✝¹ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝¹) (CategoryTheory.Catego …
    s_g✝¹ : Eq (CategoryTheory.CategoryStruct.comp s✝¹ S.g) (CategoryTheory.Catego …
    id✝¹ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝¹ S.f) (CategoryThe …
    r✝ : Quiver.Hom S.X₂ S.X₁
    s✝ : Quiver.Hom S.X₃ S.X₂
    f_r✝ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝) (CategoryTheory.Category …
    s_g✝ : Eq (CategoryTheory.CategoryStruct.comp s✝ S.g) (CategoryTheory.Category …
    id✝ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝ S.f) (CategoryTheor …
    h : Eq { r := r✝¹, s := s✝¹, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ }.s { r := …
    eq : Eq { r := r✝¹, s := s✝¹, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ }.r { r : …
    ⊢ Eq { r := r✝¹, s := s✝¹, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ } { r := r✝, …
  -/
  obtain rfl := eq
  /-
    case mk.mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    this : CategoryTheory.Mono S.f
    r✝ : Quiver.Hom S.X₂ S.X₁
    s✝¹ : Quiver.Hom S.X₃ S.X₂
    f_r✝¹ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝) (CategoryTheory.Categor …
    s_g✝¹ : Eq (CategoryTheory.CategoryStruct.comp s✝¹ S.g) (CategoryTheory.Catego …
    id✝¹ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝ S.f) (CategoryTheo …
    s✝ : Quiver.Hom S.X₃ S.X₂
    s_g✝ : Eq (CategoryTheory.CategoryStruct.comp s✝ S.g) (CategoryTheory.Category …
    f_r✝ : Eq (CategoryTheory.CategoryStruct.comp S.f { r := r✝, s := s✝¹, f_r :=  …
    id✝ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp { r := r✝, s := s✝¹, f …
    h : Eq { r := r✝, s := s✝¹, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ }.s { r :=  …
    ⊢ Eq { r := r✝, s := s✝¹, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ } { r := { r  …
  -/
  obtain rfl := h
  /-
    case mk.mk
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    this : CategoryTheory.Mono S.f
    r✝ : Quiver.Hom S.X₂ S.X₁
    s✝ : Quiver.Hom S.X₃ S.X₂
    f_r✝¹ : Eq (CategoryTheory.CategoryStruct.comp S.f r✝) (CategoryTheory.Categor …
    s_g✝¹ : Eq (CategoryTheory.CategoryStruct.comp s✝ S.g) (CategoryTheory.Categor …
    id✝¹ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp r✝ S.f) (CategoryTheo …
    f_r✝ : Eq (CategoryTheory.CategoryStruct.comp S.f { r := r✝, s := s✝, f_r := f …
    s_g✝ : Eq (CategoryTheory.CategoryStruct.comp { r := r✝, s := s✝, f_r := f_r✝¹ …
    id✝ : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp { r := r✝, s := s✝, f_ …
    ⊢ Eq { r := r✝, s := s✝, f_r := f_r✝¹, s_g := s_g✝¹, id := id✝¹ } { r := { r : …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The left homology data on a short complex equipped with a splitting. -/
@[simps]
noncomputable def leftHomologyData [HasZeroObject C] (s : S.Splitting) :
    LeftHomologyData S := by
  have hi := KernelFork.IsLimit.ofι S.f S.zero
    (fun x _ => x ≫ s.r)
    (fun x hx => by simp only [assoc, s.r_f, comp_sub, comp_id,
      sub_eq_self, reassoc_of% hx, zero_comp])
    (fun x _ b hb => by simp only [← hb, assoc, f_r, comp_id])
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.85863, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.85867, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Preadditive D
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    s : S.Splitting
    hi : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
    ⊢ S.LeftHomologyData
  -/
  let f' := hi.lift (KernelFork.ofι S.f S.zero)
  have hf' : f' = 𝟙 _ := by
    apply Fork.IsLimit.hom_ext hi
    dsimp
    erw [Fork.IsLimit.lift_ι hi]
    simp only [Fork.ι_ofι, id_comp]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.85863, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.85867, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Preadditive D
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    s : S.Splitting
    hi : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
    f' : Quiver.Hom (CategoryTheory.Limits.KernelFork.ofι S.f ⋯).pt (CategoryTheor …
    hf' : Eq f' (CategoryTheory.CategoryStruct.id (CategoryTheory.Limits.KernelFor …
    ⊢ S.LeftHomologyData
  -/
  have wπ : f' ≫ (0 : S.X₁ ⟶ 0) = 0 := comp_zero
  have hπ : IsColimit (CokernelCofork.ofπ 0 wπ) := CokernelCofork.IsColimit.ofEpiOfIsZero _
      (by rw [hf']; infer_instance) (isZero_zero _)
  exact
    { K := S.X₁
      H := 0
      i := S.f
      wi := S.zero
      hi := hi
      π := 0
      wπ := wπ
      hπ := hπ }


/-- The right homology data on a short complex equipped with a splitting. -/
@[simps]
noncomputable def rightHomologyData [HasZeroObject C] (s : S.Splitting) :
    RightHomologyData S := by
  have hp := CokernelCofork.IsColimit.ofπ S.g S.zero
    (fun x _ => s.s ≫ x)
    (fun x hx => by simp only [s.g_s_assoc, sub_comp, id_comp, sub_eq_self, assoc, hx, comp_zero])
    (fun x _ b hb => by simp only [← hb, s.s_g_assoc])
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.89996, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.90000, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Preadditive D
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    s : S.Splitting
    hp : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    ⊢ S.RightHomologyData
  -/
  let g' := hp.desc (CokernelCofork.ofπ S.g S.zero)
  have hg' : g' = 𝟙 _ := by
    apply Cofork.IsColimit.hom_ext hp
    dsimp
    erw [Cofork.IsColimit.π_desc hp]
    simp only [Cofork.π_ofπ, comp_id]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.89996, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.90000, u_2} D
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Preadditive D
    S : CategoryTheory.ShortComplex C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    s : S.Splitting
    hp : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    g' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ S.g ⋯).pt (CategoryT …
    hg' : Eq g' (CategoryTheory.CategoryStruct.id (CategoryTheory.Limits.CokernelC …
    ⊢ S.RightHomologyData
  -/
  have wι : (0 : 0 ⟶ S.X₃) ≫ g' = 0 := zero_comp
  have hι : IsLimit (KernelFork.ofι 0 wι) := KernelFork.IsLimit.ofMonoOfIsZero _
      (by rw [hg']; dsimp; infer_instance) (isZero_zero _)
  exact
    { Q := S.X₃
      H := 0
      p := S.g
      wp := S.zero
      hp := hp
      ι := 0
      wι := wι
      hι := hι }


/-- The homology data on a short complex equipped with a splitting. -/
@[simps]
noncomputable def homologyData [HasZeroObject C] (s : S.Splitting) : S.HomologyData where
  left := s.leftHomologyData
  right := s.rightHomologyData
  iso := Iso.refl 0


/-- A short complex equipped with a splitting is exact. -/
lemma exact [HasZeroObject C] (s : S.Splitting) : S.Exact :=
  ⟨s.homologyData, isZero_zero _⟩


/-- If a short complex `S` is equipped with a splitting, then `S.X₁` is the kernel of `S.g`. -/
noncomputable def fIsKernel [HasZeroObject C] (s : S.Splitting) :
    IsLimit (KernelFork.ofι S.f S.zero) :=
  s.homologyData.left.hi


/-- If a short complex `S` is equipped with a splitting, then `S.X₃` is the cokernel of `S.f`. -/
noncomputable def gIsCokernel [HasZeroObject C] (s : S.Splitting) :
    IsColimit (CokernelCofork.ofπ S.g S.zero) :=
  s.homologyData.right.hp


/-- If a short complex `S` has a splitting and `F` is an additive functor, then
`S.map F` also has a splitting. -/
@[simps]
def map (s : S.Splitting) (F : C ⥤ D) [F.Additive] : (S.map F).Splitting where
  r := F.map s.r
  s := F.map s.s
  f_r := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.96868, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.96872, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      s : S.Splitting
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map F).f (F.map s.r)) (CategoryThe …
    -/
    dsimp [ShortComplex.map]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.96868, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.96872, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      s : S.Splitting
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map S.f) (F.map s.r)) (CategoryThe …
    -/
    rw [← F.map_comp, f_r, F.map_id]
    /-
      🎉 no goals
    -/
  s_g := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.96868, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.96872, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      s : S.Splitting
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map s.s) (S.map F).g) (CategoryThe …
    -/
    dsimp [ShortComplex.map]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.96868, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.96872, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      s : S.Splitting
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map s.s) (F.map S.g)) (CategoryThe …
    -/
    simp only [← F.map_comp, s_g, F.map_id]
    /-
      🎉 no goals
    -/
  id := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.96868, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.96872, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      s : S.Splitting
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (F.map s.r) (S.map F).f) ( …
    -/
    dsimp [ShortComplex.map]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁴ : CategoryTheory.Category.{?u.96868, u_1} C
      inst✝³ : CategoryTheory.Category.{?u.96872, u_2} D
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      s : S.Splitting
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (F.map s.r) (F.map S.f)) ( …
    -/
    simp only [← F.map_id, ← s.id, Functor.map_comp, Functor.map_add]
    /-
      🎉 no goals
    -/


/-- A splitting on a short complex induces splittings on isomorphic short complexes. -/
@[simps]
def ofIso {S₁ S₂ : ShortComplex C} (s : S₁.Splitting) (e : S₁ ≅ S₂) : S₂.Splitting where
  r := e.inv.τ₂ ≫ s.r ≫ e.hom.τ₁
  s := e.inv.τ₃ ≫ s.s ≫ e.hom.τ₂
            /-
              C : Type u_1
              D : Type u_2
              inst✝³ : CategoryTheory.Category.{?u.98702, u_1} C
              inst✝² : CategoryTheory.Category.{?u.98706, u_2} D
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Preadditive D
              S S₁ S₂ : CategoryTheory.ShortComplex C
              s : S₁.Splitting
              e : CategoryTheory.Iso S₁ S₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp S₂.f (CategoryTheory.CategoryStruct.c …
            -/
  f_r := by rw [← e.inv.comm₁₂_assoc, s.f_r_assoc, ← comp_τ₁, e.inv_hom_id, id_τ₁]
            /-
              🎉 no goals
            -/
            /-
              C : Type u_1
              D : Type u_2
              inst✝³ : CategoryTheory.Category.{?u.98702, u_1} C
              inst✝² : CategoryTheory.Category.{?u.98706, u_2} D
              inst✝¹ : CategoryTheory.Preadditive C
              inst✝ : CategoryTheory.Preadditive D
              S S₁ S₂ : CategoryTheory.ShortComplex C
              s : S₁.Splitting
              e : CategoryTheory.Iso S₁ S₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp e …
            -/
  s_g := by rw [assoc, assoc, e.hom.comm₂₃, s.s_g_assoc, ← comp_τ₃, e.inv_hom_id, id_τ₃]
            /-
              🎉 no goals
            -/
  id := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.98702, u_1} C
      inst✝² : CategoryTheory.Category.{?u.98706, u_2} D
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Preadditive D
      S S₁ S₂ : CategoryTheory.ShortComplex C
      s : S₁.Splitting
      e : CategoryTheory.Iso S₁ S₂
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
    -/
    have eq := e.inv.τ₂ ≫= s.id =≫ e.hom.τ₂
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.98702, u_1} C
      inst✝² : CategoryTheory.Category.{?u.98706, u_2} D
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Preadditive D
      S S₁ S₂ : CategoryTheory.ShortComplex C
      s : S₁.Splitting
      e : CategoryTheory.Iso S₁ S₂
      eq : Eq (CategoryTheory.CategoryStruct.comp e.inv.τ₂ (CategoryTheory.CategoryS …
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
    -/
    rw [id_comp, ← comp_τ₂, e.inv_hom_id, id_τ₂] at eq
    rw [← eq, assoc, assoc, add_comp, assoc, assoc, comp_add,
      e.hom.comm₁₂, e.inv.comm₂₃_assoc]


/-- The obvious splitting of the short complex `X₁ ⟶ X₁ ⊞ X₂ ⟶ X₂`. -/
noncomputable def ofHasBinaryBiproduct (X₁ X₂ : C) [HasBinaryBiproduct X₁ X₂] :
                                                                               /-
                                                                                 C : Type u_1
                                                                                 D : Type u_2
                                                                                 inst✝⁴ : CategoryTheory.Category.{?u.114393, u_1} C
                                                                                 inst✝³ : CategoryTheory.Category.{?u.114397, u_2} D
                                                                                 inst✝² : CategoryTheory.Preadditive C
                                                                                 inst✝¹ : CategoryTheory.Preadditive D
                                                                                 S : CategoryTheory.ShortComplex C
                                                                                 X₁ X₂ : C
                                                                                 inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₁ X₂
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl Cate …
                                                                               -/
    Splitting (ShortComplex.mk (biprod.inl : X₁ ⟶ _) (biprod.snd : _ ⟶ X₂) (by simp)) where
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  r := biprod.fst
  s := biprod.inr


/-- The obvious splitting of a short complex when `S.X₁` is zero and `S.g` is an isomorphism. -/
noncomputable def ofIsZeroOfIsIso (hf : IsZero S.X₁) (hg : IsIso S.g) : Splitting S where
  r := 0
  s := inv S.g
  f_r := hf.eq_of_src _ _


/-- The obvious splitting of a short complex when `S.f` is an isomorphism and `S.X₃` is zero. -/
noncomputable def ofIsIsoOfIsZero (hf : IsIso S.f) (hg : IsZero S.X₃) : Splitting S where
  r := inv S.f
  s := 0
  s_g := hg.eq_of_src _ _


/-- The splitting of the short complex `S.op` deduced from a splitting of `S`. -/
@[simps]
def op (h : Splitting S) : Splitting S.op where
  r := h.s.op
  s := h.r.op
                                 /-
                                   C : Type u_1
                                   D : Type u_2
                                   inst✝³ : CategoryTheory.Category.{?u.122501, u_1} C
                                   inst✝² : CategoryTheory.Category.{?u.122505, u_2} D
                                   inst✝¹ : CategoryTheory.Preadditive C
                                   inst✝ : CategoryTheory.Preadditive D
                                   S : CategoryTheory.ShortComplex C
                                   h : S.Splitting
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp S.op.f h.s.op).unop (CategoryTheory.C …
                                 -/
  f_r := Quiver.Hom.unop_inj (by simp)
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   C : Type u_1
                                   D : Type u_2
                                   inst✝³ : CategoryTheory.Category.{?u.122501, u_1} C
                                   inst✝² : CategoryTheory.Category.{?u.122505, u_2} D
                                   inst✝¹ : CategoryTheory.Preadditive C
                                   inst✝ : CategoryTheory.Preadditive D
                                   S : CategoryTheory.ShortComplex C
                                   h : S.Splitting
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp h.r.op S.op.g).unop (CategoryTheory.C …
                                 -/
  s_g := Quiver.Hom.unop_inj (by simp)
                                 /-
                                   🎉 no goals
                                 -/
  id := Quiver.Hom.unop_inj (by
    simp only [op_X₂, Opposite.unop_op, op_X₁, op_f, op_X₃, op_g, unop_add, unop_comp,
      Quiver.Hom.unop_op, unop_id, ← h.id]
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.122501, u_1} C
      inst✝² : CategoryTheory.Category.{?u.122505, u_2} D
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Preadditive D
      S : CategoryTheory.ShortComplex C
      h : S.Splitting
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp S.g h.s) (CategoryTheory.C …
    -/
    /-
      🎉 no goals
    -/
    abel)
    /-
      🎉 no goals
    -/


/-- The splitting of the short complex `S.unop` deduced from a splitting of `S`. -/
@[simps]
def unop {S : ShortComplex Cᵒᵖ} (h : Splitting S) : Splitting S.unop where
  r := h.s.unop
  s := h.r.unop
                               /-
                                 C : Type u_1
                                 D : Type u_2
                                 inst✝³ : CategoryTheory.Category.{?u.125516, u_1} C
                                 inst✝² : CategoryTheory.Category.{?u.125520, u_2} D
                                 inst✝¹ : CategoryTheory.Preadditive C
                                 inst✝ : CategoryTheory.Preadditive D
                                 S✝ : CategoryTheory.ShortComplex C
                                 S : CategoryTheory.ShortComplex (Opposite C)
                                 h : S.Splitting
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp S.unop.f h.s.unop).op (CategoryTheory …
                               -/
  f_r := Quiver.Hom.op_inj (by simp)
                               /-
                                 🎉 no goals
                               -/
                               /-
                                 C : Type u_1
                                 D : Type u_2
                                 inst✝³ : CategoryTheory.Category.{?u.125516, u_1} C
                                 inst✝² : CategoryTheory.Category.{?u.125520, u_2} D
                                 inst✝¹ : CategoryTheory.Preadditive C
                                 inst✝ : CategoryTheory.Preadditive D
                                 S✝ : CategoryTheory.ShortComplex C
                                 S : CategoryTheory.ShortComplex (Opposite C)
                                 h : S.Splitting
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp h.r.unop S.unop.g).op (CategoryTheory …
                               -/
  s_g := Quiver.Hom.op_inj (by simp)
                               /-
                                 🎉 no goals
                               -/
  id := Quiver.Hom.op_inj (by
    simp only [unop_X₂, Opposite.op_unop, unop_X₁, unop_f, unop_X₃, unop_g, op_add,
      op_comp, Quiver.Hom.op_unop, op_id, ← h.id]
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{?u.125516, u_1} C
      inst✝² : CategoryTheory.Category.{?u.125520, u_2} D
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Preadditive D
      S✝ : CategoryTheory.ShortComplex C
      S : CategoryTheory.ShortComplex (Opposite C)
      h : S.Splitting
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp S.g h.s) (CategoryTheory.C …
    -/
    /-
      🎉 no goals
    -/
    abel)
    /-
      🎉 no goals
    -/


/-- The isomorphism `S.X₂ ≅ S.X₁ ⊞ S.X₃` induced by a splitting of the short complex `S`. -/
@[simps]
noncomputable def isoBinaryBiproduct (h : Splitting S) [HasBinaryBiproduct S.X₁ S.X₃] :
    S.X₂ ≅ S.X₁ ⊞ S.X₃ where
  hom := biprod.lift h.r S.g
  inv := biprod.desc S.f h.s
                   /-
                     C : Type u_1
                     D : Type u_2
                     inst✝⁴ : CategoryTheory.Category.{?u.128436, u_1} C
                     inst✝³ : CategoryTheory.Category.{?u.128440, u_2} D
                     inst✝² : CategoryTheory.Preadditive C
                     inst✝¹ : CategoryTheory.Preadditive D
                     S : CategoryTheory.ShortComplex C
                     h : S.Splitting
                     inst✝ : CategoryTheory.Limits.HasBinaryBiproduct S.X₁ S.X₃
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift h. …
                   -/
  hom_inv_id := by simp [h.id]
                   /-
                     🎉 no goals
                   -/


lemma isIso_f' (hS : S.Exact) (h : S.LeftHomologyData) [Mono S.f] :
    IsIso h.f' := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    h : S.LeftHomologyData
    inst✝ : CategoryTheory.Mono S.f
    ⊢ CategoryTheory.IsIso h.f'
  -/
  have := hS.epi_f' h
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    h : S.LeftHomologyData
    inst✝ : CategoryTheory.Mono S.f
    this : CategoryTheory.Epi h.f'
    ⊢ CategoryTheory.IsIso h.f'
  -/
  have := mono_of_mono_fac h.f'_i
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    h : S.LeftHomologyData
    inst✝ : CategoryTheory.Mono S.f
    this✝ : CategoryTheory.Epi h.f'
    this : CategoryTheory.Mono h.f'
    ⊢ CategoryTheory.IsIso h.f'
  -/
  exact isIso_of_mono_of_epi h.f'
  /-
    🎉 no goals
  -/


lemma isIso_toCycles (hS : S.Exact) [Mono S.f] [S.HasLeftHomology]:
    IsIso S.toCycles :=
  hS.isIso_f' _


lemma isIso_g' (hS : S.Exact) (h : S.RightHomologyData) [Epi S.g] :
    IsIso h.g' := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    h : S.RightHomologyData
    inst✝ : CategoryTheory.Epi S.g
    ⊢ CategoryTheory.IsIso h.g'
  -/
  have := hS.mono_g' h
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    h : S.RightHomologyData
    inst✝ : CategoryTheory.Epi S.g
    this : CategoryTheory.Mono h.g'
    ⊢ CategoryTheory.IsIso h.g'
  -/
  have := epi_of_epi_fac h.p_g'
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    h : S.RightHomologyData
    inst✝ : CategoryTheory.Epi S.g
    this✝ : CategoryTheory.Mono h.g'
    this : CategoryTheory.Epi h.g'
    ⊢ CategoryTheory.IsIso h.g'
  -/
  exact isIso_of_mono_of_epi h.g'
  /-
    🎉 no goals
  -/


lemma isIso_fromOpcycles (hS : S.Exact) [Epi S.g] [S.HasRightHomology] :
    IsIso S.fromOpcycles :=
  hS.isIso_g' _


/-- In a balanced category, if a short complex `S` is exact and `S.f` is a mono, then
`S.X₁` is the kernel of `S.g`. -/
noncomputable def fIsKernel (hS : S.Exact) [Mono S.f] : IsLimit (KernelFork.ofι S.f S.zero) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.135579, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.135583, u_2} D
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Preadditive D
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    inst✝ : CategoryTheory.Mono S.f
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
  -/
  have := hS.hasHomology
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.135579, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.135583, u_2} D
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Preadditive D
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    inst✝ : CategoryTheory.Mono S.f
    this : S.HasHomology
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
  -/
  have := hS.isIso_toCycles
  exact IsLimit.ofIsoLimit S.cyclesIsKernel
    (Fork.ext (asIso S.toCycles).symm (by simp))


lemma map_of_mono_of_preservesKernel (hS : S.Exact) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [(S.map F).HasHomology] (_ : Mono S.f)
    (_ : PreservesLimit (parallelPair S.g 0) F) :
    (S.map F).Exact :=
  exact_of_f_is_kernel _ (KernelFork.mapIsLimit _ hS.fIsKernel F)


/-- In a balanced category, if a short complex `S` is exact and `S.g` is an epi, then
`S.X₃` is the cokernel of `S.g`. -/
noncomputable def gIsCokernel (hS : S.Exact) [Epi S.g] :
    IsColimit (CokernelCofork.ofπ S.g S.zero) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.139980, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.139984, u_2} D
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Preadditive D
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    inst✝ : CategoryTheory.Epi S.g
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ S. …
  -/
  have := hS.hasHomology
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁵ : CategoryTheory.Category.{?u.139980, u_1} C
    inst✝⁴ : CategoryTheory.Category.{?u.139984, u_2} D
    inst✝³ : CategoryTheory.Preadditive C
    inst✝² : CategoryTheory.Preadditive D
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    hS : S.Exact
    inst✝ : CategoryTheory.Epi S.g
    this : S.HasHomology
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ S. …
  -/
  have := hS.isIso_fromOpcycles
  exact IsColimit.ofIsoColimit S.opcyclesIsCokernel
    (Cofork.ext (asIso S.fromOpcycles) (by simp))


lemma map_of_epi_of_preservesCokernel (hS : S.Exact) (F : C ⥤ D)
    [F.PreservesZeroMorphisms] [(S.map F).HasHomology] (_ : Epi S.g)
    (_ : PreservesColimit (parallelPair S.f 0) F) :
    (S.map F).Exact :=
  exact_of_g_is_cokernel _ (CokernelCofork.mapIsColimit _ hS.gIsCokernel F)


/-- If a short complex `S` in a balanced category is exact and such that `S.f` is a mono,
then a morphism `k : A ⟶ S.X₂` such that `k ≫ S.g = 0` lifts to a morphism `A ⟶ S.X₁`. -/
noncomputable def lift (hS : S.Exact) {A : C} (k : A ⟶ S.X₂) (hk : k ≫ S.g = 0) [Mono S.f] :
    A ⟶ S.X₁ := hS.fIsKernel.lift (KernelFork.ofι k hk)


@[reassoc (attr := simp)]
lemma lift_f (hS : S.Exact) {A : C} (k : A ⟶ S.X₂) (hk : k ≫ S.g = 0) [Mono S.f] :
    hS.lift k hk ≫ S.f = k :=
  Fork.IsLimit.lift_ι _


lemma lift' (hS : S.Exact) {A : C} (k : A ⟶ S.X₂) (hk : k ≫ S.g = 0) [Mono S.f] :
    ∃ (l : A ⟶ S.X₁), l ≫ S.f = k :=
                    /-
                      C : Type u_1
                      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
                      inst✝² : CategoryTheory.Preadditive C
                      S : CategoryTheory.ShortComplex C
                      inst✝¹ : CategoryTheory.Balanced C
                      hS : S.Exact
                      A : C
                      k : Quiver.Hom A S.X₂
                      hk : Eq (CategoryTheory.CategoryStruct.comp k S.g) 0
                      inst✝ : CategoryTheory.Mono S.f
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (hS.lift k hk) S.f) k
                    -/
  ⟨hS.lift k hk, by simp⟩
                    /-
                      🎉 no goals
                    -/


/-- If a short complex `S` in a balanced category is exact and such that `S.g` is an epi,
then a morphism `k : S.X₂ ⟶ A` such that `S.f ≫ k = 0` descends to a morphism `S.X₃ ⟶ A`. -/
noncomputable def desc (hS : S.Exact) {A : C} (k : S.X₂ ⟶ A) (hk : S.f ≫ k = 0) [Epi S.g] :
    S.X₃ ⟶ A := hS.gIsCokernel.desc (CokernelCofork.ofπ k hk)


@[reassoc (attr := simp)]
lemma g_desc (hS : S.Exact) {A : C} (k : S.X₂ ⟶ A) (hk : S.f ≫ k = 0) [Epi S.g] :
    S.g ≫ hS.desc k hk = k :=
  Cofork.IsColimit.π_desc (hS.gIsCokernel)


lemma desc' (hS : S.Exact) {A : C} (k : S.X₂ ⟶ A) (hk : S.f ≫ k = 0) [Epi S.g] :
    ∃ (l : S.X₃ ⟶ A), S.g ≫ l = k :=
                    /-
                      C : Type u_1
                      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
                      inst✝² : CategoryTheory.Preadditive C
                      S : CategoryTheory.ShortComplex C
                      inst✝¹ : CategoryTheory.Balanced C
                      hS : S.Exact
                      A : C
                      k : Quiver.Hom S.X₂ A
                      hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                      inst✝ : CategoryTheory.Epi S.g
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp S.g (hS.desc k hk)) k
                    -/
  ⟨hS.desc k hk, by simp⟩
                    /-
                      🎉 no goals
                    -/


lemma mono_τ₂_of_exact_of_mono {S₁ S₂ : ShortComplex C} (φ : S₁ ⟶ S₂)
    (h₁ : S₁.Exact) [Mono S₁.f] [Mono S₂.f] [Mono φ.τ₁] [Mono φ.τ₃] : Mono φ.τ₂ := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.Exact
    inst✝³ : CategoryTheory.Mono S₁.f
    inst✝² : CategoryTheory.Mono S₂.f
    inst✝¹ : CategoryTheory.Mono φ.τ₁
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.Mono φ.τ₂
  -/
  rw [mono_iff_cancel_zero]
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.Exact
    inst✝³ : CategoryTheory.Mono S₁.f
    inst✝² : CategoryTheory.Mono S₂.f
    inst✝¹ : CategoryTheory.Mono φ.τ₁
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ ∀ (P : C) (g : Quiver.Hom P S₁.X₂), Eq (CategoryTheory.CategoryStruct.comp g …
  -/
  intro A x₂ hx₂
  obtain ⟨x₁, hx₁⟩ : ∃ x₁, x₁ ≫ S₁.f = x₂ := ⟨_, h₁.lift_f x₂
    (by simp only [← cancel_mono φ.τ₃, assoc, zero_comp, ← φ.comm₂₃, reassoc_of% hx₂])⟩
  /-
    case intro
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.Exact
    inst✝³ : CategoryTheory.Mono S₁.f
    inst✝² : CategoryTheory.Mono S₂.f
    inst✝¹ : CategoryTheory.Mono φ.τ₁
    inst✝ : CategoryTheory.Mono φ.τ₃
    A : C
    x₂ : Quiver.Hom A S₁.X₂
    hx₂ : Eq (CategoryTheory.CategoryStruct.comp x₂ φ.τ₂) 0
    x₁ : Quiver.Hom A S₁.X₁
    hx₁ : Eq (CategoryTheory.CategoryStruct.comp x₁ S₁.f) x₂
    ⊢ Eq x₂ 0
  -/
  suffices x₁ = 0 by rw [← hx₁, this, zero_comp]
  simp only [← cancel_mono φ.τ₁, ← cancel_mono S₂.f, assoc, φ.comm₁₂, zero_comp,
    reassoc_of% hx₁, hx₂]


lemma epi_τ₂_of_exact_of_epi {S₁ S₂ : ShortComplex C} (φ : S₁ ⟶ S₂)
    (h₂ : S₂.Exact) [Epi S₁.g] [Epi S₂.g] [Epi φ.τ₁] [Epi φ.τ₃] : Epi φ.τ₂ := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₂ : S₂.Exact
    inst✝³ : CategoryTheory.Epi S₁.g
    inst✝² : CategoryTheory.Epi S₂.g
    inst✝¹ : CategoryTheory.Epi φ.τ₁
    inst✝ : CategoryTheory.Epi φ.τ₃
    ⊢ CategoryTheory.Epi φ.τ₂
  -/
  have : Mono S₁.op.f := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₂ : S₂.Exact
    inst✝³ : CategoryTheory.Epi S₁.g
    inst✝² : CategoryTheory.Epi S₂.g
    inst✝¹ : CategoryTheory.Epi φ.τ₁
    inst✝ : CategoryTheory.Epi φ.τ₃
    this : CategoryTheory.Mono S₁.op.f
    ⊢ CategoryTheory.Epi φ.τ₂
  -/
  have : Mono S₂.op.f := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₂ : S₂.Exact
    inst✝³ : CategoryTheory.Epi S₁.g
    inst✝² : CategoryTheory.Epi S₂.g
    inst✝¹ : CategoryTheory.Epi φ.τ₁
    inst✝ : CategoryTheory.Epi φ.τ₃
    this✝ : CategoryTheory.Mono S₁.op.f
    this : CategoryTheory.Mono S₂.op.f
    ⊢ CategoryTheory.Epi φ.τ₂
  -/
  have : Mono (opMap φ).τ₁ := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₂ : S₂.Exact
    inst✝³ : CategoryTheory.Epi S₁.g
    inst✝² : CategoryTheory.Epi S₂.g
    inst✝¹ : CategoryTheory.Epi φ.τ₁
    inst✝ : CategoryTheory.Epi φ.τ₃
    this✝¹ : CategoryTheory.Mono S₁.op.f
    this✝ : CategoryTheory.Mono S₂.op.f
    this : CategoryTheory.Mono (CategoryTheory.ShortComplex.opMap φ).τ₁
    ⊢ CategoryTheory.Epi φ.τ₂
  -/
  have : Mono (opMap φ).τ₃ := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₂ : S₂.Exact
    inst✝³ : CategoryTheory.Epi S₁.g
    inst✝² : CategoryTheory.Epi S₂.g
    inst✝¹ : CategoryTheory.Epi φ.τ₁
    inst✝ : CategoryTheory.Epi φ.τ₃
    this✝² : CategoryTheory.Mono S₁.op.f
    this✝¹ : CategoryTheory.Mono S₂.op.f
    this✝ : CategoryTheory.Mono (CategoryTheory.ShortComplex.opMap φ).τ₁
    this : CategoryTheory.Mono (CategoryTheory.ShortComplex.opMap φ).τ₃
    ⊢ CategoryTheory.Epi φ.τ₂
  -/
  have := mono_τ₂_of_exact_of_mono (opMap φ) h₂.op
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Balanced C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₂ : S₂.Exact
    inst✝³ : CategoryTheory.Epi S₁.g
    inst✝² : CategoryTheory.Epi S₂.g
    inst✝¹ : CategoryTheory.Epi φ.τ₁
    inst✝ : CategoryTheory.Epi φ.τ₃
    this✝³ : CategoryTheory.Mono S₁.op.f
    this✝² : CategoryTheory.Mono S₂.op.f
    this✝¹ : CategoryTheory.Mono (CategoryTheory.ShortComplex.opMap φ).τ₁
    this✝ : CategoryTheory.Mono (CategoryTheory.ShortComplex.opMap φ).τ₃
    this : CategoryTheory.Mono (CategoryTheory.ShortComplex.opMap φ).τ₂
    ⊢ CategoryTheory.Epi φ.τ₂
  -/
  exact unop_epi_of_mono (opMap φ).τ₂
  /-
    🎉 no goals
  -/


lemma exact_and_mono_f_iff_f_is_kernel [S.HasHomology] :
    S.Exact ∧ Mono S.f ↔ Nonempty (IsLimit (KernelFork.ofι S.f S.zero)) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    inst✝ : S.HasHomology
    ⊢ Iff (And S.Exact (CategoryTheory.Mono S.f)) (Nonempty (CategoryTheory.Limits …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : CategoryTheory.Balanced C
      inst✝ : S.HasHomology
      ⊢ And S.Exact (CategoryTheory.Mono S.f) → Nonempty (CategoryTheory.Limits.IsLi …
    -/
  · intro ⟨hS, _⟩
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : CategoryTheory.Balanced C
      inst✝ : S.HasHomology
      hS : S.Exact
      right✝ : CategoryTheory.Mono S.f
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.of …
    -/
    exact ⟨hS.fIsKernel⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : CategoryTheory.Balanced C
      inst✝ : S.HasHomology
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.of …
    -/
  · intro ⟨hS⟩
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : CategoryTheory.Balanced C
      inst✝ : S.HasHomology
      hS : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι S.f ⋯)
      ⊢ And S.Exact (CategoryTheory.Mono S.f)
    -/
    exact ⟨S.exact_of_f_is_kernel hS, mono_of_isLimit_fork hS⟩
    /-
      🎉 no goals
    -/


lemma exact_and_epi_g_iff_g_is_cokernel [S.HasHomology] :
    S.Exact ∧ Epi S.g ↔ Nonempty (IsColimit (CokernelCofork.ofπ S.g S.zero)) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : CategoryTheory.Balanced C
    inst✝ : S.HasHomology
    ⊢ Iff (And S.Exact (CategoryTheory.Epi S.g)) (Nonempty (CategoryTheory.Limits. …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : CategoryTheory.Balanced C
      inst✝ : S.HasHomology
      ⊢ And S.Exact (CategoryTheory.Epi S.g) → Nonempty (CategoryTheory.Limits.IsCol …
    -/
  · intro ⟨hS, _⟩
    /-
      case mp
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : CategoryTheory.Balanced C
      inst✝ : S.HasHomology
      hS : S.Exact
      right✝ : CategoryTheory.Epi S.g
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCof …
    -/
    exact ⟨hS.gIsCokernel⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : CategoryTheory.Balanced C
      inst✝ : S.HasHomology
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCof …
    -/
  · intro ⟨hS⟩
    /-
      case mpr
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      S : CategoryTheory.ShortComplex C
      inst✝¹ : CategoryTheory.Balanced C
      inst✝ : S.HasHomology
      hS : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
      ⊢ And S.Exact (CategoryTheory.Epi S.g)
    -/
    exact ⟨S.exact_of_g_is_cokernel hS, epi_of_isColimit_cofork hS⟩
    /-
      🎉 no goals
    -/


/-- Given a morphism of short complexes `φ : S₁ ⟶ S₂` in an abelian category, if `S₁.f`
and `S₁.g` are zero (e.g. when `S₁` is of the form `0 ⟶ S₁.X₂ ⟶ 0`) and `S₂.f = 0`
(e.g when `S₂` is of the form `0 ⟶ S₂.X₂ ⟶ S₂.X₃`), then `φ` is a quasi-isomorphism iff
the obvious short complex `S₁.X₂ ⟶ S₂.X₂ ⟶ S₂.X₃` is exact and `φ.τ₂` is a mono). -/
lemma quasiIso_iff_of_zeros {S₁ S₂ : ShortComplex C} (φ : S₁ ⟶ S₂)
    (hf₁ : S₁.f = 0) (hg₁ : S₁.g = 0) (hf₂ : S₂.f = 0) :
    QuasiIso φ ↔
                                     /-
                                       C : Type u_1
                                       D : Type u_2
                                       inst✝² : CategoryTheory.Category.{?u.159880, u_1} C
                                       inst✝¹ : CategoryTheory.Category.{?u.159884, u_2} D
                                       inst✝ : CategoryTheory.Abelian C
                                       S₁ S₂ : CategoryTheory.ShortComplex C
                                       φ : Quiver.Hom S₁ S₂
                                       hf₁ : Eq S₁.f 0
                                       hg₁ : Eq S₁.g 0
                                       hf₂ : Eq S₂.f 0
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
                                     -/
      (ShortComplex.mk φ.τ₂ S₂.g (by rw [φ.comm₂₃, hg₁, zero_comp])).Exact ∧ Mono φ.τ₂ := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hf₁ : Eq S₁.f 0
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (And (CategoryTheory.ShortCompl …
  -/
  have w : φ.τ₂ ≫ S₂.g = 0 := by rw [φ.comm₂₃, hg₁, zero_comp]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hf₁ : Eq S₁.f 0
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (And (CategoryTheory.ShortCompl …
  -/
  rw [quasiIso_iff_isIso_liftCycles φ hf₁ hg₁ hf₂]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hf₁ : Eq S₁.f 0
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
    ⊢ Iff (CategoryTheory.IsIso (S₂.liftCycles φ.τ₂ ⋯)) (And (CategoryTheory.Short …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hf₁ : Eq S₁.f 0
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
      ⊢ CategoryTheory.IsIso (S₂.liftCycles φ.τ₂ ⋯) → And (CategoryTheory.ShortCompl …
    -/
  · intro h
    have : Mono φ.τ₂ := by
      rw [← S₂.liftCycles_i φ.τ₂ w]
      apply mono_comp
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hf₁ : Eq S₁.f 0
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
      h : CategoryTheory.IsIso (S₂.liftCycles φ.τ₂ ⋯)
      this : CategoryTheory.Mono φ.τ₂
      ⊢ And (CategoryTheory.ShortComplex.mk φ.τ₂ S₂.g ⋯).Exact (CategoryTheory.Mono  …
    -/
    refine ⟨?_, this⟩
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hf₁ : Eq S₁.f 0
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
      h : CategoryTheory.IsIso (S₂.liftCycles φ.τ₂ ⋯)
      this : CategoryTheory.Mono φ.τ₂
      ⊢ (CategoryTheory.ShortComplex.mk φ.τ₂ S₂.g ⋯).Exact
    -/
    apply exact_of_f_is_kernel
    exact IsLimit.ofIsoLimit S₂.cyclesIsKernel
      (Fork.ext (asIso (S₂.liftCycles φ.τ₂ w)).symm (by simp))
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hf₁ : Eq S₁.f 0
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
      ⊢ And (CategoryTheory.ShortComplex.mk φ.τ₂ S₂.g ⋯).Exact (CategoryTheory.Mono  …
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hf₁ : Eq S₁.f 0
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
      h₁ : (CategoryTheory.ShortComplex.mk φ.τ₂ S₂.g ⋯).Exact
      h₂ : CategoryTheory.Mono φ.τ₂
      ⊢ CategoryTheory.IsIso (S₂.liftCycles φ.τ₂ ⋯)
    -/
    refine ⟨⟨h₁.lift S₂.iCycles (by simp), ?_, ?_⟩⟩
      /-
        case mpr.intro.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        S₁ S₂ : CategoryTheory.ShortComplex C
        φ : Quiver.Hom S₁ S₂
        hf₁ : Eq S₁.f 0
        hg₁ : Eq S₁.g 0
        hf₂ : Eq S₂.f 0
        w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
        h₁ : (CategoryTheory.ShortComplex.mk φ.τ₂ S₂.g ⋯).Exact
        h₂ : CategoryTheory.Mono φ.τ₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S₂.liftCycles φ.τ₂ ⋯) (h₁.lift S₂.iC …
      -/
    · rw [← cancel_mono φ.τ₂, assoc, h₁.lift_f, liftCycles_i, id_comp]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
        inst✝ : CategoryTheory.Abelian C
        S₁ S₂ : CategoryTheory.ShortComplex C
        φ : Quiver.Hom S₁ S₂
        hf₁ : Eq S₁.f 0
        hg₁ : Eq S₁.g 0
        hf₂ : Eq S₂.f 0
        w : Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ S₂.g) 0
        h₁ : (CategoryTheory.ShortComplex.mk φ.τ₂ S₂.g ⋯).Exact
        h₂ : CategoryTheory.Mono φ.τ₂
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (h₁.lift S₂.iCycles ⋯) (S₂.liftCycles …
      -/
    · rw [← cancel_mono S₂.iCycles, assoc, liftCycles_i, h₁.lift_f, id_comp]
      /-
        🎉 no goals
      -/


/-- Given a morphism of short complexes `φ : S₁ ⟶ S₂` in an abelian category, if `S₁.g = 0`
(e.g when `S₁` is of the form `S₁.X₁ ⟶ S₁.X₂ ⟶ 0`) and both `S₂.f` and `S₂.g` are zero
(e.g when `S₂` is of the form `0 ⟶ S₂.X₂ ⟶ 0`), then `φ` is a quasi-isomorphism iff
the obvious short complex `S₁.X₂ ⟶ S₁.X₂ ⟶ S₂.X₂` is exact and `φ.τ₂` is an epi). -/
lemma quasiIso_iff_of_zeros' {S₁ S₂ : ShortComplex C} (φ : S₁ ⟶ S₂)
    (hg₁ : S₁.g = 0) (hf₂ : S₂.f = 0) (hg₂ : S₂.g = 0) :
    QuasiIso φ ↔
                                     /-
                                       C : Type u_1
                                       D : Type u_2
                                       inst✝² : CategoryTheory.Category.{?u.211317, u_1} C
                                       inst✝¹ : CategoryTheory.Category.{?u.211321, u_2} D
                                       inst✝ : CategoryTheory.Abelian C
                                       S₁ S₂ : CategoryTheory.ShortComplex C
                                       φ : Quiver.Hom S₁ S₂
                                       hg₁ : Eq S₁.g 0
                                       hf₂ : Eq S₂.f 0
                                       hg₂ : Eq S₂.g 0
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp S₁.f φ.τ₂) 0
                                     -/
      (ShortComplex.mk S₁.f φ.τ₂ (by rw [← φ.comm₁₂, hf₂, comp_zero])).Exact ∧ Epi φ.τ₂ := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    hg₂ : Eq S₂.g 0
    ⊢ Iff (CategoryTheory.ShortComplex.QuasiIso φ) (And (CategoryTheory.ShortCompl …
  -/
  rw [← quasiIso_opMap_iff, quasiIso_iff_of_zeros]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    hg₂ : Eq S₂.g 0
    ⊢ Iff (And (CategoryTheory.ShortComplex.mk (CategoryTheory.ShortComplex.opMap  …
  -/
  rotate_left
    /-
      case hf₁
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      hg₂ : Eq S₂.g 0
      ⊢ Eq S₂.op.f 0
    -/
  · dsimp
    /-
      case hf₁
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      hg₂ : Eq S₂.g 0
      ⊢ Eq S₂.g.op 0
    -/
    rw [hg₂, op_zero]
    /-
      🎉 no goals
    -/
    /-
      case hg₁
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      hg₂ : Eq S₂.g 0
      ⊢ Eq S₂.op.g 0
    -/
  · dsimp
    /-
      case hg₁
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      hg₂ : Eq S₂.g 0
      ⊢ Eq S₂.f.op 0
    -/
    rw [hf₂, op_zero]
    /-
      🎉 no goals
    -/
    /-
      case hf₂
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      hg₂ : Eq S₂.g 0
      ⊢ Eq S₁.op.f 0
    -/
  · dsimp
    /-
      case hf₂
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      hg₁ : Eq S₁.g 0
      hf₂ : Eq S₂.f 0
      hg₂ : Eq S₂.g 0
      ⊢ Eq S₁.g.op 0
    -/
    rw [hg₁, op_zero]
    /-
      🎉 no goals
    -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    hg₂ : Eq S₂.g 0
    ⊢ Iff (And (CategoryTheory.ShortComplex.mk (CategoryTheory.ShortComplex.opMap  …
  -/
  rw [← exact_unop_iff]
  have : Mono φ.τ₂.op ↔ Epi φ.τ₂ :=
    ⟨fun _ => unop_epi_of_mono φ.τ₂.op, fun _ => op_mono_of_epi _⟩
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    hg₁ : Eq S₁.g 0
    hf₂ : Eq S₂.f 0
    hg₂ : Eq S₂.g 0
    this : Iff (CategoryTheory.Mono φ.τ₂.op) (CategoryTheory.Epi φ.τ₂)
    ⊢ Iff (And (CategoryTheory.ShortComplex.mk (CategoryTheory.ShortComplex.opMap  …
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- If `S` is an exact short complex and `f : S.X₂ ⟶ J` is a morphism to an injective object `J`
such that `S.f ≫ f = 0`, this is a morphism `φ : S.X₃ ⟶ J` such that `S.g ≫ φ = f`. -/
noncomputable def Exact.descToInjective
    (hS : S.Exact) {J : C} (f : S.X₂ ⟶ J) [Injective J] (hf : S.f ≫ f = 0) :
    S.X₃ ⟶ J := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.218463, u_1} C
    inst✝² : CategoryTheory.Category.{?u.218467, u_2} D
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    J : C
    f : Quiver.Hom S.X₂ J
    inst✝ : CategoryTheory.Injective J
    hf : Eq (CategoryTheory.CategoryStruct.comp S.f f) 0
    ⊢ Quiver.Hom S.X₃ J
  -/
  have := hS.mono_fromOpcycles
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.218463, u_1} C
    inst✝² : CategoryTheory.Category.{?u.218467, u_2} D
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    J : C
    f : Quiver.Hom S.X₂ J
    inst✝ : CategoryTheory.Injective J
    hf : Eq (CategoryTheory.CategoryStruct.comp S.f f) 0
    this : CategoryTheory.Mono S.fromOpcycles
    ⊢ Quiver.Hom S.X₃ J
  -/
  exact Injective.factorThru (S.descOpcycles f hf) S.fromOpcycles
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp, nolint unusedHavesSuffices)]
lemma Exact.comp_descToInjective
    (hS : S.Exact) {J : C} (f : S.X₂ ⟶ J) [Injective J] (hf : S.f ≫ f = 0) :
    S.g ≫ hS.descToInjective f hf = f := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    J : C
    f : Quiver.Hom S.X₂ J
    inst✝ : CategoryTheory.Injective J
    hf : Eq (CategoryTheory.CategoryStruct.comp S.f f) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.g (hS.descToInjective f hf)) f
  -/
  have := hS.mono_fromOpcycles
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    J : C
    f : Quiver.Hom S.X₂ J
    inst✝ : CategoryTheory.Injective J
    hf : Eq (CategoryTheory.CategoryStruct.comp S.f f) 0
    this : CategoryTheory.Mono S.fromOpcycles
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.g (hS.descToInjective f hf)) f
  -/
  dsimp [descToInjective]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    J : C
    f : Quiver.Hom S.X₂ J
    inst✝ : CategoryTheory.Injective J
    hf : Eq (CategoryTheory.CategoryStruct.comp S.f f) 0
    this : CategoryTheory.Mono S.fromOpcycles
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.g (CategoryTheory.Injective.factorT …
  -/
  simp only [← p_fromOpcycles, assoc, Injective.comp_factorThru, p_descOpcycles]
  /-
    🎉 no goals
  -/


/-- If `S` is an exact short complex and `f : P ⟶ S.X₂` is a morphism from a projective object `P`
such that `f ≫ S.g = 0`, this is a morphism `φ : P ⟶ S.X₁` such that `φ ≫ S.f = f`. -/
noncomputable def Exact.liftFromProjective
    (hS : S.Exact) {P : C} (f : P ⟶ S.X₂) [Projective P] (hf : f ≫ S.g = 0) :
    P ⟶ S.X₁ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.221352, u_1} C
    inst✝² : CategoryTheory.Category.{?u.221356, u_2} D
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    P : C
    f : Quiver.Hom P S.X₂
    inst✝ : CategoryTheory.Projective P
    hf : Eq (CategoryTheory.CategoryStruct.comp f S.g) 0
    ⊢ Quiver.Hom P S.X₁
  -/
  have := hS.epi_toCycles
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{?u.221352, u_1} C
    inst✝² : CategoryTheory.Category.{?u.221356, u_2} D
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    P : C
    f : Quiver.Hom P S.X₂
    inst✝ : CategoryTheory.Projective P
    hf : Eq (CategoryTheory.CategoryStruct.comp f S.g) 0
    this : CategoryTheory.Epi S.toCycles
    ⊢ Quiver.Hom P S.X₁
  -/
  exact Projective.factorThru (S.liftCycles f hf) S.toCycles
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp, nolint unusedHavesSuffices)]
lemma Exact.liftFromProjective_comp
    (hS : S.Exact) {P : C} (f : P ⟶ S.X₂) [Projective P] (hf : f ≫ S.g = 0) :
    hS.liftFromProjective f hf ≫ S.f = f := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    P : C
    f : Quiver.Hom P S.X₂
    inst✝ : CategoryTheory.Projective P
    hf : Eq (CategoryTheory.CategoryStruct.comp f S.g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (hS.liftFromProjective f hf) S.f) f
  -/
  have := hS.epi_toCycles
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    P : C
    f : Quiver.Hom P S.X₂
    inst✝ : CategoryTheory.Projective P
    hf : Eq (CategoryTheory.CategoryStruct.comp f S.g) 0
    this : CategoryTheory.Epi S.toCycles
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (hS.liftFromProjective f hf) S.f) f
  -/
  dsimp [liftFromProjective]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    hS : S.Exact
    P : C
    f : Quiver.Hom P S.X₂
    inst✝ : CategoryTheory.Projective P
    hf : Eq (CategoryTheory.CategoryStruct.comp f S.g) 0
    this : CategoryTheory.Epi S.toCycles
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Projective.factorThru …
  -/
  rw [← toCycles_i, Projective.factorThru_comp_assoc, liftCycles_i]
  /-
    🎉 no goals
  -/



@[deprecated (since := "2024-07-09")] alias _root_.CategoryTheory.Exact.lift :=
  Exact.liftFromProjective

@[deprecated (since := "2024-07-09")] alias _root_.CategoryTheory.Exact.lift_comp :=
  Exact.liftFromProjective_comp

@[deprecated (since := "2024-07-09")] alias _root_.CategoryTheory.Injective.Exact.desc :=
  Exact.descToInjective

@[deprecated (since := "2024-07-09")] alias _root_.CategoryTheory.Injective.Exact.comp_desc :=
  Exact.comp_descToInjective


instance : F.PreservesMonomorphisms where
  preserves {X Y} f hf := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : CategoryTheory.Preadditive D
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.Limits.HasZeroObject D
      inst✝¹ : F.PreservesZeroMorphisms
      inst✝ : F.PreservesHomology
      X Y : C
      f : Quiver.Hom X Y
      hf : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono (F.map f)
    -/
    let S := ShortComplex.mk (0 : X ⟶ X) f zero_comp
    exact ((S.map F).exact_iff_mono (by simp [S])).1
      (((S.exact_iff_mono rfl).2 hf).map F)



instance : F.PreservesEpimorphisms where
  preserves {X Y} f hf := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
      F : CategoryTheory.Functor C D
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : CategoryTheory.Preadditive D
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.Limits.HasZeroObject D
      inst✝¹ : F.PreservesZeroMorphisms
      inst✝ : F.PreservesHomology
      X Y : C
      f : Quiver.Hom X Y
      hf : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi (F.map f)
    -/
    let S := ShortComplex.mk f (0 : Y ⟶ Y) comp_zero
    exact ((S.map F).exact_iff_epi (by simp [S])).1
      (((S.exact_iff_epi rfl).2 hf).map F)



/-- This is the splitting of a short complex `S` in a balanced category induced by
a section of the morphism `S.g : S.X₂ ⟶ S.X₃` -/
noncomputable def ofExactOfSection (S : ShortComplex C) (hS : S.Exact) (s : S.X₃ ⟶ S.X₂)
    (s_g : s ≫ S.g = 𝟙 S.X₃) (hf : Mono S.f) :
    S.Splitting where
                                      /-
                                        C : Type u_1
                                        D : Type u_2
                                        inst✝³ : CategoryTheory.Category.{?u.228106, u_1} C
                                        inst✝² : CategoryTheory.Category.{?u.228110, u_2} D
                                        inst✝¹ : CategoryTheory.Preadditive C
                                        inst✝ : CategoryTheory.Balanced C
                                        S : CategoryTheory.ShortComplex C
                                        hS : S.Exact
                                        s : Quiver.Hom S.X₃ S.X₂
                                        s_g : Eq (CategoryTheory.CategoryStruct.comp s S.g) (CategoryTheory.CategorySt …
                                        hf : CategoryTheory.Mono S.f
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (CategoryTheory.CategorySt …
                                      -/
  r := hS.lift (𝟙 S.X₂ - S.g ≫ s) (by simp [s_g])
                                      /-
                                        🎉 no goals
                                      -/
  s := s
  f_r := by rw [← cancel_mono S.f, assoc, Exact.lift_f, comp_sub, comp_id,
    zero_assoc, zero_comp, sub_zero, id_comp]
  s_g := s_g


/-- This is the splitting of a short complex `S` in a balanced category induced by
a retraction of the morphism `S.f : S.X₁ ⟶ S.X₂` -/
noncomputable def ofExactOfRetraction (S : ShortComplex C) (hS : S.Exact) (r : S.X₂ ⟶ S.X₁)
    (f_r : S.f ≫ r = 𝟙 S.X₁) (hg : Epi S.g) :
    S.Splitting where
  r := r
                                      /-
                                        C : Type u_1
                                        D : Type u_2
                                        inst✝³ : CategoryTheory.Category.{?u.232448, u_1} C
                                        inst✝² : CategoryTheory.Category.{?u.232452, u_2} D
                                        inst✝¹ : CategoryTheory.Preadditive C
                                        inst✝ : CategoryTheory.Balanced C
                                        S : CategoryTheory.ShortComplex C
                                        hS : S.Exact
                                        r : Quiver.Hom S.X₂ S.X₁
                                        f_r : Eq (CategoryTheory.CategoryStruct.comp S.f r) (CategoryTheory.CategorySt …
                                        hg : CategoryTheory.Epi S.g
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f (HSub.hSub (CategoryTheory.Catego …
                                      -/
  s := hS.desc (𝟙 S.X₂ - r ≫ S.f) (by simp [reassoc_of% f_r])
                                      /-
                                        🎉 no goals
                                      -/
  f_r := f_r
  s_g := by
    rw [← cancel_epi S.g, Exact.g_desc_assoc, sub_comp, id_comp, assoc, zero,
      comp_zero, sub_zero, comp_id]


