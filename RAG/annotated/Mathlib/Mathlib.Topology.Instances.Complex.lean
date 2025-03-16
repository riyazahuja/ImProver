/-- The only closed subfields of `ℂ` are `ℝ` and `ℂ`. -/
theorem Complex.subfield_eq_of_closed {K : Subfield ℂ} (hc : IsClosed (K : Set ℂ)) :
    K = ofRealHom.fieldRange ∨ K = ⊤ := by
  suffices range (ofReal : ℝ → ℂ) ⊆ K by
    rw [range_subset_iff, ← coe_algebraMap] at this
    have :=
      (Subalgebra.isSimpleOrder_of_finrank finrank_real_complex).eq_bot_or_eq_top
        (Subfield.toIntermediateField K this).toSubalgebra
    simp_rw [← SetLike.coe_set_eq, IntermediateField.coe_toSubalgebra] at this ⊢
    exact this
  suffices range (ofReal : ℝ → ℂ) ⊆ closure (Set.range ((ofReal : ℝ → ℂ) ∘ ((↑) : ℚ → ℝ))) by
    refine subset_trans this ?_
    rw [← IsClosed.closure_eq hc]
    apply closure_mono
    rintro _ ⟨_, rfl⟩
    simp only [Function.comp_apply, ofReal_ratCast, SetLike.mem_coe, SubfieldClass.ratCast_mem]
  /-
    K : Subfield Complex
    hc : IsClosed ↑K
    ⊢ HasSubset.Subset (Set.range Complex.ofReal) (closure (Set.range (Function.co …
  -/
  nth_rw 1 [range_comp]
  /-
    K : Subfield Complex
    hc : IsClosed ↑K
    ⊢ HasSubset.Subset (Set.range Complex.ofReal) (closure (Set.image Complex.ofRe …
  -/
  refine subset_trans ?_ (image_closure_subset_closure_image continuous_ofReal)
  /-
    K : Subfield Complex
    hc : IsClosed ↑K
    ⊢ HasSubset.Subset (Set.range Complex.ofReal) (Set.image Complex.ofReal (closu …
  -/
  rw [DenseRange.closure_range Rat.isDenseEmbedding_coe_real.dense]
  /-
    K : Subfield Complex
    hc : IsClosed ↑K
    ⊢ HasSubset.Subset (Set.range Complex.ofReal) (Set.image Complex.ofReal Set.un …
  -/
  simp only [image_univ]
  /-
    K : Subfield Complex
    hc : IsClosed ↑K
    ⊢ HasSubset.Subset (Set.range Complex.ofReal) (Set.range Complex.ofReal)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Let `K` a subfield of `ℂ` and let `ψ : K →+* ℂ` a ring homomorphism. Assume that `ψ` is uniform
continuous, then `ψ` is either the inclusion map or the composition of the inclusion map with the
complex conjugation. -/
theorem Complex.uniformContinuous_ringHom_eq_id_or_conj (K : Subfield ℂ) {ψ : K →+* ℂ}
    (hc : UniformContinuous ψ) : ψ.toFun = K.subtype ∨ ψ.toFun = conj ∘ K.subtype := by
  /-
    K : Subfield Complex
    ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
    hc : UniformContinuous ⇑ψ
    ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
  -/
  letI : TopologicalDivisionRing ℂ := TopologicalDivisionRing.mk
  letI : TopologicalRing K.topologicalClosure :=
    Subring.instTopologicalRing K.topologicalClosure.toSubring
  /-
    K : Subfield Complex
    ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
    hc : UniformContinuous ⇑ψ
    this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
    this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
    ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
  -/
  set ι : K → K.topologicalClosure := ⇑(Subfield.inclusion K.le_topologicalClosure)
  have ui : IsUniformInducing ι :=
    ⟨by
      rw [uniformity_subtype, uniformity_subtype, Filter.comap_comap]
      congr ⟩
  /-
    K : Subfield Complex
    ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
    hc : UniformContinuous ⇑ψ
    this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
    this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
    ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
    ui : IsUniformInducing ι
    ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
  -/
  let di := ui.isDenseInducing (?_ : DenseRange ι)
  · -- extψ : closure(K) →+* ℂ is the extension of ψ : K →+* ℂ
    /-
      case refine_2
      K : Subfield Complex
      ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
      hc : UniformContinuous ⇑ψ
      this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
      this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
      ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
      ui : IsUniformInducing ι
      di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
      ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
    -/
    let extψ := IsDenseInducing.extendRingHom ui di.dense hc
    /-
      case refine_2
      K : Subfield Complex
      ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
      hc : UniformContinuous ⇑ψ
      this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
      this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
      ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
      ui : IsUniformInducing ι
      di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
      extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
      ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
    -/
    haveI hψ := (uniformContinuous_uniformly_extend ui di.dense hc).continuous
    /-
      case refine_2
      K : Subfield Complex
      ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
      hc : UniformContinuous ⇑ψ
      this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
      this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
      ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
      ui : IsUniformInducing ι
      di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
      extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
      hψ : Continuous (⋯.extend ⇑ψ)
      ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
    -/
    cases' Complex.subfield_eq_of_closed (Subfield.isClosed_topologicalClosure K) with h h
      /-
        case refine_2.inl
        K : Subfield Complex
        ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
        hc : UniformContinuous ⇑ψ
        this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
        this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
        ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
        ui : IsUniformInducing ι
        di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
        extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
        hψ : Continuous (⋯.extend ⇑ψ)
        h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
        ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
      -/
    · left
      /-
        case refine_2.inl.h
        K : Subfield Complex
        ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
        hc : UniformContinuous ⇑ψ
        this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
        this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
        ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
        ui : IsUniformInducing ι
        di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
        extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
        hψ : Continuous (⋯.extend ⇑ψ)
        h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
        ⊢ Eq (↑↑ψ).toFun ⇑K.subtype
      -/
      let j := RingEquiv.subfieldCongr h
      -- ψ₁ is the continuous ring hom `ℝ →+* ℂ` constructed from `j : closure (K) ≃+* ℝ`
      -- and `extψ : closure (K) →+* ℂ`
      /-
        case refine_2.inl.h
        K : Subfield Complex
        ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
        hc : UniformContinuous ⇑ψ
        this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
        this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
        ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
        ui : IsUniformInducing ι
        di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
        extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
        hψ : Continuous (⋯.extend ⇑ψ)
        h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
        j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
        ⊢ Eq (↑↑ψ).toFun ⇑K.subtype
      -/
      let ψ₁ := RingHom.comp extψ (RingHom.comp j.symm.toRingHom ofRealHom.rangeRestrict)
      -- Porting note: was `by continuity!` and was used inline
      have hψ₁ : Continuous ψ₁ := by
        simpa only [RingHom.coe_comp] using hψ.comp ((continuous_algebraMap ℝ ℂ).subtype_mk _)
      /-
        case refine_2.inl.h
        K : Subfield Complex
        ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
        hc : UniformContinuous ⇑ψ
        this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
        this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
        ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
        ui : IsUniformInducing ι
        di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
        extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
        hψ : Continuous (⋯.extend ⇑ψ)
        h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
        j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
        ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
        hψ₁ : Continuous ⇑ψ₁
        ⊢ Eq (↑↑ψ).toFun ⇑K.subtype
      -/
      ext1 x
      /-
        case refine_2.inl.h.h
        K : Subfield Complex
        ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
        hc : UniformContinuous ⇑ψ
        this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
        this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
        ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
        ui : IsUniformInducing ι
        di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
        extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
        hψ : Continuous (⋯.extend ⇑ψ)
        h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
        j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
        ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
        hψ₁ : Continuous ⇑ψ₁
        x : Subtype fun x => Membership.mem K x
        ⊢ Eq ((↑↑ψ).toFun x) (K.subtype x)
      -/
      rsuffices ⟨r, hr⟩ : ∃ r : ℝ, ofRealHom.rangeRestrict r = j (ι x)
      · have :=
          RingHom.congr_fun (ringHom_eq_ofReal_of_continuous hψ₁) r
        -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
        /-
          case refine_2.inl.h.h.intro
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝¹ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this✝ : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure  …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
          j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
          ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
          hψ₁ : Continuous ⇑ψ₁
          x : Subtype fun x => Membership.mem K x
          r : Real
          hr : Eq (Complex.ofRealHom.rangeRestrict r) (j (ι x))
          this : Eq (ψ₁ r) (Complex.ofRealHom r)
          ⊢ Eq ((↑↑ψ).toFun x) (K.subtype x)
        -/
        erw [RingHom.comp_apply, RingHom.comp_apply, hr, RingEquiv.toRingHom_eq_coe] at this
        /-
          case refine_2.inl.h.h.intro
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝¹ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this✝ : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure  …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
          j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
          ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
          hψ₁ : Continuous ⇑ψ₁
          x : Subtype fun x => Membership.mem K x
          r : Real
          hr : Eq (Complex.ofRealHom.rangeRestrict r) (j (ι x))
          this : Eq (extψ (↑j.symm (j (ι x)))) (Complex.ofRealHom r)
          ⊢ Eq ((↑↑ψ).toFun x) (K.subtype x)
        -/
        convert this using 1
          /-
            case h.e'_2
            K : Subfield Complex
            ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
            hc : UniformContinuous ⇑ψ
            this✝¹ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
            this✝ : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure  …
            ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
            ui : IsUniformInducing ι
            di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
            extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
            hψ : Continuous (⋯.extend ⇑ψ)
            h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
            j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
            ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
            hψ₁ : Continuous ⇑ψ₁
            x : Subtype fun x => Membership.mem K x
            r : Real
            hr : Eq (Complex.ofRealHom.rangeRestrict r) (j (ι x))
            this : Eq (extψ (↑j.symm (j (ι x)))) (Complex.ofRealHom r)
            ⊢ Eq ((↑↑ψ).toFun x) (extψ (↑j.symm (j (ι x))))
          -/
        · exact (IsDenseInducing.extend_eq di hc.continuous _).symm
          /-
            🎉 no goals
          -/
          /-
            case h.e'_3
            K : Subfield Complex
            ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
            hc : UniformContinuous ⇑ψ
            this✝¹ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
            this✝ : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure  …
            ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
            ui : IsUniformInducing ι
            di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
            extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
            hψ : Continuous (⋯.extend ⇑ψ)
            h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
            j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
            ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
            hψ₁ : Continuous ⇑ψ₁
            x : Subtype fun x => Membership.mem K x
            r : Real
            hr : Eq (Complex.ofRealHom.rangeRestrict r) (j (ι x))
            this : Eq (extψ (↑j.symm (j (ι x)))) (Complex.ofRealHom r)
            ⊢ Eq (K.subtype x) (Complex.ofRealHom r)
          -/
        · rw [← ofRealHom.coe_rangeRestrict, hr]
          /-
            case h.e'_3
            K : Subfield Complex
            ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
            hc : UniformContinuous ⇑ψ
            this✝¹ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
            this✝ : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure  …
            ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
            ui : IsUniformInducing ι
            di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
            extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
            hψ : Continuous (⋯.extend ⇑ψ)
            h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
            j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
            ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
            hψ₁ : Continuous ⇑ψ₁
            x : Subtype fun x => Membership.mem K x
            r : Real
            hr : Eq (Complex.ofRealHom.rangeRestrict r) (j (ι x))
            this : Eq (extψ (↑j.symm (j (ι x)))) (Complex.ofRealHom r)
            ⊢ Eq (K.subtype x) ↑(j (ι x))
          -/
          rfl
          /-
            🎉 no goals
          -/
      /-
        K : Subfield Complex
        ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
        hc : UniformContinuous ⇑ψ
        this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
        this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
        ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
        ui : IsUniformInducing ι
        di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
        extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
        hψ : Continuous (⋯.extend ⇑ψ)
        h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
        j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
        ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
        hψ₁ : Continuous ⇑ψ₁
        x : Subtype fun x => Membership.mem K x
        ⊢ Exists fun r => Eq (Complex.ofRealHom.rangeRestrict r) (j (ι x))
      -/
      obtain ⟨r, hr⟩ := SetLike.coe_mem (j (ι x))
      /-
        case intro
        K : Subfield Complex
        ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
        hc : UniformContinuous ⇑ψ
        this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
        this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
        ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
        ui : IsUniformInducing ι
        di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
        extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
        hψ : Continuous (⋯.extend ⇑ψ)
        h : Eq K.topologicalClosure Complex.ofRealHom.fieldRange
        j : RingEquiv (Subtype fun x => Membership.mem K.topologicalClosure x) (Subtyp …
        ψ₁ : RingHom Real Complex := extψ.comp (j.symm.toRingHom.comp Complex.ofRealHo …
        hψ₁ : Continuous ⇑ψ₁
        x : Subtype fun x => Membership.mem K x
        r : Real
        hr : Eq (Complex.ofRealHom r) ↑(j (ι x))
        ⊢ Exists fun r => Eq (Complex.ofRealHom.rangeRestrict r) (j (ι x))
      -/
      exact ⟨r, Subtype.ext hr⟩
      /-
        🎉 no goals
      -/
    · -- ψ₁ is the continuous ring hom `ℂ →+* ℂ` constructed from `closure (K) ≃+* ℂ`
      -- and `extψ : closure (K) →+* ℂ`
      let ψ₁ :=
        RingHom.comp extψ
          (RingHom.comp (RingEquiv.subfieldCongr h).symm.toRingHom
            (@Subfield.topEquiv ℂ _).symm.toRingHom)
      -- Porting note: was `by continuity!` and was used inline
      have hψ₁ : Continuous ψ₁ := by
        simpa only [RingHom.coe_comp] using hψ.comp (continuous_id.subtype_mk _)
      /-
        case refine_2.inr
        K : Subfield Complex
        ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
        hc : UniformContinuous ⇑ψ
        this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
        this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
        ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
        ui : IsUniformInducing ι
        di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
        extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
        hψ : Continuous (⋯.extend ⇑ψ)
        h : Eq K.topologicalClosure Top.top
        ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h).symm.to …
        hψ₁ : Continuous ⇑ψ₁
        ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
      -/
      cases' ringHom_eq_id_or_conj_of_continuous hψ₁ with h h
        /-
          case refine_2.inr.inl
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h✝ : Eq K.topologicalClosure Top.top
          ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h✝).symm.t …
          hψ₁ : Continuous ⇑ψ₁
          h : Eq ψ₁ (RingHom.id Complex)
          ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
        -/
      · left
        /-
          case refine_2.inr.inl.h
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h✝ : Eq K.topologicalClosure Top.top
          ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h✝).symm.t …
          hψ₁ : Continuous ⇑ψ₁
          h : Eq ψ₁ (RingHom.id Complex)
          ⊢ Eq (↑↑ψ).toFun ⇑K.subtype
        -/
        ext1 z
        /-
          case refine_2.inr.inl.h.h
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h✝ : Eq K.topologicalClosure Top.top
          ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h✝).symm.t …
          hψ₁ : Continuous ⇑ψ₁
          h : Eq ψ₁ (RingHom.id Complex)
          z : Subtype fun x => Membership.mem K x
          ⊢ Eq ((↑↑ψ).toFun z) (K.subtype z)
        -/
        convert RingHom.congr_fun h z using 1
        /-
          case h.e'_2
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h✝ : Eq K.topologicalClosure Top.top
          ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h✝).symm.t …
          hψ₁ : Continuous ⇑ψ₁
          h : Eq ψ₁ (RingHom.id Complex)
          z : Subtype fun x => Membership.mem K x
          ⊢ Eq ((↑↑ψ).toFun z) (ψ₁ ↑z)
        -/
        exact (IsDenseInducing.extend_eq di hc.continuous z).symm
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inr.inr
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h✝ : Eq K.topologicalClosure Top.top
          ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h✝).symm.t …
          hψ₁ : Continuous ⇑ψ₁
          h : Eq ψ₁ (starRingEnd Complex)
          ⊢ Or (Eq (↑↑ψ).toFun ⇑K.subtype) (Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd  …
        -/
      · right
        /-
          case refine_2.inr.inr.h
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h✝ : Eq K.topologicalClosure Top.top
          ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h✝).symm.t …
          hψ₁ : Continuous ⇑ψ₁
          h : Eq ψ₁ (starRingEnd Complex)
          ⊢ Eq (↑↑ψ).toFun (Function.comp ⇑(starRingEnd Complex) ⇑K.subtype)
        -/
        ext1 z
        /-
          case refine_2.inr.inr.h.h
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h✝ : Eq K.topologicalClosure Top.top
          ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h✝).symm.t …
          hψ₁ : Continuous ⇑ψ₁
          h : Eq ψ₁ (starRingEnd Complex)
          z : Subtype fun x => Membership.mem K x
          ⊢ Eq ((↑↑ψ).toFun z) (Function.comp (⇑(starRingEnd Complex)) (⇑K.subtype) z)
        -/
        convert RingHom.congr_fun h z using 1
        /-
          case h.e'_2
          K : Subfield Complex
          ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
          hc : UniformContinuous ⇑ψ
          this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
          this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
          ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
          ui : IsUniformInducing ι
          di : IsDenseInducing ι := IsUniformInducing.isDenseInducing ui ?refine_1
          extψ : RingHom (Subtype fun x => Membership.mem K.topologicalClosure x) Comple …
          hψ : Continuous (⋯.extend ⇑ψ)
          h✝ : Eq K.topologicalClosure Top.top
          ψ₁ : RingHom Complex Complex := extψ.comp ((RingEquiv.subfieldCongr h✝).symm.t …
          hψ₁ : Continuous ⇑ψ₁
          h : Eq ψ₁ (starRingEnd Complex)
          z : Subtype fun x => Membership.mem K x
          ⊢ Eq ((↑↑ψ).toFun z) (ψ₁ ↑z)
        -/
        exact (IsDenseInducing.extend_eq di hc.continuous z).symm
        /-
          🎉 no goals
        -/
  · let j : { x // x ∈ closure (id '' { x | (K : Set ℂ) x }) } → (K.topologicalClosure : Set ℂ) :=
      fun x =>
      ⟨x, by
        convert x.prop
        simp only [id, Set.image_id']
        rfl ⟩
    convert DenseRange.comp (Function.Surjective.denseRange _)
      (IsDenseEmbedding.id.subtype (K : Set ℂ)).dense (by continuity : Continuous j)
    /-
      case refine_1
      K : Subfield Complex
      ψ : RingHom (Subtype fun x => Membership.mem K x) Complex
      hc : UniformContinuous ⇑ψ
      this✝ : TopologicalDivisionRing Complex := TopologicalDivisionRing.mk
      this : TopologicalRing (Subtype fun x => Membership.mem K.topologicalClosure x …
      ι : (Subtype fun x => Membership.mem K x) → Subtype fun x => Membership.mem K. …
      ui : IsUniformInducing ι
      j : (Subtype fun x => Membership.mem (closure (Set.image id (setOf fun x => ↑K …
      ⊢ Function.Surjective j
    -/
    rintro ⟨y, hy⟩
    use
      ⟨y, by
        convert hy
        simp only [id, Set.image_id']
        rfl ⟩


