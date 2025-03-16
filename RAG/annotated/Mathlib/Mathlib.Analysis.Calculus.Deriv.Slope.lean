/-- If the domain has dimension one, then Fréchet derivative is equivalent to the classical
definition with a limit. In this version we have to take the limit along the subset `-{x}`,
because for `y=x` the slope equals zero due to the convention `0⁻¹=0`. -/
theorem hasDerivAtFilter_iff_tendsto_slope {x : 𝕜} {L : Filter 𝕜} :
    HasDerivAtFilter f f' x L ↔ Tendsto (slope f x) (L ⊓ 𝓟 {x}ᶜ) (𝓝 f') :=
  calc HasDerivAtFilter f f' x L
    ↔ Tendsto (fun y ↦ slope f x y - (y - x)⁻¹ • (y - x) • f') L (𝓝 0) := by
        simp only [hasDerivAtFilter_iff_tendsto, ← norm_inv, ← norm_smul,
          ← tendsto_zero_iff_norm_tendsto_zero, slope_def_module, smul_sub]
  _ ↔ Tendsto (fun y ↦ slope f x y - (y - x)⁻¹ • (y - x) • f') (L ⊓ 𝓟 {x}ᶜ) (𝓝 0) :=
                                                                   /-
                                                                     𝕜 : Type u
                                                                     inst✝² : NontriviallyNormedField 𝕜
                                                                     F : Type v
                                                                     inst✝¹ : NormedAddCommGroup F
                                                                     inst✝ : NormedSpace 𝕜 F
                                                                     f : 𝕜 → F
                                                                     f' : F
                                                                     x : 𝕜
                                                                     L : Filter 𝕜
                                                                     ⊢ ∀ (a : 𝕜), Not (Membership.mem (HasCompl.compl (Singleton.singleton x)) a) → …
                                                                   -/
        .symm <| tendsto_inf_principal_nhds_iff_of_forall_eq <| by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  _ ↔ Tendsto (fun y ↦ slope f x y - f') (L ⊓ 𝓟 {x}ᶜ) (𝓝 0) := tendsto_congr' <| by
        /-
          𝕜 : Type u
          inst✝² : NontriviallyNormedField 𝕜
          F : Type v
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          f : 𝕜 → F
          f' : F
          x : 𝕜
          L : Filter 𝕜
          ⊢ (Min.min L (Filter.principal (HasCompl.compl (Singleton.singleton x)))).Even …
        -/
        refine (EqOn.eventuallyEq fun y hy ↦ ?_).filter_mono inf_le_right
        /-
          𝕜 : Type u
          inst✝² : NontriviallyNormedField 𝕜
          F : Type v
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          f : 𝕜 → F
          f' : F
          x : 𝕜
          L : Filter 𝕜
          y : 𝕜
          hy : Membership.mem (HasCompl.compl (Singleton.singleton x)) y
          ⊢ Eq (HSub.hSub (slope f x y) (HSMul.hSMul (Inv.inv (HSub.hSub y x)) (HSMul.hS …
        -/
        rw [inv_smul_smul₀ (sub_ne_zero.2 hy) f']
        /-
          🎉 no goals
        -/
  _ ↔ Tendsto (slope f x) (L ⊓ 𝓟 {x}ᶜ) (𝓝 f') := by
        /-
          𝕜 : Type u
          inst✝² : NontriviallyNormedField 𝕜
          F : Type v
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          f : 𝕜 → F
          f' : F
          x : 𝕜
          L : Filter 𝕜
          ⊢ Iff (Filter.Tendsto (fun y => HSub.hSub (slope f x y) f') (Min.min L (Filter …
        -/
        rw [← nhds_translation_sub f', tendsto_comap_iff]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem hasDerivWithinAt_iff_tendsto_slope :
    HasDerivWithinAt f f' s x ↔ Tendsto (slope f x) (𝓝[s \ {x}] x) (𝓝 f') := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    s : Set 𝕜
    ⊢ Iff (HasDerivWithinAt f f' s x) (Filter.Tendsto (slope f x) (nhdsWithin x (S …
  -/
  simp only [HasDerivWithinAt, nhdsWithin, diff_eq, ← inf_assoc, inf_principal.symm]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    s : Set 𝕜
    ⊢ Iff (HasDerivAtFilter f f' x (Min.min (nhds x) (Filter.principal s))) (Filte …
  -/
  exact hasDerivAtFilter_iff_tendsto_slope
  /-
    🎉 no goals
  -/


theorem hasDerivWithinAt_iff_tendsto_slope' (hs : x ∉ s) :
    HasDerivWithinAt f f' s x ↔ Tendsto (slope f x) (𝓝[s] x) (𝓝 f') := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    s : Set 𝕜
    hs : Not (Membership.mem s x)
    ⊢ Iff (HasDerivWithinAt f f' s x) (Filter.Tendsto (slope f x) (nhdsWithin x s) …
  -/
  rw [hasDerivWithinAt_iff_tendsto_slope, diff_singleton_eq_self hs]
  /-
    🎉 no goals
  -/


theorem hasDerivAt_iff_tendsto_slope : HasDerivAt f f' x ↔ Tendsto (slope f x) (𝓝[≠] x) (𝓝 f') :=
  hasDerivAtFilter_iff_tendsto_slope


theorem hasDerivAt_iff_tendsto_slope_zero :
    HasDerivAt f f' x ↔ Tendsto (fun t ↦ t⁻¹ • (f (x + t) - f x)) (𝓝[≠] 0) (𝓝 f') := by
  have : 𝓝[≠] x = Filter.map (fun t ↦ x + t) (𝓝[≠] 0) := by
    simp [nhdsWithin, map_add_left_nhds_zero x, Filter.map_inf, add_right_injective x]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    f' : F
    x : 𝕜
    this : Eq (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.map  …
    ⊢ Iff (HasDerivAt f f' x) (Filter.Tendsto (fun t => HSMul.hSMul (Inv.inv t) (H …
  -/
  simp [hasDerivAt_iff_tendsto_slope, this, slope, Function.comp_def]
  /-
    🎉 no goals
  -/


alias ⟨HasDerivAt.tendsto_slope_zero, _⟩ := hasDerivAt_iff_tendsto_slope_zero


theorem HasDerivAt.tendsto_slope_zero_right [PartialOrder 𝕜] (h : HasDerivAt f f' x) :
    Tendsto (fun t ↦ t⁻¹ • (f (x + t) - f x)) (𝓝[>] 0) (𝓝 f') :=
  h.tendsto_slope_zero.mono_left (nhdsGT_le_nhdsNE 0)


theorem HasDerivAt.tendsto_slope_zero_left [PartialOrder 𝕜] (h : HasDerivAt f f' x) :
    Tendsto (fun t ↦ t⁻¹ • (f (x + t) - f x)) (𝓝[<] 0) (𝓝 f') :=
  h.tendsto_slope_zero.mono_left (nhdsLT_le_nhdsNE 0)


/-- Given a set `t` such that `s ∩ t` is dense in `s`, then the range of `derivWithin f s` is
contained in the closure of the submodule spanned by the image of `t`. -/
theorem range_derivWithin_subset_closure_span_image
    (f : 𝕜 → F) {s t : Set 𝕜} (h : s ⊆ closure (s ∩ t)) :
    range (derivWithin f s) ⊆ closure (Submodule.span 𝕜 (f '' t)) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s t : Set 𝕜
    h : HasSubset.Subset s (closure (Inter.inter s t))
    ⊢ HasSubset.Subset (Set.range (derivWithin f s)) (closure ↑(Submodule.span 𝕜 ( …
  -/
  rintro - ⟨x, rfl⟩
  /-
    case intro
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s t : Set 𝕜
    h : HasSubset.Subset s (closure (Inter.inter s t))
    x : 𝕜
    ⊢ Membership.mem (closure ↑(Submodule.span 𝕜 (Set.image f t))) (derivWithin f  …
  -/
  rcases eq_or_neBot (𝓝[s \ {x}] x) with H|H
    /-
      case intro.inl
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : Eq (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))) Bot.bot
      ⊢ Membership.mem (closure ↑(Submodule.span 𝕜 (Set.image f t))) (derivWithin f  …
    -/
  · simpa [derivWithin_zero_of_isolated H] using subset_closure (zero_mem _)
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s t : Set 𝕜
    h : HasSubset.Subset s (closure (Inter.inter s t))
    x : 𝕜
    H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
    ⊢ Membership.mem (closure ↑(Submodule.span 𝕜 (Set.image f t))) (derivWithin f  …
  -/
  by_cases H' : DifferentiableWithinAt 𝕜 f s x; swap
    /-
      case neg
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : Not (DifferentiableWithinAt 𝕜 f s x)
      ⊢ Membership.mem (closure ↑(Submodule.span 𝕜 (Set.image f t))) (derivWithin f  …
    -/
  · rw [derivWithin_zero_of_not_differentiableWithinAt H']
    /-
      case neg
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : Not (DifferentiableWithinAt 𝕜 f s x)
      ⊢ Membership.mem (closure ↑(Submodule.span 𝕜 (Set.image f t))) 0
    -/
    exact subset_closure (zero_mem _)
    /-
      🎉 no goals
    -/
  have I : (𝓝[(s ∩ t) \ {x}] x).NeBot := by
    rw [← mem_closure_iff_nhdsWithin_neBot] at H ⊢
    have A : closure (s \ {x}) ⊆ closure (closure (s ∩ t) \ {x}) :=
      closure_mono (diff_subset_diff_left h)
    have B : closure (s ∩ t) \ {x} ⊆ closure ((s ∩ t) \ {x}) := by
      convert closure_diff; exact closure_singleton.symm
    simpa using A.trans (closure_mono B) H
  have : Tendsto (slope f x) (𝓝[(s ∩ t) \ {x}] x) (𝓝 (derivWithin f s x)) := by
    apply Tendsto.mono_left (hasDerivWithinAt_iff_tendsto_slope.1 H'.hasDerivWithinAt)
    rw [inter_comm, inter_diff_assoc]
    exact nhdsWithin_mono _ inter_subset_right
  /-
    case pos
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s t : Set 𝕜
    h : HasSubset.Subset s (closure (Inter.inter s t))
    x : 𝕜
    H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
    H' : DifferentiableWithinAt 𝕜 f s x
    I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
    this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
    ⊢ Membership.mem (closure ↑(Submodule.span 𝕜 (Set.image f t))) (derivWithin f  …
  -/
  rw [← closure_closure, ← Submodule.topologicalClosure_coe]
  /-
    case pos
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s t : Set 𝕜
    h : HasSubset.Subset s (closure (Inter.inter s t))
    x : 𝕜
    H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
    H' : DifferentiableWithinAt 𝕜 f s x
    I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
    this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
    ⊢ Membership.mem (closure ↑(Submodule.span 𝕜 (Set.image f t)).topologicalClosu …
  -/
  apply mem_closure_of_tendsto this
  /-
    case pos
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s t : Set 𝕜
    h : HasSubset.Subset s (closure (Inter.inter s t))
    x : 𝕜
    H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
    H' : DifferentiableWithinAt 𝕜 f s x
    I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
    this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (↑(Submodule.span 𝕜 (Set.image  …
  -/
  filter_upwards [self_mem_nhdsWithin] with y hy
  /-
    case h
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s t : Set 𝕜
    h : HasSubset.Subset s (closure (Inter.inter s t))
    x : 𝕜
    H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
    H' : DifferentiableWithinAt 𝕜 f s x
    I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
    this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
    y : 𝕜
    hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
    ⊢ Membership.mem (↑(Submodule.span 𝕜 (Set.image f t)).topologicalClosure) (slo …
  -/
  simp only [slope, vsub_eq_sub, SetLike.mem_coe]
  /-
    case h
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s t : Set 𝕜
    h : HasSubset.Subset s (closure (Inter.inter s t))
    x : 𝕜
    H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
    H' : DifferentiableWithinAt 𝕜 f s x
    I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
    this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
    y : 𝕜
    hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
    ⊢ Membership.mem (Submodule.span 𝕜 (Set.image f t)).topologicalClosure (HSMul. …
  -/
  refine Submodule.smul_mem _ _ (Submodule.sub_mem _ ?_ ?_)
    /-
      case h.refine_1
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : DifferentiableWithinAt 𝕜 f s x
      I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
      this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
      y : 𝕜
      hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
      ⊢ Membership.mem (Submodule.span 𝕜 (Set.image f t)).topologicalClosure (f y)
    -/
  · apply Submodule.le_topologicalClosure
    /-
      case h.refine_1.a
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : DifferentiableWithinAt 𝕜 f s x
      I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
      this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
      y : 𝕜
      hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
      ⊢ Membership.mem (Submodule.span 𝕜 (Set.image f t)) (f y)
    -/
    apply Submodule.subset_span
    /-
      case h.refine_1.a.a
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : DifferentiableWithinAt 𝕜 f s x
      I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
      this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
      y : 𝕜
      hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
      ⊢ Membership.mem (Set.image f t) (f y)
    -/
    exact mem_image_of_mem _ hy.1.2
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : DifferentiableWithinAt 𝕜 f s x
      I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
      this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
      y : 𝕜
      hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
      ⊢ Membership.mem (Submodule.span 𝕜 (Set.image f t)).topologicalClosure (f x)
    -/
  · apply Submodule.closure_subset_topologicalClosure_span
    suffices A : f x ∈ closure (f '' (s ∩ t)) from
      closure_mono (image_subset _ inter_subset_right) A
    /-
      case h.refine_2.a
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : DifferentiableWithinAt 𝕜 f s x
      I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
      this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
      y : 𝕜
      hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
      ⊢ Membership.mem (closure (Set.image f (Inter.inter s t))) (f x)
    -/
    apply ContinuousWithinAt.mem_closure_image
      /-
        case h.refine_2.a.h
        𝕜 : Type u
        inst✝² : NontriviallyNormedField 𝕜
        F : Type v
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : 𝕜 → F
        s t : Set 𝕜
        h : HasSubset.Subset s (closure (Inter.inter s t))
        x : 𝕜
        H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
        H' : DifferentiableWithinAt 𝕜 f s x
        I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
        this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
        y : 𝕜
        hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
        ⊢ ContinuousWithinAt f (Inter.inter s t) x
      -/
    · apply H'.continuousWithinAt.mono inter_subset_left
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2.a.hx
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : DifferentiableWithinAt 𝕜 f s x
      I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
      this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
      y : 𝕜
      hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
      ⊢ Membership.mem (closure (Inter.inter s t)) x
    -/
    rw [mem_closure_iff_nhdsWithin_neBot]
    /-
      case h.refine_2.a.hx
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s t : Set 𝕜
      h : HasSubset.Subset s (closure (Inter.inter s t))
      x : 𝕜
      H : (nhdsWithin x (SDiff.sdiff s (Singleton.singleton x))).NeBot
      H' : DifferentiableWithinAt 𝕜 f s x
      I : (nhdsWithin x (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x))).NeBot
      this : Filter.Tendsto (slope f x) (nhdsWithin x (SDiff.sdiff (Inter.inter s t) …
      y : 𝕜
      hy : Membership.mem (SDiff.sdiff (Inter.inter s t) (Singleton.singleton x)) y
      ⊢ (nhdsWithin x (Inter.inter s t)).NeBot
    -/
    exact I.mono (nhdsWithin_mono _ diff_subset)
    /-
      🎉 no goals
    -/


/-- Given a dense set `t`, then the range of `deriv f` is contained in the closure of the submodule
spanned by the image of `t`. -/
theorem range_deriv_subset_closure_span_image
    (f : 𝕜 → F) {t : Set 𝕜} (h : Dense t) :
    range (deriv f) ⊆ closure (Submodule.span 𝕜 (f '' t)) := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    t : Set 𝕜
    h : Dense t
    ⊢ HasSubset.Subset (Set.range (deriv f)) (closure ↑(Submodule.span 𝕜 (Set.imag …
  -/
  rw [← derivWithin_univ]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    t : Set 𝕜
    h : Dense t
    ⊢ HasSubset.Subset (Set.range (derivWithin f Set.univ)) (closure ↑(Submodule.s …
  -/
  apply range_derivWithin_subset_closure_span_image
  /-
    case h
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    t : Set 𝕜
    h : Dense t
    ⊢ HasSubset.Subset Set.univ (closure (Inter.inter Set.univ t))
  -/
  simp [dense_iff_closure_eq.1 h]
  /-
    🎉 no goals
  -/


theorem isSeparable_range_derivWithin [SeparableSpace 𝕜] (f : 𝕜 → F) (s : Set 𝕜) :
    IsSeparable (range (derivWithin f s)) := by
  obtain ⟨t, ts, t_count, ht⟩ : ∃ t, t ⊆ s ∧ Set.Countable t ∧ s ⊆ closure t :=
    (IsSeparable.of_separableSpace s).exists_countable_dense_subset
  /-
    case intro.intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace.SeparableSpace 𝕜
    f : 𝕜 → F
    s t : Set 𝕜
    ts : HasSubset.Subset t s
    t_count : t.Countable
    ht : HasSubset.Subset s (closure t)
    ⊢ TopologicalSpace.IsSeparable (Set.range (derivWithin f s))
  -/
  have : s ⊆ closure (s ∩ t) := by rwa [inter_eq_self_of_subset_right ts]
  /-
    case intro.intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace.SeparableSpace 𝕜
    f : 𝕜 → F
    s t : Set 𝕜
    ts : HasSubset.Subset t s
    t_count : t.Countable
    ht : HasSubset.Subset s (closure t)
    this : HasSubset.Subset s (closure (Inter.inter s t))
    ⊢ TopologicalSpace.IsSeparable (Set.range (derivWithin f s))
  -/
  apply IsSeparable.mono _ (range_derivWithin_subset_closure_span_image f this)
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace.SeparableSpace 𝕜
    f : 𝕜 → F
    s t : Set 𝕜
    ts : HasSubset.Subset t s
    t_count : t.Countable
    ht : HasSubset.Subset s (closure t)
    this : HasSubset.Subset s (closure (Inter.inter s t))
    ⊢ TopologicalSpace.IsSeparable (closure ↑(Submodule.span 𝕜 (Set.image f t)))
  -/
  exact (Countable.image t_count f).isSeparable.span.closure
  /-
    🎉 no goals
  -/


theorem isSeparable_range_deriv [SeparableSpace 𝕜] (f : 𝕜 → F) :
    IsSeparable (range (deriv f)) := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace.SeparableSpace 𝕜
    f : 𝕜 → F
    ⊢ TopologicalSpace.IsSeparable (Set.range (deriv f))
  -/
  rw [← derivWithin_univ]
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : TopologicalSpace.SeparableSpace 𝕜
    f : 𝕜 → F
    ⊢ TopologicalSpace.IsSeparable (Set.range (derivWithin f Set.univ))
  -/
  exact isSeparable_range_derivWithin _ _
  /-
    🎉 no goals
  -/


lemma HasDerivAt.continuousAt_div [DecidableEq 𝕜] {f : 𝕜 → 𝕜} {c a : 𝕜} (hf : HasDerivAt f a c) :
    ContinuousAt (Function.update (fun x ↦ (f x - f c) / (x - c)) c a) c := by
  /-
    𝕜 : Type u
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : DecidableEq 𝕜
    f : 𝕜 → 𝕜
    c a : 𝕜
    hf : HasDerivAt f a c
    ⊢ ContinuousAt (Function.update (fun x => HDiv.hDiv (HSub.hSub (f x) (f c)) (H …
  -/
  rw [← slope_fun_def_field]
  /-
    𝕜 : Type u
    inst✝¹ : NontriviallyNormedField 𝕜
    inst✝ : DecidableEq 𝕜
    f : 𝕜 → 𝕜
    c a : 𝕜
    hf : HasDerivAt f a c
    ⊢ ContinuousAt (Function.update (slope f c) c a) c
  -/
  exact continuousAt_update_same.mpr <| hasDerivAt_iff_tendsto_slope.mp hf
  /-
    🎉 no goals
  -/


theorem HasDerivWithinAt.limsup_slope_le (hf : HasDerivWithinAt f f' s x) (hr : f' < r) :
    ∀ᶠ z in 𝓝[s \ {x}] x, slope f x z < r :=
  hasDerivWithinAt_iff_tendsto_slope.1 hf (IsOpen.mem_nhds isOpen_Iio hr)


theorem HasDerivWithinAt.limsup_slope_le' (hf : HasDerivWithinAt f f' s x) (hs : x ∉ s)
    (hr : f' < r) : ∀ᶠ z in 𝓝[s] x, slope f x z < r :=
  (hasDerivWithinAt_iff_tendsto_slope' hs).1 hf (IsOpen.mem_nhds isOpen_Iio hr)


theorem HasDerivWithinAt.liminf_right_slope_le (hf : HasDerivWithinAt f f' (Ici x) x)
    (hr : f' < r) : ∃ᶠ z in 𝓝[>] x, slope f x z < r :=
  (hf.Ioi_of_Ici.limsup_slope_le' (lt_irrefl x) hr).frequently


/-- If `f` has derivative `f'` within `s` at `x`, then for any `r > ‖f'‖` the ratio
`‖f z - f x‖ / ‖z - x‖` is less than `r` in some neighborhood of `x` within `s`.
In other words, the limit superior of this ratio as `z` tends to `x` along `s`
is less than or equal to `‖f'‖`. -/
theorem HasDerivWithinAt.limsup_norm_slope_le (hf : HasDerivWithinAt f f' s x) (hr : ‖f'‖ < r) :
    ∀ᶠ z in 𝓝[s] x, ‖z - x‖⁻¹ * ‖f z - f x‖ < r := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    ⊢ Filter.Eventually (fun z => LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub  …
  -/
  have hr₀ : 0 < r := lt_of_le_of_lt (norm_nonneg f') hr
  have A : ∀ᶠ z in 𝓝[s \ {x}] x, ‖(z - x)⁻¹ • (f z - f x)‖ ∈ Iio r :=
    (hasDerivWithinAt_iff_tendsto_slope.1 hf).norm (IsOpen.mem_nhds isOpen_Iio hr)
  have B : ∀ᶠ z in 𝓝[{x}] x, ‖(z - x)⁻¹ • (f z - f x)‖ ∈ Iio r :=
    mem_of_superset self_mem_nhdsWithin (singleton_subset_iff.2 <| by simp [hr₀])
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    hr₀ : LT.lt 0 r
    A : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    B : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    ⊢ Filter.Eventually (fun z => LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub  …
  -/
  have C := mem_sup.2 ⟨A, B⟩
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    hr₀ : LT.lt 0 r
    A : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    B : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    C : Membership.mem (Max.max (nhdsWithin x (SDiff.sdiff s (Singleton.singleton  …
    ⊢ Filter.Eventually (fun z => LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub  …
  -/
  rw [← nhdsWithin_union, diff_union_self, nhdsWithin_union, mem_sup] at C
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    hr₀ : LT.lt 0 r
    A : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    B : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    C : And (Membership.mem (nhdsWithin x s) (setOf fun x_1 => (fun z => Membershi …
    ⊢ Filter.Eventually (fun z => LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub  …
  -/
  filter_upwards [C.1]
  /-
    case h
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    hr₀ : LT.lt 0 r
    A : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    B : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    C : And (Membership.mem (nhdsWithin x s) (setOf fun x_1 => (fun z => Membershi …
    ⊢ ∀ (a : Real), Membership.mem (Set.Iio r) (Norm.norm (HSMul.hSMul (Inv.inv (H …
  -/
  simp only [norm_smul, mem_Iio, norm_inv]
  /-
    case h
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    hr₀ : LT.lt 0 r
    A : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    B : Filter.Eventually (fun z => Membership.mem (Set.Iio r) (Norm.norm (HSMul.h …
    C : And (Membership.mem (nhdsWithin x s) (setOf fun x_1 => (fun z => Membershi …
    ⊢ ∀ (a : Real), LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub a x))) (Norm.n …
  -/
  exact fun _ => id
  /-
    🎉 no goals
  -/


/-- If `f` has derivative `f'` within `s` at `x`, then for any `r > ‖f'‖` the ratio
`(‖f z‖ - ‖f x‖) / ‖z - x‖` is less than `r` in some neighborhood of `x` within `s`.
In other words, the limit superior of this ratio as `z` tends to `x` along `s`
is less than or equal to `‖f'‖`.

This lemma is a weaker version of `HasDerivWithinAt.limsup_norm_slope_le`
where `‖f z‖ - ‖f x‖` is replaced by `‖f z - f x‖`. -/
theorem HasDerivWithinAt.limsup_slope_norm_le (hf : HasDerivWithinAt f f' s x) (hr : ‖f'‖ < r) :
    ∀ᶠ z in 𝓝[s] x, ‖z - x‖⁻¹ * (‖f z‖ - ‖f x‖) < r := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    ⊢ Filter.Eventually (fun z => LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub  …
  -/
  apply (hf.limsup_norm_slope_le hr).mono
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    ⊢ ∀ (x_1 : Real), LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub x_1 x))) (No …
  -/
  intro z hz
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    z : Real
    hz : LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub z x))) (Norm.norm (HSub.h …
    ⊢ LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub z x))) (HSub.hSub (Norm.norm …
  -/
  refine lt_of_le_of_lt (mul_le_mul_of_nonneg_left (norm_sub_norm_le _ _) ?_) hz
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    s : Set Real
    x r : Real
    hf : HasDerivWithinAt f f' s x
    hr : LT.lt (Norm.norm f') r
    z : Real
    hz : LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub z x))) (Norm.norm (HSub.h …
    ⊢ LE.le 0 (Inv.inv (Norm.norm (HSub.hSub z x)))
  -/
  exact inv_nonneg.2 (norm_nonneg _)
  /-
    🎉 no goals
  -/


/-- If `f` has derivative `f'` within `(x, +∞)` at `x`, then for any `r > ‖f'‖` the ratio
`‖f z - f x‖ / ‖z - x‖` is frequently less than `r` as `z → x+0`.
In other words, the limit inferior of this ratio as `z` tends to `x+0`
is less than or equal to `‖f'‖`. See also `HasDerivWithinAt.limsup_norm_slope_le`
for a stronger version using limit superior and any set `s`. -/
theorem HasDerivWithinAt.liminf_right_norm_slope_le (hf : HasDerivWithinAt f f' (Ici x) x)
    (hr : ‖f'‖ < r) : ∃ᶠ z in 𝓝[>] x, ‖z - x‖⁻¹ * ‖f z - f x‖ < r :=
  (hf.Ioi_of_Ici.limsup_norm_slope_le hr).frequently


/-- If `f` has derivative `f'` within `(x, +∞)` at `x`, then for any `r > ‖f'‖` the ratio
`(‖f z‖ - ‖f x‖) / (z - x)` is frequently less than `r` as `z → x+0`.
In other words, the limit inferior of this ratio as `z` tends to `x+0`
is less than or equal to `‖f'‖`.

See also

* `HasDerivWithinAt.limsup_norm_slope_le` for a stronger version using
  limit superior and any set `s`;
* `HasDerivWithinAt.liminf_right_norm_slope_le` for a stronger version using
  `‖f z - f xp‖` instead of `‖f z‖ - ‖f x‖`. -/
theorem HasDerivWithinAt.liminf_right_slope_norm_le (hf : HasDerivWithinAt f f' (Ici x) x)
    (hr : ‖f'‖ < r) : ∃ᶠ z in 𝓝[>] x, (z - x)⁻¹ * (‖f z‖ - ‖f x‖) < r := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    x r : Real
    hf : HasDerivWithinAt f f' (Set.Ici x) x
    hr : LT.lt (Norm.norm f') r
    ⊢ Filter.Frequently (fun z => LT.lt (HMul.hMul (Inv.inv (HSub.hSub z x)) (HSub …
  -/
  have := (hf.Ioi_of_Ici.limsup_slope_norm_le hr).frequently
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    x r : Real
    hf : HasDerivWithinAt f f' (Set.Ici x) x
    hr : LT.lt (Norm.norm f') r
    this : Filter.Frequently (fun x_1 => LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSu …
    ⊢ Filter.Frequently (fun z => LT.lt (HMul.hMul (Inv.inv (HSub.hSub z x)) (HSub …
  -/
  refine this.mp (Eventually.mono self_mem_nhdsWithin fun z hxz hz ↦ ?_)
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    f' : E
    x r : Real
    hf : HasDerivWithinAt f f' (Set.Ici x) x
    hr : LT.lt (Norm.norm f') r
    this : Filter.Frequently (fun x_1 => LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSu …
    z : Real
    hxz : LT.lt x z
    hz : LT.lt (HMul.hMul (Inv.inv (Norm.norm (HSub.hSub z x))) (HSub.hSub (Norm.n …
    ⊢ LT.lt (HMul.hMul (Inv.inv (HSub.hSub z x)) (HSub.hSub (Norm.norm (f z)) (Nor …
  -/
  rwa [Real.norm_eq_abs, abs_of_pos (sub_pos_of_lt hxz)] at hz
  /-
    🎉 no goals
  -/


