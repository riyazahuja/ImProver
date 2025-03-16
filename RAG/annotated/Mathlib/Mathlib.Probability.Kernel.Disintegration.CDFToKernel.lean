/-- a function `f : α × β → ℚ → ℝ` is called a rational conditional kernel CDF of `κ` with respect
to `ν` if is measurable, if `fun b ↦ f (a, b) x` is `(ν a)`-integrable for all `a : α` and `x : ℝ`
and for all measurable sets `s : Set β`, `∫ b in s, f (a, b) x ∂(ν a) = (κ a (s ×ˢ Iic x)).toReal`.
Also the `ℚ → ℝ` function `f (a, b)` should satisfy the properties of a Sieltjes function for
`(ν a)`-almost all `b : β`. -/
structure IsRatCondKernelCDF (f : α × β → ℚ → ℝ) (κ : Kernel α (β × ℝ)) (ν : Kernel α β) :
    Prop where
  measurable : Measurable f
  isRatStieltjesPoint_ae (a : α) : ∀ᵐ b ∂(ν a), IsRatStieltjesPoint f (a, b)
  integrable (a : α) (q : ℚ) : Integrable (fun b ↦ f (a, b) q) (ν a)
  setIntegral (a : α) {s : Set β} (_hs : MeasurableSet s) (q : ℚ) :
    ∫ b in s, f (a, b) q ∂(ν a) = (κ a (s ×ˢ Iic (q : ℝ))).toReal


lemma IsRatCondKernelCDF.mono (hf : IsRatCondKernelCDF f κ ν) (a : α) :
    ∀ᵐ b ∂(ν a), Monotone (f (a, b)) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    ⊢ Filter.Eventually (fun b => Monotone (f { fst := a, snd := b })) (MeasureThe …
  -/
  filter_upwards [hf.isRatStieltjesPoint_ae a] with b hb using hb.mono
  /-
    🎉 no goals
  -/


lemma IsRatCondKernelCDF.tendsto_atTop_one (hf : IsRatCondKernelCDF f κ ν) (a : α) :
    ∀ᵐ b ∂(ν a), Tendsto (f (a, b)) atTop (𝓝 1) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    ⊢ Filter.Eventually (fun b => Filter.Tendsto (f { fst := a, snd := b }) Filter …
  -/
  filter_upwards [hf.isRatStieltjesPoint_ae a] with b hb using hb.tendsto_atTop_one
  /-
    🎉 no goals
  -/


lemma IsRatCondKernelCDF.tendsto_atBot_zero (hf : IsRatCondKernelCDF f κ ν) (a : α) :
    ∀ᵐ b ∂(ν a), Tendsto (f (a, b)) atBot (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    ⊢ Filter.Eventually (fun b => Filter.Tendsto (f { fst := a, snd := b }) Filter …
  -/
  filter_upwards [hf.isRatStieltjesPoint_ae a] with b hb using hb.tendsto_atBot_zero
  /-
    🎉 no goals
  -/


lemma IsRatCondKernelCDF.iInf_rat_gt_eq (hf : IsRatCondKernelCDF f κ ν) (a : α) :
    ∀ᵐ b ∂(ν a), ∀ q, ⨅ r : Ioi q, f (a, b) r = f (a, b) q := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    ⊢ Filter.Eventually (fun b => ∀ (q : Rat), Eq (iInf fun r => f { fst := a, snd …
  -/
  filter_upwards [hf.isRatStieltjesPoint_ae a] with b hb using hb.iInf_rat_gt_eq
  /-
    🎉 no goals
  -/


lemma stieltjesOfMeasurableRat_ae_eq (hf : IsRatCondKernelCDF f κ ν) (a : α) (q : ℚ) :
    (fun b ↦ stieltjesOfMeasurableRat f hf.measurable (a, b) q) =ᵐ[ν a] fun b ↦ f (a, b) q := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    q : Rat
    ⊢ (MeasureTheory.ae (ν a)).EventuallyEq (fun b => ↑(ProbabilityTheory.stieltje …
  -/
  filter_upwards [hf.isRatStieltjesPoint_ae a] with a ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a✝ : α
    q : Rat
    a : β
    ha : ProbabilityTheory.IsRatStieltjesPoint f { fst := a✝, snd := a }
    ⊢ Eq (↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst := a✝, snd := a } …
  -/
  rw [stieltjesOfMeasurableRat_eq, toRatCDF_of_isRatStieltjesPoint ha]
  /-
    🎉 no goals
  -/


lemma setIntegral_stieltjesOfMeasurableRat_rat (hf : IsRatCondKernelCDF f κ ν) (a : α) (q : ℚ)
    {s : Set β} (hs : MeasurableSet s) :
    ∫ b in s, stieltjesOfMeasurableRat f hf.measurable (a, b) q ∂(ν a)
      = (κ a (s ×ˢ Iic (q : ℝ))).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    q : Rat
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict s) fun b => ↑(ProbabilityTheory.s …
  -/
  rw [setIntegral_congr_ae hs (g := fun b ↦ f (a, b) q) ?_, hf.setIntegral a hs]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    q : Rat
    s : Set β
    hs : MeasurableSet s
    ⊢ Filter.Eventually (fun x => Membership.mem s x → Eq (↑(ProbabilityTheory.sti …
  -/
  filter_upwards [stieltjesOfMeasurableRat_ae_eq hf a q] with b hb using fun _ ↦ hb
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_stieltjesOfMeasurableRat_rat := setIntegral_stieltjesOfMeasurableRat_rat


lemma setLIntegral_stieltjesOfMeasurableRat_rat [IsFiniteKernel κ] (hf : IsRatCondKernelCDF f κ ν)
    (a : α) (q : ℚ) {s : Set β} (hs : MeasurableSet s) :
    ∫⁻ b in s, ENNReal.ofReal (stieltjesOfMeasurableRat f hf.measurable (a, b) q) ∂(ν a)
      = κ a (s ×ˢ Iic (q : ℝ)) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    q : Rat
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑(Pr …
  -/
  rw [← ofReal_integral_eq_lintegral_ofReal]
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      q : Rat
      s : Set β
      hs : MeasurableSet s
      ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral ((ν a).restrict s) fun x => ↑(Pro …
    -/
  · rw [setIntegral_stieltjesOfMeasurableRat_rat hf a q hs, ENNReal.ofReal_toReal]
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      q : Rat
      s : Set β
      hs : MeasurableSet s
      ⊢ Ne ((κ a) (SProd.sprod s (Set.Iic ↑q))) Top.top
    -/
    exact measure_ne_top _ _
    /-
      🎉 no goals
    -/
    /-
      case hfi
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      q : Rat
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasureTheory.Integrable (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurable …
    -/
  · refine Integrable.restrict ?_
    /-
      case hfi
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      q : Rat
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasureTheory.Integrable (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurable …
    -/
    rw [integrable_congr (stieltjesOfMeasurableRat_ae_eq hf a q)]
    /-
      case hfi
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      q : Rat
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasureTheory.Integrable (fun b => f { fst := a, snd := b } q) (ν a)
    -/
    exact hf.integrable a q
    /-
      🎉 no goals
    -/
    /-
      case f_nn
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      q : Rat
      s : Set β
      hs : MeasurableSet s
      ⊢ (MeasureTheory.ae ((ν a).restrict s)).EventuallyLE 0 fun b => ↑(ProbabilityT …
    -/
  · exact ae_of_all _ (fun x ↦ stieltjesOfMeasurableRat_nonneg _ _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_stieltjesOfMeasurableRat_rat := setLIntegral_stieltjesOfMeasurableRat_rat


lemma setLIntegral_stieltjesOfMeasurableRat [IsFiniteKernel κ] (hf : IsRatCondKernelCDF f κ ν)
    (a : α) (x : ℝ) {s : Set β} (hs : MeasurableSet s) :
    ∫⁻ b in s, ENNReal.ofReal (stieltjesOfMeasurableRat f hf.measurable (a, b) x) ∂(ν a)
      = κ a (s ×ˢ Iic x) := by
  -- We have the result for `x : ℚ` thanks to `setLIntegral_stieltjesOfMeasurableRat_rat`.
  -- We use a monotone convergence argument to extend it to the reals.
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑(Pr …
  -/
  by_cases hρ_zero : (ν a).restrict s = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Eq ((ν a).restrict s) 0
      ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑(Pr …
    -/
  · rw [hρ_zero, lintegral_zero_measure]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Eq ((ν a).restrict s) 0
      ⊢ Eq 0 ((κ a) (SProd.sprod s (Set.Iic x)))
    -/
    have ⟨q, hq⟩ := exists_rat_gt x
    suffices κ a (s ×ˢ Iic (q : ℝ)) = 0 by
      symm
      refine measure_mono_null (fun p ↦ ?_) this
      simp only [mem_prod, mem_Iic, and_imp]
      exact fun h1 h2 ↦ ⟨h1, h2.trans hq.le⟩
    suffices (κ a (s ×ˢ Iic (q : ℝ))).toReal = 0 by
      rw [ENNReal.toReal_eq_zero_iff] at this
      simpa [measure_ne_top] using this
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Eq ((ν a).restrict s) 0
      q : Rat
      hq : LT.lt x ↑q
      ⊢ Eq ((κ a) (SProd.sprod s (Set.Iic ↑q))).toReal 0
    -/
    rw [← hf.setIntegral a hs q]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Eq ((ν a).restrict s) 0
      q : Rat
      hq : LT.lt x ↑q
      ⊢ Eq (MeasureTheory.integral ((ν a).restrict s) fun b => f { fst := a, snd :=  …
    -/
    simp [hρ_zero]
    /-
      🎉 no goals
    -/
  have h : ∫⁻ b in s, ENNReal.ofReal (stieltjesOfMeasurableRat f hf.measurable (a, b) x) ∂(ν a)
      = ∫⁻ b in s, ⨅ r : { r' : ℚ // x < r' },
        ENNReal.ofReal (stieltjesOfMeasurableRat f hf.measurable (a, b) r) ∂(ν a) := by
    congr with b : 1
    simp_rw [← measure_stieltjesOfMeasurableRat_Iic]
    rw [← Monotone.measure_iInter]
    · congr with y : 1
      simp only [mem_Iic, mem_iInter, Subtype.forall]
      refine ⟨fun h a ha ↦ h.trans ?_, fun h ↦ ?_⟩
      · exact mod_cast ha.le
      · refine le_of_forall_lt_rat_imp_le fun q hq ↦ h q ?_
        exact mod_cast hq
    · exact fun r r' hrr' ↦ Iic_subset_Iic.mpr <| mod_cast hrr'
    · exact fun _ ↦ nullMeasurableSet_Iic
    · obtain ⟨q, hq⟩ := exists_rat_gt x
      exact ⟨⟨q, hq⟩, measure_ne_top _ _⟩
  have h_nonempty : Nonempty { r' : ℚ // x < ↑r' } := by
    obtain ⟨r, hrx⟩ := exists_rat_gt x
    exact ⟨⟨r, hrx⟩⟩
  /-
    case neg
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    hρ_zero : Not (Eq ((ν a).restrict s) 0)
    h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
    h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑(Pr …
  -/
  rw [h, lintegral_iInf_directed_of_measurable hρ_zero fun q : { r' : ℚ // x < ↑r' } ↦ ?_]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    hρ_zero : Not (Eq ((ν a).restrict s) 0)
    h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
    h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
    ⊢ Eq (iInf fun b => MeasureTheory.lintegral ((ν a).restrict s) fun a_1 => ENNR …
  -/
  rotate_left
    /-
      case neg.hf_int
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      ⊢ ∀ (b : Subtype fun r' => LT.lt x ↑r'), Ne (MeasureTheory.lintegral ((ν a).re …
    -/
  · intro b
    /-
      case neg.hf_int
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      b : Subtype fun r' => LT.lt x ↑r'
      ⊢ Ne (MeasureTheory.lintegral ((ν a).restrict s) fun a_1 => ENNReal.ofReal (↑( …
    -/
    rw [setLIntegral_stieltjesOfMeasurableRat_rat hf a _ hs]
    /-
      case neg.hf_int
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      b : Subtype fun r' => LT.lt x ↑r'
      ⊢ Ne ((κ a) (SProd.sprod s (Set.Iic ↑↑b))) Top.top
    -/
    exact measure_ne_top _ _
    /-
      🎉 no goals
    -/
    /-
      case neg.h_directed
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      ⊢ Directed (fun x1 x2 => GE.ge x1 x2) fun r b => ENNReal.ofReal (↑(Probability …
    -/
  · refine Monotone.directed_ge fun i j hij b ↦ ?_
    /-
      case neg.h_directed
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      i j : Subtype fun r' => LT.lt x ↑r'
      hij : LE.le i j
      b : β
      ⊢ LE.le (ENNReal.ofReal (↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fs …
    -/
    simp_rw [← measure_stieltjesOfMeasurableRat_Iic]
    /-
      case neg.h_directed
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      i j : Subtype fun r' => LT.lt x ↑r'
      hij : LE.le i j
      b : β
      ⊢ LE.le ((ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst := a, snd := b  …
    -/
    refine measure_mono (Iic_subset_Iic.mpr ?_)
    /-
      case neg.h_directed
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      i j : Subtype fun r' => LT.lt x ↑r'
      hij : LE.le i j
      b : β
      ⊢ LE.le ↑↑i ↑↑j
    -/
    exact mod_cast hij
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      q : Subtype fun r' => LT.lt x ↑r'
      ⊢ Measurable fun b => ENNReal.ofReal (↑(ProbabilityTheory.stieltjesOfMeasurabl …
    -/
  · refine Measurable.ennreal_ofReal ?_
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      q : Subtype fun r' => LT.lt x ↑r'
      ⊢ Measurable fun b => ↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst : …
    -/
    exact (measurable_stieltjesOfMeasurableRat hf.measurable _).comp measurable_prod_mk_left
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    hρ_zero : Not (Eq ((ν a).restrict s) 0)
    h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
    h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
    ⊢ Eq (iInf fun b => MeasureTheory.lintegral ((ν a).restrict s) fun a_1 => ENNR …
  -/
  simp_rw [setLIntegral_stieltjesOfMeasurableRat_rat hf _ _ hs]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    hρ_zero : Not (Eq ((ν a).restrict s) 0)
    h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
    h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
    ⊢ Eq (iInf fun b => (κ a) (SProd.sprod s (Set.Iic ↑↑b))) ((κ a) (SProd.sprod s …
  -/
  rw [← Monotone.measure_iInter]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      ⊢ Eq ((κ a) (Set.iInter fun i => SProd.sprod s (Set.Iic ↑↑i))) ((κ a) (SProd.s …
    -/
  · rw [← prod_iInter]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      ⊢ Eq ((κ a) (SProd.sprod s (Set.iInter fun i => Set.Iic ↑↑i))) ((κ a) (SProd.s …
    -/
    congr with y
    /-
      case neg.h.e_6.h.e_a.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      y : Real
      ⊢ Iff (Membership.mem (Set.iInter fun i => Set.Iic ↑↑i) y) (Membership.mem (Se …
    -/
    simp only [mem_iInter, mem_Iic, Subtype.forall, Subtype.coe_mk]
    /-
      case neg.h.e_6.h.e_a.h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      y : Real
      ⊢ Iff (∀ (a : Rat), LT.lt x ↑a → LE.le y ↑a) (LE.le y x)
    -/
    exact ⟨le_of_forall_lt_rat_imp_le, fun hyx q hq ↦ hyx.trans hq.le⟩
    /-
      🎉 no goals
    -/
    /-
      case neg.hs
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      ⊢ Monotone fun b => SProd.sprod s (Set.Iic ↑↑b)
    -/
  · exact fun i j hij ↦ prod_mono_right (by gcongr)
    /-
      🎉 no goals
    -/
    /-
      case neg.hsm
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      ⊢ ∀ (i : Subtype fun r' => LT.lt x ↑r'), MeasureTheory.NullMeasurableSet (SPro …
    -/
  · exact fun i ↦ (hs.prod measurableSet_Iic).nullMeasurableSet
    /-
      🎉 no goals
    -/
    /-
      case neg.hfin
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      hρ_zero : Not (Eq ((ν a).restrict s) 0)
      h : Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑( …
      h_nonempty : Nonempty (Subtype fun r' => LT.lt x ↑r')
      ⊢ Exists fun i => Ne ((κ a) (SProd.sprod s (Set.Iic ↑↑i))) Top.top
    -/
  · exact ⟨h_nonempty.some, measure_ne_top _ _⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_stieltjesOfMeasurableRat := setLIntegral_stieltjesOfMeasurableRat


lemma lintegral_stieltjesOfMeasurableRat [IsFiniteKernel κ] (hf : IsRatCondKernelCDF f κ ν)
    (a : α) (x : ℝ) :
    ∫⁻ b, ENNReal.ofReal (stieltjesOfMeasurableRat f hf.measurable (a, b) x) ∂(ν a)
      = κ a (univ ×ˢ Iic x) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    ⊢ Eq (MeasureTheory.lintegral (ν a) fun b => ENNReal.ofReal (↑(ProbabilityTheo …
  -/
  rw [← setLIntegral_univ, setLIntegral_stieltjesOfMeasurableRat hf _ _ MeasurableSet.univ]
  /-
    🎉 no goals
  -/


lemma integrable_stieltjesOfMeasurableRat [IsFiniteKernel κ] (hf : IsRatCondKernelCDF f κ ν)
    (a : α) (x : ℝ) :
    Integrable (fun b ↦ stieltjesOfMeasurableRat f hf.measurable (a, b) x) (ν a) := by
  have : (fun b ↦ stieltjesOfMeasurableRat f hf.measurable (a, b) x)
      = fun b ↦ (ENNReal.ofReal (stieltjesOfMeasurableRat f hf.measurable (a, b) x)).toReal := by
    ext t
    rw [ENNReal.toReal_ofReal]
    exact stieltjesOfMeasurableRat_nonneg _ _ _
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    this : Eq (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst :=  …
    ⊢ MeasureTheory.Integrable (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurable …
  -/
  rw [this]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    this : Eq (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst :=  …
    ⊢ MeasureTheory.Integrable (fun b => (ENNReal.ofReal (↑(ProbabilityTheory.stie …
  -/
  refine integrable_toReal_of_lintegral_ne_top ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      this : Eq (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst :=  …
      ⊢ AEMeasurable (fun b => ENNReal.ofReal (↑(ProbabilityTheory.stieltjesOfMeasur …
    -/
  · refine (Measurable.ennreal_ofReal ?_).aemeasurable
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      this : Eq (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst :=  …
      ⊢ Measurable fun b => ↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst : …
    -/
    exact (measurable_stieltjesOfMeasurableRat hf.measurable x).comp measurable_prod_mk_left
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      this : Eq (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst :=  …
      ⊢ Ne (MeasureTheory.lintegral (ν a) fun x_1 => ENNReal.ofReal (↑(ProbabilityTh …
    -/
  · rw [lintegral_stieltjesOfMeasurableRat hf]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      this : Eq (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurableRat f ⋯ { fst :=  …
      ⊢ Ne ((κ a) (SProd.sprod Set.univ (Set.Iic x))) Top.top
    -/
    exact measure_ne_top _ _
    /-
      🎉 no goals
    -/


lemma setIntegral_stieltjesOfMeasurableRat [IsFiniteKernel κ] (hf : IsRatCondKernelCDF f κ ν)
    (a : α) (x : ℝ) {s : Set β} (hs : MeasurableSet s) :
    ∫ b in s, stieltjesOfMeasurableRat f hf.measurable (a, b) x ∂(ν a)
      = (κ a (s ×ˢ Iic x)).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict s) fun b => ↑(ProbabilityTheory.s …
  -/
  rw [← ENNReal.ofReal_eq_ofReal_iff, ENNReal.ofReal_toReal]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral ((ν a).restrict s) fun b => ↑(Pro …
  -/
  rotate_left
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      ⊢ Ne ((κ a) (SProd.sprod s (Set.Iic x))) Top.top
    -/
  · exact measure_ne_top _ _
    /-
      🎉 no goals
    -/
    /-
      case hp
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      ⊢ LE.le 0 (MeasureTheory.integral ((ν a).restrict s) fun b => ↑(ProbabilityThe …
    -/
  · exact setIntegral_nonneg hs (fun _ _ ↦ stieltjesOfMeasurableRat_nonneg _ _ _)
    /-
      🎉 no goals
    -/
    /-
      case hq
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      ⊢ LE.le 0 ((κ a) (SProd.sprod s (Set.Iic x))).toReal
    -/
  · exact ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral ((ν a).restrict s) fun b => ↑(Pro …
  -/
  rw [ofReal_integral_eq_lintegral_ofReal, setLIntegral_stieltjesOfMeasurableRat hf _ _ hs]
    /-
      case hfi
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      ⊢ MeasureTheory.Integrable (fun b => ↑(ProbabilityTheory.stieltjesOfMeasurable …
    -/
  · exact (integrable_stieltjesOfMeasurableRat hf _ _).restrict
    /-
      🎉 no goals
    -/
    /-
      case f_nn
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
      a : α
      x : Real
      s : Set β
      hs : MeasurableSet s
      ⊢ (MeasureTheory.ae ((ν a).restrict s)).EventuallyLE 0 fun b => ↑(ProbabilityT …
    -/
  · exact ae_of_all _ (fun _ ↦ stieltjesOfMeasurableRat_nonneg _ _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_stieltjesOfMeasurableRat := setIntegral_stieltjesOfMeasurableRat


lemma integral_stieltjesOfMeasurableRat [IsFiniteKernel κ] (hf : IsRatCondKernelCDF f κ ν)
    (a : α) (x : ℝ) :
    ∫ b, stieltjesOfMeasurableRat f hf.measurable (a, b) x ∂(ν a)
      = (κ a (univ ×ˢ Iic x)).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsRatCondKernelCDF f κ ν
    a : α
    x : Real
    ⊢ Eq (MeasureTheory.integral (ν a) fun b => ↑(ProbabilityTheory.stieltjesOfMea …
  -/
  rw [← setIntegral_univ, setIntegral_stieltjesOfMeasurableRat hf _ _ MeasurableSet.univ]
  /-
    🎉 no goals
  -/


/-- This property implies `IsRatCondKernelCDF`. The measurability, integrability and integral
conditions are the same, but the limit properties of `IsRatCondKernelCDF` are replaced by
limits of integrals. -/
structure IsRatCondKernelCDFAux (f : α × β → ℚ → ℝ) (κ : Kernel α (β × ℝ)) (ν : Kernel α β) :
    Prop where
  measurable : Measurable f
  mono' (a : α) {q r : ℚ} (_hqr : q ≤ r) : ∀ᵐ c ∂(ν a), f (a, c) q ≤ f (a, c) r
  nonneg' (a : α) (q : ℚ) : ∀ᵐ c ∂(ν a), 0 ≤ f (a, c) q
  le_one' (a : α) (q : ℚ) : ∀ᵐ c ∂(ν a), f (a, c) q ≤ 1
  /- Same as `Tendsto (fun q : ℚ ↦ ∫ c, f (a, c) q ∂(ν a)) atBot (𝓝 0)` but slightly easier
  to prove in the current applications of this definition (some integral convergence lemmas
  currently apply only to `ℕ`, not `ℚ`) -/
  tendsto_integral_of_antitone (a : α) (seq : ℕ → ℚ) (_hs : Antitone seq)
    (_hs_tendsto : Tendsto seq atTop atBot) :
    Tendsto (fun m ↦ ∫ c, f (a, c) (seq m) ∂(ν a)) atTop (𝓝 0)
  /- Same as `Tendsto (fun q : ℚ ↦ ∫ c, f (a, c) q ∂(ν a)) atTop (𝓝 (ν a univ).toReal)` but
  slightly easier to prove in the current applications of this definition (some integral convergence
  lemmas currently apply only to `ℕ`, not `ℚ`) -/
  tendsto_integral_of_monotone (a : α) (seq : ℕ → ℚ) (_hs : Monotone seq)
    (_hs_tendsto : Tendsto seq atTop atTop) :
    Tendsto (fun m ↦ ∫ c, f (a, c) (seq m) ∂(ν a)) atTop (𝓝 (ν a univ).toReal)
  integrable (a : α) (q : ℚ) : Integrable (fun c ↦ f (a, c) q) (ν a)
  setIntegral (a : α) {A : Set β} (_hA : MeasurableSet A) (q : ℚ) :
    ∫ c in A, f (a, c) q ∂(ν a) = (κ a (A ×ˢ Iic ↑q)).toReal


lemma IsRatCondKernelCDFAux.measurable_right (hf : IsRatCondKernelCDFAux f κ ν) (a : α) (q : ℚ) :
    Measurable (fun t ↦ f (a, t) q) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    q : Rat
    ⊢ Measurable fun t => f { fst := a, snd := t } q
  -/
  let h := hf.measurable
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    q : Rat
    h : Measurable f := hf.measurable
    ⊢ Measurable fun t => f { fst := a, snd := t } q
  -/
  rw [measurable_pi_iff] at h
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    q : Rat
    h : ∀ (a : Rat), Measurable fun x => f x a
    ⊢ Measurable fun t => f { fst := a, snd := t } q
  -/
  exact (h q).comp measurable_prod_mk_left
  /-
    🎉 no goals
  -/


lemma IsRatCondKernelCDFAux.mono (hf : IsRatCondKernelCDFAux f κ ν) (a : α) :
    ∀ᵐ c ∂(ν a), Monotone (f (a, c)) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    ⊢ Filter.Eventually (fun c => Monotone (f { fst := a, snd := c })) (MeasureThe …
  -/
  unfold Monotone
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    ⊢ Filter.Eventually (fun c => ∀ ⦃a_1 b : Rat⦄, LE.le a_1 b → LE.le (f { fst := …
  -/
  simp_rw [ae_all_iff]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    ⊢ ∀ (i i_1 : Rat), LE.le i i_1 → Filter.Eventually (fun a_1 => LE.le (f { fst  …
  -/
  exact fun _ _ hqr ↦ hf.mono' a hqr
  /-
    🎉 no goals
  -/


lemma IsRatCondKernelCDFAux.nonneg (hf : IsRatCondKernelCDFAux f κ ν) (a : α) :
    ∀ᵐ c ∂(ν a), ∀ q, 0 ≤ f (a, c) q := ae_all_iff.mpr <| hf.nonneg' a


lemma IsRatCondKernelCDFAux.le_one (hf : IsRatCondKernelCDFAux f κ ν) (a : α) :
    ∀ᵐ c ∂(ν a), ∀ q, f (a, c) q ≤ 1 := ae_all_iff.mpr <| hf.le_one' a


lemma IsRatCondKernelCDFAux.tendsto_zero_of_antitone (hf : IsRatCondKernelCDFAux f κ ν)
    [IsFiniteKernel ν] (a : α) (seq : ℕ → ℚ) (hseq : Antitone seq)
    (hseq_tendsto : Tendsto seq atTop atBot) :
    ∀ᵐ c ∂(ν a), Tendsto (fun m ↦ f (a, c) (seq m)) atTop (𝓝 0) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Rat
    hseq : Antitone seq
    hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atBot
    ⊢ Filter.Eventually (fun c => Filter.Tendsto (fun m => f { fst := a, snd := c  …
  -/
  refine tendsto_of_integral_tendsto_of_antitone ?_ (integrable_const _) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Antitone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atBot
      ⊢ ∀ (n : Nat), MeasureTheory.Integrable (fun c => f { fst := a, snd := c } (se …
    -/
  · exact fun n ↦ hf.integrable a (seq n)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Antitone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atBot
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (ν a) fun a_1 => f { fst :=  …
    -/
  · rw [integral_zero]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Antitone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atBot
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (ν a) fun a_1 => f { fst :=  …
    -/
    exact hf.tendsto_integral_of_antitone a seq hseq hseq_tendsto
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Antitone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atBot
      ⊢ Filter.Eventually (fun a_1 => Antitone fun i => f { fst := a, snd := a_1 } ( …
    -/
  · filter_upwards [hf.mono a] with t ht using fun n m hnm ↦ ht (hseq hnm)
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Antitone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atBot
      ⊢ Filter.Eventually (fun a_1 => ∀ (i : Nat), LE.le 0 (f { fst := a, snd := a_1 …
    -/
  · filter_upwards [hf.nonneg a] with c hc using fun i ↦ hc (seq i)
    /-
      🎉 no goals
    -/


lemma IsRatCondKernelCDFAux.tendsto_one_of_monotone (hf : IsRatCondKernelCDFAux f κ ν)
    [IsFiniteKernel ν] (a : α) (seq : ℕ → ℚ) (hseq : Monotone seq)
    (hseq_tendsto : Tendsto seq atTop atTop) :
    ∀ᵐ c ∂(ν a), Tendsto (fun m ↦ f (a, c) (seq m)) atTop (𝓝 1) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    seq : Nat → Rat
    hseq : Monotone seq
    hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
    ⊢ Filter.Eventually (fun c => Filter.Tendsto (fun m => f { fst := a, snd := c  …
  -/
  refine tendsto_of_integral_tendsto_of_monotone ?_ (integrable_const _) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Monotone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
      ⊢ ∀ (n : Nat), MeasureTheory.Integrable (fun c => f { fst := a, snd := c } (se …
    -/
  · exact fun n ↦ hf.integrable a (seq n)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Monotone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (ν a) fun a_1 => f { fst :=  …
    -/
  · rw [MeasureTheory.integral_const, smul_eq_mul, mul_one]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Monotone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
      ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (ν a) fun a_1 => f { fst :=  …
    -/
    exact hf.tendsto_integral_of_monotone a seq hseq hseq_tendsto
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Monotone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
      ⊢ Filter.Eventually (fun a_1 => Monotone fun i => f { fst := a, snd := a_1 } ( …
    -/
  · filter_upwards [hf.mono a] with t ht using fun n m hnm ↦ ht (hseq hnm)
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      seq : Nat → Rat
      hseq : Monotone seq
      hseq_tendsto : Filter.Tendsto seq Filter.atTop Filter.atTop
      ⊢ Filter.Eventually (fun a_1 => ∀ (i : Nat), LE.le (f { fst := a, snd := a_1 } …
    -/
  · filter_upwards [hf.le_one a] with c hc using fun i ↦ hc (seq i)
    /-
      🎉 no goals
    -/


lemma IsRatCondKernelCDFAux.tendsto_atTop_one (hf : IsRatCondKernelCDFAux f κ ν) [IsFiniteKernel ν]
    (a : α) :
    ∀ᵐ t ∂(ν a), Tendsto (f (a, t)) atTop (𝓝 1) := by
  suffices ∀ᵐ t ∂(ν a), Tendsto (fun (n : ℕ) ↦ f (a, t) n) atTop (𝓝 1) by
    filter_upwards [this, hf.mono a] with t ht h_mono
    rw [tendsto_iff_tendsto_subseq_of_monotone h_mono tendsto_natCast_atTop_atTop]
    exact ht
  filter_upwards [hf.tendsto_one_of_monotone a Nat.cast Nat.mono_cast tendsto_natCast_atTop_atTop]
    with x hx using hx


lemma IsRatCondKernelCDFAux.tendsto_atBot_zero (hf : IsRatCondKernelCDFAux f κ ν) [IsFiniteKernel ν]
    (a : α) :
    ∀ᵐ t ∂(ν a), Tendsto (f (a, t)) atBot (𝓝 0) := by
  suffices ∀ᵐ t ∂(ν a), Tendsto (fun q : ℚ ↦ f (a, t) (-q)) atTop (𝓝 0) by
    filter_upwards [this] with t ht
    have h_eq_neg : f (a, t) = fun q : ℚ ↦ f (a, t) (- -q) := by
      simp_rw [neg_neg]
    rw [h_eq_neg]
    convert ht.comp tendsto_neg_atBot_atTop
    simp
  suffices ∀ᵐ t ∂(ν a), Tendsto (fun (n : ℕ) ↦ f (a, t) (-n)) atTop (𝓝 0) by
    filter_upwards [this, hf.mono a] with t ht h_mono
    have h_anti : Antitone (fun q ↦ f (a, t) (-q)) := h_mono.comp_antitone monotone_id.neg
    exact (tendsto_iff_tendsto_subseq_of_antitone h_anti tendsto_natCast_atTop_atTop).mpr ht
  exact hf.tendsto_zero_of_antitone _ _ Nat.mono_cast.neg
    (tendsto_neg_atBot_iff.mpr tendsto_natCast_atTop_atTop)


lemma IsRatCondKernelCDFAux.bddBelow_range (hf : IsRatCondKernelCDFAux f κ ν) (a : α) :
    ∀ᵐ t ∂(ν a), ∀ q : ℚ, BddBelow (range fun (r : Ioi q) ↦ f (a, t) r) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    ⊢ Filter.Eventually (fun t => ∀ (q : Rat), BddBelow (Set.range fun r => f { fs …
  -/
  filter_upwards [hf.nonneg a] with c hc
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    c : β
    hc : ∀ (q : Rat), LE.le 0 (f { fst := a, snd := c } q)
    ⊢ ∀ (q : Rat), BddBelow (Set.range fun r => f { fst := a, snd := c } ↑r)
  -/
  refine fun q ↦ ⟨0, ?_⟩
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    a : α
    c : β
    hc : ∀ (q : Rat), LE.le 0 (f { fst := a, snd := c } q)
    q : Rat
    ⊢ Membership.mem (lowerBounds (Set.range fun r => f { fst := a, snd := c } ↑r) …
  -/
  simp [mem_lowerBounds, hc]
  /-
    🎉 no goals
  -/


lemma IsRatCondKernelCDFAux.integrable_iInf_rat_gt (hf : IsRatCondKernelCDFAux f κ ν)
    [IsFiniteKernel ν] (a : α) (q : ℚ) :
    Integrable (fun t ↦ ⨅ r : Ioi q, f (a, t) r) (ν a) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    q : Rat
    ⊢ MeasureTheory.Integrable (fun t => iInf fun r => f { fst := a, snd := t } ↑r …
  -/
  rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    q : Rat
    ⊢ MeasureTheory.Memℒp (fun t => iInf fun r => f { fst := a, snd := t } ↑r) 1 ( …
  -/
  refine ⟨(Measurable.iInf fun i ↦ hf.measurable_right a _).aestronglyMeasurable, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    q : Rat
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun t => iInf fun r => f { fst := a, snd := t  …
  -/
  refine (?_ : _ ≤ (ν a univ : ℝ≥0∞)).trans_lt (measure_lt_top _ _)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    q : Rat
    ⊢ LE.le (MeasureTheory.eLpNorm (fun t => iInf fun r => f { fst := a, snd := t  …
  -/
  refine (eLpNorm_le_of_ae_bound (C := 1) ?_).trans (by simp)
  filter_upwards [hf.bddBelow_range a, hf.nonneg a, hf.le_one a]
    with t hbdd_below h_nonneg h_le_one
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    q : Rat
    t : β
    hbdd_below : ∀ (q : Rat), BddBelow (Set.range fun r => f { fst := a, snd := t  …
    h_nonneg : ∀ (q : Rat), LE.le 0 (f { fst := a, snd := t } q)
    h_le_one : ∀ (q : Rat), LE.le (f { fst := a, snd := t } q) 1
    ⊢ LE.le (Norm.norm (iInf fun r => f { fst := a, snd := t } ↑r)) 1
  -/
  rw [Real.norm_eq_abs, abs_of_nonneg]
    /-
      case h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      q : Rat
      t : β
      hbdd_below : ∀ (q : Rat), BddBelow (Set.range fun r => f { fst := a, snd := t  …
      h_nonneg : ∀ (q : Rat), LE.le 0 (f { fst := a, snd := t } q)
      h_le_one : ∀ (q : Rat), LE.le (f { fst := a, snd := t } q) 1
      ⊢ LE.le (iInf fun r => f { fst := a, snd := t } ↑r) 1
    -/
  · refine ciInf_le_of_le ?_ ?_ ?_
      /-
        case h.refine_1
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ : ProbabilityTheory.Kernel α (Prod β Real)
        ν : ProbabilityTheory.Kernel α β
        f : Prod α β → Rat → Real
        hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        q : Rat
        t : β
        hbdd_below : ∀ (q : Rat), BddBelow (Set.range fun r => f { fst := a, snd := t  …
        h_nonneg : ∀ (q : Rat), LE.le 0 (f { fst := a, snd := t } q)
        h_le_one : ∀ (q : Rat), LE.le (f { fst := a, snd := t } q) 1
        ⊢ BddBelow (Set.range fun r => f { fst := a, snd := t } ↑r)
      -/
    · exact hbdd_below _
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ : ProbabilityTheory.Kernel α (Prod β Real)
        ν : ProbabilityTheory.Kernel α β
        f : Prod α β → Rat → Real
        hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        q : Rat
        t : β
        hbdd_below : ∀ (q : Rat), BddBelow (Set.range fun r => f { fst := a, snd := t  …
        h_nonneg : ∀ (q : Rat), LE.le 0 (f { fst := a, snd := t } q)
        h_le_one : ∀ (q : Rat), LE.le (f { fst := a, snd := t } q) 1
        ⊢ ↑(Set.Ioi q)
      -/
    · exact ⟨q + 1, by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case h.refine_3
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ : ProbabilityTheory.Kernel α (Prod β Real)
        ν : ProbabilityTheory.Kernel α β
        f : Prod α β → Rat → Real
        hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        q : Rat
        t : β
        hbdd_below : ∀ (q : Rat), BddBelow (Set.range fun r => f { fst := a, snd := t  …
        h_nonneg : ∀ (q : Rat), LE.le 0 (f { fst := a, snd := t } q)
        h_le_one : ∀ (q : Rat), LE.le (f { fst := a, snd := t } q) 1
        ⊢ LE.le (f { fst := a, snd := t } ↑⟨HAdd.hAdd q 1, ⋯⟩) 1
      -/
    · exact h_le_one _
      /-
        🎉 no goals
      -/
    /-
      case h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      q : Rat
      t : β
      hbdd_below : ∀ (q : Rat), BddBelow (Set.range fun r => f { fst := a, snd := t  …
      h_nonneg : ∀ (q : Rat), LE.le 0 (f { fst := a, snd := t } q)
      h_le_one : ∀ (q : Rat), LE.le (f { fst := a, snd := t } q) 1
      ⊢ LE.le 0 (iInf fun r => f { fst := a, snd := t } ↑r)
    -/
  · exact le_ciInf fun r ↦ h_nonneg _
    /-
      🎉 no goals
    -/


lemma _root_.MeasureTheory.Measure.iInf_rat_gt_prod_Iic {ρ : Measure (α × ℝ)} [IsFiniteMeasure ρ]
    {s : Set α} (hs : MeasurableSet s) (t : ℚ) :
    ⨅ r : { r' : ℚ // t < r' }, ρ (s ×ˢ Iic (r : ℝ)) = ρ (s ×ˢ Iic (t : ℝ)) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ρ : MeasureTheory.Measure (Prod α Real)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    s : Set α
    hs : MeasurableSet s
    t : Rat
    ⊢ Eq (iInf fun r => ρ (SProd.sprod s (Set.Iic ↑↑r))) (ρ (SProd.sprod s (Set.Ii …
  -/
  rw [← Monotone.measure_iInter]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      s : Set α
      hs : MeasurableSet s
      t : Rat
      ⊢ Eq (ρ (Set.iInter fun i => SProd.sprod s (Set.Iic ↑↑i))) (ρ (SProd.sprod s ( …
    -/
  · rw [← prod_iInter]
    /-
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      s : Set α
      hs : MeasurableSet s
      t : Rat
      ⊢ Eq (ρ (SProd.sprod s (Set.iInter fun i => Set.Iic ↑↑i))) (ρ (SProd.sprod s ( …
    -/
    congr with x : 1
    /-
      case h.e_6.h.e_a.h
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      s : Set α
      hs : MeasurableSet s
      t : Rat
      x : Real
      ⊢ Iff (Membership.mem (Set.iInter fun i => Set.Iic ↑↑i) x) (Membership.mem (Se …
    -/
    simp only [mem_iInter, mem_Iic, Subtype.forall, Subtype.coe_mk]
    /-
      case h.e_6.h.e_a.h
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      s : Set α
      hs : MeasurableSet s
      t : Rat
      x : Real
      ⊢ Iff (∀ (a : Rat), LT.lt t a → LE.le x ↑a) (LE.le x ↑t)
    -/
    refine ⟨fun h ↦ ?_, fun h a hta ↦ h.trans ?_⟩
      /-
        case h.e_6.h.e_a.h.refine_1
        α : Type u_1
        mα : MeasurableSpace α
        ρ : MeasureTheory.Measure (Prod α Real)
        inst✝ : MeasureTheory.IsFiniteMeasure ρ
        s : Set α
        hs : MeasurableSet s
        t : Rat
        x : Real
        h : ∀ (a : Rat), LT.lt t a → LE.le x ↑a
        ⊢ LE.le x ↑t
      -/
    · refine le_of_forall_lt_rat_imp_le fun q htq ↦ h q ?_
      /-
        case h.e_6.h.e_a.h.refine_1
        α : Type u_1
        mα : MeasurableSpace α
        ρ : MeasureTheory.Measure (Prod α Real)
        inst✝ : MeasureTheory.IsFiniteMeasure ρ
        s : Set α
        hs : MeasurableSet s
        t : Rat
        x : Real
        h : ∀ (a : Rat), LT.lt t a → LE.le x ↑a
        q : Rat
        htq : LT.lt ↑t ↑q
        ⊢ LT.lt t q
      -/
      exact mod_cast htq
      /-
        🎉 no goals
      -/
      /-
        case h.e_6.h.e_a.h.refine_2
        α : Type u_1
        mα : MeasurableSpace α
        ρ : MeasureTheory.Measure (Prod α Real)
        inst✝ : MeasureTheory.IsFiniteMeasure ρ
        s : Set α
        hs : MeasurableSet s
        t : Rat
        x : Real
        h : LE.le x ↑t
        a : Rat
        hta : LT.lt t a
        ⊢ LE.le ↑t ↑a
      -/
    · exact mod_cast hta.le
      /-
        🎉 no goals
      -/
    /-
      case hs
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      s : Set α
      hs : MeasurableSet s
      t : Rat
      ⊢ Monotone fun r => SProd.sprod s (Set.Iic ↑↑r)
    -/
  · exact fun r r' hrr' ↦ prod_mono_right <| by gcongr
    /-
      🎉 no goals
    -/
    /-
      case hsm
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      s : Set α
      hs : MeasurableSet s
      t : Rat
      ⊢ ∀ (i : Subtype fun r' => LT.lt t r'), MeasureTheory.NullMeasurableSet (SProd …
    -/
  · exact fun _ => (hs.prod measurableSet_Iic).nullMeasurableSet
    /-
      🎉 no goals
    -/
    /-
      case hfin
      α : Type u_1
      mα : MeasurableSpace α
      ρ : MeasureTheory.Measure (Prod α Real)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      s : Set α
      hs : MeasurableSet s
      t : Rat
      ⊢ Exists fun i => Ne (ρ (SProd.sprod s (Set.Iic ↑↑i))) Top.top
    -/
  · exact ⟨⟨t + 1, lt_add_one _⟩, measure_ne_top ρ _⟩
    /-
      🎉 no goals
    -/


lemma IsRatCondKernelCDFAux.setIntegral_iInf_rat_gt (hf : IsRatCondKernelCDFAux f κ ν)
    [IsFiniteKernel κ] [IsFiniteKernel ν] (a : α) (q : ℚ) {A : Set β} (hA : MeasurableSet A) :
    ∫ t in A, ⨅ r : Ioi q, f (a, t) r ∂(ν a) = (κ a (A ×ˢ Iic (q : ℝ))).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    q : Rat
    A : Set β
    hA : MeasurableSet A
    ⊢ Eq (MeasureTheory.integral ((ν a).restrict A) fun t => iInf fun r => f { fst …
  -/
  refine le_antisymm ?_ ?_
  · have h : ∀ r : Ioi q, ∫ t in A, ⨅ r' : Ioi q, f (a, t) r' ∂(ν a)
        ≤ (κ a (A ×ˢ Iic (r : ℝ))).toReal := by
      intro r
      rw [← hf.setIntegral a hA]
      refine setIntegral_mono_ae ?_ ?_ ?_
      · exact (hf.integrable_iInf_rat_gt _ _).integrableOn
      · exact (hf.integrable _ _).integrableOn
      · filter_upwards [hf.bddBelow_range a] with t ht using ciInf_le (ht _) r
    calc ∫ t in A, ⨅ r : Ioi q, f (a, t) r ∂(ν a)
      ≤ ⨅ r : Ioi q, (κ a (A ×ˢ Iic (r : ℝ))).toReal := le_ciInf h
    _ = (κ a (A ×ˢ Iic (q : ℝ))).toReal := by
        rw [← Measure.iInf_rat_gt_prod_Iic hA q]
        exact (ENNReal.toReal_iInf (fun r ↦ measure_ne_top _ _)).symm
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      q : Rat
      A : Set β
      hA : MeasurableSet A
      ⊢ LE.le ((κ a) (SProd.sprod A (Set.Iic ↑q))).toReal (MeasureTheory.integral (( …
    -/
  · rw [← hf.setIntegral a hA]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      q : Rat
      A : Set β
      hA : MeasurableSet A
      ⊢ LE.le (MeasureTheory.integral ((ν a).restrict A) fun c => f { fst := a, snd  …
    -/
    refine setIntegral_mono_ae ?_ ?_ ?_
      /-
        case refine_2.refine_1
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ : ProbabilityTheory.Kernel α (Prod β Real)
        ν : ProbabilityTheory.Kernel α β
        f : Prod α β → Rat → Real
        hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
        inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        q : Rat
        A : Set β
        hA : MeasurableSet A
        ⊢ MeasureTheory.IntegrableOn (fun c => f { fst := a, snd := c } q) A (ν a)
      -/
    · exact (hf.integrable _ _).integrableOn
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ : ProbabilityTheory.Kernel α (Prod β Real)
        ν : ProbabilityTheory.Kernel α β
        f : Prod α β → Rat → Real
        hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
        inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        q : Rat
        A : Set β
        hA : MeasurableSet A
        ⊢ MeasureTheory.IntegrableOn (fun t => iInf fun r => f { fst := a, snd := t }  …
      -/
    · exact (hf.integrable_iInf_rat_gt _ _).integrableOn
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_3
        α : Type u_1
        β : Type u_2
        mα : MeasurableSpace α
        mβ : MeasurableSpace β
        κ : ProbabilityTheory.Kernel α (Prod β Real)
        ν : ProbabilityTheory.Kernel α β
        f : Prod α β → Rat → Real
        hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
        inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
        inst✝ : ProbabilityTheory.IsFiniteKernel ν
        a : α
        q : Rat
        A : Set β
        hA : MeasurableSet A
        ⊢ (MeasureTheory.ae (ν a)).EventuallyLE (fun c => f { fst := a, snd := c } q)  …
      -/
    · filter_upwards [hf.mono a] with c h_mono using le_ciInf (fun r ↦ h_mono (le_of_lt r.prop))
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-04-17")]
alias IsRatCondKernelCDFAux.set_integral_iInf_rat_gt :=
  IsRatCondKernelCDFAux.setIntegral_iInf_rat_gt


lemma IsRatCondKernelCDFAux.iInf_rat_gt_eq (hf : IsRatCondKernelCDFAux f κ ν) [IsFiniteKernel κ]
    [IsFiniteKernel ν] (a : α) :
    ∀ᵐ t ∂(ν a), ∀ q : ℚ, ⨅ r : Ioi q, f (a, t) r = f (a, t) q := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    ⊢ Filter.Eventually (fun t => ∀ (q : Rat), Eq (iInf fun r => f { fst := a, snd …
  -/
  rw [ae_all_iff]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    ⊢ ∀ (i : Rat), Filter.Eventually (fun a_1 => Eq (iInf fun r => f { fst := a, s …
  -/
  refine fun q ↦ ae_eq_of_forall_setIntegral_eq_of_sigmaFinite (μ := ν a) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      q : Rat
      ⊢ ∀ (s : Set β), MeasurableSet s → LT.lt ((ν a) s) Top.top → MeasureTheory.Int …
    -/
  · exact fun _ _ _ ↦ (hf.integrable_iInf_rat_gt _ _).integrableOn
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      q : Rat
      ⊢ ∀ (s : Set β), MeasurableSet s → LT.lt ((ν a) s) Top.top → MeasureTheory.Int …
    -/
  · exact fun _ _ _ ↦ (hf.integrable a _).integrableOn
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      q : Rat
      ⊢ ∀ (s : Set β), MeasurableSet s → LT.lt ((ν a) s) Top.top → Eq (MeasureTheory …
    -/
  · intro s hs _
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → Rat → Real
      hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
      inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
      inst✝ : ProbabilityTheory.IsFiniteKernel ν
      a : α
      q : Rat
      s : Set β
      hs : MeasurableSet s
      a✝ : LT.lt ((ν a) s) Top.top
      ⊢ Eq (MeasureTheory.integral ((ν a).restrict s) fun x => iInf fun r => f { fst …
    -/
    rw [hf.setIntegral _ hs, hf.setIntegral_iInf_rat_gt _ _ hs]
    /-
      🎉 no goals
    -/


lemma IsRatCondKernelCDFAux.isRatStieltjesPoint_ae (hf : IsRatCondKernelCDFAux f κ ν)
    [IsFiniteKernel κ] [IsFiniteKernel ν] (a : α) :
    ∀ᵐ t ∂(ν a), IsRatStieltjesPoint f (a, t) := by
  filter_upwards [hf.tendsto_atTop_one a, hf.tendsto_atBot_zero a,
    hf.iInf_rat_gt_eq a, hf.mono a] with t ht_top ht_bot ht_iInf h_mono
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → Rat → Real
    hf : ProbabilityTheory.IsRatCondKernelCDFAux f κ ν
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsFiniteKernel ν
    a : α
    t : β
    ht_top : Filter.Tendsto (f { fst := a, snd := t }) Filter.atTop (nhds 1)
    ht_bot : Filter.Tendsto (f { fst := a, snd := t }) Filter.atBot (nhds 0)
    ht_iInf : ∀ (q : Rat), Eq (iInf fun r => f { fst := a, snd := t } ↑r) (f { fst …
    h_mono : Monotone (f { fst := a, snd := t })
    ⊢ ProbabilityTheory.IsRatStieltjesPoint f { fst := a, snd := t }
  -/
  exact ⟨h_mono, ht_top, ht_bot, ht_iInf⟩
  /-
    🎉 no goals
  -/


lemma IsRatCondKernelCDFAux.isRatCondKernelCDF (hf : IsRatCondKernelCDFAux f κ ν) [IsFiniteKernel κ]
    [IsFiniteKernel ν] :
    IsRatCondKernelCDF f κ ν where
  measurable := hf.measurable
  isRatStieltjesPoint_ae := hf.isRatStieltjesPoint_ae
  integrable := hf.integrable
  setIntegral := hf.setIntegral


/-- A function `f : α × β → StieltjesFunction` is called a conditional kernel CDF of `κ` with
respect to `ν` if it is measurable, tends to 0 at -∞ and to 1 at +∞ for all `p : α × β`,
`fun b ↦ f (a, b) x` is `(ν a)`-integrable for all `a : α` and `x : ℝ` and for all
measurable sets `s : Set β`, `∫ b in s, f (a, b) x ∂(ν a) = (κ a (s ×ˢ Iic x)).toReal`. -/
structure IsCondKernelCDF (f : α × β → StieltjesFunction) (κ : Kernel α (β × ℝ)) (ν : Kernel α β) :
    Prop where
  measurable (x : ℝ) : Measurable fun p ↦ f p x
  integrable (a : α) (x : ℝ) : Integrable (fun b ↦ f (a, b) x) (ν a)
  tendsto_atTop_one (p : α × β) : Tendsto (f p) atTop (𝓝 1)
  tendsto_atBot_zero (p : α × β) : Tendsto (f p) atBot (𝓝 0)
  setIntegral (a : α) {s : Set β} (_hs : MeasurableSet s) (x : ℝ) :
    ∫ b in s, f (a, b) x ∂(ν a) = (κ a (s ×ˢ Iic x)).toReal


lemma IsCondKernelCDF.nonneg (hf : IsCondKernelCDF f κ ν) (p : α × β) (x : ℝ) : 0 ≤ f p x :=
  Monotone.le_of_tendsto (f p).mono (hf.tendsto_atBot_zero p) x


lemma IsCondKernelCDF.le_one (hf : IsCondKernelCDF f κ ν) (p : α × β) (x : ℝ) : f p x ≤ 1 :=
  Monotone.ge_of_tendsto (f p).mono (hf.tendsto_atTop_one p) x


lemma IsCondKernelCDF.integral
    {f : α × β → StieltjesFunction} (hf : IsCondKernelCDF f κ ν) (a : α) (x : ℝ) :
    ∫ b, f (a, b) x ∂(ν a) = (κ a (univ ×ˢ Iic x)).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    x : Real
    ⊢ Eq (MeasureTheory.integral (ν a) fun b => ↑(f { fst := a, snd := b }) x) ((κ …
  -/
  rw [← hf.setIntegral _ MeasurableSet.univ, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


lemma IsCondKernelCDF.setLIntegral [IsFiniteKernel κ]
    {f : α × β → StieltjesFunction} (hf : IsCondKernelCDF f κ ν)
    (a : α) {s : Set β} (hs : MeasurableSet s) (x : ℝ) :
    ∫⁻ b in s, ENNReal.ofReal (f (a, b) x) ∂(ν a) = κ a (s ×ˢ Iic x) := by
  rw [← ofReal_integral_eq_lintegral_ofReal (hf.integrable a x).restrict
    (ae_of_all _ (fun _ ↦ hf.nonneg _ _)), hf.setIntegral a hs x, ENNReal.ofReal_toReal]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : Prod α β → StieltjesFunction
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    s : Set β
    hs : MeasurableSet s
    x : Real
    ⊢ Ne ((κ a) (SProd.sprod s (Set.Iic x))) Top.top
  -/
  exact measure_ne_top _ _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias IsCondKernelCDF.set_lintegral := IsCondKernelCDF.setLIntegral


lemma IsCondKernelCDF.lintegral [IsFiniteKernel κ]
    {f : α × β → StieltjesFunction} (hf : IsCondKernelCDF f κ ν) (a : α) (x : ℝ) :
    ∫⁻ b, ENNReal.ofReal (f (a, b) x) ∂(ν a) = κ a (univ ×ˢ Iic x) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    f : Prod α β → StieltjesFunction
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    x : Real
    ⊢ Eq (MeasureTheory.lintegral (ν a) fun b => ENNReal.ofReal (↑(f { fst := a, s …
  -/
  rw [← hf.setLIntegral _ MeasurableSet.univ, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


lemma isCondKernelCDF_stieltjesOfMeasurableRat {f : α × β → ℚ → ℝ} (hf : IsRatCondKernelCDF f κ ν)
    [IsFiniteKernel κ] :
    IsCondKernelCDF (stieltjesOfMeasurableRat f hf.measurable) κ ν where
  measurable := measurable_stieltjesOfMeasurableRat hf.measurable
  integrable := integrable_stieltjesOfMeasurableRat hf
  tendsto_atTop_one := tendsto_stieltjesOfMeasurableRat_atTop hf.measurable
  tendsto_atBot_zero := tendsto_stieltjesOfMeasurableRat_atBot hf.measurable
  setIntegral a _ hs x := setIntegral_stieltjesOfMeasurableRat hf a x hs


/-- A function `f : α × β → StieltjesFunction` with the property `IsCondKernelCDF f κ ν` gives a
Markov kernel from `α × β` to `ℝ`, by taking for each `p : α × β` the measure defined by `f p`. -/
noncomputable
def IsCondKernelCDF.toKernel (f : α × β → StieltjesFunction) (hf : IsCondKernelCDF f κ ν) :
    Kernel (α × β) ℝ where
  toFun p := (f p).measure
  measurable' := StieltjesFunction.measurable_measure hf.measurable
    hf.tendsto_atBot_zero hf.tendsto_atTop_one


lemma IsCondKernelCDF.toKernel_apply {hf : IsCondKernelCDF f κ ν} (p : α × β) :
    hf.toKernel f p = (f p).measure := rfl


instance instIsMarkovKernel_toKernel {hf : IsCondKernelCDF f κ ν} :
    IsMarkovKernel (hf.toKernel f) :=
  ⟨fun _ ↦ (f _).isProbabilityMeasure (hf.tendsto_atBot_zero _) (hf.tendsto_atTop_one _)⟩


lemma IsCondKernelCDF.toKernel_Iic {hf : IsCondKernelCDF f κ ν} (p : α × β) (x : ℝ) :
    hf.toKernel f p (Iic x) = ENNReal.ofReal (f p x) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    x✝ : MeasurableSpace β
    f : Prod α β → StieltjesFunction
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    p : Prod α β
    x : Real
    ⊢ Eq (((ProbabilityTheory.IsCondKernelCDF.toKernel f hf) p) (Set.Iic x)) (ENNR …
  -/
  rw [IsCondKernelCDF.toKernel_apply p, (f p).measure_Iic (hf.tendsto_atBot_zero p)]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    x✝ : MeasurableSpace β
    f : Prod α β → StieltjesFunction
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    p : Prod α β
    x : Real
    ⊢ Eq (ENNReal.ofReal (HSub.hSub (↑(f p) x) 0)) (ENNReal.ofReal (↑(f p) x))
  -/
  simp
  /-
    🎉 no goals
  -/


lemma setLIntegral_toKernel_Iic [IsFiniteKernel κ] (hf : IsCondKernelCDF f κ ν)
    (a : α) (x : ℝ) {s : Set β} (hs : MeasurableSet s) :
    ∫⁻ b in s, hf.toKernel f (a, b) (Iic x) ∂(ν a) = κ a (s ×ˢ Iic x) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ((ProbabilityTheory. …
  -/
  simp_rw [IsCondKernelCDF.toKernel_Iic]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    x : Real
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ENNReal.ofReal (↑(f  …
  -/
  exact hf.setLIntegral _ hs _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_toKernel_Iic := setLIntegral_toKernel_Iic


lemma setLIntegral_toKernel_univ [IsFiniteKernel κ] (hf : IsCondKernelCDF f κ ν)
    (a : α) {s : Set β} (hs : MeasurableSet s) :
    ∫⁻ b in s, hf.toKernel f (a, b) univ ∂(ν a) = κ a (s ×ˢ univ) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ((ProbabilityTheory. …
  -/
  rw [← Real.iUnion_Iic_rat, prod_iUnion]
  have h_dir : Directed (fun x y ↦ x ⊆ y) fun q : ℚ ↦ Iic (q : ℝ) := by
    refine Monotone.directed_le fun r r' hrr' ↦ Iic_subset_Iic.mpr ?_
    exact mod_cast hrr'
  have h_dir_prod : Directed (fun x y ↦ x ⊆ y) fun q : ℚ ↦ s ×ˢ Iic (q : ℝ) := by
    refine Monotone.directed_le fun i j hij ↦ ?_
    refine prod_subset_prod_iff.mpr (Or.inl ⟨subset_rfl, Iic_subset_Iic.mpr ?_⟩)
    exact mod_cast hij
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    s : Set β
    hs : MeasurableSet s
    h_dir : Directed (fun x y => HasSubset.Subset x y) fun q => Set.Iic ↑q
    h_dir_prod : Directed (fun x y => HasSubset.Subset x y) fun q => SProd.sprod s …
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => ((ProbabilityTheory. …
  -/
  simp_rw [h_dir.measure_iUnion, h_dir_prod.measure_iUnion]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    s : Set β
    hs : MeasurableSet s
    h_dir : Directed (fun x y => HasSubset.Subset x y) fun q => Set.Iic ↑q
    h_dir_prod : Directed (fun x y => HasSubset.Subset x y) fun q => SProd.sprod s …
    ⊢ Eq (MeasureTheory.lintegral ((ν a).restrict s) fun b => iSup fun i => ((Prob …
  -/
  rw [lintegral_iSup_directed]
    /-
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → StieltjesFunction
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsCondKernelCDF f κ ν
      a : α
      s : Set β
      hs : MeasurableSet s
      h_dir : Directed (fun x y => HasSubset.Subset x y) fun q => Set.Iic ↑q
      h_dir_prod : Directed (fun x y => HasSubset.Subset x y) fun q => SProd.sprod s …
      ⊢ Eq (iSup fun b => MeasureTheory.lintegral ((ν a).restrict s) fun a_1 => ((Pr …
    -/
  · simp_rw [setLIntegral_toKernel_Iic hf _ _ hs]
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → StieltjesFunction
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsCondKernelCDF f κ ν
      a : α
      s : Set β
      hs : MeasurableSet s
      h_dir : Directed (fun x y => HasSubset.Subset x y) fun q => Set.Iic ↑q
      h_dir_prod : Directed (fun x y => HasSubset.Subset x y) fun q => SProd.sprod s …
      ⊢ ∀ (b : Rat), AEMeasurable (fun b_1 => ((ProbabilityTheory.IsCondKernelCDF.to …
    -/
  · refine fun q ↦ Measurable.aemeasurable ?_
    /-
      case hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → StieltjesFunction
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsCondKernelCDF f κ ν
      a : α
      s : Set β
      hs : MeasurableSet s
      h_dir : Directed (fun x y => HasSubset.Subset x y) fun q => Set.Iic ↑q
      h_dir_prod : Directed (fun x y => HasSubset.Subset x y) fun q => SProd.sprod s …
      q : Rat
      ⊢ Measurable fun b => ((ProbabilityTheory.IsCondKernelCDF.toKernel f hf) { fst …
    -/
    exact (Kernel.measurable_coe _ measurableSet_Iic).comp measurable_prod_mk_left
    /-
      🎉 no goals
    -/
    /-
      case h_directed
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → StieltjesFunction
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsCondKernelCDF f κ ν
      a : α
      s : Set β
      hs : MeasurableSet s
      h_dir : Directed (fun x y => HasSubset.Subset x y) fun q => Set.Iic ↑q
      h_dir_prod : Directed (fun x y => HasSubset.Subset x y) fun q => SProd.sprod s …
      ⊢ Directed (fun x1 x2 => LE.le x1 x2) fun i b => ((ProbabilityTheory.IsCondKer …
    -/
  · refine Monotone.directed_le fun i j hij t ↦ measure_mono (Iic_subset_Iic.mpr ?_)
    /-
      case h_directed
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α (Prod β Real)
      ν : ProbabilityTheory.Kernel α β
      f : Prod α β → StieltjesFunction
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      hf : ProbabilityTheory.IsCondKernelCDF f κ ν
      a : α
      s : Set β
      hs : MeasurableSet s
      h_dir : Directed (fun x y => HasSubset.Subset x y) fun q => Set.Iic ↑q
      h_dir_prod : Directed (fun x y => HasSubset.Subset x y) fun q => SProd.sprod s …
      i j : Rat
      hij : LE.le i j
      t : β
      ⊢ LE.le ↑i ↑j
    -/
    exact mod_cast hij
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_toKernel_univ := setLIntegral_toKernel_univ


lemma lintegral_toKernel_univ [IsFiniteKernel κ] (hf : IsCondKernelCDF f κ ν) (a : α) :
    ∫⁻ b, hf.toKernel f (a, b) univ ∂(ν a) = κ a univ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    ⊢ Eq (MeasureTheory.lintegral (ν a) fun b => ((ProbabilityTheory.IsCondKernelC …
  -/
  rw [← setLIntegral_univ, setLIntegral_toKernel_univ hf a MeasurableSet.univ, univ_prod_univ]
  /-
    🎉 no goals
  -/


lemma setLIntegral_toKernel_prod [IsFiniteKernel κ] (hf : IsCondKernelCDF f κ ν)
    (a : α) {s : Set β} (hs : MeasurableSet s) {t : Set ℝ} (ht : MeasurableSet t) :
    ∫⁻ b in s, hf.toKernel f (a, b) t ∂(ν a) = κ a (s ×ˢ t) := by
  -- `setLIntegral_toKernel_Iic` gives the result for `t = Iic x`. These sets form a
  -- π-system that generates the Borel σ-algebra, hence we can get the same equality for any
  -- measurable set `t`.
  induction t, ht
    using MeasurableSpace.induction_on_inter (borel_eq_generateFrom_Iic ℝ) isPiSystem_Iic with
  | empty => simp only [measure_empty, lintegral_const, zero_mul, prod_empty]
  | basic t ht =>
    obtain ⟨q, rfl⟩ := ht
    exact setLIntegral_toKernel_Iic hf a _ hs
  | compl t ht iht =>
    calc ∫⁻ b in s, hf.toKernel f (a, b) tᶜ ∂(ν a)
      = ∫⁻ b in s, hf.toKernel f (a, b) univ - hf.toKernel f (a, b) t ∂(ν a) := by
          congr with x; rw [measure_compl ht (measure_ne_top (hf.toKernel f (a, x)) _)]
    _ = ∫⁻ b in s, hf.toKernel f (a, b) univ ∂(ν a)
          - ∫⁻ b in s, hf.toKernel f (a, b) t ∂(ν a) := by
        rw [lintegral_sub]
        · exact (Kernel.measurable_coe (hf.toKernel f) ht).comp measurable_prod_mk_left
        · rw [iht]
          exact measure_ne_top _ _
        · exact Eventually.of_forall fun a ↦ measure_mono (subset_univ _)
    _ = κ a (s ×ˢ univ) - κ a (s ×ˢ t) := by
        rw [setLIntegral_toKernel_univ hf a hs, iht]
    _ = κ a (s ×ˢ tᶜ) := by
        rw [← measure_diff _ (hs.prod ht).nullMeasurableSet (measure_ne_top _ _)]
        · rw [prod_diff_prod, compl_eq_univ_diff]
          simp only [diff_self, empty_prod, union_empty]
        · rw [prod_subset_prod_iff]
          exact Or.inl ⟨subset_rfl, subset_univ t⟩
  | iUnion f hf_disj hf_meas ihf =>
    simp_rw [measure_iUnion hf_disj hf_meas]
    rw [lintegral_tsum, prod_iUnion, measure_iUnion]
    · simp_rw [ihf]
    · exact hf_disj.mono fun i j h ↦ h.set_prod_right _ _
    · exact fun i ↦ MeasurableSet.prod hs (hf_meas i)
    · exact fun i ↦
        ((Kernel.measurable_coe _ (hf_meas i)).comp measurable_prod_mk_left).aemeasurable.restrict


@[deprecated (since := "2024-06-29")]
alias set_lintegral_toKernel_prod := setLIntegral_toKernel_prod


lemma lintegral_toKernel_mem [IsFiniteKernel κ] (hf : IsCondKernelCDF f κ ν)
    (a : α) {s : Set (β × ℝ)} (hs : MeasurableSet s) :
    ∫⁻ b, hf.toKernel f (a, b) {y | (b, y) ∈ s} ∂(ν a) = κ a s := by
  -- `setLIntegral_toKernel_prod` gives the result for sets of the form `t₁ × t₂`. These
  -- sets form a π-system that generates the product σ-algebra, hence we can get the same equality
  -- for any measurable set `s`.
  induction s, hs
    using MeasurableSpace.induction_on_inter generateFrom_prod.symm isPiSystem_prod with
  | empty =>
    simp only [mem_empty_iff_false, setOf_false, measure_empty, lintegral_const, zero_mul]
  | basic s hs =>
    rcases hs with ⟨t₁, ht₁, t₂, ht₂, rfl⟩
    simp only [mem_setOf_eq] at ht₁ ht₂
    have h_prod_eq_snd : ∀ a ∈ t₁, {x : ℝ | (a, x) ∈ t₁ ×ˢ t₂} = t₂ := by
      intro a ha
      simp only [ha, prod_mk_mem_set_prod_eq, true_and, setOf_mem_eq]
    rw [← lintegral_add_compl _ ht₁]
    have h_eq1 : ∫⁻ x in t₁, hf.toKernel f (a, x) {y : ℝ | (x, y) ∈ t₁ ×ˢ t₂} ∂(ν a)
        = ∫⁻ x in t₁, hf.toKernel f (a, x) t₂ ∂(ν a) := by
      refine setLIntegral_congr_fun ht₁ (Eventually.of_forall fun a ha ↦ ?_)
      rw [h_prod_eq_snd a ha]
    have h_eq2 :
        ∫⁻ x in t₁ᶜ, hf.toKernel f (a, x) {y : ℝ | (x, y) ∈ t₁ ×ˢ t₂} ∂(ν a) = 0 := by
      suffices h_eq_zero :
          ∀ x ∈ t₁ᶜ, hf.toKernel f (a, x) {y : ℝ | (x, y) ∈ t₁ ×ˢ t₂} = 0 by
        rw [setLIntegral_congr_fun ht₁.compl (Eventually.of_forall h_eq_zero)]
        simp only [lintegral_const, zero_mul]
      intro a hat₁
      rw [mem_compl_iff] at hat₁
      simp only [hat₁, prod_mk_mem_set_prod_eq, false_and, setOf_false, measure_empty]
    rw [h_eq1, h_eq2, add_zero]
    exact setLIntegral_toKernel_prod hf a ht₁ ht₂
  | compl t ht ht_eq =>
    calc ∫⁻ b, hf.toKernel f (a, b) {y : ℝ | (b, y) ∈ tᶜ} ∂(ν a)
      = ∫⁻ b, hf.toKernel f (a, b) {y : ℝ | (b, y) ∈ t}ᶜ ∂(ν a) := rfl
    _ = ∫⁻ b, hf.toKernel f (a, b) univ
          - hf.toKernel f (a, b) {y : ℝ | (b, y) ∈ t} ∂(ν a) := by
        congr with x : 1
        exact measure_compl (measurable_prod_mk_left ht)
          (measure_ne_top (hf.toKernel f (a, x)) _)
    _ = ∫⁻ x, hf.toKernel f (a, x) univ ∂(ν a) -
          ∫⁻ x, hf.toKernel f (a, x) {y : ℝ | (x, y) ∈ t} ∂(ν a) := by
        have h_le : (fun x ↦ hf.toKernel f (a, x) {y : ℝ | (x, y) ∈ t})
              ≤ᵐ[ν a] fun x ↦ hf.toKernel f (a, x) univ :=
          Eventually.of_forall fun _ ↦ measure_mono (subset_univ _)
        rw [lintegral_sub _ _ h_le]
        · exact Kernel.measurable_kernel_prod_mk_left' ht a
        refine ((lintegral_mono_ae h_le).trans_lt ?_).ne
        rw [lintegral_toKernel_univ hf]
        exact measure_lt_top _ univ
    _ = κ a univ - κ a t := by rw [ht_eq, lintegral_toKernel_univ hf]
    _ = κ a tᶜ := (measure_compl ht (measure_ne_top _ _)).symm
  | iUnion f' hf_disj hf_meas hf_eq =>
    have h_eq : ∀ a, {x | (a, x) ∈ ⋃ i, f' i} = ⋃ i, {x | (a, x) ∈ f' i} := by
      intro a; ext x; simp only [mem_iUnion, mem_setOf_eq]
    simp_rw [h_eq]
    have h_disj : ∀ a, Pairwise (Disjoint on fun i ↦ {x | (a, x) ∈ f' i}) := by
      intro a i j hij
      have h_disj := hf_disj hij
      rw [Function.onFun, disjoint_iff_inter_eq_empty] at h_disj ⊢
      ext1 x
      simp only [mem_inter_iff, mem_setOf_eq, mem_empty_iff_false, iff_false]
      intro h_mem_both
      suffices (a, x) ∈ ∅ by rwa [mem_empty_iff_false] at this
      rwa [← h_disj, mem_inter_iff]
    calc ∫⁻ b, hf.toKernel f (a, b) (⋃ i, {y | (b, y) ∈ f' i}) ∂(ν a)
      = ∫⁻ b, ∑' i, hf.toKernel f (a, b) {y | (b, y) ∈ f' i} ∂(ν a) := by
          congr with x : 1
          rw [measure_iUnion (h_disj x) fun i ↦ measurable_prod_mk_left (hf_meas i)]
    _ = ∑' i, ∫⁻ b, hf.toKernel f (a, b) {y | (b, y) ∈ f' i} ∂(ν a) :=
          lintegral_tsum fun i ↦ (Kernel.measurable_kernel_prod_mk_left' (hf_meas i) a).aemeasurable
    _ = ∑' i, κ a (f' i) := by simp_rw [hf_eq]
    _ = κ a (iUnion f') := (measure_iUnion hf_disj hf_meas).symm


lemma compProd_toKernel [IsFiniteKernel κ] [IsSFiniteKernel ν] (hf : IsCondKernelCDF f κ ν) :
    ν ⊗ₖ hf.toKernel f = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel ν
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    ⊢ Eq (ν.compProd (ProbabilityTheory.IsCondKernelCDF.toKernel f hf)) κ
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α (Prod β Real)
    ν : ProbabilityTheory.Kernel α β
    f : Prod α β → StieltjesFunction
    inst✝¹ : ProbabilityTheory.IsFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel ν
    hf : ProbabilityTheory.IsCondKernelCDF f κ ν
    a : α
    s : Set (Prod β Real)
    hs : MeasurableSet s
    ⊢ Eq (((ν.compProd (ProbabilityTheory.IsCondKernelCDF.toKernel f hf)) a) s) (( …
  -/
  rw [Kernel.compProd_apply hs, lintegral_toKernel_mem hf a hs]
  /-
    🎉 no goals
  -/


