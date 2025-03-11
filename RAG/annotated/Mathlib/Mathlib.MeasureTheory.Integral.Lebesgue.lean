local infixr:25 " →ₛ " => SimpleFunc


/-- The **lower Lebesgue integral** of a function `f` with respect to a measure `μ`. -/
irreducible_def lintegral {_ : MeasurableSpace α} (μ : Measure α) (f : α → ℝ≥0∞) : ℝ≥0∞ :=
  ⨆ (g : α →ₛ ℝ≥0∞) (_ : ⇑g ≤ f), g.lintegral μ


@[inherit_doc MeasureTheory.lintegral]
notation3 "∫⁻ "(...)", "r:60:(scoped f => f)" ∂"μ:70 => lintegral μ r


@[inherit_doc MeasureTheory.lintegral]
notation3 "∫⁻ "(...)", "r:60:(scoped f => lintegral volume f) => r


@[inherit_doc MeasureTheory.lintegral]
notation3"∫⁻ "(...)" in "s", "r:60:(scoped f => f)" ∂"μ:70 => lintegral (Measure.restrict μ s) r


@[inherit_doc MeasureTheory.lintegral]
notation3"∫⁻ "(...)" in "s", "r:60:(scoped f => lintegral (Measure.restrict volume s) f) => r


theorem SimpleFunc.lintegral_eq_lintegral {m : MeasurableSpace α} (f : α →ₛ ℝ≥0∞) (μ : Measure α) :
    ∫⁻ a, f a ∂μ = f.lintegral μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.lintegral μ fun a => f a) (f.lintegral μ)
  -/
  rw [MeasureTheory.lintegral]
  exact le_antisymm (iSup₂_le fun g hg => lintegral_mono hg <| le_rfl)
    (le_iSup₂_of_le f le_rfl le_rfl)


@[gcongr, mono]
theorem lintegral_mono' {m : MeasurableSpace α} ⦃μ ν : Measure α⦄ (hμν : μ ≤ ν) ⦃f g : α → ℝ≥0∞⦄
    (hfg : f ≤ g) : ∫⁻ a, f a ∂μ ≤ ∫⁻ a, g a ∂ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : LE.le μ ν
    f g : α → ENNReal
    hfg : LE.le f g
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral ν fu …
  -/
  rw [lintegral, lintegral]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    hμν : LE.le μ ν
    f g : α → ENNReal
    hfg : LE.le f g
    ⊢ LE.le (iSup fun g => iSup fun x => g.lintegral μ) (iSup fun g_1 => iSup fun  …
  -/
  exact iSup_mono fun φ => iSup_mono' fun hφ => ⟨le_trans hφ hfg, lintegral_mono (le_refl φ) hμν⟩
  /-
    🎉 no goals
  -/

-- version where `hfg` is an explicit forall, so that `@[gcongr]` can recognize it

@[gcongr] theorem lintegral_mono_fn' ⦃f g : α → ℝ≥0∞⦄ (hfg : ∀ x, f x ≤ g x) (h2 : μ ≤ ν) :
    ∫⁻ a, f a ∂μ ≤ ∫⁻ a, g a ∂ν :=
  lintegral_mono' h2 hfg


theorem lintegral_mono ⦃f g : α → ℝ≥0∞⦄ (hfg : f ≤ g) : ∫⁻ a, f a ∂μ ≤ ∫⁻ a, g a ∂μ :=
  lintegral_mono' (le_refl μ) hfg

-- version where `hfg` is an explicit forall, so that `@[gcongr]` can recognize it

@[gcongr] theorem lintegral_mono_fn ⦃f g : α → ℝ≥0∞⦄ (hfg : ∀ x, f x ≤ g x) :
    ∫⁻ a, f a ∂μ ≤ ∫⁻ a, g a ∂μ :=
  lintegral_mono hfg


theorem lintegral_mono_nnreal {f g : α → ℝ≥0} (h : f ≤ g) : ∫⁻ a, f a ∂μ ≤ ∫⁻ a, g a ∂μ :=
  lintegral_mono fun a => ENNReal.coe_le_coe.2 (h a)


theorem iSup_lintegral_measurable_le_eq_lintegral (f : α → ℝ≥0∞) :
    ⨆ (g : α → ℝ≥0∞) (_ : Measurable g) (_ : g ≤ f), ∫⁻ a, g a ∂μ = ∫⁻ a, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Eq (iSup fun g => iSup fun x => iSup fun x => MeasureTheory.lintegral μ fun  …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ⊢ LE.le (iSup fun g => iSup fun x => iSup fun x => MeasureTheory.lintegral μ f …
    -/
  · exact iSup_le fun i => iSup_le fun _ => iSup_le fun h'i => lintegral_mono h'i
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => f a) (iSup fun g => iSup fun x =>  …
    -/
  · rw [lintegral]
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      ⊢ LE.le (iSup fun g => iSup fun x => g.lintegral μ) (iSup fun g => iSup fun x  …
    -/
    refine iSup₂_le fun i hi => le_iSup₂_of_le i i.measurable <| le_iSup_of_le hi ?_
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      i : MeasureTheory.SimpleFunc α ENNReal
      hi : LE.le ⇑i fun a => f a
      ⊢ LE.le (i.lintegral μ) (MeasureTheory.lintegral μ fun a => i a)
    -/
    exact le_of_eq (i.lintegral_eq_lintegral _).symm
    /-
      🎉 no goals
    -/


theorem lintegral_mono_set {_ : MeasurableSpace α} ⦃μ : Measure α⦄ {s t : Set α} {f : α → ℝ≥0∞}
    (hst : s ⊆ t) : ∫⁻ x in s, f x ∂μ ≤ ∫⁻ x in t, f x ∂μ :=
  lintegral_mono' (Measure.restrict_mono hst (le_refl μ)) (le_refl f)


theorem lintegral_mono_set' {_ : MeasurableSpace α} ⦃μ : Measure α⦄ {s t : Set α} {f : α → ℝ≥0∞}
    (hst : s ≤ᵐ[μ] t) : ∫⁻ x in s, f x ∂μ ≤ ∫⁻ x in t, f x ∂μ :=
  lintegral_mono' (Measure.restrict_mono' hst (le_refl μ)) (le_refl f)


theorem monotone_lintegral {_ : MeasurableSpace α} (μ : Measure α) : Monotone (lintegral μ) :=
  lintegral_mono


@[simp]
theorem lintegral_const (c : ℝ≥0∞) : ∫⁻ _, c ∂μ = c * μ univ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun x => c) (HMul.hMul c (μ Set.univ))
  -/
  rw [← SimpleFunc.const_lintegral, ← SimpleFunc.lintegral_eq_lintegral, SimpleFunc.coe_const]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun x => c) (MeasureTheory.lintegral μ fun a = …
  -/
  rfl
  /-
    🎉 no goals
  -/


                                                  /-
                                                    α : Type u_1
                                                    m : MeasurableSpace α
                                                    μ : MeasureTheory.Measure α
                                                    ⊢ Eq (MeasureTheory.lintegral μ fun x => 0) 0
                                                  -/
theorem lintegral_zero : ∫⁻ _ : α, 0 ∂μ = 0 := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem lintegral_zero_fun : lintegral μ (0 : α → ℝ≥0∞) = 0 :=
  lintegral_zero


                                                           /-
                                                             α : Type u_1
                                                             m : MeasurableSpace α
                                                             μ : MeasureTheory.Measure α
                                                             ⊢ Eq (MeasureTheory.lintegral μ fun x => 1) (μ Set.univ)
                                                           -/
theorem lintegral_one : ∫⁻ _, (1 : ℝ≥0∞) ∂μ = μ univ := by rw [lintegral_const, one_mul]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem setLIntegral_const (s : Set α) (c : ℝ≥0∞) : ∫⁻ _ in s, c ∂μ = c * μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    c : ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => c) (HMul.hMul c (μ s))
  -/
  rw [lintegral_const, Measure.restrict_apply_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_const := setLIntegral_const


                                                           /-
                                                             α : Type u_1
                                                             m : MeasurableSpace α
                                                             μ : MeasureTheory.Measure α
                                                             s : Set α
                                                             ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => 1) (μ s)
                                                           -/
theorem setLIntegral_one (s) : ∫⁻ _ in s, 1 ∂μ = μ s := by rw [setLIntegral_const, one_mul]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_one := setLIntegral_one


theorem setLIntegral_const_lt_top [IsFiniteMeasure μ] (s : Set α) {c : ℝ≥0∞} (hc : c ≠ ∞) :
    ∫⁻ _ in s, c ∂μ < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set α
    c : ENNReal
    hc : Ne c Top.top
    ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict s) fun x => c) Top.top
  -/
  rw [lintegral_const]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Set α
    c : ENNReal
    hc : Ne c Top.top
    ⊢ LT.lt (HMul.hMul c ((μ.restrict s) Set.univ)) Top.top
  -/
  exact ENNReal.mul_lt_top hc.lt_top (measure_lt_top (μ.restrict s) univ)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_const_lt_top := setLIntegral_const_lt_top


theorem lintegral_const_lt_top [IsFiniteMeasure μ] {c : ℝ≥0∞} (hc : c ≠ ∞) : ∫⁻ _, c ∂μ < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : ENNReal
    hc : Ne c Top.top
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => c) Top.top
  -/
  simpa only [Measure.restrict_univ] using setLIntegral_const_lt_top (univ : Set α) hc
  /-
    🎉 no goals
  -/


/-- For any function `f : α → ℝ≥0∞`, there exists a measurable function `g ≤ f` with the same
integral. -/
theorem exists_measurable_le_lintegral_eq (f : α → ℝ≥0∞) :
    ∃ g : α → ℝ≥0∞, Measurable g ∧ g ≤ f ∧ ∫⁻ a, f a ∂μ = ∫⁻ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (Eq (MeasureTheory.linte …
  -/
  rcases eq_or_ne (∫⁻ a, f a ∂μ) 0 with h₀ | h₀
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h₀ : Eq (MeasureTheory.lintegral μ fun a => f a) 0
      ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (Eq (MeasureTheory.linte …
    -/
  · exact ⟨0, measurable_zero, zero_le f, h₀.trans lintegral_zero.symm⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h₀ : Ne (MeasureTheory.lintegral μ fun a => f a) 0
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (Eq (MeasureTheory.linte …
  -/
  rcases exists_seq_strictMono_tendsto' h₀.bot_lt with ⟨L, _, hLf, hL_tendsto⟩
  have : ∀ n, ∃ g : α → ℝ≥0∞, Measurable g ∧ g ≤ f ∧ L n < ∫⁻ a, g a ∂μ := by
    intro n
    simpa only [← iSup_lintegral_measurable_le_eq_lintegral f, lt_iSup_iff, exists_prop] using
      (hLf n).2
  /-
    case inr.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h₀ : Ne (MeasureTheory.lintegral μ fun a => f a) 0
    L : Nat → ENNReal
    left✝ : StrictMono L
    hLf : ∀ (n : Nat), Membership.mem (Set.Ioo Bot.bot (MeasureTheory.lintegral μ  …
    hL_tendsto : Filter.Tendsto L Filter.atTop (nhds (MeasureTheory.lintegral μ fu …
    this : ∀ (n : Nat), Exists fun g => And (Measurable g) (And (LE.le g f) (LT.lt …
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (Eq (MeasureTheory.linte …
  -/
  choose g hgm hgf hLg using this
  refine
    ⟨fun x => ⨆ n, g n x, .iSup hgm, fun x => iSup_le fun n => hgf n x, le_antisymm ?_ ?_⟩
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h₀ : Ne (MeasureTheory.lintegral μ fun a => f a) 0
      L : Nat → ENNReal
      left✝ : StrictMono L
      hLf : ∀ (n : Nat), Membership.mem (Set.Ioo Bot.bot (MeasureTheory.lintegral μ  …
      hL_tendsto : Filter.Tendsto L Filter.atTop (nhds (MeasureTheory.lintegral μ fu …
      g : Nat → α → ENNReal
      hgm : ∀ (n : Nat), Measurable (g n)
      hgf : ∀ (n : Nat), LE.le (g n) f
      hLg : ∀ (n : Nat), LT.lt (L n) (MeasureTheory.lintegral μ fun a => g n a)
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ fu …
    -/
  · refine le_of_tendsto' hL_tendsto fun n => (hLg n).le.trans <| lintegral_mono fun x => ?_
    /-
      case inr.intro.intro.intro.refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h₀ : Ne (MeasureTheory.lintegral μ fun a => f a) 0
      L : Nat → ENNReal
      left✝ : StrictMono L
      hLf : ∀ (n : Nat), Membership.mem (Set.Ioo Bot.bot (MeasureTheory.lintegral μ  …
      hL_tendsto : Filter.Tendsto L Filter.atTop (nhds (MeasureTheory.lintegral μ fu …
      g : Nat → α → ENNReal
      hgm : ∀ (n : Nat), Measurable (g n)
      hgf : ∀ (n : Nat), LE.le (g n) f
      hLg : ∀ (n : Nat), LT.lt (L n) (MeasureTheory.lintegral μ fun a => g n a)
      n : Nat
      x : α
      ⊢ LE.le (g n x) (iSup fun n => g n x)
    -/
    exact le_iSup (fun n => g n x) n
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro.refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      h₀ : Ne (MeasureTheory.lintegral μ fun a => f a) 0
      L : Nat → ENNReal
      left✝ : StrictMono L
      hLf : ∀ (n : Nat), Membership.mem (Set.Ioo Bot.bot (MeasureTheory.lintegral μ  …
      hL_tendsto : Filter.Tendsto L Filter.atTop (nhds (MeasureTheory.lintegral μ fu …
      g : Nat → α → ENNReal
      hgm : ∀ (n : Nat), Measurable (g n)
      hgf : ∀ (n : Nat), LE.le (g n) f
      hLg : ∀ (n : Nat), LT.lt (L n) (MeasureTheory.lintegral μ fun a => g n a)
      ⊢ LE.le (MeasureTheory.lintegral μ fun a => (fun x => iSup fun n => g n x) a)  …
    -/
  · exact lintegral_mono fun x => iSup_le fun n => hgf n x
    /-
      🎉 no goals
    -/


/-- `∫⁻ a in s, f a ∂μ` is defined as the supremum of integrals of simple functions
`φ : α →ₛ ℝ≥0∞` such that `φ ≤ f`. This lemma says that it suffices to take
functions `φ : α →ₛ ℝ≥0`. -/
theorem lintegral_eq_nnreal {m : MeasurableSpace α} (f : α → ℝ≥0∞) (μ : Measure α) :
    ∫⁻ a, f a ∂μ =
      ⨆ (φ : α →ₛ ℝ≥0) (_ : ∀ x, ↑(φ x) ≤ f x), (φ.map ((↑) : ℝ≥0 → ℝ≥0∞)).lintegral μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.lintegral μ fun a => f a) (iSup fun φ => iSup fun x => (Me …
  -/
  rw [lintegral]
  refine
    le_antisymm (iSup₂_le fun φ hφ ↦ ?_) (iSup_mono' fun φ ↦ ⟨φ.map ((↑) : ℝ≥0 → ℝ≥0∞), le_rfl⟩)
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    φ : MeasureTheory.SimpleFunc α ENNReal
    hφ : LE.le ⇑φ fun a => f a
    ⊢ LE.le (φ.lintegral μ) (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc …
  -/
  by_cases h : ∀ᵐ a ∂μ, φ a ≠ ∞
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ)
      ⊢ LE.le (φ.lintegral μ) (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc …
    -/
  · let ψ := φ.map ENNReal.toNNReal
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ)
      ψ : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.map ENNReal. …
      ⊢ LE.le (φ.lintegral μ) (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc …
    -/
    replace h : ψ.map ((↑) : ℝ≥0 → ℝ≥0∞) =ᵐ[μ] φ := h.mono fun a => ENNReal.coe_toNNReal
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      ψ : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.map ENNReal. …
      h : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.SimpleFunc.map ENNReal.o …
      ⊢ LE.le (φ.lintegral μ) (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc …
    -/
    have : ∀ x, ↑(ψ x) ≤ f x := fun x => le_trans ENNReal.coe_toNNReal_le_self (hφ x)
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      ψ : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.map ENNReal. …
      h : (MeasureTheory.ae μ).EventuallyEq ⇑(MeasureTheory.SimpleFunc.map ENNReal.o …
      this : ∀ (x : α), LE.le (↑(ψ x)) (f x)
      ⊢ LE.le (φ.lintegral μ) (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc …
    -/
    exact le_iSup₂_of_le (φ.map ENNReal.toNNReal) this (ge_of_eq <| lintegral_congr h)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Not (Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ))
      ⊢ LE.le (φ.lintegral μ) (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc …
    -/
  · have h_meas : μ (φ ⁻¹' {∞}) ≠ 0 := mt measure_zero_iff_ae_nmem.1 h
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Not (Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ))
      h_meas : Ne (μ (Set.preimage (⇑φ) (Singleton.singleton Top.top))) 0
      ⊢ LE.le (φ.lintegral μ) (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc …
    -/
    refine le_trans le_top (ge_of_eq <| (iSup_eq_top _).2 fun b hb => ?_)
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Not (Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ))
      h_meas : Ne (μ (Set.preimage (⇑φ) (Singleton.singleton Top.top))) 0
      b : ENNReal
      hb : LT.lt b Top.top
      ⊢ Exists fun i => LT.lt b (iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal …
    -/
    obtain ⟨n, hn⟩ : ∃ n : ℕ, b < n * μ (φ ⁻¹' {∞}) := exists_nat_mul_gt h_meas (ne_of_lt hb)
    /-
      case neg.intro
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Not (Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ))
      h_meas : Ne (μ (Set.preimage (⇑φ) (Singleton.singleton Top.top))) 0
      b : ENNReal
      hb : LT.lt b Top.top
      n : Nat
      hn : LT.lt b (HMul.hMul (↑n) (μ (Set.preimage (⇑φ) (Singleton.singleton Top.to …
      ⊢ Exists fun i => LT.lt b (iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal …
    -/
    use (const α (n : ℝ≥0)).restrict (φ ⁻¹' {∞})
    simp only [lt_iSup_iff, exists_prop, coe_restrict, φ.measurableSet_preimage, coe_const,
      ENNReal.coe_indicator, map_coe_ennreal_restrict, SimpleFunc.map_const, ENNReal.coe_natCast,
      restrict_const_lintegral]
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Not (Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ))
      h_meas : Ne (μ (Set.preimage (⇑φ) (Singleton.singleton Top.top))) 0
      b : ENNReal
      hb : LT.lt b Top.top
      n : Nat
      hn : LT.lt b (HMul.hMul (↑n) (μ (Set.preimage (⇑φ) (Singleton.singleton Top.to …
      ⊢ And (∀ (x : α), LE.le ((Set.preimage (⇑φ) (Singleton.singleton Top.top)).ind …
    -/
    refine ⟨indicator_le fun x hx => le_trans ?_ (hφ _), hn⟩
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Not (Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ))
      h_meas : Ne (μ (Set.preimage (⇑φ) (Singleton.singleton Top.top))) 0
      b : ENNReal
      hb : LT.lt b Top.top
      n : Nat
      hn : LT.lt b (HMul.hMul (↑n) (μ (Set.preimage (⇑φ) (Singleton.singleton Top.to …
      x : α
      hx : Membership.mem (Set.preimage (⇑φ) (Singleton.singleton Top.top)) x
      ⊢ LE.le (↑(Function.const α (↑n) x)) (φ x)
    -/
    simp only [mem_preimage, mem_singleton_iff] at hx
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      f : α → ENNReal
      μ : MeasureTheory.Measure α
      φ : MeasureTheory.SimpleFunc α ENNReal
      hφ : LE.le ⇑φ fun a => f a
      h : Not (Filter.Eventually (fun a => Ne (φ a) Top.top) (MeasureTheory.ae μ))
      h_meas : Ne (μ (Set.preimage (⇑φ) (Singleton.singleton Top.top))) 0
      b : ENNReal
      hb : LT.lt b Top.top
      n : Nat
      hn : LT.lt b (HMul.hMul (↑n) (μ (Set.preimage (⇑φ) (Singleton.singleton Top.to …
      x : α
      hx : Eq (φ x) Top.top
      ⊢ LE.le (↑(Function.const α (↑n) x)) (φ x)
    -/
    simp only [hx, le_top]
    /-
      🎉 no goals
    -/


theorem exists_simpleFunc_forall_lintegral_sub_lt_of_pos {f : α → ℝ≥0∞} (h : ∫⁻ x, f x ∂μ ≠ ∞)
    {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ φ : α →ₛ ℝ≥0,
      (∀ x, ↑(φ x) ≤ f x) ∧
        ∀ ψ : α →ₛ ℝ≥0, (∀ x, ↑(ψ x) ≤ f x) → (map (↑) (ψ - φ)).lintegral μ < ε := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun φ => And (∀ (x : α), LE.le (↑(φ x)) (f x)) (∀ (ψ : MeasureTheory. …
  -/
  rw [lintegral_eq_nnreal] at h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun φ => And (∀ (x : α), LE.le (↑(φ x)) (f x)) (∀ (ψ : MeasureTheory. …
  -/
  have := ENNReal.lt_add_right h hε
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    this : LT.lt (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNRea …
    ⊢ Exists fun φ => And (∀ (x : α), LE.le (↑(φ x)) (f x)) (∀ (ψ : MeasureTheory. …
  -/
  erw [ENNReal.biSup_add] at this <;> [skip; exact ⟨0, fun x => zero_le _⟩]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    this : LT.lt (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNRea …
    ⊢ Exists fun φ => And (∀ (x : α), LE.le (↑(φ x)) (f x)) (∀ (ψ : MeasureTheory. …
  -/
  simp_rw [lt_iSup_iff, iSup_lt_iff, iSup_le_iff] at this
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    this : Exists fun i => Exists fun h => Exists fun b => And (LT.lt b (HAdd.hAdd …
    ⊢ Exists fun φ => And (∀ (x : α), LE.le (↑(φ x)) (f x)) (∀ (ψ : MeasureTheory. …
  -/
  rcases this with ⟨φ, hle : ∀ x, ↑(φ x) ≤ f x, b, hbφ, hb⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    φ : MeasureTheory.SimpleFunc α NNReal
    hle : ∀ (x : α), LE.le (↑(φ x)) (f x)
    b : ENNReal
    hbφ : LT.lt b (HAdd.hAdd ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).li …
    hb : ∀ (i : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(i x)) (f  …
    ⊢ Exists fun φ => And (∀ (x : α), LE.le (↑(φ x)) (f x)) (∀ (ψ : MeasureTheory. …
  -/
  refine ⟨φ, hle, fun ψ hψ => ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    φ : MeasureTheory.SimpleFunc α NNReal
    hle : ∀ (x : α), LE.le (↑(φ x)) (f x)
    b : ENNReal
    hbφ : LT.lt b (HAdd.hAdd ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).li …
    hb : ∀ (i : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(i x)) (f  …
    ψ : MeasureTheory.SimpleFunc α NNReal
    hψ : ∀ (x : α), LE.le (↑(ψ x)) (f x)
    ⊢ LT.lt ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal (HSub.hSub ψ φ)).linte …
  -/
  have : (map (↑) φ).lintegral μ ≠ ∞ := ne_top_of_le_ne_top h (by exact le_iSup₂ (α := ℝ≥0∞) φ hle)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    φ : MeasureTheory.SimpleFunc α NNReal
    hle : ∀ (x : α), LE.le (↑(φ x)) (f x)
    b : ENNReal
    hbφ : LT.lt b (HAdd.hAdd ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).li …
    hb : ∀ (i : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(i x)) (f  …
    ψ : MeasureTheory.SimpleFunc α NNReal
    hψ : ∀ (x : α), LE.le (↑(ψ x)) (f x)
    this : Ne ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).lintegral μ) Top. …
    ⊢ LT.lt ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal (HSub.hSub ψ φ)).linte …
  -/
  rw [← ENNReal.add_lt_add_iff_left this, ← add_lintegral, ← SimpleFunc.map_add @ENNReal.coe_add]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    φ : MeasureTheory.SimpleFunc α NNReal
    hle : ∀ (x : α), LE.le (↑(φ x)) (f x)
    b : ENNReal
    hbφ : LT.lt b (HAdd.hAdd ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).li …
    hb : ∀ (i : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(i x)) (f  …
    ψ : MeasureTheory.SimpleFunc α NNReal
    hψ : ∀ (x : α), LE.le (↑(ψ x)) (f x)
    this : Ne ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).lintegral μ) Top. …
    ⊢ LT.lt ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal (HAdd.hAdd φ (HSub.hSu …
  -/
  refine (hb _ fun x => le_trans ?_ (max_le (hle x) (hψ x))).trans_lt hbφ
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    φ : MeasureTheory.SimpleFunc α NNReal
    hle : ∀ (x : α), LE.le (↑(φ x)) (f x)
    b : ENNReal
    hbφ : LT.lt b (HAdd.hAdd ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).li …
    hb : ∀ (i : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(i x)) (f  …
    ψ : MeasureTheory.SimpleFunc α NNReal
    hψ : ∀ (x : α), LE.le (↑(ψ x)) (f x)
    this : Ne ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).lintegral μ) Top. …
    x : α
    ⊢ LE.le (↑((HAdd.hAdd φ (HSub.hSub ψ φ)) x)) (Max.max ↑(φ x) ↑(ψ x))
  -/
  norm_cast
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    φ : MeasureTheory.SimpleFunc α NNReal
    hle : ∀ (x : α), LE.le (↑(φ x)) (f x)
    b : ENNReal
    hbφ : LT.lt b (HAdd.hAdd ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).li …
    hb : ∀ (i : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(i x)) (f  …
    ψ : MeasureTheory.SimpleFunc α NNReal
    hψ : ∀ (x : α), LE.le (↑(ψ x)) (f x)
    this : Ne ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).lintegral μ) Top. …
    x : α
    ⊢ LE.le ((HAdd.hAdd φ (HSub.hSub ψ φ)) x) (Max.max (φ x) (ψ x))
  -/
  simp only [add_apply, sub_apply, add_tsub_eq_max]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofNN …
    ε : ENNReal
    hε : Ne ε 0
    φ : MeasureTheory.SimpleFunc α NNReal
    hle : ∀ (x : α), LE.le (↑(φ x)) (f x)
    b : ENNReal
    hbφ : LT.lt b (HAdd.hAdd ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).li …
    hb : ∀ (i : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(i x)) (f  …
    ψ : MeasureTheory.SimpleFunc α NNReal
    hψ : ∀ (x : α), LE.le (↑(ψ x)) (f x)
    this : Ne ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal φ).lintegral μ) Top. …
    x : α
    ⊢ LE.le (Max.max (φ x) (ψ x)) (Max.max (φ x) (ψ x))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem iSup_lintegral_le {ι : Sort*} (f : ι → α → ℝ≥0∞) :
    ⨆ i, ∫⁻ a, f i a ∂μ ≤ ∫⁻ a, ⨆ i, f i a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_5
    f : ι → α → ENNReal
    ⊢ LE.le (iSup fun i => MeasureTheory.lintegral μ fun a => f i a) (MeasureTheor …
  -/
  simp only [← iSup_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_5
    f : ι → α → ENNReal
    ⊢ LE.le (iSup fun i => MeasureTheory.lintegral μ fun a => f i a) (MeasureTheor …
  -/
  exact (monotone_lintegral μ).le_map_iSup
  /-
    🎉 no goals
  -/


theorem iSup₂_lintegral_le {ι : Sort*} {ι' : ι → Sort*} (f : ∀ i, ι' i → α → ℝ≥0∞) :
    ⨆ (i) (j), ∫⁻ a, f i j a ∂μ ≤ ∫⁻ a, ⨆ (i) (j), f i j a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_5
    ι' : ι → Sort u_6
    f : (i : ι) → ι' i → α → ENNReal
    ⊢ LE.le (iSup fun i => iSup fun j => MeasureTheory.lintegral μ fun a => f i j  …
  -/
  convert (monotone_lintegral μ).le_map_iSup₂ f with a
  /-
    case h.e'_4.h.e'_4.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_5
    ι' : ι → Sort u_6
    f : (i : ι) → ι' i → α → ENNReal
    a : α
    ⊢ Eq (iSup fun i => iSup fun j => f i j a) (iSup (fun i => iSup fun j => f i j …
  -/
  simp only [iSup_apply]
  /-
    🎉 no goals
  -/


theorem le_iInf_lintegral {ι : Sort*} (f : ι → α → ℝ≥0∞) :
    ∫⁻ a, ⨅ i, f i a ∂μ ≤ ⨅ i, ∫⁻ a, f i a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_5
    f : ι → α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => iInf fun i => f i a) (iInf fun i = …
  -/
  simp only [← iInf_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_5
    f : ι → α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => iInf (fun i => f i) a) (iInf fun i …
  -/
  exact (monotone_lintegral μ).map_iInf_le
  /-
    🎉 no goals
  -/


theorem le_iInf₂_lintegral {ι : Sort*} {ι' : ι → Sort*} (f : ∀ i, ι' i → α → ℝ≥0∞) :
    ∫⁻ a, ⨅ (i) (h : ι' i), f i h a ∂μ ≤ ⨅ (i) (h : ι' i), ∫⁻ a, f i h a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_5
    ι' : ι → Sort u_6
    f : (i : ι) → ι' i → α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => iInf fun i => iInf fun h => f i h  …
  -/
  convert (monotone_lintegral μ).map_iInf₂_le f with a
  /-
    case h.e'_3.h.e'_4.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Sort u_5
    ι' : ι → Sort u_6
    f : (i : ι) → ι' i → α → ENNReal
    a : α
    ⊢ Eq (iInf fun i => iInf fun h => f i h a) (iInf (fun i => iInf fun j => f i j …
  -/
  simp only [iInf_apply]
  /-
    🎉 no goals
  -/


theorem lintegral_mono_ae {f g : α → ℝ≥0∞} (h : ∀ᵐ a ∂μ, f a ≤ g a) :
    ∫⁻ a, f a ∂μ ≤ ∫⁻ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ fu …
  -/
  rcases exists_measurable_superset_of_null h with ⟨t, hts, ht, ht0⟩
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
    t : Set α
    hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
    ht : MeasurableSet t
    ht0 : Eq (μ t) 0
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ fu …
  -/
  have : ∀ᵐ x ∂μ, x ∉ t := measure_zero_iff_ae_nmem.1 ht0
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
    t : Set α
    hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
    ht : MeasurableSet t
    ht0 : Eq (μ t) 0
    this : Filter.Eventually (fun x => Not (Membership.mem t x)) (MeasureTheory.ae …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ fu …
  -/
  rw [lintegral, lintegral]
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
    t : Set α
    hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
    ht : MeasurableSet t
    ht0 : Eq (μ t) 0
    this : Filter.Eventually (fun x => Not (Membership.mem t x)) (MeasureTheory.ae …
    ⊢ LE.le (iSup fun g => iSup fun x => g.lintegral μ) (iSup fun g_1 => iSup fun  …
  -/
  refine iSup₂_le fun s hfs ↦ le_iSup₂_of_le (s.restrict tᶜ) ?_ ?_
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
      t : Set α
      hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
      ht : MeasurableSet t
      ht0 : Eq (μ t) 0
      this : Filter.Eventually (fun x => Not (Membership.mem t x)) (MeasureTheory.ae …
      s : MeasureTheory.SimpleFunc α ENNReal
      hfs : LE.le ⇑s fun a => f a
      ⊢ LE.le ⇑(s.restrict (HasCompl.compl t)) fun a => g a
    -/
  · intro a
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
      t : Set α
      hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
      ht : MeasurableSet t
      ht0 : Eq (μ t) 0
      this : Filter.Eventually (fun x => Not (Membership.mem t x)) (MeasureTheory.ae …
      s : MeasureTheory.SimpleFunc α ENNReal
      hfs : LE.le ⇑s fun a => f a
      a : α
      ⊢ LE.le ((s.restrict (HasCompl.compl t)) a) ((fun a => g a) a)
    -/
    by_cases h : a ∈ t <;>
      simp only [restrict_apply s ht.compl, mem_compl_iff, h, not_true, not_false_eq_true,
        indicator_of_not_mem, zero_le, not_false_eq_true, indicator_of_mem]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      h✝ : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
      t : Set α
      hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
      ht : MeasurableSet t
      ht0 : Eq (μ t) 0
      this : Filter.Eventually (fun x => Not (Membership.mem t x)) (MeasureTheory.ae …
      s : MeasureTheory.SimpleFunc α ENNReal
      hfs : LE.le ⇑s fun a => f a
      a : α
      h : Not (Membership.mem t a)
      ⊢ LE.le (s a) (g a)
    -/
    exact le_trans (hfs a) (by_contradiction fun hnfg => h (hts hnfg))
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
      t : Set α
      hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
      ht : MeasurableSet t
      ht0 : Eq (μ t) 0
      this : Filter.Eventually (fun x => Not (Membership.mem t x)) (MeasureTheory.ae …
      s : MeasureTheory.SimpleFunc α ENNReal
      hfs : LE.le ⇑s fun a => f a
      ⊢ LE.le (s.lintegral μ) ((s.restrict (HasCompl.compl t)).lintegral μ)
    -/
  · refine le_of_eq (SimpleFunc.lintegral_congr <| this.mono fun a hnt => ?_)
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
      t : Set α
      hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
      ht : MeasurableSet t
      ht0 : Eq (μ t) 0
      this : Filter.Eventually (fun x => Not (Membership.mem t x)) (MeasureTheory.ae …
      s : MeasureTheory.SimpleFunc α ENNReal
      hfs : LE.le ⇑s fun a => f a
      a : α
      hnt : Not (Membership.mem t a)
      ⊢ Eq (s a) ((s.restrict (HasCompl.compl t)) a)
    -/
    by_cases hat : a ∈ t <;> simp only [restrict_apply s ht.compl, mem_compl_iff, hat, not_true,
      not_false_eq_true, indicator_of_not_mem, not_false_eq_true, indicator_of_mem]
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      h : Filter.Eventually (fun a => LE.le (f a) (g a)) (MeasureTheory.ae μ)
      t : Set α
      hts : HasSubset.Subset (HasCompl.compl (setOf fun x => (fun a => LE.le (f a) ( …
      ht : MeasurableSet t
      ht0 : Eq (μ t) 0
      this : Filter.Eventually (fun x => Not (Membership.mem t x)) (MeasureTheory.ae …
      s : MeasureTheory.SimpleFunc α ENNReal
      hfs : LE.le ⇑s fun a => f a
      a : α
      hnt : Not (Membership.mem t a)
      hat : Membership.mem t a
      ⊢ Eq (s a) 0
    -/
    exact (hnt hat).elim
    /-
      🎉 no goals
    -/


/-- Lebesgue integral over a set is monotone in function.

This version assumes that the upper estimate is an a.e. measurable function
and the estimate holds a.e. on the set.
See also `setLIntegral_mono_ae'` for a version that assumes measurability of the set
but assumes no regularity of either function. -/
theorem setLIntegral_mono_ae {s : Set α} {f g : α → ℝ≥0∞} (hg : AEMeasurable g (μ.restrict s))
    (hfg : ∀ᵐ x ∂μ, x ∈ s → f x ≤ g x) : ∫⁻ x in s, f x ∂μ ≤ ∫⁻ x in s, g x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → ENNReal
    hg : AEMeasurable g (μ.restrict s)
    hfg : Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) (g x)) (Mea …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => f x) (MeasureTheory.l …
  -/
  rcases exists_measurable_le_lintegral_eq (μ.restrict s) f with ⟨f', hf'm, hle, hf'⟩
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → ENNReal
    hg : AEMeasurable g (μ.restrict s)
    hfg : Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) (g x)) (Mea …
    f' : α → ENNReal
    hf'm : Measurable f'
    hle : LE.le f' f
    hf' : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => f x) (MeasureTheory.l …
  -/
  rw [hf']
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → ENNReal
    hg : AEMeasurable g (μ.restrict s)
    hfg : Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) (g x)) (Mea …
    f' : α → ENNReal
    hf'm : Measurable f'
    hle : LE.le f' f
    hf' : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => f' a) (MeasureTheory. …
  -/
  apply lintegral_mono_ae
  /-
    case intro.intro.intro.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f g : α → ENNReal
    hg : AEMeasurable g (μ.restrict s)
    hfg : Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) (g x)) (Mea …
    f' : α → ENNReal
    hf'm : Measurable f'
    hle : LE.le f' f
    hf' : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
    ⊢ Filter.Eventually (fun a => LE.le (f' a) (g a)) (MeasureTheory.ae (μ.restric …
  -/
  rw [ae_restrict_iff₀]
    /-
      case intro.intro.intro.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f g : α → ENNReal
      hg : AEMeasurable g (μ.restrict s)
      hfg : Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) (g x)) (Mea …
      f' : α → ENNReal
      hf'm : Measurable f'
      hle : LE.le f' f
      hf' : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
      ⊢ Filter.Eventually (fun x => Membership.mem s x → LE.le (f' x) (g x)) (Measur …
    -/
  · exact hfg.mono fun x hx hxs ↦ (hle x).trans (hx hxs)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f g : α → ENNReal
      hg : AEMeasurable g (μ.restrict s)
      hfg : Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) (g x)) (Mea …
      f' : α → ENNReal
      hf'm : Measurable f'
      hle : LE.le f' f
      hf' : Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory. …
      ⊢ MeasureTheory.NullMeasurableSet (setOf fun x => LE.le (f' x) (g x)) (μ.restr …
    -/
  · exact nullMeasurableSet_le hf'm.aemeasurable hg
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_mono_ae := setLIntegral_mono_ae


theorem setLIntegral_mono {s : Set α} {f g : α → ℝ≥0∞} (hg : Measurable g)
    (hfg : ∀ x ∈ s, f x ≤ g x) : ∫⁻ x in s, f x ∂μ ≤ ∫⁻ x in s, g x ∂μ :=
  setLIntegral_mono_ae hg.aemeasurable (ae_of_all _ hfg)


@[deprecated (since := "2024-06-29")]
alias set_lintegral_mono := setLIntegral_mono


theorem setLIntegral_mono_ae' {s : Set α} {f g : α → ℝ≥0∞} (hs : MeasurableSet s)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → f x ≤ g x) : ∫⁻ x in s, f x ∂μ ≤ ∫⁻ x in s, g x ∂μ :=
  lintegral_mono_ae <| (ae_restrict_iff' hs).2 hfg


@[deprecated (since := "2024-06-29")]
alias set_lintegral_mono_ae' := setLIntegral_mono_ae'


theorem setLIntegral_mono' {s : Set α} {f g : α → ℝ≥0∞} (hs : MeasurableSet s)
    (hfg : ∀ x ∈ s, f x ≤ g x) : ∫⁻ x in s, f x ∂μ ≤ ∫⁻ x in s, g x ∂μ :=
  setLIntegral_mono_ae' hs (ae_of_all _ hfg)


@[deprecated (since := "2024-06-29")]
alias set_lintegral_mono' := setLIntegral_mono'


theorem setLIntegral_le_lintegral (s : Set α) (f : α → ℝ≥0∞) :
    ∫⁻ x in s, f x ∂μ ≤ ∫⁻ x, f x ∂μ :=
  lintegral_mono' Measure.restrict_le_self le_rfl


@[deprecated (since := "2024-06-29")]
alias set_lintegral_le_lintegral := setLIntegral_le_lintegral


theorem lintegral_congr_ae {f g : α → ℝ≥0∞} (h : f =ᵐ[μ] g) : ∫⁻ a, f a ∂μ = ∫⁻ a, g a ∂μ :=
  le_antisymm (lintegral_mono_ae <| h.le) (lintegral_mono_ae <| h.symm.le)


theorem lintegral_congr {f g : α → ℝ≥0∞} (h : ∀ a, f a = g a) : ∫⁻ a, f a ∂μ = ∫⁻ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    h : ∀ (a : α), Eq (f a) (g a)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ fun a …
  -/
  simp only [h]
  /-
    🎉 no goals
  -/


theorem setLIntegral_congr {f : α → ℝ≥0∞} {s t : Set α} (h : s =ᵐ[μ] t) :
                                                /-
                                                  α : Type u_1
                                                  m : MeasurableSpace α
                                                  μ : MeasureTheory.Measure α
                                                  f : α → ENNReal
                                                  s t : Set α
                                                  h : (MeasureTheory.ae μ).EventuallyEq s t
                                                  ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => f x) (MeasureTheory.lint …
                                                -/
    ∫⁻ x in s, f x ∂μ = ∫⁻ x in t, f x ∂μ := by rw [Measure.restrict_congr_set h]
                                                /-
                                                  🎉 no goals
                                                -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_congr := setLIntegral_congr


theorem setLIntegral_congr_fun {f g : α → ℝ≥0∞} {s : Set α} (hs : MeasurableSet s)
    (hfg : ∀ᵐ x ∂μ, x ∈ s → f x = g x) : ∫⁻ x in s, f x ∂μ = ∫⁻ x in s, g x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    hfg : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (Measur …
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => f x) (MeasureTheory.lint …
  -/
  rw [lintegral_congr_ae]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    hfg : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (Measur …
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq f g
  -/
  rw [EventuallyEq]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    s : Set α
    hs : MeasurableSet s
    hfg : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (g x)) (Measur …
    ⊢ Filter.Eventually (fun x => Eq (f x) (g x)) (MeasureTheory.ae (μ.restrict s))
  -/
  rwa [ae_restrict_iff' hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_congr_fun := setLIntegral_congr_fun


theorem lintegral_ofReal_le_lintegral_nnnorm (f : α → ℝ) :
    ∫⁻ x, ENNReal.ofReal (f x) ∂μ ≤ ∫⁻ x, ‖f x‖₊ ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (MeasureTheo …
  -/
  simp_rw [← ofReal_norm_eq_coe_nnnorm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x)) (MeasureTheo …
  -/
  refine lintegral_mono fun x => ENNReal.ofReal_le_ofReal ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    x : α
    ⊢ LE.le (f x) (Norm.norm (f x))
  -/
  rw [Real.norm_eq_abs]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    x : α
    ⊢ LE.le (f x) (abs (f x))
  -/
  exact le_abs_self (f x)
  /-
    🎉 no goals
  -/


theorem lintegral_nnnorm_eq_of_ae_nonneg {f : α → ℝ} (h_nonneg : 0 ≤ᵐ[μ] f) :
    ∫⁻ x, ‖f x‖₊ ∂μ = ∫⁻ x, ENNReal.ofReal (f x) ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    h_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Eq (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm (f x))) (MeasureTheor …
  -/
  apply lintegral_congr_ae
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    h_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => ↑(NNNorm.nnnorm (f a))) fun a => …
  -/
  filter_upwards [h_nonneg] with x hx
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    h_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
    x : α
    hx : LE.le (0 x) (f x)
    ⊢ Eq (↑(NNNorm.nnnorm (f x))) (ENNReal.ofReal (f x))
  -/
  rw [Real.nnnorm_of_nonneg hx, ENNReal.ofReal_eq_coe_nnreal hx]
  /-
    🎉 no goals
  -/


theorem lintegral_nnnorm_eq_of_nonneg {f : α → ℝ} (h_nonneg : 0 ≤ f) :
    ∫⁻ x, ‖f x‖₊ ∂μ = ∫⁻ x, ENNReal.ofReal (f x) ∂μ :=
  lintegral_nnnorm_eq_of_ae_nonneg (Filter.Eventually.of_forall h_nonneg)


/-- **Monotone convergence theorem** -- sometimes called **Beppo-Levi convergence**.
See `lintegral_iSup_directed` for a more general form. -/
theorem lintegral_iSup {f : ℕ → α → ℝ≥0∞} (hf : ∀ n, Measurable (f n)) (h_mono : Monotone f) :
    ∫⁻ a, ⨆ n, f n a ∂μ = ⨆ n, ∫⁻ a, f n a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun n => f n a) (iSup fun n => M …
  -/
  set c : ℝ≥0 → ℝ≥0∞ := (↑)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun n => f n a) (iSup fun n => M …
  -/
  set F := fun a : α => ⨆ n, f n a
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    F : α → ENNReal := fun a => iSup fun n => f n a
    ⊢ Eq (MeasureTheory.lintegral μ F) (iSup fun n => MeasureTheory.lintegral μ fu …
  -/
  refine le_antisymm ?_ (iSup_lintegral_le _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    F : α → ENNReal := fun a => iSup fun n => f n a
    ⊢ LE.le (MeasureTheory.lintegral μ F) (iSup fun n => MeasureTheory.lintegral μ …
  -/
  rw [lintegral_eq_nnreal]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    F : α → ENNReal := fun a => iSup fun n => f n a
    ⊢ LE.le (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofN …
  -/
  refine iSup_le fun s => iSup_le fun hsf => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    F : α → ENNReal := fun a => iSup fun n => f n a
    s : MeasureTheory.SimpleFunc α NNReal
    hsf : ∀ (x : α), LE.le (↑(s x)) (iSup fun n => f n x)
    ⊢ LE.le ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal s).lintegral μ) (iSup  …
  -/
  refine ENNReal.le_of_forall_lt_one_mul_le fun a ha => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    F : α → ENNReal := fun a => iSup fun n => f n a
    s : MeasureTheory.SimpleFunc α NNReal
    hsf : ∀ (x : α), LE.le (↑(s x)) (iSup fun n => f n x)
    a : ENNReal
    ha : LT.lt a 1
    ⊢ LE.le (HMul.hMul a ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal s).linteg …
  -/
  rcases ENNReal.lt_iff_exists_coe.1 ha with ⟨r, rfl, _⟩
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    F : α → ENNReal := fun a => iSup fun n => f n a
    s : MeasureTheory.SimpleFunc α NNReal
    hsf : ∀ (x : α), LE.le (↑(s x)) (iSup fun n => f n x)
    r : NNReal
    right✝ : LT.lt (↑r) 1
    ha : LT.lt (↑r) 1
    ⊢ LE.le (HMul.hMul (↑r) ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal s).lin …
  -/
  have ha : r < 1 := ENNReal.coe_lt_coe.1 ha
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    F : α → ENNReal := fun a => iSup fun n => f n a
    s : MeasureTheory.SimpleFunc α NNReal
    hsf : ∀ (x : α), LE.le (↑(s x)) (iSup fun n => f n x)
    r : NNReal
    right✝ : LT.lt (↑r) 1
    ha✝ : LT.lt (↑r) 1
    ha : LT.lt r 1
    ⊢ LE.le (HMul.hMul (↑r) ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal s).lin …
  -/
  let rs := s.map fun a => r * a
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), Measurable (f n)
    h_mono : Monotone f
    c : NNReal → ENNReal := ENNReal.ofNNReal
    F : α → ENNReal := fun a => iSup fun n => f n a
    s : MeasureTheory.SimpleFunc α NNReal
    hsf : ∀ (x : α), LE.le (↑(s x)) (iSup fun n => f n x)
    r : NNReal
    right✝ : LT.lt (↑r) 1
    ha✝ : LT.lt (↑r) 1
    ha : LT.lt r 1
    rs : MeasureTheory.SimpleFunc α NNReal := MeasureTheory.SimpleFunc.map (fun a  …
    ⊢ LE.le (HMul.hMul (↑r) ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal s).lin …
  -/
  have eq_rs : rs.map c = (const α r : α →ₛ ℝ≥0∞) * map c s := rfl
  have eq : ∀ p, rs.map c ⁻¹' {p} = ⋃ n, rs.map c ⁻¹' {p} ∩ { a | p ≤ f n a } := by
    intro p
    rw [← inter_iUnion]; nth_rw 1 [← inter_univ (map c rs ⁻¹' {p})]
    refine Set.ext fun x => and_congr_right fun hx => (iff_of_eq (true_iff _)).2 ?_
    by_cases p_eq : p = 0
    · simp [p_eq]
    simp only [coe_map, mem_preimage, Function.comp_apply, mem_singleton_iff] at hx
    subst hx
    have : r * s x ≠ 0 := by rwa [Ne, ← ENNReal.coe_eq_zero]
    have : s x ≠ 0 := right_ne_zero_of_mul this
    have : (rs.map c) x < ⨆ n : ℕ, f n x := by
      refine lt_of_lt_of_le (ENNReal.coe_lt_coe.2 ?_) (hsf x)
      suffices r * s x < 1 * s x by simpa
      exact mul_lt_mul_of_pos_right ha (pos_iff_ne_zero.2 this)
    rcases lt_iSup_iff.1 this with ⟨i, hi⟩
    exact mem_iUnion.2 ⟨i, le_of_lt hi⟩
  have mono : ∀ r : ℝ≥0∞, Monotone fun n => rs.map c ⁻¹' {r} ∩ { a | r ≤ f n a } := by
    intro r i j h
    refine inter_subset_inter_right _ ?_
    simp_rw [subset_def, mem_setOf]
    intro x hx
    exact le_trans hx (h_mono h x)
  have h_meas : ∀ n, MeasurableSet {a : α | map c rs a ≤ f n a} := fun n =>
    measurableSet_le (SimpleFunc.measurable _) (hf n)
  calc
    (r : ℝ≥0∞) * (s.map c).lintegral μ = ∑ r ∈ (rs.map c).range, r * μ (rs.map c ⁻¹' {r}) := by
      rw [← const_mul_lintegral, eq_rs, SimpleFunc.lintegral]
    _ = ∑ r ∈ (rs.map c).range, r * μ (⋃ n, rs.map c ⁻¹' {r} ∩ { a | r ≤ f n a }) := by
      simp only [(eq _).symm]
    _ = ∑ r ∈ (rs.map c).range, ⨆ n, r * μ (rs.map c ⁻¹' {r} ∩ { a | r ≤ f n a }) :=
      Finset.sum_congr rfl fun x _ => by rw [(mono x).measure_iUnion, ENNReal.mul_iSup]
    _ = ⨆ n, ∑ r ∈ (rs.map c).range, r * μ (rs.map c ⁻¹' {r} ∩ { a | r ≤ f n a }) := by
      refine ENNReal.finsetSum_iSup_of_monotone fun p i j h ↦ ?_
      gcongr _ * μ ?_
      exact mono p h
    _ ≤ ⨆ n : ℕ, ((rs.map c).restrict { a | (rs.map c) a ≤ f n a }).lintegral μ := by
      gcongr with n
      rw [restrict_lintegral _ (h_meas n)]
      refine le_of_eq (Finset.sum_congr rfl fun r _ => ?_)
      congr 2 with a
      refine and_congr_right ?_
      simp +contextual
    _ ≤ ⨆ n, ∫⁻ a, f n a ∂μ := by
      simp only [← SimpleFunc.lintegral_eq_lintegral]
      gcongr with n a
      simp only [map_apply] at h_meas
      simp only [coe_map, restrict_apply _ (h_meas _), (· ∘ ·)]
      exact indicator_apply_le id


/-- Monotone convergence theorem -- sometimes called Beppo-Levi convergence. Version with
ae_measurable functions. -/
theorem lintegral_iSup' {f : ℕ → α → ℝ≥0∞} (hf : ∀ n, AEMeasurable (f n) μ)
    (h_mono : ∀ᵐ x ∂μ, Monotone fun n => f n x) : ∫⁻ a, ⨆ n, f n a ∂μ = ⨆ n, ∫⁻ a, f n a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun n => f n a) (iSup fun n => M …
  -/
  simp_rw [← iSup_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup (fun i => f i) a) (iSup fun n => …
  -/
  let p : α → (ℕ → ℝ≥0∞) → Prop := fun _ f' => Monotone f'
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    p : α → (Nat → ENNReal) → Prop := fun x f' => Monotone f'
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup (fun i => f i) a) (iSup fun n => …
  -/
  have hp : ∀ᵐ x ∂μ, p x fun i => f i x := h_mono
  have h_ae_seq_mono : Monotone (aeSeq hf p) := by
    intro n m hnm x
    by_cases hx : x ∈ aeSeqSet hf p
    · exact aeSeq.prop_of_mem_aeSeqSet hf hx hnm
    · simp only [aeSeq, hx, if_false, le_rfl]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    p : α → (Nat → ENNReal) → Prop := fun x f' => Monotone f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_mono : Monotone (aeSeq hf p)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup (fun i => f i) a) (iSup fun n => …
  -/
  rw [lintegral_congr_ae (aeSeq.iSup hf hp).symm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    p : α → (Nat → ENNReal) → Prop := fun x f' => Monotone f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_mono : Monotone (aeSeq hf p)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup (fun n => aeSeq hf p n) a) (iSup …
  -/
  simp_rw [iSup_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    p : α → (Nat → ENNReal) → Prop := fun x f' => Monotone f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_mono : Monotone (aeSeq hf p)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun i => aeSeq hf p i a) (iSup f …
  -/
  rw [lintegral_iSup (aeSeq.measurable hf p) h_ae_seq_mono]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    p : α → (Nat → ENNReal) → Prop := fun x f' => Monotone f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_mono : Monotone (aeSeq hf p)
    ⊢ Eq (iSup fun n => MeasureTheory.lintegral μ fun a => aeSeq hf p n a) (iSup f …
  -/
  congr with n
  /-
    case e_s.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    p : α → (Nat → ENNReal) → Prop := fun x f' => Monotone f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_mono : Monotone (aeSeq hf p)
    n : Nat
    ⊢ Eq (MeasureTheory.lintegral μ fun a => aeSeq hf p n a) (MeasureTheory.linteg …
  -/
  exact lintegral_congr_ae (aeSeq.aeSeq_n_eq_fun_n_ae hf hp n)
  /-
    🎉 no goals
  -/


/-- Monotone convergence theorem expressed with limits -/
theorem lintegral_tendsto_of_tendsto_of_monotone {f : ℕ → α → ℝ≥0∞} {F : α → ℝ≥0∞}
    (hf : ∀ n, AEMeasurable (f n) μ) (h_mono : ∀ᵐ x ∂μ, Monotone fun n => f n x)
    (h_tendsto : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 <| F x)) :
    Tendsto (fun n => ∫⁻ x, f n x ∂μ) atTop (𝓝 <| ∫⁻ x, F x ∂μ) := by
  have : Monotone fun n => ∫⁻ x, f n x ∂μ := fun i j hij =>
    lintegral_mono_ae (h_mono.mono fun x hx => hx hij)
  suffices key : ∫⁻ x, F x ∂μ = ⨆ n, ∫⁻ x, f n x ∂μ by
    rw [key]
    exact tendsto_atTop_iSup this
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    F : α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    this : Monotone fun n => MeasureTheory.lintegral μ fun x => f n x
    ⊢ Eq (MeasureTheory.lintegral μ fun x => F x) (iSup fun n => MeasureTheory.lin …
  -/
  rw [← lintegral_iSup' hf h_mono]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    F : α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_mono : Filter.Eventually (fun x => Monotone fun n => f n x) (MeasureTheory.a …
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    this : Monotone fun n => MeasureTheory.lintegral μ fun x => f n x
    ⊢ Eq (MeasureTheory.lintegral μ fun x => F x) (MeasureTheory.lintegral μ fun a …
  -/
  refine lintegral_congr_ae ?_
  filter_upwards [h_mono, h_tendsto] with _ hx_mono hx_tendsto using
    tendsto_nhds_unique hx_tendsto (tendsto_atTop_iSup hx_mono)


theorem lintegral_eq_iSup_eapprox_lintegral {f : α → ℝ≥0∞} (hf : Measurable f) :
    ∫⁻ a, f a ∂μ = ⨆ n, (eapprox f n).lintegral μ :=
  calc
    ∫⁻ a, f a ∂μ = ∫⁻ a, ⨆ n, (eapprox f n : α → ℝ≥0∞) a ∂μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        hf : Measurable f
        ⊢ Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ fun a …
      -/
      congr; ext a; rw [iSup_eapprox_apply hf]
                    /-
                      🎉 no goals
                    -/
    _ = ⨆ n, ∫⁻ a, (eapprox f n : α → ℝ≥0∞) a ∂μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        hf : Measurable f
        ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun n => (MeasureTheory.SimpleFu …
      -/
      apply lintegral_iSup
        /-
          case hf
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : α → ENNReal
          hf : Measurable f
          ⊢ ∀ (n : Nat), Measurable ⇑(MeasureTheory.SimpleFunc.eapprox f n)
        -/
      · measurability
        /-
          🎉 no goals
        -/
        /-
          case h_mono
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : α → ENNReal
          hf : Measurable f
          ⊢ Monotone fun n => ⇑(MeasureTheory.SimpleFunc.eapprox f n)
        -/
      · intro i j h
        /-
          case h_mono
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : α → ENNReal
          hf : Measurable f
          i j : Nat
          h : LE.le i j
          ⊢ LE.le ((fun n => ⇑(MeasureTheory.SimpleFunc.eapprox f n)) i) ((fun n => ⇑(Me …
        -/
        exact monotone_eapprox f h
        /-
          🎉 no goals
        -/
    _ = ⨆ n, (eapprox f n).lintegral μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → ENNReal
        hf : Measurable f
        ⊢ Eq (iSup fun n => MeasureTheory.lintegral μ fun a => (MeasureTheory.SimpleFu …
      -/
      congr; ext n; rw [(eapprox f n).lintegral_eq_lintegral]
                    /-
                      🎉 no goals
                    -/


lemma lintegral_eapprox_le_lintegral {f : α → ℝ≥0∞} (hf : Measurable f) (n : ℕ) :
    (eapprox f n).lintegral μ ≤ ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    n : Nat
    ⊢ LE.le ((MeasureTheory.SimpleFunc.eapprox f n).lintegral μ) (MeasureTheory.li …
  -/
  rw [lintegral_eq_iSup_eapprox_lintegral hf]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    n : Nat
    ⊢ LE.le ((MeasureTheory.SimpleFunc.eapprox f n).lintegral μ) (iSup fun n => (M …
  -/
  exact le_iSup (fun n ↦ (eapprox f n).lintegral μ) n
  /-
    🎉 no goals
  -/


lemma measure_support_eapprox_lt_top {f : α → ℝ≥0∞} (hf_meas : Measurable f)
    (hf : ∫⁻ x, f x ∂μ ≠ ∞) (n : ℕ) :
    μ (support (eapprox f n)) < ∞ :=
  measure_support_lt_top_of_lintegral_ne_top <|
    ((lintegral_eapprox_le_lintegral hf_meas n).trans_lt hf.lt_top).ne


/-- If `f` has finite integral, then `∫⁻ x in s, f x ∂μ` is absolutely continuous in `s`: it tends
to zero as `μ s` tends to zero. This lemma states this fact in terms of `ε` and `δ`. -/
theorem exists_pos_setLIntegral_lt_of_measure_lt {f : α → ℝ≥0∞} (h : ∫⁻ x, f x ∂μ ≠ ∞) {ε : ℝ≥0∞}
    (hε : ε ≠ 0) : ∃ δ > 0, ∀ s, μ s < δ → ∫⁻ x in s, f x ∂μ < ε := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (s : Set α), LT.lt (μ s) δ → LT.lt (Measu …
  -/
  rcases exists_between (pos_iff_ne_zero.mpr hε) with ⟨ε₂, hε₂0, hε₂ε⟩
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ε₂ : ENNReal
    hε₂0 : LT.lt 0 ε₂
    hε₂ε : LT.lt ε₂ ε
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (s : Set α), LT.lt (μ s) δ → LT.lt (Measu …
  -/
  rcases exists_between hε₂0 with ⟨ε₁, hε₁0, hε₁₂⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ε₂ : ENNReal
    hε₂0 : LT.lt 0 ε₂
    hε₂ε : LT.lt ε₂ ε
    ε₁ : ENNReal
    hε₁0 : LT.lt 0 ε₁
    hε₁₂ : LT.lt ε₁ ε₂
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (s : Set α), LT.lt (μ s) δ → LT.lt (Measu …
  -/
  rcases exists_simpleFunc_forall_lintegral_sub_lt_of_pos h hε₁0.ne' with ⟨φ, _, hφ⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ε₂ : ENNReal
    hε₂0 : LT.lt 0 ε₂
    hε₂ε : LT.lt ε₂ ε
    ε₁ : ENNReal
    hε₁0 : LT.lt 0 ε₁
    hε₁₂ : LT.lt ε₁ ε₂
    φ : MeasureTheory.SimpleFunc α NNReal
    left✝ : ∀ (x : α), LE.le (↑(φ x)) (f x)
    hφ : ∀ (ψ : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(ψ x)) (f  …
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (s : Set α), LT.lt (μ s) δ → LT.lt (Measu …
  -/
  rcases φ.exists_forall_le with ⟨C, hC⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ε₂ : ENNReal
    hε₂0 : LT.lt 0 ε₂
    hε₂ε : LT.lt ε₂ ε
    ε₁ : ENNReal
    hε₁0 : LT.lt 0 ε₁
    hε₁₂ : LT.lt ε₁ ε₂
    φ : MeasureTheory.SimpleFunc α NNReal
    left✝ : ∀ (x : α), LE.le (↑(φ x)) (f x)
    hφ : ∀ (ψ : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(ψ x)) (f  …
    C : NNReal
    hC : ∀ (x : α), LE.le (φ x) C
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (s : Set α), LT.lt (μ s) δ → LT.lt (Measu …
  -/
  use (ε₂ - ε₁) / C, ENNReal.div_pos_iff.2 ⟨(tsub_pos_iff_lt.2 hε₁₂).ne', ENNReal.coe_ne_top⟩
  /-
    case right
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ε₂ : ENNReal
    hε₂0 : LT.lt 0 ε₂
    hε₂ε : LT.lt ε₂ ε
    ε₁ : ENNReal
    hε₁0 : LT.lt 0 ε₁
    hε₁₂ : LT.lt ε₁ ε₂
    φ : MeasureTheory.SimpleFunc α NNReal
    left✝ : ∀ (x : α), LE.le (↑(φ x)) (f x)
    hφ : ∀ (ψ : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(ψ x)) (f  …
    C : NNReal
    hC : ∀ (x : α), LE.le (φ x) C
    ⊢ ∀ (s : Set α), LT.lt (μ s) (HDiv.hDiv (HSub.hSub ε₂ ε₁) ↑C) → LT.lt (Measure …
  -/
  refine fun s hs => lt_of_le_of_lt ?_ hε₂ε
  /-
    case right
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ε₂ : ENNReal
    hε₂0 : LT.lt 0 ε₂
    hε₂ε : LT.lt ε₂ ε
    ε₁ : ENNReal
    hε₁0 : LT.lt 0 ε₁
    hε₁₂ : LT.lt ε₁ ε₂
    φ : MeasureTheory.SimpleFunc α NNReal
    left✝ : ∀ (x : α), LE.le (↑(φ x)) (f x)
    hφ : ∀ (ψ : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(ψ x)) (f  …
    C : NNReal
    hC : ∀ (x : α), LE.le (φ x) C
    s : Set α
    hs : LT.lt (μ s) (HDiv.hDiv (HSub.hSub ε₂ ε₁) ↑C)
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => f x) ε₂
  -/
  simp only [lintegral_eq_nnreal, iSup_le_iff]
  /-
    case right
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ε₂ : ENNReal
    hε₂0 : LT.lt 0 ε₂
    hε₂ε : LT.lt ε₂ ε
    ε₁ : ENNReal
    hε₁0 : LT.lt 0 ε₁
    hε₁₂ : LT.lt ε₁ ε₂
    φ : MeasureTheory.SimpleFunc α NNReal
    left✝ : ∀ (x : α), LE.le (↑(φ x)) (f x)
    hφ : ∀ (ψ : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(ψ x)) (f  …
    C : NNReal
    hC : ∀ (x : α), LE.le (φ x) C
    s : Set α
    hs : LT.lt (μ s) (HDiv.hDiv (HSub.hSub ε₂ ε₁) ↑C)
    ⊢ ∀ (i : MeasureTheory.SimpleFunc α NNReal), (∀ (x : α), LE.le (↑(i x)) (f x)) …
  -/
  intro ψ hψ
  calc
    (map (↑) ψ).lintegral (μ.restrict s) ≤
        (map (↑) φ).lintegral (μ.restrict s) + (map (↑) (ψ - φ)).lintegral (μ.restrict s) := by
      rw [← SimpleFunc.add_lintegral, ← SimpleFunc.map_add @ENNReal.coe_add]
      refine SimpleFunc.lintegral_mono (fun x => ?_) le_rfl
      simp only [add_tsub_eq_max, le_max_right, coe_map, Function.comp_apply, SimpleFunc.coe_add,
        SimpleFunc.coe_sub, Pi.add_apply, Pi.sub_apply, ENNReal.coe_max (φ x) (ψ x)]
    _ ≤ (map (↑) φ).lintegral (μ.restrict s) + ε₁ := by
      gcongr
      refine le_trans ?_ (hφ _ hψ).le
      exact SimpleFunc.lintegral_mono le_rfl Measure.restrict_le_self
    _ ≤ (SimpleFunc.const α (C : ℝ≥0∞)).lintegral (μ.restrict s) + ε₁ := by
      gcongr
      exact fun x ↦ ENNReal.coe_le_coe.2 (hC x)
    _ = C * μ s + ε₁ := by
      simp only [← SimpleFunc.lintegral_eq_lintegral, coe_const, lintegral_const,
        Measure.restrict_apply, MeasurableSet.univ, univ_inter, Function.const]
    _ ≤ C * ((ε₂ - ε₁) / C) + ε₁ := by gcongr
    _ ≤ ε₂ - ε₁ + ε₁ := by gcongr; apply mul_div_le
    _ = ε₂ := tsub_add_cancel_of_le hε₁₂.le


@[deprecated (since := "2024-06-29")]
alias exists_pos_set_lintegral_lt_of_measure_lt := exists_pos_setLIntegral_lt_of_measure_lt


/-- If `f` has finite integral, then `∫⁻ x in s, f x ∂μ` is absolutely continuous in `s`: it tends
to zero as `μ s` tends to zero. -/
theorem tendsto_setLIntegral_zero {ι} {f : α → ℝ≥0∞} (h : ∫⁻ x, f x ∂μ ≠ ∞) {l : Filter ι}
    {s : ι → Set α} (hl : Tendsto (μ ∘ s) l (𝓝 0)) :
    Tendsto (fun i => ∫⁻ x in s i, f x ∂μ) l (𝓝 0) := by
  simp only [ENNReal.nhds_zero, tendsto_iInf, tendsto_principal, mem_Iio,
    ← pos_iff_ne_zero] at hl ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    l : Filter ι
    s : ι → Set α
    hl : ∀ (i : ENNReal), LT.lt 0 i → Filter.Eventually (fun a => LT.lt (Function. …
    ⊢ ∀ (i : ENNReal), LT.lt 0 i → Filter.Eventually (fun a => LT.lt (MeasureTheor …
  -/
  intro ε ε0
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    l : Filter ι
    s : ι → Set α
    hl : ∀ (i : ENNReal), LT.lt 0 i → Filter.Eventually (fun a => LT.lt (Function. …
    ε : ENNReal
    ε0 : LT.lt 0 ε
    ⊢ Filter.Eventually (fun a => LT.lt (MeasureTheory.lintegral (μ.restrict (s a) …
  -/
  rcases exists_pos_setLIntegral_lt_of_measure_lt h ε0.ne' with ⟨δ, δ0, hδ⟩
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    f : α → ENNReal
    h : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    l : Filter ι
    s : ι → Set α
    hl : ∀ (i : ENNReal), LT.lt 0 i → Filter.Eventually (fun a => LT.lt (Function. …
    ε : ENNReal
    ε0 : LT.lt 0 ε
    δ : ENNReal
    δ0 : GT.gt δ 0
    hδ : ∀ (s : Set α), LT.lt (μ s) δ → LT.lt (MeasureTheory.lintegral (μ.restrict …
    ⊢ Filter.Eventually (fun a => LT.lt (MeasureTheory.lintegral (μ.restrict (s a) …
  -/
  exact (hl δ δ0).mono fun i => hδ _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias tendsto_set_lintegral_zero := tendsto_setLIntegral_zero


/-- The sum of the lower Lebesgue integrals of two functions is less than or equal to the integral
of their sum. The other inequality needs one of these functions to be (a.e.-)measurable. -/
theorem le_lintegral_add (f g : α → ℝ≥0∞) :
    ∫⁻ a, f a ∂μ + ∫⁻ a, g a ∂μ ≤ ∫⁻ a, f a + g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    ⊢ LE.le (HAdd.hAdd (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lin …
  -/
  simp only [lintegral]
  refine ENNReal.biSup_add_biSup_le' (p := fun h : α →ₛ ℝ≥0∞ => h ≤ f)
    (q := fun h : α →ₛ ℝ≥0∞ => h ≤ g) ⟨0, zero_le f⟩ ⟨0, zero_le g⟩ fun f' hf' g' hg' => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    f' : MeasureTheory.SimpleFunc α ENNReal
    hf' : (fun h => LE.le (⇑h) f) f'
    g' : MeasureTheory.SimpleFunc α ENNReal
    hg' : (fun h => LE.le (⇑h) g) g'
    ⊢ LE.le (HAdd.hAdd (f'.lintegral μ) (g'.lintegral μ)) (iSup fun g_1 => iSup fu …
  -/
  exact le_iSup₂_of_le (f' + g') (add_le_add hf' hg') (add_lintegral _ _).ge
  /-
    🎉 no goals
  -/

-- Use stronger lemmas `lintegral_add_left`/`lintegral_add_right` instead

theorem lintegral_add_aux {f g : α → ℝ≥0∞} (hf : Measurable f) (hg : Measurable g) :
    ∫⁻ a, f a + g a ∂μ = ∫⁻ a, f a ∂μ + ∫⁻ a, g a ∂μ :=
  calc
    ∫⁻ a, f a + g a ∂μ =
        ∫⁻ a, (⨆ n, (eapprox f n : α → ℝ≥0∞) a) + ⨆ n, (eapprox g n : α → ℝ≥0∞) a ∂μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : α → ENNReal
        hf : Measurable f
        hg : Measurable g
        ⊢ Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd (f a) (g a)) (MeasureTheory …
      -/
      simp only [iSup_eapprox_apply, hf, hg]
      /-
        🎉 no goals
      -/
    _ = ∫⁻ a, ⨆ n, (eapprox f n + eapprox g n : α → ℝ≥0∞) a ∂μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : α → ENNReal
        hf : Measurable f
        hg : Measurable g
        ⊢ Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd (iSup fun n => (MeasureTheo …
      -/
      congr; funext a
      /-
        case e_f.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : α → ENNReal
        hf : Measurable f
        hg : Measurable g
        a : α
        ⊢ Eq (HAdd.hAdd (iSup fun n => (MeasureTheory.SimpleFunc.eapprox f n) a) (iSup …
      -/
      rw [ENNReal.iSup_add_iSup_of_monotone]
        /-
          case e_f.h
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          a : α
          ⊢ Eq (iSup fun a_1 => HAdd.hAdd ((MeasureTheory.SimpleFunc.eapprox f a_1) a) ( …
        -/
      · simp only [Pi.add_apply]
        /-
          🎉 no goals
        -/
        /-
          case e_f.h.hf
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          a : α
          ⊢ Monotone fun n => (MeasureTheory.SimpleFunc.eapprox f n) a
        -/
      · intro i j h
        /-
          case e_f.h.hf
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          a : α
          i j : Nat
          h : LE.le i j
          ⊢ LE.le ((fun n => (MeasureTheory.SimpleFunc.eapprox f n) a) i) ((fun n => (Me …
        -/
        exact monotone_eapprox _ h a
        /-
          🎉 no goals
        -/
        /-
          case e_f.h.hg
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          a : α
          ⊢ Monotone fun n => (MeasureTheory.SimpleFunc.eapprox g n) a
        -/
      · intro i j h
        /-
          case e_f.h.hg
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          a : α
          i j : Nat
          h : LE.le i j
          ⊢ LE.le ((fun n => (MeasureTheory.SimpleFunc.eapprox g n) a) i) ((fun n => (Me …
        -/
        exact monotone_eapprox _ h a
        /-
          🎉 no goals
        -/
    _ = ⨆ n, (eapprox f n).lintegral μ + (eapprox g n).lintegral μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : α → ENNReal
        hf : Measurable f
        hg : Measurable g
        ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun n => HAdd.hAdd (⇑(MeasureThe …
      -/
      rw [lintegral_iSup]
        /-
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          ⊢ Eq (iSup fun n => MeasureTheory.lintegral μ fun a => HAdd.hAdd (⇑(MeasureThe …
        -/
      · congr
        /-
          case e_s
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          ⊢ Eq (fun n => MeasureTheory.lintegral μ fun a => HAdd.hAdd (⇑(MeasureTheory.S …
        -/
        funext n
        /-
          case e_s.h
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          n : Nat
          ⊢ Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd (⇑(MeasureTheory.SimpleFunc …
        -/
        rw [← SimpleFunc.add_lintegral, ← SimpleFunc.lintegral_eq_lintegral]
        /-
          case e_s.h
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          n : Nat
          ⊢ Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd (⇑(MeasureTheory.SimpleFunc …
        -/
        simp only [Pi.add_apply, SimpleFunc.coe_add]
        /-
          🎉 no goals
        -/
        /-
          case hf
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          ⊢ ∀ (n : Nat), Measurable (HAdd.hAdd ⇑(MeasureTheory.SimpleFunc.eapprox f n) ⇑ …
        -/
      · fun_prop
        /-
          🎉 no goals
        -/
        /-
          case h_mono
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          ⊢ Monotone fun n => HAdd.hAdd ⇑(MeasureTheory.SimpleFunc.eapprox f n) ⇑(Measur …
        -/
      · intro i j h a
        /-
          case h_mono
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          i j : Nat
          h : LE.le i j
          a : α
          ⊢ LE.le ((fun n => HAdd.hAdd ⇑(MeasureTheory.SimpleFunc.eapprox f n) ⇑(Measure …
        -/
        dsimp
        /-
          case h_mono
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f g : α → ENNReal
          hf : Measurable f
          hg : Measurable g
          i j : Nat
          h : LE.le i j
          a : α
          ⊢ LE.le (HAdd.hAdd ((MeasureTheory.SimpleFunc.eapprox f i) a) ((MeasureTheory. …
        -/
                   /-
                     🎉 no goals
                   -/
        gcongr <;> exact monotone_eapprox _ h _
                   /-
                     🎉 no goals
                   -/
    _ = (⨆ n, (eapprox f n).lintegral μ) + ⨆ n, (eapprox g n).lintegral μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : α → ENNReal
        hf : Measurable f
        hg : Measurable g
        ⊢ Eq (iSup fun n => HAdd.hAdd ((MeasureTheory.SimpleFunc.eapprox f n).lintegra …
      -/
      refine (ENNReal.iSup_add_iSup_of_monotone ?_ ?_).symm <;>
          /-
            case refine_1
            α : Type u_1
            m : MeasurableSpace α
            μ : MeasureTheory.Measure α
            f g : α → ENNReal
            hf : Measurable f
            hg : Measurable g
            ⊢ Monotone fun n => (MeasureTheory.SimpleFunc.eapprox f n).lintegral μ
          -/
          /-
            case refine_1
            α : Type u_1
            m : MeasurableSpace α
            μ : MeasureTheory.Measure α
            f g : α → ENNReal
            hf : Measurable f
            hg : Measurable g
            i j : Nat
            h : LE.le i j
            ⊢ LE.le ((fun n => (MeasureTheory.SimpleFunc.eapprox f n).lintegral μ) i) ((fu …
          -/
          /-
            🎉 no goals
          -/
          /-
            case refine_2
            α : Type u_1
            m : MeasurableSpace α
            μ : MeasureTheory.Measure α
            f g : α → ENNReal
            hf : Measurable f
            hg : Measurable g
            i j : Nat
            h : LE.le i j
            ⊢ LE.le ((fun n => (MeasureTheory.SimpleFunc.eapprox g n).lintegral μ) i) ((fu …
          -/
          exact SimpleFunc.lintegral_mono (monotone_eapprox _ h) le_rfl
          /-
            🎉 no goals
          -/
    _ = ∫⁻ a, f a ∂μ + ∫⁻ a, g a ∂μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : α → ENNReal
        hf : Measurable f
        hg : Measurable g
        ⊢ Eq (HAdd.hAdd (iSup fun n => (MeasureTheory.SimpleFunc.eapprox f n).lintegra …
      -/
      rw [lintegral_eq_iSup_eapprox_lintegral hf, lintegral_eq_iSup_eapprox_lintegral hg]
      /-
        🎉 no goals
      -/


/-- If `f g : α → ℝ≥0∞` are two functions and one of them is (a.e.) measurable, then the Lebesgue
integral of `f + g` equals the sum of integrals. This lemma assumes that `f` is integrable, see also
`MeasureTheory.lintegral_add_right` and primed versions of these lemmas. -/
@[simp]
theorem lintegral_add_left {f : α → ℝ≥0∞} (hf : Measurable f) (g : α → ℝ≥0∞) :
    ∫⁻ a, f a + g a ∂μ = ∫⁻ a, f a ∂μ + ∫⁻ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    g : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd (f a) (g a)) (HAdd.hAdd (Me …
  -/
  refine le_antisymm ?_ (le_lintegral_add _ _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    g : α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HAdd.hAdd (f a) (g a)) (HAdd.hAdd  …
  -/
  rcases exists_measurable_le_lintegral_eq μ fun a => f a + g a with ⟨φ, hφm, hφ_le, hφ_eq⟩
  calc
    ∫⁻ a, f a + g a ∂μ = ∫⁻ a, φ a ∂μ := hφ_eq
    _ ≤ ∫⁻ a, f a + (φ a - f a) ∂μ := lintegral_mono fun a => le_add_tsub
    _ = ∫⁻ a, f a ∂μ + ∫⁻ a, φ a - f a ∂μ := lintegral_add_aux hf (hφm.sub hf)
    _ ≤ ∫⁻ a, f a ∂μ + ∫⁻ a, g a ∂μ :=
      add_le_add_left (lintegral_mono fun a => tsub_le_iff_left.2 <| hφ_le a) _


theorem lintegral_add_left' {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) (g : α → ℝ≥0∞) :
    ∫⁻ a, f a + g a ∂μ = ∫⁻ a, f a ∂μ + ∫⁻ a, g a ∂μ := by
  rw [lintegral_congr_ae hf.ae_eq_mk, ← lintegral_add_left hf.measurable_mk,
    lintegral_congr_ae (hf.ae_eq_mk.add (ae_eq_refl g))]


theorem lintegral_add_right' (f : α → ℝ≥0∞) {g : α → ℝ≥0∞} (hg : AEMeasurable g μ) :
    ∫⁻ a, f a + g a ∂μ = ∫⁻ a, f a ∂μ + ∫⁻ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hg : AEMeasurable g μ
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd (f a) (g a)) (HAdd.hAdd (Me …
  -/
  simpa only [add_comm] using lintegral_add_left' hg f
  /-
    🎉 no goals
  -/


/-- If `f g : α → ℝ≥0∞` are two functions and one of them is (a.e.) measurable, then the Lebesgue
integral of `f + g` equals the sum of integrals. This lemma assumes that `g` is integrable, see also
`MeasureTheory.lintegral_add_left` and primed versions of these lemmas. -/
@[simp]
theorem lintegral_add_right (f : α → ℝ≥0∞) {g : α → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ a, f a + g a ∂μ = ∫⁻ a, f a ∂μ + ∫⁻ a, g a ∂μ :=
  lintegral_add_right' f hg.aemeasurable


@[simp]
theorem lintegral_smul_measure (c : ℝ≥0∞) (f : α → ℝ≥0∞) : ∫⁻ a, f a ∂c • μ = c * ∫⁻ a, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (HSMul.hSMul c μ) fun a => f a) (HMul.hMul c (Me …
  -/
  simp only [lintegral, iSup_subtype', SimpleFunc.lintegral_smul, ENNReal.mul_iSup, smul_eq_mul]
  /-
    🎉 no goals
  -/


lemma setLIntegral_smul_measure (c : ℝ≥0∞) (f : α → ℝ≥0∞) (s : Set α) :
    ∫⁻ a in s, f a ∂(c • μ) = c * ∫⁻ a in s, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    f : α → ENNReal
    s : Set α
    ⊢ Eq (MeasureTheory.lintegral ((HSMul.hSMul c μ).restrict s) fun a => f a) (HM …
  -/
  rw [Measure.restrict_smul, lintegral_smul_measure]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_smul_measure := setLIntegral_smul_measure


@[simp]
theorem lintegral_zero_measure {m : MeasurableSpace α} (f : α → ℝ≥0∞) :
    ∫⁻ a, f a ∂(0 : Measure α) = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral 0 fun a => f a) 0
  -/
  simp [lintegral]
  /-
    🎉 no goals
  -/


@[simp]
theorem lintegral_add_measure (f : α → ℝ≥0∞) (μ ν : Measure α) :
    ∫⁻ a, f a ∂(μ + ν) = ∫⁻ a, f a ∂μ + ∫⁻ a, f a ∂ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    ⊢ Eq (MeasureTheory.lintegral (HAdd.hAdd μ ν) fun a => f a) (HAdd.hAdd (Measur …
  -/
  simp only [lintegral, SimpleFunc.lintegral_add, iSup_subtype']
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    ⊢ Eq (iSup fun x => HAdd.hAdd ((↑x).lintegral μ) ((↑x).lintegral ν)) (HAdd.hAd …
  -/
  refine (ENNReal.iSup_add_iSup ?_).symm
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    ⊢ ∀ (i j : Subtype fun i => LE.le ⇑i fun a => f a), Exists fun k => LE.le (HAd …
  -/
  rintro ⟨φ, hφ⟩ ⟨ψ, hψ⟩
  /-
    case mk.mk
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    φ : MeasureTheory.SimpleFunc α ENNReal
    hφ : LE.le ⇑φ fun a => f a
    ψ : MeasureTheory.SimpleFunc α ENNReal
    hψ : LE.le ⇑ψ fun a => f a
    ⊢ Exists fun k => LE.le (HAdd.hAdd ((↑⟨φ, hφ⟩).lintegral μ) ((↑⟨ψ, hψ⟩).linteg …
  -/
  refine ⟨⟨φ ⊔ ψ, sup_le hφ hψ⟩, ?_⟩
  /-
    case mk.mk
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    φ : MeasureTheory.SimpleFunc α ENNReal
    hφ : LE.le ⇑φ fun a => f a
    ψ : MeasureTheory.SimpleFunc α ENNReal
    hψ : LE.le ⇑ψ fun a => f a
    ⊢ LE.le (HAdd.hAdd ((↑⟨φ, hφ⟩).lintegral μ) ((↑⟨ψ, hψ⟩).lintegral ν)) (HAdd.hA …
  -/
  apply_rules [add_le_add, SimpleFunc.lintegral_mono, le_rfl] -- TODO: use `gcongr`
  /-
    case mk.mk.h₁.hfg
    α : Type u_1
    m : MeasurableSpace α
    f : α → ENNReal
    μ ν : MeasureTheory.Measure α
    φ : MeasureTheory.SimpleFunc α ENNReal
    hφ : LE.le ⇑φ fun a => f a
    ψ : MeasureTheory.SimpleFunc α ENNReal
    hψ : LE.le ⇑ψ fun a => f a
    ⊢ LE.le ↑⟨φ, hφ⟩ ↑⟨Max.max φ ψ, ⋯⟩
  -/
  exacts [le_sup_left, le_sup_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem lintegral_finset_sum_measure {ι} (s : Finset ι) (f : α → ℝ≥0∞) (μ : ι → Measure α) :
    ∫⁻ a, f a ∂(∑ i ∈ s, μ i) = ∑ i ∈ s, ∫⁻ a, f a ∂μ i :=
  let F : Measure α →+ ℝ≥0∞ :=
    { toFun := (lintegral · f),
      map_zero' := lintegral_zero_measure f,
      map_add' := lintegral_add_measure f }
  map_sum F μ s


@[simp]
theorem lintegral_sum_measure {m : MeasurableSpace α} {ι} (f : α → ℝ≥0∞) (μ : ι → Measure α) :
    ∫⁻ a, f a ∂Measure.sum μ = ∑' i, ∫⁻ a, f a ∂μ i := by
  simp_rw [ENNReal.tsum_eq_iSup_sum, ← lintegral_finset_sum_measure,
    lintegral, SimpleFunc.lintegral_sum, ENNReal.tsum_eq_iSup_sum,
    SimpleFunc.lintegral_finset_sum, iSup_comm (ι := Finset ι)]


theorem hasSum_lintegral_measure {ι} {_ : MeasurableSpace α} (f : α → ℝ≥0∞) (μ : ι → Measure α) :
    HasSum (fun i => ∫⁻ a, f a ∂μ i) (∫⁻ a, f a ∂Measure.sum μ) :=
  (lintegral_sum_measure f μ).symm ▸ ENNReal.summable.hasSum


@[simp]
theorem lintegral_of_isEmpty {α} [MeasurableSpace α] [IsEmpty α] (μ : Measure α) (f : α → ℝ≥0∞) :
    ∫⁻ x, f x ∂μ = 0 := by
  /-
    α : Type u_5
    inst✝¹ : MeasurableSpace α
    inst✝ : IsEmpty α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f x) 0
  -/
  have : Subsingleton (Measure α) := inferInstance
  /-
    α : Type u_5
    inst✝¹ : MeasurableSpace α
    inst✝ : IsEmpty α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    this : Subsingleton (MeasureTheory.Measure α)
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f x) 0
  -/
  convert lintegral_zero_measure f
  /-
    🎉 no goals
  -/


theorem setLIntegral_empty (f : α → ℝ≥0∞) : ∫⁻ x in ∅, f x ∂μ = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict EmptyCollection.emptyCollection) fun …
  -/
  rw [Measure.restrict_empty, lintegral_zero_measure]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_empty := setLIntegral_empty


theorem setLIntegral_univ (f : α → ℝ≥0∞) : ∫⁻ x in univ, f x ∂μ = ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict Set.univ) fun x => f x) (MeasureTheo …
  -/
  rw [Measure.restrict_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_univ := setLIntegral_univ


theorem setLIntegral_measure_zero (s : Set α) (f : α → ℝ≥0∞) (hs' : μ s = 0) :
    ∫⁻ x in s, f x ∂μ = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hs' : Eq (μ s) 0
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => f x) 0
  -/
  convert lintegral_zero_measure _
  /-
    case h.e'_2.h.e'_3.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hs' : Eq (μ s) 0
    ⊢ Eq (μ.restrict s) 0
  -/
  exact Measure.restrict_eq_zero.2 hs'
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_measure_zero := setLIntegral_measure_zero


theorem lintegral_finset_sum' (s : Finset β) {f : β → α → ℝ≥0∞}
    (hf : ∀ b ∈ s, AEMeasurable (f b) μ) :
    ∫⁻ a, ∑ b ∈ s, f b a ∂μ = ∑ b ∈ s, ∫⁻ a, f b a ∂μ := by
  classical
  induction' s using Finset.induction_on with a s has ih
  · simp
  · simp only [Finset.sum_insert has]
    rw [Finset.forall_mem_insert] at hf
    rw [lintegral_add_left' hf.1, ih hf.2]


theorem lintegral_finset_sum (s : Finset β) {f : β → α → ℝ≥0∞} (hf : ∀ b ∈ s, Measurable (f b)) :
    ∫⁻ a, ∑ b ∈ s, f b a ∂μ = ∑ b ∈ s, ∫⁻ a, f b a ∂μ :=
  lintegral_finset_sum' s fun b hb => (hf b hb).aemeasurable


@[simp]
theorem lintegral_const_mul (r : ℝ≥0∞) {f : α → ℝ≥0∞} (hf : Measurable f) :
    ∫⁻ a, r * f a ∂μ = r * ∫⁻ a, f a ∂μ :=
  calc
    ∫⁻ a, r * f a ∂μ = ∫⁻ a, ⨆ n, (const α r * eapprox f n) a ∂μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        r : ENNReal
        f : α → ENNReal
        hf : Measurable f
        ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (MeasureTheory.lin …
      -/
      congr
      /-
        case e_f
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        r : ENNReal
        f : α → ENNReal
        hf : Measurable f
        ⊢ Eq (fun a => HMul.hMul r (f a)) fun a => iSup fun n => (HMul.hMul (MeasureTh …
      -/
      funext a
      /-
        case e_f.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        r : ENNReal
        f : α → ENNReal
        hf : Measurable f
        a : α
        ⊢ Eq (HMul.hMul r (f a)) (iSup fun n => (HMul.hMul (MeasureTheory.SimpleFunc.c …
      -/
      rw [← iSup_eapprox_apply hf, ENNReal.mul_iSup]
      /-
        case e_f.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        r : ENNReal
        f : α → ENNReal
        hf : Measurable f
        a : α
        ⊢ Eq (iSup fun i => HMul.hMul r ((MeasureTheory.SimpleFunc.eapprox f i) a)) (i …
      -/
      simp
      /-
        🎉 no goals
      -/
    _ = ⨆ n, r * (eapprox f n).lintegral μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        r : ENNReal
        f : α → ENNReal
        hf : Measurable f
        ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun n => (HMul.hMul (MeasureTheo …
      -/
      rw [lintegral_iSup]
        /-
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          r : ENNReal
          f : α → ENNReal
          hf : Measurable f
          ⊢ Eq (iSup fun n => MeasureTheory.lintegral μ fun a => (HMul.hMul (MeasureTheo …
        -/
      · congr
        /-
          case e_s
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          r : ENNReal
          f : α → ENNReal
          hf : Measurable f
          ⊢ Eq (fun n => MeasureTheory.lintegral μ fun a => (HMul.hMul (MeasureTheory.Si …
        -/
        funext n
        /-
          case e_s.h
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          r : ENNReal
          f : α → ENNReal
          hf : Measurable f
          n : Nat
          ⊢ Eq (MeasureTheory.lintegral μ fun a => (HMul.hMul (MeasureTheory.SimpleFunc. …
        -/
        rw [← SimpleFunc.const_mul_lintegral, ← SimpleFunc.lintegral_eq_lintegral]
        /-
          🎉 no goals
        -/
        /-
          case hf
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          r : ENNReal
          f : α → ENNReal
          hf : Measurable f
          ⊢ ∀ (n : Nat), Measurable ⇑(HMul.hMul (MeasureTheory.SimpleFunc.const α r) (Me …
        -/
      · intro n
        /-
          case hf
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          r : ENNReal
          f : α → ENNReal
          hf : Measurable f
          n : Nat
          ⊢ Measurable ⇑(HMul.hMul (MeasureTheory.SimpleFunc.const α r) (MeasureTheory.S …
        -/
        exact SimpleFunc.measurable _
        /-
          🎉 no goals
        -/
        /-
          case h_mono
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          r : ENNReal
          f : α → ENNReal
          hf : Measurable f
          ⊢ Monotone fun n => ⇑(HMul.hMul (MeasureTheory.SimpleFunc.const α r) (MeasureT …
        -/
      · intro i j h a
        /-
          case h_mono
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          r : ENNReal
          f : α → ENNReal
          hf : Measurable f
          i j : Nat
          h : LE.le i j
          a : α
          ⊢ LE.le ((fun n => ⇑(HMul.hMul (MeasureTheory.SimpleFunc.const α r) (MeasureTh …
        -/
        exact mul_le_mul_left' (monotone_eapprox _ h _) _
        /-
          🎉 no goals
        -/
                               /-
                                 α : Type u_1
                                 m : MeasurableSpace α
                                 μ : MeasureTheory.Measure α
                                 r : ENNReal
                                 f : α → ENNReal
                                 hf : Measurable f
                                 ⊢ Eq (iSup fun n => HMul.hMul r ((MeasureTheory.SimpleFunc.eapprox f n).linteg …
                               -/
    _ = r * ∫⁻ a, f a ∂μ := by rw [← ENNReal.mul_iSup, lintegral_eq_iSup_eapprox_lintegral hf]
                               /-
                                 🎉 no goals
                               -/


theorem lintegral_const_mul'' (r : ℝ≥0∞) {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) :
    ∫⁻ a, r * f a ∂μ = r * ∫⁻ a, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (HMul.hMul r (Meas …
  -/
  have A : ∫⁻ a, f a ∂μ = ∫⁻ a, hf.mk f a ∂μ := lintegral_congr_ae hf.ae_eq_mk
  have B : ∫⁻ a, r * f a ∂μ = ∫⁻ a, r * hf.mk f a ∂μ :=
    lintegral_congr_ae (EventuallyEq.fun_comp hf.ae_eq_mk _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hf : AEMeasurable f μ
    A : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ fun …
    B : Eq (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (MeasureTheory.l …
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (HMul.hMul r (Meas …
  -/
  rw [A, B, lintegral_const_mul _ hf.measurable_mk]
  /-
    🎉 no goals
  -/


theorem lintegral_const_mul_le (r : ℝ≥0∞) (f : α → ℝ≥0∞) :
    r * ∫⁻ a, f a ∂μ ≤ ∫⁻ a, r * f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    ⊢ LE.le (HMul.hMul r (MeasureTheory.lintegral μ fun a => f a)) (MeasureTheory. …
  -/
  rw [lintegral, ENNReal.mul_iSup]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    ⊢ LE.le (iSup fun i => HMul.hMul r (iSup fun x => i.lintegral μ)) (MeasureTheo …
  -/
  refine iSup_le fun s => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    s : MeasureTheory.SimpleFunc α ENNReal
    ⊢ LE.le (HMul.hMul r (iSup fun x => s.lintegral μ)) (MeasureTheory.lintegral μ …
  -/
  rw [ENNReal.mul_iSup, iSup_le_iff]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    s : MeasureTheory.SimpleFunc α ENNReal
    ⊢ (LE.le ⇑s fun a => f a) → LE.le (HMul.hMul r (s.lintegral μ)) (MeasureTheory …
  -/
  intro hs
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    s : MeasureTheory.SimpleFunc α ENNReal
    hs : LE.le ⇑s fun a => f a
    ⊢ LE.le (HMul.hMul r (s.lintegral μ)) (MeasureTheory.lintegral μ fun a => HMul …
  -/
  rw [← SimpleFunc.const_mul_lintegral, lintegral]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    s : MeasureTheory.SimpleFunc α ENNReal
    hs : LE.le ⇑s fun a => f a
    ⊢ LE.le ((HMul.hMul (MeasureTheory.SimpleFunc.const α r) s).lintegral μ) (iSup …
  -/
  refine le_iSup_of_le (const α r * s) (le_iSup_of_le (fun x => ?_) le_rfl)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    s : MeasureTheory.SimpleFunc α ENNReal
    hs : LE.le ⇑s fun a => f a
    x : α
    ⊢ LE.le ((HMul.hMul (MeasureTheory.SimpleFunc.const α r) s) x) ((fun a => HMul …
  -/
  exact mul_le_mul_left' (hs x) _
  /-
    🎉 no goals
  -/


theorem lintegral_const_mul' (r : ℝ≥0∞) (f : α → ℝ≥0∞) (hr : r ≠ ∞) :
    ∫⁻ a, r * f a ∂μ = r * ∫⁻ a, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hr : Ne r Top.top
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (HMul.hMul r (Meas …
  -/
  by_cases h : r = 0
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      r : ENNReal
      f : α → ENNReal
      hr : Ne r Top.top
      h : Eq r 0
      ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (HMul.hMul r (Meas …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hr : Ne r Top.top
    h : Not (Eq r 0)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (HMul.hMul r (Meas …
  -/
  apply le_antisymm _ (lintegral_const_mul_le r f)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hr : Ne r Top.top
    h : Not (Eq r 0)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (HMul.hMul r (M …
  -/
  have rinv : r * r⁻¹ = 1 := ENNReal.mul_inv_cancel h hr
  have rinv' : r⁻¹ * r = 1 := by
    rw [mul_comm]
    exact rinv
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hr : Ne r Top.top
    h : Not (Eq r 0)
    rinv : Eq (HMul.hMul r (Inv.inv r)) 1
    rinv' : Eq (HMul.hMul (Inv.inv r) r) 1
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (HMul.hMul r (M …
  -/
  have := lintegral_const_mul_le (μ := μ) r⁻¹ fun x => r * f x
  simp? [(mul_assoc _ _ _).symm, rinv'] at this says
    simp only [(mul_assoc _ _ _).symm, rinv', one_mul] at this
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    hr : Ne r Top.top
    h : Not (Eq r 0)
    rinv : Eq (HMul.hMul r (Inv.inv r)) 1
    rinv' : Eq (HMul.hMul (Inv.inv r) r) 1
    this : LE.le (HMul.hMul (Inv.inv r) (MeasureTheory.lintegral μ fun a => HMul.h …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => HMul.hMul r (f a)) (HMul.hMul r (M …
  -/
  simpa [(mul_assoc _ _ _).symm, rinv] using mul_le_mul_left' this r
  /-
    🎉 no goals
  -/


theorem lintegral_mul_const (r : ℝ≥0∞) {f : α → ℝ≥0∞} (hf : Measurable f) :
                                                /-
                                                  α : Type u_1
                                                  m : MeasurableSpace α
                                                  μ : MeasureTheory.Measure α
                                                  r : ENNReal
                                                  f : α → ENNReal
                                                  hf : Measurable f
                                                  ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul (f a) r) (HMul.hMul (Measur …
                                                -/
    ∫⁻ a, f a * r ∂μ = (∫⁻ a, f a ∂μ) * r := by simp_rw [mul_comm, lintegral_const_mul r hf]
                                                /-
                                                  🎉 no goals
                                                -/


theorem lintegral_mul_const'' (r : ℝ≥0∞) {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) :
                                                /-
                                                  α : Type u_1
                                                  m : MeasurableSpace α
                                                  μ : MeasureTheory.Measure α
                                                  r : ENNReal
                                                  f : α → ENNReal
                                                  hf : AEMeasurable f μ
                                                  ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul (f a) r) (HMul.hMul (Measur …
                                                -/
    ∫⁻ a, f a * r ∂μ = (∫⁻ a, f a ∂μ) * r := by simp_rw [mul_comm, lintegral_const_mul'' r hf]
                                                /-
                                                  🎉 no goals
                                                -/


theorem lintegral_mul_const_le (r : ℝ≥0∞) (f : α → ℝ≥0∞) :
    (∫⁻ a, f a ∂μ) * r ≤ ∫⁻ a, f a * r ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    r : ENNReal
    f : α → ENNReal
    ⊢ LE.le (HMul.hMul (MeasureTheory.lintegral μ fun a => f a) r) (MeasureTheory. …
  -/
  simp_rw [mul_comm, lintegral_const_mul_le r f]
  /-
    🎉 no goals
  -/


theorem lintegral_mul_const' (r : ℝ≥0∞) (f : α → ℝ≥0∞) (hr : r ≠ ∞) :
                                                /-
                                                  α : Type u_1
                                                  m : MeasurableSpace α
                                                  μ : MeasureTheory.Measure α
                                                  r : ENNReal
                                                  f : α → ENNReal
                                                  hr : Ne r Top.top
                                                  ⊢ Eq (MeasureTheory.lintegral μ fun a => HMul.hMul (f a) r) (HMul.hMul (Measur …
                                                -/
    ∫⁻ a, f a * r ∂μ = (∫⁻ a, f a ∂μ) * r := by simp_rw [mul_comm, lintegral_const_mul' r f hr]
                                                /-
                                                  🎉 no goals
                                                -/

/- A double integral of a product where each factor contains only one variable
  is a product of integrals -/

theorem lintegral_lintegral_mul {β} [MeasurableSpace β] {ν : Measure β} {f : α → ℝ≥0∞}
    {g : β → ℝ≥0∞} (hf : AEMeasurable f μ) (hg : AEMeasurable g ν) :
    ∫⁻ x, ∫⁻ y, f x * g y ∂ν ∂μ = (∫⁻ x, f x ∂μ) * ∫⁻ y, g y ∂ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f : α → ENNReal
    g : β → ENNReal
    hf : AEMeasurable f μ
    hg : AEMeasurable g ν
    ⊢ Eq (MeasureTheory.lintegral μ fun x => MeasureTheory.lintegral ν fun y => HM …
  -/
  simp [lintegral_const_mul'' _ hg, lintegral_mul_const'' _ hf]
  /-
    🎉 no goals
  -/

-- TODO: Need a better way of rewriting inside of an integral

theorem lintegral_rw₁ {f f' : α → β} (h : f =ᵐ[μ] f') (g : β → ℝ≥0∞) :
    ∫⁻ a, g (f a) ∂μ = ∫⁻ a, g (f' a) ∂μ :=
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               m : MeasurableSpace α
                                               μ : MeasureTheory.Measure α
                                               f f' : α → β
                                               h✝ : (MeasureTheory.ae μ).EventuallyEq f f'
                                               g : β → ENNReal
                                               a : α
                                               h : Eq (f a) (f' a)
                                               ⊢ Eq ((fun a => g (f a)) a) ((fun a => g (f' a)) a)
                                             -/
  lintegral_congr_ae <| h.mono fun a h => by dsimp only; rw [h]
                                                         /-
                                                           🎉 no goals
                                                         -/

-- TODO: Need a better way of rewriting inside of an integral

theorem lintegral_rw₂ {f₁ f₁' : α → β} {f₂ f₂' : α → γ} (h₁ : f₁ =ᵐ[μ] f₁') (h₂ : f₂ =ᵐ[μ] f₂')
    (g : β → γ → ℝ≥0∞) : ∫⁻ a, g (f₁ a) (f₂ a) ∂μ = ∫⁻ a, g (f₁' a) (f₂' a) ∂μ :=
                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             γ : Type u_3
                                                             m : MeasurableSpace α
                                                             μ : MeasureTheory.Measure α
                                                             f₁ f₁' : α → β
                                                             f₂ f₂' : α → γ
                                                             h₁✝ : (MeasureTheory.ae μ).EventuallyEq f₁ f₁'
                                                             h₂✝ : (MeasureTheory.ae μ).EventuallyEq f₂ f₂'
                                                             g : β → γ → ENNReal
                                                             x✝ : α
                                                             h₂ : Eq (f₂ x✝) (f₂' x✝)
                                                             h₁ : Eq (f₁ x✝) (f₁' x✝)
                                                             ⊢ Eq ((fun a => g (f₁ a) (f₂ a)) x✝) ((fun a => g (f₁' a) (f₂' a)) x✝)
                                                           -/
  lintegral_congr_ae <| h₁.mp <| h₂.mono fun _ h₂ h₁ => by dsimp only; rw [h₁, h₂]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem lintegral_indicator_le (f : α → ℝ≥0∞) (s : Set α) :
    ∫⁻ a, s.indicator f a ∂μ ≤ ∫⁻ a in s, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => s.indicator f a) (MeasureTheory.li …
  -/
  simp only [lintegral]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    ⊢ LE.le (iSup fun g => iSup fun x => g.lintegral μ) (iSup fun g => iSup fun x  …
  -/
  apply iSup_le (fun g ↦ (iSup_le (fun hg ↦ ?_)))
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    ⊢ LE.le (g.lintegral μ) (iSup fun g => iSup fun x => g.lintegral (μ.restrict s))
  -/
  have : g ≤ f := hg.trans (indicator_le_self s f)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    ⊢ LE.le (g.lintegral μ) (iSup fun g => iSup fun x => g.lintegral (μ.restrict s))
  -/
  refine le_iSup_of_le g (le_iSup_of_le this (le_of_eq ?_))
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    ⊢ Eq (g.lintegral μ) (g.lintegral (μ.restrict s))
  -/
  rw [lintegral_restrict, SimpleFunc.lintegral]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    ⊢ Eq (g.range.sum fun x => HMul.hMul x (μ (Set.preimage (⇑g) (Singleton.single …
  -/
  congr with t
  /-
    case e_f.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    t : ENNReal
    ⊢ Eq (HMul.hMul t (μ (Set.preimage (⇑g) (Singleton.singleton t)))) (HMul.hMul  …
  -/
  by_cases H : t = 0
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      s : Set α
      g : MeasureTheory.SimpleFunc α ENNReal
      hg : LE.le ⇑g fun a => s.indicator f a
      this : LE.le (⇑g) f
      t : ENNReal
      H : Eq t 0
      ⊢ Eq (HMul.hMul t (μ (Set.preimage (⇑g) (Singleton.singleton t)))) (HMul.hMul  …
    -/
  · simp [H]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    t : ENNReal
    H : Not (Eq t 0)
    ⊢ Eq (HMul.hMul t (μ (Set.preimage (⇑g) (Singleton.singleton t)))) (HMul.hMul  …
  -/
  congr with x
  /-
    case neg.e_a.h.e_6.h.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    t : ENNReal
    H : Not (Eq t 0)
    x : α
    ⊢ Iff (Membership.mem (Set.preimage (⇑g) (Singleton.singleton t)) x) (Membersh …
  -/
  simp only [mem_preimage, mem_singleton_iff, mem_inter_iff, iff_self_and]
  /-
    case neg.e_a.h.e_6.h.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    t : ENNReal
    H : Not (Eq t 0)
    x : α
    ⊢ Eq (g x) t → Membership.mem s x
  -/
  rintro rfl
  /-
    case neg.e_a.h.e_6.h.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    x : α
    H : Not (Eq (g x) 0)
    ⊢ Membership.mem s x
  -/
  contrapose! H
  /-
    case neg.e_a.h.e_6.h.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    g : MeasureTheory.SimpleFunc α ENNReal
    hg : LE.le ⇑g fun a => s.indicator f a
    this : LE.le (⇑g) f
    x : α
    H : Not (Membership.mem s x)
    ⊢ Eq (g x) 0
  -/
  simpa [H] using hg x
  /-
    🎉 no goals
  -/


@[simp]
theorem lintegral_indicator {s : Set α} (hs : MeasurableSet s) (f : α → ℝ≥0∞) :
    ∫⁻ a, s.indicator f a ∂μ = ∫⁻ a in s, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun a => s.indicator f a) (MeasureTheory.linte …
  -/
  apply le_antisymm (lintegral_indicator_le f s)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory.l …
  -/
  simp only [lintegral, ← restrict_lintegral_eq_lintegral_restrict _ hs, iSup_subtype']
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    ⊢ LE.le (iSup fun x => ((↑x).restrict s).lintegral μ) (iSup fun x => (↑x).lint …
  -/
  refine iSup_mono' (Subtype.forall.2 fun φ hφ => ?_)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    φ : MeasureTheory.SimpleFunc α ENNReal
    hφ : LE.le ⇑φ fun a => f a
    ⊢ Exists fun i' => LE.le (((↑⟨φ, hφ⟩).restrict s).lintegral μ) ((↑i').lintegra …
  -/
  refine ⟨⟨φ.restrict s, fun x => ?_⟩, le_rfl⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    φ : MeasureTheory.SimpleFunc α ENNReal
    hφ : LE.le ⇑φ fun a => f a
    x : α
    ⊢ LE.le ((φ.restrict s) x) ((fun a => s.indicator f a) x)
  -/
  simp [hφ x, hs, indicator_le_indicator]
  /-
    🎉 no goals
  -/


lemma setLIntegral_indicator {s t : Set α} (hs : MeasurableSet s) (f : α → ℝ≥0∞) :
    ∫⁻ a in t, s.indicator f a ∂μ = ∫⁻ a in s ∩ t, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict t) fun a => s.indicator f a) (Measur …
  -/
  rw [lintegral_indicator hs, Measure.restrict_restrict hs]
  /-
    🎉 no goals
  -/


theorem lintegral_indicator₀ {s : Set α} (hs : NullMeasurableSet s μ) (f : α → ℝ≥0∞) :
    ∫⁻ a, s.indicator f a ∂μ = ∫⁻ a in s, f a ∂μ := by
  rw [← lintegral_congr_ae (indicator_ae_eq_of_ae_eq_set hs.toMeasurable_ae_eq),
    lintegral_indicator (measurableSet_toMeasurable _ _),
    Measure.restrict_congr_set hs.toMeasurable_ae_eq]


lemma setLIntegral_indicator₀ (f : α → ℝ≥0∞) {s t : Set α}
    (hs : NullMeasurableSet s (μ.restrict t)) :
    ∫⁻ a in t, s.indicator f a ∂μ = ∫⁻ a in s ∩ t, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s t : Set α
    hs : MeasureTheory.NullMeasurableSet s (μ.restrict t)
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict t) fun a => s.indicator f a) (Measur …
  -/
  rw [lintegral_indicator₀ hs, Measure.restrict_restrict₀ hs]
  /-
    🎉 no goals
  -/


theorem lintegral_indicator_const_le (s : Set α) (c : ℝ≥0∞) :
    ∫⁻ a, s.indicator (fun _ => c) a ∂μ ≤ c * μ s :=
  (lintegral_indicator_le _ _).trans (setLIntegral_const s c).le


theorem lintegral_indicator_const₀ {s : Set α} (hs : NullMeasurableSet s μ) (c : ℝ≥0∞) :
    ∫⁻ a, s.indicator (fun _ => c) a ∂μ = c * μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasureTheory.NullMeasurableSet s μ
    c : ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun a => s.indicator (fun x => c) a) (HMul.hMu …
  -/
  rw [lintegral_indicator₀ hs, setLIntegral_const]
  /-
    🎉 no goals
  -/


theorem lintegral_indicator_const {s : Set α} (hs : MeasurableSet s) (c : ℝ≥0∞) :
    ∫⁻ a, s.indicator (fun _ => c) a ∂μ = c * μ s :=
  lintegral_indicator_const₀ hs.nullMeasurableSet c


lemma setLIntegral_eq_of_support_subset {s : Set α} {f : α → ℝ≥0∞} (hsf : f.support ⊆ s) :
    ∫⁻ x in s, f x ∂μ = ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => f x) (MeasureTheory.lint …
  -/
  apply le_antisymm (setLIntegral_le_lintegral s fun x ↦ f x)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => f x) (MeasureTheory.lintegral (μ.r …
  -/
  apply le_trans (le_of_eq _) (lintegral_indicator_le _ _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f x) (MeasureTheory.lintegral μ fun a …
  -/
  congr with x
  /-
    case e_f.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    x : α
    ⊢ Eq (f x) (s.indicator f x)
  -/
  simp only [indicator]
  /-
    case e_f.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hsf : HasSubset.Subset (Function.support f) s
    x : α
    ⊢ Eq (f x) (ite (Membership.mem s x) (f x) 0)
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hsf : HasSubset.Subset (Function.support f) s
      x : α
      h : Membership.mem s x
      ⊢ Eq (f x) (f x)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hsf : HasSubset.Subset (Function.support f) s
      x : α
      h : Not (Membership.mem s x)
      ⊢ Eq (f x) 0
    -/
  · exact Function.support_subset_iff'.1 hsf x h
    /-
      🎉 no goals
    -/


theorem setLIntegral_eq_const {f : α → ℝ≥0∞} (hf : Measurable f) (r : ℝ≥0∞) :
    ∫⁻ x in { x | f x = r }, f x ∂μ = r * μ { x | f x = r } := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    r : ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (setOf fun x => Eq (f x) r)) fun x = …
  -/
  have : ∀ᵐ x ∂μ, x ∈ { x | f x = r } → f x = r := ae_of_all μ fun _ hx => hx
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    r : ENNReal
    this : Filter.Eventually (fun x => Membership.mem (setOf fun x => Eq (f x) r)  …
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (setOf fun x => Eq (f x) r)) fun x = …
  -/
  rw [setLIntegral_congr_fun _ this]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Measurable f
      r : ENNReal
      this : Filter.Eventually (fun x => Membership.mem (setOf fun x => Eq (f x) r)  …
      ⊢ Eq (MeasureTheory.lintegral (μ.restrict (setOf fun x => Eq (f x) r)) fun x = …
    -/
  · rw [lintegral_const, Measure.restrict_apply MeasurableSet.univ, Set.univ_inter]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Measurable f
      r : ENNReal
      this : Filter.Eventually (fun x => Membership.mem (setOf fun x => Eq (f x) r)  …
      ⊢ MeasurableSet (setOf fun x => Eq (f x) r)
    -/
  · exact hf (measurableSet_singleton r)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_eq_const := setLIntegral_eq_const


theorem lintegral_indicator_one_le (s : Set α) : ∫⁻ a, s.indicator 1 a ∂μ ≤ μ s :=
  (lintegral_indicator_const_le _ _).trans <| (one_mul _).le


@[simp]
theorem lintegral_indicator_one₀ {s : Set α} (hs : NullMeasurableSet s μ) :
    ∫⁻ a, s.indicator 1 a ∂μ = μ s :=
  (lintegral_indicator_const₀ hs _).trans <| one_mul _


@[simp]
theorem lintegral_indicator_one {s : Set α} (hs : MeasurableSet s) :
    ∫⁻ a, s.indicator 1 a ∂μ = μ s :=
  (lintegral_indicator_const hs _).trans <| one_mul _


/-- A version of **Markov's inequality** for two functions. It doesn't follow from the standard
Markov's inequality because we only assume measurability of `g`, not `f`. -/
theorem lintegral_add_mul_meas_add_le_le_lintegral {f g : α → ℝ≥0∞} (hle : f ≤ᵐ[μ] g)
    (hg : AEMeasurable g μ) (ε : ℝ≥0∞) :
    ∫⁻ a, f a ∂μ + ε * μ { x | f x + ε ≤ g x } ≤ ∫⁻ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hle : (MeasureTheory.ae μ).EventuallyLE f g
    hg : AEMeasurable g μ
    ε : ENNReal
    ⊢ LE.le (HAdd.hAdd (MeasureTheory.lintegral μ fun a => f a) (HMul.hMul ε (μ (s …
  -/
  rcases exists_measurable_le_lintegral_eq μ f with ⟨φ, hφm, hφ_le, hφ_eq⟩
  calc
    ∫⁻ x, f x ∂μ + ε * μ { x | f x + ε ≤ g x } = ∫⁻ x, φ x ∂μ + ε * μ { x | f x + ε ≤ g x } := by
      rw [hφ_eq]
    _ ≤ ∫⁻ x, φ x ∂μ + ε * μ { x | φ x + ε ≤ g x } := by
      gcongr
      exact fun x => (add_le_add_right (hφ_le _) _).trans
    _ = ∫⁻ x, φ x + indicator { x | φ x + ε ≤ g x } (fun _ => ε) x ∂μ := by
      rw [lintegral_add_left hφm, lintegral_indicator₀, setLIntegral_const]
      exact measurableSet_le (hφm.nullMeasurable.measurable'.add_const _) hg.nullMeasurable
    _ ≤ ∫⁻ x, g x ∂μ := lintegral_mono_ae (hle.mono fun x hx₁ => ?_)
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hle : (MeasureTheory.ae μ).EventuallyLE f g
    hg : AEMeasurable g μ
    ε : ENNReal
    φ : α → ENNReal
    hφm : Measurable φ
    hφ_le : LE.le φ f
    hφ_eq : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ …
    x : α
    hx₁ : LE.le (f x) (g x)
    ⊢ LE.le (HAdd.hAdd (φ x) ((setOf fun x => LE.le (HAdd.hAdd (φ x) ε) (g x)).ind …
  -/
  simp only [indicator_apply]; split_ifs with hx₂
  /-
    case pos
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hle : (MeasureTheory.ae μ).EventuallyLE f g
    hg : AEMeasurable g μ
    ε : ENNReal
    φ : α → ENNReal
    hφm : Measurable φ
    hφ_le : LE.le φ f
    hφ_eq : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ …
    x : α
    hx₁ : LE.le (f x) (g x)
    hx₂ : Membership.mem (setOf fun x => LE.le (HAdd.hAdd (φ x) ε) (g x)) x
    ⊢ LE.le (HAdd.hAdd (φ x) ε) (g x)
  -/
  exacts [hx₂, (add_zero _).trans_le <| (hφ_le x).trans hx₁]
  /-
    🎉 no goals
  -/


/-- **Markov's inequality** also known as **Chebyshev's first inequality**. -/
theorem mul_meas_ge_le_lintegral₀ {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) (ε : ℝ≥0∞) :
    ε * μ { x | ε ≤ f x } ≤ ∫⁻ a, f a ∂μ := by
  simpa only [lintegral_zero, zero_add] using
    lintegral_add_mul_meas_add_le_le_lintegral (ae_of_all _ fun x => zero_le (f x)) hf ε


/-- **Markov's inequality** also known as **Chebyshev's first inequality**. For a version assuming
`AEMeasurable`, see `mul_meas_ge_le_lintegral₀`. -/
theorem mul_meas_ge_le_lintegral {f : α → ℝ≥0∞} (hf : Measurable f) (ε : ℝ≥0∞) :
    ε * μ { x | ε ≤ f x } ≤ ∫⁻ a, f a ∂μ :=
  mul_meas_ge_le_lintegral₀ hf.aemeasurable ε


lemma meas_le_lintegral₀ {f : α → ℝ≥0∞} (hf : AEMeasurable f μ)
    {s : Set α} (hs : ∀ x ∈ s, 1 ≤ f x) : μ s ≤ ∫⁻ a, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → LE.le 1 (f x)
    ⊢ LE.le (μ s) (MeasureTheory.lintegral μ fun a => f a)
  -/
  apply le_trans _ (mul_meas_ge_le_lintegral₀ hf 1)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → LE.le 1 (f x)
    ⊢ LE.le (μ s) (HMul.hMul 1 (μ (setOf fun x => LE.le 1 (f x))))
  -/
  rw [one_mul]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → LE.le 1 (f x)
    ⊢ LE.le (μ s) (μ (setOf fun x => LE.le 1 (f x)))
  -/
  exact measure_mono hs
  /-
    🎉 no goals
  -/


lemma lintegral_le_meas {s : Set α} {f : α → ℝ≥0∞} (hf : ∀ a, f a ≤ 1) (h'f : ∀ a ∈ sᶜ, f a = 0) :
    ∫⁻ a, f a ∂μ ≤ μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hf : ∀ (a : α), LE.le (f a) 1
    h'f : ∀ (a : α), Membership.mem (HasCompl.compl s) a → Eq (f a) 0
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => f a) (μ s)
  -/
  apply (lintegral_mono (fun x ↦ ?_)).trans (lintegral_indicator_one_le s)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    f : α → ENNReal
    hf : ∀ (a : α), LE.le (f a) 1
    h'f : ∀ (a : α), Membership.mem (HasCompl.compl s) a → Eq (f a) 0
    x : α
    ⊢ LE.le (f x) (s.indicator 1 x)
  -/
  by_cases hx : x ∈ s
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hf : ∀ (a : α), LE.le (f a) 1
      h'f : ∀ (a : α), Membership.mem (HasCompl.compl s) a → Eq (f a) 0
      x : α
      hx : Membership.mem s x
      ⊢ LE.le (f x) (s.indicator 1 x)
    -/
  · simpa [hx] using hf x
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      f : α → ENNReal
      hf : ∀ (a : α), LE.le (f a) 1
      h'f : ∀ (a : α), Membership.mem (HasCompl.compl s) a → Eq (f a) 0
      x : α
      hx : Not (Membership.mem s x)
      ⊢ LE.le (f x) (s.indicator 1 x)
    -/
  · simpa [hx] using h'f x hx
    /-
      🎉 no goals
    -/


lemma setLIntegral_le_meas {s t : Set α} (hs : MeasurableSet s)
    {f : α → ℝ≥0∞} (hf : ∀ a ∈ s, a ∈ t → f a ≤ 1)
    (hf' : ∀ a ∈ s, a ∉ t → f a = 0) : ∫⁻ a in s, f a ∂μ ≤ μ t := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t a → LE.le (f a) 1
    hf' : ∀ (a : α), Membership.mem s a → Not (Membership.mem t a) → Eq (f a) 0
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (μ t)
  -/
  rw [← lintegral_indicator hs]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t a → LE.le (f a) 1
    hf' : ∀ (a : α), Membership.mem s a → Not (Membership.mem t a) → Eq (f a) 0
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => s.indicator f a) (μ t)
  -/
  refine lintegral_le_meas (fun a ↦ ?_) (by aesop)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    hf : ∀ (a : α), Membership.mem s a → Membership.mem t a → LE.le (f a) 1
    hf' : ∀ (a : α), Membership.mem s a → Not (Membership.mem t a) → Eq (f a) 0
    a : α
    ⊢ LE.le (s.indicator f a) 1
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  by_cases has : a ∈ s <;> [by_cases hat : a ∈ t; skip] <;> simp [*]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem lintegral_eq_top_of_measure_eq_top_ne_zero {f : α → ℝ≥0∞} (hf : AEMeasurable f μ)
    (hμf : μ {x | f x = ∞} ≠ 0) : ∫⁻ x, f x ∂μ = ∞ :=
  eq_top_iff.mpr <|
    calc
                                      /-
                                        α : Type u_1
                                        m : MeasurableSpace α
                                        μ : MeasureTheory.Measure α
                                        f : α → ENNReal
                                        hf : AEMeasurable f μ
                                        hμf : Ne (μ (setOf fun x => Eq (f x) Top.top)) 0
                                        ⊢ Eq Top.top (HMul.hMul Top.top (μ (setOf fun x => LE.le Top.top (f x))))
                                      -/
      ∞ = ∞ * μ { x | ∞ ≤ f x } := by simp [mul_eq_top, hμf]
                                      /-
                                        🎉 no goals
                                      -/
      _ ≤ ∫⁻ x, f x ∂μ := mul_meas_ge_le_lintegral₀ hf ∞


theorem setLintegral_eq_top_of_measure_eq_top_ne_zero {f : α → ℝ≥0∞} {s : Set α}
    (hf : AEMeasurable f (μ.restrict s)) (hμf : μ ({x ∈ s | f x = ∞}) ≠ 0) :
    ∫⁻ x in s, f x ∂μ = ∞ :=
  lintegral_eq_top_of_measure_eq_top_ne_zero hf <|
                          /-
                            α : Type u_1
                            m : MeasurableSpace α
                            μ : MeasureTheory.Measure α
                            f : α → ENNReal
                            s : Set α
                            hf : AEMeasurable f (μ.restrict s)
                            hμf : Ne (μ (setOf fun x => And (Membership.mem s x) (Eq (f x) Top.top))) 0
                            ⊢ LE.le (μ (setOf fun x => And (Membership.mem s x) (Eq (f x) Top.top))) ((μ.r …
                          -/
    mt (eq_bot_mono <| by rw [← setOf_inter_eq_sep]; exact Measure.le_restrict_apply _ _) hμf
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem measure_eq_top_of_lintegral_ne_top {f : α → ℝ≥0∞}
    (hf : AEMeasurable f μ) (hμf : ∫⁻ x, f x ∂μ ≠ ∞) : μ {x | f x = ∞} = 0 :=
  of_not_not fun h => hμf <| lintegral_eq_top_of_measure_eq_top_ne_zero hf h


theorem measure_eq_top_of_setLintegral_ne_top {f : α → ℝ≥0∞} {s : Set α}
    (hf : AEMeasurable f (μ.restrict s)) (hμf : ∫⁻ x in s, f x ∂μ ≠ ∞) :
    μ ({x ∈ s | f x = ∞}) = 0 :=
  of_not_not fun h => hμf <| setLintegral_eq_top_of_measure_eq_top_ne_zero hf h


/-- **Markov's inequality**, also known as **Chebyshev's first inequality**. -/
theorem meas_ge_le_lintegral_div {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) {ε : ℝ≥0∞} (hε : ε ≠ 0)
    (hε' : ε ≠ ∞) : μ { x | ε ≤ f x } ≤ (∫⁻ a, f a ∂μ) / ε :=
  (ENNReal.le_div_iff_mul_le (Or.inl hε) (Or.inl hε')).2 <| by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : AEMeasurable f μ
      ε : ENNReal
      hε : Ne ε 0
      hε' : Ne ε Top.top
      ⊢ LE.le (HMul.hMul (μ (setOf fun x => LE.le ε (f x))) ε) (MeasureTheory.linteg …
    -/
    rw [mul_comm]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : AEMeasurable f μ
      ε : ENNReal
      hε : Ne ε 0
      hε' : Ne ε Top.top
      ⊢ LE.le (HMul.hMul ε (μ (setOf fun x => LE.le ε (f x)))) (MeasureTheory.linteg …
    -/
    exact mul_meas_ge_le_lintegral₀ hf ε
    /-
      🎉 no goals
    -/


theorem ae_eq_of_ae_le_of_lintegral_le {f g : α → ℝ≥0∞} (hfg : f ≤ᵐ[μ] g) (hf : ∫⁻ x, f x ∂μ ≠ ∞)
    (hg : AEMeasurable g μ) (hgf : ∫⁻ x, g x ∂μ ≤ ∫⁻ x, f x ∂μ) : f =ᵐ[μ] g := by
  have : ∀ n : ℕ, ∀ᵐ x ∂μ, g x < f x + (n : ℝ≥0∞)⁻¹ := by
    intro n
    simp only [ae_iff, not_lt]
    have : ∫⁻ x, f x ∂μ + (↑n)⁻¹ * μ { x : α | f x + (n : ℝ≥0∞)⁻¹ ≤ g x } ≤ ∫⁻ x, f x ∂μ :=
      (lintegral_add_mul_meas_add_le_le_lintegral hfg hg n⁻¹).trans hgf
    rw [(ENNReal.cancel_of_ne hf).add_le_iff_nonpos_right, nonpos_iff_eq_zero, mul_eq_zero] at this
    exact this.resolve_left (ENNReal.inv_ne_zero.2 (ENNReal.natCast_ne_top _))
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    hg : AEMeasurable g μ
    hgf : LE.le (MeasureTheory.lintegral μ fun x => g x) (MeasureTheory.lintegral  …
    this : ∀ (n : Nat), Filter.Eventually (fun x => LT.lt (g x) (HAdd.hAdd (f x) ( …
    ⊢ (MeasureTheory.ae μ).EventuallyEq f g
  -/
  refine hfg.mp ((ae_all_iff.2 this).mono fun x hlt hle => hle.antisymm ?_)
  suffices Tendsto (fun n : ℕ => f x + (n : ℝ≥0∞)⁻¹) atTop (𝓝 (f x)) from
    ge_of_tendsto' this fun i => (hlt i).le
  simpa only [inv_top, add_zero] using
    tendsto_const_nhds.add (ENNReal.tendsto_inv_iff.2 ENNReal.tendsto_nat_nhds_top)


@[simp]
theorem lintegral_eq_zero_iff' {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) :
    ∫⁻ a, f a ∂μ = 0 ↔ f =ᵐ[μ] 0 :=
                                  /-
                                    α : Type u_1
                                    m : MeasurableSpace α
                                    μ : MeasureTheory.Measure α
                                    f : α → ENNReal
                                    hf : AEMeasurable f μ
                                    ⊢ Ne (MeasureTheory.lintegral μ fun x => 0) Top.top
                                  -/
  have : ∫⁻ _ : α, 0 ∂μ ≠ ∞ := by simp [lintegral_zero, zero_ne_top]
                                  /-
                                    🎉 no goals
                                  -/
  ⟨fun h =>
    (ae_eq_of_ae_le_of_lintegral_le (ae_of_all _ <| zero_le f) this hf
        (h.trans lintegral_zero.symm).le).symm,
    fun h => (lintegral_congr_ae h).trans lintegral_zero⟩


@[simp]
theorem lintegral_eq_zero_iff {f : α → ℝ≥0∞} (hf : Measurable f) : ∫⁻ a, f a ∂μ = 0 ↔ f =ᵐ[μ] 0 :=
  lintegral_eq_zero_iff' hf.aemeasurable


theorem setLIntegral_eq_zero_iff' {s : Set α} (hs : MeasurableSet s)
    {f : α → ℝ≥0∞} (hf : AEMeasurable f (μ.restrict s)) :
    ∫⁻ a in s, f a ∂μ = 0 ↔ ∀ᵐ x ∂μ, x ∈ s → f x = 0 :=
  (lintegral_eq_zero_iff' hf).trans (ae_restrict_iff' hs)


theorem setLIntegral_eq_zero_iff {s : Set α} (hs : MeasurableSet s) {f : α → ℝ≥0∞}
    (hf : Measurable f) : ∫⁻ a in s, f a ∂μ = 0 ↔ ∀ᵐ x ∂μ, x ∈ s → f x = 0 :=
  setLIntegral_eq_zero_iff' hs hf.aemeasurable


theorem lintegral_pos_iff_support {f : α → ℝ≥0∞} (hf : Measurable f) :
    (0 < ∫⁻ a, f a ∂μ) ↔ 0 < μ (Function.support f) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    ⊢ Iff (LT.lt 0 (MeasureTheory.lintegral μ fun a => f a)) (LT.lt 0 (μ (Function …
  -/
  simp [pos_iff_ne_zero, hf, Filter.EventuallyEq, ae_iff, Function.support]
  /-
    🎉 no goals
  -/


theorem setLintegral_pos_iff {f : α → ℝ≥0∞} (hf : Measurable f) {s : Set α} :
    0 < ∫⁻ a in s, f a ∂μ ↔ 0 < μ (Function.support f ∩ s) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    s : Set α
    ⊢ Iff (LT.lt 0 (MeasureTheory.lintegral (μ.restrict s) fun a => f a)) (LT.lt 0 …
  -/
  rw [lintegral_pos_iff_support hf, Measure.restrict_apply (measurableSet_support hf)]
  /-
    🎉 no goals
  -/


/-- Weaker version of the monotone convergence theorem -/
theorem lintegral_iSup_ae {f : ℕ → α → ℝ≥0∞} (hf : ∀ n, Measurable (f n))
    (h_mono : ∀ n, ∀ᵐ a ∂μ, f n a ≤ f n.succ a) : ∫⁻ a, ⨆ n, f n a ∂μ = ⨆ n, ∫⁻ a, f n a ∂μ := by
  classical
  let ⟨s, hs⟩ := exists_measurable_superset_of_null (ae_iff.1 (ae_all_iff.2 h_mono))
  let g n a := if a ∈ s then 0 else f n a
  have g_eq_f : ∀ᵐ a ∂μ, ∀ n, g n a = f n a :=
    (measure_zero_iff_ae_nmem.1 hs.2.2).mono fun a ha n => if_neg ha
  calc
    ∫⁻ a, ⨆ n, f n a ∂μ = ∫⁻ a, ⨆ n, g n a ∂μ :=
      lintegral_congr_ae <| g_eq_f.mono fun a ha => by simp only [ha]
    _ = ⨆ n, ∫⁻ a, g n a ∂μ :=
      (lintegral_iSup (fun n => measurable_const.piecewise hs.2.1 (hf n))
        (monotone_nat_of_le_succ fun n a => ?_))
    _ = ⨆ n, ∫⁻ a, f n a ∂μ := by simp only [lintegral_congr_ae (g_eq_f.mono fun _a ha => ha _)]
  simp only [g]
  split_ifs with h
  · rfl
  · have := Set.not_mem_subset hs.1 h
    simp only [not_forall, not_le, mem_setOf_eq, not_exists, not_lt] at this
    exact this n


theorem lintegral_sub' {f g : α → ℝ≥0∞} (hg : AEMeasurable g μ) (hg_fin : ∫⁻ a, g a ∂μ ≠ ∞)
    (h_le : g ≤ᵐ[μ] f) : ∫⁻ a, f a - g a ∂μ = ∫⁻ a, f a ∂μ - ∫⁻ a, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hg : AEMeasurable g μ
    hg_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
    h_le : (MeasureTheory.ae μ).EventuallyLE g f
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HSub.hSub (f a) (g a)) (HSub.hSub (Me …
  -/
  refine ENNReal.eq_sub_of_add_eq hg_fin ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hg : AEMeasurable g μ
    hg_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
    h_le : (MeasureTheory.ae μ).EventuallyLE g f
    ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral μ fun a => HSub.hSub (f a) (g a)) (Me …
  -/
  rw [← lintegral_add_right' _ hg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hg : AEMeasurable g μ
    hg_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
    h_le : (MeasureTheory.ae μ).EventuallyLE g f
    ⊢ Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd (HSub.hSub (f a) (g a)) (g  …
  -/
  exact lintegral_congr_ae (h_le.mono fun x hx => tsub_add_cancel_of_le hx)
  /-
    🎉 no goals
  -/


theorem lintegral_sub {f g : α → ℝ≥0∞} (hg : Measurable g) (hg_fin : ∫⁻ a, g a ∂μ ≠ ∞)
    (h_le : g ≤ᵐ[μ] f) : ∫⁻ a, f a - g a ∂μ = ∫⁻ a, f a ∂μ - ∫⁻ a, g a ∂μ :=
  lintegral_sub' hg.aemeasurable hg_fin h_le


theorem lintegral_sub_le' (f g : α → ℝ≥0∞) (hf : AEMeasurable f μ) :
    ∫⁻ x, g x ∂μ - ∫⁻ x, f x ∂μ ≤ ∫⁻ x, g x - f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ LE.le (HSub.hSub (MeasureTheory.lintegral μ fun x => g x) (MeasureTheory.lin …
  -/
  rw [tsub_le_iff_right]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : AEMeasurable f μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheory.lin …
  -/
  by_cases hfi : ∫⁻ x, f x ∂μ = ∞
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hfi : Eq (MeasureTheory.lintegral μ fun x => f x) Top.top
      ⊢ LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheory.lin …
    -/
  · rw [hfi, add_top]
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hfi : Eq (MeasureTheory.lintegral μ fun x => f x) Top.top
      ⊢ LE.le (MeasureTheory.lintegral μ fun x => g x) Top.top
    -/
    exact le_top
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hfi : Not (Eq (MeasureTheory.lintegral μ fun x => f x) Top.top)
      ⊢ LE.le (MeasureTheory.lintegral μ fun x => g x) (HAdd.hAdd (MeasureTheory.lin …
    -/
  · rw [← lintegral_add_right' _ hf]
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hfi : Not (Eq (MeasureTheory.lintegral μ fun x => f x) Top.top)
      ⊢ LE.le (MeasureTheory.lintegral μ fun x => g x) (MeasureTheory.lintegral μ fu …
    -/
    gcongr
    /-
      case neg.hfg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : AEMeasurable f μ
      hfi : Not (Eq (MeasureTheory.lintegral μ fun x => f x) Top.top)
      x✝ : α
      ⊢ LE.le (g x✝) (HAdd.hAdd (HSub.hSub (g x✝) (f x✝)) (f x✝))
    -/
    exact le_tsub_add
    /-
      🎉 no goals
    -/


theorem lintegral_sub_le (f g : α → ℝ≥0∞) (hf : Measurable f) :
    ∫⁻ x, g x ∂μ - ∫⁻ x, f x ∂μ ≤ ∫⁻ x, g x - f x ∂μ :=
  lintegral_sub_le' f g hf.aemeasurable


theorem lintegral_strict_mono_of_ae_le_of_frequently_ae_lt {f g : α → ℝ≥0∞} (hg : AEMeasurable g μ)
    (hfi : ∫⁻ x, f x ∂μ ≠ ∞) (h_le : f ≤ᵐ[μ] g) (h : ∃ᵐ x ∂μ, f x ≠ g x) :
    ∫⁻ x, f x ∂μ < ∫⁻ x, g x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h_le : (MeasureTheory.ae μ).EventuallyLE f g
    h : Filter.Frequently (fun x => Ne (f x) (g x)) (MeasureTheory.ae μ)
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => f x) (MeasureTheory.lintegral μ fu …
  -/
  contrapose! h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h_le : (MeasureTheory.ae μ).EventuallyLE f g
    h : LE.le (MeasureTheory.lintegral μ fun x => g x) (MeasureTheory.lintegral μ  …
    ⊢ Not (Filter.Frequently (fun x => Ne (f x) (g x)) (MeasureTheory.ae μ))
  -/
  simp only [not_frequently, Ne, Classical.not_not]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h_le : (MeasureTheory.ae μ).EventuallyLE f g
    h : LE.le (MeasureTheory.lintegral μ fun x => g x) (MeasureTheory.lintegral μ  …
    ⊢ Filter.Eventually (fun x => Eq (f x) (g x)) (MeasureTheory.ae μ)
  -/
  exact ae_eq_of_ae_le_of_lintegral_le h_le hfi hg h
  /-
    🎉 no goals
  -/


theorem lintegral_strict_mono_of_ae_le_of_ae_lt_on {f g : α → ℝ≥0∞} (hg : AEMeasurable g μ)
    (hfi : ∫⁻ x, f x ∂μ ≠ ∞) (h_le : f ≤ᵐ[μ] g) {s : Set α} (hμs : μ s ≠ 0)
    (h : ∀ᵐ x ∂μ, x ∈ s → f x < g x) : ∫⁻ x, f x ∂μ < ∫⁻ x, g x ∂μ :=
  lintegral_strict_mono_of_ae_le_of_frequently_ae_lt hg hfi h_le <|
    ((frequently_ae_mem_iff.2 hμs).and_eventually h).mono fun _x hx => (hx.2 hx.1).ne


theorem lintegral_strict_mono {f g : α → ℝ≥0∞} (hμ : μ ≠ 0) (hg : AEMeasurable g μ)
    (hfi : ∫⁻ x, f x ∂μ ≠ ∞) (h : ∀ᵐ x ∂μ, f x < g x) : ∫⁻ x, f x ∂μ < ∫⁻ x, g x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hμ : Ne μ 0
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h : Filter.Eventually (fun x => LT.lt (f x) (g x)) (MeasureTheory.ae μ)
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => f x) (MeasureTheory.lintegral μ fu …
  -/
  rw [Ne, ← Measure.measure_univ_eq_zero] at hμ
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hμ : Not (Eq (μ Set.univ) 0)
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h : Filter.Eventually (fun x => LT.lt (f x) (g x)) (MeasureTheory.ae μ)
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => f x) (MeasureTheory.lintegral μ fu …
  -/
  refine lintegral_strict_mono_of_ae_le_of_ae_lt_on hg hfi (ae_le_of_ae_lt h) hμ ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hμ : Not (Eq (μ Set.univ) 0)
    hg : AEMeasurable g μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h : Filter.Eventually (fun x => LT.lt (f x) (g x)) (MeasureTheory.ae μ)
    ⊢ Filter.Eventually (fun x => Membership.mem Set.univ x → LT.lt (f x) (g x)) ( …
  -/
  simpa using h
  /-
    🎉 no goals
  -/


theorem setLIntegral_strict_mono {f g : α → ℝ≥0∞} {s : Set α} (hsm : MeasurableSet s)
    (hs : μ s ≠ 0) (hg : Measurable g) (hfi : ∫⁻ x in s, f x ∂μ ≠ ∞)
    (h : ∀ᵐ x ∂μ, x ∈ s → f x < g x) : ∫⁻ x in s, f x ∂μ < ∫⁻ x in s, g x ∂μ :=
                            /-
                              α : Type u_1
                              m : MeasurableSpace α
                              μ : MeasureTheory.Measure α
                              f g : α → ENNReal
                              s : Set α
                              hsm : MeasurableSet s
                              hs : Ne (μ s) 0
                              hg : Measurable g
                              hfi : Ne (MeasureTheory.lintegral (μ.restrict s) fun x => f x) Top.top
                              h : Filter.Eventually (fun x => Membership.mem s x → LT.lt (f x) (g x)) (Measu …
                              ⊢ Ne (μ.restrict s) 0
                            -/
  lintegral_strict_mono (by simp [hs]) hg.aemeasurable hfi ((ae_restrict_iff' hsm).mpr h)
                            /-
                              🎉 no goals
                            -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_strict_mono := setLIntegral_strict_mono


/-- Monotone convergence theorem for nonincreasing sequences of functions -/
theorem lintegral_iInf_ae {f : ℕ → α → ℝ≥0∞} (h_meas : ∀ n, Measurable (f n))
    (h_mono : ∀ n : ℕ, f n.succ ≤ᵐ[μ] f n) (h_fin : ∫⁻ a, f 0 a ∂μ ≠ ∞) :
    ∫⁻ a, ⨅ n, f n a ∂μ = ⨅ n, ∫⁻ a, f n a ∂μ :=
  have fn_le_f0 : ∫⁻ a, ⨅ n, f n a ∂μ ≤ ∫⁻ a, f 0 a ∂μ :=
    lintegral_mono fun _ => iInf_le_of_le 0 le_rfl
  have fn_le_f0' : ⨅ n, ∫⁻ a, f n a ∂μ ≤ ∫⁻ a, f 0 a ∂μ := iInf_le_of_le 0 le_rfl
  (ENNReal.sub_right_inj h_fin fn_le_f0 fn_le_f0').1 <|
    show ∫⁻ a, f 0 a ∂μ - ∫⁻ a, ⨅ n, f n a ∂μ = ∫⁻ a, f 0 a ∂μ - ⨅ n, ∫⁻ a, f n a ∂μ from
      calc
        ∫⁻ a, f 0 a ∂μ - ∫⁻ a, ⨅ n, f n a ∂μ = ∫⁻ a, f 0 a - ⨅ n, f n a ∂μ :=
          (lintegral_sub (.iInf h_meas)
              (ne_top_of_le_ne_top h_fin <| lintegral_mono fun _ => iInf_le _ _)
              (ae_of_all _ fun _ => iInf_le _ _)).symm
        _ = ∫⁻ a, ⨆ n, f 0 a - f n a ∂μ := congr rfl (funext fun _ => ENNReal.sub_iInf)
        _ = ⨆ n, ∫⁻ a, f 0 a - f n a ∂μ :=
          (lintegral_iSup_ae (fun n => (h_meas 0).sub (h_meas n)) fun n =>
            (h_mono n).mono fun _ ha => tsub_le_tsub le_rfl ha)
        _ = ⨆ n, ∫⁻ a, f 0 a ∂μ - ∫⁻ a, f n a ∂μ :=
          (have h_mono : ∀ᵐ a ∂μ, ∀ n : ℕ, f n.succ a ≤ f n a := ae_all_iff.2 h_mono
          have h_mono : ∀ n, ∀ᵐ a ∂μ, f n a ≤ f 0 a := fun n =>
            h_mono.mono fun a h => by
              /-
                α : Type u_1
                m : MeasurableSpace α
                μ : MeasureTheory.Measure α
                f : Nat → α → ENNReal
                h_meas : ∀ (n : Nat), Measurable (f n)
                h_mono✝ : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n.succ) (f n)
                h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
                fn_le_f0 : LE.le (MeasureTheory.lintegral μ fun a => iInf fun n => f n a) (Mea …
                fn_le_f0' : LE.le (iInf fun n => MeasureTheory.lintegral μ fun a => f n a) (Me …
                h_mono : Filter.Eventually (fun a => ∀ (n : Nat), LE.le (f n.succ a) (f n a))  …
                n : Nat
                a : α
                h : ∀ (n : Nat), LE.le (f n.succ a) (f n a)
                ⊢ LE.le (f n a) (f 0 a)
              -/
              induction' n with n ih
                /-
                  case zero
                  α : Type u_1
                  m : MeasurableSpace α
                  μ : MeasureTheory.Measure α
                  f : Nat → α → ENNReal
                  h_meas : ∀ (n : Nat), Measurable (f n)
                  h_mono✝ : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n.succ) (f n)
                  h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
                  fn_le_f0 : LE.le (MeasureTheory.lintegral μ fun a => iInf fun n => f n a) (Mea …
                  fn_le_f0' : LE.le (iInf fun n => MeasureTheory.lintegral μ fun a => f n a) (Me …
                  h_mono : Filter.Eventually (fun a => ∀ (n : Nat), LE.le (f n.succ a) (f n a))  …
                  a : α
                  h : ∀ (n : Nat), LE.le (f n.succ a) (f n a)
                  ⊢ LE.le (f 0 a) (f 0 a)
                -/
              · exact le_rfl
                /-
                  🎉 no goals
                -/
                /-
                  case succ
                  α : Type u_1
                  m : MeasurableSpace α
                  μ : MeasureTheory.Measure α
                  f : Nat → α → ENNReal
                  h_meas : ∀ (n : Nat), Measurable (f n)
                  h_mono✝ : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n.succ) (f n)
                  h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
                  fn_le_f0 : LE.le (MeasureTheory.lintegral μ fun a => iInf fun n => f n a) (Mea …
                  fn_le_f0' : LE.le (iInf fun n => MeasureTheory.lintegral μ fun a => f n a) (Me …
                  h_mono : Filter.Eventually (fun a => ∀ (n : Nat), LE.le (f n.succ a) (f n a))  …
                  a : α
                  h : ∀ (n : Nat), LE.le (f n.succ a) (f n a)
                  n : Nat
                  ih : LE.le (f n a) (f 0 a)
                  ⊢ LE.le (f (HAdd.hAdd n 1) a) (f 0 a)
                -/
              · exact le_trans (h n) ih
                /-
                  🎉 no goals
                -/
          congr_arg iSup <|
            funext fun n =>
              lintegral_sub (h_meas _) (ne_top_of_le_ne_top h_fin <| lintegral_mono_ae <| h_mono n)
                (h_mono n))
        _ = ∫⁻ a, f 0 a ∂μ - ⨅ n, ∫⁻ a, f n a ∂μ := ENNReal.sub_iInf.symm


/-- Monotone convergence theorem for nonincreasing sequences of functions -/
theorem lintegral_iInf {f : ℕ → α → ℝ≥0∞} (h_meas : ∀ n, Measurable (f n)) (h_anti : Antitone f)
    (h_fin : ∫⁻ a, f 0 a ∂μ ≠ ∞) : ∫⁻ a, ⨅ n, f n a ∂μ = ⨅ n, ∫⁻ a, f n a ∂μ :=
  lintegral_iInf_ae h_meas (fun n => ae_of_all _ <| h_anti n.le_succ) h_fin


theorem lintegral_iInf' {f : ℕ → α → ℝ≥0∞} (h_meas : ∀ n, AEMeasurable (f n) μ)
    (h_anti : ∀ᵐ a ∂μ, Antitone (fun i ↦ f i a)) (h_fin : ∫⁻ a, f 0 a ∂μ ≠ ∞) :
    ∫⁻ a, ⨅ n, f n a ∂μ = ⨅ n, ∫⁻ a, f n a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
    h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf fun n => f n a) (iInf fun n => M …
  -/
  simp_rw [← iInf_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
    h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf (fun i => f i) a) (iInf fun n => …
  -/
  let p : α → (ℕ → ℝ≥0∞) → Prop := fun _ f' => Antitone f'
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
    h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
    p : α → (Nat → ENNReal) → Prop := fun x f' => Antitone f'
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf (fun i => f i) a) (iInf fun n => …
  -/
  have hp : ∀ᵐ x ∂μ, p x fun i => f i x := h_anti
  have h_ae_seq_mono : Antitone (aeSeq h_meas p) := by
    intro n m hnm x
    by_cases hx : x ∈ aeSeqSet h_meas p
    · exact aeSeq.prop_of_mem_aeSeqSet h_meas hx hnm
    · simp only [aeSeq, hx, if_false]
      exact le_rfl
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
    h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
    p : α → (Nat → ENNReal) → Prop := fun x f' => Antitone f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_mono : Antitone (aeSeq h_meas p)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf (fun i => f i) a) (iInf fun n => …
  -/
  rw [lintegral_congr_ae (aeSeq.iInf h_meas hp).symm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
    h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
    p : α → (Nat → ENNReal) → Prop := fun x f' => Antitone f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_mono : Antitone (aeSeq h_meas p)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf (fun n => aeSeq h_meas p n) a) ( …
  -/
  simp_rw [iInf_apply]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
    h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
    p : α → (Nat → ENNReal) → Prop := fun x f' => Antitone f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_mono : Antitone (aeSeq h_meas p)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf fun i => aeSeq h_meas p i a) (iI …
  -/
  rw [lintegral_iInf (aeSeq.measurable h_meas p) h_ae_seq_mono]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → ENNReal
      h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
      h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
      p : α → (Nat → ENNReal) → Prop := fun x f' => Antitone f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_mono : Antitone (aeSeq h_meas p)
      ⊢ Eq (iInf fun n => MeasureTheory.lintegral μ fun a => aeSeq h_meas p n a) (iI …
    -/
  · congr
    /-
      case e_s
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → ENNReal
      h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
      h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
      p : α → (Nat → ENNReal) → Prop := fun x f' => Antitone f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_mono : Antitone (aeSeq h_meas p)
      ⊢ Eq (fun n => MeasureTheory.lintegral μ fun a => aeSeq h_meas p n a) fun n => …
    -/
    exact funext fun n ↦ lintegral_congr_ae (aeSeq.aeSeq_n_eq_fun_n_ae h_meas hp n)
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : Nat → α → ENNReal
      h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
      h_anti : Filter.Eventually (fun a => Antitone fun i => f i a) (MeasureTheory.a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
      p : α → (Nat → ENNReal) → Prop := fun x f' => Antitone f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_mono : Antitone (aeSeq h_meas p)
      ⊢ Ne (MeasureTheory.lintegral μ fun a => aeSeq h_meas p 0 a) Top.top
    -/
  · rwa [lintegral_congr_ae (aeSeq.aeSeq_n_eq_fun_n_ae h_meas hp 0)]
    /-
      🎉 no goals
    -/


/-- Monotone convergence for an infimum over a directed family and indexed by a countable type -/
theorem lintegral_iInf_directed_of_measurable {mα : MeasurableSpace α} [Countable β]
    {f : β → α → ℝ≥0∞} {μ : Measure α} (hμ : μ ≠ 0) (hf : ∀ b, Measurable (f b))
    (hf_int : ∀ b, ∫⁻ a, f b a ∂μ ≠ ∞) (h_directed : Directed (· ≥ ·) f) :
    ∫⁻ a, ⨅ b, f b a ∂μ = ⨅ b, ∫⁻ a, f b a ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    inst✝ : Countable β
    f : β → α → ENNReal
    μ : MeasureTheory.Measure α
    hμ : Ne μ 0
    hf : ∀ (b : β), Measurable (f b)
    hf_int : ∀ (b : β), Ne (MeasureTheory.lintegral μ fun a => f b a) Top.top
    h_directed : Directed (fun x1 x2 => GE.ge x1 x2) f
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf fun b => f b a) (iInf fun b => M …
  -/
  cases nonempty_encodable β
  /-
    case intro
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    inst✝ : Countable β
    f : β → α → ENNReal
    μ : MeasureTheory.Measure α
    hμ : Ne μ 0
    hf : ∀ (b : β), Measurable (f b)
    hf_int : ∀ (b : β), Ne (MeasureTheory.lintegral μ fun a => f b a) Top.top
    h_directed : Directed (fun x1 x2 => GE.ge x1 x2) f
    val✝ : Encodable β
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf fun b => f b a) (iInf fun b => M …
  -/
  cases isEmpty_or_nonempty β
  · simp only [iInf_of_empty, lintegral_const,
      ENNReal.top_mul (Measure.measure_univ_ne_zero.mpr hμ)]
  /-
    case intro.inr
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    inst✝ : Countable β
    f : β → α → ENNReal
    μ : MeasureTheory.Measure α
    hμ : Ne μ 0
    hf : ∀ (b : β), Measurable (f b)
    hf_int : ∀ (b : β), Ne (MeasureTheory.lintegral μ fun a => f b a) Top.top
    h_directed : Directed (fun x1 x2 => GE.ge x1 x2) f
    val✝ : Encodable β
    h✝ : Nonempty β
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf fun b => f b a) (iInf fun b => M …
  -/
  inhabit β
  have : ∀ a, ⨅ b, f b a = ⨅ n, f (h_directed.sequence f n) a := by
    refine fun a =>
      le_antisymm (le_iInf fun n => iInf_le _ _)
        (le_iInf fun b => iInf_le_of_le (Encodable.encode b + 1) ?_)
    exact h_directed.sequence_le b a
  -- Porting note: used `∘` below to deal with its reduced reducibility
  calc
    ∫⁻ a, ⨅ b, f b a ∂μ
    _ = ∫⁻ a, ⨅ n, (f ∘ h_directed.sequence f) n a ∂μ := by simp only [this, Function.comp_apply]
    _ = ⨅ n, ∫⁻ a, (f ∘ h_directed.sequence f) n a ∂μ := by
      rw [lintegral_iInf ?_ h_directed.sequence_anti]
      · exact hf_int _
      · exact fun n => hf _
    _ = ⨅ b, ∫⁻ a, f b a ∂μ := by
      refine le_antisymm (le_iInf fun b => ?_) (le_iInf fun n => ?_)
      · exact iInf_le_of_le (Encodable.encode b + 1) (lintegral_mono <| h_directed.sequence_le b)
      · exact iInf_le (fun b => ∫⁻ a, f b a ∂μ) _


/-- Known as Fatou's lemma, version with `AEMeasurable` functions -/
theorem lintegral_liminf_le' {f : ℕ → α → ℝ≥0∞} (h_meas : ∀ n, AEMeasurable (f n) μ) :
    ∫⁻ a, liminf (fun n => f n a) atTop ∂μ ≤ liminf (fun n => ∫⁻ a, f n a ∂μ) atTop :=
  calc
    ∫⁻ a, liminf (fun n => f n a) atTop ∂μ = ∫⁻ a, ⨆ n : ℕ, ⨅ i ≥ n, f i a ∂μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : Nat → α → ENNReal
        h_meas : ∀ (n : Nat), AEMeasurable (f n) μ
        ⊢ Eq (MeasureTheory.lintegral μ fun a => Filter.liminf (fun n => f n a) Filter …
      -/
      simp only [liminf_eq_iSup_iInf_of_nat]
      /-
        🎉 no goals
      -/
    _ = ⨆ n : ℕ, ∫⁻ a, ⨅ i ≥ n, f i a ∂μ :=
      (lintegral_iSup' (fun _ => .biInf _ (to_countable _) (fun i _ ↦ h_meas i))
        (ae_of_all μ fun _ _ _ hnm => iInf_le_iInf_of_subset fun _ hi => le_trans hnm hi))
    _ ≤ ⨆ n : ℕ, ⨅ i ≥ n, ∫⁻ a, f i a ∂μ := iSup_mono fun _ => le_iInf₂_lintegral _
    _ = atTop.liminf fun n => ∫⁻ a, f n a ∂μ := Filter.liminf_eq_iSup_iInf_of_nat.symm


/-- Known as Fatou's lemma -/
theorem lintegral_liminf_le {f : ℕ → α → ℝ≥0∞} (h_meas : ∀ n, Measurable (f n)) :
    ∫⁻ a, liminf (fun n => f n a) atTop ∂μ ≤ liminf (fun n => ∫⁻ a, f n a ∂μ) atTop :=
  lintegral_liminf_le' fun n => (h_meas n).aemeasurable


theorem limsup_lintegral_le {f : ℕ → α → ℝ≥0∞} (g : α → ℝ≥0∞) (hf_meas : ∀ n, Measurable (f n))
    (h_bound : ∀ n, f n ≤ᵐ[μ] g) (h_fin : ∫⁻ a, g a ∂μ ≠ ∞) :
    limsup (fun n => ∫⁻ a, f n a ∂μ) atTop ≤ ∫⁻ a, limsup (fun n => f n a) atTop ∂μ :=
  calc
    limsup (fun n => ∫⁻ a, f n a ∂μ) atTop = ⨅ n : ℕ, ⨆ i ≥ n, ∫⁻ a, f i a ∂μ :=
      limsup_eq_iInf_iSup_of_nat
    _ ≤ ⨅ n : ℕ, ∫⁻ a, ⨆ i ≥ n, f i a ∂μ := iInf_mono fun _ => iSup₂_lintegral_le _
    _ = ∫⁻ a, ⨅ n : ℕ, ⨆ i ≥ n, f i a ∂μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : Nat → α → ENNReal
        g : α → ENNReal
        hf_meas : ∀ (n : Nat), Measurable (f n)
        h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
        h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
        ⊢ Eq (iInf fun n => MeasureTheory.lintegral μ fun a => iSup fun i => iSup fun  …
      -/
      refine (lintegral_iInf ?_ ?_ ?_).symm
        /-
          case refine_1
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : Nat → α → ENNReal
          g : α → ENNReal
          hf_meas : ∀ (n : Nat), Measurable (f n)
          h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
          h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
          ⊢ ∀ (n : Nat), Measurable fun a => iSup fun i => iSup fun h => f i a
        -/
      · intro n
        /-
          case refine_1
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : Nat → α → ENNReal
          g : α → ENNReal
          hf_meas : ∀ (n : Nat), Measurable (f n)
          h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
          h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
          n : Nat
          ⊢ Measurable fun a => iSup fun i => iSup fun h => f i a
        -/
        exact .biSup _ (to_countable _) (fun i _ ↦ hf_meas i)
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : Nat → α → ENNReal
          g : α → ENNReal
          hf_meas : ∀ (n : Nat), Measurable (f n)
          h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
          h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
          ⊢ Antitone fun n a => iSup fun i => iSup fun h => f i a
        -/
      · intro n m hnm a
        /-
          case refine_2
          α : Type u_1
          m✝ : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : Nat → α → ENNReal
          g : α → ENNReal
          hf_meas : ∀ (n : Nat), Measurable (f n)
          h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
          h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
          n m : Nat
          hnm : LE.le n m
          a : α
          ⊢ LE.le ((fun n a => iSup fun i => iSup fun h => f i a) m a) ((fun n a => iSup …
        -/
        exact iSup_le_iSup_of_subset fun i hi => le_trans hnm hi
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : Nat → α → ENNReal
          g : α → ENNReal
          hf_meas : ∀ (n : Nat), Measurable (f n)
          h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
          h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
          ⊢ Ne (MeasureTheory.lintegral μ fun a => iSup fun i => iSup fun h => f i a) To …
        -/
      · refine ne_top_of_le_ne_top h_fin (lintegral_mono_ae ?_)
        /-
          case refine_3
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : Nat → α → ENNReal
          g : α → ENNReal
          hf_meas : ∀ (n : Nat), Measurable (f n)
          h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
          h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
          ⊢ Filter.Eventually (fun a => LE.le (iSup fun i => iSup fun h => f i a) (g a)) …
        -/
        refine (ae_all_iff.2 h_bound).mono fun n hn => ?_
        /-
          case refine_3
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          f : Nat → α → ENNReal
          g : α → ENNReal
          hf_meas : ∀ (n : Nat), Measurable (f n)
          h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
          h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
          n : α
          hn : ∀ (i : Nat), LE.le (f i n) (g n)
          ⊢ LE.le (iSup fun i => iSup fun h => f i n) (g n)
        -/
        exact iSup_le fun i => iSup_le fun _ => hn i
        /-
          🎉 no goals
        -/
                                                     /-
                                                       α : Type u_1
                                                       m : MeasurableSpace α
                                                       μ : MeasureTheory.Measure α
                                                       f : Nat → α → ENNReal
                                                       g : α → ENNReal
                                                       hf_meas : ∀ (n : Nat), Measurable (f n)
                                                       h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (f n) g
                                                       h_fin : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
                                                       ⊢ Eq (MeasureTheory.lintegral μ fun a => iInf fun n => iSup fun i => iSup fun  …
                                                     -/
    _ = ∫⁻ a, limsup (fun n => f n a) atTop ∂μ := by simp only [limsup_eq_iInf_iSup_of_nat]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- Dominated convergence theorem for nonnegative functions -/
theorem tendsto_lintegral_of_dominated_convergence {F : ℕ → α → ℝ≥0∞} {f : α → ℝ≥0∞}
    (bound : α → ℝ≥0∞) (hF_meas : ∀ n, Measurable (F n)) (h_bound : ∀ n, F n ≤ᵐ[μ] bound)
    (h_fin : ∫⁻ a, bound a ∂μ ≠ ∞) (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) atTop (𝓝 (f a))) :
    Tendsto (fun n => ∫⁻ a, F n a ∂μ) atTop (𝓝 (∫⁻ a, f a ∂μ)) :=
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    F : Nat → α → ENNReal
    f bound : α → ENNReal
    hF_meas : ∀ (n : Nat), Measurable (F n)
    h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => Measu …
  -/
  /-
    🎉 no goals
  -/
  tendsto_of_le_liminf_of_limsup_le
  /-
    🎉 no goals
  -/
    (calc
      ∫⁻ a, f a ∂μ = ∫⁻ a, liminf (fun n : ℕ => F n a) atTop ∂μ :=
        lintegral_congr_ae <| h_lim.mono fun _ h => h.liminf_eq.symm
      _ ≤ liminf (fun n => ∫⁻ a, F n a ∂μ) atTop := lintegral_liminf_le hF_meas
      )
    (calc
      limsup (fun n : ℕ => ∫⁻ a, F n a ∂μ) atTop ≤ ∫⁻ a, limsup (fun n => F n a) atTop ∂μ :=
        limsup_lintegral_le _ hF_meas h_bound h_fin
      _ = ∫⁻ a, f a ∂μ := lintegral_congr_ae <| h_lim.mono fun _ h => h.limsup_eq
      )


/-- Dominated convergence theorem for nonnegative functions which are just almost everywhere
measurable. -/
theorem tendsto_lintegral_of_dominated_convergence' {F : ℕ → α → ℝ≥0∞} {f : α → ℝ≥0∞}
    (bound : α → ℝ≥0∞) (hF_meas : ∀ n, AEMeasurable (F n) μ) (h_bound : ∀ n, F n ≤ᵐ[μ] bound)
    (h_fin : ∫⁻ a, bound a ∂μ ≠ ∞) (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) atTop (𝓝 (f a))) :
    Tendsto (fun n => ∫⁻ a, F n a ∂μ) atTop (𝓝 (∫⁻ a, f a ∂μ)) := by
  have : ∀ n, ∫⁻ a, F n a ∂μ = ∫⁻ a, (hF_meas n).mk (F n) a ∂μ := fun n =>
    lintegral_congr_ae (hF_meas n).ae_eq_mk
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    F : Nat → α → ENNReal
    f bound : α → ENNReal
    hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
    h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    this : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureTheo …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun a => F n a) Filter.at …
  -/
  simp_rw [this]
  apply
    tendsto_lintegral_of_dominated_convergence bound (fun n => (hF_meas n).measurable_mk) _ h_fin
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → ENNReal
      f bound : α → ENNReal
      hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
      h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      this : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureTheo …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => AEMeasurable.mk (F n) ⋯ …
    -/
  · have : ∀ n, ∀ᵐ a ∂μ, (hF_meas n).mk (F n) a = F n a := fun n => (hF_meas n).ae_eq_mk.symm
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → ENNReal
      f bound : α → ENNReal
      hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
      h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      this✝ : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureThe …
      this : ∀ (n : Nat), Filter.Eventually (fun a => Eq (AEMeasurable.mk (F n) ⋯ a) …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => AEMeasurable.mk (F n) ⋯ …
    -/
    have : ∀ᵐ a ∂μ, ∀ n, (hF_meas n).mk (F n) a = F n a := ae_all_iff.mpr this
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → ENNReal
      f bound : α → ENNReal
      hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
      h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      this✝¹ : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureTh …
      this✝ : ∀ (n : Nat), Filter.Eventually (fun a => Eq (AEMeasurable.mk (F n) ⋯ a …
      this : Filter.Eventually (fun a => ∀ (n : Nat), Eq (AEMeasurable.mk (F n) ⋯ a) …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => AEMeasurable.mk (F n) ⋯ …
    -/
    filter_upwards [this, h_lim] with a H H'
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → ENNReal
      f bound : α → ENNReal
      hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
      h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      this✝¹ : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureTh …
      this✝ : ∀ (n : Nat), Filter.Eventually (fun a => Eq (AEMeasurable.mk (F n) ⋯ a …
      this : Filter.Eventually (fun a => ∀ (n : Nat), Eq (AEMeasurable.mk (F n) ⋯ a) …
      a : α
      H : ∀ (n : Nat), Eq (AEMeasurable.mk (F n) ⋯ a) (F n a)
      H' : Filter.Tendsto (fun n => F n a) Filter.atTop (nhds (f a))
      ⊢ Filter.Tendsto (fun n => AEMeasurable.mk (F n) ⋯ a) Filter.atTop (nhds (f a))
    -/
    simp_rw [H]
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → ENNReal
      f bound : α → ENNReal
      hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
      h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      this✝¹ : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureTh …
      this✝ : ∀ (n : Nat), Filter.Eventually (fun a => Eq (AEMeasurable.mk (F n) ⋯ a …
      this : Filter.Eventually (fun a => ∀ (n : Nat), Eq (AEMeasurable.mk (F n) ⋯ a) …
      a : α
      H : ∀ (n : Nat), Eq (AEMeasurable.mk (F n) ⋯ a) (F n a)
      H' : Filter.Tendsto (fun n => F n a) Filter.atTop (nhds (f a))
      ⊢ Filter.Tendsto (fun n => F n a) Filter.atTop (nhds (f a))
    -/
    exact H'
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → ENNReal
      f bound : α → ENNReal
      hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
      h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      this : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureTheo …
      ⊢ ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (AEMeasurable.mk (F n) ⋯) bound
    -/
  · intro n
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → ENNReal
      f bound : α → ENNReal
      hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
      h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      this : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureTheo …
      n : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyLE (AEMeasurable.mk (F n) ⋯) bound
    -/
    filter_upwards [h_bound n, (hF_meas n).ae_eq_mk] with a H H'
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Nat → α → ENNReal
      f bound : α → ENNReal
      hF_meas : ∀ (n : Nat), AEMeasurable (F n) μ
      h_bound : ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F n) bound
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      this : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => F n a) (MeasureTheo …
      n : Nat
      a : α
      H : LE.le (F n a) (bound a)
      H' : Eq (F n a) (AEMeasurable.mk (F n) ⋯ a)
      ⊢ LE.le (AEMeasurable.mk (F n) ⋯ a) (bound a)
    -/
    rwa [H'] at H
    /-
      🎉 no goals
    -/


/-- Dominated convergence theorem for filters with a countable basis -/
theorem tendsto_lintegral_filter_of_dominated_convergence {ι} {l : Filter ι}
    [l.IsCountablyGenerated] {F : ι → α → ℝ≥0∞} {f : α → ℝ≥0∞} (bound : α → ℝ≥0∞)
    (hF_meas : ∀ᶠ n in l, Measurable (F n)) (h_bound : ∀ᶠ n in l, ∀ᵐ a ∂μ, F n a ≤ bound a)
    (h_fin : ∫⁻ a, bound a ∂μ ≠ ∞) (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) l (𝓝 (f a))) :
    Tendsto (fun n => ∫⁻ a, F n a ∂μ) l (𝓝 <| ∫⁻ a, f a ∂μ) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    F : ι → α → ENNReal
    f bound : α → ENNReal
    hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun a => F n a) l (nhds ( …
  -/
  rw [tendsto_iff_seq_tendsto]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    F : ι → α → ENNReal
    f bound : α → ENNReal
    hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
    ⊢ ∀ (x : Nat → ι), Filter.Tendsto x Filter.atTop l → Filter.Tendsto (Function. …
  -/
  intro x xl
  have hxl := by
    rw [tendsto_atTop'] at xl
    exact xl
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    F : ι → α → ENNReal
    f bound : α → ENNReal
    hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
    ⊢ Filter.Tendsto (Function.comp (fun n => MeasureTheory.lintegral μ fun a => F …
  -/
  have h := inter_mem hF_meas h_bound
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    F : ι → α → ENNReal
    f bound : α → ENNReal
    hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
    h : Membership.mem l (Inter.inter (setOf fun x => (fun n => Measurable (F n))  …
    ⊢ Filter.Tendsto (Function.comp (fun n => MeasureTheory.lintegral μ fun a => F …
  -/
  replace h := hxl _ h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    F : ι → α → ENNReal
    f bound : α → ENNReal
    hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
    h : Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem (Inter.inter (setO …
    ⊢ Filter.Tendsto (Function.comp (fun n => MeasureTheory.lintegral μ fun a => F …
  -/
  rcases h with ⟨k, h⟩
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    F : ι → α → ENNReal
    f bound : α → ENNReal
    hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
    k : Nat
    h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
    ⊢ Filter.Tendsto (Function.comp (fun n => MeasureTheory.lintegral μ fun a => F …
  -/
  rw [← tendsto_add_atTop_iff_nat k]
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    ι : Type u_5
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    F : ι → α → ENNReal
    f bound : α → ENNReal
    hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
    h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
    h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
    x : Nat → ι
    xl : Filter.Tendsto x Filter.atTop l
    hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
    k : Nat
    h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
    ⊢ Filter.Tendsto (fun n => Function.comp (fun n => MeasureTheory.lintegral μ f …
  -/
  refine tendsto_lintegral_of_dominated_convergence ?_ ?_ ?_ ?_ ?_
    /-
      case intro.refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ α → ENNReal
    -/
  · exact bound
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ ∀ (n : Nat), Measurable (F (x (HAdd.hAdd n k)))
    -/
  · intro
    /-
      case intro.refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      n✝ : Nat
      ⊢ Measurable (F (x (HAdd.hAdd n✝ k)))
    -/
    refine (h _ ?_).1
    /-
      case intro.refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      n✝ : Nat
      ⊢ GE.ge (HAdd.hAdd n✝ k) k
    -/
    exact Nat.le_add_left _ _
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ ∀ (n : Nat), (MeasureTheory.ae μ).EventuallyLE (F (x (HAdd.hAdd n k))) bound
    -/
  · intro
    /-
      case intro.refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      n✝ : Nat
      ⊢ (MeasureTheory.ae μ).EventuallyLE (F (x (HAdd.hAdd n✝ k))) bound
    -/
    refine (h _ ?_).2
    /-
      case intro.refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      n✝ : Nat
      ⊢ GE.ge (HAdd.hAdd n✝ k) k
    -/
    exact Nat.le_add_left _ _
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_4
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
    -/
  · assumption
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_5
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds (f …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => F (x (HAdd.hAdd n k)) a …
    -/
  · refine h_lim.mono fun a h_lim => ?_
    /-
      case intro.refine_5
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim✝ : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      a : α
      h_lim : Filter.Tendsto (fun n => F n a) l (nhds (f a))
      ⊢ Filter.Tendsto (fun n => F (x (HAdd.hAdd n k)) a) Filter.atTop (nhds (f a))
    -/
    apply @Tendsto.comp _ _ _ (fun n => x (n + k)) fun n => F n a
      /-
        case intro.refine_5.hg
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        ι : Type u_5
        l : Filter ι
        inst✝ : l.IsCountablyGenerated
        F : ι → α → ENNReal
        f bound : α → ENNReal
        hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
        h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
        h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
        h_lim✝ : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds ( …
        x : Nat → ι
        xl : Filter.Tendsto x Filter.atTop l
        hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
        k : Nat
        h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
        a : α
        h_lim : Filter.Tendsto (fun n => F n a) l (nhds (f a))
        ⊢ Filter.Tendsto (fun n => F n a) ?intro.refine_5.y (nhds (f a))
      -/
    · assumption
      /-
        🎉 no goals
      -/
    /-
      case intro.refine_5.hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim✝ : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      a : α
      h_lim : Filter.Tendsto (fun n => F n a) l (nhds (f a))
      ⊢ Filter.Tendsto (fun n => x (HAdd.hAdd n k)) Filter.atTop l
    -/
    rw [tendsto_add_atTop_iff_nat]
    /-
      case intro.refine_5.hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      ι : Type u_5
      l : Filter ι
      inst✝ : l.IsCountablyGenerated
      F : ι → α → ENNReal
      f bound : α → ENNReal
      hF_meas : Filter.Eventually (fun n => Measurable (F n)) l
      h_bound : Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (F n a …
      h_fin : Ne (MeasureTheory.lintegral μ fun a => bound a) Top.top
      h_lim✝ : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) l (nhds ( …
      x : Nat → ι
      xl : Filter.Tendsto x Filter.atTop l
      hxl : ∀ (s : Set ι), Membership.mem l s → Exists fun a => ∀ (b : Nat), GE.ge b …
      k : Nat
      h : ∀ (b : Nat), GE.ge b k → Membership.mem (Inter.inter (setOf fun x => (fun  …
      a : α
      h_lim : Filter.Tendsto (fun n => F n a) l (nhds (f a))
      ⊢ Filter.Tendsto x Filter.atTop l
    -/
    assumption
    /-
      🎉 no goals
    -/


theorem lintegral_tendsto_of_tendsto_of_antitone {f : ℕ → α → ℝ≥0∞} {F : α → ℝ≥0∞}
    (hf : ∀ n, AEMeasurable (f n) μ) (h_anti : ∀ᵐ x ∂μ, Antitone fun n ↦ f n x)
    (h0 : ∫⁻ a, f 0 a ∂μ ≠ ∞)
    (h_tendsto : ∀ᵐ x ∂μ, Tendsto (fun n ↦ f n x) atTop (𝓝 (F x))) :
    Tendsto (fun n ↦ ∫⁻ x, f n x ∂μ) atTop (𝓝 (∫⁻ x, F x ∂μ)) := by
  have : Antitone fun n ↦ ∫⁻ x, f n x ∂μ := fun i j hij ↦
    lintegral_mono_ae (h_anti.mono fun x hx ↦ hx hij)
  suffices key : ∫⁻ x, F x ∂μ = ⨅ n, ∫⁻ x, f n x ∂μ by
    rw [key]
    exact tendsto_atTop_iInf this
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    F : α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_anti : Filter.Eventually (fun x => Antitone fun n => f n x) (MeasureTheory.a …
    h0 : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    this : Antitone fun n => MeasureTheory.lintegral μ fun x => f n x
    ⊢ Eq (MeasureTheory.lintegral μ fun x => F x) (iInf fun n => MeasureTheory.lin …
  -/
  rw [← lintegral_iInf' hf h_anti h0]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : Nat → α → ENNReal
    F : α → ENNReal
    hf : ∀ (n : Nat), AEMeasurable (f n) μ
    h_anti : Filter.Eventually (fun x => Antitone fun n => f n x) (MeasureTheory.a …
    h0 : Ne (MeasureTheory.lintegral μ fun a => f 0 a) Top.top
    h_tendsto : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter …
    this : Antitone fun n => MeasureTheory.lintegral μ fun x => f n x
    ⊢ Eq (MeasureTheory.lintegral μ fun x => F x) (MeasureTheory.lintegral μ fun a …
  -/
  refine lintegral_congr_ae ?_
  filter_upwards [h_anti, h_tendsto] with _ hx_anti hx_tendsto
    using tendsto_nhds_unique hx_tendsto (tendsto_atTop_iInf hx_anti)


/-- Monotone convergence for a supremum over a directed family and indexed by a countable type -/
theorem lintegral_iSup_directed_of_measurable [Countable β] {f : β → α → ℝ≥0∞}
    (hf : ∀ b, Measurable (f b)) (h_directed : Directed (· ≤ ·) f) :
    ∫⁻ a, ⨆ b, f b a ∂μ = ⨆ b, ∫⁻ a, f b a ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    f : β → α → ENNReal
    hf : ∀ (b : β), Measurable (f b)
    h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun b => f b a) (iSup fun b => M …
  -/
  cases nonempty_encodable β
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    f : β → α → ENNReal
    hf : ∀ (b : β), Measurable (f b)
    h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
    val✝ : Encodable β
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun b => f b a) (iSup fun b => M …
  -/
  cases isEmpty_or_nonempty β
    /-
      case intro.inl
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable β
      f : β → α → ENNReal
      hf : ∀ (b : β), Measurable (f b)
      h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
      val✝ : Encodable β
      h✝ : IsEmpty β
      ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun b => f b a) (iSup fun b => M …
    -/
  · simp [iSup_of_empty]
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    f : β → α → ENNReal
    hf : ∀ (b : β), Measurable (f b)
    h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
    val✝ : Encodable β
    h✝ : Nonempty β
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun b => f b a) (iSup fun b => M …
  -/
  inhabit β
  have : ∀ a, ⨆ b, f b a = ⨆ n, f (h_directed.sequence f n) a := by
    intro a
    refine le_antisymm (iSup_le fun b => ?_) (iSup_le fun n => le_iSup (fun n => f n a) _)
    exact le_iSup_of_le (encode b + 1) (h_directed.le_sequence b a)
  calc
    ∫⁻ a, ⨆ b, f b a ∂μ = ∫⁻ a, ⨆ n, f (h_directed.sequence f n) a ∂μ := by simp only [this]
    _ = ⨆ n, ∫⁻ a, f (h_directed.sequence f n) a ∂μ :=
      (lintegral_iSup (fun n => hf _) h_directed.sequence_mono)
    _ = ⨆ b, ∫⁻ a, f b a ∂μ := by
      refine le_antisymm (iSup_le fun n => ?_) (iSup_le fun b => ?_)
      · exact le_iSup (fun b => ∫⁻ a, f b a ∂μ) _
      · exact le_iSup_of_le (encode b + 1) (lintegral_mono <| h_directed.le_sequence b)


/-- Monotone convergence for a supremum over a directed family and indexed by a countable type. -/
theorem lintegral_iSup_directed [Countable β] {f : β → α → ℝ≥0∞} (hf : ∀ b, AEMeasurable (f b) μ)
    (h_directed : Directed (· ≤ ·) f) : ∫⁻ a, ⨆ b, f b a ∂μ = ⨆ b, ∫⁻ a, f b a ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    f : β → α → ENNReal
    hf : ∀ (b : β), AEMeasurable (f b) μ
    h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup fun b => f b a) (iSup fun b => M …
  -/
  simp_rw [← iSup_apply]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    f : β → α → ENNReal
    hf : ∀ (b : β), AEMeasurable (f b) μ
    h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup (fun i => f i) a) (iSup fun b => …
  -/
  let p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
  have hp : ∀ᵐ x ∂μ, p x fun i => f i x := by
    filter_upwards [] with x i j
    obtain ⟨z, hz₁, hz₂⟩ := h_directed i j
    exact ⟨z, hz₁ x, hz₂ x⟩
  have h_ae_seq_directed : Directed LE.le (aeSeq hf p) := by
    intro b₁ b₂
    obtain ⟨z, hz₁, hz₂⟩ := h_directed b₁ b₂
    refine ⟨z, ?_, ?_⟩ <;>
      · intro x
        by_cases hx : x ∈ aeSeqSet hf p
        · repeat rw [aeSeq.aeSeq_eq_fun_of_mem_aeSeqSet hf hx]
          apply_rules [hz₁, hz₂]
        · simp only [aeSeq, hx, if_false]
          exact le_rfl
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    f : β → α → ENNReal
    hf : ∀ (b : β), AEMeasurable (f b) μ
    h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
    p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
    hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
    h_ae_seq_directed : Directed LE.le (aeSeq hf p)
    ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup (fun i => f i) a) (iSup fun b => …
  -/
  convert lintegral_iSup_directed_of_measurable (aeSeq.measurable hf p) h_ae_seq_directed using 1
    /-
      case h.e'_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable β
      f : β → α → ENNReal
      hf : ∀ (b : β), AEMeasurable (f b) μ
      h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
      p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_directed : Directed LE.le (aeSeq hf p)
      ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup (fun i => f i) a) (MeasureTheory …
    -/
  · simp_rw [← iSup_apply]
    /-
      case h.e'_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable β
      f : β → α → ENNReal
      hf : ∀ (b : β), AEMeasurable (f b) μ
      h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
      p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_directed : Directed LE.le (aeSeq hf p)
      ⊢ Eq (MeasureTheory.lintegral μ fun a => iSup (fun i => f i) a) (MeasureTheory …
    -/
    rw [lintegral_congr_ae (aeSeq.iSup hf hp).symm]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable β
      f : β → α → ENNReal
      hf : ∀ (b : β), AEMeasurable (f b) μ
      h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
      p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_directed : Directed LE.le (aeSeq hf p)
      ⊢ Eq (iSup fun b => MeasureTheory.lintegral μ fun a => f b a) (iSup fun b => M …
    -/
  · congr 1
    /-
      case h.e'_3.e_s
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable β
      f : β → α → ENNReal
      hf : ∀ (b : β), AEMeasurable (f b) μ
      h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
      p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_directed : Directed LE.le (aeSeq hf p)
      ⊢ Eq (fun b => MeasureTheory.lintegral μ fun a => f b a) fun b => MeasureTheor …
    -/
    ext1 b
    /-
      case h.e'_3.e_s.h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable β
      f : β → α → ENNReal
      hf : ∀ (b : β), AEMeasurable (f b) μ
      h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
      p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_directed : Directed LE.le (aeSeq hf p)
      b : β
      ⊢ Eq (MeasureTheory.lintegral μ fun a => f b a) (MeasureTheory.lintegral μ fun …
    -/
    rw [lintegral_congr_ae]
    /-
      case h.e'_3.e_s.h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable β
      f : β → α → ENNReal
      hf : ∀ (b : β), AEMeasurable (f b) μ
      h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
      p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_directed : Directed LE.le (aeSeq hf p)
      b : β
      ⊢ (MeasureTheory.ae μ).EventuallyEq (f b) (aeSeq hf p b)
    -/
    apply EventuallyEq.symm
    /-
      case h.e'_3.e_s.h.H
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : Countable β
      f : β → α → ENNReal
      hf : ∀ (b : β), AEMeasurable (f b) μ
      h_directed : Directed (fun x1 x2 => LE.le x1 x2) f
      p : α → (β → ENNReal) → Prop := fun x f' => Directed LE.le f'
      hp : Filter.Eventually (fun x => p x fun i => f i x) (MeasureTheory.ae μ)
      h_ae_seq_directed : Directed LE.le (aeSeq hf p)
      b : β
      ⊢ (MeasureTheory.ae μ).EventuallyEq (aeSeq hf p b) (f b)
    -/
    exact aeSeq.aeSeq_n_eq_fun_n_ae hf hp _
    /-
      🎉 no goals
    -/


theorem lintegral_tsum [Countable β] {f : β → α → ℝ≥0∞} (hf : ∀ i, AEMeasurable (f i) μ) :
    ∫⁻ a, ∑' i, f i a ∂μ = ∑' i, ∫⁻ a, f i a ∂μ := by
  classical
  simp only [ENNReal.tsum_eq_iSup_sum]
  rw [lintegral_iSup_directed]
  · simp [lintegral_finset_sum' _ fun i _ => hf i]
  · intro b
    exact Finset.aemeasurable_sum _ fun i _ => hf i
  · intro s t
    use s ∪ t
    constructor
    · exact fun a => Finset.sum_le_sum_of_subset Finset.subset_union_left
    · exact fun a => Finset.sum_le_sum_of_subset Finset.subset_union_right


theorem lintegral_iUnion₀ [Countable β] {s : β → Set α} (hm : ∀ i, NullMeasurableSet (s i) μ)
    (hd : Pairwise (AEDisjoint μ on s)) (f : α → ℝ≥0∞) :
    ∫⁻ a in ⋃ i, s i, f a ∂μ = ∑' i, ∫⁻ a in s i, f a ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    s : β → Set α
    hm : ∀ (i : β), MeasureTheory.NullMeasurableSet (s i) μ
    hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Set.iUnion fun i => s i)) fun a =>  …
  -/
  simp only [Measure.restrict_iUnion_ae hd hm, lintegral_sum_measure]
  /-
    🎉 no goals
  -/


theorem lintegral_iUnion [Countable β] {s : β → Set α} (hm : ∀ i, MeasurableSet (s i))
    (hd : Pairwise (Disjoint on s)) (f : α → ℝ≥0∞) :
    ∫⁻ a in ⋃ i, s i, f a ∂μ = ∑' i, ∫⁻ a in s i, f a ∂μ :=
  lintegral_iUnion₀ (fun i => (hm i).nullMeasurableSet) hd.aedisjoint f


theorem lintegral_biUnion₀ {t : Set β} {s : β → Set α} (ht : t.Countable)
    (hm : ∀ i ∈ t, NullMeasurableSet (s i) μ) (hd : t.Pairwise (AEDisjoint μ on s)) (f : α → ℝ≥0∞) :
    ∫⁻ a in ⋃ i ∈ t, s i, f a ∂μ = ∑' i : t, ∫⁻ a in s i, f a ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Set β
    s : β → Set α
    ht : t.Countable
    hm : ∀ (i : β), Membership.mem t i → MeasureTheory.NullMeasurableSet (s i) μ
    hd : t.Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Set.iUnion fun i => Set.iUnion fun  …
  -/
  haveI := ht.toEncodable
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    t : Set β
    s : β → Set α
    ht : t.Countable
    hm : ∀ (i : β), Membership.mem t i → MeasureTheory.NullMeasurableSet (s i) μ
    hd : t.Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
    f : α → ENNReal
    this : Encodable ↑t
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Set.iUnion fun i => Set.iUnion fun  …
  -/
  rw [biUnion_eq_iUnion, lintegral_iUnion₀ (SetCoe.forall'.1 hm) (hd.subtype _ _)]
  /-
    🎉 no goals
  -/


theorem lintegral_biUnion {t : Set β} {s : β → Set α} (ht : t.Countable)
    (hm : ∀ i ∈ t, MeasurableSet (s i)) (hd : t.PairwiseDisjoint s) (f : α → ℝ≥0∞) :
    ∫⁻ a in ⋃ i ∈ t, s i, f a ∂μ = ∑' i : t, ∫⁻ a in s i, f a ∂μ :=
  lintegral_biUnion₀ ht (fun i hi => (hm i hi).nullMeasurableSet) hd.aedisjoint f


theorem lintegral_biUnion_finset₀ {s : Finset β} {t : β → Set α}
    (hd : Set.Pairwise (↑s) (AEDisjoint μ on t)) (hm : ∀ b ∈ s, NullMeasurableSet (t b) μ)
    (f : α → ℝ≥0∞) : ∫⁻ a in ⋃ b ∈ s, t b, f a ∂μ = ∑ b ∈ s, ∫⁻ a in t b, f a ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Finset β
    t : β → Set α
    hd : (↑s).Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) t)
    hm : ∀ (b : β), Membership.mem s b → MeasureTheory.NullMeasurableSet (t b) μ
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Set.iUnion fun b => Set.iUnion fun  …
  -/
  simp only [← Finset.mem_coe, lintegral_biUnion₀ s.countable_toSet hm hd, ← Finset.tsum_subtype']
  /-
    🎉 no goals
  -/


theorem lintegral_biUnion_finset {s : Finset β} {t : β → Set α} (hd : Set.PairwiseDisjoint (↑s) t)
    (hm : ∀ b ∈ s, MeasurableSet (t b)) (f : α → ℝ≥0∞) :
    ∫⁻ a in ⋃ b ∈ s, t b, f a ∂μ = ∑ b ∈ s, ∫⁻ a in t b, f a ∂μ :=
  lintegral_biUnion_finset₀ hd.aedisjoint (fun b hb => (hm b hb).nullMeasurableSet) f


theorem lintegral_iUnion_le [Countable β] (s : β → Set α) (f : α → ℝ≥0∞) :
    ∫⁻ a in ⋃ i, s i, f a ∂μ ≤ ∑' i, ∫⁻ a in s i, f a ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    s : β → Set α
    f : α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict (Set.iUnion fun i => s i)) fun a  …
  -/
  rw [← lintegral_sum_measure]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : Countable β
    s : β → Set α
    f : α → ENNReal
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict (Set.iUnion fun i => s i)) fun a  …
  -/
  exact lintegral_mono' restrict_iUnion_le le_rfl
  /-
    🎉 no goals
  -/


theorem lintegral_union {f : α → ℝ≥0∞} {A B : Set α} (hB : MeasurableSet B) (hAB : Disjoint A B) :
    ∫⁻ a in A ∪ B, f a ∂μ = ∫⁻ a in A, f a ∂μ + ∫⁻ a in B, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    A B : Set α
    hB : MeasurableSet B
    hAB : Disjoint A B
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Union.union A B)) fun a => f a) (HA …
  -/
  rw [restrict_union hAB hB, lintegral_add_measure]
  /-
    🎉 no goals
  -/


theorem lintegral_union_le (f : α → ℝ≥0∞) (s t : Set α) :
    ∫⁻ a in s ∪ t, f a ∂μ ≤ ∫⁻ a in s, f a ∂μ + ∫⁻ a in t, f a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s t : Set α
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict (Union.union s t)) fun a => f a)  …
  -/
  rw [← lintegral_add_measure]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s t : Set α
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict (Union.union s t)) fun a => f a)  …
  -/
  exact lintegral_mono' (restrict_union_le _ _) le_rfl
  /-
    🎉 no goals
  -/


theorem lintegral_inter_add_diff {B : Set α} (f : α → ℝ≥0∞) (A : Set α) (hB : MeasurableSet B) :
    ∫⁻ x in A ∩ B, f x ∂μ + ∫⁻ x in A \ B, f x ∂μ = ∫⁻ x in A, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    B : Set α
    f : α → ENNReal
    A : Set α
    hB : MeasurableSet B
    ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral (μ.restrict (Inter.inter A B)) fun x  …
  -/
  rw [← lintegral_add_measure, restrict_inter_add_diff _ hB]
  /-
    🎉 no goals
  -/


theorem lintegral_add_compl (f : α → ℝ≥0∞) {A : Set α} (hA : MeasurableSet A) :
    ∫⁻ x in A, f x ∂μ + ∫⁻ x in Aᶜ, f x ∂μ = ∫⁻ x, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    A : Set α
    hA : MeasurableSet A
    ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral (μ.restrict A) fun x => f x) (Measure …
  -/
  rw [← lintegral_add_measure, Measure.restrict_add_restrict_compl hA]
  /-
    🎉 no goals
  -/


theorem setLintegral_compl {f : α → ℝ≥0∞} {s : Set α} (hsm : MeasurableSet s)
    (hfs : ∫⁻ x in s, f x ∂μ ≠ ∞) :
    ∫⁻ x in sᶜ, f x ∂μ = ∫⁻ x, f x ∂μ - ∫⁻ x in s, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    s : Set α
    hsm : MeasurableSet s
    hfs : Ne (MeasureTheory.lintegral (μ.restrict s) fun x => f x) Top.top
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (HasCompl.compl s)) fun x => f x) (H …
  -/
  rw [← lintegral_add_compl (μ := μ) f hsm, ENNReal.add_sub_cancel_left hfs]
  /-
    🎉 no goals
  -/


theorem setLIntegral_iUnion_of_directed {ι : Type*} [Countable ι]
    (f : α → ℝ≥0∞) {s : ι → Set α} (hd : Directed (· ⊆ ·) s) :
    ∫⁻ x in ⋃ i, s i, f x ∂μ = ⨆ i, ∫⁻ x in s i, f x ∂μ := by
  simp only [lintegral_def, iSup_comm (ι := ι),
    SimpleFunc.lintegral_restrict_iUnion_of_directed _ hd]


theorem lintegral_max {f g : α → ℝ≥0∞} (hf : Measurable f) (hg : Measurable g) :
    ∫⁻ x, max (f x) (g x) ∂μ =
      ∫⁻ x in { x | f x ≤ g x }, g x ∂μ + ∫⁻ x in { x | g x < f x }, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral μ fun x => Max.max (f x) (g x)) (HAdd.hAdd (Meas …
  -/
  have hm : MeasurableSet { x | f x ≤ g x } := measurableSet_le hf hg
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Measurable f
    hg : Measurable g
    hm : MeasurableSet (setOf fun x => LE.le (f x) (g x))
    ⊢ Eq (MeasureTheory.lintegral μ fun x => Max.max (f x) (g x)) (HAdd.hAdd (Meas …
  -/
  rw [← lintegral_add_compl (fun x => max (f x) (g x)) hm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Measurable f
    hg : Measurable g
    hm : MeasurableSet (setOf fun x => LE.le (f x) (g x))
    ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral (μ.restrict (setOf fun x => LE.le (f  …
  -/
  simp only [← compl_setOf, ← not_le]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Measurable f
    hg : Measurable g
    hm : MeasurableSet (setOf fun x => LE.le (f x) (g x))
    ⊢ Eq (HAdd.hAdd (MeasureTheory.lintegral (μ.restrict (setOf fun x => LE.le (f  …
  -/
  refine congr_arg₂ (· + ·) (setLIntegral_congr_fun hm ?_) (setLIntegral_congr_fun hm.compl ?_)
  exacts [ae_of_all _ fun x => max_eq_right (a := f x) (b := g x),
    ae_of_all _ fun x (hx : ¬ f x ≤ g x) => max_eq_left (not_le.1 hx).le]


theorem setLIntegral_max {f g : α → ℝ≥0∞} (hf : Measurable f) (hg : Measurable g) (s : Set α) :
    ∫⁻ x in s, max (f x) (g x) ∂μ =
      ∫⁻ x in s ∩ { x | f x ≤ g x }, g x ∂μ + ∫⁻ x in s ∩ { x | g x < f x }, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Measurable f
    hg : Measurable g
    s : Set α
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => Max.max (f x) (g x)) (HA …
  -/
  rw [lintegral_max hf hg, restrict_restrict, restrict_restrict, inter_comm s, inter_comm s]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → ENNReal
    hf : Measurable f
    hg : Measurable g
    s : Set α
    ⊢ MeasurableSet (setOf fun x => LT.lt (g x) (f x))
  -/
  exacts [measurableSet_lt hg hf, measurableSet_le hf hg]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_max := setLIntegral_max


theorem lintegral_map {mβ : MeasurableSpace β} {f : β → ℝ≥0∞} {g : α → β} (hf : Measurable f)
    (hg : Measurable g) : ∫⁻ a, f a ∂map g μ = ∫⁻ a, f (g a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map g μ) fun a => f a) (M …
  -/
  erw [lintegral_eq_iSup_eapprox_lintegral hf, lintegral_eq_iSup_eapprox_lintegral (hf.comp hg)]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq (iSup fun n => (MeasureTheory.SimpleFunc.eapprox f n).lintegral (MeasureT …
  -/
  congr with n : 1
  /-
    case e_s.h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    hf : Measurable f
    hg : Measurable g
    n : Nat
    ⊢ Eq ((MeasureTheory.SimpleFunc.eapprox f n).lintegral (MeasureTheory.Measure. …
  -/
  convert SimpleFunc.lintegral_map _ hg
  /-
    case h.e'_3.h.e'_3
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    hf : Measurable f
    hg : Measurable g
    n : Nat
    ⊢ Eq (MeasureTheory.SimpleFunc.eapprox (Function.comp f g) n) ((MeasureTheory. …
  -/
  ext1 x; simp only [eapprox_comp hf hg, coe_comp]
          /-
            🎉 no goals
          -/


theorem lintegral_map' {mβ : MeasurableSpace β} {f : β → ℝ≥0∞} {g : α → β}
    (hf : AEMeasurable f (Measure.map g μ)) (hg : AEMeasurable g μ) :
    ∫⁻ a, f a ∂Measure.map g μ = ∫⁻ a, f (g a) ∂μ :=
  calc
    ∫⁻ a, f a ∂Measure.map g μ = ∫⁻ a, hf.mk f a ∂Measure.map g μ :=
      lintegral_congr_ae hf.ae_eq_mk
    _ = ∫⁻ a, hf.mk f a ∂Measure.map (hg.mk g) μ := by
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        mβ : MeasurableSpace β
        f : β → ENNReal
        g : α → β
        hf : AEMeasurable f (MeasureTheory.Measure.map g μ)
        hg : AEMeasurable g μ
        ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map g μ) fun a => AEMeasu …
      -/
      congr 1
      /-
        case e_μ
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        mβ : MeasurableSpace β
        f : β → ENNReal
        g : α → β
        hf : AEMeasurable f (MeasureTheory.Measure.map g μ)
        hg : AEMeasurable g μ
        ⊢ Eq (MeasureTheory.Measure.map g μ) (MeasureTheory.Measure.map (AEMeasurable. …
      -/
      exact Measure.map_congr hg.ae_eq_mk
      /-
        🎉 no goals
      -/
    _ = ∫⁻ a, hf.mk f (hg.mk g a) ∂μ := lintegral_map hf.measurable_mk hg.measurable_mk
    _ = ∫⁻ a, hf.mk f (g a) ∂μ := lintegral_congr_ae <| hg.ae_eq_mk.symm.fun_comp _
    _ = ∫⁻ a, f (g a) ∂μ := lintegral_congr_ae (ae_eq_comp hg hf.ae_eq_mk.symm)


theorem lintegral_map_le {mβ : MeasurableSpace β} (f : β → ℝ≥0∞) {g : α → β} (hg : Measurable g) :
    ∫⁻ a, f a ∂Measure.map g μ ≤ ∫⁻ a, f (g a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    hg : Measurable g
    ⊢ LE.le (MeasureTheory.lintegral (MeasureTheory.Measure.map g μ) fun a => f a) …
  -/
  rw [← iSup_lintegral_measurable_le_eq_lintegral, ← iSup_lintegral_measurable_le_eq_lintegral]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    hg : Measurable g
    ⊢ LE.le (iSup fun g_1 => iSup fun x => iSup fun x => MeasureTheory.lintegral ( …
  -/
  refine iSup₂_le fun i hi => iSup_le fun h'i => ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    hg : Measurable g
    i : β → ENNReal
    hi : Measurable i
    h'i : LE.le i f
    ⊢ LE.le (MeasureTheory.lintegral (MeasureTheory.Measure.map g μ) fun a => i a) …
  -/
  refine le_iSup₂_of_le (i ∘ g) (hi.comp hg) ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mβ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    hg : Measurable g
    i : β → ENNReal
    hi : Measurable i
    h'i : LE.le i f
    ⊢ LE.le (MeasureTheory.lintegral (MeasureTheory.Measure.map g μ) fun a => i a) …
  -/
  exact le_iSup_of_le (fun x => h'i (g x)) (le_of_eq (lintegral_map hi hg))
  /-
    🎉 no goals
  -/


theorem lintegral_comp [MeasurableSpace β] {f : β → ℝ≥0∞} {g : α → β} (hf : Measurable f)
    (hg : Measurable g) : lintegral μ (f ∘ g) = ∫⁻ a, f a ∂map g μ :=
  (lintegral_map hf hg).symm


theorem setLIntegral_map [MeasurableSpace β] {f : β → ℝ≥0∞} {g : α → β} {s : Set β}
    (hs : MeasurableSet s) (hf : Measurable f) (hg : Measurable g) :
    ∫⁻ y in s, f y ∂map g μ = ∫⁻ x in g ⁻¹' s, f (g x) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    f : β → ENNReal
    g : α → β
    s : Set β
    hs : MeasurableSet s
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral ((MeasureTheory.Measure.map g μ).restrict s) fun …
  -/
  rw [restrict_map hg hs, lintegral_map hf hg]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_map := setLIntegral_map


theorem lintegral_indicator_const_comp {mβ : MeasurableSpace β} {f : α → β} {s : Set β}
    (hf : Measurable f) (hs : MeasurableSet s) (c : ℝ≥0∞) :
    ∫⁻ a, s.indicator (fun _ => c) (f a) ∂μ = c * μ (f ⁻¹' s) := by
  erw [lintegral_comp (measurable_const.indicator hs) hf, lintegral_indicator_const hs,
    Measure.map_apply hf hs]


/-- If `g : α → β` is a measurable embedding and `f : β → ℝ≥0∞` is any function (not necessarily
measurable), then `∫⁻ a, f a ∂(map g μ) = ∫⁻ a, f (g a) ∂μ`. Compare with `lintegral_map` which
applies to any measurable `g : α → β` but requires that `f` is measurable as well. -/
theorem _root_.MeasurableEmbedding.lintegral_map [MeasurableSpace β] {g : α → β}
    (hg : MeasurableEmbedding g) (f : β → ℝ≥0∞) : ∫⁻ a, f a ∂map g μ = ∫⁻ a, f (g a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    g : α → β
    hg : MeasurableEmbedding g
    f : β → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map g μ) fun a => f a) (M …
  -/
  rw [lintegral, lintegral]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    g : α → β
    hg : MeasurableEmbedding g
    f : β → ENNReal
    ⊢ Eq (iSup fun g_1 => iSup fun x => g_1.lintegral (MeasureTheory.Measure.map g …
  -/
  refine le_antisymm (iSup₂_le fun f₀ hf₀ => ?_) (iSup₂_le fun f₀ hf₀ => ?_)
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasurableSpace β
      g : α → β
      hg : MeasurableEmbedding g
      f : β → ENNReal
      f₀ : MeasureTheory.SimpleFunc β ENNReal
      hf₀ : LE.le ⇑f₀ fun a => f a
      ⊢ LE.le (f₀.lintegral (MeasureTheory.Measure.map g μ)) (iSup fun g_1 => iSup f …
    -/
  · rw [SimpleFunc.lintegral_map _ hg.measurable]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasurableSpace β
      g : α → β
      hg : MeasurableEmbedding g
      f : β → ENNReal
      f₀ : MeasureTheory.SimpleFunc β ENNReal
      hf₀ : LE.le ⇑f₀ fun a => f a
      ⊢ LE.le ((f₀.comp g ⋯).lintegral μ) (iSup fun g_1 => iSup fun x => g_1.lintegr …
    -/
    have : (f₀.comp g hg.measurable : α → ℝ≥0∞) ≤ f ∘ g := fun x => hf₀ (g x)
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasurableSpace β
      g : α → β
      hg : MeasurableEmbedding g
      f : β → ENNReal
      f₀ : MeasureTheory.SimpleFunc β ENNReal
      hf₀ : LE.le ⇑f₀ fun a => f a
      this : LE.le (⇑(f₀.comp g ⋯)) (Function.comp f g)
      ⊢ LE.le ((f₀.comp g ⋯).lintegral μ) (iSup fun g_1 => iSup fun x => g_1.lintegr …
    -/
    exact le_iSup_of_le (comp f₀ g hg.measurable) (by exact le_iSup (α := ℝ≥0∞) _ this)
    /-
      🎉 no goals
    -/
  · rw [← f₀.extend_comp_eq hg (const _ 0), ← SimpleFunc.lintegral_map, ←
      SimpleFunc.lintegral_eq_lintegral, ← lintegral]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasurableSpace β
      g : α → β
      hg : MeasurableEmbedding g
      f : β → ENNReal
      f₀ : MeasureTheory.SimpleFunc α ENNReal
      hf₀ : LE.le ⇑f₀ fun a => f (g a)
      ⊢ LE.le (MeasureTheory.lintegral (MeasureTheory.Measure.map g μ) fun a => (f₀. …
    -/
    refine lintegral_mono_ae (hg.ae_map_iff.2 <| Eventually.of_forall fun x => ?_)
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasurableSpace β
      g : α → β
      hg : MeasurableEmbedding g
      f : β → ENNReal
      f₀ : MeasureTheory.SimpleFunc α ENNReal
      hf₀ : LE.le ⇑f₀ fun a => f (g a)
      x : α
      ⊢ LE.le ((f₀.extend g hg (MeasureTheory.SimpleFunc.const β 0)) (g x)) (f (g x))
    -/
    exact (extend_apply _ _ _ _).trans_le (hf₀ _)
    /-
      🎉 no goals
    -/


/-- The `lintegral` transforms appropriately under a measurable equivalence `g : α ≃ᵐ β`.
(Compare `lintegral_map`, which applies to a wider class of functions `g : α → β`, but requires
measurability of the function being integrated.) -/
theorem lintegral_map_equiv [MeasurableSpace β] (f : β → ℝ≥0∞) (g : α ≃ᵐ β) :
    ∫⁻ a, f a ∂map g μ = ∫⁻ a, f (g a) ∂μ :=
  g.measurableEmbedding.lintegral_map f


protected theorem MeasurePreserving.lintegral_map_equiv [MeasurableSpace β] {ν : Measure β}
    (f : β → ℝ≥0∞) (g : α ≃ᵐ β) (hg : MeasurePreserving g μ ν) :
    ∫⁻ a, f a ∂ν = ∫⁻ a, f (g a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f : β → ENNReal
    g : MeasurableEquiv α β
    hg : MeasureTheory.MeasurePreserving (⇑g) μ ν
    ⊢ Eq (MeasureTheory.lintegral ν fun a => f a) (MeasureTheory.lintegral μ fun a …
  -/
  rw [← MeasureTheory.lintegral_map_equiv f g, hg.map_eq]
  /-
    🎉 no goals
  -/


theorem MeasurePreserving.lintegral_comp {mb : MeasurableSpace β} {ν : Measure β} {g : α → β}
    (hg : MeasurePreserving g μ ν) {f : β → ℝ≥0∞} (hf : Measurable f) :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            m : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            mb : MeasurableSpace β
                                            ν : MeasureTheory.Measure β
                                            g : α → β
                                            hg : MeasureTheory.MeasurePreserving g μ ν
                                            f : β → ENNReal
                                            hf : Measurable f
                                            ⊢ Eq (MeasureTheory.lintegral μ fun a => f (g a)) (MeasureTheory.lintegral ν f …
                                          -/
    ∫⁻ a, f (g a) ∂μ = ∫⁻ b, f b ∂ν := by rw [← hg.map_eq, lintegral_map hf hg.measurable]
                                          /-
                                            🎉 no goals
                                          -/


theorem MeasurePreserving.lintegral_comp_emb {mb : MeasurableSpace β} {ν : Measure β} {g : α → β}
    (hg : MeasurePreserving g μ ν) (hge : MeasurableEmbedding g) (f : β → ℝ≥0∞) :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            m : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            mb : MeasurableSpace β
                                            ν : MeasureTheory.Measure β
                                            g : α → β
                                            hg : MeasureTheory.MeasurePreserving g μ ν
                                            hge : MeasurableEmbedding g
                                            f : β → ENNReal
                                            ⊢ Eq (MeasureTheory.lintegral μ fun a => f (g a)) (MeasureTheory.lintegral ν f …
                                          -/
    ∫⁻ a, f (g a) ∂μ = ∫⁻ b, f b ∂ν := by rw [← hg.map_eq, hge.lintegral_map]
                                          /-
                                            🎉 no goals
                                          -/


theorem MeasurePreserving.setLIntegral_comp_preimage {mb : MeasurableSpace β} {ν : Measure β}
    {g : α → β} (hg : MeasurePreserving g μ ν) {s : Set β} (hs : MeasurableSet s) {f : β → ℝ≥0∞}
    (hf : Measurable f) : ∫⁻ a in g ⁻¹' s, f (g a) ∂μ = ∫⁻ b in s, f b ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mb : MeasurableSpace β
    ν : MeasureTheory.Measure β
    g : α → β
    hg : MeasureTheory.MeasurePreserving g μ ν
    s : Set β
    hs : MeasurableSet s
    f : β → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Set.preimage g s)) fun a => f (g a) …
  -/
  rw [← hg.map_eq, setLIntegral_map hs hf hg.measurable]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias MeasurePreserving.set_lintegral_comp_preimage := MeasurePreserving.setLIntegral_comp_preimage


theorem MeasurePreserving.setLIntegral_comp_preimage_emb {mb : MeasurableSpace β} {ν : Measure β}
    {g : α → β} (hg : MeasurePreserving g μ ν) (hge : MeasurableEmbedding g) (f : β → ℝ≥0∞)
    (s : Set β) : ∫⁻ a in g ⁻¹' s, f (g a) ∂μ = ∫⁻ b in s, f b ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mb : MeasurableSpace β
    ν : MeasureTheory.Measure β
    g : α → β
    hg : MeasureTheory.MeasurePreserving g μ ν
    hge : MeasurableEmbedding g
    f : β → ENNReal
    s : Set β
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Set.preimage g s)) fun a => f (g a) …
  -/
  rw [← hg.map_eq, hge.restrict_map, hge.lintegral_map]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias MeasurePreserving.set_lintegral_comp_preimage_emb :=
  MeasurePreserving.setLIntegral_comp_preimage_emb


theorem MeasurePreserving.setLIntegral_comp_emb {mb : MeasurableSpace β} {ν : Measure β}
    {g : α → β} (hg : MeasurePreserving g μ ν) (hge : MeasurableEmbedding g) (f : β → ℝ≥0∞)
    (s : Set α) : ∫⁻ a in s, f (g a) ∂μ = ∫⁻ b in g '' s, f b ∂ν := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    mb : MeasurableSpace β
    ν : MeasureTheory.Measure β
    g : α → β
    hg : MeasureTheory.MeasurePreserving g μ ν
    hge : MeasurableEmbedding g
    f : β → ENNReal
    s : Set α
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f (g a)) (MeasureTheory. …
  -/
  rw [← hg.setLIntegral_comp_preimage_emb hge, preimage_image_eq _ hge.injective]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-29")]
alias MeasurePreserving.set_lintegral_comp_emb := MeasurePreserving.setLIntegral_comp_emb


theorem lintegral_subtype_comap {s : Set α} (hs : MeasurableSet s) (f : α → ℝ≥0∞) :
    ∫⁻ x : s, f x ∂(μ.comap (↑)) = ∫⁻ x in s, f x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.comap Subtype.val μ) fun  …
  -/
  rw [← (MeasurableEmbedding.subtype_coe hs).lintegral_map, map_comap_subtype_coe hs]
  /-
    🎉 no goals
  -/


theorem setLIntegral_subtype {s : Set α} (hs : MeasurableSet s) (t : Set s) (f : α → ℝ≥0∞) :
    ∫⁻ x in t, f x ∂(μ.comap (↑)) = ∫⁻ x in (↑) '' t, f x ∂μ := by
  rw [(MeasurableEmbedding.subtype_coe hs).restrict_comap, lintegral_subtype_comap hs,
    restrict_restrict hs, inter_eq_right.2 (Subtype.coe_image_subset _ _)]


/-- If `f : α → ℝ≥0∞` has finite integral, then there exists a measurable set `s` of finite measure
such that the integral of `f` over `sᶜ` is less than a given positive number.

Also used to prove an `Lᵖ`-norm version in
`MeasureTheory.Memℒp.exists_eLpNorm_indicator_compl_le`. -/
theorem exists_setLintegral_compl_lt {f : α → ℝ≥0∞} (hf : ∫⁻ a, f a ∂μ ≠ ∞)
    {ε : ℝ≥0∞} (hε : ε ≠ 0) :
    ∃ s : Set α, MeasurableSet s ∧ μ s < ∞ ∧ ∫⁻ a in sᶜ, f a ∂μ < ε := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    ε : ENNReal
    hε : Ne ε 0
    ⊢ Exists fun s => And (MeasurableSet s) (And (LT.lt (μ s) Top.top) (LT.lt (Mea …
  -/
  by_cases hf₀ : ∫⁻ a, f a ∂μ = 0
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
      ε : ENNReal
      hε : Ne ε 0
      hf₀ : Eq (MeasureTheory.lintegral μ fun a => f a) 0
      ⊢ Exists fun s => And (MeasurableSet s) (And (LT.lt (μ s) Top.top) (LT.lt (Mea …
    -/
  · exact ⟨∅, .empty, by simp, by simpa [hf₀, pos_iff_ne_zero]⟩
    /-
      🎉 no goals
    -/
  obtain ⟨g, hgf, hg_meas, hgsupp, hgε⟩ :
      ∃ g ≤ f, Measurable g ∧ μ (support g) < ∞ ∧ ∫⁻ a, f a ∂μ - ε < ∫⁻ a, g a ∂μ := by
    obtain ⟨g, hgf, hgε⟩ : ∃ (g : α →ₛ ℝ≥0∞) (_ : g ≤ f), ∫⁻ a, f a ∂μ - ε < g.lintegral μ := by
      simpa only [← lt_iSup_iff, ← lintegral_def] using ENNReal.sub_lt_self hf hf₀ hε
    refine ⟨g, hgf, g.measurable, ?_, by rwa [g.lintegral_eq_lintegral]⟩
    exact SimpleFunc.FinMeasSupp.of_lintegral_ne_top <| ne_top_of_le_ne_top hf <|
      g.lintegral_eq_lintegral μ ▸ lintegral_mono hgf
  /-
    case neg.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    ε : ENNReal
    hε : Ne ε 0
    hf₀ : Not (Eq (MeasureTheory.lintegral μ fun a => f a) 0)
    g : α → ENNReal
    hgf : LE.le g f
    hg_meas : Measurable g
    hgsupp : LT.lt (μ (Function.support g)) Top.top
    hgε : LT.lt (HSub.hSub (MeasureTheory.lintegral μ fun a => f a) ε) (MeasureThe …
    ⊢ Exists fun s => And (MeasurableSet s) (And (LT.lt (μ s) Top.top) (LT.lt (Mea …
  -/
  refine ⟨_, measurableSet_support hg_meas, hgsupp, ?_⟩
  calc
    ∫⁻ a in (support g)ᶜ, f a ∂μ
      = ∫⁻ a in (support g)ᶜ, f a - g a ∂μ := setLIntegral_congr_fun
      (measurableSet_support hg_meas).compl <| ae_of_all _ <| by intro; simp_all
    _ ≤ ∫⁻ a, f a - g a ∂μ := setLIntegral_le_lintegral _ _
    _ = ∫⁻ a, f a ∂μ - ∫⁻ a, g a ∂μ :=
      lintegral_sub hg_meas (ne_top_of_le_ne_top hf <| lintegral_mono hgf) (ae_of_all _ hgf)
    _ < ε := ENNReal.sub_lt_of_lt_add (lintegral_mono hgf) <|
      ENNReal.lt_add_of_sub_lt_left (.inl hf) hgε


/-- For any function `f : α → ℝ≥0∞`, there exists a measurable function `g ≤ f` with the same
integral over any measurable set. -/
theorem exists_measurable_le_setLintegral_eq_of_integrable {f : α → ℝ≥0∞} (hf : ∫⁻ a, f a ∂μ ≠ ∞) :
    ∃ (g : α → ℝ≥0∞), Measurable g ∧ g ≤ f ∧ ∀ s : Set α, MeasurableSet s →
      ∫⁻ a in s, f a ∂μ = ∫⁻ a in s, g a ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), Measurab …
  -/
  obtain ⟨g, hmg, hgf, hifg⟩ := exists_measurable_le_lintegral_eq (μ := μ) f
  /-
    case intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    g : α → ENNReal
    hmg : Measurable g
    hgf : LE.le g f
    hifg : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ  …
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), Measurab …
  -/
  use g, hmg, hgf
  /-
    case right
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    g : α → ENNReal
    hmg : Measurable g
    hgf : LE.le g f
    hifg : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ  …
    ⊢ ∀ (s : Set α), MeasurableSet s → Eq (MeasureTheory.lintegral (μ.restrict s)  …
  -/
  refine fun s hms ↦ le_antisymm ?_ (lintegral_mono hgf)
  /-
    case right
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
    g : α → ENNReal
    hmg : Measurable g
    hgf : LE.le g f
    hifg : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ  …
    s : Set α
    hms : MeasurableSet s
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory.l …
  -/
  rw [← compl_compl s, setLintegral_compl hms.compl, setLintegral_compl hms.compl, hifg]
    /-
      case right
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
      g : α → ENNReal
      hmg : Measurable g
      hgf : LE.le g f
      hifg : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ  …
      s : Set α
      hms : MeasurableSet s
      ⊢ LE.le (HSub.hSub (MeasureTheory.lintegral μ fun a => g a) (MeasureTheory.lin …
    -/
  · gcongr; apply hgf
            /-
              🎉 no goals
            -/
    /-
      case right
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
      g : α → ENNReal
      hmg : Measurable g
      hgf : LE.le g f
      hifg : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ  …
      s : Set α
      hms : MeasurableSet s
      ⊢ Ne (MeasureTheory.lintegral (μ.restrict (HasCompl.compl s)) fun x => g x) To …
    -/
  · rw [hifg] at hf
    /-
      case right
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun a => g a) Top.top
      hmg : Measurable g
      hgf : LE.le g f
      hifg : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ  …
      s : Set α
      hms : MeasurableSet s
      ⊢ Ne (MeasureTheory.lintegral (μ.restrict (HasCompl.compl s)) fun x => g x) To …
    -/
    exact ne_top_of_le_ne_top hf (setLIntegral_le_lintegral _ _)
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun a => f a) Top.top
      g : α → ENNReal
      hmg : Measurable g
      hgf : LE.le g f
      hifg : Eq (MeasureTheory.lintegral μ fun a => f a) (MeasureTheory.lintegral μ  …
      s : Set α
      hms : MeasurableSet s
      ⊢ Ne (MeasureTheory.lintegral (μ.restrict (HasCompl.compl s)) fun x => f x) To …
    -/
  · exact ne_top_of_le_ne_top hf (setLIntegral_le_lintegral _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_subtype := setLIntegral_subtype


theorem lintegral_dirac' (a : α) {f : α → ℝ≥0∞} (hf : Measurable f) : ∫⁻ a, f a ∂dirac a = f a := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    a : α
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.dirac a) fun a => f a) (f …
  -/
  simp [lintegral_congr_ae (ae_eq_dirac' hf)]
  /-
    🎉 no goals
  -/


theorem lintegral_dirac [MeasurableSingletonClass α] (a : α) (f : α → ℝ≥0∞) :
                                   /-
                                     α : Type u_1
                                     inst✝¹ : MeasurableSpace α
                                     inst✝ : MeasurableSingletonClass α
                                     a : α
                                     f : α → ENNReal
                                     ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.dirac a) fun a => f a) (f …
                                   -/
    ∫⁻ a, f a ∂dirac a = f a := by simp [lintegral_congr_ae (ae_eq_dirac f)]
                                   /-
                                     🎉 no goals
                                   -/


theorem setLIntegral_dirac' {a : α} {f : α → ℝ≥0∞} (hf : Measurable f) {s : Set α}
    (hs : MeasurableSet s) [Decidable (a ∈ s)] :
    ∫⁻ x in s, f x ∂Measure.dirac a = if a ∈ s then f a else 0 := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    a : α
    f : α → ENNReal
    hf : Measurable f
    s : Set α
    hs : MeasurableSet s
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (MeasureTheory.lintegral ((MeasureTheory.Measure.dirac a).restrict s) fun …
  -/
  rw [restrict_dirac' hs]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    a : α
    f : α → ENNReal
    hf : Measurable f
    s : Set α
    hs : MeasurableSet s
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (MeasureTheory.lintegral (ite (Membership.mem s a) (MeasureTheory.Measure …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      a : α
      f : α → ENNReal
      hf : Measurable f
      s : Set α
      hs : MeasurableSet s
      inst✝ : Decidable (Membership.mem s a)
      h✝ : Membership.mem s a
      ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.dirac a) fun x => f x) (f …
    -/
  · exact lintegral_dirac' _ hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      a : α
      f : α → ENNReal
      hf : Measurable f
      s : Set α
      hs : MeasurableSet s
      inst✝ : Decidable (Membership.mem s a)
      h✝ : Not (Membership.mem s a)
      ⊢ Eq (MeasureTheory.lintegral 0 fun x => f x) 0
    -/
  · exact lintegral_zero_measure _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_dirac' := setLIntegral_dirac'


theorem setLIntegral_dirac {a : α} (f : α → ℝ≥0∞) (s : Set α) [MeasurableSingletonClass α]
    [Decidable (a ∈ s)] : ∫⁻ x in s, f x ∂Measure.dirac a = if a ∈ s then f a else 0 := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    a : α
    f : α → ENNReal
    s : Set α
    inst✝¹ : MeasurableSingletonClass α
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (MeasureTheory.lintegral ((MeasureTheory.Measure.dirac a).restrict s) fun …
  -/
  rw [restrict_dirac]
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    a : α
    f : α → ENNReal
    s : Set α
    inst✝¹ : MeasurableSingletonClass α
    inst✝ : Decidable (Membership.mem s a)
    ⊢ Eq (MeasureTheory.lintegral (ite (Membership.mem s a) (MeasureTheory.Measure …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      inst✝² : MeasurableSpace α
      a : α
      f : α → ENNReal
      s : Set α
      inst✝¹ : MeasurableSingletonClass α
      inst✝ : Decidable (Membership.mem s a)
      h✝ : Membership.mem s a
      ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.dirac a) fun x => f x) (f …
    -/
  · exact lintegral_dirac _ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : MeasurableSpace α
      a : α
      f : α → ENNReal
      s : Set α
      inst✝¹ : MeasurableSingletonClass α
      inst✝ : Decidable (Membership.mem s a)
      h✝ : Not (Membership.mem s a)
      ⊢ Eq (MeasureTheory.lintegral 0 fun x => f x) 0
    -/
  · exact lintegral_zero_measure _
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-29")]
alias set_lintegral_dirac := setLIntegral_dirac


theorem lintegral_count' {f : α → ℝ≥0∞} (hf : Measurable f) : ∫⁻ a, f a ∂count = ∑' a, f a := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (MeasureTheory.lintegral MeasureTheory.Measure.count fun a => f a) (tsum  …
  -/
  rw [count, lintegral_sum_measure]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (tsum fun i => MeasureTheory.lintegral (MeasureTheory.Measure.dirac i) fu …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (fun i => MeasureTheory.lintegral (MeasureTheory.Measure.dirac i) fun a = …
  -/
  exact funext fun a => lintegral_dirac' a hf
  /-
    🎉 no goals
  -/


theorem lintegral_count [MeasurableSingletonClass α] (f : α → ℝ≥0∞) :
    ∫⁻ a, f a ∂count = ∑' a, f a := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral MeasureTheory.Measure.count fun a => f a) (tsum  …
  -/
  rw [count, lintegral_sum_measure]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    f : α → ENNReal
    ⊢ Eq (tsum fun i => MeasureTheory.lintegral (MeasureTheory.Measure.dirac i) fu …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    f : α → ENNReal
    ⊢ Eq (fun i => MeasureTheory.lintegral (MeasureTheory.Measure.dirac i) fun a = …
  -/
  exact funext fun a => lintegral_dirac a f
  /-
    🎉 no goals
  -/


theorem _root_.ENNReal.tsum_const_eq [MeasurableSingletonClass α] (c : ℝ≥0∞) :
                                                         /-
                                                           α : Type u_1
                                                           inst✝¹ : MeasurableSpace α
                                                           inst✝ : MeasurableSingletonClass α
                                                           c : ENNReal
                                                           ⊢ Eq (tsum fun x => c) (HMul.hMul c (MeasureTheory.Measure.count Set.univ))
                                                         -/
    ∑' _ : α, c = c * Measure.count (univ : Set α) := by rw [← lintegral_count, lintegral_const]
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- Markov's inequality for the counting measure with hypothesis using `tsum` in `ℝ≥0∞`. -/
theorem _root_.ENNReal.count_const_le_le_of_tsum_le [MeasurableSingletonClass α] {a : α → ℝ≥0∞}
    (a_mble : Measurable a) {c : ℝ≥0∞} (tsum_le_c : ∑' i, a i ≤ c) {ε : ℝ≥0∞} (ε_ne_zero : ε ≠ 0)
    (ε_ne_top : ε ≠ ∞) : Measure.count { i : α | ε ≤ a i } ≤ c / ε := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α → ENNReal
    a_mble : Measurable a
    c : ENNReal
    tsum_le_c : LE.le (tsum fun i => a i) c
    ε : ENNReal
    ε_ne_zero : Ne ε 0
    ε_ne_top : Ne ε Top.top
    ⊢ LE.le (MeasureTheory.Measure.count (setOf fun i => LE.le ε (a i))) (HDiv.hDi …
  -/
  rw [← lintegral_count] at tsum_le_c
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α → ENNReal
    a_mble : Measurable a
    c : ENNReal
    tsum_le_c : LE.le (MeasureTheory.lintegral MeasureTheory.Measure.count fun a_1 …
    ε : ENNReal
    ε_ne_zero : Ne ε 0
    ε_ne_top : Ne ε Top.top
    ⊢ LE.le (MeasureTheory.Measure.count (setOf fun i => LE.le ε (a i))) (HDiv.hDi …
  -/
  apply (MeasureTheory.meas_ge_le_lintegral_div a_mble.aemeasurable ε_ne_zero ε_ne_top).trans
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α → ENNReal
    a_mble : Measurable a
    c : ENNReal
    tsum_le_c : LE.le (MeasureTheory.lintegral MeasureTheory.Measure.count fun a_1 …
    ε : ENNReal
    ε_ne_zero : Ne ε 0
    ε_ne_top : Ne ε Top.top
    ⊢ LE.le (HDiv.hDiv (MeasureTheory.lintegral MeasureTheory.Measure.count fun a_ …
  -/
  exact ENNReal.div_le_div tsum_le_c rfl.le
  /-
    🎉 no goals
  -/


/-- Markov's inequality for counting measure with hypothesis using `tsum` in `ℝ≥0`. -/
theorem _root_.NNReal.count_const_le_le_of_tsum_le [MeasurableSingletonClass α] {a : α → ℝ≥0}
    (a_mble : Measurable a) (a_summable : Summable a) {c : ℝ≥0} (tsum_le_c : ∑' i, a i ≤ c)
    {ε : ℝ≥0} (ε_ne_zero : ε ≠ 0) : Measure.count { i : α | ε ≤ a i } ≤ c / ε := by
  rw [show (fun i => ε ≤ a i) = fun i => (ε : ℝ≥0∞) ≤ ((↑) ∘ a) i by
      funext i
      simp only [ENNReal.coe_le_coe, Function.comp]]
  apply
    ENNReal.count_const_le_le_of_tsum_le (measurable_coe_nnreal_ennreal.comp a_mble) _
      (mod_cast ε_ne_zero) (@ENNReal.coe_ne_top ε)
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α → NNReal
    a_mble : Measurable a
    a_summable : Summable a
    c : NNReal
    tsum_le_c : LE.le (tsum fun i => a i) c
    ε : NNReal
    ε_ne_zero : Ne ε 0
    ⊢ LE.le (tsum fun i => Function.comp ENNReal.ofNNReal a i) ↑c
  -/
  convert ENNReal.coe_le_coe.mpr tsum_le_c
  /-
    case h.e'_3
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α → NNReal
    a_mble : Measurable a
    a_summable : Summable a
    c : NNReal
    tsum_le_c : LE.le (tsum fun i => a i) c
    ε : NNReal
    ε_ne_zero : Ne ε 0
    ⊢ Eq (tsum fun i => Function.comp ENNReal.ofNNReal a i) ↑(tsum fun i => a i)
  -/
  simp_rw [Function.comp_apply]
  /-
    case h.e'_3
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    a : α → NNReal
    a_mble : Measurable a
    a_summable : Summable a
    c : NNReal
    tsum_le_c : LE.le (tsum fun i => a i) c
    ε : NNReal
    ε_ne_zero : Ne ε 0
    ⊢ Eq (tsum fun i => ↑(a i)) ↑(tsum fun i => a i)
  -/
  rw [ENNReal.tsum_coe_eq a_summable.hasSum]
  /-
    🎉 no goals
  -/


theorem lintegral_countable' [Countable α] [MeasurableSingletonClass α] (f : α → ℝ≥0∞) :
    ∫⁻ a, f a ∂μ = ∑' a, f a * μ {a} := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable α
    inst✝ : MeasurableSingletonClass α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun a => f a) (tsum fun a => HMul.hMul (f a) ( …
  -/
  conv_lhs => rw [← sum_smul_dirac μ, lintegral_sum_measure]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable α
    inst✝ : MeasurableSingletonClass α
    f : α → ENNReal
    ⊢ Eq (tsum fun i => MeasureTheory.lintegral (HSMul.hSMul (μ (Singleton.singlet …
  -/
  congr 1 with a : 1
  /-
    case e_f.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : Countable α
    inst✝ : MeasurableSingletonClass α
    f : α → ENNReal
    a : α
    ⊢ Eq (MeasureTheory.lintegral (HSMul.hSMul (μ (Singleton.singleton a)) (Measur …
  -/
  rw [lintegral_smul_measure, lintegral_dirac, mul_comm]
  /-
    🎉 no goals
  -/


theorem lintegral_singleton' {f : α → ℝ≥0∞} (hf : Measurable f) (a : α) :
    ∫⁻ x in {a}, f x ∂μ = f a * μ {a} := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    a : α
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Singleton.singleton a)) fun x => f  …
  -/
  simp only [restrict_singleton, lintegral_smul_measure, lintegral_dirac' _ hf, mul_comm]
  /-
    🎉 no goals
  -/


theorem lintegral_singleton [MeasurableSingletonClass α] (f : α → ℝ≥0∞) (a : α) :
    ∫⁻ x in {a}, f x ∂μ = f a * μ {a} := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSingletonClass α
    f : α → ENNReal
    a : α
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict (Singleton.singleton a)) fun x => f  …
  -/
  simp only [restrict_singleton, lintegral_smul_measure, lintegral_dirac, mul_comm]
  /-
    🎉 no goals
  -/


theorem lintegral_countable [MeasurableSingletonClass α] (f : α → ℝ≥0∞) {s : Set α}
    (hs : s.Countable) : ∫⁻ a in s, f a ∂μ = ∑' a : s, f a * μ {(a : α)} :=
  calc
                                                           /-
                                                             α : Type u_1
                                                             m : MeasurableSpace α
                                                             μ : MeasureTheory.Measure α
                                                             inst✝ : MeasurableSingletonClass α
                                                             f : α → ENNReal
                                                             s : Set α
                                                             hs : s.Countable
                                                             ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory.lint …
                                                           -/
    ∫⁻ a in s, f a ∂μ = ∫⁻ a in ⋃ x ∈ s, {x}, f a ∂μ := by rw [biUnion_of_singleton]
                                                           /-
                                                             🎉 no goals
                                                           -/
    _ = ∑' a : s, ∫⁻ x in {(a : α)}, f x ∂μ :=
      (lintegral_biUnion hs (fun _ _ => measurableSet_singleton _) (pairwiseDisjoint_fiber id s) _)
                                          /-
                                            α : Type u_1
                                            m : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            inst✝ : MeasurableSingletonClass α
                                            f : α → ENNReal
                                            s : Set α
                                            hs : s.Countable
                                            ⊢ Eq (tsum fun a => MeasureTheory.lintegral (μ.restrict (Singleton.singleton ↑ …
                                          -/
    _ = ∑' a : s, f a * μ {(a : α)} := by simp only [lintegral_singleton]
                                          /-
                                            🎉 no goals
                                          -/


theorem lintegral_insert [MeasurableSingletonClass α] {a : α} {s : Set α} (h : a ∉ s)
    (f : α → ℝ≥0∞) : ∫⁻ x in insert a s, f x ∂μ = f a * μ {a} + ∫⁻ x in s, f x ∂μ := by
  rw [← union_singleton, lintegral_union (measurableSet_singleton a), lintegral_singleton,
    add_comm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSingletonClass α
    a : α
    s : Set α
    h : Not (Membership.mem s a)
    f : α → ENNReal
    ⊢ Disjoint s (Singleton.singleton a)
  -/
  rwa [disjoint_singleton_right]
  /-
    🎉 no goals
  -/


theorem lintegral_finset [MeasurableSingletonClass α] (s : Finset α) (f : α → ℝ≥0∞) :
    ∫⁻ x in s, f x ∂μ = ∑ x ∈ s, f x * μ {x} := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSingletonClass α
    s : Finset α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict ↑s) fun x => f x) (s.sum fun x => HM …
  -/
  simp only [lintegral_countable _ s.countable_toSet, ← Finset.tsum_subtype']
  /-
    🎉 no goals
  -/


theorem lintegral_fintype [MeasurableSingletonClass α] [Fintype α] (f : α → ℝ≥0∞) :
    ∫⁻ x, f x ∂μ = ∑ x, f x * μ {x} := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSingletonClass α
    inst✝ : Fintype α
    f : α → ENNReal
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f x) (Finset.univ.sum fun x => HMul.h …
  -/
  rw [← lintegral_finset, Finset.coe_univ, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


theorem lintegral_unique [Unique α] (f : α → ℝ≥0∞) : ∫⁻ x, f x ∂μ = f default * μ univ :=
  calc
    ∫⁻ x, f x ∂μ = ∫⁻ _, f default ∂μ := lintegral_congr <| Unique.forall_iff.2 rfl
    _ = f default * μ univ := lintegral_const _


theorem ae_lt_top' {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) (h2f : ∫⁻ x, f x ∂μ ≠ ∞) :
    ∀ᵐ x ∂μ, f x < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    h2f : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ⊢ Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
  -/
  simp_rw [ae_iff, ENNReal.not_lt_top]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    h2f : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ⊢ Eq (μ (setOf fun a => Eq (f a) Top.top)) 0
  -/
  exact measure_eq_top_of_lintegral_ne_top hf h2f
  /-
    🎉 no goals
  -/


theorem ae_lt_top {f : α → ℝ≥0∞} (hf : Measurable f) (h2f : ∫⁻ x, f x ∂μ ≠ ∞) :
    ∀ᵐ x ∂μ, f x < ∞ :=
  ae_lt_top' hf.aemeasurable h2f


/-- Lebesgue integral of a bounded function over a set of finite measure is finite.
Note that this lemma assumes no regularity of either `f` or `s`. -/
theorem setLIntegral_lt_top_of_le_nnreal {s : Set α} (hs : μ s ≠ ∞) {f : α → ℝ≥0∞}
    (hbdd : ∃ y : ℝ≥0, ∀ x ∈ s, f x ≤ y) : ∫⁻ x in s, f x ∂μ < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Ne (μ s) Top.top
    f : α → ENNReal
    hbdd : Exists fun y => ∀ (x : α), Membership.mem s x → LE.le (f x) ↑y
    ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict s) fun x => f x) Top.top
  -/
  obtain ⟨M, hM⟩ := hbdd
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Ne (μ s) Top.top
    f : α → ENNReal
    M : NNReal
    hM : ∀ (x : α), Membership.mem s x → LE.le (f x) ↑M
    ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict s) fun x => f x) Top.top
  -/
  refine lt_of_le_of_lt (setLIntegral_mono measurable_const hM) ?_
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : Ne (μ s) Top.top
    f : α → ENNReal
    M : NNReal
    hM : ∀ (x : α), Membership.mem s x → LE.le (f x) ↑M
    ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict s) fun x => ↑M) Top.top
  -/
  simp [ENNReal.mul_lt_top, hs.lt_top]
  /-
    🎉 no goals
  -/


/-- Lebesgue integral of a bounded function over a set of finite measure is finite.
Note that this lemma assumes no regularity of either `f` or `s`. -/
theorem setLIntegral_lt_top_of_bddAbove {s : Set α} (hs : μ s ≠ ∞) {f : α → ℝ≥0}
    (hbdd : BddAbove (f '' s)) : ∫⁻ x in s, f x ∂μ < ∞ :=
  setLIntegral_lt_top_of_le_nnreal hs <| hbdd.imp fun _M hM _x hx ↦
    ENNReal.coe_le_coe.2 <| hM (mem_image_of_mem f hx)


@[deprecated (since := "2024-06-29")]
alias set_lintegral_lt_top_of_bddAbove := setLIntegral_lt_top_of_bddAbove


theorem setLIntegral_lt_top_of_isCompact [TopologicalSpace α] {s : Set α}
    (hs : μ s ≠ ∞) (hsc : IsCompact s) {f : α → ℝ≥0} (hf : Continuous f) :
    ∫⁻ x in s, f x ∂μ < ∞ :=
  setLIntegral_lt_top_of_bddAbove hs (hsc.image hf).bddAbove


@[deprecated (since := "2024-06-29")]
alias set_lintegral_lt_top_of_isCompact := setLIntegral_lt_top_of_isCompact


theorem _root_.IsFiniteMeasure.lintegral_lt_top_of_bounded_to_ennreal {α : Type*}
    [MeasurableSpace α] (μ : Measure α) [μ_fin : IsFiniteMeasure μ] {f : α → ℝ≥0∞}
    (f_bdd : ∃ c : ℝ≥0, ∀ x, f x ≤ c) : ∫⁻ x, f x ∂μ < ∞ := by
  /-
    α : Type u_5
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    μ_fin : MeasureTheory.IsFiniteMeasure μ
    f : α → ENNReal
    f_bdd : Exists fun c => ∀ (x : α), LE.le (f x) ↑c
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => f x) Top.top
  -/
  rw [← μ.restrict_univ]
  /-
    α : Type u_5
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    μ_fin : MeasureTheory.IsFiniteMeasure μ
    f : α → ENNReal
    f_bdd : Exists fun c => ∀ (x : α), LE.le (f x) ↑c
    ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict Set.univ) fun x => f x) Top.top
  -/
  refine setLIntegral_lt_top_of_le_nnreal (measure_ne_top _ _) ?_
  /-
    α : Type u_5
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    μ_fin : MeasureTheory.IsFiniteMeasure μ
    f : α → ENNReal
    f_bdd : Exists fun c => ∀ (x : α), LE.le (f x) ↑c
    ⊢ Exists fun y => ∀ (x : α), Membership.mem Set.univ x → LE.le (f x) ↑y
  -/
  simpa using f_bdd
  /-
    🎉 no goals
  -/


/-- If a monotone sequence of functions has an upper bound and the sequence of integrals of these
functions tends to the integral of the upper bound, then the sequence of functions converges
almost everywhere to the upper bound. Auxiliary version assuming moreover that the
functions in the sequence are ae measurable. -/
lemma tendsto_of_lintegral_tendsto_of_monotone_aux {α : Type*} {mα : MeasurableSpace α}
    {f : ℕ → α → ℝ≥0∞} {F : α → ℝ≥0∞} {μ : Measure α}
    (hf_meas : ∀ n, AEMeasurable (f n) μ) (hF_meas : AEMeasurable F μ)
    (hf_tendsto : Tendsto (fun i ↦ ∫⁻ a, f i a ∂μ) atTop (𝓝 (∫⁻ a, F a ∂μ)))
    (hf_mono : ∀ᵐ a ∂μ, Monotone (fun i ↦ f i a))
    (h_bound : ∀ᵐ a ∂μ, ∀ i, f i a ≤ F a) (h_int_finite : ∫⁻ a, F a ∂μ ≠ ∞) :
    ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) atTop (𝓝 (F a)) := by
  have h_bound_finite : ∀ᵐ a ∂μ, F a ≠ ∞ := by
    filter_upwards [ae_lt_top' hF_meas h_int_finite] with a ha using ha.ne
  have h_exists : ∀ᵐ a ∂μ, ∃ l, Tendsto (fun i ↦ f i a) atTop (𝓝 l) := by
    filter_upwards [h_bound, h_bound_finite, hf_mono] with a h_le h_fin h_mono
    have h_tendsto : Tendsto (fun i ↦ f i a) atTop atTop ∨
        ∃ l, Tendsto (fun i ↦ f i a) atTop (𝓝 l) := tendsto_of_monotone h_mono
    cases' h_tendsto with h_absurd h_tendsto
    · rw [tendsto_atTop_atTop_iff_of_monotone h_mono] at h_absurd
      obtain ⟨i, hi⟩ := h_absurd (F a + 1)
      refine absurd (hi.trans (h_le _)) (not_le.mpr ?_)
      exact ENNReal.lt_add_right h_fin one_ne_zero
    · exact h_tendsto
  classical
  let F' : α → ℝ≥0∞ := fun a ↦ if h : ∃ l, Tendsto (fun i ↦ f i a) atTop (𝓝 l)
    then h.choose else ∞
  have hF'_tendsto : ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) atTop (𝓝 (F' a)) := by
    filter_upwards [h_exists] with a ha
    simp_rw [F', dif_pos ha]
    exact ha.choose_spec
  suffices F' =ᵐ[μ] F by
    filter_upwards [this, hF'_tendsto] with a h_eq h_tendsto using h_eq ▸ h_tendsto
  have hF'_le : F' ≤ᵐ[μ] F := by
    filter_upwards [h_bound, hF'_tendsto] with a h_le h_tendsto
    exact le_of_tendsto' h_tendsto (fun m ↦ h_le _)
  suffices ∫⁻ a, F' a ∂μ = ∫⁻ a, F a ∂μ from
    ae_eq_of_ae_le_of_lintegral_le hF'_le (this ▸ h_int_finite) hF_meas this.symm.le
  refine tendsto_nhds_unique ?_ hf_tendsto
  exact lintegral_tendsto_of_tendsto_of_monotone hf_meas hf_mono hF'_tendsto


/-- If a monotone sequence of functions has an upper bound and the sequence of integrals of these
functions tends to the integral of the upper bound, then the sequence of functions converges
almost everywhere to the upper bound. -/
lemma tendsto_of_lintegral_tendsto_of_monotone {α : Type*} {mα : MeasurableSpace α}
    {f : ℕ → α → ℝ≥0∞} {F : α → ℝ≥0∞} {μ : Measure α}
    (hF_meas : AEMeasurable F μ)
    (hf_tendsto : Tendsto (fun i ↦ ∫⁻ a, f i a ∂μ) atTop (𝓝 (∫⁻ a, F a ∂μ)))
    (hf_mono : ∀ᵐ a ∂μ, Monotone (fun i ↦ f i a))
    (h_bound : ∀ᵐ a ∂μ, ∀ i, f i a ≤ F a) (h_int_finite : ∫⁻ a, F a ∂μ ≠ ∞) :
    ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) atTop (𝓝 (F a)) := by
  have : ∀ n, ∃ g : α → ℝ≥0∞, Measurable g ∧ g ≤ f n ∧ ∫⁻ a, f n a ∂μ = ∫⁻ a, g a ∂μ :=
    fun n ↦ exists_measurable_le_lintegral_eq _ _
  /-
    α : Type u_5
    mα : MeasurableSpace α
    f : Nat → α → ENNReal
    F : α → ENNReal
    μ : MeasureTheory.Measure α
    hF_meas : AEMeasurable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f i a …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Measu …
    h_int_finite : Ne (MeasureTheory.lintegral μ fun a => F a) Top.top
    this : ∀ (n : Nat), Exists fun g => And (Measurable g) (And (LE.le g (f n)) (E …
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  choose g gmeas gf hg using this
  /-
    α : Type u_5
    mα : MeasurableSpace α
    f : Nat → α → ENNReal
    F : α → ENNReal
    μ : MeasureTheory.Measure α
    hF_meas : AEMeasurable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f i a …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Measu …
    h_int_finite : Ne (MeasureTheory.lintegral μ fun a => F a) Top.top
    g : Nat → α → ENNReal
    gmeas : ∀ (n : Nat), Measurable (g n)
    gf : ∀ (n : Nat), LE.le (g n) (f n)
    hg : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => f n a) (MeasureTheory …
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  let g' : ℕ → α → ℝ≥0∞ := Nat.rec (g 0) (fun n I x ↦ max (g (n+1) x) (I x))
  have M n : Measurable (g' n) := by
    induction n with
    | zero => simp [g', gmeas 0]
    | succ n ih => exact Measurable.max (gmeas (n+1)) ih
  have I : ∀ n x, g n x ≤ g' n x := by
    intro n x
    cases n with | zero | succ => simp [g']
  have I' : ∀ᵐ x ∂μ, ∀ n, g' n x ≤ f n x := by
    filter_upwards [hf_mono] with x hx n
    induction n with
    | zero => simpa [g'] using gf 0 x
    | succ n ih => exact max_le (gf (n+1) x) (ih.trans (hx (Nat.le_succ n)))
  have Int_eq n : ∫⁻ x, g' n x ∂μ = ∫⁻ x, f n x ∂μ := by
    apply le_antisymm
    · apply lintegral_mono_ae
      filter_upwards [I'] with x hx using hx n
    · rw [hg n]
      exact lintegral_mono (I n)
  have : ∀ᵐ a ∂μ, Tendsto (fun i ↦ g' i a) atTop (𝓝 (F a)) := by
    apply tendsto_of_lintegral_tendsto_of_monotone_aux _ hF_meas _ _ _ h_int_finite
    · exact fun n ↦ (M n).aemeasurable
    · simp_rw [Int_eq]
      exact hf_tendsto
    · exact Eventually.of_forall (fun x ↦ monotone_nat_of_le_succ (fun n ↦ le_max_right _ _))
    · filter_upwards [h_bound, I'] with x h'x hx n using (hx n).trans (h'x n)
  /-
    α : Type u_5
    mα : MeasurableSpace α
    f : Nat → α → ENNReal
    F : α → ENNReal
    μ : MeasureTheory.Measure α
    hF_meas : AEMeasurable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f i a …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Measu …
    h_int_finite : Ne (MeasureTheory.lintegral μ fun a => F a) Top.top
    g : Nat → α → ENNReal
    gmeas : ∀ (n : Nat), Measurable (g n)
    gf : ∀ (n : Nat), LE.le (g n) (f n)
    hg : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => f n a) (MeasureTheory …
    g' : Nat → α → ENNReal := fun t => Nat.rec (motive := fun x => α → ENNReal) (g …
    M : ∀ (n : Nat), Measurable (g' n)
    I : ∀ (n : Nat) (x : α), LE.le (g n x) (g' n x)
    I' : Filter.Eventually (fun x => ∀ (n : Nat), LE.le (g' n x) (f n x)) (Measure …
    Int_eq : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun x => g' n x) (MeasureT …
    this : Filter.Eventually (fun a => Filter.Tendsto (fun i => g' i a) Filter.atT …
    ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) Filter.atTop (nh …
  -/
  filter_upwards [this, I', h_bound] with x hx h'x h''x
  /-
    case h
    α : Type u_5
    mα : MeasurableSpace α
    f : Nat → α → ENNReal
    F : α → ENNReal
    μ : MeasureTheory.Measure α
    hF_meas : AEMeasurable F μ
    hf_tendsto : Filter.Tendsto (fun i => MeasureTheory.lintegral μ fun a => f i a …
    hf_mono : Filter.Eventually (fun a => Monotone fun i => f i a) (MeasureTheory. …
    h_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (f i a) (F a)) (Measu …
    h_int_finite : Ne (MeasureTheory.lintegral μ fun a => F a) Top.top
    g : Nat → α → ENNReal
    gmeas : ∀ (n : Nat), Measurable (g n)
    gf : ∀ (n : Nat), LE.le (g n) (f n)
    hg : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => f n a) (MeasureTheory …
    g' : Nat → α → ENNReal := fun t => Nat.rec (motive := fun x => α → ENNReal) (g …
    M : ∀ (n : Nat), Measurable (g' n)
    I : ∀ (n : Nat) (x : α), LE.le (g n x) (g' n x)
    I' : Filter.Eventually (fun x => ∀ (n : Nat), LE.le (g' n x) (f n x)) (Measure …
    Int_eq : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun x => g' n x) (MeasureT …
    this : Filter.Eventually (fun a => Filter.Tendsto (fun i => g' i a) Filter.atT …
    x : α
    hx : Filter.Tendsto (fun i => g' i x) Filter.atTop (nhds (F x))
    h'x : ∀ (n : Nat), LE.le (g' n x) (f n x)
    h''x : ∀ (i : Nat), LE.le (f i x) (F x)
    ⊢ Filter.Tendsto (fun i => f i x) Filter.atTop (nhds (F x))
  -/
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le hx tendsto_const_nhds h'x h''x
  /-
    🎉 no goals
  -/


/-- If an antitone sequence of functions has a lower bound and the sequence of integrals of these
functions tends to the integral of the lower bound, then the sequence of functions converges
almost everywhere to the lower bound. -/
lemma tendsto_of_lintegral_tendsto_of_antitone {α : Type*} {mα : MeasurableSpace α}
    {f : ℕ → α → ℝ≥0∞} {F : α → ℝ≥0∞} {μ : Measure α}
    (hf_meas : ∀ n, AEMeasurable (f n) μ)
    (hf_tendsto : Tendsto (fun i ↦ ∫⁻ a, f i a ∂μ) atTop (𝓝 (∫⁻ a, F a ∂μ)))
    (hf_mono : ∀ᵐ a ∂μ, Antitone (fun i ↦ f i a))
    (h_bound : ∀ᵐ a ∂μ, ∀ i, F a ≤ f i a) (h0 : ∫⁻ a, f 0 a ∂μ ≠ ∞) :
    ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) atTop (𝓝 (F a)) := by
  have h_int_finite : ∫⁻ a, F a ∂μ ≠ ∞ := by
    refine ((lintegral_mono_ae ?_).trans_lt h0.lt_top).ne
    filter_upwards [h_bound] with a ha using ha 0
  have h_exists : ∀ᵐ a ∂μ, ∃ l, Tendsto (fun i ↦ f i a) atTop (𝓝 l) := by
    filter_upwards [hf_mono] with a h_mono
    rcases _root_.tendsto_of_antitone h_mono with h | h
    · refine ⟨0, h.mono_right ?_⟩
      rw [OrderBot.atBot_eq]
      exact pure_le_nhds _
    · exact h
  classical
  let F' : α → ℝ≥0∞ := fun a ↦ if h : ∃ l, Tendsto (fun i ↦ f i a) atTop (𝓝 l)
    then h.choose else ∞
  have hF'_tendsto : ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) atTop (𝓝 (F' a)) := by
    filter_upwards [h_exists] with a ha
    simp_rw [F', dif_pos ha]
    exact ha.choose_spec
  suffices F' =ᵐ[μ] F by
    filter_upwards [this, hF'_tendsto] with a h_eq h_tendsto using h_eq ▸ h_tendsto
  have hF'_le : F ≤ᵐ[μ] F' := by
    filter_upwards [h_bound, hF'_tendsto] with a h_le h_tendsto
    exact ge_of_tendsto' h_tendsto (fun m ↦ h_le _)
  suffices ∫⁻ a, F' a ∂μ = ∫⁻ a, F a ∂μ by
    refine (ae_eq_of_ae_le_of_lintegral_le hF'_le h_int_finite ?_ this.le).symm
    exact ENNReal.aemeasurable_of_tendsto hf_meas hF'_tendsto
  refine tendsto_nhds_unique ?_ hf_tendsto
  exact lintegral_tendsto_of_tendsto_of_antitone hf_meas hf_mono h0 hF'_tendsto


variable (μ) in
/-- If `μ` is an s-finite measure, then for any function `f`
there exists a measurable function `g ≤ f`
that has the same Lebesgue integral over every set.

For the integral over the whole space, the statement is true without extra assumptions,
see `exists_measurable_le_lintegral_eq`.
See also `MeasureTheory.Measure.restrict_toMeasurable_of_sFinite` for a similar result. -/
theorem exists_measurable_le_forall_setLIntegral_eq [SFinite μ] (f : α → ℝ≥0∞) :
    ∃ g : α → ℝ≥0∞, Measurable g ∧ g ≤ f ∧ ∀ s, ∫⁻ a in s, f a ∂μ = ∫⁻ a in s, g a ∂μ := by
  -- We only need to prove the `≤` inequality for the integrals, the other one follows from `g ≤ f`.
  rsuffices ⟨g, hgm, hgle, hleg⟩ :
      ∃ g : α → ℝ≥0∞, Measurable g ∧ g ≤ f ∧ ∀ s, ∫⁻ a in s, f a ∂μ ≤ ∫⁻ a in s, g a ∂μ
    /-
      case intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f g : α → ENNReal
      hgm : Measurable g
      hgle : LE.le g f
      hleg : ∀ (s : Set α), LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => f …
      ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), Eq (Meas …
    -/
  · exact ⟨g, hgm, hgle, fun s ↦ (hleg s).antisymm (lintegral_mono hgle)⟩
    /-
      🎉 no goals
    -/
  -- Without loss of generality, `μ` is a finite measure.
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    f : α → ENNReal
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), LE.le (M …
  -/
  wlog h : IsFiniteMeasure μ generalizing μ
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → ENNReal
      this : ∀ (μ : MeasureTheory.Measure α) [inst : MeasureTheory.SFinite μ], Measu …
      h : Not (MeasureTheory.IsFiniteMeasure μ)
      ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), LE.le (M …
    -/
  · choose g hgm hgle hgint using fun n ↦ @this (sfiniteSeq μ n) _ inferInstance
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → ENNReal
      this : ∀ (μ : MeasureTheory.Measure α) [inst : MeasureTheory.SFinite μ], Measu …
      h : Not (MeasureTheory.IsFiniteMeasure μ)
      g : Nat → α → ENNReal
      hgm : ∀ (n : Nat), Measurable (g n)
      hgle : ∀ (n : Nat), LE.le (g n) f
      hgint : ∀ (n : Nat) (s : Set α), LE.le (MeasureTheory.lintegral ((MeasureTheor …
      ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), LE.le (M …
    -/
    refine ⟨fun x ↦ ⨆ n, g n x, .iSup hgm, fun x ↦ iSup_le (hgle · x), fun s ↦ ?_⟩
    rw [← sum_sfiniteSeq μ, Measure.restrict_sum_of_countable,
      lintegral_sum_measure, lintegral_sum_measure]
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SFinite μ
      f : α → ENNReal
      this : ∀ (μ : MeasureTheory.Measure α) [inst : MeasureTheory.SFinite μ], Measu …
      h : Not (MeasureTheory.IsFiniteMeasure μ)
      g : Nat → α → ENNReal
      hgm : ∀ (n : Nat), Measurable (g n)
      hgle : ∀ (n : Nat), LE.le (g n) f
      hgint : ∀ (n : Nat) (s : Set α), LE.le (MeasureTheory.lintegral ((MeasureTheor …
      s : Set α
      ⊢ LE.le (tsum fun i => MeasureTheory.lintegral ((MeasureTheory.sfiniteSeq μ i) …
    -/
    exact ENNReal.tsum_le_tsum fun n ↦ (hgint n s).trans (lintegral_mono fun x ↦ le_iSup (g · x) _)
    /-
      🎉 no goals
    -/
  -- According to `exists_measurable_le_lintegral_eq`, for any natural `n`
  -- we can choose a measurable function $g_{n}$
  -- such that $g_{n}(x) ≤ \min (f(x), n)$ for all $x$
  -- and both sides have the same integral over the whole space w.r.t. $μ$.
  have (n : ℕ): ∃ g : α → ℝ≥0∞, Measurable g ∧ g ≤ f ∧ g ≤ n ∧
      ∫⁻ a, min (f a) n ∂μ = ∫⁻ a, g a ∂μ := by
    simpa [and_assoc] using exists_measurable_le_lintegral_eq μ (f ⊓ n)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    h : MeasureTheory.IsFiniteMeasure μ
    this : ∀ (n : Nat), Exists fun g => And (Measurable g) (And (LE.le g f) (And ( …
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), LE.le (M …
  -/
  choose g hgm hgf hgle hgint using this
  -- Let `φ` be the pointwise supremum of the functions $g_{n}$.
  -- Clearly, `φ` is a measurable function and `φ ≤ f`.
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    h : MeasureTheory.IsFiniteMeasure μ
    g : Nat → α → ENNReal
    hgm : ∀ (n : Nat), Measurable (g n)
    hgf : ∀ (n : Nat), LE.le (g n) f
    hgle : ∀ (n : Nat), LE.le (g n) ↑n
    hgint : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => Min.min (f a) ↑n)  …
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), LE.le (M …
  -/
  set φ : α → ℝ≥0∞ := fun x ↦ ⨆ n, g n x
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    h : MeasureTheory.IsFiniteMeasure μ
    g : Nat → α → ENNReal
    hgm : ∀ (n : Nat), Measurable (g n)
    hgf : ∀ (n : Nat), LE.le (g n) f
    hgle : ∀ (n : Nat), LE.le (g n) ↑n
    hgint : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => Min.min (f a) ↑n)  …
    φ : α → ENNReal := fun x => iSup fun n => g n x
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), LE.le (M …
  -/
  have hφm : Measurable φ := by measurability
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    h : MeasureTheory.IsFiniteMeasure μ
    g : Nat → α → ENNReal
    hgm : ∀ (n : Nat), Measurable (g n)
    hgf : ∀ (n : Nat), LE.le (g n) f
    hgle : ∀ (n : Nat), LE.le (g n) ↑n
    hgint : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => Min.min (f a) ↑n)  …
    φ : α → ENNReal := fun x => iSup fun n => g n x
    hφm : Measurable φ
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), LE.le (M …
  -/
  have hφle : φ ≤ f := fun x ↦ iSup_le (hgf · x)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    h : MeasureTheory.IsFiniteMeasure μ
    g : Nat → α → ENNReal
    hgm : ∀ (n : Nat), Measurable (g n)
    hgf : ∀ (n : Nat), LE.le (g n) f
    hgle : ∀ (n : Nat), LE.le (g n) ↑n
    hgint : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => Min.min (f a) ↑n)  …
    φ : α → ENNReal := fun x => iSup fun n => g n x
    hφm : Measurable φ
    hφle : LE.le φ f
    ⊢ Exists fun g => And (Measurable g) (And (LE.le g f) (∀ (s : Set α), LE.le (M …
  -/
  refine ⟨φ, hφm, hφle, fun s ↦ ?_⟩
  -- Now we show the inequality between set integrals.
  -- Choose a simple function `ψ ≤ f` with values in `ℝ≥0` and prove for `ψ`.
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    h : MeasureTheory.IsFiniteMeasure μ
    g : Nat → α → ENNReal
    hgm : ∀ (n : Nat), Measurable (g n)
    hgf : ∀ (n : Nat), LE.le (g n) f
    hgle : ∀ (n : Nat), LE.le (g n) ↑n
    hgint : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => Min.min (f a) ↑n)  …
    φ : α → ENNReal := fun x => iSup fun n => g n x
    hφm : Measurable φ
    hφle : LE.le φ f
    s : Set α
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => f a) (MeasureTheory.l …
  -/
  rw [lintegral_eq_nnreal]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ✝ : MeasureTheory.Measure α
    f : α → ENNReal
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SFinite μ
    h : MeasureTheory.IsFiniteMeasure μ
    g : Nat → α → ENNReal
    hgm : ∀ (n : Nat), Measurable (g n)
    hgf : ∀ (n : Nat), LE.le (g n) f
    hgle : ∀ (n : Nat), LE.le (g n) ↑n
    hgint : ∀ (n : Nat), Eq (MeasureTheory.lintegral μ fun a => Min.min (f a) ↑n)  …
    φ : α → ENNReal := fun x => iSup fun n => g n x
    hφm : Measurable φ
    hφle : LE.le φ f
    s : Set α
    ⊢ LE.le (iSup fun φ => iSup fun x => (MeasureTheory.SimpleFunc.map ENNReal.ofN …
  -/
  refine iSup₂_le fun ψ hψ ↦ ?_
  -- Choose `n` such that `ψ x ≤ n` for all `x`.
  obtain ⟨n, hn⟩ : ∃ n : ℕ, ∀ x, ψ x ≤ n := by
    rcases ψ.range.bddAbove with ⟨C, hC⟩
    exact ⟨⌈C⌉₊, fun x ↦ (hC <| ψ.mem_range_self x).trans (Nat.le_ceil _)⟩
  calc
    (ψ.map (↑)).lintegral (μ.restrict s) = ∫⁻ a in s, ψ a ∂μ :=
      SimpleFunc.lintegral_eq_lintegral .. |>.symm
    _ ≤ ∫⁻ a in s, min (f a) n ∂μ :=
      lintegral_mono fun a ↦ le_min (hψ _) (ENNReal.coe_le_coe.2 (hn a))
    _ ≤ ∫⁻ a in s, g n a ∂μ := by
      have : ∫⁻ a in (toMeasurable μ s)ᶜ, min (f a) n ∂μ ≠ ∞ :=
        IsFiniteMeasure.lintegral_lt_top_of_bounded_to_ennreal _ ⟨n, fun _ ↦ min_le_right ..⟩ |>.ne
      have hsm : MeasurableSet (toMeasurable μ s) := measurableSet_toMeasurable ..
      apply ENNReal.le_of_add_le_add_right this
      rw [← μ.restrict_toMeasurable_of_sFinite, lintegral_add_compl _ hsm, hgint,
        ← lintegral_add_compl _ hsm]
      gcongr with x
      exact le_min (hgf n x) (hgle n x)
    _ ≤ _ := lintegral_mono fun x ↦ le_iSup (g · x) n


/-- In a sigma-finite measure space, there exists an integrable function which is
positive everywhere (and with an arbitrarily small integral). -/
theorem exists_pos_lintegral_lt_of_sigmaFinite (μ : Measure α) [SigmaFinite μ] {ε : ℝ≥0∞}
    (ε0 : ε ≠ 0) : ∃ g : α → ℝ≥0, (∀ x, 0 < g x) ∧ Measurable g ∧ ∫⁻ x, g x ∂μ < ε := by
  /- Let `s` be a covering of `α` by pairwise disjoint measurable sets of finite measure. Let
    `δ : ℕ → ℝ≥0` be a positive function such that `∑' i, μ (s i) * δ i < ε`. Then the function that
     is equal to `δ n` on `s n` is a positive function with integral less than `ε`. -/
  /-
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ε : ENNReal
    ε0 : Ne ε 0
    ⊢ Exists fun g => And (∀ (x : α), LT.lt 0 (g x)) (And (Measurable g) (LT.lt (M …
  -/
  set s : ℕ → Set α := disjointed (spanningSets μ)
  have : ∀ n, μ (s n) < ∞ := fun n =>
    (measure_mono <| disjointed_subset _ _).trans_lt (measure_spanningSets_lt_top μ n)
  obtain ⟨δ, δpos, δsum⟩ : ∃ δ : ℕ → ℝ≥0, (∀ i, 0 < δ i) ∧ (∑' i, μ (s i) * δ i) < ε :=
    ENNReal.exists_pos_tsum_mul_lt_of_countable ε0 _ fun n => (this n).ne
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ε : ENNReal
    ε0 : Ne ε 0
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets μ)
    this : ∀ (n : Nat), LT.lt (μ (s n)) Top.top
    δ : Nat → NNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    δsum : LT.lt (tsum fun i => HMul.hMul (μ (s i)) ↑(δ i)) ε
    ⊢ Exists fun g => And (∀ (x : α), LT.lt 0 (g x)) (And (Measurable g) (LT.lt (M …
  -/
  set N : α → ℕ := spanningSetsIndex μ
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ε : ENNReal
    ε0 : Ne ε 0
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets μ)
    this : ∀ (n : Nat), LT.lt (μ (s n)) Top.top
    δ : Nat → NNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    δsum : LT.lt (tsum fun i => HMul.hMul (μ (s i)) ↑(δ i)) ε
    N : α → Nat := MeasureTheory.spanningSetsIndex μ
    ⊢ Exists fun g => And (∀ (x : α), LT.lt 0 (g x)) (And (Measurable g) (LT.lt (M …
  -/
  have hN_meas : Measurable N := measurableSet_spanningSetsIndex μ
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ε : ENNReal
    ε0 : Ne ε 0
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets μ)
    this : ∀ (n : Nat), LT.lt (μ (s n)) Top.top
    δ : Nat → NNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    δsum : LT.lt (tsum fun i => HMul.hMul (μ (s i)) ↑(δ i)) ε
    N : α → Nat := MeasureTheory.spanningSetsIndex μ
    hN_meas : Measurable N
    ⊢ Exists fun g => And (∀ (x : α), LT.lt 0 (g x)) (And (Measurable g) (LT.lt (M …
  -/
  have hNs : ∀ n, N ⁻¹' {n} = s n := preimage_spanningSetsIndex_singleton μ
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ε : ENNReal
    ε0 : Ne ε 0
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets μ)
    this : ∀ (n : Nat), LT.lt (μ (s n)) Top.top
    δ : Nat → NNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    δsum : LT.lt (tsum fun i => HMul.hMul (μ (s i)) ↑(δ i)) ε
    N : α → Nat := MeasureTheory.spanningSetsIndex μ
    hN_meas : Measurable N
    hNs : ∀ (n : Nat), Eq (Set.preimage N (Singleton.singleton n)) (s n)
    ⊢ Exists fun g => And (∀ (x : α), LT.lt 0 (g x)) (And (Measurable g) (LT.lt (M …
  -/
  refine ⟨δ ∘ N, fun x => δpos _, measurable_from_nat.comp hN_meas, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ε : ENNReal
    ε0 : Ne ε 0
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets μ)
    this : ∀ (n : Nat), LT.lt (μ (s n)) Top.top
    δ : Nat → NNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    δsum : LT.lt (tsum fun i => HMul.hMul (μ (s i)) ↑(δ i)) ε
    N : α → Nat := MeasureTheory.spanningSetsIndex μ
    hN_meas : Measurable N
    hNs : ∀ (n : Nat), Eq (Set.preimage N (Singleton.singleton n)) (s n)
    ⊢ LT.lt (MeasureTheory.lintegral μ fun x => ↑(Function.comp δ N x)) ε
  -/
  erw [lintegral_comp measurable_from_nat.coe_nnreal_ennreal hN_meas]
  /-
    case intro.intro
    α : Type u_1
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ε : ENNReal
    ε0 : Ne ε 0
    s : Nat → Set α := disjointed (MeasureTheory.spanningSets μ)
    this : ∀ (n : Nat), LT.lt (μ (s n)) Top.top
    δ : Nat → NNReal
    δpos : ∀ (i : Nat), LT.lt 0 (δ i)
    δsum : LT.lt (tsum fun i => HMul.hMul (μ (s i)) ↑(δ i)) ε
    N : α → Nat := MeasureTheory.spanningSetsIndex μ
    hN_meas : Measurable N
    hNs : ∀ (n : Nat), Eq (Set.preimage N (Singleton.singleton n)) (s n)
    ⊢ LT.lt (MeasureTheory.lintegral (MeasureTheory.Measure.map N μ) fun a => ↑(δ  …
  -/
  simpa [N, hNs, lintegral_countable', measurableSet_spanningSetsIndex, mul_comm] using δsum
  /-
    🎉 no goals
  -/


theorem lintegral_trim {μ : Measure α} (hm : m ≤ m0) {f : α → ℝ≥0∞} (hf : Measurable[m] f) :
    ∫⁻ a, f a ∂μ.trim hm = ∫⁻ a, f a ∂μ := by
  refine
    @Measurable.ennreal_induction α m (fun f => ∫⁻ a, f a ∂μ.trim hm = ∫⁻ a, f a ∂μ) ?_ ?_ ?_ f hf
    /-
      case refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : α → ENNReal
      hf : Measurable f
      ⊢ ∀ (c : ENNReal) ⦃s : Set α⦄, MeasurableSet s → (fun f => Eq (MeasureTheory.l …
    -/
  · intro c s hs
    rw [lintegral_indicator hs, lintegral_indicator (hm s hs), setLIntegral_const,
      setLIntegral_const]
    /-
      case refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : α → ENNReal
      hf : Measurable f
      c : ENNReal
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq (HMul.hMul c ((μ.trim hm) s)) (HMul.hMul c (μ s))
    -/
    suffices h_trim_s : μ.trim hm s = μ s by rw [h_trim_s]
    /-
      case refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : α → ENNReal
      hf : Measurable f
      c : ENNReal
      s : Set α
      hs : MeasurableSet s
      ⊢ Eq ((μ.trim hm) s) (μ s)
    -/
    exact trim_measurableSet_eq hm hs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : α → ENNReal
      hf : Measurable f
      ⊢ ∀ ⦃f g : α → ENNReal⦄, Disjoint (Function.support f) (Function.support g) →  …
    -/
  · intro f g _ hf _ hf_prop hg_prop
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      f g : α → ENNReal
      a✝¹ : Disjoint (Function.support f) (Function.support g)
      hf : Measurable f
      a✝ : Measurable g
      hf_prop : Eq (MeasureTheory.lintegral (μ.trim hm) fun a => f a) (MeasureTheory …
      hg_prop : Eq (MeasureTheory.lintegral (μ.trim hm) fun a => g a) (MeasureTheory …
      ⊢ Eq (MeasureTheory.lintegral (μ.trim hm) fun a => HAdd.hAdd f g a) (MeasureTh …
    -/
    have h_m := lintegral_add_left (μ := Measure.trim μ hm) hf g
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      f g : α → ENNReal
      a✝¹ : Disjoint (Function.support f) (Function.support g)
      hf : Measurable f
      a✝ : Measurable g
      hf_prop : Eq (MeasureTheory.lintegral (μ.trim hm) fun a => f a) (MeasureTheory …
      hg_prop : Eq (MeasureTheory.lintegral (μ.trim hm) fun a => g a) (MeasureTheory …
      h_m : Eq (MeasureTheory.lintegral (μ.trim hm) fun a => HAdd.hAdd (f a) (g a))  …
      ⊢ Eq (MeasureTheory.lintegral (μ.trim hm) fun a => HAdd.hAdd f g a) (MeasureTh …
    -/
    have h_m0 := lintegral_add_left (μ := μ) (Measurable.mono hf hm le_rfl) g
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      f g : α → ENNReal
      a✝¹ : Disjoint (Function.support f) (Function.support g)
      hf : Measurable f
      a✝ : Measurable g
      hf_prop : Eq (MeasureTheory.lintegral (μ.trim hm) fun a => f a) (MeasureTheory …
      hg_prop : Eq (MeasureTheory.lintegral (μ.trim hm) fun a => g a) (MeasureTheory …
      h_m : Eq (MeasureTheory.lintegral (μ.trim hm) fun a => HAdd.hAdd (f a) (g a))  …
      h_m0 : Eq (MeasureTheory.lintegral μ fun a => HAdd.hAdd (f a) (g a)) (HAdd.hAd …
      ⊢ Eq (MeasureTheory.lintegral (μ.trim hm) fun a => HAdd.hAdd f g a) (MeasureTh …
    -/
    rwa [hf_prop, hg_prop, ← h_m0] at h_m
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : α → ENNReal
      hf : Measurable f
      ⊢ ∀ ⦃f : Nat → α → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone f → (∀ …
    -/
  · intro f hf hf_mono hf_prop
    /-
      case refine_3
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      f : Nat → α → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      hf_mono : Monotone f
      hf_prop : ∀ (n : Nat), (fun f => Eq (MeasureTheory.lintegral (μ.trim hm) fun a …
      ⊢ Eq (MeasureTheory.lintegral (μ.trim hm) fun a => (fun x => iSup fun n => f n …
    -/
    rw [lintegral_iSup hf hf_mono]
    /-
      case refine_3
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      f : Nat → α → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      hf_mono : Monotone f
      hf_prop : ∀ (n : Nat), (fun f => Eq (MeasureTheory.lintegral (μ.trim hm) fun a …
      ⊢ Eq (iSup fun n => MeasureTheory.lintegral (μ.trim hm) fun a => f n a) (Measu …
    -/
    rw [lintegral_iSup (fun n => Measurable.mono (hf n) hm le_rfl) hf_mono]
    /-
      case refine_3
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      f : Nat → α → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      hf_mono : Monotone f
      hf_prop : ∀ (n : Nat), (fun f => Eq (MeasureTheory.lintegral (μ.trim hm) fun a …
      ⊢ Eq (iSup fun n => MeasureTheory.lintegral (μ.trim hm) fun a => f n a) (iSup  …
    -/
    congr with n
    /-
      case refine_3.e_s.h
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f✝ : α → ENNReal
      hf✝ : Measurable f✝
      f : Nat → α → ENNReal
      hf : ∀ (n : Nat), Measurable (f n)
      hf_mono : Monotone f
      hf_prop : ∀ (n : Nat), (fun f => Eq (MeasureTheory.lintegral (μ.trim hm) fun a …
      n : Nat
      ⊢ Eq (MeasureTheory.lintegral (μ.trim hm) fun a => f n a) (MeasureTheory.linte …
    -/
    exact hf_prop n
    /-
      🎉 no goals
    -/


theorem lintegral_trim_ae {μ : Measure α} (hm : m ≤ m0) {f : α → ℝ≥0∞}
    (hf : AEMeasurable f (μ.trim hm)) : ∫⁻ a, f a ∂μ.trim hm = ∫⁻ a, f a ∂μ := by
  rw [lintegral_congr_ae (ae_eq_of_ae_eq_trim hf.ae_eq_mk), lintegral_congr_ae hf.ae_eq_mk,
    lintegral_trim hm hf.measurable_mk]


theorem univ_le_of_forall_fin_meas_le {μ : Measure α} (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    (C : ℝ≥0∞) {f : Set α → ℝ≥0∞} (hf : ∀ s, MeasurableSet[m] s → μ s ≠ ∞ → f s ≤ C)
    (h_F_lim :
      ∀ S : ℕ → Set α, (∀ n, MeasurableSet[m] (S n)) → Monotone S → f (⋃ n, S n) ≤ ⨆ n, f (S n)) :
    f univ ≤ C := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : Set α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (f s) C
    h_F_lim : ∀ (S : Nat → Set α), (∀ (n : Nat), MeasurableSet (S n)) → Monotone S …
    ⊢ LE.le (f Set.univ) C
  -/
  let S := @spanningSets _ m (μ.trim hm) _
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : Set α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (f s) C
    h_F_lim : ∀ (S : Nat → Set α), (∀ (n : Nat), MeasurableSet (S n)) → Monotone S …
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    ⊢ LE.le (f Set.univ) C
  -/
  have hS_mono : Monotone S := @monotone_spanningSets _ m (μ.trim hm) _
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : Set α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (f s) C
    h_F_lim : ∀ (S : Nat → Set α), (∀ (n : Nat), MeasurableSet (S n)) → Monotone S …
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    hS_mono : Monotone S
    ⊢ LE.le (f Set.univ) C
  -/
  have hS_meas : ∀ n, MeasurableSet[m] (S n) := @measurableSet_spanningSets _ m (μ.trim hm) _
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : Set α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (f s) C
    h_F_lim : ∀ (S : Nat → Set α), (∀ (n : Nat), MeasurableSet (S n)) → Monotone S …
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    hS_mono : Monotone S
    hS_meas : ∀ (n : Nat), MeasurableSet (S n)
    ⊢ LE.le (f Set.univ) C
  -/
  rw [← @iUnion_spanningSets _ m (μ.trim hm)]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : Set α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (f s) C
    h_F_lim : ∀ (S : Nat → Set α), (∀ (n : Nat), MeasurableSet (S n)) → Monotone S …
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    hS_mono : Monotone S
    hS_meas : ∀ (n : Nat), MeasurableSet (S n)
    ⊢ LE.le (f (Set.iUnion fun i => MeasureTheory.spanningSets (μ.trim hm) i)) C
  -/
  refine (h_F_lim S hS_meas hS_mono).trans ?_
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : Set α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (f s) C
    h_F_lim : ∀ (S : Nat → Set α), (∀ (n : Nat), MeasurableSet (S n)) → Monotone S …
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    hS_mono : Monotone S
    hS_meas : ∀ (n : Nat), MeasurableSet (S n)
    ⊢ LE.le (iSup fun n => f (S n)) C
  -/
  refine iSup_le fun n => hf (S n) (hS_meas n) ?_
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : Set α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (f s) C
    h_F_lim : ∀ (S : Nat → Set α), (∀ (n : Nat), MeasurableSet (S n)) → Monotone S …
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    hS_mono : Monotone S
    hS_meas : ∀ (n : Nat), MeasurableSet (S n)
    n : Nat
    ⊢ Ne (μ (S n)) Top.top
  -/
  exact ((le_trim hm).trans_lt (@measure_spanningSets_lt_top _ m (μ.trim hm) _ n)).ne
  /-
    🎉 no goals
  -/


/-- If the Lebesgue integral of a function is bounded by some constant on all sets with finite
measure in a sub-σ-algebra and the measure is σ-finite on that sub-σ-algebra, then the integral
over the whole space is bounded by that same constant. -/
theorem lintegral_le_of_forall_fin_meas_trim_le {μ : Measure α} (hm : m ≤ m0)
    [SigmaFinite (μ.trim hm)] (C : ℝ≥0∞) {f : α → ℝ≥0∞}
    (hf : ∀ s, MeasurableSet[m] s → μ s ≠ ∞ → ∫⁻ x in s, f x ∂μ ≤ C) : ∫⁻ x, f x ∂μ ≤ C := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (MeasureTheory. …
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => f x) C
  -/
  have : ∫⁻ x in univ, f x ∂μ = ∫⁻ x, f x ∂μ := by simp only [Measure.restrict_univ]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (MeasureTheory. …
    this : Eq (MeasureTheory.lintegral (μ.restrict Set.univ) fun x => f x) (Measur …
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => f x) C
  -/
  rw [← this]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (MeasureTheory. …
    this : Eq (MeasureTheory.lintegral (μ.restrict Set.univ) fun x => f x) (Measur …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict Set.univ) fun x => f x) C
  -/
  refine univ_le_of_forall_fin_meas_le hm C hf fun S _ hS_mono => ?_
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (MeasureTheory. …
    this : Eq (MeasureTheory.lintegral (μ.restrict Set.univ) fun x => f x) (Measur …
    S : Nat → Set α
    x✝ : ∀ (n : Nat), MeasurableSet (S n)
    hS_mono : Monotone S
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict (Set.iUnion fun n => S n)) fun x  …
  -/
  rw [setLIntegral_iUnion_of_directed]
  /-
    case hd
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    C : ENNReal
    f : α → ENNReal
    hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (MeasureTheory. …
    this : Eq (MeasureTheory.lintegral (μ.restrict Set.univ) fun x => f x) (Measur …
    S : Nat → Set α
    x✝ : ∀ (n : Nat), MeasurableSet (S n)
    hS_mono : Monotone S
    ⊢ Directed (fun x1 x2 => HasSubset.Subset x1 x2) S
  -/
  exact directed_of_isDirected_le hS_mono
  /-
    🎉 no goals
  -/


@[deprecated lintegral_le_of_forall_fin_meas_trim_le (since := "2024-07-14")]
alias lintegral_le_of_forall_fin_meas_le' := lintegral_le_of_forall_fin_meas_trim_le

alias lintegral_le_of_forall_fin_meas_le_of_measurable := lintegral_le_of_forall_fin_meas_trim_le


/-- If the Lebesgue integral of a function is bounded by some constant on all sets with finite
measure and the measure is σ-finite, then the integral over the whole space is bounded by that same
constant. -/
theorem lintegral_le_of_forall_fin_meas_le [MeasurableSpace α] {μ : Measure α} [SigmaFinite μ]
    (C : ℝ≥0∞) {f : α → ℝ≥0∞}
    (hf : ∀ s, MeasurableSet s → μ s ≠ ∞ → ∫⁻ x in s, f x ∂μ ≤ C) : ∫⁻ x, f x ∂μ ≤ C :=
                                           /-
                                             α : Type u_1
                                             inst✝¹ : MeasurableSpace α
                                             μ : MeasureTheory.Measure α
                                             inst✝ : MeasureTheory.SigmaFinite μ
                                             C : ENNReal
                                             f : α → ENNReal
                                             hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (MeasureTheory. …
                                             ⊢ MeasureTheory.SigmaFinite (μ.trim ⋯)
                                           -/
  have : SigmaFinite (μ.trim le_rfl) := by rwa [trim_eq_self]
                                           /-
                                             🎉 no goals
                                           -/
  lintegral_le_of_forall_fin_meas_trim_le _ C hf


theorem SimpleFunc.exists_lt_lintegral_simpleFunc_of_lt_lintegral {m : MeasurableSpace α}
    {μ : Measure α} [SigmaFinite μ] {f : α →ₛ ℝ≥0} {L : ℝ≥0∞} (hL : L < ∫⁻ x, f x ∂μ) :
    ∃ g : α →ₛ ℝ≥0, (∀ x, g x ≤ f x) ∧ ∫⁻ x, g x ∂μ < ∞ ∧ L < ∫⁻ x, g x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : MeasureTheory.SimpleFunc α NNReal
    L : ENNReal
    hL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f x))
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (LT.lt (MeasureTheor …
  -/
  induction' f using MeasureTheory.SimpleFunc.induction with c s hs f₁ f₂ _ h₁ h₂ generalizing L
  · simp only [hs, const_zero, coe_piecewise, coe_const, SimpleFunc.coe_zero, univ_inter,
      piecewise_eq_indicator, lintegral_indicator, lintegral_const, Measure.restrict_apply',
      ENNReal.coe_indicator, Function.const_apply] at hL
    have c_ne_zero : c ≠ 0 := by
      intro hc
      simp only [hc, ENNReal.coe_zero, zero_mul, not_lt_zero] at hL
    have : L / c < μ s := by
      rwa [ENNReal.div_lt_iff, mul_comm]
      · simp only [c_ne_zero, Ne, ENNReal.coe_eq_zero, not_false_iff, true_or]
      · simp only [Ne, coe_ne_top, not_false_iff, true_or]
    obtain ⟨t, ht, ts, mlt, t_top⟩ :
      ∃ t : Set α, MeasurableSet t ∧ t ⊆ s ∧ L / ↑c < μ t ∧ μ t < ∞ :=
      Measure.exists_subset_measure_lt_top hs this
    /-
      case h_ind.intro.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      c : NNReal
      s : Set α
      hs : MeasurableSet s
      L : ENNReal
      hL : LT.lt L (HMul.hMul (↑c) (μ s))
      c_ne_zero : Ne c 0
      this : LT.lt (HDiv.hDiv L ↑c) (μ s)
      t : Set α
      ht : MeasurableSet t
      ts : HasSubset.Subset t s
      mlt : LT.lt (HDiv.hDiv L ↑c) (μ t)
      t_top : LT.lt (μ t) Top.top
      ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((MeasureTheory.SimpleFunc.piece …
    -/
    refine ⟨piecewise t ht (const α c) (const α 0), fun x => ?_, ?_, ?_⟩
      /-
        case h_ind.intro.intro.intro.intro.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        c : NNReal
        s : Set α
        hs : MeasurableSet s
        L : ENNReal
        hL : LT.lt L (HMul.hMul (↑c) (μ s))
        c_ne_zero : Ne c 0
        this : LT.lt (HDiv.hDiv L ↑c) (μ s)
        t : Set α
        ht : MeasurableSet t
        ts : HasSubset.Subset t s
        mlt : LT.lt (HDiv.hDiv L ↑c) (μ t)
        t_top : LT.lt (μ t) Top.top
        x : α
        ⊢ LE.le ((MeasureTheory.SimpleFunc.piecewise t ht (MeasureTheory.SimpleFunc.co …
      -/
    · refine indicator_le_indicator_of_subset ts (fun x => ?_) x
      /-
        case h_ind.intro.intro.intro.intro.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        c : NNReal
        s : Set α
        hs : MeasurableSet s
        L : ENNReal
        hL : LT.lt L (HMul.hMul (↑c) (μ s))
        c_ne_zero : Ne c 0
        this : LT.lt (HDiv.hDiv L ↑c) (μ s)
        t : Set α
        ht : MeasurableSet t
        ts : HasSubset.Subset t s
        mlt : LT.lt (HDiv.hDiv L ↑c) (μ t)
        t_top : LT.lt (μ t) Top.top
        x✝ x : α
        ⊢ LE.le (0 x) ((MeasureTheory.SimpleFunc.const α c) x)
      -/
      exact zero_le _
      /-
        🎉 no goals
      -/
    · simp only [ht, const_zero, coe_piecewise, coe_const, SimpleFunc.coe_zero, univ_inter,
        piecewise_eq_indicator, ENNReal.coe_indicator, Function.const_apply, lintegral_indicator,
        lintegral_const, Measure.restrict_apply', ENNReal.mul_lt_top ENNReal.coe_lt_top t_top]
    · simp only [ht, const_zero, coe_piecewise, coe_const, SimpleFunc.coe_zero,
        piecewise_eq_indicator, ENNReal.coe_indicator, Function.const_apply, lintegral_indicator,
        lintegral_const, Measure.restrict_apply', univ_inter]
      /-
        case h_ind.intro.intro.intro.intro.refine_3
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        c : NNReal
        s : Set α
        hs : MeasurableSet s
        L : ENNReal
        hL : LT.lt L (HMul.hMul (↑c) (μ s))
        c_ne_zero : Ne c 0
        this : LT.lt (HDiv.hDiv L ↑c) (μ s)
        t : Set α
        ht : MeasurableSet t
        ts : HasSubset.Subset t s
        mlt : LT.lt (HDiv.hDiv L ↑c) (μ t)
        t_top : LT.lt (μ t) Top.top
        ⊢ LT.lt L (HMul.hMul (↑c) (μ t))
      -/
      rwa [mul_comm, ← ENNReal.div_lt_iff]
        /-
          case h_ind.intro.intro.intro.intro.refine_3.h0
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          inst✝ : MeasureTheory.SigmaFinite μ
          c : NNReal
          s : Set α
          hs : MeasurableSet s
          L : ENNReal
          hL : LT.lt L (HMul.hMul (↑c) (μ s))
          c_ne_zero : Ne c 0
          this : LT.lt (HDiv.hDiv L ↑c) (μ s)
          t : Set α
          ht : MeasurableSet t
          ts : HasSubset.Subset t s
          mlt : LT.lt (HDiv.hDiv L ↑c) (μ t)
          t_top : LT.lt (μ t) Top.top
          ⊢ Or (Ne (↑c) 0) (Ne L 0)
        -/
      · simp only [c_ne_zero, Ne, ENNReal.coe_eq_zero, not_false_iff, true_or]
        /-
          🎉 no goals
        -/
        /-
          case h_ind.intro.intro.intro.intro.refine_3.ht
          α : Type u_1
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          inst✝ : MeasureTheory.SigmaFinite μ
          c : NNReal
          s : Set α
          hs : MeasurableSet s
          L : ENNReal
          hL : LT.lt L (HMul.hMul (↑c) (μ s))
          c_ne_zero : Ne c 0
          this : LT.lt (HDiv.hDiv L ↑c) (μ s)
          t : Set α
          ht : MeasurableSet t
          ts : HasSubset.Subset t s
          mlt : LT.lt (HDiv.hDiv L ↑c) (μ t)
          t_top : LT.lt (μ t) Top.top
          ⊢ Or (Ne (↑c) Top.top) (Ne L Top.top)
        -/
      · simp only [Ne, coe_ne_top, not_false_iff, true_or]
        /-
          🎉 no goals
        -/
  · replace hL : L < ∫⁻ x, f₁ x ∂μ + ∫⁻ x, f₂ x ∂μ := by
      rwa [← lintegral_add_left f₁.measurable.coe_nnreal_ennreal]
    /-
      case h_add
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
      h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
      L : ENNReal
      hL : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (MeasureT …
      ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
    -/
    by_cases hf₁ : ∫⁻ x, f₁ x ∂μ = 0
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hL : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (MeasureT …
        hf₁ : Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0
        ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
      -/
    · simp only [hf₁, zero_add] at hL
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hf₁ : Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0
        hL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
      -/
      rcases h₂ hL with ⟨g, g_le, g_top, gL⟩
      /-
        case pos.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hf₁ : Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0
        hL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        g : MeasureTheory.SimpleFunc α NNReal
        g_le : ∀ (x : α), LE.le (g x) (f₂ x)
        g_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        gL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(g x))
        ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
      -/
      refine ⟨g, fun x => (g_le x).trans ?_, g_top, gL⟩
      /-
        case pos.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hf₁ : Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0
        hL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        g : MeasureTheory.SimpleFunc α NNReal
        g_le : ∀ (x : α), LE.le (g x) (f₂ x)
        g_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        gL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(g x))
        x : α
        ⊢ LE.le (f₂ x) ((HAdd.hAdd f₁ f₂) x)
      -/
      simp only [SimpleFunc.coe_add, Pi.add_apply, le_add_iff_nonneg_left, zero_le']
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
      h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
      L : ENNReal
      hL : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (MeasureT …
      hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
      ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
    -/
    by_cases hf₂ : ∫⁻ x, f₂ x ∂μ = 0
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hL : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (MeasureT …
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0
        ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
      -/
    · simp only [hf₂, add_zero] at hL
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0
        hL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
      -/
      rcases h₁ hL with ⟨g, g_le, g_top, gL⟩
      /-
        case pos.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0
        hL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        g : MeasureTheory.SimpleFunc α NNReal
        g_le : ∀ (x : α), LE.le (g x) (f₁ x)
        g_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        gL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(g x))
        ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
      -/
      refine ⟨g, fun x => (g_le x).trans ?_, g_top, gL⟩
      /-
        case pos.intro.intro.intro
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0
        hL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        g : MeasureTheory.SimpleFunc α NNReal
        g_le : ∀ (x : α), LE.le (g x) (f₁ x)
        g_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
        gL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(g x))
        x : α
        ⊢ LE.le (f₁ x) ((HAdd.hAdd f₁ f₂) x)
      -/
      simp only [SimpleFunc.coe_add, Pi.add_apply, le_add_iff_nonneg_right, zero_le']
      /-
        🎉 no goals
      -/
    obtain ⟨L₁, hL₁, L₂, hL₂, hL⟩ : ∃ L₁ < ∫⁻ x, f₁ x ∂μ, ∃ L₂ < ∫⁻ x, f₂ x ∂μ, L < L₁ + L₂ :=
      ENNReal.exists_lt_add_of_lt_add hL hf₁ hf₂
    /-
      case neg.intro.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
      h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
      L : ENNReal
      hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
      hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
      hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
      L₁ : ENNReal
      hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
      L₂ : ENNReal
      hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
      hL : LT.lt L (HAdd.hAdd L₁ L₂)
      ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
    -/
    rcases h₁ hL₁ with ⟨g₁, g₁_le, g₁_top, hg₁⟩
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
      h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
      L : ENNReal
      hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
      hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
      hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
      L₁ : ENNReal
      hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
      L₂ : ENNReal
      hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
      hL : LT.lt L (HAdd.hAdd L₁ L₂)
      g₁ : MeasureTheory.SimpleFunc α NNReal
      g₁_le : ∀ (x : α), LE.le (g₁ x) (f₁ x)
      g₁_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) Top.top
      hg₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(g₁ x))
      ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
    -/
    rcases h₂ hL₂ with ⟨g₂, g₂_le, g₂_top, hg₂⟩
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
      a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
      h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
      h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
      L : ENNReal
      hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
      hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
      hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
      L₁ : ENNReal
      hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
      L₂ : ENNReal
      hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
      hL : LT.lt L (HAdd.hAdd L₁ L₂)
      g₁ : MeasureTheory.SimpleFunc α NNReal
      g₁_le : ∀ (x : α), LE.le (g₁ x) (f₁ x)
      g₁_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) Top.top
      hg₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(g₁ x))
      g₂ : MeasureTheory.SimpleFunc α NNReal
      g₂_le : ∀ (x : α), LE.le (g₂ x) (f₂ x)
      g₂_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) Top.top
      hg₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(g₂ x))
      ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) ((HAdd.hAdd f₁ f₂) x)) (And (LT. …
    -/
    refine ⟨g₁ + g₂, fun x => add_le_add (g₁_le x) (g₂_le x), ?_, ?_⟩
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
        L₁ : ENNReal
        hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        L₂ : ENNReal
        hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        hL : LT.lt L (HAdd.hAdd L₁ L₂)
        g₁ : MeasureTheory.SimpleFunc α NNReal
        g₁_le : ∀ (x : α), LE.le (g₁ x) (f₁ x)
        g₁_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) Top.top
        hg₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(g₁ x))
        g₂ : MeasureTheory.SimpleFunc α NNReal
        g₂_le : ∀ (x : α), LE.le (g₂ x) (f₂ x)
        g₂_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) Top.top
        hg₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(g₂ x))
        ⊢ LT.lt (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd g₁ g₂) x)) Top.top
      -/
    · apply lt_of_le_of_lt _ (add_lt_top.2 ⟨g₁_top, g₂_top⟩)
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
        L₁ : ENNReal
        hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        L₂ : ENNReal
        hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        hL : LT.lt L (HAdd.hAdd L₁ L₂)
        g₁ : MeasureTheory.SimpleFunc α NNReal
        g₁_le : ∀ (x : α), LE.le (g₁ x) (f₁ x)
        g₁_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) Top.top
        hg₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(g₁ x))
        g₂ : MeasureTheory.SimpleFunc α NNReal
        g₂_le : ∀ (x : α), LE.le (g₂ x) (f₂ x)
        g₂_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) Top.top
        hg₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(g₂ x))
        ⊢ LE.le (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd g₁ g₂) x)) (HAdd.hAdd …
      -/
      rw [← lintegral_add_left g₁.measurable.coe_nnreal_ennreal]
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
        L₁ : ENNReal
        hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        L₂ : ENNReal
        hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        hL : LT.lt L (HAdd.hAdd L₁ L₂)
        g₁ : MeasureTheory.SimpleFunc α NNReal
        g₁_le : ∀ (x : α), LE.le (g₁ x) (f₁ x)
        g₁_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) Top.top
        hg₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(g₁ x))
        g₂ : MeasureTheory.SimpleFunc α NNReal
        g₂_le : ∀ (x : α), LE.le (g₂ x) (f₂ x)
        g₂_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) Top.top
        hg₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(g₂ x))
        ⊢ LE.le (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd g₁ g₂) x)) (MeasureTh …
      -/
      exact le_rfl
      /-
        🎉 no goals
      -/
      /-
        case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
        L₁ : ENNReal
        hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        L₂ : ENNReal
        hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        hL : LT.lt L (HAdd.hAdd L₁ L₂)
        g₁ : MeasureTheory.SimpleFunc α NNReal
        g₁_le : ∀ (x : α), LE.le (g₁ x) (f₁ x)
        g₁_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) Top.top
        hg₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(g₁ x))
        g₂ : MeasureTheory.SimpleFunc α NNReal
        g₂_le : ∀ (x : α), LE.le (g₂ x) (f₂ x)
        g₂_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) Top.top
        hg₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(g₂ x))
        ⊢ LT.lt L (MeasureTheory.lintegral μ fun x => ↑((HAdd.hAdd g₁ g₂) x))
      -/
    · apply hL.trans ((ENNReal.add_lt_add hg₁ hg₂).trans_le _)
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
        L₁ : ENNReal
        hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        L₂ : ENNReal
        hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        hL : LT.lt L (HAdd.hAdd L₁ L₂)
        g₁ : MeasureTheory.SimpleFunc α NNReal
        g₁_le : ∀ (x : α), LE.le (g₁ x) (f₁ x)
        g₁_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) Top.top
        hg₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(g₁ x))
        g₂ : MeasureTheory.SimpleFunc α NNReal
        g₂_le : ∀ (x : α), LE.le (g₂ x) (f₂ x)
        g₂_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) Top.top
        hg₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(g₂ x))
        ⊢ LE.le (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) (MeasureTheory …
      -/
      rw [← lintegral_add_left g₁.measurable.coe_nnreal_ennreal]
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.SigmaFinite μ
        f₁ f₂ : MeasureTheory.SimpleFunc α NNReal
        a✝ : Disjoint (Function.support ⇑f₁) (Function.support ⇑f₂)
        h₁ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) → E …
        h₂ : ∀ {L : ENNReal}, LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) → E …
        L : ENNReal
        hL✝ : LT.lt L (HAdd.hAdd (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) (Measure …
        hf₁ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₁ x)) 0)
        hf₂ : Not (Eq (MeasureTheory.lintegral μ fun x => ↑(f₂ x)) 0)
        L₁ : ENNReal
        hL₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(f₁ x))
        L₂ : ENNReal
        hL₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(f₂ x))
        hL : LT.lt L (HAdd.hAdd L₁ L₂)
        g₁ : MeasureTheory.SimpleFunc α NNReal
        g₁_le : ∀ (x : α), LE.le (g₁ x) (f₁ x)
        g₁_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₁ x)) Top.top
        hg₁ : LT.lt L₁ (MeasureTheory.lintegral μ fun x => ↑(g₁ x))
        g₂ : MeasureTheory.SimpleFunc α NNReal
        g₂_le : ∀ (x : α), LE.le (g₂ x) (f₂ x)
        g₂_top : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g₂ x)) Top.top
        hg₂ : LT.lt L₂ (MeasureTheory.lintegral μ fun x => ↑(g₂ x))
        ⊢ LE.le (MeasureTheory.lintegral μ fun a => HAdd.hAdd ↑(g₁ a) ↑(g₂ a)) (Measur …
      -/
      simp only [coe_add, Pi.add_apply, ENNReal.coe_add, le_rfl]
      /-
        🎉 no goals
      -/


theorem exists_lt_lintegral_simpleFunc_of_lt_lintegral {m : MeasurableSpace α} {μ : Measure α}
    [SigmaFinite μ] {f : α → ℝ≥0} {L : ℝ≥0∞} (hL : L < ∫⁻ x, f x ∂μ) :
    ∃ g : α →ₛ ℝ≥0, (∀ x, g x ≤ f x) ∧ ∫⁻ x, g x ∂μ < ∞ ∧ L < ∫⁻ x, g x ∂μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    L : ENNReal
    hL : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(f x))
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (LT.lt (MeasureTheor …
  -/
  simp_rw [lintegral_eq_nnreal, lt_iSup_iff] at hL
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    L : ENNReal
    hL : Exists fun i => Exists fun i_1 => LT.lt L ((MeasureTheory.SimpleFunc.map  …
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (LT.lt (MeasureTheor …
  -/
  rcases hL with ⟨g₀, hg₀, g₀L⟩
  have h'L : L < ∫⁻ x, g₀ x ∂μ := by
    convert g₀L
    rw [← SimpleFunc.lintegral_eq_lintegral, coe_map]
    simp only [Function.comp_apply]
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    L : ENNReal
    g₀ : MeasureTheory.SimpleFunc α NNReal
    hg₀ : ∀ (x : α), LE.le ↑(g₀ x) ↑(f x)
    g₀L : LT.lt L ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal g₀).lintegral μ)
    h'L : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(g₀ x))
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (LT.lt (MeasureTheor …
  -/
  rcases SimpleFunc.exists_lt_lintegral_simpleFunc_of_lt_lintegral h'L with ⟨g, hg, gL, gtop⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    f : α → NNReal
    L : ENNReal
    g₀ : MeasureTheory.SimpleFunc α NNReal
    hg₀ : ∀ (x : α), LE.le ↑(g₀ x) ↑(f x)
    g₀L : LT.lt L ((MeasureTheory.SimpleFunc.map ENNReal.ofNNReal g₀).lintegral μ)
    h'L : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(g₀ x))
    g : MeasureTheory.SimpleFunc α NNReal
    hg : ∀ (x : α), LE.le (g x) (g₀ x)
    gL : LT.lt (MeasureTheory.lintegral μ fun x => ↑(g x)) Top.top
    gtop : LT.lt L (MeasureTheory.lintegral μ fun x => ↑(g x))
    ⊢ Exists fun g => And (∀ (x : α), LE.le (g x) (f x)) (And (LT.lt (MeasureTheor …
  -/
  exact ⟨g, fun x => (hg x).trans (coe_le_coe.1 (hg₀ x)), gL, gtop⟩
  /-
    🎉 no goals
  -/


/-- If the indicators of measurable sets `Aᵢ` tend pointwise almost everywhere to the indicator
of a measurable set `A` and we eventually have `Aᵢ ⊆ B` for some set `B` of finite measure, then
the measures of `Aᵢ` tend to the measure of `A`. -/
lemma tendsto_measure_of_ae_tendsto_indicator {μ : Measure α} (A_mble : MeasurableSet A)
    (As_mble : ∀ i, MeasurableSet (As i)) {B : Set α} (B_mble : MeasurableSet B)
    (B_finmeas : μ B ≠ ∞) (As_le_B : ∀ᶠ i in L, As i ⊆ B)
    (h_lim : ∀ᵐ x ∂μ, ∀ᶠ i in L, x ∈ As i ↔ x ∈ A) :
    Tendsto (fun i ↦ μ (As i)) L (𝓝 (μ A)) := by
  simp_rw [← MeasureTheory.lintegral_indicator_one A_mble,
           ← MeasureTheory.lintegral_indicator_one (As_mble _)]
  refine tendsto_lintegral_filter_of_dominated_convergence (B.indicator (1 : α → ℝ≥0∞))
          (Eventually.of_forall ?_) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_5
      inst✝¹ : MeasurableSpace α
      A : Set α
      ι : Type u_6
      L : Filter ι
      inst✝ : L.IsCountablyGenerated
      As : ι → Set α
      μ : MeasureTheory.Measure α
      A_mble : MeasurableSet A
      As_mble : ∀ (i : ι), MeasurableSet (As i)
      B : Set α
      B_mble : MeasurableSet B
      B_finmeas : Ne (μ B) Top.top
      As_le_B : Filter.Eventually (fun i => HasSubset.Subset (As i) B) L
      h_lim : Filter.Eventually (fun x => Filter.Eventually (fun i => Iff (Membershi …
      ⊢ ∀ (x : ι), Measurable ((As x).indicator 1)
    -/
  · exact fun i ↦ Measurable.indicator measurable_const (As_mble i)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_5
      inst✝¹ : MeasurableSpace α
      A : Set α
      ι : Type u_6
      L : Filter ι
      inst✝ : L.IsCountablyGenerated
      As : ι → Set α
      μ : MeasureTheory.Measure α
      A_mble : MeasurableSet A
      As_mble : ∀ (i : ι), MeasurableSet (As i)
      B : Set α
      B_mble : MeasurableSet B
      B_finmeas : Ne (μ B) Top.top
      As_le_B : Filter.Eventually (fun i => HasSubset.Subset (As i) B) L
      h_lim : Filter.Eventually (fun x => Filter.Eventually (fun i => Iff (Membershi …
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le ((As n).indica …
    -/
  · filter_upwards [As_le_B] with i hi
    /-
      case h
      α : Type u_5
      inst✝¹ : MeasurableSpace α
      A : Set α
      ι : Type u_6
      L : Filter ι
      inst✝ : L.IsCountablyGenerated
      As : ι → Set α
      μ : MeasureTheory.Measure α
      A_mble : MeasurableSet A
      As_mble : ∀ (i : ι), MeasurableSet (As i)
      B : Set α
      B_mble : MeasurableSet B
      B_finmeas : Ne (μ B) Top.top
      As_le_B : Filter.Eventually (fun i => HasSubset.Subset (As i) B) L
      h_lim : Filter.Eventually (fun x => Filter.Eventually (fun i => Iff (Membershi …
      i : ι
      hi : HasSubset.Subset (As i) B
      ⊢ Filter.Eventually (fun a => LE.le ((As i).indicator 1 a) (B.indicator 1 a))  …
    -/
    exact Eventually.of_forall (fun x ↦ indicator_le_indicator_of_subset hi (by simp) x)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_5
      inst✝¹ : MeasurableSpace α
      A : Set α
      ι : Type u_6
      L : Filter ι
      inst✝ : L.IsCountablyGenerated
      As : ι → Set α
      μ : MeasureTheory.Measure α
      A_mble : MeasurableSet A
      As_mble : ∀ (i : ι), MeasurableSet (As i)
      B : Set α
      B_mble : MeasurableSet B
      B_finmeas : Ne (μ B) Top.top
      As_le_B : Filter.Eventually (fun i => HasSubset.Subset (As i) B) L
      h_lim : Filter.Eventually (fun x => Filter.Eventually (fun i => Iff (Membershi …
      ⊢ Ne (MeasureTheory.lintegral μ fun a => B.indicator 1 a) Top.top
    -/
  · rwa [← lintegral_indicator_one B_mble] at B_finmeas
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_5
      inst✝¹ : MeasurableSpace α
      A : Set α
      ι : Type u_6
      L : Filter ι
      inst✝ : L.IsCountablyGenerated
      As : ι → Set α
      μ : MeasureTheory.Measure α
      A_mble : MeasurableSet A
      As_mble : ∀ (i : ι), MeasurableSet (As i)
      B : Set α
      B_mble : MeasurableSet B
      B_finmeas : Ne (μ B) Top.top
      As_le_B : Filter.Eventually (fun i => HasSubset.Subset (As i) B) L
      h_lim : Filter.Eventually (fun x => Filter.Eventually (fun i => Iff (Membershi …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => (As n).indicator 1 a) L …
    -/
  · simpa only [Pi.one_def, tendsto_indicator_const_apply_iff_eventually] using h_lim
    /-
      🎉 no goals
    -/


/-- If `μ` is a finite measure and the indicators of measurable sets `Aᵢ` tend pointwise
almost everywhere to the indicator of a measurable set `A`, then the measures `μ Aᵢ` tend to
the measure `μ A`. -/
lemma tendsto_measure_of_ae_tendsto_indicator_of_isFiniteMeasure
    {μ : Measure α} [IsFiniteMeasure μ] (A_mble : MeasurableSet A)
    (As_mble : ∀ i, MeasurableSet (As i)) (h_lim : ∀ᵐ x ∂μ, ∀ᶠ i in L, x ∈ As i ↔ x ∈ A) :
    Tendsto (fun i ↦ μ (As i)) L (𝓝 (μ A)) :=
  tendsto_measure_of_ae_tendsto_indicator L A_mble As_mble MeasurableSet.univ
    (measure_ne_top μ univ) (Eventually.of_forall (fun i ↦ subset_univ (As i))) h_lim


