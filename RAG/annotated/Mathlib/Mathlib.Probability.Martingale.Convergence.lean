/-- If a stochastic process has bounded upcrossing from below `a` to above `b`,
then it does not frequently visit both below `a` and above `b`. -/
theorem not_frequently_of_upcrossings_lt_top (hab : a < b) (hω : upcrossings a b f ω ≠ ∞) :
    ¬((∃ᶠ n in atTop, f n ω < a) ∧ ∃ᶠ n in atTop, b < f n ω) := by
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    hab : LT.lt a b
    hω : Ne (MeasureTheory.upcrossings a b f ω) Top.top
    ⊢ Not (And (Filter.Frequently (fun n => LT.lt (f n ω) a) Filter.atTop) (Filter …
  -/
  rw [← lt_top_iff_ne_top, upcrossings_lt_top_iff] at hω
  replace hω : ∃ k, ∀ N, upcrossingsBefore a b f N ω < k := by
    obtain ⟨k, hk⟩ := hω
    exact ⟨k + 1, fun N => lt_of_le_of_lt (hk N) k.lt_succ_self⟩
  /-
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    hab : LT.lt a b
    hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
    ⊢ Not (And (Filter.Frequently (fun n => LT.lt (f n ω) a) Filter.atTop) (Filter …
  -/
  rintro ⟨h₁, h₂⟩
  /-
    case intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    hab : LT.lt a b
    hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
    h₁ : Filter.Frequently (fun n => LT.lt (f n ω) a) Filter.atTop
    h₂ : Filter.Frequently (fun n => LT.lt b (f n ω)) Filter.atTop
    ⊢ False
  -/
  rw [frequently_atTop] at h₁ h₂
  /-
    case intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    hab : LT.lt a b
    hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
    h₁ : ∀ (a_1 : Nat), Exists fun b => And (GE.ge b a_1) (LT.lt (f b ω) a)
    h₂ : ∀ (a : Nat), Exists fun b_1 => And (GE.ge b_1 a) (LT.lt b (f b_1 ω))
    ⊢ False
  -/
  refine Classical.not_not.2 hω ?_
  /-
    case intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    hab : LT.lt a b
    hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
    h₁ : ∀ (a_1 : Nat), Exists fun b => And (GE.ge b a_1) (LT.lt (f b ω) a)
    h₂ : ∀ (a : Nat), Exists fun b_1 => And (GE.ge b_1 a) (LT.lt b (f b_1 ω))
    ⊢ Not (Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b …
  -/
  push_neg
  /-
    case intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    hab : LT.lt a b
    hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
    h₁ : ∀ (a_1 : Nat), Exists fun b => And (GE.ge b a_1) (LT.lt (f b ω) a)
    h₂ : ∀ (a : Nat), Exists fun b_1 => And (GE.ge b_1 a) (LT.lt b (f b_1 ω))
    ⊢ ∀ (k : Nat), Exists fun N => LE.le k (MeasureTheory.upcrossingsBefore a b f  …
  -/
  intro k
  /-
    case intro
    Ω : Type u_1
    a b : Real
    f : Nat → Ω → Real
    ω : Ω
    hab : LT.lt a b
    hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
    h₁ : ∀ (a_1 : Nat), Exists fun b => And (GE.ge b a_1) (LT.lt (f b ω) a)
    h₂ : ∀ (a : Nat), Exists fun b_1 => And (GE.ge b_1 a) (LT.lt b (f b_1 ω))
    k : Nat
    ⊢ Exists fun N => LE.le k (MeasureTheory.upcrossingsBefore a b f N ω)
  -/
  induction' k with k ih
    /-
      case intro.zero
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
      h₁ : ∀ (a_1 : Nat), Exists fun b => And (GE.ge b a_1) (LT.lt (f b ω) a)
      h₂ : ∀ (a : Nat), Exists fun b_1 => And (GE.ge b_1 a) (LT.lt b (f b_1 ω))
      ⊢ Exists fun N => LE.le 0 (MeasureTheory.upcrossingsBefore a b f N ω)
    -/
  · simp only [zero_le, exists_const]
    /-
      🎉 no goals
    -/
    /-
      case intro.succ
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
      h₁ : ∀ (a_1 : Nat), Exists fun b => And (GE.ge b a_1) (LT.lt (f b ω) a)
      h₂ : ∀ (a : Nat), Exists fun b_1 => And (GE.ge b_1 a) (LT.lt b (f b_1 ω))
      k : Nat
      ih : Exists fun N => LE.le k (MeasureTheory.upcrossingsBefore a b f N ω)
      ⊢ Exists fun N => LE.le (HAdd.hAdd k 1) (MeasureTheory.upcrossingsBefore a b f …
    -/
  · obtain ⟨N, hN⟩ := ih
    /-
      case intro.succ.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
      h₁ : ∀ (a_1 : Nat), Exists fun b => And (GE.ge b a_1) (LT.lt (f b ω) a)
      h₂ : ∀ (a : Nat), Exists fun b_1 => And (GE.ge b_1 a) (LT.lt b (f b_1 ω))
      k N : Nat
      hN : LE.le k (MeasureTheory.upcrossingsBefore a b f N ω)
      ⊢ Exists fun N => LE.le (HAdd.hAdd k 1) (MeasureTheory.upcrossingsBefore a b f …
    -/
    obtain ⟨N₁, hN₁, hN₁'⟩ := h₁ N
    /-
      case intro.succ.intro.intro.intro
      Ω : Type u_1
      a b : Real
      f : Nat → Ω → Real
      ω : Ω
      hab : LT.lt a b
      hω : Exists fun k => ∀ (N : Nat), LT.lt (MeasureTheory.upcrossingsBefore a b f …
      h₁ : ∀ (a_1 : Nat), Exists fun b => And (GE.ge b a_1) (LT.lt (f b ω) a)
      h₂ : ∀ (a : Nat), Exists fun b_1 => And (GE.ge b_1 a) (LT.lt b (f b_1 ω))
      k N : Nat
      hN : LE.le k (MeasureTheory.upcrossingsBefore a b f N ω)
      N₁ : Nat
      hN₁ : GE.ge N₁ N
      hN₁' : LT.lt (f N₁ ω) a
      ⊢ Exists fun N => LE.le (HAdd.hAdd k 1) (MeasureTheory.upcrossingsBefore a b f …
    -/
    obtain ⟨N₂, hN₂, hN₂'⟩ := h₂ N₁
    exact ⟨N₂ + 1, Nat.succ_le_of_lt <|
      lt_of_le_of_lt hN (upcrossingsBefore_lt_of_exists_upcrossing hab hN₁ hN₁' hN₂ hN₂')⟩


/-- A stochastic process that frequently visits below `a` and above `b` has infinite upcrossings. -/
theorem upcrossings_eq_top_of_frequently_lt (hab : a < b) (h₁ : ∃ᶠ n in atTop, f n ω < a)
    (h₂ : ∃ᶠ n in atTop, b < f n ω) : upcrossings a b f ω = ∞ :=
  by_contradiction fun h => not_frequently_of_upcrossings_lt_top hab h ⟨h₁, h₂⟩


/-- A realization of a stochastic process with bounded upcrossings and bounded liminfs is
convergent.

We use the spelling `< ∞` instead of the standard `≠ ∞` in the assumptions since it is not as easy
to change `<` to `≠` under binders. -/
theorem tendsto_of_uncrossing_lt_top (hf₁ : liminf (fun n => (‖f n ω‖₊ : ℝ≥0∞)) atTop < ∞)
    (hf₂ : ∀ a b : ℚ, a < b → upcrossings a b f ω < ∞) :
    ∃ c, Tendsto (fun n => f n ω) atTop (𝓝 c) := by
  /-
    Ω : Type u_1
    f : Nat → Ω → Real
    ω : Ω
    hf₁ : LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n ω))) Filter.atTop) T …
    hf₂ : ∀ (a b : Rat), LT.lt a b → LT.lt (MeasureTheory.upcrossings (↑a) (↑b) f  …
    ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
  -/
  by_cases h : IsBoundedUnder (· ≤ ·) atTop fun n => |f n ω|
    /-
      case pos
      Ω : Type u_1
      f : Nat → Ω → Real
      ω : Ω
      hf₁ : LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n ω))) Filter.atTop) T …
      hf₂ : ∀ (a b : Rat), LT.lt a b → LT.lt (MeasureTheory.upcrossings (↑a) (↑b) f  …
      h : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n => abs …
      ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
    -/
  · rw [isBoundedUnder_le_abs] at h
    /-
      case pos
      Ω : Type u_1
      f : Nat → Ω → Real
      ω : Ω
      hf₁ : LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n ω))) Filter.atTop) T …
      hf₂ : ∀ (a b : Rat), LT.lt a b → LT.lt (MeasureTheory.upcrossings (↑a) (↑b) f  …
      h : And (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n = …
      ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
    -/
    refine tendsto_of_no_upcrossings Rat.denseRange_cast ?_ h.1 h.2
    /-
      case pos
      Ω : Type u_1
      f : Nat → Ω → Real
      ω : Ω
      hf₁ : LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n ω))) Filter.atTop) T …
      hf₂ : ∀ (a b : Rat), LT.lt a b → LT.lt (MeasureTheory.upcrossings (↑a) (↑b) f  …
      h : And (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n = …
      ⊢ ∀ (a : Real), Membership.mem (Set.range Rat.cast) a → ∀ (b : Real), Membersh …
    -/
    rintro _ ⟨a, rfl⟩ _ ⟨b, rfl⟩ hab
    /-
      case pos.intro.intro
      Ω : Type u_1
      f : Nat → Ω → Real
      ω : Ω
      hf₁ : LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n ω))) Filter.atTop) T …
      hf₂ : ∀ (a b : Rat), LT.lt a b → LT.lt (MeasureTheory.upcrossings (↑a) (↑b) f  …
      h : And (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n = …
      a b : Rat
      hab : LT.lt ↑a ↑b
      ⊢ Not (And (Filter.Frequently (fun n => LT.lt (f n ω) ↑a) Filter.atTop) (Filte …
    -/
    exact not_frequently_of_upcrossings_lt_top hab (hf₂ a b (Rat.cast_lt.1 hab)).ne
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      f : Nat → Ω → Real
      ω : Ω
      hf₁ : LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n ω))) Filter.atTop) T …
      hf₂ : ∀ (a b : Rat), LT.lt a b → LT.lt (MeasureTheory.upcrossings (↑a) (↑b) f  …
      h : Not (Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) Filter.atTop fun n = …
      ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
    -/
  · obtain ⟨a, b, hab, h₁, h₂⟩ := ENNReal.exists_upcrossings_of_not_bounded_under hf₁.ne h
    exact
      False.elim ((hf₂ a b hab).ne (upcrossings_eq_top_of_frequently_lt (Rat.cast_lt.2 hab) h₁ h₂))


/-- An L¹-bounded submartingale has bounded upcrossings almost everywhere. -/
theorem Submartingale.upcrossings_ae_lt_top' [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hbdd : ∀ n, eLpNorm (f n) 1 μ ≤ R) (hab : a < b) : ∀ᵐ ω ∂μ, upcrossings a b f ω < ∞ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    a b : Real
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    hab : LT.lt a b
    ⊢ Filter.Eventually (fun ω => LT.lt (MeasureTheory.upcrossings a b f ω) Top.to …
  -/
  refine ae_lt_top (hf.adapted.measurable_upcrossings hab) ?_
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    a b : Real
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    hab : LT.lt a b
    ⊢ Ne (MeasureTheory.lintegral μ fun x => MeasureTheory.upcrossings a b f x) To …
  -/
  have := hf.mul_lintegral_upcrossings_le_lintegral_pos_part a b
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    a b : Real
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    hab : LT.lt a b
    this : LE.le (HMul.hMul (ENNReal.ofReal (HSub.hSub b a)) (MeasureTheory.linteg …
    ⊢ Ne (MeasureTheory.lintegral μ fun x => MeasureTheory.upcrossings a b f x) To …
  -/
  rw [mul_comm, ← ENNReal.le_div_iff_mul_le] at this
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      a b : Real
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
      hab : LT.lt a b
      this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
      ⊢ Ne (MeasureTheory.lintegral μ fun x => MeasureTheory.upcrossings a b f x) To …
    -/
  · refine (lt_of_le_of_lt this (ENNReal.div_lt_top ?_ ?_)).ne
    · have hR' : ∀ n, ∫⁻ ω, ‖f n ω - a‖₊ ∂μ ≤ R + ‖a‖₊ * μ Set.univ := by
        simp_rw [eLpNorm_one_eq_lintegral_nnnorm] at hbdd
        intro n
        refine (lintegral_mono ?_ : ∫⁻ ω, ‖f n ω - a‖₊ ∂μ ≤ ∫⁻ ω, ‖f n ω‖₊ + ‖a‖₊ ∂μ).trans ?_
        · intro ω
          simp_rw [sub_eq_add_neg, ← nnnorm_neg a, ← ENNReal.coe_add, ENNReal.coe_le_coe]
          exact nnnorm_add_le _ _
        · simp_rw [lintegral_add_right _ measurable_const, lintegral_const]
          exact add_le_add (hbdd _) le_rfl
      refine ne_of_lt (iSup_lt_iff.2 ⟨R + ‖a‖₊ * μ Set.univ, ENNReal.add_lt_top.2
        ⟨ENNReal.coe_lt_top, ENNReal.mul_lt_top ENNReal.coe_lt_top (measure_lt_top _ _)⟩,
        fun n => le_trans ?_ (hR' n)⟩)
      /-
        case refine_1
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        a b : Real
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
        hab : LT.lt a b
        this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
        hR' : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm ( …
        n : Nat
        ⊢ LE.le (MeasureTheory.lintegral μ fun ω => ENNReal.ofReal (PosPart.posPart (H …
      -/
      refine lintegral_mono fun ω => ?_
      /-
        case refine_1
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        a b : Real
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
        hab : LT.lt a b
        this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
        hR' : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm ( …
        n : Nat
        ω : Ω
        ⊢ LE.le (ENNReal.ofReal (PosPart.posPart (HSub.hSub (f n ω) a))) ↑(NNNorm.nnno …
      -/
      rw [ENNReal.ofReal_le_iff_le_toReal, ENNReal.coe_toReal, coe_nnnorm]
        /-
          case refine_1
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          ℱ : MeasureTheory.Filtration Nat m0
          a b : Real
          f : Nat → Ω → Real
          R : NNReal
          inst✝ : MeasureTheory.IsFiniteMeasure μ
          hf : MeasureTheory.Submartingale f ℱ μ
          hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
          hab : LT.lt a b
          this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
          hR' : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm ( …
          n : Nat
          ω : Ω
          ⊢ LE.le (PosPart.posPart (HSub.hSub (f n ω) a)) (Norm.norm (HSub.hSub (f n ω)  …
        -/
      · by_cases hnonneg : 0 ≤ f n ω - a
          /-
            case pos
            Ω : Type u_1
            m0 : MeasurableSpace Ω
            μ : MeasureTheory.Measure Ω
            ℱ : MeasureTheory.Filtration Nat m0
            a b : Real
            f : Nat → Ω → Real
            R : NNReal
            inst✝ : MeasureTheory.IsFiniteMeasure μ
            hf : MeasureTheory.Submartingale f ℱ μ
            hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
            hab : LT.lt a b
            this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
            hR' : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm ( …
            n : Nat
            ω : Ω
            hnonneg : LE.le 0 (HSub.hSub (f n ω) a)
            ⊢ LE.le (PosPart.posPart (HSub.hSub (f n ω) a)) (Norm.norm (HSub.hSub (f n ω)  …
          -/
        · rw [posPart_eq_self.2 hnonneg, Real.norm_eq_abs, abs_of_nonneg hnonneg]
          /-
            🎉 no goals
          -/
          /-
            case neg
            Ω : Type u_1
            m0 : MeasurableSpace Ω
            μ : MeasureTheory.Measure Ω
            ℱ : MeasureTheory.Filtration Nat m0
            a b : Real
            f : Nat → Ω → Real
            R : NNReal
            inst✝ : MeasureTheory.IsFiniteMeasure μ
            hf : MeasureTheory.Submartingale f ℱ μ
            hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
            hab : LT.lt a b
            this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
            hR' : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm ( …
            n : Nat
            ω : Ω
            hnonneg : Not (LE.le 0 (HSub.hSub (f n ω) a))
            ⊢ LE.le (PosPart.posPart (HSub.hSub (f n ω) a)) (Norm.norm (HSub.hSub (f n ω)  …
          -/
        · rw [posPart_eq_zero.2 (not_le.1 hnonneg).le]
          /-
            case neg
            Ω : Type u_1
            m0 : MeasurableSpace Ω
            μ : MeasureTheory.Measure Ω
            ℱ : MeasureTheory.Filtration Nat m0
            a b : Real
            f : Nat → Ω → Real
            R : NNReal
            inst✝ : MeasureTheory.IsFiniteMeasure μ
            hf : MeasureTheory.Submartingale f ℱ μ
            hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
            hab : LT.lt a b
            this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
            hR' : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm ( …
            n : Nat
            ω : Ω
            hnonneg : Not (LE.le 0 (HSub.hSub (f n ω) a))
            ⊢ LE.le 0 (Norm.norm (HSub.hSub (f n ω) a))
          -/
          exact norm_nonneg _
          /-
            🎉 no goals
          -/
        /-
          case refine_1
          Ω : Type u_1
          m0 : MeasurableSpace Ω
          μ : MeasureTheory.Measure Ω
          ℱ : MeasureTheory.Filtration Nat m0
          a b : Real
          f : Nat → Ω → Real
          R : NNReal
          inst✝ : MeasureTheory.IsFiniteMeasure μ
          hf : MeasureTheory.Submartingale f ℱ μ
          hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
          hab : LT.lt a b
          this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
          hR' : ∀ (n : Nat), LE.le (MeasureTheory.lintegral μ fun ω => ↑(NNNorm.nnnorm ( …
          n : Nat
          ω : Ω
          ⊢ Ne (↑(NNNorm.nnnorm (HSub.hSub (f n ω) a))) Top.top
        -/
      · simp only [Ne, ENNReal.coe_ne_top, not_false_iff]
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        a b : Real
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
        hab : LT.lt a b
        this : LE.le (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcrossings a b …
        ⊢ Ne (ENNReal.ofReal (HSub.hSub b a)) 0
      -/
    · simp only [hab, Ne, ENNReal.ofReal_eq_zero, sub_nonpos, not_le]
      /-
        🎉 no goals
      -/
    /-
      case h0
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      a b : Real
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
      hab : LT.lt a b
      this : LE.le (HMul.hMul (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcr …
      ⊢ Or (Ne (ENNReal.ofReal (HSub.hSub b a)) 0) (Ne (iSup fun N => MeasureTheory. …
    -/
  · simp only [hab, Ne, ENNReal.ofReal_eq_zero, sub_nonpos, not_le, true_or]
    /-
      🎉 no goals
    -/
    /-
      case ht
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      a b : Real
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
      hab : LT.lt a b
      this : LE.le (HMul.hMul (MeasureTheory.lintegral μ fun ω => MeasureTheory.upcr …
      ⊢ Or (Ne (ENNReal.ofReal (HSub.hSub b a)) Top.top) (Ne (iSup fun N => MeasureT …
    -/
  · simp only [Ne, ENNReal.ofReal_ne_top, not_false_iff, true_or]
    /-
      🎉 no goals
    -/


theorem Submartingale.upcrossings_ae_lt_top [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hbdd : ∀ n, eLpNorm (f n) 1 μ ≤ R) : ∀ᵐ ω ∂μ, ∀ a b : ℚ, a < b → upcrossings a b f ω < ∞ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    ⊢ Filter.Eventually (fun ω => ∀ (a b : Rat), LT.lt a b → LT.lt (MeasureTheory. …
  -/
  simp only [ae_all_iff, eventually_imp_distrib_left]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    ⊢ ∀ (i i_1 : Rat), LT.lt i i_1 → Filter.Eventually (fun x => LT.lt (MeasureThe …
  -/
  rintro a b hab
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    a b : Rat
    hab : LT.lt a b
    ⊢ Filter.Eventually (fun x => LT.lt (MeasureTheory.upcrossings (↑a) (↑b) f x)  …
  -/
  exact hf.upcrossings_ae_lt_top' hbdd (Rat.cast_lt.2 hab)
  /-
    🎉 no goals
  -/


/-- An L¹-bounded submartingale converges almost everywhere. -/
theorem Submartingale.exists_ae_tendsto_of_bdd [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hbdd : ∀ n, eLpNorm (f n) 1 μ ≤ R) : ∀ᵐ ω ∂μ, ∃ c, Tendsto (fun n => f n ω) atTop (𝓝 c) := by
  filter_upwards [hf.upcrossings_ae_lt_top hbdd, ae_bdd_liminf_atTop_of_eLpNorm_bdd one_ne_zero
    (fun n => (hf.stronglyMeasurable n).measurable.mono (ℱ.le n) le_rfl) hbdd] with ω h₁ h₂
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    ω : Ω
    h₁ : ∀ (a b : Rat), LT.lt a b → LT.lt (MeasureTheory.upcrossings (↑a) (↑b) f ω …
    h₂ : LT.lt (Filter.liminf (fun n => ↑(NNNorm.nnnorm (f n ω))) Filter.atTop) To …
    ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
  -/
  exact tendsto_of_uncrossing_lt_top h₂ h₁
  /-
    🎉 no goals
  -/


theorem Submartingale.exists_ae_trim_tendsto_of_bdd [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hbdd : ∀ n, eLpNorm (f n) 1 μ ≤ R) :
    ∀ᵐ ω ∂μ.trim (sSup_le fun _ ⟨_, hn⟩ => hn ▸ ℱ.le _ : ⨆ n, ℱ n ≤ m0),
      ∃ c, Tendsto (fun n => f n ω) atTop (𝓝 c) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    ⊢ Filter.Eventually (fun ω => Exists fun c => Filter.Tendsto (fun n => f n ω)  …
  -/
  letI := (⨆ n, ℱ n)
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
    this : MeasurableSpace Ω := iSup fun n => ↑ℱ n
    ⊢ Filter.Eventually (fun ω => Exists fun c => Filter.Tendsto (fun n => f n ω)  …
  -/
  rw [ae_iff, trim_measurableSet_eq]
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      hbdd : ∀ (n : Nat), LE.le (MeasureTheory.eLpNorm (f n) 1 μ) ↑R
      this : MeasurableSpace Ω := iSup fun n => ↑ℱ n
      ⊢ Eq (μ (setOf fun a => Not (Exists fun c => Filter.Tendsto (fun n => f n a) F …
    -/
  · exact hf.exists_ae_tendsto_of_bdd hbdd
    /-
      🎉 no goals
    -/
  · exact MeasurableSet.compl <| measurableSet_exists_tendsto
      fun n => (hf.stronglyMeasurable n).measurable.mono (le_sSup ⟨n, rfl⟩) le_rfl


/-- **Almost everywhere martingale convergence theorem**: An L¹-bounded submartingale converges
almost everywhere to a `⨆ n, ℱ n`-measurable function. -/
theorem Submartingale.ae_tendsto_limitProcess [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hbdd : ∀ n, eLpNorm (f n) 1 μ ≤ R) :
    ∀ᵐ ω ∂μ, Tendsto (fun n => f n ω) atTop (𝓝 (ℱ.limitProcess f μ ω)) := by
  classical
  suffices
      ∃ g, StronglyMeasurable[⨆ n, ℱ n] g ∧ ∀ᵐ ω ∂μ, Tendsto (fun n => f n ω) atTop (𝓝 (g ω)) by
    rw [limitProcess, dif_pos this]
    exact (Classical.choose_spec this).2
  set g' : Ω → ℝ := fun ω => if h : ∃ c, Tendsto (fun n => f n ω) atTop (𝓝 c) then h.choose else 0
  have hle : ⨆ n, ℱ n ≤ m0 := sSup_le fun m ⟨n, hn⟩ => hn ▸ ℱ.le _
  have hg' : ∀ᵐ ω ∂μ.trim hle, Tendsto (fun n => f n ω) atTop (𝓝 (g' ω)) := by
    filter_upwards [hf.exists_ae_trim_tendsto_of_bdd hbdd] with ω hω
    simp_rw [g', dif_pos hω]
    exact hω.choose_spec
  have hg'm : @AEStronglyMeasurable _ _ _ (⨆ n, ℱ n) g' (μ.trim hle) :=
    (@aemeasurable_of_tendsto_metrizable_ae' _ _ (⨆ n, ℱ n) _ _ _ _ _ _ _
      (fun n => ((hf.stronglyMeasurable n).measurable.mono (le_sSup ⟨n, rfl⟩ : ℱ n ≤ ⨆ n, ℱ n)
        le_rfl).aemeasurable) hg').aestronglyMeasurable
  obtain ⟨g, hgm, hae⟩ := hg'm
  have hg : ∀ᵐ ω ∂μ.trim hle, Tendsto (fun n => f n ω) atTop (𝓝 (g ω)) := by
    filter_upwards [hae, hg'] with ω hω hg'ω
    exact hω ▸ hg'ω
  exact ⟨g, hgm, measure_eq_zero_of_trim_eq_zero hle hg⟩


/-- The limiting process of an Lᵖ-bounded submartingale is Lᵖ. -/
theorem Submartingale.memℒp_limitProcess {p : ℝ≥0∞} (hf : Submartingale f ℱ μ)
    (hbdd : ∀ n, eLpNorm (f n) p μ ≤ R) : Memℒp (ℱ.limitProcess f μ) p μ :=
  memℒp_limitProcess_of_eLpNorm_bdd
    (fun n => ((hf.stronglyMeasurable n).mono (ℱ.le n)).aestronglyMeasurable) hbdd


/-- Part a of the **L¹ martingale convergence theorem**: a uniformly integrable submartingale
adapted to the filtration `ℱ` converges a.e. and in L¹ to an integrable function which is
measurable with respect to the σ-algebra `⨆ n, ℱ n`. -/
theorem Submartingale.tendsto_eLpNorm_one_limitProcess (hf : Submartingale f ℱ μ)
    (hunif : UniformIntegrable f 1 μ) :
    Tendsto (fun n => eLpNorm (f n - ℱ.limitProcess f μ) 1 μ) atTop (𝓝 0) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hunif : MeasureTheory.UniformIntegrable f 1 μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) (MeasureTheo …
  -/
  obtain ⟨R, hR⟩ := hunif.2.2
  have hmeas : ∀ n, AEStronglyMeasurable (f n) μ := fun n =>
    ((hf.stronglyMeasurable n).mono (ℱ.le _)).aestronglyMeasurable
  exact tendsto_Lp_finite_of_tendstoInMeasure le_rfl ENNReal.one_ne_top hmeas
    (memℒp_limitProcess_of_eLpNorm_bdd hmeas hR) hunif.2.1
    (tendstoInMeasure_of_tendsto_ae hmeas <| hf.ae_tendsto_limitProcess hR)


@[deprecated (since := "2024-07-27")]
alias Submartingale.tendsto_snorm_one_limitProcess := Submartingale.tendsto_eLpNorm_one_limitProcess


theorem Submartingale.ae_tendsto_limitProcess_of_uniformIntegrable (hf : Submartingale f ℱ μ)
    (hunif : UniformIntegrable f 1 μ) :
    ∀ᵐ ω ∂μ, Tendsto (fun n => f n ω) atTop (𝓝 (ℱ.limitProcess f μ ω)) :=
  let ⟨_, hR⟩ := hunif.2.2
  hf.ae_tendsto_limitProcess hR


/-- If a martingale `f` adapted to `ℱ` converges in L¹ to `g`, then for all `n`, `f n` is almost
everywhere equal to `𝔼[g | ℱ n]`. -/
theorem Martingale.eq_condexp_of_tendsto_eLpNorm {μ : Measure Ω} (hf : Martingale f ℱ μ)
    (hg : Integrable g μ) (hgtends : Tendsto (fun n => eLpNorm (f n - g) 1 μ) atTop (𝓝 0)) (n : ℕ) :
    f n =ᵐ[μ] μ[g|ℱ n] := by
  rw [← sub_ae_eq_zero, ← eLpNorm_eq_zero_iff (((hf.stronglyMeasurable n).mono (ℱ.le _)).sub
    (stronglyMeasurable_condexp.mono (ℱ.le _))).aestronglyMeasurable one_ne_zero]
  have ht : Tendsto (fun m => eLpNorm (μ[f m - g|ℱ n]) 1 μ) atTop (𝓝 0) :=
    haveI hint : ∀ m, Integrable (f m - g) μ := fun m => (hf.integrable m).sub hg
    tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds hgtends (fun m => zero_le _)
      fun m => eLpNorm_one_condexp_le_eLpNorm _
  have hev : ∀ m ≥ n, eLpNorm (μ[f m - g|ℱ n]) 1 μ = eLpNorm (f n - μ[g|ℱ n]) 1 μ := by
    refine fun m hm => eLpNorm_congr_ae ((condexp_sub (hf.integrable m) hg).trans ?_)
    filter_upwards [hf.2 n m hm] with x hx
    simp only [hx, Pi.sub_apply]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    g : Ω → Real
    μ : MeasureTheory.Measure Ω
    hf : MeasureTheory.Martingale f ℱ μ
    hg : MeasureTheory.Integrable g μ
    hgtends : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (f n) g) 1 …
    n : Nat
    ht : Filter.Tendsto (fun m => MeasureTheory.eLpNorm (MeasureTheory.condexp (↑ℱ …
    hev : ∀ (m : Nat), GE.ge m n → Eq (MeasureTheory.eLpNorm (MeasureTheory.condex …
    ⊢ Eq (MeasureTheory.eLpNorm (HSub.hSub (f n) (MeasureTheory.condexp (↑ℱ n) μ g …
  -/
  exact tendsto_nhds_unique (tendsto_atTop_of_eventually_const hev) ht
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Martingale.eq_condexp_of_tendsto_snorm := Martingale.eq_condexp_of_tendsto_eLpNorm


/-- Part b of the **L¹ martingale convergence theorem**: if `f` is a uniformly integrable martingale
adapted to the filtration `ℱ`, then for all `n`, `f n` is almost everywhere equal to the conditional
expectation of its limiting process wrt. `ℱ n`. -/
theorem Martingale.ae_eq_condexp_limitProcess (hf : Martingale f ℱ μ)
    (hbdd : UniformIntegrable f 1 μ) (n : ℕ) : f n =ᵐ[μ] μ[ℱ.limitProcess f μ|ℱ n] :=
  let ⟨_, hR⟩ := hbdd.2.2
  hf.eq_condexp_of_tendsto_eLpNorm ((memℒp_limitProcess_of_eLpNorm_bdd hbdd.1 hR).integrable le_rfl)
    (hf.submartingale.tendsto_eLpNorm_one_limitProcess hbdd) n


/-- Part c of the **L¹ martingale convergence theorem**: Given an integrable function `g` which
is measurable with respect to `⨆ n, ℱ n` where `ℱ` is a filtration, the martingale defined by
`𝔼[g | ℱ n]` converges almost everywhere to `g`.

This martingale also converges to `g` in L¹ and this result is provided by
`MeasureTheory.Integrable.tendsto_eLpNorm_condexp` -/
theorem Integrable.tendsto_ae_condexp (hg : Integrable g μ)
    (hgmeas : StronglyMeasurable[⨆ n, ℱ n] g) :
    ∀ᵐ x ∂μ, Tendsto (fun n => (μ[g|ℱ n]) x) atTop (𝓝 (g x)) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : Ω → Real
    hg : MeasureTheory.Integrable g μ
    hgmeas : MeasureTheory.StronglyMeasurable g
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.condexp ( …
  -/
  have hle : ⨆ n, ℱ n ≤ m0 := sSup_le fun m ⟨n, hn⟩ => hn ▸ ℱ.le _
  have hunif : UniformIntegrable (fun n => μ[g|ℱ n]) 1 μ :=
    hg.uniformIntegrable_condexp_filtration
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : Ω → Real
    hg : MeasureTheory.Integrable g μ
    hgmeas : MeasureTheory.StronglyMeasurable g
    hle : LE.le (iSup fun n => ↑ℱ n) m0
    hunif : MeasureTheory.UniformIntegrable (fun n => MeasureTheory.condexp (↑ℱ n) …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.condexp ( …
  -/
  obtain ⟨R, hR⟩ := hunif.2.2
  have hlimint : Integrable (ℱ.limitProcess (fun n => μ[g|ℱ n]) μ) μ :=
    (memℒp_limitProcess_of_eLpNorm_bdd hunif.1 hR).integrable le_rfl
  suffices g =ᵐ[μ] ℱ.limitProcess (fun n x => (μ[g|ℱ n]) x) μ by
    filter_upwards [this, (martingale_condexp g ℱ μ).submartingale.ae_tendsto_limitProcess hR] with
      x heq ht
    rwa [heq]
  have : ∀ n s, MeasurableSet[ℱ n] s →
      ∫ x in s, g x ∂μ = ∫ x in s, ℱ.limitProcess (fun n x => (μ[g|ℱ n]) x) μ x ∂μ := by
    intro n s hs
    rw [← setIntegral_condexp (ℱ.le n) hg hs, ← setIntegral_condexp (ℱ.le n) hlimint hs]
    refine setIntegral_congr_ae (ℱ.le _ _ hs) ?_
    filter_upwards [(martingale_condexp g ℱ μ).ae_eq_condexp_limitProcess hunif n] with x hx _
    rw [hx]
  refine ae_eq_of_forall_setIntegral_eq_of_sigmaFinite' hle (fun s _ _ => hg.integrableOn)
    (fun s _ _ => hlimint.integrableOn) (fun s hs _ => ?_) hgmeas.aeStronglyMeasurable'
    stronglyMeasurable_limitProcess.aeStronglyMeasurable'
  have hpi : IsPiSystem {s | ∃ n, MeasurableSet[ℱ n] s} := by
    rw [Set.setOf_exists]
    exact isPiSystem_iUnion_of_monotone _ (fun n ↦ (ℱ n).isPiSystem_measurableSet) fun _ _ ↦ ℱ.mono
  induction s, hs
    using MeasurableSpace.induction_on_inter (MeasurableSpace.measurableSpace_iSup_eq ℱ) hpi with
  | empty =>
    simp only [measure_empty, Measure.restrict_empty, integral_zero_measure]
  | basic s hs =>
    rcases hs with ⟨n, hn⟩
    exact this n _ hn
  | compl t htmeas ht =>
    have hgeq := @setIntegral_compl _ _ (⨆ n, ℱ n) _ _ _ _ _ htmeas (hg.trim hle hgmeas)
    have hheq := @setIntegral_compl _ _ (⨆ n, ℱ n) _ _ _ _ _ htmeas
      (hlimint.trim hle stronglyMeasurable_limitProcess)
    rw [setIntegral_trim hle hgmeas htmeas.compl,
      setIntegral_trim hle stronglyMeasurable_limitProcess htmeas.compl, hgeq, hheq, ←
      setIntegral_trim hle hgmeas htmeas, ←
      setIntegral_trim hle stronglyMeasurable_limitProcess htmeas, ← integral_trim hle hgmeas, ←
      integral_trim hle stronglyMeasurable_limitProcess, ← setIntegral_univ,
      this 0 _ MeasurableSet.univ, setIntegral_univ, ht (measure_lt_top _ _)]
  | iUnion f hf hfmeas heq =>
    rw [integral_iUnion (fun n => hle _ (hfmeas n)) hf hg.integrableOn,
      integral_iUnion (fun n => hle _ (hfmeas n)) hf hlimint.integrableOn]
    exact tsum_congr fun n => heq _ (measure_lt_top _ _)


/-- Part c of the **L¹ martingale convergence theorem**: Given an integrable function `g` which
is measurable with respect to `⨆ n, ℱ n` where `ℱ` is a filtration, the martingale defined by
`𝔼[g | ℱ n]` converges in L¹ to `g`.

This martingale also converges to `g` almost everywhere and this result is provided by
`MeasureTheory.Integrable.tendsto_ae_condexp` -/
theorem Integrable.tendsto_eLpNorm_condexp (hg : Integrable g μ)
    (hgmeas : StronglyMeasurable[⨆ n, ℱ n] g) :
    Tendsto (fun n => eLpNorm (μ[g|ℱ n] - g) 1 μ) atTop (𝓝 0) :=
  tendsto_Lp_finite_of_tendstoInMeasure le_rfl ENNReal.one_ne_top
    (fun n => (stronglyMeasurable_condexp.mono (ℱ.le n)).aestronglyMeasurable)
    (memℒp_one_iff_integrable.2 hg) hg.uniformIntegrable_condexp_filtration.2.1
    (tendstoInMeasure_of_tendsto_ae
      (fun n => (stronglyMeasurable_condexp.mono (ℱ.le n)).aestronglyMeasurable)
      (hg.tendsto_ae_condexp hgmeas))


@[deprecated (since := "2024-07-27")]
alias Integrable.tendsto_snorm_condexp := Integrable.tendsto_eLpNorm_condexp


/-- **Lévy's upward theorem**, almost everywhere version: given a function `g` and a filtration
`ℱ`, the sequence defined by `𝔼[g | ℱ n]` converges almost everywhere to `𝔼[g | ⨆ n, ℱ n]`. -/
theorem tendsto_ae_condexp (g : Ω → ℝ) :
    ∀ᵐ x ∂μ, Tendsto (fun n => (μ[g|ℱ n]) x) atTop (𝓝 ((μ[g|⨆ n, ℱ n]) x)) := by
  have ht : ∀ᵐ x ∂μ, Tendsto (fun n => (μ[μ[g|⨆ n, ℱ n]|ℱ n]) x) atTop (𝓝 ((μ[g|⨆ n, ℱ n]) x)) :=
    integrable_condexp.tendsto_ae_condexp stronglyMeasurable_condexp
  have heq : ∀ n, ∀ᵐ x ∂μ, (μ[μ[g|⨆ n, ℱ n]|ℱ n]) x = (μ[g|ℱ n]) x := fun n =>
    condexp_condexp_of_le (le_iSup _ n) (iSup_le fun n => ℱ.le n)
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : Ω → Real
    ht : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.condex …
    heq : ∀ (n : Nat), Filter.Eventually (fun x => Eq (MeasureTheory.condexp (↑ℱ n …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.condexp ( …
  -/
  rw [← ae_all_iff] at heq
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : Ω → Real
    ht : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.condex …
    heq : Filter.Eventually (fun a => ∀ (i : Nat), Eq (MeasureTheory.condexp (↑ℱ i …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.condexp ( …
  -/
  filter_upwards [heq, ht] with x hxeq hxt
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : Ω → Real
    ht : Filter.Eventually (fun x => Filter.Tendsto (fun n => MeasureTheory.condex …
    heq : Filter.Eventually (fun a => ∀ (i : Nat), Eq (MeasureTheory.condexp (↑ℱ i …
    x : Ω
    hxeq : ∀ (i : Nat), Eq (MeasureTheory.condexp (↑ℱ i) μ (MeasureTheory.condexp  …
    hxt : Filter.Tendsto (fun n => MeasureTheory.condexp (↑ℱ n) μ (MeasureTheory.c …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.condexp (↑ℱ n) μ g x) Filter.atTop (n …
  -/
  exact hxt.congr hxeq
  /-
    🎉 no goals
  -/


/-- **Lévy's upward theorem**, L¹ version: given a function `g` and a filtration `ℱ`, the
sequence defined by `𝔼[g | ℱ n]` converges in L¹ to `𝔼[g | ⨆ n, ℱ n]`. -/
theorem tendsto_eLpNorm_condexp (g : Ω → ℝ) :
    Tendsto (fun n => eLpNorm (μ[g|ℱ n] - μ[g|⨆ n, ℱ n]) 1 μ) atTop (𝓝 0) := by
  have ht : Tendsto (fun n => eLpNorm (μ[μ[g|⨆ n, ℱ n]|ℱ n] - μ[g|⨆ n, ℱ n]) 1 μ) atTop (𝓝 0) :=
    integrable_condexp.tendsto_eLpNorm_condexp stronglyMeasurable_condexp
  have heq : ∀ n, ∀ᵐ x ∂μ, (μ[μ[g|⨆ n, ℱ n]|ℱ n]) x = (μ[g|ℱ n]) x := fun n =>
    condexp_condexp_of_le (le_iSup _ n) (iSup_le fun n => ℱ.le n)
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : Ω → Real
    ht : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (MeasureTheory. …
    heq : ∀ (n : Nat), Filter.Eventually (fun x => Eq (MeasureTheory.condexp (↑ℱ n …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (MeasureTheory.con …
  -/
  refine ht.congr fun n => eLpNorm_congr_ae ?_
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : Ω → Real
    ht : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (MeasureTheory. …
    heq : ∀ (n : Nat), Filter.Eventually (fun x => Eq (MeasureTheory.condexp (↑ℱ n …
    n : Nat
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSub.hSub (MeasureTheory.condexp (↑ℱ n) μ …
  -/
  filter_upwards [heq n] with x hxeq
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    g : Ω → Real
    ht : Filter.Tendsto (fun n => MeasureTheory.eLpNorm (HSub.hSub (MeasureTheory. …
    heq : ∀ (n : Nat), Filter.Eventually (fun x => Eq (MeasureTheory.condexp (↑ℱ n …
    n : Nat
    x : Ω
    hxeq : Eq (MeasureTheory.condexp (↑ℱ n) μ (MeasureTheory.condexp (iSup fun n = …
    ⊢ Eq (HSub.hSub (MeasureTheory.condexp (↑ℱ n) μ (MeasureTheory.condexp (iSup f …
  -/
  simp only [hxeq, Pi.sub_apply]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias tendsto_snorm_condexp := tendsto_eLpNorm_condexp


