/-- `leastGE f r n` is the stopping time corresponding to the first time `f ≥ r`. -/
noncomputable def leastGE (f : ℕ → Ω → ℝ) (r : ℝ) (n : ℕ) :=
  hitting f (Set.Ici r) 0 n


theorem Adapted.isStoppingTime_leastGE (r : ℝ) (n : ℕ) (hf : Adapted ℱ f) :
    IsStoppingTime ℱ (leastGE f r n) :=
  hitting_isStoppingTime hf measurableSet_Ici


theorem leastGE_le {i : ℕ} {r : ℝ} (ω : Ω) : leastGE f r i ω ≤ i :=
  hitting_le ω

-- The following four lemmas shows `leastGE` behaves like a stopped process. Ideally we should
-- define `leastGE` as a stopping time and take its stopped process. However, we can't do that
-- with our current definition since a stopping time takes only finite indices. An upcoming
-- refactor should hopefully make it possible to have stopping times taking infinity as a value

theorem leastGE_mono {n m : ℕ} (hnm : n ≤ m) (r : ℝ) (ω : Ω) : leastGE f r n ω ≤ leastGE f r m ω :=
  hitting_mono hnm


theorem leastGE_eq_min (π : Ω → ℕ) (r : ℝ) (ω : Ω) {n : ℕ} (hπn : ∀ ω, π ω ≤ n) :
    leastGE f r (π ω) ω = min (π ω) (leastGE f r n ω) := by
  classical
  refine le_antisymm (le_min (leastGE_le _) (leastGE_mono (hπn ω) r ω)) ?_
  by_cases hle : π ω ≤ leastGE f r n ω
  · rw [min_eq_left hle, leastGE]
    by_cases h : ∃ j ∈ Set.Icc 0 (π ω), f j ω ∈ Set.Ici r
    · refine hle.trans (Eq.le ?_)
      rw [leastGE, ← hitting_eq_hitting_of_exists (hπn ω) h]
    · simp only [hitting, if_neg h, le_rfl]
  · rw [min_eq_right (not_le.1 hle).le, leastGE, leastGE, ←
      hitting_eq_hitting_of_exists (hπn ω) _]
    rw [not_le, leastGE, hitting_lt_iff _ (hπn ω)] at hle
    exact
      let ⟨j, hj₁, hj₂⟩ := hle
      ⟨j, ⟨hj₁.1, hj₁.2.le⟩, hj₂⟩


theorem stoppedValue_stoppedValue_leastGE (f : ℕ → Ω → ℝ) (π : Ω → ℕ) (r : ℝ) {n : ℕ}
    (hπn : ∀ ω, π ω ≤ n) : stoppedValue (fun i => stoppedValue f (leastGE f r i)) π =
      stoppedValue (stoppedProcess f (leastGE f r n)) π := by
  /-
    Ω : Type u_1
    f : Nat → Ω → Real
    π : Ω → Nat
    r : Real
    n : Nat
    hπn : ∀ (ω : Ω), LE.le (π ω) n
    ⊢ Eq (MeasureTheory.stoppedValue (fun i => MeasureTheory.stoppedValue f (Measu …
  -/
  ext1 ω
  /-
    case h
    Ω : Type u_1
    f : Nat → Ω → Real
    π : Ω → Nat
    r : Real
    n : Nat
    hπn : ∀ (ω : Ω), LE.le (π ω) n
    ω : Ω
    ⊢ Eq (MeasureTheory.stoppedValue (fun i => MeasureTheory.stoppedValue f (Measu …
  -/
  simp (config := { unfoldPartialApp := true }) only [stoppedProcess, stoppedValue]
  /-
    case h
    Ω : Type u_1
    f : Nat → Ω → Real
    π : Ω → Nat
    r : Real
    n : Nat
    hπn : ∀ (ω : Ω), LE.le (π ω) n
    ω : Ω
    ⊢ Eq (f (MeasureTheory.leastGE f r (π ω) ω) ω) (f (Min.min (π ω) (MeasureTheor …
  -/
  rw [leastGE_eq_min _ _ _ hπn]
  /-
    🎉 no goals
  -/


theorem Submartingale.stoppedValue_leastGE [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ) (r : ℝ) :
    Submartingale (fun i => stoppedValue f (leastGE f r i)) ℱ μ := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    r : Real
    ⊢ MeasureTheory.Submartingale (fun i => MeasureTheory.stoppedValue f (MeasureT …
  -/
  rw [submartingale_iff_expected_stoppedValue_mono]
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      r : Real
      ⊢ ∀ (τ π : Ω → Nat), MeasureTheory.IsStoppingTime ℱ τ → MeasureTheory.IsStoppi …
    -/
  · intro σ π hσ hπ hσ_le_π hπ_bdd
    /-
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      r : Real
      σ π : Ω → Nat
      hσ : MeasureTheory.IsStoppingTime ℱ σ
      hπ : MeasureTheory.IsStoppingTime ℱ π
      hσ_le_π : LE.le σ π
      hπ_bdd : Exists fun N => ∀ (x : Ω), LE.le (π x) N
      ⊢ LE.le (MeasureTheory.integral μ fun x => MeasureTheory.stoppedValue (fun i = …
    -/
    obtain ⟨n, hπ_le_n⟩ := hπ_bdd
    /-
      case intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      r : Real
      σ π : Ω → Nat
      hσ : MeasureTheory.IsStoppingTime ℱ σ
      hπ : MeasureTheory.IsStoppingTime ℱ π
      hσ_le_π : LE.le σ π
      n : Nat
      hπ_le_n : ∀ (x : Ω), LE.le (π x) n
      ⊢ LE.le (MeasureTheory.integral μ fun x => MeasureTheory.stoppedValue (fun i = …
    -/
    simp_rw [stoppedValue_stoppedValue_leastGE f σ r fun i => (hσ_le_π i).trans (hπ_le_n i)]
    /-
      case intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      r : Real
      σ π : Ω → Nat
      hσ : MeasureTheory.IsStoppingTime ℱ σ
      hπ : MeasureTheory.IsStoppingTime ℱ π
      hσ_le_π : LE.le σ π
      n : Nat
      hπ_le_n : ∀ (x : Ω), LE.le (π x) n
      ⊢ LE.le (MeasureTheory.integral μ fun x => MeasureTheory.stoppedValue (Measure …
    -/
    simp_rw [stoppedValue_stoppedValue_leastGE f π r hπ_le_n]
    /-
      case intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      r : Real
      σ π : Ω → Nat
      hσ : MeasureTheory.IsStoppingTime ℱ σ
      hπ : MeasureTheory.IsStoppingTime ℱ π
      hσ_le_π : LE.le σ π
      n : Nat
      hπ_le_n : ∀ (x : Ω), LE.le (π x) n
      ⊢ LE.le (MeasureTheory.integral μ fun x => MeasureTheory.stoppedValue (Measure …
    -/
    refine hf.expected_stoppedValue_mono ?_ ?_ ?_ fun ω => (min_le_left _ _).trans (hπ_le_n ω)
      /-
        case intro.refine_1
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        r : Real
        σ π : Ω → Nat
        hσ : MeasureTheory.IsStoppingTime ℱ σ
        hπ : MeasureTheory.IsStoppingTime ℱ π
        hσ_le_π : LE.le σ π
        n : Nat
        hπ_le_n : ∀ (x : Ω), LE.le (π x) n
        ⊢ MeasureTheory.IsStoppingTime ℱ fun x => Min.min (σ x) (MeasureTheory.leastGE …
      -/
    · exact hσ.min (hf.adapted.isStoppingTime_leastGE _ _)
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_2
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        r : Real
        σ π : Ω → Nat
        hσ : MeasureTheory.IsStoppingTime ℱ σ
        hπ : MeasureTheory.IsStoppingTime ℱ π
        hσ_le_π : LE.le σ π
        n : Nat
        hπ_le_n : ∀ (x : Ω), LE.le (π x) n
        ⊢ MeasureTheory.IsStoppingTime ℱ fun x => Min.min (π x) (MeasureTheory.leastGE …
      -/
    · exact hπ.min (hf.adapted.isStoppingTime_leastGE _ _)
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_3
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        r : Real
        σ π : Ω → Nat
        hσ : MeasureTheory.IsStoppingTime ℱ σ
        hπ : MeasureTheory.IsStoppingTime ℱ π
        hσ_le_π : LE.le σ π
        n : Nat
        hπ_le_n : ∀ (x : Ω), LE.le (π x) n
        ⊢ LE.le (fun x => Min.min (σ x) (MeasureTheory.leastGE f r n x)) fun x => Min. …
      -/
    · exact fun ω => min_le_min (hσ_le_π ω) le_rfl
      /-
        🎉 no goals
      -/
  · exact fun i => stronglyMeasurable_stoppedValue_of_le hf.adapted.progMeasurable_of_discrete
      (hf.adapted.isStoppingTime_leastGE _ _) leastGE_le
  · exact fun i => integrable_stoppedValue _ (hf.adapted.isStoppingTime_leastGE _ _) hf.integrable
      leastGE_le


theorem norm_stoppedValue_leastGE_le (hr : 0 ≤ r) (hf0 : f 0 = 0)
    (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) (i : ℕ) :
    ∀ᵐ ω ∂μ, stoppedValue f (leastGE f r i) ω ≤ r + R := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Nat → Ω → Real
    r : Real
    R : NNReal
    hr : LE.le 0 r
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    i : Nat
    ⊢ Filter.Eventually (fun ω => LE.le (MeasureTheory.stoppedValue f (MeasureTheo …
  -/
  filter_upwards [hbdd] with ω hbddω
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Nat → Ω → Real
    r : Real
    R : NNReal
    hr : LE.le 0 r
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    i : Nat
    ω : Ω
    hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
    ⊢ LE.le (MeasureTheory.stoppedValue f (MeasureTheory.leastGE f r i) ω) (HAdd.h …
  -/
  change f (leastGE f r i ω) ω ≤ r + R
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    f : Nat → Ω → Real
    r : Real
    R : NNReal
    hr : LE.le 0 r
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    i : Nat
    ω : Ω
    hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
    ⊢ LE.le (f (MeasureTheory.leastGE f r i ω) ω) (HAdd.hAdd r ↑R)
  -/
  by_cases heq : leastGE f r i ω = 0
    /-
      case pos
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      r : Real
      R : NNReal
      hr : LE.le 0 r
      hf0 : Eq (f 0) 0
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      i : Nat
      ω : Ω
      hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
      heq : Eq (MeasureTheory.leastGE f r i ω) 0
      ⊢ LE.le (f (MeasureTheory.leastGE f r i ω) ω) (HAdd.hAdd r ↑R)
    -/
  · rw [heq, hf0, Pi.zero_apply]
    /-
      case pos
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      r : Real
      R : NNReal
      hr : LE.le 0 r
      hf0 : Eq (f 0) 0
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      i : Nat
      ω : Ω
      hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
      heq : Eq (MeasureTheory.leastGE f r i ω) 0
      ⊢ LE.le 0 (HAdd.hAdd r ↑R)
    -/
    exact add_nonneg hr R.coe_nonneg
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      r : Real
      R : NNReal
      hr : LE.le 0 r
      hf0 : Eq (f 0) 0
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      i : Nat
      ω : Ω
      hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
      heq : Not (Eq (MeasureTheory.leastGE f r i ω) 0)
      ⊢ LE.le (f (MeasureTheory.leastGE f r i ω) ω) (HAdd.hAdd r ↑R)
    -/
  · obtain ⟨k, hk⟩ := Nat.exists_eq_succ_of_ne_zero heq
    /-
      case neg.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      r : Real
      R : NNReal
      hr : LE.le 0 r
      hf0 : Eq (f 0) 0
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      i : Nat
      ω : Ω
      hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
      heq : Not (Eq (MeasureTheory.leastGE f r i ω) 0)
      k : Nat
      hk : Eq (MeasureTheory.leastGE f r i ω) k.succ
      ⊢ LE.le (f (MeasureTheory.leastGE f r i ω) ω) (HAdd.hAdd r ↑R)
    -/
    rw [hk, add_comm, ← sub_le_iff_le_add]
    /-
      case neg.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      r : Real
      R : NNReal
      hr : LE.le 0 r
      hf0 : Eq (f 0) 0
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      i : Nat
      ω : Ω
      hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
      heq : Not (Eq (MeasureTheory.leastGE f r i ω) 0)
      k : Nat
      hk : Eq (MeasureTheory.leastGE f r i ω) k.succ
      ⊢ LE.le (HSub.hSub (f k.succ ω) r) ↑R
    -/
    have := not_mem_of_lt_hitting (hk.symm ▸ k.lt_succ_self : k < leastGE f r i ω) (zero_le _)
    /-
      case neg.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      r : Real
      R : NNReal
      hr : LE.le 0 r
      hf0 : Eq (f 0) 0
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      i : Nat
      ω : Ω
      hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
      heq : Not (Eq (MeasureTheory.leastGE f r i ω) 0)
      k : Nat
      hk : Eq (MeasureTheory.leastGE f r i ω) k.succ
      this : Not (Membership.mem (Set.Ici r) (f k ω))
      ⊢ LE.le (HSub.hSub (f k.succ ω) r) ↑R
    -/
    simp only [Set.mem_union, Set.mem_Iic, Set.mem_Ici, not_or, not_le] at this
    /-
      case neg.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      f : Nat → Ω → Real
      r : Real
      R : NNReal
      hr : LE.le 0 r
      hf0 : Eq (f 0) 0
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      i : Nat
      ω : Ω
      hbddω : ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd.hAdd i 1) ω) (f i ω))) ↑R
      heq : Not (Eq (MeasureTheory.leastGE f r i ω) 0)
      k : Nat
      hk : Eq (MeasureTheory.leastGE f r i ω) k.succ
      this : LT.lt (f k ω) r
      ⊢ LE.le (HSub.hSub (f k.succ ω) r) ↑R
    -/
    exact (sub_lt_sub_left this _).le.trans ((le_abs_self _).trans (hbddω _))
    /-
      🎉 no goals
    -/


theorem Submartingale.stoppedValue_leastGE_eLpNorm_le [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hr : 0 ≤ r) (hf0 : f 0 = 0) (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) (i : ℕ) :
    eLpNorm (stoppedValue f (leastGE f r i)) 1 μ ≤ 2 * μ Set.univ * ENNReal.ofReal (r + R) := by
  refine eLpNorm_one_le_of_le' ((hf.stoppedValue_leastGE r).integrable _) ?_
    (norm_stoppedValue_leastGE_le hr hf0 hbdd i)
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    r : Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hr : LE.le 0 r
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    i : Nat
    ⊢ LE.le 0 (MeasureTheory.integral μ fun x => MeasureTheory.stoppedValue f (Mea …
  -/
  rw [← setIntegral_univ]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    r : Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hr : LE.le 0 r
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    i : Nat
    ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict Set.univ) fun x => MeasureTheory …
  -/
  refine le_trans ?_ ((hf.stoppedValue_leastGE r).setIntegral_le (zero_le _) MeasurableSet.univ)
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    r : Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hr : LE.le 0 r
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    i : Nat
    ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict Set.univ) fun ω => MeasureTheory …
  -/
  simp_rw [stoppedValue, leastGE, hitting_of_le le_rfl, hf0, integral_zero', le_rfl]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Submartingale.stoppedValue_leastGE_snorm_le := Submartingale.stoppedValue_leastGE_eLpNorm_le


theorem Submartingale.stoppedValue_leastGE_eLpNorm_le' [IsFiniteMeasure μ]
    (hf : Submartingale f ℱ μ) (hr : 0 ≤ r) (hf0 : f 0 = 0)
    (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) (i : ℕ) :
    eLpNorm (stoppedValue f (leastGE f r i)) 1 μ ≤
      ENNReal.toNNReal (2 * μ Set.univ * ENNReal.ofReal (r + R)) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    r : Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hr : LE.le 0 r
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    i : Nat
    ⊢ LE.le (MeasureTheory.eLpNorm (MeasureTheory.stoppedValue f (MeasureTheory.le …
  -/
  refine (hf.stoppedValue_leastGE_eLpNorm_le hr hf0 hbdd i).trans ?_
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    r : Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hr : LE.le 0 r
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    i : Nat
    ⊢ LE.le (HMul.hMul (HMul.hMul 2 (μ Set.univ)) (ENNReal.ofReal (HAdd.hAdd r ↑R) …
  -/
  simp [ENNReal.coe_toNNReal (measure_ne_top μ _), ENNReal.coe_toNNReal]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias Submartingale.stoppedValue_leastGE_snorm_le' := Submartingale.stoppedValue_leastGE_eLpNorm_le'


/-- This lemma is superseded by `Submartingale.bddAbove_iff_exists_tendsto`. -/
theorem Submartingale.exists_tendsto_of_abs_bddAbove_aux [IsFiniteMeasure μ]
    (hf : Submartingale f ℱ μ) (hf0 : f 0 = 0) (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) :
    ∀ᵐ ω ∂μ, BddAbove (Set.range fun n => f n ω) → ∃ c, Tendsto (fun n => f n ω) atTop (𝓝 c) := by
  have ht :
    ∀ᵐ ω ∂μ, ∀ i : ℕ, ∃ c, Tendsto (fun n => stoppedValue f (leastGE f i n) ω) atTop (𝓝 c) := by
    rw [ae_all_iff]
    exact fun i => Submartingale.exists_ae_tendsto_of_bdd (hf.stoppedValue_leastGE i)
      (hf.stoppedValue_leastGE_eLpNorm_le' i.cast_nonneg hf0 hbdd)
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ht : Filter.Eventually (fun ω => ∀ (i : Nat), Exists fun c => Filter.Tendsto ( …
    ⊢ Filter.Eventually (fun ω => BddAbove (Set.range fun n => f n ω) → Exists fun …
  -/
  filter_upwards [ht] with ω hω hωb
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
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ht : Filter.Eventually (fun ω => ∀ (i : Nat), Exists fun c => Filter.Tendsto ( …
    ω : Ω
    hω : ∀ (i : Nat), Exists fun c => Filter.Tendsto (fun n => MeasureTheory.stopp …
    hωb : BddAbove (Set.range fun n => f n ω)
    ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
  -/
  rw [BddAbove] at hωb
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
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ht : Filter.Eventually (fun ω => ∀ (i : Nat), Exists fun c => Filter.Tendsto ( …
    ω : Ω
    hω : ∀ (i : Nat), Exists fun c => Filter.Tendsto (fun n => MeasureTheory.stopp …
    hωb : (upperBounds (Set.range fun n => f n ω)).Nonempty
    ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
  -/
  obtain ⟨i, hi⟩ := exists_nat_gt hωb.some
  have hib : ∀ n, f n ω < i := by
    intro n
    exact lt_of_le_of_lt ((mem_upperBounds.1 hωb.some_mem) _ ⟨n, rfl⟩) hi
  have heq : ∀ n, stoppedValue f (leastGE f i n) ω = f n ω := by
    intro n
    rw [leastGE]; unfold hitting; rw [stoppedValue]
    rw [if_neg]
    simp only [Set.mem_Icc, Set.mem_union, Set.mem_Ici]
    push_neg
    exact fun j _ => hib j
  /-
    case h.intro
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hf0 : Eq (f 0) 0
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ht : Filter.Eventually (fun ω => ∀ (i : Nat), Exists fun c => Filter.Tendsto ( …
    ω : Ω
    hω : ∀ (i : Nat), Exists fun c => Filter.Tendsto (fun n => MeasureTheory.stopp …
    hωb : (upperBounds (Set.range fun n => f n ω)).Nonempty
    i : Nat
    hi : LT.lt hωb.some ↑i
    hib : ∀ (n : Nat), LT.lt (f n ω) ↑i
    heq : ∀ (n : Nat), Eq (MeasureTheory.stoppedValue f (MeasureTheory.leastGE f ( …
    ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
  -/
  simp only [← heq, hω i]
  /-
    🎉 no goals
  -/


theorem Submartingale.bddAbove_iff_exists_tendsto_aux [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hf0 : f 0 = 0) (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) :
    ∀ᵐ ω ∂μ, BddAbove (Set.range fun n => f n ω) ↔ ∃ c, Tendsto (fun n => f n ω) atTop (𝓝 c) := by
  filter_upwards [hf.exists_tendsto_of_abs_bddAbove_aux hf0 hbdd] with ω hω using
    ⟨hω, fun ⟨c, hc⟩ => hc.bddAbove_range⟩


/-- One sided martingale bound: If `f` is a submartingale which has uniformly bounded differences,
then for almost every `ω`, `f n ω` is bounded above (in `n`) if and only if it converges. -/
theorem Submartingale.bddAbove_iff_exists_tendsto [IsFiniteMeasure μ] (hf : Submartingale f ℱ μ)
    (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) :
    ∀ᵐ ω ∂μ, BddAbove (Set.range fun n => f n ω) ↔ ∃ c, Tendsto (fun n => f n ω) atTop (𝓝 c) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    ⊢ Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (Exist …
  -/
  set g : ℕ → Ω → ℝ := fun n ω => f n ω - f 0 ω
  have hg : Submartingale g ℱ μ :=
    hf.sub_martingale (martingale_const_fun _ _ (hf.adapted 0) (hf.integrable 0))
  have hg0 : g 0 = 0 := by
    ext ω
    simp only [g, sub_self, Pi.zero_apply]
  have hgbdd : ∀ᵐ ω ∂μ, ∀ i : ℕ, |g (i + 1) ω - g i ω| ≤ ↑R := by
    simpa only [g, sub_sub_sub_cancel_right]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Submartingale f ℱ μ
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
    hg : MeasureTheory.Submartingale g ℱ μ
    hg0 : Eq (g 0) 0
    hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
    ⊢ Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (Exist …
  -/
  filter_upwards [hg.bddAbove_iff_exists_tendsto_aux hg0 hgbdd] with ω hω
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
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
    hg : MeasureTheory.Submartingale g ℱ μ
    hg0 : Eq (g 0) 0
    hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
    ω : Ω
    hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
    ⊢ Iff (BddAbove (Set.range fun n => f n ω)) (Exists fun c => Filter.Tendsto (f …
  -/
  convert hω using 1
    /-
      case h.e'_1.a
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
      hg : MeasureTheory.Submartingale g ℱ μ
      hg0 : Eq (g 0) 0
      hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
      ω : Ω
      hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
      ⊢ Iff (BddAbove (Set.range fun n => f n ω)) (BddAbove (Set.range fun n => g n  …
    -/
  · refine ⟨fun h => ?_, fun h => ?_⟩ <;> obtain ⟨b, hb⟩ := h <;>
    /-
      case h.e'_1.a.refine_1.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
      hg : MeasureTheory.Submartingale g ℱ μ
      hg0 : Eq (g 0) 0
      hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
      ω : Ω
      hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
      b : Real
      hb : Membership.mem (upperBounds (Set.range fun n => f n ω)) b
      ⊢ BddAbove (Set.range fun n => g n ω)
    -/
    refine ⟨b + |f 0 ω|, fun y hy => ?_⟩ <;> obtain ⟨n, rfl⟩ := hy
      /-
        case h.e'_1.a.refine_1.intro.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
        hg : MeasureTheory.Submartingale g ℱ μ
        hg0 : Eq (g 0) 0
        hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
        ω : Ω
        hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
        b : Real
        hb : Membership.mem (upperBounds (Set.range fun n => f n ω)) b
        n : Nat
        ⊢ LE.le ((fun n => g n ω) n) (HAdd.hAdd b (abs (f 0 ω)))
      -/
    · simp_rw [g, sub_eq_add_neg]
      /-
        case h.e'_1.a.refine_1.intro.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
        hg : MeasureTheory.Submartingale g ℱ μ
        hg0 : Eq (g 0) 0
        hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
        ω : Ω
        hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
        b : Real
        hb : Membership.mem (upperBounds (Set.range fun n => f n ω)) b
        n : Nat
        ⊢ LE.le (HAdd.hAdd (f n ω) (Neg.neg (f 0 ω))) (HAdd.hAdd b (abs (f 0 ω)))
      -/
      exact add_le_add (hb ⟨n, rfl⟩) (neg_le_abs _)
      /-
        🎉 no goals
      -/
      /-
        case h.e'_1.a.refine_2.intro.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
        hg : MeasureTheory.Submartingale g ℱ μ
        hg0 : Eq (g 0) 0
        hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
        ω : Ω
        hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
        b : Real
        hb : Membership.mem (upperBounds (Set.range fun n => g n ω)) b
        n : Nat
        ⊢ LE.le ((fun n => f n ω) n) (HAdd.hAdd b (abs (f 0 ω)))
      -/
    · exact sub_le_iff_le_add.1 (le_trans (sub_le_sub_left (le_abs_self _) _) (hb ⟨n, rfl⟩))
      /-
        🎉 no goals
      -/
    /-
      case h.e'_2.a
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Submartingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
      hg : MeasureTheory.Submartingale g ℱ μ
      hg0 : Eq (g 0) 0
      hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
      ω : Ω
      hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
      ⊢ Iff (Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c))  …
    -/
  · refine ⟨fun h => ?_, fun h => ?_⟩ <;> obtain ⟨c, hc⟩ := h
      /-
        case h.e'_2.a.refine_1.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
        hg : MeasureTheory.Submartingale g ℱ μ
        hg0 : Eq (g 0) 0
        hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
        ω : Ω
        hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
        c : Real
        hc : Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
        ⊢ Exists fun c => Filter.Tendsto (fun n => g n ω) Filter.atTop (nhds c)
      -/
    · exact ⟨c - f 0 ω, hc.sub_const _⟩
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2.a.refine_2.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
        hg : MeasureTheory.Submartingale g ℱ μ
        hg0 : Eq (g 0) 0
        hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
        ω : Ω
        hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
        c : Real
        hc : Filter.Tendsto (fun n => g n ω) Filter.atTop (nhds c)
        ⊢ Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds c)
      -/
    · refine ⟨c + f 0 ω, ?_⟩
      /-
        case h.e'_2.a.refine_2.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
        hg : MeasureTheory.Submartingale g ℱ μ
        hg0 : Eq (g 0) 0
        hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
        ω : Ω
        hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
        c : Real
        hc : Filter.Tendsto (fun n => g n ω) Filter.atTop (nhds c)
        ⊢ Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds (HAdd.hAdd c (f 0 ω)))
      -/
      have := hc.add_const (f 0 ω)
      /-
        case h.e'_2.a.refine_2.intro
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hf : MeasureTheory.Submartingale f ℱ μ
        hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        g : Nat → Ω → Real := fun n ω => HSub.hSub (f n ω) (f 0 ω)
        hg : MeasureTheory.Submartingale g ℱ μ
        hg0 : Eq (g 0) 0
        hgbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (g (HAd …
        ω : Ω
        hω : Iff (BddAbove (Set.range fun n => g n ω)) (Exists fun c => Filter.Tendsto …
        c : Real
        hc : Filter.Tendsto (fun n => g n ω) Filter.atTop (nhds c)
        this : Filter.Tendsto (fun k => HAdd.hAdd (g k ω) (f 0 ω)) Filter.atTop (nhds  …
        ⊢ Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds (HAdd.hAdd c (f 0 ω)))
      -/
      simpa only [g, sub_add_cancel]
      /-
        🎉 no goals
      -/


theorem Martingale.bddAbove_range_iff_bddBelow_range [IsFiniteMeasure μ] (hf : Martingale f ℱ μ)
    (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) :
    ∀ᵐ ω ∂μ, BddAbove (Set.range fun n => f n ω) ↔ BddBelow (Set.range fun n => f n ω) := by
  have hbdd' : ∀ᵐ ω ∂μ, ∀ i, |(-f) (i + 1) ω - (-f) i ω| ≤ R := by
    filter_upwards [hbdd] with ω hω i
    erw [← abs_neg, neg_sub, sub_neg_eq_add, neg_add_eq_sub]
    exact hω i
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Martingale f ℱ μ
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
    ⊢ Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (BddBe …
  -/
  have hup := hf.submartingale.bddAbove_iff_exists_tendsto hbdd
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Martingale f ℱ μ
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
    hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
    ⊢ Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (BddBe …
  -/
  have hdown := hf.neg.submartingale.bddAbove_iff_exists_tendsto hbdd'
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Martingale f ℱ μ
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
    hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
    hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
    ⊢ Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (BddBe …
  -/
  filter_upwards [hup, hdown] with ω hω₁ hω₂
  have : (∃ c, Tendsto (fun n => f n ω) atTop (𝓝 c)) ↔
      ∃ c, Tendsto (fun n => (-f) n ω) atTop (𝓝 c) := by
    constructor <;> rintro ⟨c, hc⟩
    · exact ⟨-c, hc.neg⟩
    · refine ⟨-c, ?_⟩
      convert hc.neg
      simp only [neg_neg, Pi.neg_apply]
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Martingale f ℱ μ
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
    hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
    hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
    ω : Ω
    hω₁ : Iff (BddAbove (Set.range fun n => f n ω)) (Exists fun c => Filter.Tendst …
    hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω)) (Exists fun c => Filte …
    this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds …
    ⊢ Iff (BddAbove (Set.range fun n => f n ω)) (BddBelow (Set.range fun n => f n  …
  -/
  rw [hω₁, this, ← hω₂]
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.Martingale f ℱ μ
    hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
    hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
    hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
    ω : Ω
    hω₁ : Iff (BddAbove (Set.range fun n => f n ω)) (Exists fun c => Filter.Tendst …
    hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω)) (Exists fun c => Filte …
    this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω) Filter.atTop (nhds …
    ⊢ Iff (BddAbove (Set.range fun n => Neg.neg f n ω)) (BddBelow (Set.range fun n …
  -/
  constructor <;> rintro ⟨c, hc⟩ <;> refine ⟨-c, fun ω hω => ?_⟩
    /-
      case h.mp.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Martingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
      hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
      hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
      ω✝ : Ω
      hω₁ : Iff (BddAbove (Set.range fun n => f n ω✝)) (Exists fun c => Filter.Tends …
      hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω✝)) (Exists fun c => Filt …
      this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω✝) Filter.atTop (nhd …
      c : Real
      hc : Membership.mem (upperBounds (Set.range fun n => Neg.neg f n ω✝)) c
      ω : Real
      hω : Membership.mem (Set.range fun n => f n ω✝) ω
      ⊢ LE.le (Neg.neg c) ω
    -/
  · rw [mem_upperBounds] at hc
    /-
      case h.mp.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Martingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
      hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
      hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
      ω✝ : Ω
      hω₁ : Iff (BddAbove (Set.range fun n => f n ω✝)) (Exists fun c => Filter.Tends …
      hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω✝)) (Exists fun c => Filt …
      this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω✝) Filter.atTop (nhd …
      c : Real
      hc : ∀ (x : Real), Membership.mem (Set.range fun n => Neg.neg f n ω✝) x → LE.l …
      ω : Real
      hω : Membership.mem (Set.range fun n => f n ω✝) ω
      ⊢ LE.le (Neg.neg c) ω
    -/
    refine neg_le.2 (hc _ ?_)
    /-
      case h.mp.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Martingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
      hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
      hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
      ω✝ : Ω
      hω₁ : Iff (BddAbove (Set.range fun n => f n ω✝)) (Exists fun c => Filter.Tends …
      hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω✝)) (Exists fun c => Filt …
      this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω✝) Filter.atTop (nhd …
      c : Real
      hc : ∀ (x : Real), Membership.mem (Set.range fun n => Neg.neg f n ω✝) x → LE.l …
      ω : Real
      hω : Membership.mem (Set.range fun n => f n ω✝) ω
      ⊢ Membership.mem (Set.range fun n => Neg.neg f n ω✝) (Neg.neg ω)
    -/
    simpa only [Pi.neg_apply, Set.mem_range, neg_inj]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Martingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
      hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
      hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
      ω✝ : Ω
      hω₁ : Iff (BddAbove (Set.range fun n => f n ω✝)) (Exists fun c => Filter.Tends …
      hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω✝)) (Exists fun c => Filt …
      this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω✝) Filter.atTop (nhd …
      c : Real
      hc : Membership.mem (lowerBounds (Set.range fun n => f n ω✝)) c
      ω : Real
      hω : Membership.mem (Set.range fun n => Neg.neg f n ω✝) ω
      ⊢ LE.le ω (Neg.neg c)
    -/
  · rw [mem_lowerBounds] at hc
    /-
      case h.mpr.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Martingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
      hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
      hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
      ω✝ : Ω
      hω₁ : Iff (BddAbove (Set.range fun n => f n ω✝)) (Exists fun c => Filter.Tends …
      hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω✝)) (Exists fun c => Filt …
      this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω✝) Filter.atTop (nhd …
      c : Real
      hc : ∀ (x : Real), Membership.mem (Set.range fun n => f n ω✝) x → LE.le c x
      ω : Real
      hω : Membership.mem (Set.range fun n => Neg.neg f n ω✝) ω
      ⊢ LE.le ω (Neg.neg c)
    -/
    simp_rw [Set.mem_range, Pi.neg_apply, neg_eq_iff_eq_neg] at hω
    /-
      case h.mpr.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Martingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
      hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
      hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
      ω✝ : Ω
      hω₁ : Iff (BddAbove (Set.range fun n => f n ω✝)) (Exists fun c => Filter.Tends …
      hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω✝)) (Exists fun c => Filt …
      this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω✝) Filter.atTop (nhd …
      c : Real
      hc : ∀ (x : Real), Membership.mem (Set.range fun n => f n ω✝) x → LE.le c x
      ω : Real
      hω : Exists fun y => Eq (f y ω✝) (Neg.neg ω)
      ⊢ LE.le ω (Neg.neg c)
    -/
    refine le_neg.1 (hc _ ?_)
    /-
      case h.mpr.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hf : MeasureTheory.Martingale f ℱ μ
      hbdd : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      hbdd' : Filter.Eventually (fun ω => ∀ (i : Nat), LE.le (abs (HSub.hSub (Neg.ne …
      hup : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => f n ω)) (E …
      hdown : Filter.Eventually (fun ω => Iff (BddAbove (Set.range fun n => Neg.neg  …
      ω✝ : Ω
      hω₁ : Iff (BddAbove (Set.range fun n => f n ω✝)) (Exists fun c => Filter.Tends …
      hω₂ : Iff (BddAbove (Set.range fun n => Neg.neg f n ω✝)) (Exists fun c => Filt …
      this : Iff (Exists fun c => Filter.Tendsto (fun n => f n ω✝) Filter.atTop (nhd …
      c : Real
      hc : ∀ (x : Real), Membership.mem (Set.range fun n => f n ω✝) x → LE.le c x
      ω : Real
      hω : Exists fun y => Eq (f y ω✝) (Neg.neg ω)
      ⊢ Membership.mem (Set.range fun n => f n ω✝) (Neg.neg ω)
    -/
    simpa only [Set.mem_range]
    /-
      🎉 no goals
    -/


theorem Martingale.ae_not_tendsto_atTop_atTop [IsFiniteMeasure μ] (hf : Martingale f ℱ μ)
    (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) :
    ∀ᵐ ω ∂μ, ¬Tendsto (fun n => f n ω) atTop atTop := by
  filter_upwards [hf.bddAbove_range_iff_bddBelow_range hbdd] with ω hω htop using
    unbounded_of_tendsto_atTop htop (hω.2 <| bddBelow_range_of_tendsto_atTop_atTop htop)


theorem Martingale.ae_not_tendsto_atTop_atBot [IsFiniteMeasure μ] (hf : Martingale f ℱ μ)
    (hbdd : ∀ᵐ ω ∂μ, ∀ i, |f (i + 1) ω - f i ω| ≤ R) :
    ∀ᵐ ω ∂μ, ¬Tendsto (fun n => f n ω) atTop atBot := by
  filter_upwards [hf.bddAbove_range_iff_bddBelow_range hbdd] with ω hω htop using
    unbounded_of_tendsto_atBot htop (hω.1 <| bddAbove_range_of_tendsto_atTop_atBot htop)


/-- Auxiliary definition required to prove Lévy's generalization of the Borel-Cantelli lemmas for
which we will take the martingale part. -/
noncomputable def process (s : ℕ → Set Ω) (n : ℕ) : Ω → ℝ :=
  ∑ k ∈ Finset.range n, (s (k + 1)).indicator 1


                                             /-
                                               Ω : Type u_1
                                               s : Nat → Set Ω
                                               ⊢ Eq (MeasureTheory.BorelCantelli.process s 0) 0
                                             -/
theorem process_zero : process s 0 = 0 := by rw [process, Finset.range_zero, Finset.sum_empty]
                                             /-
                                               🎉 no goals
                                             -/


theorem adapted_process (hs : ∀ n, MeasurableSet[ℱ n] (s n)) : Adapted ℱ (process s) := fun _ =>
  Finset.stronglyMeasurable_sum' _ fun _ hk =>
    stronglyMeasurable_one.indicator <| ℱ.mono (Finset.mem_range.1 hk) _ <| hs _


theorem martingalePart_process_ae_eq (ℱ : Filtration ℕ m0) (μ : Measure Ω) (s : ℕ → Set Ω) (n : ℕ) :
    martingalePart (process s) ℱ μ n =
      ∑ k ∈ Finset.range n, ((s (k + 1)).indicator 1 - μ[(s (k + 1)).indicator 1|ℱ k]) := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    ℱ : MeasureTheory.Filtration Nat m0
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    n : Nat
    ⊢ Eq (MeasureTheory.martingalePart (MeasureTheory.BorelCantelli.process s) ℱ μ …
  -/
  simp only [martingalePart_eq_sum, process_zero, zero_add]
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    ℱ : MeasureTheory.Filtration Nat m0
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    n : Nat
    ⊢ Eq ((Finset.range n).sum fun i => HSub.hSub (HSub.hSub (MeasureTheory.BorelC …
  -/
  refine Finset.sum_congr rfl fun k _ => ?_
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    ℱ : MeasureTheory.Filtration Nat m0
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    n k : Nat
    x✝ : Membership.mem (Finset.range n) k
    ⊢ Eq (HSub.hSub (HSub.hSub (MeasureTheory.BorelCantelli.process s (HAdd.hAdd k …
  -/
  simp only [process, Finset.sum_range_succ_sub_sum]
  /-
    🎉 no goals
  -/


theorem predictablePart_process_ae_eq (ℱ : Filtration ℕ m0) (μ : Measure Ω) (s : ℕ → Set Ω)
    (n : ℕ) : predictablePart (process s) ℱ μ n =
    ∑ k ∈ Finset.range n, μ[(s (k + 1)).indicator (1 : Ω → ℝ)|ℱ k] := by
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    ℱ : MeasureTheory.Filtration Nat m0
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    n : Nat
    ⊢ Eq (MeasureTheory.predictablePart (MeasureTheory.BorelCantelli.process s) ℱ  …
  -/
  have := martingalePart_process_ae_eq ℱ μ s n
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    ℱ : MeasureTheory.Filtration Nat m0
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    n : Nat
    this : Eq (MeasureTheory.martingalePart (MeasureTheory.BorelCantelli.process s …
    ⊢ Eq (MeasureTheory.predictablePart (MeasureTheory.BorelCantelli.process s) ℱ  …
  -/
  simp_rw [martingalePart, process, Finset.sum_sub_distrib] at this
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    ℱ : MeasureTheory.Filtration Nat m0
    μ : MeasureTheory.Measure Ω
    s : Nat → Set Ω
    n : Nat
    this : Eq (HSub.hSub ((Finset.range n).sum fun k => (s (HAdd.hAdd k 1)).indica …
    ⊢ Eq (MeasureTheory.predictablePart (MeasureTheory.BorelCantelli.process s) ℱ  …
  -/
  exact sub_right_injective this
  /-
    🎉 no goals
  -/


theorem process_difference_le (s : ℕ → Set Ω) (ω : Ω) (n : ℕ) :
    |process s (n + 1) ω - process s n ω| ≤ (1 : ℝ≥0) := by
  /-
    Ω : Type u_1
    s : Nat → Set Ω
    ω : Ω
    n : Nat
    ⊢ LE.le (abs (HSub.hSub (MeasureTheory.BorelCantelli.process s (HAdd.hAdd n 1) …
  -/
  norm_cast
  rw [process, process, Finset.sum_apply, Finset.sum_apply,
    Finset.sum_range_succ_sub_sum, ← Real.norm_eq_abs, norm_indicator_eq_indicator_norm]
  /-
    Ω : Type u_1
    s : Nat → Set Ω
    ω : Ω
    n : Nat
    ⊢ LE.le ((s (HAdd.hAdd n 1)).indicator (fun a => Norm.norm (1 a)) ω) 1
  -/
  refine Set.indicator_le' (fun _ _ => ?_) (fun _ _ => zero_le_one) _
  /-
    Ω : Type u_1
    s : Nat → Set Ω
    ω : Ω
    n : Nat
    x✝¹ : Ω
    x✝ : Membership.mem (s (HAdd.hAdd n 1)) x✝¹
    ⊢ LE.le (Norm.norm (1 x✝¹)) 1
  -/
  rw [Pi.one_apply, norm_one]
  /-
    🎉 no goals
  -/


theorem integrable_process (μ : Measure Ω) [IsFiniteMeasure μ] (hs : ∀ n, MeasurableSet[ℱ n] (s n))
    (n : ℕ) : Integrable (process s n) μ :=
  integrable_finset_sum' _ fun _ _ =>
    IntegrableOn.integrable_indicator (integrable_const 1) <| ℱ.le _ _ <| hs _


/-- An a.e. monotone adapted process `f` with uniformly bounded differences converges to `+∞` if
and only if its predictable part also converges to `+∞`. -/
theorem tendsto_sum_indicator_atTop_iff [IsFiniteMeasure μ]
    (hfmono : ∀ᵐ ω ∂μ, ∀ n, f n ω ≤ f (n + 1) ω) (hf : Adapted ℱ f) (hint : ∀ n, Integrable (f n) μ)
    (hbdd : ∀ᵐ ω ∂μ, ∀ n, |f (n + 1) ω - f n ω| ≤ R) :
    ∀ᵐ ω ∂μ, Tendsto (fun n => f n ω) atTop atTop ↔
      Tendsto (fun n => predictablePart f ℱ μ n ω) atTop atTop := by
  have h₁ := (martingale_martingalePart hf hint).ae_not_tendsto_atTop_atTop
    (martingalePart_bdd_difference ℱ hbdd)
  have h₂ := (martingale_martingalePart hf hint).ae_not_tendsto_atTop_atBot
    (martingalePart_bdd_difference ℱ hbdd)
  have h₃ : ∀ᵐ ω ∂μ, ∀ n, 0 ≤ (μ[f (n + 1) - f n|ℱ n]) ω := by
    refine ae_all_iff.2 fun n => condexp_nonneg ?_
    filter_upwards [ae_all_iff.1 hfmono n] with ω hω using sub_nonneg.2 hω
  /-
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
    hf : MeasureTheory.Adapted ℱ f
    hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
    h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
    h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
    ⊢ Filter.Eventually (fun ω => Iff (Filter.Tendsto (fun n => f n ω) Filter.atTo …
  -/
  filter_upwards [h₁, h₂, h₃, hfmono] with ω hω₁ hω₂ hω₃ hω₄
  /-
    case h
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    f : Nat → Ω → Real
    R : NNReal
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
    hf : MeasureTheory.Adapted ℱ f
    hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
    hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
    h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
    h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
    h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
    ω : Ω
    hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
    hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
    hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
    hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
    ⊢ Iff (Filter.Tendsto (fun n => f n ω) Filter.atTop Filter.atTop) (Filter.Tend …
  -/
  constructor <;> intro ht
    /-
      case h.mp
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
      hf : MeasureTheory.Adapted ℱ f
      hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
      ω : Ω
      hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
      hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
      ht : Filter.Tendsto (fun n => f n ω) Filter.atTop Filter.atTop
      ⊢ Filter.Tendsto (fun n => MeasureTheory.predictablePart f ℱ μ n ω) Filter.atT …
    -/
  · refine tendsto_atTop_atTop_of_monotone' ?_ ?_
      /-
        case h.mp.refine_1
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
        hf : MeasureTheory.Adapted ℱ f
        hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
        hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
        h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
        h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
        ω : Ω
        hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
        hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
        hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
        hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
        ht : Filter.Tendsto (fun n => f n ω) Filter.atTop Filter.atTop
        ⊢ Monotone fun n => MeasureTheory.predictablePart f ℱ μ n ω
      -/
    · intro n m hnm
      /-
        case h.mp.refine_1
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
        hf : MeasureTheory.Adapted ℱ f
        hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
        hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
        h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
        h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
        ω : Ω
        hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
        hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
        hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
        hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
        ht : Filter.Tendsto (fun n => f n ω) Filter.atTop Filter.atTop
        n m : Nat
        hnm : LE.le n m
        ⊢ LE.le ((fun n => MeasureTheory.predictablePart f ℱ μ n ω) n) ((fun n => Meas …
      -/
      simp only [predictablePart, Finset.sum_apply]
      /-
        case h.mp.refine_1
        Ω : Type u_1
        m0 : MeasurableSpace Ω
        μ : MeasureTheory.Measure Ω
        ℱ : MeasureTheory.Filtration Nat m0
        f : Nat → Ω → Real
        R : NNReal
        inst✝ : MeasureTheory.IsFiniteMeasure μ
        hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
        hf : MeasureTheory.Adapted ℱ f
        hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
        hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
        h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
        h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
        h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
        ω : Ω
        hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
        hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
        hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
        hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
        ht : Filter.Tendsto (fun n => f n ω) Filter.atTop Filter.atTop
        n m : Nat
        hnm : LE.le n m
        ⊢ LE.le ((Finset.range n).sum fun c => MeasureTheory.condexp (↑ℱ c) μ (HSub.hS …
      -/
      exact Finset.sum_mono_set_of_nonneg hω₃ (Finset.range_mono hnm)
      /-
        🎉 no goals
      -/
    /-
      case h.mp.refine_2
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
      hf : MeasureTheory.Adapted ℱ f
      hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
      ω : Ω
      hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
      hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
      ht : Filter.Tendsto (fun n => f n ω) Filter.atTop Filter.atTop
      ⊢ Not (BddAbove (Set.range fun n => MeasureTheory.predictablePart f ℱ μ n ω))
    -/
    rintro ⟨b, hbdd⟩
    /-
      case h.mp.refine_2.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
      hf : MeasureTheory.Adapted ℱ f
      hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hbdd✝ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAd …
      h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
      ω : Ω
      hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
      hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
      ht : Filter.Tendsto (fun n => f n ω) Filter.atTop Filter.atTop
      b : Real
      hbdd : Membership.mem (upperBounds (Set.range fun n => MeasureTheory.predictab …
      ⊢ False
    -/
    rw [← tendsto_neg_atBot_iff] at ht
    /-
      case h.mp.refine_2.intro
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
      hf : MeasureTheory.Adapted ℱ f
      hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hbdd✝ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAd …
      h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
      ω : Ω
      hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
      hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
      ht : Filter.Tendsto (fun x => Neg.neg (f x ω)) Filter.atTop Filter.atBot
      b : Real
      hbdd : Membership.mem (upperBounds (Set.range fun n => MeasureTheory.predictab …
      ⊢ False
    -/
    simp only [martingalePart, sub_eq_add_neg] at hω₁
    exact hω₁ (tendsto_atTop_add_right_of_le _ (-b) (tendsto_neg_atBot_iff.1 ht) fun n =>
      neg_le_neg (hbdd ⟨n, rfl⟩))
    /-
      case h.mpr
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
      hf : MeasureTheory.Adapted ℱ f
      hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
      ω : Ω
      hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
      hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
      ht : Filter.Tendsto (fun n => MeasureTheory.predictablePart f ℱ μ n ω) Filter. …
      ⊢ Filter.Tendsto (fun n => f n ω) Filter.atTop Filter.atTop
    -/
  · refine tendsto_atTop_atTop_of_monotone' (monotone_nat_of_le_succ hω₄) ?_
    /-
      case h.mpr
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      f : Nat → Ω → Real
      R : NNReal
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      hfmono : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd  …
      hf : MeasureTheory.Adapted ℱ f
      hint : ∀ (n : Nat), MeasureTheory.Integrable (f n) μ
      hbdd : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le (abs (HSub.hSub (f (HAdd …
      h₁ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₂ : Filter.Eventually (fun ω => Not (Filter.Tendsto (fun n => MeasureTheory.m …
      h₃ : Filter.Eventually (fun ω => ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp ( …
      ω : Ω
      hω₁ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₂ : Not (Filter.Tendsto (fun n => MeasureTheory.martingalePart f ℱ μ n ω) Fi …
      hω₃ : ∀ (n : Nat), LE.le 0 (MeasureTheory.condexp (↑ℱ n) μ (HSub.hSub (f (HAdd …
      hω₄ : ∀ (n : Nat), LE.le (f n ω) (f (HAdd.hAdd n 1) ω)
      ht : Filter.Tendsto (fun n => MeasureTheory.predictablePart f ℱ μ n ω) Filter. …
      ⊢ Not (BddAbove (Set.range fun n => f n ω))
    -/
    rintro ⟨b, hbdd⟩
    exact hω₂ ((tendsto_atBot_add_left_of_ge _ b fun n =>
      hbdd ⟨n, rfl⟩) <| tendsto_neg_atBot_iff.2 ht)


theorem tendsto_sum_indicator_atTop_iff' [IsFiniteMeasure μ] {s : ℕ → Set Ω}
    (hs : ∀ n, MeasurableSet[ℱ n] (s n)) : ∀ᵐ ω ∂μ,
    Tendsto (fun n => ∑ k ∈ Finset.range n,
      (s (k + 1)).indicator (1 : Ω → ℝ) ω) atTop atTop ↔
    Tendsto (fun n => ∑ k ∈ Finset.range n,
      (μ[(s (k + 1)).indicator (1 : Ω → ℝ)|ℱ k]) ω) atTop atTop := by
  have := tendsto_sum_indicator_atTop_iff (Eventually.of_forall fun ω n => ?_) (adapted_process hs)
    (integrable_process μ hs) (Eventually.of_forall <| process_difference_le s)
  /-
    case refine_2
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Nat → Set Ω
    hs : ∀ (n : Nat), MeasurableSet (s n)
    this : Filter.Eventually (fun ω => Iff (Filter.Tendsto (fun n => MeasureTheory …
    ⊢ Filter.Eventually (fun ω => Iff (Filter.Tendsto (fun n => (Finset.range n).s …
  -/
  swap
  · rw [process, process, ← sub_nonneg, Finset.sum_apply, Finset.sum_apply,
      Finset.sum_range_succ_sub_sum]
    /-
      case refine_1
      Ω : Type u_1
      m0 : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      ℱ : MeasureTheory.Filtration Nat m0
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      s : Nat → Set Ω
      hs : ∀ (n : Nat), MeasurableSet (s n)
      ω : Ω
      n : Nat
      ⊢ LE.le 0 ((s (HAdd.hAdd n 1)).indicator 1 ω)
    -/
    exact Set.indicator_nonneg (fun _ _ => zero_le_one) _
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Nat → Set Ω
    hs : ∀ (n : Nat), MeasurableSet (s n)
    this : Filter.Eventually (fun ω => Iff (Filter.Tendsto (fun n => MeasureTheory …
    ⊢ Filter.Eventually (fun ω => Iff (Filter.Tendsto (fun n => (Finset.range n).s …
  -/
  simp_rw [process, predictablePart_process_ae_eq] at this
  /-
    case refine_2
    Ω : Type u_1
    m0 : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    ℱ : MeasureTheory.Filtration Nat m0
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    s : Nat → Set Ω
    hs : ∀ (n : Nat), MeasurableSet (s n)
    this : Filter.Eventually (fun ω => Iff (Filter.Tendsto (fun n => (Finset.range …
    ⊢ Filter.Eventually (fun ω => Iff (Filter.Tendsto (fun n => (Finset.range n).s …
  -/
  simpa using this
  /-
    🎉 no goals
  -/


/-- **Lévy's generalization of the Borel-Cantelli lemma**: given a sequence of sets `s` and a
filtration `ℱ` such that for all `n`, `s n` is `ℱ n`-measurable, `limsup s atTop` is almost
everywhere equal to the set for which `∑ k, ℙ(s (k + 1) | ℱ k) = ∞`. -/
theorem ae_mem_limsup_atTop_iff (μ : Measure Ω) [IsFiniteMeasure μ] {s : ℕ → Set Ω}
    (hs : ∀ n, MeasurableSet[ℱ n] (s n)) : ∀ᵐ ω ∂μ, ω ∈ limsup s atTop ↔
    Tendsto (fun n => ∑ k ∈ Finset.range n,
      (μ[(s (k + 1)).indicator (1 : Ω → ℝ)|ℱ k]) ω) atTop atTop :=
  (limsup_eq_tendsto_sum_indicator_atTop ℝ s).symm ▸ tendsto_sum_indicator_atTop_iff' hs


