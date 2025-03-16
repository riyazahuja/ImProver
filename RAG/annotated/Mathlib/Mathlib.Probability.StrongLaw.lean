/-- Truncating a real-valued function to the interval `(-A, A]`. -/
def truncation (f : α → ℝ) (A : ℝ) :=
  indicator (Set.Ioc (-A) A) id ∘ f


theorem _root_.MeasureTheory.AEStronglyMeasurable.truncation (hf : AEStronglyMeasurable f μ)
    {A : ℝ} : AEStronglyMeasurable (truncation f A) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    ⊢ MeasureTheory.AEStronglyMeasurable (ProbabilityTheory.truncation f A) μ
  -/
  apply AEStronglyMeasurable.comp_aemeasurable _ hf.aemeasurable
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    ⊢ MeasureTheory.AEStronglyMeasurable ((Set.Ioc (Neg.neg A) A).indicator id) (M …
  -/
  exact (stronglyMeasurable_id.indicator measurableSet_Ioc).aestronglyMeasurable
  /-
    🎉 no goals
  -/


theorem abs_truncation_le_bound (f : α → ℝ) (A : ℝ) (x : α) : |truncation f A x| ≤ |A| := by
  /-
    α : Type u_1
    f : α → Real
    A : Real
    x : α
    ⊢ LE.le (abs (ProbabilityTheory.truncation f A x)) (abs A)
  -/
  simp only [truncation, Set.indicator, Set.mem_Icc, id, Function.comp_apply]
  /-
    α : Type u_1
    f : α → Real
    A : Real
    x : α
    ⊢ LE.le (abs (ite (Membership.mem (Set.Ioc (Neg.neg A) A) (f x)) (f x) 0)) (ab …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      f : α → Real
      A : Real
      x : α
      h : Membership.mem (Set.Ioc (Neg.neg A) A) (f x)
      ⊢ LE.le (abs (f x)) (abs A)
    -/
  · exact abs_le_abs h.2 (neg_le.2 h.1.le)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      f : α → Real
      A : Real
      x : α
      h : Not (Membership.mem (Set.Ioc (Neg.neg A) A) (f x))
      ⊢ LE.le (abs 0) (abs A)
    -/
  · simp [abs_nonneg]
    /-
      🎉 no goals
    -/


@[simp]
                                                               /-
                                                                 α : Type u_1
                                                                 f : α → Real
                                                                 ⊢ Eq (ProbabilityTheory.truncation f 0) 0
                                                               -/
theorem truncation_zero (f : α → ℝ) : truncation f 0 = 0 := by simp [truncation]; rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem abs_truncation_le_abs_self (f : α → ℝ) (A : ℝ) (x : α) : |truncation f A x| ≤ |f x| := by
  /-
    α : Type u_1
    f : α → Real
    A : Real
    x : α
    ⊢ LE.le (abs (ProbabilityTheory.truncation f A x)) (abs (f x))
  -/
  simp only [truncation, indicator, Set.mem_Icc, id, Function.comp_apply]
  /-
    α : Type u_1
    f : α → Real
    A : Real
    x : α
    ⊢ LE.le (abs (ite (Membership.mem (Set.Ioc (Neg.neg A) A) (f x)) (f x) 0)) (ab …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      f : α → Real
      A : Real
      x : α
      h✝ : Membership.mem (Set.Ioc (Neg.neg A) A) (f x)
      ⊢ LE.le (abs (f x)) (abs (f x))
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      f : α → Real
      A : Real
      x : α
      h✝ : Not (Membership.mem (Set.Ioc (Neg.neg A) A) (f x))
      ⊢ LE.le (abs 0) (abs (f x))
    -/
  · simp [abs_nonneg]
    /-
      🎉 no goals
    -/


theorem truncation_eq_self {f : α → ℝ} {A : ℝ} {x : α} (h : |f x| < A) :
    truncation f A x = f x := by
  /-
    α : Type u_1
    f : α → Real
    A : Real
    x : α
    h : LT.lt (abs (f x)) A
    ⊢ Eq (ProbabilityTheory.truncation f A x) (f x)
  -/
  simp only [truncation, indicator, Set.mem_Icc, id, Function.comp_apply, ite_eq_left_iff]
  /-
    α : Type u_1
    f : α → Real
    A : Real
    x : α
    h : LT.lt (abs (f x)) A
    ⊢ Not (Membership.mem (Set.Ioc (Neg.neg A) A) (f x)) → Eq 0 (f x)
  -/
  intro H
  /-
    α : Type u_1
    f : α → Real
    A : Real
    x : α
    h : LT.lt (abs (f x)) A
    H : Not (Membership.mem (Set.Ioc (Neg.neg A) A) (f x))
    ⊢ Eq 0 (f x)
  -/
  apply H.elim
  /-
    α : Type u_1
    f : α → Real
    A : Real
    x : α
    h : LT.lt (abs (f x)) A
    H : Not (Membership.mem (Set.Ioc (Neg.neg A) A) (f x))
    ⊢ Membership.mem (Set.Ioc (Neg.neg A) A) (f x)
  -/
  simp [(abs_lt.1 h).1, (abs_lt.1 h).2.le]
  /-
    🎉 no goals
  -/


theorem truncation_eq_of_nonneg {f : α → ℝ} {A : ℝ} (h : ∀ x, 0 ≤ f x) :
    truncation f A = indicator (Set.Ioc 0 A) id ∘ f := by
  /-
    α : Type u_1
    f : α → Real
    A : Real
    h : ∀ (x : α), LE.le 0 (f x)
    ⊢ Eq (ProbabilityTheory.truncation f A) (Function.comp ((Set.Ioc 0 A).indicato …
  -/
  ext x
  /-
    case h
    α : Type u_1
    f : α → Real
    A : Real
    h : ∀ (x : α), LE.le 0 (f x)
    x : α
    ⊢ Eq (ProbabilityTheory.truncation f A x) (Function.comp ((Set.Ioc 0 A).indica …
  -/
  rcases (h x).lt_or_eq with (hx | hx)
    /-
      case h.inl
      α : Type u_1
      f : α → Real
      A : Real
      h : ∀ (x : α), LE.le 0 (f x)
      x : α
      hx : LT.lt 0 (f x)
      ⊢ Eq (ProbabilityTheory.truncation f A x) (Function.comp ((Set.Ioc 0 A).indica …
    -/
  · simp only [truncation, indicator, hx, Set.mem_Ioc, id, Function.comp_apply]
    /-
      case h.inl
      α : Type u_1
      f : α → Real
      A : Real
      h : ∀ (x : α), LE.le 0 (f x)
      x : α
      hx : LT.lt 0 (f x)
      ⊢ Eq (ite (And (LT.lt (Neg.neg A) (f x)) (LE.le (f x) A)) (f x) 0) (ite (And T …
    -/
    by_cases h'x : f x ≤ A
      /-
        case pos
        α : Type u_1
        f : α → Real
        A : Real
        h : ∀ (x : α), LE.le 0 (f x)
        x : α
        hx : LT.lt 0 (f x)
        h'x : LE.le (f x) A
        ⊢ Eq (ite (And (LT.lt (Neg.neg A) (f x)) (LE.le (f x) A)) (f x) 0) (ite (And T …
      -/
    · have : -A < f x := by linarith [h x]
      /-
        case pos
        α : Type u_1
        f : α → Real
        A : Real
        h : ∀ (x : α), LE.le 0 (f x)
        x : α
        hx : LT.lt 0 (f x)
        h'x : LE.le (f x) A
        this : LT.lt (Neg.neg A) (f x)
        ⊢ Eq (ite (And (LT.lt (Neg.neg A) (f x)) (LE.le (f x) A)) (f x) 0) (ite (And T …
      -/
      simp only [this, true_and]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        f : α → Real
        A : Real
        h : ∀ (x : α), LE.le 0 (f x)
        x : α
        hx : LT.lt 0 (f x)
        h'x : Not (LE.le (f x) A)
        ⊢ Eq (ite (And (LT.lt (Neg.neg A) (f x)) (LE.le (f x) A)) (f x) 0) (ite (And T …
      -/
    · simp only [h'x, and_false]
      /-
        🎉 no goals
      -/
    /-
      case h.inr
      α : Type u_1
      f : α → Real
      A : Real
      h : ∀ (x : α), LE.le 0 (f x)
      x : α
      hx : Eq 0 (f x)
      ⊢ Eq (ProbabilityTheory.truncation f A x) (Function.comp ((Set.Ioc 0 A).indica …
    -/
  · simp only [truncation, indicator, hx, id, Function.comp_apply, ite_self]
    /-
      🎉 no goals
    -/


theorem truncation_nonneg {f : α → ℝ} (A : ℝ) {x : α} (h : 0 ≤ f x) : 0 ≤ truncation f A x :=
  Set.indicator_apply_nonneg fun _ => h


theorem _root_.MeasureTheory.AEStronglyMeasurable.memℒp_truncation [IsFiniteMeasure μ]
    (hf : AEStronglyMeasurable f μ) {A : ℝ} {p : ℝ≥0∞} : Memℒp (truncation f A) p μ :=
  Memℒp.of_bound hf.truncation |A| (Eventually.of_forall fun _ => abs_truncation_le_bound _ _ _)


theorem _root_.MeasureTheory.AEStronglyMeasurable.integrable_truncation [IsFiniteMeasure μ]
    (hf : AEStronglyMeasurable f μ) {A : ℝ} : Integrable (truncation f A) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    ⊢ MeasureTheory.Integrable (ProbabilityTheory.truncation f A) μ
  -/
  rw [← memℒp_one_iff_integrable]; exact hf.memℒp_truncation
                                   /-
                                     🎉 no goals
                                   -/


theorem moment_truncation_eq_intervalIntegral (hf : AEStronglyMeasurable f μ) {A : ℝ} (hA : 0 ≤ A)
    {n : ℕ} (hn : n ≠ 0) : ∫ x, truncation f A x ^ n ∂μ = ∫ y in -A..A, y ^ n ∂Measure.map f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    hA : LE.le 0 A
    n : Nat
    hn : Ne n 0
    ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (ProbabilityTheory.truncatio …
  -/
  have M : MeasurableSet (Set.Ioc (-A) A) := measurableSet_Ioc
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    hA : LE.le 0 A
    n : Nat
    hn : Ne n 0
    M : MeasurableSet (Set.Ioc (Neg.neg A) A)
    ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (ProbabilityTheory.truncatio …
  -/
  change ∫ x, (fun z => indicator (Set.Ioc (-A) A) id z ^ n) (f x) ∂μ = _
  rw [← integral_map (f := fun z => _ ^ n) hf.aemeasurable, intervalIntegral.integral_of_le,
    ← integral_indicator M]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEStronglyMeasurable f μ
      A : Real
      hA : LE.le 0 A
      n : Nat
      hn : Ne n 0
      M : MeasurableSet (Set.Ioc (Neg.neg A) A)
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map f μ) fun y => HPow.hPo …
    -/
  · simp only [indicator, zero_pow hn, id, ite_pow]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEStronglyMeasurable f μ
      A : Real
      hA : LE.le 0 A
      n : Nat
      hn : Ne n 0
      M : MeasurableSet (Set.Ioc (Neg.neg A) A)
      ⊢ LE.le (Neg.neg A) A
    -/
  · linarith
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.AEStronglyMeasurable f μ
      A : Real
      hA : LE.le 0 A
      n : Nat
      hn : Ne n 0
      M : MeasurableSet (Set.Ioc (Neg.neg A) A)
      ⊢ MeasureTheory.AEStronglyMeasurable (fun z => HPow.hPow ((Set.Ioc (Neg.neg A) …
    -/
  · exact ((measurable_id.indicator M).pow_const n).aestronglyMeasurable
    /-
      🎉 no goals
    -/


theorem moment_truncation_eq_intervalIntegral_of_nonneg (hf : AEStronglyMeasurable f μ) {A : ℝ}
    {n : ℕ} (hn : n ≠ 0) (h'f : 0 ≤ f) :
    ∫ x, truncation f A x ^ n ∂μ = ∫ y in (0)..A, y ^ n ∂Measure.map f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    n : Nat
    hn : Ne n 0
    h'f : LE.le 0 f
    ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (ProbabilityTheory.truncatio …
  -/
  have M : MeasurableSet (Set.Ioc 0 A) := measurableSet_Ioc
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    n : Nat
    hn : Ne n 0
    h'f : LE.le 0 f
    M : MeasurableSet (Set.Ioc 0 A)
    ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (ProbabilityTheory.truncatio …
  -/
  have M' : MeasurableSet (Set.Ioc A 0) := measurableSet_Ioc
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    n : Nat
    hn : Ne n 0
    h'f : LE.le 0 f
    M : MeasurableSet (Set.Ioc 0 A)
    M' : MeasurableSet (Set.Ioc A 0)
    ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (ProbabilityTheory.truncatio …
  -/
  rw [truncation_eq_of_nonneg h'f]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    n : Nat
    hn : Ne n 0
    h'f : LE.le 0 f
    M : MeasurableSet (Set.Ioc 0 A)
    M' : MeasurableSet (Set.Ioc A 0)
    ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (Function.comp ((Set.Ioc 0 A …
  -/
  change ∫ x, (fun z => indicator (Set.Ioc 0 A) id z ^ n) (f x) ∂μ = _
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    n : Nat
    hn : Ne n 0
    h'f : LE.le 0 f
    M : MeasurableSet (Set.Ioc 0 A)
    M' : MeasurableSet (Set.Ioc A 0)
    ⊢ Eq (MeasureTheory.integral μ fun x => (fun z => HPow.hPow ((Set.Ioc 0 A).ind …
  -/
  rcases le_or_lt 0 A with (hA | hA)
  · rw [← integral_map (f := fun z => _ ^ n) hf.aemeasurable, intervalIntegral.integral_of_le hA,
      ← integral_indicator M]
      /-
        case inl
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LE.le 0 A
        ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map f μ) fun y => HPow.hPo …
      -/
    · simp only [indicator, zero_pow hn, id, ite_pow]
      /-
        🎉 no goals
      -/
      /-
        case inl
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LE.le 0 A
        ⊢ MeasureTheory.AEStronglyMeasurable (fun z => HPow.hPow ((Set.Ioc 0 A).indica …
      -/
    · exact ((measurable_id.indicator M).pow_const n).aestronglyMeasurable
      /-
        🎉 no goals
      -/
  · rw [← integral_map (f := fun z => _ ^ n) hf.aemeasurable, intervalIntegral.integral_of_ge hA.le,
      ← integral_indicator M']
    · simp only [Set.Ioc_eq_empty_of_le hA.le, zero_pow hn, Set.indicator_empty, integral_zero,
        zero_eq_neg]
      /-
        case inr
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LT.lt A 0
        ⊢ Eq (MeasureTheory.integral (MeasureTheory.Measure.map f μ) fun x => (Set.Ioc …
      -/
      apply integral_eq_zero_of_ae
      have : ∀ᵐ x ∂Measure.map f μ, (0 : ℝ) ≤ x :=
        (ae_map_iff hf.aemeasurable measurableSet_Ici).2 (Eventually.of_forall h'f)
      /-
        case inr.hf
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LT.lt A 0
        this : Filter.Eventually (fun x => LE.le 0 x) (MeasureTheory.ae (MeasureTheory …
        ⊢ (MeasureTheory.ae (MeasureTheory.Measure.map f μ)).EventuallyEq ((Set.Ioc A  …
      -/
      filter_upwards [this] with x hx
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LT.lt A 0
        this : Filter.Eventually (fun x => LE.le 0 x) (MeasureTheory.ae (MeasureTheory …
        x : Real
        hx : LE.le 0 x
        ⊢ Eq ((Set.Ioc A 0).indicator (fun x => HPow.hPow x n) x) (0 x)
      -/
      simp only [indicator, Set.mem_Ioc, Pi.zero_apply, ite_eq_right_iff, and_imp]
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LT.lt A 0
        this : Filter.Eventually (fun x => LE.le 0 x) (MeasureTheory.ae (MeasureTheory …
        x : Real
        hx : LE.le 0 x
        ⊢ LT.lt A x → LE.le x 0 → Eq (HPow.hPow x n) 0
      -/
      intro _ h''x
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LT.lt A 0
        this : Filter.Eventually (fun x => LE.le 0 x) (MeasureTheory.ae (MeasureTheory …
        x : Real
        hx : LE.le 0 x
        a✝ : LT.lt A x
        h''x : LE.le x 0
        ⊢ Eq (HPow.hPow x n) 0
      -/
      have : x = 0 := by linarith
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LT.lt A 0
        this✝ : Filter.Eventually (fun x => LE.le 0 x) (MeasureTheory.ae (MeasureTheor …
        x : Real
        hx : LE.le 0 x
        a✝ : LT.lt A x
        h''x : LE.le x 0
        this : Eq x 0
        ⊢ Eq (HPow.hPow x n) 0
      -/
      simp [this, zero_pow hn]
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → Real
        hf : MeasureTheory.AEStronglyMeasurable f μ
        A : Real
        n : Nat
        hn : Ne n 0
        h'f : LE.le 0 f
        M : MeasurableSet (Set.Ioc 0 A)
        M' : MeasurableSet (Set.Ioc A 0)
        hA : LT.lt A 0
        ⊢ MeasureTheory.AEStronglyMeasurable (fun z => HPow.hPow ((Set.Ioc 0 A).indica …
      -/
    · exact ((measurable_id.indicator M).pow_const n).aestronglyMeasurable
      /-
        🎉 no goals
      -/


theorem integral_truncation_eq_intervalIntegral (hf : AEStronglyMeasurable f μ) {A : ℝ}
    (hA : 0 ≤ A) : ∫ x, truncation f A x ∂μ = ∫ y in -A..A, y ∂Measure.map f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    hA : LE.le 0 A
    ⊢ Eq (MeasureTheory.integral μ fun x => ProbabilityTheory.truncation f A x) (i …
  -/
  simpa using moment_truncation_eq_intervalIntegral hf hA one_ne_zero
  /-
    🎉 no goals
  -/


theorem integral_truncation_eq_intervalIntegral_of_nonneg (hf : AEStronglyMeasurable f μ) {A : ℝ}
    (h'f : 0 ≤ f) : ∫ x, truncation f A x ∂μ = ∫ y in (0)..A, y ∂Measure.map f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    A : Real
    h'f : LE.le 0 f
    ⊢ Eq (MeasureTheory.integral μ fun x => ProbabilityTheory.truncation f A x) (i …
  -/
  simpa using moment_truncation_eq_intervalIntegral_of_nonneg hf one_ne_zero h'f
  /-
    🎉 no goals
  -/


theorem integral_truncation_le_integral_of_nonneg (hf : Integrable f μ) (h'f : 0 ≤ f) {A : ℝ} :
    ∫ x, truncation f A x ∂μ ≤ ∫ x, f x ∂μ := by
  apply integral_mono_of_nonneg
    (Eventually.of_forall fun x => ?_) hf (Eventually.of_forall fun x => ?_)
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      h'f : LE.le 0 f
      A : Real
      x : α
      ⊢ LE.le (0 x) (ProbabilityTheory.truncation f A x)
    -/
  · exact truncation_nonneg _ (h'f x)
    /-
      🎉 no goals
    -/
  · calc
      truncation f A x ≤ |truncation f A x| := le_abs_self _
      _ ≤ |f x| := abs_truncation_le_abs_self _ _ _
      _ = f x := abs_of_nonneg (h'f x)


/-- If a function is integrable, then the integral of its truncated versions converges to the
integral of the whole function. -/
theorem tendsto_integral_truncation {f : α → ℝ} (hf : Integrable f μ) :
    Tendsto (fun A => ∫ x, truncation f A x ∂μ) atTop (𝓝 (∫ x, f x ∂μ)) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ Filter.Tendsto (fun A => MeasureTheory.integral μ fun x => ProbabilityTheory …
  -/
  refine tendsto_integral_filter_of_dominated_convergence (fun x => abs (f x)) ?_ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ Filter.Eventually (fun n => MeasureTheory.AEStronglyMeasurable (ProbabilityT …
    -/
  · exact Eventually.of_forall fun A ↦ hf.aestronglyMeasurable.truncation
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm.norm (Pr …
    -/
  · filter_upwards with A
    /-
      case refine_2.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      A : Real
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (ProbabilityTheory.truncation f …
    -/
    filter_upwards with x
    /-
      case refine_2.h.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      A : Real
      x : α
      ⊢ LE.le (Norm.norm (ProbabilityTheory.truncation f A x)) (abs (f x))
    -/
    rw [Real.norm_eq_abs]
    /-
      case refine_2.h.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      A : Real
      x : α
      ⊢ LE.le (abs (ProbabilityTheory.truncation f A x)) (abs (f x))
    -/
    exact abs_truncation_le_abs_self _ _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.Integrable (fun x => abs (f x)) μ
    -/
  · exact hf.abs
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => ProbabilityTheory.trunc …
    -/
  · filter_upwards with x
    /-
      case refine_4.h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      x : α
      ⊢ Filter.Tendsto (fun n => ProbabilityTheory.truncation f n x) Filter.atTop (n …
    -/
    apply tendsto_const_nhds.congr' _
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      x : α
      ⊢ Filter.atTop.EventuallyEq (fun x_1 => f x) fun n => ProbabilityTheory.trunca …
    -/
    filter_upwards [Ioi_mem_atTop (abs (f x))] with A hA
    /-
      case h
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → Real
      hf : MeasureTheory.Integrable f μ
      x : α
      A : Real
      hA : Membership.mem (Set.Ioi (abs (f x))) A
      ⊢ Eq (f x) (ProbabilityTheory.truncation f A x)
    -/
    exact (truncation_eq_self hA).symm
    /-
      🎉 no goals
    -/


theorem IdentDistrib.truncation {β : Type*} [MeasurableSpace β] {ν : Measure β} {f : α → ℝ}
    {g : β → ℝ} (h : IdentDistrib f g μ ν) {A : ℝ} :
    IdentDistrib (truncation f A) (truncation g A) μ ν :=
  h.comp (measurable_id.indicator measurableSet_Ioc)


                                                /-
                                                  Ω : Type u_1
                                                  inst✝¹ : MeasureTheory.MeasureSpace Ω
                                                  inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
                                                  X : Ω → Real
                                                  ⊢ MeasureTheory.Measure Ω
                                                -/
theorem sum_prob_mem_Ioc_le {X : Ω → ℝ} (hint : Integrable X) (hnonneg : 0 ≤ X) {K : ℕ} {N : ℕ}
                                                /-
                                                  🎉 no goals
                                                -/
    (hKN : K ≤ N) :
    ∑ j ∈ range K, ℙ {ω | X ω ∈ Set.Ioc (j : ℝ) N} ≤ ENNReal.ofReal (𝔼[X] + 1) := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Ω → Real
    hint : MeasureTheory.Integrable X MeasureTheory.MeasureSpace.volume
    hnonneg : LE.le 0 X
    K N : Nat
    hKN : LE.le K N
    ⊢ LE.le ((Finset.range K).sum fun j => MeasureTheory.MeasureSpace.volume (setO …
  -/
  let ρ : Measure ℝ := Measure.map X ℙ
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Ω → Real
    hint : MeasureTheory.Integrable X MeasureTheory.MeasureSpace.volume
    hnonneg : LE.le 0 X
    K N : Nat
    hKN : LE.le K N
    ρ : MeasureTheory.Measure Real := MeasureTheory.Measure.map X MeasureTheory.Me …
    ⊢ LE.le ((Finset.range K).sum fun j => MeasureTheory.MeasureSpace.volume (setO …
  -/
  haveI : IsProbabilityMeasure ρ := isProbabilityMeasure_map hint.aemeasurable
  have A : ∑ j ∈ range K, ∫ _ in j..N, (1 : ℝ) ∂ρ ≤ 𝔼[X] + 1 :=
    calc
      ∑ j ∈ range K, ∫ _ in j..N, (1 : ℝ) ∂ρ =
          ∑ j ∈ range K, ∑ i ∈ Ico j N, ∫ _ in i..(i + 1 : ℕ), (1 : ℝ) ∂ρ := by
        apply sum_congr rfl fun j hj => ?_
        rw [intervalIntegral.sum_integral_adjacent_intervals_Ico ((mem_range.1 hj).le.trans hKN)]
        intro k _
        exact continuous_const.intervalIntegrable _ _
      _ = ∑ i ∈ range N, ∑ j ∈ range (min (i + 1) K), ∫ _ in i..(i + 1 : ℕ), (1 : ℝ) ∂ρ := by
        simp_rw [sum_sigma']
        refine sum_nbij' (fun p ↦ ⟨p.2, p.1⟩) (fun p ↦ ⟨p.2, p.1⟩) ?_ ?_ ?_ ?_ ?_ <;>
          aesop (add simp Nat.lt_succ_iff)
      _ ≤ ∑ i ∈ range N, (i + 1) * ∫ _ in i..(i + 1 : ℕ), (1 : ℝ) ∂ρ := by
        apply sum_le_sum fun i _ => ?_
        simp only [Nat.cast_add, Nat.cast_one, sum_const, card_range, nsmul_eq_mul, Nat.cast_min]
        refine mul_le_mul_of_nonneg_right (min_le_left _ _) ?_
        apply intervalIntegral.integral_nonneg
        · simp only [le_add_iff_nonneg_right, zero_le_one]
        · simp only [zero_le_one, imp_true_iff]
      _ ≤ ∑ i ∈ range N, ∫ x in i..(i + 1 : ℕ), x + 1 ∂ρ := by
        apply sum_le_sum fun i _ => ?_
        have I : (i : ℝ) ≤ (i + 1 : ℕ) := by
          simp only [Nat.cast_add, Nat.cast_one, le_add_iff_nonneg_right, zero_le_one]
        simp_rw [intervalIntegral.integral_of_le I, ← integral_mul_left]
        apply setIntegral_mono_on
        · exact continuous_const.integrableOn_Ioc
        · exact (continuous_id.add continuous_const).integrableOn_Ioc
        · exact measurableSet_Ioc
        · intro x hx
          simp only [Nat.cast_add, Nat.cast_one, Set.mem_Ioc] at hx
          simp [hx.1.le]
      _ = ∫ x in (0)..N, x + 1 ∂ρ := by
        rw [intervalIntegral.sum_integral_adjacent_intervals fun k _ => ?_]
        · norm_cast
        · exact (continuous_id.add continuous_const).intervalIntegrable _ _
      _ = ∫ x in (0)..N, x ∂ρ + ∫ x in (0)..N, 1 ∂ρ := by
        rw [intervalIntegral.integral_add]
        · exact continuous_id.intervalIntegrable _ _
        · exact continuous_const.intervalIntegrable _ _
      _ = 𝔼[truncation X N] + ∫ x in (0)..N, 1 ∂ρ := by
        rw [integral_truncation_eq_intervalIntegral_of_nonneg hint.1 hnonneg]
      _ ≤ 𝔼[X] + ∫ x in (0)..N, 1 ∂ρ :=
        (add_le_add_right (integral_truncation_le_integral_of_nonneg hint hnonneg) _)
      _ ≤ 𝔼[X] + 1 := by
        refine add_le_add le_rfl ?_
        rw [intervalIntegral.integral_of_le (Nat.cast_nonneg _)]
        simp only [integral_const, Measure.restrict_apply', measurableSet_Ioc, Set.univ_inter,
          Algebra.id.smul_eq_mul, mul_one]
        rw [← ENNReal.one_toReal]
        exact ENNReal.toReal_mono ENNReal.one_ne_top prob_le_one
  have B : ∀ a b, ℙ {ω | X ω ∈ Set.Ioc a b} = ENNReal.ofReal (∫ _ in Set.Ioc a b, (1 : ℝ) ∂ρ) := by
    intro a b
    rw [ofReal_setIntegral_one ρ _,
      Measure.map_apply_of_aemeasurable hint.aemeasurable measurableSet_Ioc]
    rfl
  calc
    ∑ j ∈ range K, ℙ {ω | X ω ∈ Set.Ioc (j : ℝ) N} =
        ∑ j ∈ range K, ENNReal.ofReal (∫ _ in Set.Ioc (j : ℝ) N, (1 : ℝ) ∂ρ) := by simp_rw [B]
    _ = ENNReal.ofReal (∑ j ∈ range K, ∫ _ in Set.Ioc (j : ℝ) N, (1 : ℝ) ∂ρ) := by
      rw [ENNReal.ofReal_sum_of_nonneg]
      simp only [integral_const, Algebra.id.smul_eq_mul, mul_one, ENNReal.toReal_nonneg,
        imp_true_iff]
    _ = ENNReal.ofReal (∑ j ∈ range K, ∫ _ in (j : ℝ)..N, (1 : ℝ) ∂ρ) := by
      congr 1
      refine sum_congr rfl fun j hj => ?_
      rw [intervalIntegral.integral_of_le (Nat.cast_le.2 ((mem_range.1 hj).le.trans hKN))]
    _ ≤ ENNReal.ofReal (𝔼[X] + 1) := ENNReal.ofReal_le_ofReal A


                                                     /-
                                                       Ω : Type u_1
                                                       inst✝¹ : MeasureTheory.MeasureSpace Ω
                                                       inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
                                                       X : Ω → Real
                                                       ⊢ MeasureTheory.Measure Ω
                                                     -/
theorem tsum_prob_mem_Ioi_lt_top {X : Ω → ℝ} (hint : Integrable X) (hnonneg : 0 ≤ X) :
                                                     /-
                                                       🎉 no goals
                                                     -/
    (∑' j : ℕ, ℙ {ω | X ω ∈ Set.Ioi (j : ℝ)}) < ∞ := by
  suffices ∀ K : ℕ, ∑ j ∈ range K, ℙ {ω | X ω ∈ Set.Ioi (j : ℝ)} ≤ ENNReal.ofReal (𝔼[X] + 1) from
    (le_of_tendsto_of_tendsto (ENNReal.tendsto_nat_tsum _) tendsto_const_nhds
      (Eventually.of_forall this)).trans_lt ENNReal.ofReal_lt_top
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Ω → Real
    hint : MeasureTheory.Integrable X MeasureTheory.MeasureSpace.volume
    hnonneg : LE.le 0 X
    ⊢ ∀ (K : Nat), LE.le ((Finset.range K).sum fun j => MeasureTheory.MeasureSpace …
  -/
  intro K
  have A : Tendsto (fun N : ℕ => ∑ j ∈ range K, ℙ {ω | X ω ∈ Set.Ioc (j : ℝ) N}) atTop
      (𝓝 (∑ j ∈ range K, ℙ {ω | X ω ∈ Set.Ioi (j : ℝ)})) := by
    refine tendsto_finset_sum _ fun i _ => ?_
    have : {ω | X ω ∈ Set.Ioi (i : ℝ)} = ⋃ N : ℕ, {ω | X ω ∈ Set.Ioc (i : ℝ) N} := by
      apply Set.Subset.antisymm _ _
      · intro ω hω
        obtain ⟨N, hN⟩ : ∃ N : ℕ, X ω ≤ N := exists_nat_ge (X ω)
        exact Set.mem_iUnion.2 ⟨N, hω, hN⟩
      · simp (config := {contextual := true}) only [Set.mem_Ioc, Set.mem_Ioi,
          Set.iUnion_subset_iff, Set.setOf_subset_setOf, imp_true_iff]
    rw [this]
    apply tendsto_measure_iUnion_atTop
    intro m n hmn x hx
    exact ⟨hx.1, hx.2.trans (Nat.cast_le.2 hmn)⟩
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Ω → Real
    hint : MeasureTheory.Integrable X MeasureTheory.MeasureSpace.volume
    hnonneg : LE.le 0 X
    K : Nat
    A : Filter.Tendsto (fun N => (Finset.range K).sum fun j => MeasureTheory.Measu …
    ⊢ LE.le ((Finset.range K).sum fun j => MeasureTheory.MeasureSpace.volume (setO …
  -/
  apply le_of_tendsto_of_tendsto A tendsto_const_nhds
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Ω → Real
    hint : MeasureTheory.Integrable X MeasureTheory.MeasureSpace.volume
    hnonneg : LE.le 0 X
    K : Nat
    A : Filter.Tendsto (fun N => (Finset.range K).sum fun j => MeasureTheory.Measu …
    ⊢ Filter.atTop.EventuallyLE (fun N => (Finset.range K).sum fun j => MeasureThe …
  -/
  filter_upwards [Ici_mem_atTop K] with N hN
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Ω → Real
    hint : MeasureTheory.Integrable X MeasureTheory.MeasureSpace.volume
    hnonneg : LE.le 0 X
    K : Nat
    A : Filter.Tendsto (fun N => (Finset.range K).sum fun j => MeasureTheory.Measu …
    N : Nat
    hN : Membership.mem (Set.Ici K) N
    ⊢ LE.le ((Finset.range K).sum fun j => MeasureTheory.MeasureSpace.volume (setO …
  -/
  exact sum_prob_mem_Ioc_le hint hnonneg hN
  /-
    🎉 no goals
  -/


                                                       /-
                                                         Ω : Type u_1
                                                         inst✝¹ : MeasureTheory.MeasureSpace Ω
                                                         inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
                                                         X : Ω → Real
                                                         ⊢ MeasureTheory.Measure Ω
                                                       -/
theorem sum_variance_truncation_le {X : Ω → ℝ} (hint : Integrable X) (hnonneg : 0 ≤ X) (K : ℕ) :
                                                       /-
                                                         🎉 no goals
                                                       -/
    ∑ j ∈ range K, ((j : ℝ) ^ 2)⁻¹ * 𝔼[truncation X j ^ 2] ≤ 2 * 𝔼[X] := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Ω → Real
    hint : MeasureTheory.Integrable X MeasureTheory.MeasureSpace.volume
    hnonneg : LE.le 0 X
    K : Nat
    ⊢ LE.le ((Finset.range K).sum fun j => HMul.hMul (Inv.inv (HPow.hPow (↑j) 2))  …
  -/
  set Y := fun n : ℕ => truncation X n
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Ω → Real
    hint : MeasureTheory.Integrable X MeasureTheory.MeasureSpace.volume
    hnonneg : LE.le 0 X
    K : Nat
    Y : Nat → Ω → Real := fun n => ProbabilityTheory.truncation X ↑n
    ⊢ LE.le ((Finset.range K).sum fun j => HMul.hMul (Inv.inv (HPow.hPow (↑j) 2))  …
  -/
  let ρ : Measure ℝ := Measure.map X ℙ
  have Y2 : ∀ n, 𝔼[Y n ^ 2] = ∫ x in (0)..n, x ^ 2 ∂ρ := by
    intro n
    change 𝔼[fun x => Y n x ^ 2] = _
    rw [moment_truncation_eq_intervalIntegral_of_nonneg hint.1 two_ne_zero hnonneg]
  calc
    ∑ j ∈ range K, ((j : ℝ) ^ 2)⁻¹ * 𝔼[Y j ^ 2] =
        ∑ j ∈ range K, ((j : ℝ) ^ 2)⁻¹ * ∫ x in (0)..j, x ^ 2 ∂ρ := by simp_rw [Y2]
    _ = ∑ j ∈ range K, ((j : ℝ) ^ 2)⁻¹ * ∑ k ∈ range j, ∫ x in k..(k + 1 : ℕ), x ^ 2 ∂ρ := by
      congr 1 with j
      congr 1
      rw [intervalIntegral.sum_integral_adjacent_intervals]
      · norm_cast
      intro k _
      exact (continuous_id.pow _).intervalIntegrable _ _
    _ = ∑ k ∈ range K, (∑ j ∈ Ioo k K, ((j : ℝ) ^ 2)⁻¹) * ∫ x in k..(k + 1 : ℕ), x ^ 2 ∂ρ := by
      simp_rw [mul_sum, sum_mul, sum_sigma']
      refine sum_nbij' (fun p ↦ ⟨p.2, p.1⟩) (fun p ↦ ⟨p.2, p.1⟩) ?_ ?_ ?_ ?_ ?_ <;>
        aesop (add unsafe lt_trans)
    _ ≤ ∑ k ∈ range K, 2 / (k + 1 : ℝ) * ∫ x in k..(k + 1 : ℕ), x ^ 2 ∂ρ := by
      apply sum_le_sum fun k _ => ?_
      refine mul_le_mul_of_nonneg_right (sum_Ioo_inv_sq_le _ _) ?_
      refine intervalIntegral.integral_nonneg_of_forall ?_ fun u => sq_nonneg _
      simp only [Nat.cast_add, Nat.cast_one, le_add_iff_nonneg_right, zero_le_one]
    _ ≤ ∑ k ∈ range K, ∫ x in k..(k + 1 : ℕ), 2 * x ∂ρ := by
      apply sum_le_sum fun k _ => ?_
      have Ik : (k : ℝ) ≤ (k + 1 : ℕ) := by simp
      rw [← intervalIntegral.integral_const_mul, intervalIntegral.integral_of_le Ik,
        intervalIntegral.integral_of_le Ik]
      refine setIntegral_mono_on ?_ ?_ measurableSet_Ioc fun x hx => ?_
      · apply Continuous.integrableOn_Ioc
        exact continuous_const.mul (continuous_pow 2)
      · apply Continuous.integrableOn_Ioc
        exact continuous_const.mul continuous_id'
      · calc
          ↑2 / (↑k + ↑1) * x ^ 2 = x / (k + 1) * (2 * x) := by ring
          _ ≤ 1 * (2 * x) :=
            (mul_le_mul_of_nonneg_right (by
              convert (div_le_one _).2 hx.2
              · norm_cast
              simp only [Nat.cast_add, Nat.cast_one]
              linarith only [show (0 : ℝ) ≤ k from Nat.cast_nonneg k])
              (mul_nonneg zero_le_two ((Nat.cast_nonneg k).trans hx.1.le)))
          _ = 2 * x := by rw [one_mul]
    _ = 2 * ∫ x in (0 : ℝ)..K, x ∂ρ := by
      rw [intervalIntegral.sum_integral_adjacent_intervals fun k _ => ?_]
      swap; · exact (continuous_const.mul continuous_id').intervalIntegrable _ _
      rw [intervalIntegral.integral_const_mul]
      norm_cast
    _ ≤ 2 * 𝔼[X] := mul_le_mul_of_nonneg_left (by
      rw [← integral_truncation_eq_intervalIntegral_of_nonneg hint.1 hnonneg]
      exact integral_truncation_le_integral_of_nonneg hint hnonneg) zero_le_two


                  inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
                                   X : Nat → Ω → Real
                                   ⊢ MeasureTheory.Measure Ω
                                 -/
variable (X : ℕ → Ω → ℝ) (hint : Integrable (X 0))
                                 /-
                                   🎉 no goals
                                 -/
                      /-
                        Ω : Type u_1
                        inst✝¹ : MeasureTheory.MeasureSpace Ω
                        inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
                        X : Nat → Ω → Real
                        hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
                        f g : Ω → Real
                        ⊢ MeasureTheory.Measure Ω
                      -/
                      /-
                        🎉 no goals
                      -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  (hindep : Pairwise (IndepFun on X)) (hident : ∀ i, IdentDistrib (X i) (X 0))
                                                     /-
                                                       🎉 no goals
                                                     -/
  (hnonneg : ∀ i ω, 0 ≤ X i ω)

include hint hindep hident hnonneg in
/-- The truncation of `Xᵢ` up to `i` satisfies the strong law of large numbers (with respect to
the truncated expectation) along the sequence `c^n`, for any `c > 1`, up to a given `ε > 0`.
This follows from a variance control. -/
theorem strong_law_aux1 {c : ℝ} (c_one : 1 < c) {ε : ℝ} (εpos : 0 < ε) : ∀ᵐ ω, ∀ᶠ n : ℕ in atTop,
    |∑ i ∈ range ⌊c ^ n⌋₊, truncation (X i) i ω - 𝔼[∑ i ∈ range ⌊c ^ n⌋₊, truncation (X i) i]| <
    ε * ⌊c ^ n⌋₊ := by
  /- Let `S n = ∑ i ∈ range n, Y i` where `Y i = truncation (X i) i`. We should show that
    `|S k - 𝔼[S k]| / k ≤ ε` along the sequence of powers of `c`. For this, we apply Borel-Cantelli:
    it suffices to show that the converse probabilities are summable. From Chebyshev inequality,
    this will follow from a variance control `∑' Var[S (c^i)] / (c^i)^2 < ∞`. This is checked in
    `I2` using pairwise independence to expand the variance of the sum as the sum of the variances,
    and then a straightforward but tedious computation (essentially boiling down to the fact that
    the sum of `1/(c ^ i)^2` beyond a threshold `j` is comparable to `1/j^2`).
    Note that we have written `c^i` in the above proof sketch, but rigorously one should put integer
    parts everywhere, making things more painful. We write `u i = ⌊c^i⌋₊` for brevity. -/
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Filter.Eventually (fun ω => Filter.Eventually (fun n => LT.lt (abs (HSub.hSu …
  -/
  have c_pos : 0 < c := zero_lt_one.trans c_one
  have hX : ∀ i, AEStronglyMeasurable (X i) ℙ := fun i =>
    (hident i).symm.aestronglyMeasurable_snd hint.1
  have A : ∀ i, StronglyMeasurable (indicator (Set.Ioc (-i : ℝ) i) id) := fun i =>
    stronglyMeasurable_id.indicator measurableSet_Ioc
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    c_pos : LT.lt 0 c
    hX : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) MeasureTheory.Measu …
    A : ∀ (i : Real), MeasureTheory.StronglyMeasurable ((Set.Ioc (Neg.neg i) i).in …
    ⊢ Filter.Eventually (fun ω => Filter.Eventually (fun n => LT.lt (abs (HSub.hSu …
  -/
  set Y := fun n : ℕ => truncation (X n) n
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    c_pos : LT.lt 0 c
    hX : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) MeasureTheory.Measu …
    A : ∀ (i : Real), MeasureTheory.StronglyMeasurable ((Set.Ioc (Neg.neg i) i).in …
    Y : Nat → Ω → Real := fun n => ProbabilityTheory.truncation (X n) ↑n
    ⊢ Filter.Eventually (fun ω => Filter.Eventually (fun n => LT.lt (abs (HSub.hSu …
  -/
  set S := fun n => ∑ i ∈ range n, Y i with hS
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    c_pos : LT.lt 0 c
    hX : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) MeasureTheory.Measu …
    A : ∀ (i : Real), MeasureTheory.StronglyMeasurable ((Set.Ioc (Neg.neg i) i).in …
    Y : Nat → Ω → Real := fun n => ProbabilityTheory.truncation (X n) ↑n
    S : Nat → Ω → Real := fun n => (Finset.range n).sum fun i => Y i
    hS : Eq S fun n => (Finset.range n).sum fun i => Y i
    ⊢ Filter.Eventually (fun ω => Filter.Eventually (fun n => LT.lt (abs (HSub.hSu …
  -/
  let u : ℕ → ℕ := fun n => ⌊c ^ n⌋₊
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    c_pos : LT.lt 0 c
    hX : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) MeasureTheory.Measu …
    A : ∀ (i : Real), MeasureTheory.StronglyMeasurable ((Set.Ioc (Neg.neg i) i).in …
    Y : Nat → Ω → Real := fun n => ProbabilityTheory.truncation (X n) ↑n
    S : Nat → Ω → Real := fun n => (Finset.range n).sum fun i => Y i
    hS : Eq S fun n => (Finset.range n).sum fun i => Y i
    u : Nat → Nat := fun n => Nat.floor (HPow.hPow c n)
    ⊢ Filter.Eventually (fun ω => Filter.Eventually (fun n => LT.lt (abs (HSub.hSu …
  -/
  have u_mono : Monotone u := fun i j hij => Nat.floor_mono (pow_right_mono₀ c_one.le hij)
  have I1 : ∀ K, ∑ j ∈ range K, ((j : ℝ) ^ 2)⁻¹ * Var[Y j] ≤ 2 * 𝔼[X 0] := by
    intro K
    calc
      ∑ j ∈ range K, ((j : ℝ) ^ 2)⁻¹ * Var[Y j] ≤
          ∑ j ∈ range K, ((j : ℝ) ^ 2)⁻¹ * 𝔼[truncation (X 0) j ^ 2] := by
        apply sum_le_sum fun j _ => ?_
        refine mul_le_mul_of_nonneg_left ?_ (inv_nonneg.2 (sq_nonneg _))
        rw [(hident j).truncation.variance_eq]
        exact variance_le_expectation_sq (hX 0).truncation
      _ ≤ 2 * 𝔼[X 0] := sum_variance_truncation_le hint (hnonneg 0) K
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    c_pos : LT.lt 0 c
    hX : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) MeasureTheory.Measu …
    A : ∀ (i : Real), MeasureTheory.StronglyMeasurable ((Set.Ioc (Neg.neg i) i).in …
    Y : Nat → Ω → Real := fun n => ProbabilityTheory.truncation (X n) ↑n
    S : Nat → Ω → Real := fun n => (Finset.range n).sum fun i => Y i
    hS : Eq S fun n => (Finset.range n).sum fun i => Y i
    u : Nat → Nat := fun n => Nat.floor (HPow.hPow c n)
    u_mono : Monotone u
    I1 : ∀ (K : Nat), LE.le ((Finset.range K).sum fun j => HMul.hMul (Inv.inv (HPo …
    ⊢ Filter.Eventually (fun ω => Filter.Eventually (fun n => LT.lt (abs (HSub.hSu …
  -/
  let C := c ^ 5 * (c - 1)⁻¹ ^ 3 * (2 * 𝔼[X 0])
  have I2 : ∀ N, ∑ i ∈ range N, ((u i : ℝ) ^ 2)⁻¹ * Var[S (u i)] ≤ C := by
    intro N
    calc
      ∑ i ∈ range N, ((u i : ℝ) ^ 2)⁻¹ * Var[S (u i)] =
          ∑ i ∈ range N, ((u i : ℝ) ^ 2)⁻¹ * ∑ j ∈ range (u i), Var[Y j] := by
        congr 1 with i
        congr 1
        rw [hS, IndepFun.variance_sum]
        · intro j _
          exact (hident j).aestronglyMeasurable_fst.memℒp_truncation
        · intro k _ l _ hkl
          exact (hindep hkl).comp (A k).measurable (A l).measurable
      _ = ∑ j ∈ range (u (N - 1)), (∑ i ∈ range N with j < u i, ((u i : ℝ) ^ 2)⁻¹) * Var[Y j] := by
        simp_rw [mul_sum, sum_mul, sum_sigma']
        refine sum_nbij' (fun p ↦ ⟨p.2, p.1⟩) (fun p ↦ ⟨p.2, p.1⟩) ?_ ?_ ?_ ?_ ?_
        · simp only [mem_sigma, mem_range, filter_congr_decidable, mem_filter, and_imp,
            Sigma.forall]
          exact fun a b haN hb ↦ ⟨hb.trans_le <| u_mono <| Nat.le_pred_of_lt haN, haN, hb⟩
        all_goals aesop
      _ ≤ ∑ j ∈ range (u (N - 1)), c ^ 5 * (c - 1)⁻¹ ^ 3 / ↑j ^ 2 * Var[Y j] := by
        apply sum_le_sum fun j hj => ?_
        rcases @eq_zero_or_pos _ _ j with (rfl | hj)
        · simp only [Nat.cast_zero, zero_pow, Ne, Nat.one_ne_zero,
            not_false_iff, div_zero, zero_mul]
          simp only [Y, Nat.cast_zero, truncation_zero, variance_zero, mul_zero, le_rfl]
        apply mul_le_mul_of_nonneg_right _ (variance_nonneg _ _)
        convert sum_div_nat_floor_pow_sq_le_div_sq N (Nat.cast_pos.2 hj) c_one using 2
        · simp only [u, Nat.cast_lt]
        · simp only [Y, S, u, C, one_div]
      _ = c ^ 5 * (c - 1)⁻¹ ^ 3 * ∑ j ∈ range (u (N - 1)), ((j : ℝ) ^ 2)⁻¹ * Var[Y j] := by
        simp_rw [mul_sum, div_eq_mul_inv, mul_assoc]
      _ ≤ c ^ 5 * (c - 1)⁻¹ ^ 3 * (2 * 𝔼[X 0]) := by
        apply mul_le_mul_of_nonneg_left (I1 _)
        apply mul_nonneg (pow_nonneg c_pos.le _)
        exact pow_nonneg (inv_nonneg.2 (sub_nonneg.2 c_one.le)) _
  have I3 : ∀ N, ∑ i ∈ range N, ℙ {ω | (u i * ε : ℝ) ≤ |S (u i) ω - 𝔼[S (u i)]|} ≤
      ENNReal.ofReal (ε⁻¹ ^ 2 * C) := by
    intro N
    calc
      ∑ i ∈ range N, ℙ {ω | (u i * ε : ℝ) ≤ |S (u i) ω - 𝔼[S (u i)]|} ≤
          ∑ i ∈ range N, ENNReal.ofReal (Var[S (u i)] / (u i * ε) ^ 2) := by
        refine sum_le_sum fun i _ => ?_
        apply meas_ge_le_variance_div_sq
        · exact memℒp_finset_sum' _ fun j _ => (hident j).aestronglyMeasurable_fst.memℒp_truncation
        · apply mul_pos (Nat.cast_pos.2 _) εpos
          refine zero_lt_one.trans_le ?_
          apply Nat.le_floor
          rw [Nat.cast_one]
          apply one_le_pow₀ c_one.le
      _ = ENNReal.ofReal (∑ i ∈ range N, Var[S (u i)] / (u i * ε) ^ 2) := by
        rw [ENNReal.ofReal_sum_of_nonneg fun i _ => ?_]
        exact div_nonneg (variance_nonneg _ _) (sq_nonneg _)
      _ ≤ ENNReal.ofReal (ε⁻¹ ^ 2 * C) := by
        apply ENNReal.ofReal_le_ofReal
        -- Porting note: do most of the rewrites under `conv` so as not to expand `variance`
        conv_lhs =>
          enter [2, i]
          rw [div_eq_inv_mul, ← inv_pow, mul_inv, mul_comm _ ε⁻¹, mul_pow, mul_assoc]
        rw [← mul_sum]
        refine mul_le_mul_of_nonneg_left ?_ (sq_nonneg _)
        conv_lhs => enter [2, i]; rw [inv_pow]
        exact I2 N
  have I4 : (∑' i, ℙ {ω | (u i * ε : ℝ) ≤ |S (u i) ω - 𝔼[S (u i)]|}) < ∞ :=
    (le_of_tendsto_of_tendsto' (ENNReal.tendsto_nat_tsum _) tendsto_const_nhds I3).trans_lt
      ENNReal.ofReal_lt_top
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    c_pos : LT.lt 0 c
    hX : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) MeasureTheory.Measu …
    A : ∀ (i : Real), MeasureTheory.StronglyMeasurable ((Set.Ioc (Neg.neg i) i).in …
    Y : Nat → Ω → Real := fun n => ProbabilityTheory.truncation (X n) ↑n
    S : Nat → Ω → Real := fun n => (Finset.range n).sum fun i => Y i
    hS : Eq S fun n => (Finset.range n).sum fun i => Y i
    u : Nat → Nat := fun n => Nat.floor (HPow.hPow c n)
    u_mono : Monotone u
    I1 : ∀ (K : Nat), LE.le ((Finset.range K).sum fun j => HMul.hMul (Inv.inv (HPo …
    C : Real := HMul.hMul (HMul.hMul (HPow.hPow c 5) (HPow.hPow (Inv.inv (HSub.hSu …
    I2 : ∀ (N : Nat), LE.le ((Finset.range N).sum fun i => HMul.hMul (Inv.inv (HPo …
    I3 : ∀ (N : Nat), LE.le ((Finset.range N).sum fun i => MeasureTheory.MeasureSp …
    I4 : LT.lt (tsum fun i => MeasureTheory.MeasureSpace.volume (setOf fun ω => LE …
    ⊢ Filter.Eventually (fun ω => Filter.Eventually (fun n => LT.lt (abs (HSub.hSu …
  -/
  filter_upwards [ae_eventually_not_mem I4.ne] with ω hω
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    c_pos : LT.lt 0 c
    hX : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) MeasureTheory.Measu …
    A : ∀ (i : Real), MeasureTheory.StronglyMeasurable ((Set.Ioc (Neg.neg i) i).in …
    Y : Nat → Ω → Real := fun n => ProbabilityTheory.truncation (X n) ↑n
    S : Nat → Ω → Real := fun n => (Finset.range n).sum fun i => Y i
    hS : Eq S fun n => (Finset.range n).sum fun i => Y i
    u : Nat → Nat := fun n => Nat.floor (HPow.hPow c n)
    u_mono : Monotone u
    I1 : ∀ (K : Nat), LE.le ((Finset.range K).sum fun j => HMul.hMul (Inv.inv (HPo …
    C : Real := HMul.hMul (HMul.hMul (HPow.hPow c 5) (HPow.hPow (Inv.inv (HSub.hSu …
    I2 : ∀ (N : Nat), LE.le ((Finset.range N).sum fun i => HMul.hMul (Inv.inv (HPo …
    I3 : ∀ (N : Nat), LE.le ((Finset.range N).sum fun i => MeasureTheory.MeasureSp …
    I4 : LT.lt (tsum fun i => MeasureTheory.MeasureSpace.volume (setOf fun ω => LE …
    ω : Ω
    hω : Filter.Eventually (fun n => Not (LE.le (HMul.hMul (↑(u n)) ε) (abs (HSub. …
    ⊢ Filter.Eventually (fun n => LT.lt (abs (HSub.hSub ((Finset.range (Nat.floor  …
  -/
  simp_rw [S, not_le, mul_comm, sum_apply] at hω
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ε : Real
    εpos : LT.lt 0 ε
    c_pos : LT.lt 0 c
    hX : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) MeasureTheory.Measu …
    A : ∀ (i : Real), MeasureTheory.StronglyMeasurable ((Set.Ioc (Neg.neg i) i).in …
    Y : Nat → Ω → Real := fun n => ProbabilityTheory.truncation (X n) ↑n
    S : Nat → Ω → Real := fun n => (Finset.range n).sum fun i => Y i
    hS : Eq S fun n => (Finset.range n).sum fun i => Y i
    u : Nat → Nat := fun n => Nat.floor (HPow.hPow c n)
    u_mono : Monotone u
    I1 : ∀ (K : Nat), LE.le ((Finset.range K).sum fun j => HMul.hMul (Inv.inv (HPo …
    C : Real := HMul.hMul (HMul.hMul (HPow.hPow c 5) (HPow.hPow (Inv.inv (HSub.hSu …
    I2 : ∀ (N : Nat), LE.le ((Finset.range N).sum fun i => HMul.hMul (Inv.inv (HPo …
    I3 : ∀ (N : Nat), LE.le ((Finset.range N).sum fun i => MeasureTheory.MeasureSp …
    I4 : LT.lt (tsum fun i => MeasureTheory.MeasureSpace.volume (setOf fun ω => LE …
    ω : Ω
    hω : Filter.Eventually (fun n => LT.lt (abs (HSub.hSub ((Finset.range (u n)).s …
    ⊢ Filter.Eventually (fun n => LT.lt (abs (HSub.hSub ((Finset.range (Nat.floor  …
  -/
  convert hω; simp only [Y, S, u, C, sum_apply]
              /-
                🎉 no goals
              -/


i)]|} ≤
          ∑ i ∈ range N, ENNReal.ofReal (Var[S (u i)] / (u i * ε) ^ 2) := by
        refine sum_le_sum fun i _ => ?_
        apply meas_ge_le_variance_div_sq
        · exact memℒp_finset_sum' _ fun j _ => (hident j).aestronglyMeasurable_fst.memℒp_truncation
        · apply mul_pos (Nat.cast_pos.2 _) εpos
          refine zero_lt_one.trans_le ?_
          apply Nat.le_floor
          rw [Nat.cast_one]
          apply one_le_pow₀ c_one.le
      _ = ENNReal.ofReal (∑ i ∈ range N, Var[S (u i)] / (u i * ε) ^ 2) := by
        rw [ENNReal.ofReal_sum_of_nonneg fun i _ => ?_]
        exact div_nonneg (variance_nonneg _ _) (sq_nonneg _)
      _ ≤ ENNReal.ofReal (ε⁻¹ ^ 2 * C) := by
        apply ENNReal.ofReal_le_ofReal
        -- Porting note: do most of the rewrites under `conv` so as not to expand `variance`
        conv_lhs =>
          enter [2, i]
          rw [div_eq_inv_mul, ← inv_pow, mul_inv, mul_comm _ ε⁻¹, mul_pow, mul_assoc]
        rw [← mul_sum]
        refine mul_le_mul_of_nonneg_left ?_ (sq_nonneg _)
        conv_lhs => enter [2, i]; rw [inv_pow]
        exact I2 N
  have I4 : (∑' i, ℙ {ω | (u i * ε : ℝ) ≤ |S (u i) ω - 𝔼[S (u i)]|}) < ∞ :=
    (le_of_tendsto_of_tendsto' (ENNReal.tendsto_nat_tsum _) tendsto_const_nhds I3).trans_lt
      ENNReal.ofReal_lt_top
  filter_upwards [ae_eventually_not_mem I4.ne] with ω hω
  simp_rw [S, not_le, mul_comm, sum_apply] at hω
  convert hω; simp only [Y, S, u, C, sum_apply]

include hint hindep hident hnonneg in
/- The truncation of `Xᵢ` up to `i` satisfies the strong law of large numbers
(with respect to the truncated expectation) along the sequence
`c^n`, for any `c > 1`. This follows from `strong_law_aux1` by varying `ε`. -/
theorem strong_law_aux2 {c : ℝ} (c_one : 1 < c) :
    ∀ᵐ ω, (fun n : ℕ => ∑ i ∈ range ⌊c ^ n⌋₊, truncation (X i) i ω -
      𝔼[∑ i ∈ range ⌊c ^ n⌋₊, truncation (X i) i]) =o[atTop] fun n : ℕ => (⌊c ^ n⌋₊ : ℝ) := by
  obtain ⟨v, -, v_pos, v_lim⟩ :
      ∃ v : ℕ → ℝ, StrictAnti v ∧ (∀ n : ℕ, 0 < v n) ∧ Tendsto v atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ)
  /-
    case intro.intro.intro
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    v : Nat → Real
    v_pos : ∀ (n : Nat), LT.lt 0 (v n)
    v_lim : Filter.Tendsto v Filter.atTop (nhds 0)
    ⊢ Filter.Eventually (fun ω => Asymptotics.IsLittleO Filter.atTop (fun n => HSu …
  -/
  have := fun i => strong_law_aux1 X hint hindep hident hnonneg c_one (v_pos i)
  /-
    case intro.intro.intro
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    v : Nat → Real
    v_pos : ∀ (n : Nat), LT.lt 0 (v n)
    v_lim : Filter.Tendsto v Filter.atTop (nhds 0)
    this : ∀ (i : Nat), Filter.Eventually (fun ω => Filter.Eventually (fun n => LT …
    ⊢ Filter.Eventually (fun ω => Asymptotics.IsLittleO Filter.atTop (fun n => HSu …
  -/
  filter_upwards [ae_all_iff.2 this] with ω hω
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    v : Nat → Real
    v_pos : ∀ (n : Nat), LT.lt 0 (v n)
    v_lim : Filter.Tendsto v Filter.atTop (nhds 0)
    this : ∀ (i : Nat), Filter.Eventually (fun ω => Filter.Eventually (fun n => LT …
    ω : Ω
    hω : ∀ (i : Nat), Filter.Eventually (fun n => LT.lt (abs (HSub.hSub ((Finset.r …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Nat.f …
  -/
  apply Asymptotics.isLittleO_iff.2 fun ε εpos => ?_
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    v : Nat → Real
    v_pos : ∀ (n : Nat), LT.lt 0 (v n)
    v_lim : Filter.Tendsto v Filter.atTop (nhds 0)
    this : ∀ (i : Nat), Filter.Eventually (fun ω => Filter.Eventually (fun n => LT …
    ω : Ω
    hω : ∀ (i : Nat), Filter.Eventually (fun n => LT.lt (abs (HSub.hSub ((Finset.r …
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub ((Finset.range (Nat. …
  -/
  obtain ⟨i, hi⟩ : ∃ i, v i < ε := ((tendsto_order.1 v_lim).2 ε εpos).exists
  /-
    case intro
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    v : Nat → Real
    v_pos : ∀ (n : Nat), LT.lt 0 (v n)
    v_lim : Filter.Tendsto v Filter.atTop (nhds 0)
    this : ∀ (i : Nat), Filter.Eventually (fun ω => Filter.Eventually (fun n => LT …
    ω : Ω
    hω : ∀ (i : Nat), Filter.Eventually (fun n => LT.lt (abs (HSub.hSub ((Finset.r …
    ε : Real
    εpos : LT.lt 0 ε
    i : Nat
    hi : LT.lt (v i) ε
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (HSub.hSub ((Finset.range (Nat. …
  -/
  filter_upwards [hω i] with n hn
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    v : Nat → Real
    v_pos : ∀ (n : Nat), LT.lt 0 (v n)
    v_lim : Filter.Tendsto v Filter.atTop (nhds 0)
    this : ∀ (i : Nat), Filter.Eventually (fun ω => Filter.Eventually (fun n => LT …
    ω : Ω
    hω : ∀ (i : Nat), Filter.Eventually (fun n => LT.lt (abs (HSub.hSub ((Finset.r …
    ε : Real
    εpos : LT.lt 0 ε
    i : Nat
    hi : LT.lt (v i) ε
    n : Nat
    hn : LT.lt (abs (HSub.hSub ((Finset.range (Nat.floor (HPow.hPow c n))).sum fun …
    ⊢ LE.le (Norm.norm (HSub.hSub ((Finset.range (Nat.floor (HPow.hPow c n))).sum  …
  -/
  simp only [Real.norm_eq_abs, abs_abs, Nat.abs_cast]
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    v : Nat → Real
    v_pos : ∀ (n : Nat), LT.lt 0 (v n)
    v_lim : Filter.Tendsto v Filter.atTop (nhds 0)
    this : ∀ (i : Nat), Filter.Eventually (fun ω => Filter.Eventually (fun n => LT …
    ω : Ω
    hω : ∀ (i : Nat), Filter.Eventually (fun n => LT.lt (abs (HSub.hSub ((Finset.r …
    ε : Real
    εpos : LT.lt 0 ε
    i : Nat
    hi : LT.lt (v i) ε
    n : Nat
    hn : LT.lt (abs (HSub.hSub ((Finset.range (Nat.floor (HPow.hPow c n))).sum fun …
    ⊢ LE.le (abs (HSub.hSub ((Finset.range (Nat.floor (HPow.hPow c n))).sum fun i  …
  -/
  exact hn.le.trans (mul_le_mul_of_nonneg_right hi.le (Nat.cast_nonneg _))
  /-
    🎉 no goals
  -/


 have I4 : (∑' i, ℙ {ω | (u i * ε : ℝ) ≤ |S (u i) ω - 𝔼[S (u i)]|}) < ∞ :=
    (le_of_tendsto_of_tendsto' (ENNReal.tendsto_nat_tsum _) tendsto_const_nhds I3).trans_lt
      ENNReal.ofReal_lt_top
  filter_upwards [ae_eventually_not_mem I4.ne] with ω hω
  simp_rw [S, not_le, mul_comm, sum_apply] at hω
  convert hω; simp only [Y, S, u, C, sum_apply]

include hint hindep hident hnonneg in
/- The truncation of `Xᵢ` up to `i` satisfies the strong law of large numbers
(with respect to the truncated expectation) along the sequence
`c^n`, for any `c > 1`. This follows from `strong_law_aux1` by varying `ε`. -/
theorem strong_law_aux2 {c : ℝ} (c_one : 1 < c) :
    ∀ᵐ ω, (fun n : ℕ => ∑ i ∈ range ⌊c ^ n⌋₊, truncation (X i) i ω -
      𝔼[∑ i ∈ range ⌊c ^ n⌋₊, truncation (X i) i]) =o[atTop] fun n : ℕ => (⌊c ^ n⌋₊ : ℝ) := by
  obtain ⟨v, -, v_pos, v_lim⟩ :
      ∃ v : ℕ → ℝ, StrictAnti v ∧ (∀ n : ℕ, 0 < v n) ∧ Tendsto v atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ)
  have := fun i => strong_law_aux1 X hint hindep hident hnonneg c_one (v_pos i)
  filter_upwards [ae_all_iff.2 this] with ω hω
  apply Asymptotics.isLittleO_iff.2 fun ε εpos => ?_
  obtain ⟨i, hi⟩ : ∃ i, v i < ε := ((tendsto_order.1 v_lim).2 ε εpos).exists
  filter_upwards [hω i] with n hn
  simp only [Real.norm_eq_abs, abs_abs, Nat.abs_cast]
  exact hn.le.trans (mul_le_mul_of_nonneg_right hi.le (Nat.cast_nonneg _))

include hint hident in
/-- The expectation of the truncated version of `Xᵢ` behaves asymptotically like the whole
expectation. This follows from convergence and Cesàro averaging. -/
theorem strong_law_aux3 :
    (fun n => 𝔼[∑ i ∈ range n, truncation (X i) i] - n * 𝔼[X 0]) =o[atTop] ((↑) : ℕ → ℝ) := by
  have A : Tendsto (fun i => 𝔼[truncation (X i) i]) atTop (𝓝 𝔼[X 0]) := by
    convert (tendsto_integral_truncation hint).comp tendsto_natCast_atTop_atTop using 1
    ext i
    exact (hident i).truncation.integral_eq
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    A : Filter.Tendsto (fun i => MeasureTheory.integral MeasureTheory.MeasureSpace …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub (MeasureTheory.integr …
  -/
  convert Asymptotics.isLittleO_sum_range_of_tendsto_zero (tendsto_sub_nhds_zero_iff.2 A) using 1
  /-
    case h.e'_7
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    A : Filter.Tendsto (fun i => MeasureTheory.integral MeasureTheory.MeasureSpace …
    ⊢ Eq (fun n => HSub.hSub (MeasureTheory.integral MeasureTheory.MeasureSpace.vo …
  -/
  ext1 n
  /-
    case h.e'_7.h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    A : Filter.Tendsto (fun i => MeasureTheory.integral MeasureTheory.MeasureSpace …
    n : Nat
    ⊢ Eq (HSub.hSub (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun  …
  -/
  simp only [sum_sub_distrib, sum_const, card_range, nsmul_eq_mul, sum_apply, sub_left_inj]
  /-
    case h.e'_7.h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    A : Filter.Tendsto (fun i => MeasureTheory.integral MeasureTheory.MeasureSpace …
    n : Nat
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun a => (Finse …
  -/
  rw [integral_finset_sum _ fun i _ => ?_]
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    A : Filter.Tendsto (fun i => MeasureTheory.integral MeasureTheory.MeasureSpace …
    n i : Nat
    x✝ : Membership.mem (Finset.range n) i
    ⊢ MeasureTheory.Integrable (ProbabilityTheory.truncation (X i) ↑i) MeasureTheo …
  -/
  exact ((hident i).symm.integrable_snd hint).1.integrable_truncation
  /-
    🎉 no goals
  -/


c ^ n⌋₊ : ℝ) := by
  obtain ⟨v, -, v_pos, v_lim⟩ :
      ∃ v : ℕ → ℝ, StrictAnti v ∧ (∀ n : ℕ, 0 < v n) ∧ Tendsto v atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ)
  have := fun i => strong_law_aux1 X hint hindep hident hnonneg c_one (v_pos i)
  filter_upwards [ae_all_iff.2 this] with ω hω
  apply Asymptotics.isLittleO_iff.2 fun ε εpos => ?_
  obtain ⟨i, hi⟩ : ∃ i, v i < ε := ((tendsto_order.1 v_lim).2 ε εpos).exists
  filter_upwards [hω i] with n hn
  simp only [Real.norm_eq_abs, abs_abs, Nat.abs_cast]
  exact hn.le.trans (mul_le_mul_of_nonneg_right hi.le (Nat.cast_nonneg _))

include hint hident in
/-- The expectation of the truncated version of `Xᵢ` behaves asymptotically like the whole
expectation. This follows from convergence and Cesàro averaging. -/
theorem strong_law_aux3 :
    (fun n => 𝔼[∑ i ∈ range n, truncation (X i) i] - n * 𝔼[X 0]) =o[atTop] ((↑) : ℕ → ℝ) := by
  have A : Tendsto (fun i => 𝔼[truncation (X i) i]) atTop (𝓝 𝔼[X 0]) := by
    convert (tendsto_integral_truncation hint).comp tendsto_natCast_atTop_atTop using 1
    ext i
    exact (hident i).truncation.integral_eq
  convert Asymptotics.isLittleO_sum_range_of_tendsto_zero (tendsto_sub_nhds_zero_iff.2 A) using 1
  ext1 n
  simp only [sum_sub_distrib, sum_const, card_range, nsmul_eq_mul, sum_apply, sub_left_inj]
  rw [integral_finset_sum _ fun i _ => ?_]
  exact ((hident i).symm.integrable_snd hint).1.integrable_truncation

include hint hindep hident hnonneg in
/- The truncation of `Xᵢ` up to `i` satisfies the strong law of large numbers
(with respect to the original expectation) along the sequence
`c^n`, for any `c > 1`. This follows from the version from the truncated expectation, and the
fact that the truncated and the original expectations have the same asymptotic behavior. -/
theorem strong_law_aux4 {c : ℝ} (c_one : 1 < c) :
    ∀ᵐ ω, (fun n : ℕ => ∑ i ∈ range ⌊c ^ n⌋₊, truncation (X i) i ω - ⌊c ^ n⌋₊ * 𝔼[X 0]) =o[atTop]
    fun n : ℕ => (⌊c ^ n⌋₊ : ℝ) := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ⊢ Filter.Eventually (fun ω => Asymptotics.IsLittleO Filter.atTop (fun n => HSu …
  -/
  filter_upwards [strong_law_aux2 X hint hindep hident hnonneg c_one] with ω hω
  have A : Tendsto (fun n : ℕ => ⌊c ^ n⌋₊) atTop atTop :=
    tendsto_nat_floor_atTop.comp (tendsto_pow_atTop_atTop_of_one_lt c_one)
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ω : Ω
    hω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Na …
    A : Filter.Tendsto (fun n => Nat.floor (HPow.hPow c n)) Filter.atTop Filter.at …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Nat.f …
  -/
  convert hω.add ((strong_law_aux3 X hint hident).comp_tendsto A) using 1
  /-
    case h.e'_7
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ω : Ω
    hω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Na …
    A : Filter.Tendsto (fun n => Nat.floor (HPow.hPow c n)) Filter.atTop Filter.at …
    ⊢ Eq (fun n => HSub.hSub ((Finset.range (Nat.floor (HPow.hPow c n))).sum fun i …
  -/
  ext1 n
  /-
    case h.e'_7.h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    ω : Ω
    hω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Na …
    A : Filter.Tendsto (fun n => Nat.floor (HPow.hPow c n)) Filter.atTop Filter.at …
    n : Nat
    ⊢ Eq (HSub.hSub ((Finset.range (Nat.floor (HPow.hPow c n))).sum fun i => Proba …
  -/
  simp
  /-
    🎉 no goals
  -/


] - n * 𝔼[X 0]) =o[atTop] ((↑) : ℕ → ℝ) := by
  have A : Tendsto (fun i => 𝔼[truncation (X i) i]) atTop (𝓝 𝔼[X 0]) := by
    convert (tendsto_integral_truncation hint).comp tendsto_natCast_atTop_atTop using 1
    ext i
    exact (hident i).truncation.integral_eq
  convert Asymptotics.isLittleO_sum_range_of_tendsto_zero (tendsto_sub_nhds_zero_iff.2 A) using 1
  ext1 n
  simp only [sum_sub_distrib, sum_const, card_range, nsmul_eq_mul, sum_apply, sub_left_inj]
  rw [integral_finset_sum _ fun i _ => ?_]
  exact ((hident i).symm.integrable_snd hint).1.integrable_truncation

include hint hindep hident hnonneg in
/- The truncation of `Xᵢ` up to `i` satisfies the strong law of large numbers
(with respect to the original expectation) along the sequence
`c^n`, for any `c > 1`. This follows from the version from the truncated expectation, and the
fact that the truncated and the original expectations have the same asymptotic behavior. -/
theorem strong_law_aux4 {c : ℝ} (c_one : 1 < c) :
    ∀ᵐ ω, (fun n : ℕ => ∑ i ∈ range ⌊c ^ n⌋₊, truncation (X i) i ω - ⌊c ^ n⌋₊ * 𝔼[X 0]) =o[atTop]
    fun n : ℕ => (⌊c ^ n⌋₊ : ℝ) := by
  filter_upwards [strong_law_aux2 X hint hindep hident hnonneg c_one] with ω hω
  have A : Tendsto (fun n : ℕ => ⌊c ^ n⌋₊) atTop atTop :=
    tendsto_nat_floor_atTop.comp (tendsto_pow_atTop_atTop_of_one_lt c_one)
  convert hω.add ((strong_law_aux3 X hint hident).comp_tendsto A) using 1
  ext1 n
  simp

include hint hident hnonneg in
/-- The truncated and non-truncated versions of `Xᵢ` have the same asymptotic behavior, as they
almost surely coincide at all but finitely many steps. This follows from a probability computation
and Borel-Cantelli. -/
theorem strong_law_aux5 :
    ∀ᵐ ω, (fun n : ℕ => ∑ i ∈ range n, truncation (X i) i ω - ∑ i ∈ range n, X i ω) =o[atTop]
    fun n : ℕ => (n : ℝ) := by
  have A : (∑' j : ℕ, ℙ {ω | X j ω ∈ Set.Ioi (j : ℝ)}) < ∞ := by
    convert tsum_prob_mem_Ioi_lt_top hint (hnonneg 0) using 2
    ext1 j
    exact (hident j).measure_mem_eq measurableSet_Ioi
  have B : ∀ᵐ ω, Tendsto (fun n : ℕ => truncation (X n) n ω - X n ω) atTop (𝓝 0) := by
    filter_upwards [ae_eventually_not_mem A.ne] with ω hω
    apply tendsto_const_nhds.congr' _
    filter_upwards [hω, Ioi_mem_atTop 0] with n hn npos
    simp only [truncation, indicator, Set.mem_Ioc, id, Function.comp_apply]
    split_ifs with h
    · exact (sub_self _).symm
    · have : -(n : ℝ) < X n ω := by
        apply lt_of_lt_of_le _ (hnonneg n ω)
        simpa only [Right.neg_neg_iff, Nat.cast_pos] using npos
      simp only [this, true_and, not_le] at h
      exact (hn h).elim
  /-
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    A : LT.lt (tsum fun j => MeasureTheory.MeasureSpace.volume (setOf fun ω => Mem …
    B : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSub.hSub (Probabilit …
    ⊢ Filter.Eventually (fun ω => Asymptotics.IsLittleO Filter.atTop (fun n => HSu …
  -/
  filter_upwards [B] with ω hω
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    A : LT.lt (tsum fun j => MeasureTheory.MeasureSpace.volume (setOf fun ω => Mem …
    B : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSub.hSub (Probabilit …
    ω : Ω
    hω : Filter.Tendsto (fun n => HSub.hSub (ProbabilityTheory.truncation (X n) (↑ …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range n).sum …
  -/
  convert isLittleO_sum_range_of_tendsto_zero hω using 1
  /-
    case h.e'_7
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    A : LT.lt (tsum fun j => MeasureTheory.MeasureSpace.volume (setOf fun ω => Mem …
    B : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSub.hSub (Probabilit …
    ω : Ω
    hω : Filter.Tendsto (fun n => HSub.hSub (ProbabilityTheory.truncation (X n) (↑ …
    ⊢ Eq (fun n => HSub.hSub ((Finset.range n).sum fun i => ProbabilityTheory.trun …
  -/
  ext n
  /-
    case h.e'_7.h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    A : LT.lt (tsum fun j => MeasureTheory.MeasureSpace.volume (setOf fun ω => Mem …
    B : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSub.hSub (Probabilit …
    ω : Ω
    hω : Filter.Tendsto (fun n => HSub.hSub (ProbabilityTheory.truncation (X n) (↑ …
    n : Nat
    ⊢ Eq (HSub.hSub ((Finset.range n).sum fun i => ProbabilityTheory.truncation (X …
  -/
  rw [sum_sub_distrib]
  /-
    🎉 no goals
  -/


or_atTop.comp (tendsto_pow_atTop_atTop_of_one_lt c_one)
  convert hω.add ((strong_law_aux3 X hint hident).comp_tendsto A) using 1
  ext1 n
  simp

include hint hident hnonneg in
/-- The truncated and non-truncated versions of `Xᵢ` have the same asymptotic behavior, as they
almost surely coincide at all but finitely many steps. This follows from a probability computation
and Borel-Cantelli. -/
theorem strong_law_aux5 :
    ∀ᵐ ω, (fun n : ℕ => ∑ i ∈ range n, truncation (X i) i ω - ∑ i ∈ range n, X i ω) =o[atTop]
    fun n : ℕ => (n : ℝ) := by
  have A : (∑' j : ℕ, ℙ {ω | X j ω ∈ Set.Ioi (j : ℝ)}) < ∞ := by
    convert tsum_prob_mem_Ioi_lt_top hint (hnonneg 0) using 2
    ext1 j
    exact (hident j).measure_mem_eq measurableSet_Ioi
  have B : ∀ᵐ ω, Tendsto (fun n : ℕ => truncation (X n) n ω - X n ω) atTop (𝓝 0) := by
    filter_upwards [ae_eventually_not_mem A.ne] with ω hω
    apply tendsto_const_nhds.congr' _
    filter_upwards [hω, Ioi_mem_atTop 0] with n hn npos
    simp only [truncation, indicator, Set.mem_Ioc, id, Function.comp_apply]
    split_ifs with h
    · exact (sub_self _).symm
    · have : -(n : ℝ) < X n ω := by
        apply lt_of_lt_of_le _ (hnonneg n ω)
        simpa only [Right.neg_neg_iff, Nat.cast_pos] using npos
      simp only [this, true_and, not_le] at h
      exact (hn h).elim
  filter_upwards [B] with ω hω
  convert isLittleO_sum_range_of_tendsto_zero hω using 1
  ext n
  rw [sum_sub_distrib]

include hint hindep hident hnonneg in
/- `Xᵢ` satisfies the strong law of large numbers along the sequence
`c^n`, for any `c > 1`. This follows from the version for the truncated `Xᵢ`, and the fact that
`Xᵢ` and its truncated version have the same asymptotic behavior. -/
theorem strong_law_aux6 {c : ℝ} (c_one : 1 < c) :
    ∀ᵐ ω, Tendsto (fun n : ℕ => (∑ i ∈ range ⌊c ^ n⌋₊, X i ω) / ⌊c ^ n⌋₊) atTop (𝓝 𝔼[X 0]) := by
  have H : ∀ n : ℕ, (0 : ℝ) < ⌊c ^ n⌋₊ := by
    intro n
    refine zero_lt_one.trans_le ?_
    simp only [Nat.one_le_cast, Nat.one_le_floor_iff, one_le_pow₀ c_one.le]
  filter_upwards [strong_law_aux4 X hint hindep hident hnonneg c_one,
    strong_law_aux5 X hint hident hnonneg] with ω hω h'ω
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    H : ∀ (n : Nat), LT.lt 0 ↑(Nat.floor (HPow.hPow c n))
    ω : Ω
    hω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Na …
    h'ω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range n) …
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range (Nat.floor (HPow.hPow c n) …
  -/
  rw [← tendsto_sub_nhds_zero_iff, ← Asymptotics.isLittleO_one_iff ℝ]
  have L : (fun n : ℕ => ∑ i ∈ range ⌊c ^ n⌋₊, X i ω - ⌊c ^ n⌋₊ * 𝔼[X 0]) =o[atTop] fun n =>
      (⌊c ^ n⌋₊ : ℝ) := by
    have A : Tendsto (fun n : ℕ => ⌊c ^ n⌋₊) atTop atTop :=
      tendsto_nat_floor_atTop.comp (tendsto_pow_atTop_atTop_of_one_lt c_one)
    convert hω.sub (h'ω.comp_tendsto A) using 1
    ext1 n
    simp only [Function.comp_apply, sub_sub_sub_cancel_left]
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Real
    c_one : LT.lt 1 c
    H : ∀ (n : Nat), LT.lt 0 ↑(Nat.floor (HPow.hPow c n))
    ω : Ω
    hω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Na …
    h'ω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range n) …
    L : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Nat …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HSub.hSub (HDiv.hDiv ((Finset.r …
  -/
  convert L.mul_isBigO (isBigO_refl (fun n : ℕ => (⌊c ^ n⌋₊ : ℝ)⁻¹) atTop) using 1 <;>
   /-
     case h.e'_7
     Ω : Type u_1
     inst✝¹ : MeasureTheory.MeasureSpace Ω
     inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
     X : Nat → Ω → Real
     hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
     hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
     hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
     hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
     c : Real
     c_one : LT.lt 1 c
     H : ∀ (n : Nat), LT.lt 0 ↑(Nat.floor (HPow.hPow c n))
     ω : Ω
     hω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Na …
     h'ω : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range n) …
     L : Asymptotics.IsLittleO Filter.atTop (fun n => HSub.hSub ((Finset.range (Nat …
     ⊢ Eq (fun x => HSub.hSub (HDiv.hDiv ((Finset.range (Nat.floor (HPow.hPow c x)) …
   -/
           /-
             🎉 no goals
           -/
  (ext1 n; field_simp [(H n).ne'])
           /-
             🎉 no goals
           -/


] with ω hω
  convert isLittleO_sum_range_of_tendsto_zero hω using 1
  ext n
  rw [sum_sub_distrib]

include hint hindep hident hnonneg in
/- `Xᵢ` satisfies the strong law of large numbers along the sequence
`c^n`, for any `c > 1`. This follows from the version for the truncated `Xᵢ`, and the fact that
`Xᵢ` and its truncated version have the same asymptotic behavior. -/
theorem strong_law_aux6 {c : ℝ} (c_one : 1 < c) :
    ∀ᵐ ω, Tendsto (fun n : ℕ => (∑ i ∈ range ⌊c ^ n⌋₊, X i ω) / ⌊c ^ n⌋₊) atTop (𝓝 𝔼[X 0]) := by
  have H : ∀ n : ℕ, (0 : ℝ) < ⌊c ^ n⌋₊ := by
    intro n
    refine zero_lt_one.trans_le ?_
    simp only [Nat.one_le_cast, Nat.one_le_floor_iff, one_le_pow₀ c_one.le]
  filter_upwards [strong_law_aux4 X hint hindep hident hnonneg c_one,
    strong_law_aux5 X hint hident hnonneg] with ω hω h'ω
  rw [← tendsto_sub_nhds_zero_iff, ← Asymptotics.isLittleO_one_iff ℝ]
  have L : (fun n : ℕ => ∑ i ∈ range ⌊c ^ n⌋₊, X i ω - ⌊c ^ n⌋₊ * 𝔼[X 0]) =o[atTop] fun n =>
      (⌊c ^ n⌋₊ : ℝ) := by
    have A : Tendsto (fun n : ℕ => ⌊c ^ n⌋₊) atTop atTop :=
      tendsto_nat_floor_atTop.comp (tendsto_pow_atTop_atTop_of_one_lt c_one)
    convert hω.sub (h'ω.comp_tendsto A) using 1
    ext1 n
    simp only [Function.comp_apply, sub_sub_sub_cancel_left]
  convert L.mul_isBigO (isBigO_refl (fun n : ℕ => (⌊c ^ n⌋₊ : ℝ)⁻¹) atTop) using 1 <;>
  (ext1 n; field_simp [(H n).ne'])

include hint hindep hident hnonneg in
/-- `Xᵢ` satisfies the strong law of large numbers along all integers. This follows from the
corresponding fact along the sequences `c^n`, and the fact that any integer can be sandwiched
between `c^n` and `c^(n+1)` with comparably small error if `c` is close enough to `1`
(which is formalized in `tendsto_div_of_monotone_of_tendsto_div_floor_pow`). -/
theorem strong_law_aux7 :
    ∀ᵐ ω, Tendsto (fun n : ℕ => (∑ i ∈ range n, X i ω) / n) atTop (𝓝 𝔼[X 0]) := by
  obtain ⟨c, -, cone, clim⟩ :
      ∃ c : ℕ → ℝ, StrictAnti c ∧ (∀ n : ℕ, 1 < c n) ∧ Tendsto c atTop (𝓝 1) :=
    exists_seq_strictAnti_tendsto (1 : ℝ)
  have : ∀ k, ∀ᵐ ω,
      Tendsto (fun n : ℕ => (∑ i ∈ range ⌊c k ^ n⌋₊, X i ω) / ⌊c k ^ n⌋₊) atTop (𝓝 𝔼[X 0]) :=
    fun k => strong_law_aux6 X hint hindep hident hnonneg (cone k)
  /-
    case intro.intro.intro
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Nat → Real
    cone : ∀ (n : Nat), LT.lt 1 (c n)
    clim : Filter.Tendsto c Filter.atTop (nhds 1)
    this : ∀ (k : Nat), Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv. …
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
  -/
  filter_upwards [ae_all_iff.2 this] with ω hω
  /-
    case h
    Ω : Type u_1
    inst✝¹ : MeasureTheory.MeasureSpace Ω
    inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
    hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
    hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
    c : Nat → Real
    cone : ∀ (n : Nat), LT.lt 1 (c n)
    clim : Filter.Tendsto c Filter.atTop (nhds 1)
    this : ∀ (k : Nat), Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv. …
    ω : Ω
    hω : ∀ (i : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range (Nat.floor …
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fun i => X i ω) ↑n) …
  -/
  apply tendsto_div_of_monotone_of_tendsto_div_floor_pow _ _ _ c cone clim _
    /-
      Ω : Type u_1
      inst✝¹ : MeasureTheory.MeasureSpace Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
      X : Nat → Ω → Real
      hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
      hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
      hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
      c : Nat → Real
      cone : ∀ (n : Nat), LT.lt 1 (c n)
      clim : Filter.Tendsto c Filter.atTop (nhds 1)
      this : ∀ (k : Nat), Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv. …
      ω : Ω
      hω : ∀ (i : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range (Nat.floor …
      ⊢ Monotone fun n => (Finset.range n).sum fun i => X i ω
    -/
  · intro m n hmn
    /-
      Ω : Type u_1
      inst✝¹ : MeasureTheory.MeasureSpace Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
      X : Nat → Ω → Real
      hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
      hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
      hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
      c : Nat → Real
      cone : ∀ (n : Nat), LT.lt 1 (c n)
      clim : Filter.Tendsto c Filter.atTop (nhds 1)
      this : ∀ (k : Nat), Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv. …
      ω : Ω
      hω : ∀ (i : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range (Nat.floor …
      m n : Nat
      hmn : LE.le m n
      ⊢ LE.le ((fun n => (Finset.range n).sum fun i => X i ω) m) ((fun n => (Finset. …
    -/
    exact sum_le_sum_of_subset_of_nonneg (range_mono hmn) fun i _ _ => hnonneg i ω
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      inst✝¹ : MeasureTheory.MeasureSpace Ω
      inst✝ : MeasureTheory.IsProbabilityMeasure MeasureTheory.MeasureSpace.volume
      X : Nat → Ω → Real
      hint : MeasureTheory.Integrable (X 0) MeasureTheory.MeasureSpace.volume
      hindep : Pairwise (Function.onFun (fun f g => ProbabilityTheory.IndepFun f g M …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) MeasureTheory …
      hnonneg : ∀ (i : Nat) (ω : Ω), LE.le 0 (X i ω)
      c : Nat → Real
      cone : ∀ (n : Nat), LT.lt 1 (c n)
      clim : Filter.Tendsto c Filter.atTop (nhds 1)
      this : ∀ (k : Nat), Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv. …
      ω : Ω
      hω : ∀ (i : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range (Nat.floor …
      ⊢ ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range (Nat.floor (H …
    -/
  · exact hω
    /-
      🎉 no goals
    -/


/-- **Strong law of large numbers**, almost sure version: if `X n` is a sequence of independent
identically distributed integrable real-valued random variables, then `∑ i ∈ range n, X i / n`
converges almost surely to `𝔼[X 0]`. We give here the strong version, due to Etemadi, that only
requires pairwise independence. Superseded by `strong_law_ae`, which works for random variables
taking values in any Banach space. -/
theorem strong_law_ae_real {Ω : Type*} {m : MeasurableSpace Ω} {μ : Measure Ω}
    (X : ℕ → Ω → ℝ) (hint : Integrable (X 0) μ)
    (hindep : Pairwise ((IndepFun · · μ) on X))
    (hident : ∀ i, IdentDistrib (X i) (X 0) μ μ) :
    ∀ᵐ ω ∂μ, Tendsto (fun n : ℕ => (∑ i ∈ range n, X i ω) / n) atTop (𝓝 μ[X 0]) := by
  /-
    Ω : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
  -/
  let mΩ : MeasureSpace Ω := ⟨μ⟩
  -- first get rid of the trivial case where the space is not a probability space
  /-
    Ω : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
  -/
  by_cases h : ∀ᵐ ω, X 0 ω = 0
  · have I : ∀ᵐ ω, ∀ i, X i ω = 0 := by
      rw [ae_all_iff]
      intro i
      exact (hident i).symm.ae_snd (p := fun x ↦ x = 0) measurableSet_eq h
    /-
      case pos
      Ω : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X : Nat → Ω → Real
      hint : MeasureTheory.Integrable (X 0) μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
      h : Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae MeasureTheory. …
      I : Filter.Eventually (fun ω => ∀ (i : Nat), Eq (X i ω) 0) (MeasureTheory.ae M …
      ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
    -/
    filter_upwards [I] with ω hω
    /-
      case h
      Ω : Type u_2
      m : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      X : Nat → Ω → Real
      hint : MeasureTheory.Integrable (X 0) μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
      h : Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae MeasureTheory. …
      I : Filter.Eventually (fun ω => ∀ (i : Nat), Eq (X i ω) 0) (MeasureTheory.ae M …
      ω : Ω
      hω : ∀ (i : Nat), Eq (X i ω) 0
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fun i => X i ω) ↑n) …
    -/
    simpa [hω] using (integral_eq_zero_of_ae h).symm
    /-
      🎉 no goals
    -/
  have : IsProbabilityMeasure μ :=
    hint.isProbabilityMeasure_of_indepFun (X 0) (X 1) h (hindep zero_ne_one)
  -- then consider separately the positive and the negative part, and apply the result
  -- for nonnegative functions to them.
  /-
    case neg
    Ω : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae MeasureTh …
    this : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
  -/
  let pos : ℝ → ℝ := fun x => max x 0
  /-
    case neg
    Ω : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae MeasureTh …
    this : MeasureTheory.IsProbabilityMeasure μ
    pos : Real → Real := fun x => Max.max x 0
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
  -/
  let neg : ℝ → ℝ := fun x => max (-x) 0
  /-
    case neg
    Ω : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae MeasureTh …
    this : MeasureTheory.IsProbabilityMeasure μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
  -/
  have posm : Measurable pos := measurable_id'.max measurable_const
  /-
    case neg
    Ω : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae MeasureTh …
    this : MeasureTheory.IsProbabilityMeasure μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
  -/
  have negm : Measurable neg := measurable_id'.neg.max measurable_const
  have A : ∀ᵐ ω, Tendsto (fun n : ℕ => (∑ i ∈ range n, (pos ∘ X i) ω) / n) atTop (𝓝 𝔼[pos ∘ X 0]) :=
    strong_law_aux7 _ hint.pos_part (fun i j hij => (hindep hij).comp posm posm)
      (fun i => (hident i).comp posm) fun i ω => le_max_right _ _
  have B : ∀ᵐ ω, Tendsto (fun n : ℕ => (∑ i ∈ range n, (neg ∘ X i) ω) / n) atTop (𝓝 𝔼[neg ∘ X 0]) :=
    strong_law_aux7 _ hint.neg_part (fun i j hij => (hindep hij).comp negm negm)
      (fun i => (hident i).comp negm) fun i ω => le_max_right _ _
  /-
    case neg
    Ω : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae MeasureTh …
    this : MeasureTheory.IsProbabilityMeasure μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    A : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.ra …
    B : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.ra …
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.rang …
  -/
  filter_upwards [A, B] with ω hωpos hωneg
  /-
    case h
    Ω : Type u_2
    m : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    X : Nat → Ω → Real
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    mΩ : MeasureTheory.MeasureSpace Ω := MeasureTheory.MeasureSpace.mk μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae MeasureTh …
    this : MeasureTheory.IsProbabilityMeasure μ
    pos : Real → Real := fun x => Max.max x 0
    neg : Real → Real := fun x => Max.max (Neg.neg x) 0
    posm : Measurable pos
    negm : Measurable neg
    A : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.ra …
    B : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HDiv.hDiv ((Finset.ra …
    ω : Ω
    hωpos : Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fun i => Func …
    hωneg : Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fun i => Func …
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fun i => X i ω) ↑n) …
  -/
  convert hωpos.sub hωneg using 2
  · simp only [pos, neg, ← sub_div, ← sum_sub_distrib, max_zero_sub_max_neg_zero_eq_self,
      Function.comp_apply]
  · simp only [pos, neg, ← integral_sub hint.pos_part hint.neg_part,
      max_zero_sub_max_neg_zero_eq_self, Function.comp_apply, mΩ]


/-- Preliminary lemma for the strong law of large numbers for vector-valued random variables:
the composition of the random variables with a simple function satisfies the strong law of large
numbers. -/
lemma strong_law_ae_simpleFunc_comp (X : ℕ → Ω → E) (h' : Measurable (X 0))
    (hindep : Pairwise ((IndepFun · · μ) on X))
    (hident : ∀ i, IdentDistrib (X i) (X 0) μ μ) (φ : SimpleFunc E E) :
    ∀ᵐ ω ∂μ,
      Tendsto (fun n : ℕ ↦ (n : ℝ) ⁻¹ • (∑ i ∈ range n, φ (X i ω))) atTop (𝓝 μ[φ ∘ (X 0)]) := by
  -- this follows from the one-dimensional version when `φ` takes a single value, and is then
  -- extended to the general case by linearity.
  classical
  refine SimpleFunc.induction (P := fun ψ ↦ ∀ᵐ ω ∂μ,
    Tendsto (fun n : ℕ ↦ (n : ℝ) ⁻¹ • (∑ i ∈ range n, ψ (X i ω))) atTop (𝓝 μ[ψ ∘ (X 0)])) ?_ ?_ φ
  · intro c s hs
    simp only [SimpleFunc.const_zero, SimpleFunc.coe_piecewise, SimpleFunc.coe_const,
      SimpleFunc.coe_zero, piecewise_eq_indicator, Function.comp_apply]
    let F : E → ℝ := indicator s 1
    have F_meas : Measurable F := (measurable_indicator_const_iff 1).2 hs
    let Y : ℕ → Ω → ℝ := fun n ↦ F ∘ (X n)
    have : ∀ᵐ (ω : Ω) ∂μ, Tendsto (fun (n : ℕ) ↦ (n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, Y i ω)
        atTop (𝓝 μ[Y 0]) := by
      simp only [Function.const_one, smul_eq_mul, ← div_eq_inv_mul]
      apply strong_law_ae_real
      · exact SimpleFunc.integrable_of_isFiniteMeasure
          ((SimpleFunc.piecewise s hs (SimpleFunc.const _ (1 : ℝ))
            (SimpleFunc.const _ (0 : ℝ))).comp (X 0) h')
      · exact fun i j hij ↦ IndepFun.comp (hindep hij) F_meas F_meas
      · exact fun i ↦ (hident i).comp F_meas
    filter_upwards [this] with ω hω
    have I : indicator s (Function.const E c) = (fun x ↦ (indicator s (1 : E → ℝ) x) • c) := by
      ext
      rw [← indicator_smul_const_apply]
      congr! 1
      ext
      simp
    simp only [I, integral_smul_const]
    convert Tendsto.smul_const hω c using 1
    simp [F, Y, ← sum_smul, smul_smul]
  · rintro φ ψ - hφ hψ
    filter_upwards [hφ, hψ] with ω hωφ hωψ
    convert hωφ.add hωψ using 1
    · simp [sum_add_distrib]
    · congr 1
      rw [← integral_add]
      · rfl
      · exact (φ.comp (X 0) h').integrable_of_isFiniteMeasure
      · exact (ψ.comp (X 0) h').integrable_of_isFiniteMeasure


/-- Preliminary lemma for the strong law of large numbers for vector-valued random variables,
assuming measurability in addition to integrability. This is weakened to ae measurability in
the full version `ProbabilityTheory.strong_law_ae`. -/
lemma strong_law_ae_of_measurable
    (X : ℕ → Ω → E) (hint : Integrable (X 0) μ) (h' : StronglyMeasurable (X 0))
    (hindep : Pairwise ((IndepFun · · μ) on X))
    (hident : ∀ i, IdentDistrib (X i) (X 0) μ μ) :
    ∀ᵐ ω ∂μ, Tendsto (fun n : ℕ ↦ (n : ℝ) ⁻¹ • (∑ i ∈ range n, X i ω)) atTop (𝓝 μ[X 0]) := by
  /- Choose a simple function `φ` such that `φ (X 0)` approximates well enough `X 0` -- this is
  possible as `X 0` is strongly measurable. Then `φ (X n)` approximates well `X n`.
  Then the strong law for `φ (X n)` implies the strong law for `X n`, up to a small
  error controlled by `n⁻¹ ∑_{i=0}^{n-1} ‖X i - φ (X i)‖`. This one is also controlled thanks
  to the one-dimensional law of large numbers: it converges ae to `𝔼[‖X 0 - φ (X 0)‖]`, which
  is arbitrarily small for well chosen `φ`. -/
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  let s : Set E := Set.range (X 0) ∪ {0}
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  have zero_s : 0 ∈ s := by simp [s]
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  have : SeparableSpace s := h'.separableSpace_range_union_singleton
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    this : TopologicalSpace.SeparableSpace ↑s
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  have : Nonempty s := ⟨0, zero_s⟩
  -- sequence of approximating simple functions.
  let φ : ℕ → SimpleFunc E E :=
    SimpleFunc.nearestPt (fun k => Nat.casesOn k 0 ((↑) ∘ denseSeq s) : ℕ → E)
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    this✝ : TopologicalSpace.SeparableSpace ↑s
    this : Nonempty ↑s
    φ : Nat → MeasureTheory.SimpleFunc E E := MeasureTheory.SimpleFunc.nearestPt f …
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  let Y : ℕ → ℕ → Ω → E := fun k i ↦ (φ k) ∘ (X i)
  -- strong law for `φ (X n)`
  have A : ∀ᵐ ω ∂μ, ∀ k,
      Tendsto (fun n : ℕ ↦ (n : ℝ) ⁻¹ • (∑ i ∈ range n, Y k i ω)) atTop (𝓝 μ[Y k 0]) :=
    ae_all_iff.2 (fun k ↦ strong_law_ae_simpleFunc_comp X h'.measurable hindep hident (φ k))
  -- strong law for the error `‖X i - φ (X i)‖`
  have B : ∀ᵐ ω ∂μ, ∀ k, Tendsto (fun n : ℕ ↦ (∑ i ∈ range n, ‖(X i - Y k i) ω‖) / n)
        atTop (𝓝 μ[(fun ω ↦ ‖(X 0 - Y k 0) ω‖)]) := by
    apply ae_all_iff.2 (fun k ↦ ?_)
    let G : ℕ → E → ℝ := fun k x ↦ ‖x - φ k x‖
    have G_meas : ∀ k, Measurable (G k) :=
      fun k ↦ (measurable_id.sub_stronglyMeasurable (φ k).stronglyMeasurable).norm
    have I : ∀ k i, (fun ω ↦ ‖(X i - Y k i) ω‖) = (G k) ∘ (X i) := fun k i ↦ rfl
    apply strong_law_ae_real (fun i ω ↦ ‖(X i - Y k i) ω‖)
    · exact (hint.sub ((φ k).comp (X 0) h'.measurable).integrable_of_isFiniteMeasure).norm
    · unfold Function.onFun
      simp_rw [I]
      intro i j hij
      exact (hindep hij).comp (G_meas k) (G_meas k)
    · intro i
      simp_rw [I]
      apply (hident i).comp (G_meas k)
  -- check that, when both convergences above hold, then the strong law is satisfied
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    this✝ : TopologicalSpace.SeparableSpace ↑s
    this : Nonempty ↑s
    φ : Nat → MeasureTheory.SimpleFunc E E := MeasureTheory.SimpleFunc.nearestPt f …
    Y : Nat → Nat → Ω → E := fun k i => Function.comp (⇑(φ k)) (X i)
    A : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hS …
    B : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDi …
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  filter_upwards [A, B] with ω hω h'ω
  /-
    case h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    this✝ : TopologicalSpace.SeparableSpace ↑s
    this : Nonempty ↑s
    φ : Nat → MeasureTheory.SimpleFunc E E := MeasureTheory.SimpleFunc.nearestPt f …
    Y : Nat → Nat → Ω → E := fun k i => Function.comp (⇑(φ k)) (X i)
    A : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hS …
    B : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDi …
    ω : Ω
    hω : ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.r …
    h'ω : ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fu …
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun  …
  -/
  rw [tendsto_iff_norm_sub_tendsto_zero, tendsto_order]
  /-
    case h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    this✝ : TopologicalSpace.SeparableSpace ↑s
    this : Nonempty ↑s
    φ : Nat → MeasureTheory.SimpleFunc E E := MeasureTheory.SimpleFunc.nearestPt f …
    Y : Nat → Nat → Ω → E := fun k i => Function.comp (⇑(φ k)) (X i)
    A : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hS …
    B : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDi …
    ω : Ω
    hω : ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.r …
    h'ω : ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fu …
    ⊢ And (∀ (a' : Real), LT.lt a' 0 → Filter.Eventually (fun b => LT.lt a' (Norm. …
  -/
  refine ⟨fun c hc ↦ Eventually.of_forall (fun n ↦ hc.trans_le (norm_nonneg _)), ?_⟩
  -- start with some positive `ε` (the desired precision), and fix `δ` with `3 δ < ε`.
  /-
    case h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    this✝ : TopologicalSpace.SeparableSpace ↑s
    this : Nonempty ↑s
    φ : Nat → MeasureTheory.SimpleFunc E E := MeasureTheory.SimpleFunc.nearestPt f …
    Y : Nat → Nat → Ω → E := fun k i => Function.comp (⇑(φ k)) (X i)
    A : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hS …
    B : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDi …
    ω : Ω
    hω : ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.r …
    h'ω : ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fu …
    ⊢ ∀ (a' : Real), GT.gt a' 0 → Filter.Eventually (fun b => LT.lt (Norm.norm (HS …
  -/
  intro ε (εpos : 0 < ε)
  /-
    case h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    this✝ : TopologicalSpace.SeparableSpace ↑s
    this : Nonempty ↑s
    φ : Nat → MeasureTheory.SimpleFunc E E := MeasureTheory.SimpleFunc.nearestPt f …
    Y : Nat → Nat → Ω → E := fun k i => Function.comp (⇑(φ k)) (X i)
    A : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hS …
    B : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDi …
    ω : Ω
    hω : ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.r …
    h'ω : ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fu …
    ε : Real
    εpos : LT.lt 0 ε
    ⊢ Filter.Eventually (fun b => LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul (Inv.in …
  -/
  obtain ⟨δ, δpos, hδ⟩ : ∃ δ, 0 < δ ∧ δ + δ + δ < ε := ⟨ε/4, by positivity, by linarith⟩
  -- choose `k` large enough so that `φₖ (X 0)` approximates well enough `X 0`, up to the
  -- precision `δ`.
  obtain ⟨k, hk⟩ : ∃ k, ∫ ω, ‖(X 0 - Y k 0) ω‖ ∂μ < δ := by
    simp_rw [Pi.sub_apply, norm_sub_rev (X 0 _)]
    exact ((tendsto_order.1 (tendsto_integral_norm_approxOn_sub h'.measurable hint)).2 δ
      δpos).exists
  have : ‖μ[Y k 0] - μ[X 0]‖ < δ := by
    rw [norm_sub_rev, ← integral_sub hint]
    · exact (norm_integral_le_integral_norm _).trans_lt hk
    · exact ((φ k).comp (X 0) h'.measurable).integrable_of_isFiniteMeasure
  -- consider `n` large enough for which the above convergences have taken place within `δ`.
  have I : ∀ᶠ n in atTop, (∑ i ∈ range n, ‖(X i - Y k i) ω‖) / n < δ :=
    (tendsto_order.1 (h'ω k)).2 δ hk
  have J : ∀ᶠ (n : ℕ) in atTop, ‖(n : ℝ) ⁻¹ • (∑ i ∈ range n, Y k i ω) - μ[Y k 0]‖ < δ := by
    specialize hω k
    rw [tendsto_iff_norm_sub_tendsto_zero] at hω
    exact (tendsto_order.1 hω).2 δ δpos
  /-
    case h.intro.intro.intro
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    inst✝⁵ : MeasureTheory.IsProbabilityMeasure μ
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    h' : MeasureTheory.StronglyMeasurable (X 0)
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    s : Set E := Union.union (Set.range (X 0)) (Singleton.singleton 0)
    zero_s : Membership.mem s 0
    this✝¹ : TopologicalSpace.SeparableSpace ↑s
    this✝ : Nonempty ↑s
    φ : Nat → MeasureTheory.SimpleFunc E E := MeasureTheory.SimpleFunc.nearestPt f …
    Y : Nat → Nat → Ω → E := fun k i => Function.comp (⇑(φ k)) (X i)
    A : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hS …
    B : Filter.Eventually (fun ω => ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDi …
    ω : Ω
    hω : ∀ (k : Nat), Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.r …
    h'ω : ∀ (k : Nat), Filter.Tendsto (fun n => HDiv.hDiv ((Finset.range n).sum fu …
    ε : Real
    εpos : LT.lt 0 ε
    δ : Real
    δpos : LT.lt 0 δ
    hδ : LT.lt (HAdd.hAdd (HAdd.hAdd δ δ) δ) ε
    k : Nat
    hk : LT.lt (MeasureTheory.integral μ fun ω => Norm.norm (HSub.hSub (X 0) (Y k  …
    this : LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integral μ fun x => Y k 0 x) …
    I : Filter.Eventually (fun n => LT.lt (HDiv.hDiv ((Finset.range n).sum fun i = …
    J : Filter.Eventually (fun n => LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul (Inv. …
    ⊢ Filter.Eventually (fun b => LT.lt (Norm.norm (HSub.hSub (HSMul.hSMul (Inv.in …
  -/
  filter_upwards [I, J] with n hn h'n
  -- at such an `n`, the strong law is realized up to `ε`.
  calc
  ‖(n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, X i ω - μ[X 0]‖
    = ‖(n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, (X i ω - Y k i ω) +
        ((n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, Y k i ω - μ[Y k 0]) + (μ[Y k 0] - μ[X 0])‖ := by
      congr
      simp only [Function.comp_apply, sum_sub_distrib, smul_sub]
      abel
  _ ≤ ‖(n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, (X i ω - Y k i ω)‖ +
        ‖(n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, Y k i ω - μ[Y k 0]‖ + ‖μ[Y k 0] - μ[X 0]‖ :=
      norm_add₃_le
  _ ≤ (∑ i ∈ Finset.range n, ‖X i ω - Y k i ω‖) / n + δ + δ := by
      gcongr
      simp only [Function.comp_apply, norm_smul, norm_inv, RCLike.norm_natCast,
        div_eq_inv_mul, inv_pos, Nat.cast_pos, inv_lt_zero]
      gcongr
      exact norm_sum_le _ _
  _ ≤ δ + δ + δ := by
      gcongr
      exact hn.le
  _ < ε := hδ


omit [IsProbabilityMeasure μ] in
/-- **Strong law of large numbers**, almost sure version: if `X n` is a sequence of independent
identically distributed integrable random variables taking values in a Banach space,
then `n⁻¹ • ∑ i ∈ range n, X i` converges almost surely to `𝔼[X 0]`. We give here the strong
version, due to Etemadi, that only requires pairwise independence. -/
theorem strong_law_ae (X : ℕ → Ω → E) (hint : Integrable (X 0) μ)
    (hindep : Pairwise ((IndepFun · · μ) on X))
    (hident : ∀ i, IdentDistrib (X i) (X 0) μ μ) :
    ∀ᵐ ω ∂μ, Tendsto (fun n : ℕ ↦ (n : ℝ) ⁻¹ • (∑ i ∈ range n, X i ω)) atTop (𝓝 μ[X 0]) := by
  -- First exclude the trivial case where the space is not a probability space
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  by_cases h : ∀ᵐ ω ∂μ, X 0 ω = 0
  · have I : ∀ᵐ ω ∂μ, ∀ i, X i ω = 0 := by
      rw [ae_all_iff]
      intro i
      exact (hident i).symm.ae_snd (p := fun x ↦ x = 0) measurableSet_eq h
    /-
      case pos
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      X : Nat → Ω → E
      hint : MeasureTheory.Integrable (X 0) μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      h : Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ)
      I : Filter.Eventually (fun ω => ∀ (i : Nat), Eq (X i ω) 0) (MeasureTheory.ae μ)
      ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
    -/
    filter_upwards [I] with ω hω
    /-
      case h
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      X : Nat → Ω → E
      hint : MeasureTheory.Integrable (X 0) μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      h : Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ)
      I : Filter.Eventually (fun ω => ∀ (i : Nat), Eq (X i ω) 0) (MeasureTheory.ae μ)
      ω : Ω
      hω : ∀ (i : Nat), Eq (X i ω) 0
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun  …
    -/
    simpa [hω] using (integral_eq_zero_of_ae h).symm
    /-
      🎉 no goals
    -/
  have : IsProbabilityMeasure μ :=
    hint.isProbabilityMeasure_of_indepFun (X 0) (X 1) h (hindep zero_ne_one)
  -- we reduce to the case of strongly measurable random variables, by using `Y i` which is strongly
  -- measurable and ae equal to `X i`.
  /-
    case neg
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this : MeasureTheory.IsProbabilityMeasure μ
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  have A : ∀ i, Integrable (X i) μ := fun i ↦ (hident i).integrable_iff.2 hint
  /-
    case neg
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this : MeasureTheory.IsProbabilityMeasure μ
    A : ∀ (i : Nat), MeasureTheory.Integrable (X i) μ
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  let Y : ℕ → Ω → E := fun i ↦ (A i).1.mk (X i)
  have B : ∀ᵐ ω ∂μ, ∀ n, X n ω = Y n ω :=
    ae_all_iff.2 (fun i ↦ AEStronglyMeasurable.ae_eq_mk (A i).1)
  /-
    case neg
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this : MeasureTheory.IsProbabilityMeasure μ
    A : ∀ (i : Nat), MeasureTheory.Integrable (X i) μ
    Y : Nat → Ω → E := fun i => MeasureTheory.AEStronglyMeasurable.mk (X i) ⋯
    B : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (X n ω) (Y n ω)) (MeasureTheor …
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  have Yint : Integrable (Y 0) μ := Integrable.congr hint (AEStronglyMeasurable.ae_eq_mk (A 0).1)
  have C : ∀ᵐ ω ∂μ,
      Tendsto (fun n : ℕ ↦ (n : ℝ) ⁻¹ • (∑ i ∈ range n, Y i ω)) atTop (𝓝 μ[Y 0]) := by
    apply strong_law_ae_of_measurable Y Yint ((A 0).1.stronglyMeasurable_mk)
      (fun i j hij ↦ IndepFun.ae_eq (hindep hij) (A i).1.ae_eq_mk (A j).1.ae_eq_mk)
      (fun i ↦ ((A i).1.identDistrib_mk.symm.trans (hident i)).trans (A 0).1.identDistrib_mk)
  /-
    case neg
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this : MeasureTheory.IsProbabilityMeasure μ
    A : ∀ (i : Nat), MeasureTheory.Integrable (X i) μ
    Y : Nat → Ω → E := fun i => MeasureTheory.AEStronglyMeasurable.mk (X i) ⋯
    B : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (X n ω) (Y n ω)) (MeasureTheor …
    Yint : MeasureTheory.Integrable (Y 0) μ
    C : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv  …
    ⊢ Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n …
  -/
  filter_upwards [B, C] with ω h₁ h₂
  /-
    case h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this : MeasureTheory.IsProbabilityMeasure μ
    A : ∀ (i : Nat), MeasureTheory.Integrable (X i) μ
    Y : Nat → Ω → E := fun i => MeasureTheory.AEStronglyMeasurable.mk (X i) ⋯
    B : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (X n ω) (Y n ω)) (MeasureTheor …
    Yint : MeasureTheory.Integrable (Y 0) μ
    C : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv  …
    ω : Ω
    h₁ : ∀ (n : Nat), Eq (X n ω) (Y n ω)
    h₂ : Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum f …
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun  …
  -/
  have : μ[X 0] = μ[Y 0] := integral_congr_ae (AEStronglyMeasurable.ae_eq_mk (A 0).1)
  /-
    case h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    A : ∀ (i : Nat), MeasureTheory.Integrable (X i) μ
    Y : Nat → Ω → E := fun i => MeasureTheory.AEStronglyMeasurable.mk (X i) ⋯
    B : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (X n ω) (Y n ω)) (MeasureTheor …
    Yint : MeasureTheory.Integrable (Y 0) μ
    C : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv  …
    ω : Ω
    h₁ : ∀ (n : Nat), Eq (X n ω) (Y n ω)
    h₂ : Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum f …
    this : Eq (MeasureTheory.integral μ fun x => X 0 x) (MeasureTheory.integral μ  …
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun  …
  -/
  rw [this]
  /-
    case h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    A : ∀ (i : Nat), MeasureTheory.Integrable (X i) μ
    Y : Nat → Ω → E := fun i => MeasureTheory.AEStronglyMeasurable.mk (X i) ⋯
    B : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (X n ω) (Y n ω)) (MeasureTheor …
    Yint : MeasureTheory.Integrable (Y 0) μ
    C : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv  …
    ω : Ω
    h₁ : ∀ (n : Nat), Eq (X n ω) (Y n ω)
    h₂ : Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum f …
    this : Eq (MeasureTheory.integral μ fun x => X 0 x) (MeasureTheory.integral μ  …
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun  …
  -/
  apply Tendsto.congr (fun n ↦ ?_) h₂
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    A : ∀ (i : Nat), MeasureTheory.Integrable (X i) μ
    Y : Nat → Ω → E := fun i => MeasureTheory.AEStronglyMeasurable.mk (X i) ⋯
    B : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (X n ω) (Y n ω)) (MeasureTheor …
    Yint : MeasureTheory.Integrable (Y 0) μ
    C : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv  …
    ω : Ω
    h₁ : ∀ (n : Nat), Eq (X n ω) (Y n ω)
    h₂ : Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum f …
    this : Eq (MeasureTheory.integral μ fun x => X 0 x) (MeasureTheory.integral μ  …
    n : Nat
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun i => Y i ω)) (HSMul.h …
  -/
  congr with i
  /-
    case e_a.e_f.h
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    X : Nat → Ω → E
    hint : MeasureTheory.Integrable (X 0) μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this✝ : MeasureTheory.IsProbabilityMeasure μ
    A : ∀ (i : Nat), MeasureTheory.Integrable (X i) μ
    Y : Nat → Ω → E := fun i => MeasureTheory.AEStronglyMeasurable.mk (X i) ⋯
    B : Filter.Eventually (fun ω => ∀ (n : Nat), Eq (X n ω) (Y n ω)) (MeasureTheor …
    Yint : MeasureTheory.Integrable (Y 0) μ
    C : Filter.Eventually (fun ω => Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv  …
    ω : Ω
    h₁ : ∀ (n : Nat), Eq (X n ω) (Y n ω)
    h₂ : Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum f …
    this : Eq (MeasureTheory.integral μ fun x => X 0 x) (MeasureTheory.integral μ  …
    n i : Nat
    ⊢ Eq (Y i ω) (X i ω)
  -/
  exact (h₁ i).symm
  /-
    🎉 no goals
  -/


/-- **Strong law of large numbers**, Lᵖ version: if `X n` is a sequence of independent
identically distributed random variables in Lᵖ, then `n⁻¹ • ∑ i ∈ range n, X i`
converges in `Lᵖ` to `𝔼[X 0]`. -/
theorem strong_law_Lp {p : ℝ≥0∞} (hp : 1 ≤ p) (hp' : p ≠ ∞) (X : ℕ → Ω → E)
    (hℒp : Memℒp (X 0) p μ) (hindep : Pairwise ((IndepFun · · μ) on X))
    (hident : ∀ i, IdentDistrib (X i) (X 0) μ μ) :
    Tendsto (fun (n : ℕ) => eLpNorm (fun ω => (n : ℝ) ⁻¹ • (∑ i ∈ range n, X i ω) - μ[X 0]) p μ)
      atTop (𝓝 0) := by
  -- First exclude the trivial case where the space is not a probability space
  /-
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    X : Nat → Ω → E
    hℒp : MeasureTheory.Memℒp (X 0) p μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (fun ω => HSub.hSub (HSMul.hS …
  -/
  by_cases h : ∀ᵐ ω ∂μ, X 0 ω = 0
  · have I : ∀ᵐ ω ∂μ, ∀ i, X i ω = 0 := by
      rw [ae_all_iff]
      intro i
      exact (hident i).symm.ae_snd (p := fun x ↦ x = 0) measurableSet_eq h
    have A (n : ℕ) : eLpNorm (fun ω => (n : ℝ) ⁻¹ • (∑ i ∈ range n, X i ω) - μ[X 0]) p μ = 0 := by
      simp only [integral_eq_zero_of_ae h, sub_zero]
      apply eLpNorm_eq_zero_of_ae_zero
      filter_upwards [I] with ω hω
      simp [hω]
    /-
      case pos
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      X : Nat → Ω → E
      hℒp : MeasureTheory.Memℒp (X 0) p μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      h : Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ)
      I : Filter.Eventually (fun ω => ∀ (i : Nat), Eq (X i ω) 0) (MeasureTheory.ae μ)
      A : ∀ (n : Nat), Eq (MeasureTheory.eLpNorm (fun ω => HSub.hSub (HSMul.hSMul (I …
      ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (fun ω => HSub.hSub (HSMul.hS …
    -/
    simp [A]
    /-
      🎉 no goals
    -/
  -- Then use ae convergence and uniform integrability
  have : IsProbabilityMeasure μ := Memℒp.isProbabilityMeasure_of_indepFun
    (X 0) (X 1) (zero_lt_one.trans_le hp).ne' hp' hℒp h (hindep zero_ne_one)
  have hmeas : ∀ i, AEStronglyMeasurable (X i) μ := fun i =>
    (hident i).aestronglyMeasurable_iff.2 hℒp.1
  /-
    case neg
    Ω : Type u_1
    mΩ : MeasurableSpace Ω
    μ : MeasureTheory.Measure Ω
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : CompleteSpace E
    inst✝¹ : MeasurableSpace E
    inst✝ : BorelSpace E
    p : ENNReal
    hp : LE.le 1 p
    hp' : Ne p Top.top
    X : Nat → Ω → E
    hℒp : MeasureTheory.Memℒp (X 0) p μ
    hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
    hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
    h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
    this : MeasureTheory.IsProbabilityMeasure μ
    hmeas : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.eLpNorm (fun ω => HSub.hSub (HSMul.hS …
  -/
  have hint : Integrable (X 0) μ := hℒp.integrable hp
  have havg (n : ℕ) :
      AEStronglyMeasurable (fun ω => (n : ℝ) ⁻¹ • (∑ i ∈ range n, X i ω)) μ :=
    AEStronglyMeasurable.const_smul (aestronglyMeasurable_sum _ fun i _ => hmeas i) _
  refine tendsto_Lp_finite_of_tendstoInMeasure hp hp' havg (memℒp_const _) ?_
    (tendstoInMeasure_of_tendsto_ae havg (strong_law_ae _ hint hindep hident))
  rw [(_ : (fun (n : ℕ) ω => (n : ℝ)⁻¹ • (∑ i ∈ range n, X i ω))
            = fun (n : ℕ) => (n : ℝ)⁻¹ • (∑ i ∈ range n, X i))]
    /-
      case neg
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      X : Nat → Ω → E
      hℒp : MeasureTheory.Memℒp (X 0) p μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
      this : MeasureTheory.IsProbabilityMeasure μ
      hmeas : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) μ
      hint : MeasureTheory.Integrable (X 0) μ
      havg : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fun ω => HSMul.hSMul ( …
      ⊢ MeasureTheory.UnifIntegrable (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.ran …
    -/
  · apply UniformIntegrable.unifIntegrable
    /-
      case neg.hf
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      X : Nat → Ω → E
      hℒp : MeasureTheory.Memℒp (X 0) p μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
      this : MeasureTheory.IsProbabilityMeasure μ
      hmeas : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) μ
      hint : MeasureTheory.Integrable (X 0) μ
      havg : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fun ω => HSMul.hSMul ( …
      ⊢ MeasureTheory.UniformIntegrable (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset. …
    -/
    apply uniformIntegrable_average hp
    /-
      case neg.hf
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      X : Nat → Ω → E
      hℒp : MeasureTheory.Memℒp (X 0) p μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
      this : MeasureTheory.IsProbabilityMeasure μ
      hmeas : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) μ
      hint : MeasureTheory.Integrable (X 0) μ
      havg : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fun ω => HSMul.hSMul ( …
      ⊢ MeasureTheory.UniformIntegrable X p μ
    -/
    exact Memℒp.uniformIntegrable_of_identDistrib hp hp' hℒp hident
    /-
      🎉 no goals
    -/
    /-
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      X : Nat → Ω → E
      hℒp : MeasureTheory.Memℒp (X 0) p μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
      this : MeasureTheory.IsProbabilityMeasure μ
      hmeas : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) μ
      hint : MeasureTheory.Integrable (X 0) μ
      havg : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fun ω => HSMul.hSMul ( …
      ⊢ Eq (fun n ω => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun i => X i ω …
    -/
  · ext n ω
    /-
      case h.h
      Ω : Type u_1
      mΩ : MeasurableSpace Ω
      μ : MeasureTheory.Measure Ω
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : CompleteSpace E
      inst✝¹ : MeasurableSpace E
      inst✝ : BorelSpace E
      p : ENNReal
      hp : LE.le 1 p
      hp' : Ne p Top.top
      X : Nat → Ω → E
      hℒp : MeasureTheory.Memℒp (X 0) p μ
      hindep : Pairwise (Function.onFun (fun x1 x2 => ProbabilityTheory.IndepFun x1  …
      hident : ∀ (i : Nat), ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ
      h : Not (Filter.Eventually (fun ω => Eq (X 0 ω) 0) (MeasureTheory.ae μ))
      this : MeasureTheory.IsProbabilityMeasure μ
      hmeas : ∀ (i : Nat), MeasureTheory.AEStronglyMeasurable (X i) μ
      hint : MeasureTheory.Integrable (X 0) μ
      havg : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (fun ω => HSMul.hSMul ( …
      n : Nat
      ω : Ω
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun i => X i ω)) (HSMul.h …
    -/
    simp only [Pi.smul_apply, sum_apply]
    /-
      🎉 no goals
    -/


