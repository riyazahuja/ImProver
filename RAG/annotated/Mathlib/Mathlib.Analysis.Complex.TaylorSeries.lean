include hf hz in
/-- A function that is complex differentiable on the open ball of radius `r` around `c`
is given by evaluating its Taylor series at `c` on this open ball. -/
lemma hasSum_taylorSeries_on_ball :
    HasSum (fun n : ℕ ↦ (n ! : ℂ)⁻¹ • (z - c) ^ n • iteratedDeriv n f c) (f z) := by
  obtain ⟨r', hr', hr'₀, hzr'⟩ : ∃ r' < r, 0 < r' ∧ z ∈ Metric.ball c r' := by
    obtain ⟨r', h₁, h₂⟩ := exists_between (Metric.mem_ball'.mp hz)
    exact ⟨r', h₂, Metric.pos_of_mem_ball h₁, Metric.mem_ball'.mpr h₁⟩
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    z : Complex
    hz : Membership.mem (Metric.ball c r) z
    r' : Real
    hr' : LT.lt r' r
    hr'₀ : LT.lt 0 r'
    hzr' : Membership.mem (Metric.ball c r') z
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HSMul.hSMul (HPow.hPow  …
  -/
  lift r' to NNReal using hr'₀.le
  have hz' : z - c ∈ EMetric.ball 0 r' := by
    rw [Metric.emetric_ball_nnreal]
    exact mem_ball_zero_iff.mpr hzr'
  have H := (hf.mono <| Metric.closedBall_subset_ball hr').hasFPowerSeriesOnBall hr'₀
      |>.hasSum_iteratedFDeriv hz'
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    z : Complex
    hz : Membership.mem (Metric.ball c r) z
    r' : NNReal
    hr' : LT.lt (↑r') r
    hr'₀ : LT.lt 0 ↑r'
    hzr' : Membership.mem (Metric.ball c ↑r') z
    hz' : Membership.mem (EMetric.ball 0 ↑r') (HSub.hSub z c)
    H : HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) ((iteratedFDeriv Compl …
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HSMul.hSMul (HPow.hPow  …
  -/
  simp only [add_sub_cancel] at H
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : Real
    hf : DifferentiableOn Complex f (Metric.ball c r)
    z : Complex
    hz : Membership.mem (Metric.ball c r) z
    r' : NNReal
    hr' : LT.lt (↑r') r
    hr'₀ : LT.lt 0 ↑r'
    hzr' : Membership.mem (Metric.ball c ↑r') z
    hz' : Membership.mem (EMetric.ball 0 ↑r') (HSub.hSub z c)
    H : HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) ((iteratedFDeriv Compl …
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HSMul.hSMul (HPow.hPow  …
  -/
  convert H using 4 with n
  simpa only [iteratedDeriv_eq_iteratedFDeriv, smul_eq_mul, mul_one, Finset.prod_const,
    Finset.card_fin]
    using ((iteratedFDeriv ℂ n f c).map_smul_univ (fun _ ↦ z - c) (fun _ ↦ 1)).symm


include hf hz in
/-- A function that is complex differentiable on the open ball of radius `r` around `c`
is given by evaluating its Taylor series at `c` on this open ball. -/
lemma taylorSeries_eq_on_ball :
    ∑' n : ℕ, (n ! : ℂ)⁻¹ • (z - c) ^ n • iteratedDeriv n f c = f z :=
  (hasSum_taylorSeries_on_ball hf hz).tsum_eq


include hz in
/-- A function that is complex differentiable on the open ball of radius `r` around `c`
is given by evaluating its Taylor series at `c` on this open ball. -/
lemma taylorSeries_eq_on_ball' {f : ℂ → ℂ} (hf : DifferentiableOn ℂ f (Metric.ball c r)) :
    ∑' n : ℕ, (n ! : ℂ)⁻¹ * iteratedDeriv n f c * (z - c) ^ n = f z := by
  /-
    c : Complex
    r : Real
    z : Complex
    hz : Membership.mem (Metric.ball c r) z
    f : Complex → Complex
    hf : DifferentiableOn Complex f (Metric.ball c r)
    ⊢ Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv …
  -/
  convert taylorSeries_eq_on_ball hf hz using 3 with n
  /-
    case h.e'_2.h.e'_5.h
    c : Complex
    r : Real
    z : Complex
    hz : Membership.mem (Metric.ball c r) z
    f : Complex → Complex
    hf : DifferentiableOn Complex f (Metric.ball c r)
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv n f c)) (HPow …
  -/
  rw [mul_right_comm, smul_eq_mul, smul_eq_mul, mul_assoc]
  /-
    🎉 no goals
  -/


include hf hz in
/-- A function that is complex differentiable on the open ball of radius `r ≤ ∞` around `c`
is given by evaluating its Taylor series at `c` on this open ball. -/
lemma hasSum_taylorSeries_on_emetric_ball :
    HasSum (fun n : ℕ ↦ (n ! : ℂ)⁻¹ • (z - c) ^ n • iteratedDeriv n f c) (f z) := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : ENNReal
    hf : DifferentiableOn Complex f (EMetric.ball c r)
    z : Complex
    hz : Membership.mem (EMetric.ball c r) z
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HSMul.hSMul (HPow.hPow  …
  -/
  obtain ⟨r', hzr', hr'⟩ := exists_between (EMetric.mem_ball'.mp hz)
  /-
    case intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : ENNReal
    hf : DifferentiableOn Complex f (EMetric.ball c r)
    z : Complex
    hz : Membership.mem (EMetric.ball c r) z
    r' : ENNReal
    hzr' : LT.lt (EDist.edist c z) r'
    hr' : LT.lt r' r
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HSMul.hSMul (HPow.hPow  …
  -/
  lift r' to NNReal using ne_top_of_lt hr'
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : ENNReal
    hf : DifferentiableOn Complex f (EMetric.ball c r)
    z : Complex
    hz : Membership.mem (EMetric.ball c r) z
    r' : NNReal
    hzr' : LT.lt (EDist.edist c z) ↑r'
    hr' : LT.lt (↑r') r
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HSMul.hSMul (HPow.hPow  …
  -/
  rw [← EMetric.mem_ball', Metric.emetric_ball_nnreal] at hzr'
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : ENNReal
    hf : DifferentiableOn Complex f (EMetric.ball c r)
    z : Complex
    hz : Membership.mem (EMetric.ball c r) z
    r' : NNReal
    hzr' : Membership.mem (Metric.ball c ↑r') z
    hr' : LT.lt (↑r') r
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) (HSMul.hSMul (HPow.hPow  …
  -/
  refine hasSum_taylorSeries_on_ball ?_ hzr'
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : ENNReal
    hf : DifferentiableOn Complex f (EMetric.ball c r)
    z : Complex
    hz : Membership.mem (EMetric.ball c r) z
    r' : NNReal
    hzr' : Membership.mem (Metric.ball c ↑r') z
    hr' : LT.lt (↑r') r
    ⊢ DifferentiableOn Complex f (Metric.ball c ↑r')
  -/
  rw [← Metric.emetric_ball_nnreal]
  /-
    case intro.intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    f : Complex → E
    c : Complex
    r : ENNReal
    hf : DifferentiableOn Complex f (EMetric.ball c r)
    z : Complex
    hz : Membership.mem (EMetric.ball c r) z
    r' : NNReal
    hzr' : Membership.mem (Metric.ball c ↑r') z
    hr' : LT.lt (↑r') r
    ⊢ DifferentiableOn Complex f (EMetric.ball c ↑r')
  -/
  exact hf.mono <| EMetric.ball_subset_ball hr'.le
  /-
    🎉 no goals
  -/


include hf hz in
/-- A function that is complex differentiable on the open ball of radius `r ≤ ∞` around `c`
is given by evaluating its Taylor series at `c` on this open ball. -/
lemma taylorSeries_eq_on_emetric_ball :
    ∑' n : ℕ, (n ! : ℂ)⁻¹ • (z - c) ^ n • iteratedDeriv n f c = f z :=
  (hasSum_taylorSeries_on_emetric_ball hf hz).tsum_eq


include hz in
/-- A function that is complex differentiable on the open ball of radius `r ≤ ∞` around `c`
is given by evaluating its Taylor series at `c` on this open ball. -/
lemma taylorSeries_eq_on_emetric_ball' {f : ℂ → ℂ} (hf : DifferentiableOn ℂ f (EMetric.ball c r)) :
    ∑' n : ℕ, (n ! : ℂ)⁻¹ * iteratedDeriv n f c * (z - c) ^ n = f z := by
  /-
    c : Complex
    r : ENNReal
    z : Complex
    hz : Membership.mem (EMetric.ball c r) z
    f : Complex → Complex
    hf : DifferentiableOn Complex f (EMetric.ball c r)
    ⊢ Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv …
  -/
  convert taylorSeries_eq_on_emetric_ball hf hz using 3 with n
  /-
    case h.e'_2.h.e'_5.h
    c : Complex
    r : ENNReal
    z : Complex
    hz : Membership.mem (EMetric.ball c r) z
    f : Complex → Complex
    hf : DifferentiableOn Complex f (EMetric.ball c r)
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv n f c)) (HPow …
  -/
  rw [mul_right_comm, smul_eq_mul, smul_eq_mul, mul_assoc]
  /-
    🎉 no goals
  -/


include hf in
/-- A function that is complex differentiable on the complex plane is given by evaluating
its Taylor series at any point `c`. -/
lemma hasSum_taylorSeries_of_entire :
    HasSum (fun n : ℕ ↦ (n ! : ℂ)⁻¹ • (z - c) ^ n • iteratedDeriv n f c) (f z) :=
  hasSum_taylorSeries_on_emetric_ball hf.differentiableOn <| EMetric.mem_ball.mpr <|
    edist_lt_top ..


include hf in
/-- A function that is complex differentiable on the complex plane is given by evaluating
its Taylor series at any point `c`. -/
lemma taylorSeries_eq_of_entire :
    ∑' n : ℕ, (n ! : ℂ)⁻¹ • (z - c) ^ n • iteratedDeriv n f c = f z :=
  (hasSum_taylorSeries_of_entire hf c z).tsum_eq


/-- A function that is complex differentiable on the complex plane is given by evaluating
its Taylor series at any point `c`. -/
lemma taylorSeries_eq_of_entire' {f : ℂ → ℂ} (hf : Differentiable ℂ f) :
    ∑' n : ℕ, (n ! : ℂ)⁻¹ * iteratedDeriv n f c * (z - c) ^ n = f z := by
  /-
    c z : Complex
    f : Complex → Complex
    hf : Differentiable Complex f
    ⊢ Eq (tsum fun n => HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv …
  -/
  convert taylorSeries_eq_of_entire hf c z using 3 with n
  /-
    case h.e'_2.h.e'_5.h
    c z : Complex
    f : Complex → Complex
    hf : Differentiable Complex f
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv ↑n.factorial) (iteratedDeriv n f c)) (HPow …
  -/
  rw [mul_right_comm, smul_eq_mul, smul_eq_mul, mul_assoc]
  /-
    🎉 no goals
  -/


