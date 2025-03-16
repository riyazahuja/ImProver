/-- The pmf of the geometric distribution depending on its success probability. -/
noncomputable
def geometricPMFReal (p : ℝ) (n : ℕ) : ℝ := (1-p) ^ n * p


lemma geometricPMFRealSum (hp_pos : 0 < p) (hp_le_one : p ≤ 1) :
    HasSum (fun n ↦ geometricPMFReal p n) 1 := by
  /-
    p : Real
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    ⊢ HasSum (fun n => ProbabilityTheory.geometricPMFReal p n) 1
  -/
  unfold geometricPMFReal
  /-
    p : Real
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    ⊢ HasSum (fun n => HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p) 1
  -/
  have := hasSum_geometric_of_lt_one (sub_nonneg.mpr hp_le_one) (sub_lt_self 1 hp_pos)
  /-
    p : Real
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    this : HasSum (fun n => HPow.hPow (HSub.hSub 1 p) n) (Inv.inv (HSub.hSub 1 (HS …
    ⊢ HasSum (fun n => HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p) 1
  -/
  apply (hasSum_mul_right_iff (hp_pos.ne')).mpr at this
  /-
    p : Real
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    this : HasSum (fun i => HMul.hMul (HPow.hPow (HSub.hSub 1 p) i) p) (HMul.hMul  …
    ⊢ HasSum (fun n => HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p) 1
  -/
  simp only [sub_sub_cancel] at this
  /-
    p : Real
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    this : HasSum (fun i => HMul.hMul (HPow.hPow (HSub.hSub 1 p) i) p) (HMul.hMul  …
    ⊢ HasSum (fun n => HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p) 1
  -/
  rw [inv_mul_eq_div, div_self hp_pos.ne'] at this
  /-
    p : Real
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    this : HasSum (fun i => HMul.hMul (HPow.hPow (HSub.hSub 1 p) i) p) 1
    ⊢ HasSum (fun n => HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p) 1
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- The geometric pmf is positive for all natural numbers -/
lemma geometricPMFReal_pos {n : ℕ} (hp_pos : 0 < p) (hp_lt_one : p < 1) :
    0 < geometricPMFReal p n := by
  /-
    p : Real
    n : Nat
    hp_pos : LT.lt 0 p
    hp_lt_one : LT.lt p 1
    ⊢ LT.lt 0 (ProbabilityTheory.geometricPMFReal p n)
  -/
  rw [geometricPMFReal]
  /-
    p : Real
    n : Nat
    hp_pos : LT.lt 0 p
    hp_lt_one : LT.lt p 1
    ⊢ LT.lt 0 (HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p)
  -/
  have : 0 < 1 - p := sub_pos.mpr hp_lt_one
  /-
    p : Real
    n : Nat
    hp_pos : LT.lt 0 p
    hp_lt_one : LT.lt p 1
    this : LT.lt 0 (HSub.hSub 1 p)
    ⊢ LT.lt 0 (HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p)
  -/
  positivity
  /-
    🎉 no goals
  -/


lemma geometricPMFReal_nonneg {n : ℕ} (hp_pos : 0 < p) (hp_le_one : p ≤ 1) :
    0 ≤ geometricPMFReal p n := by
  /-
    p : Real
    n : Nat
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    ⊢ LE.le 0 (ProbabilityTheory.geometricPMFReal p n)
  -/
  rw [geometricPMFReal]
  /-
    p : Real
    n : Nat
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    ⊢ LE.le 0 (HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p)
  -/
  have : 0 ≤ 1 - p := sub_nonneg.mpr hp_le_one
  /-
    p : Real
    n : Nat
    hp_pos : LT.lt 0 p
    hp_le_one : LE.le p 1
    this : LE.le 0 (HSub.hSub 1 p)
    ⊢ LE.le 0 (HMul.hMul (HPow.hPow (HSub.hSub 1 p) n) p)
  -/
  positivity
  /-
    🎉 no goals
  -/


/-- Geometric distribution with success probability `p`. -/
noncomputable
def geometricPMF (hp_pos : 0 < p) (hp_le_one : p ≤ 1) : PMF ℕ :=
  ⟨fun n ↦ ENNReal.ofReal (geometricPMFReal p n), by
    /-
      p : Real
      hp_pos : LT.lt 0 p
      hp_le_one : LE.le p 1
      ⊢ HasSum (fun n => ENNReal.ofReal (ProbabilityTheory.geometricPMFReal p n)) 1
    -/
    apply ENNReal.hasSum_coe.mpr
    /-
      p : Real
      hp_pos : LT.lt 0 p
      hp_le_one : LE.le p 1
      ⊢ HasSum (fun a => (ProbabilityTheory.geometricPMFReal p a).toNNReal) 1
    -/
    rw [← toNNReal_one]
    exact (geometricPMFRealSum hp_pos hp_le_one).toNNReal
      (fun n ↦ geometricPMFReal_nonneg hp_pos hp_le_one)⟩


/-- The geometric pmf is measurable. -/
@[measurability]
lemma measurable_geometricPMFReal : Measurable (geometricPMFReal p) := by
  /-
    p : Real
    ⊢ Measurable (ProbabilityTheory.geometricPMFReal p)
  -/
  measurability
  /-
    🎉 no goals
  -/


@[measurability]
lemma stronglyMeasurable_geometricPMFReal : StronglyMeasurable (geometricPMFReal p) :=
  stronglyMeasurable_iff_measurable.mpr measurable_geometricPMFReal


/-- Measure defined by the geometric distribution -/
noncomputable
def geometricMeasure (hp_pos : 0 < p) (hp_le_one : p ≤ 1) : Measure ℕ :=
  (geometricPMF hp_pos hp_le_one).toMeasure


lemma isProbabilityMeasureGeometric (hp_pos : 0 < p) (hp_le_one : p ≤ 1) :
    IsProbabilityMeasure (geometricMeasure hp_pos hp_le_one) :=
  PMF.toMeasure.isProbabilityMeasure (geometricPMF hp_pos hp_le_one)


