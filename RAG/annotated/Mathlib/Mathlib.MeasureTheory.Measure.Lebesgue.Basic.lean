/-- The volume on the real line (as a particular case of the volume on a finite-dimensional
inner product space) coincides with the Stieltjes measure coming from the identity function. -/
theorem volume_eq_stieltjes_id : (volume : Measure ℝ) = StieltjesFunction.id.measure := by
  haveI : IsAddLeftInvariant StieltjesFunction.id.measure :=
    ⟨fun a =>
      Eq.symm <|
        Real.measure_ext_Ioo_rat fun p q => by
          simp only [Measure.map_apply (measurable_const_add a) measurableSet_Ioo,
            sub_sub_sub_cancel_right, StieltjesFunction.measure_Ioo, StieltjesFunction.id_leftLim,
            StieltjesFunction.id_apply, id, preimage_const_add_Ioo]⟩
  have A : StieltjesFunction.id.measure (stdOrthonormalBasis ℝ ℝ).toBasis.parallelepiped = 1 := by
    change StieltjesFunction.id.measure (parallelepiped (stdOrthonormalBasis ℝ ℝ)) = 1
    rcases parallelepiped_orthonormalBasis_one_dim (stdOrthonormalBasis ℝ ℝ) with (H | H) <;>
      simp only [H, StieltjesFunction.measure_Icc, StieltjesFunction.id_apply, id, tsub_zero,
        StieltjesFunction.id_leftLim, sub_neg_eq_add, zero_add, ENNReal.ofReal_one]
  conv_rhs =>
    rw [addHaarMeasure_unique StieltjesFunction.id.measure
        (stdOrthonormalBasis ℝ ℝ).toBasis.parallelepiped, A]
  /-
    this : StieltjesFunction.id.measure.IsAddLeftInvariant
    A : Eq (StieltjesFunction.id.measure ↑(stdOrthonormalBasis Real Real).toBasis. …
    ⊢ Eq MeasureTheory.MeasureSpace.volume (HSMul.hSMul 1 (MeasureTheory.Measure.a …
  -/
  simp only [volume, Basis.addHaar, one_smul]
  /-
    🎉 no goals
  -/


theorem volume_val (s) : volume s = StieltjesFunction.id.measure s := by
  /-
    s : Set Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume s) (StieltjesFunction.id.measure s)
  -/
  simp [volume_eq_stieltjes_id]
  /-
    🎉 no goals
  -/


@[simp]
                                                                       /-
                                                                         a b : Real
                                                                         ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Ico a b)) (ENNReal.ofReal (HSub.h …
                                                                       -/
theorem volume_Ico {a b : ℝ} : volume (Ico a b) = ofReal (b - a) := by simp [volume_val]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
                                                                       /-
                                                                         a b : Real
                                                                         ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Icc a b)) (ENNReal.ofReal (HSub.h …
                                                                       -/
theorem volume_Icc {a b : ℝ} : volume (Icc a b) = ofReal (b - a) := by simp [volume_val]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
                                                                       /-
                                                                         a b : Real
                                                                         ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Ioo a b)) (ENNReal.ofReal (HSub.h …
                                                                       -/
theorem volume_Ioo {a b : ℝ} : volume (Ioo a b) = ofReal (b - a) := by simp [volume_val]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp]
                                                                       /-
                                                                         a b : Real
                                                                         ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Ioc a b)) (ENNReal.ofReal (HSub.h …
                                                                       -/
theorem volume_Ioc {a b : ℝ} : volume (Ioc a b) = ofReal (b - a) := by simp [volume_val]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


                                                                  /-
                                                                    a : Real
                                                                    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Singleton.singleton a)) 0
                                                                  -/
theorem volume_singleton {a : ℝ} : volume ({a} : Set ℝ) = 0 := by simp [volume_val]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem volume_univ : volume (univ : Set ℝ) = ∞ :=
  ENNReal.eq_top_of_forall_nnreal_le fun r =>
    calc
                                                /-
                                                  r : NNReal
                                                  ⊢ Eq (↑r) (MeasureTheory.MeasureSpace.volume (Set.Icc 0 ↑r))
                                                -/
      (r : ℝ≥0∞) = volume (Icc (0 : ℝ) r) := by simp
                                                /-
                                                  🎉 no goals
                                                -/
      _ ≤ volume univ := measure_mono (subset_univ _)


@[simp]
theorem volume_ball (a r : ℝ) : volume (Metric.ball a r) = ofReal (2 * r) := by
  /-
    a r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.ball a r)) (ENNReal.ofReal (HM …
  -/
  rw [ball_eq_Ioo, volume_Ioo, ← sub_add, add_sub_cancel_left, two_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem volume_closedBall (a r : ℝ) : volume (Metric.closedBall a r) = ofReal (2 * r) := by
  /-
    a r : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall a r)) (ENNReal.ofRe …
  -/
  rw [closedBall_eq_Icc, volume_Icc, ← sub_add, add_sub_cancel_left, two_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem volume_emetric_ball (a : ℝ) (r : ℝ≥0∞) : volume (EMetric.ball a r) = 2 * r := by
  /-
    a : Real
    r : ENNReal
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (EMetric.ball a r)) (HMul.hMul 2 r)
  -/
  rcases eq_or_ne r ∞ with (rfl | hr)
    /-
      case inl
      a : Real
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (EMetric.ball a Top.top)) (HMul.hMul 2 …
    -/
  · rw [Metric.emetric_ball_top, volume_univ, two_mul, _root_.top_add]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a : Real
      r : ENNReal
      hr : Ne r Top.top
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (EMetric.ball a r)) (HMul.hMul 2 r)
    -/
  · lift r to ℝ≥0 using hr
    rw [Metric.emetric_ball_nnreal, volume_ball, two_mul, ← NNReal.coe_add,
      ENNReal.ofReal_coe_nnreal, ENNReal.coe_add, two_mul]


@[simp]
theorem volume_emetric_closedBall (a : ℝ) (r : ℝ≥0∞) : volume (EMetric.closedBall a r) = 2 * r := by
  /-
    a : Real
    r : ENNReal
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (EMetric.closedBall a r)) (HMul.hMul 2 …
  -/
  rcases eq_or_ne r ∞ with (rfl | hr)
    /-
      case inl
      a : Real
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (EMetric.closedBall a Top.top)) (HMul. …
    -/
  · rw [EMetric.closedBall_top, volume_univ, two_mul, _root_.top_add]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a : Real
      r : ENNReal
      hr : Ne r Top.top
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (EMetric.closedBall a r)) (HMul.hMul 2 …
    -/
  · lift r to ℝ≥0 using hr
    rw [Metric.emetric_closedBall_nnreal, volume_closedBall, two_mul, ← NNReal.coe_add,
      ENNReal.ofReal_coe_nnreal, ENNReal.coe_add, two_mul]


instance noAtoms_volume : NoAtoms (volume : Measure ℝ) :=
  ⟨fun _ => volume_singleton⟩


@[simp]
theorem volume_interval {a b : ℝ} : volume (uIcc a b) = ofReal |b - a| := by
  /-
    a b : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.uIcc a b)) (ENNReal.ofReal (abs ( …
  -/
  rw [← Icc_min_max, volume_Icc, max_sub_min_eq_abs]
  /-
    🎉 no goals
  -/


@[simp]
theorem volume_Ioi {a : ℝ} : volume (Ioi a) = ∞ :=
  top_unique <|
    le_of_tendsto' ENNReal.tendsto_nat_nhds_top fun n =>
      calc
                                                  /-
                                                    a : Real
                                                    n : Nat
                                                    ⊢ Eq (↑n) (MeasureTheory.MeasureSpace.volume (Set.Ioo a (HAdd.hAdd a ↑n)))
                                                  -/
        (n : ℝ≥0∞) = volume (Ioo a (a + n)) := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/
        _ ≤ volume (Ioi a) := measure_mono Ioo_subset_Ioi_self


@[simp]
                                                      /-
                                                        a : Real
                                                        ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Ici a)) Top.top
                                                      -/
theorem volume_Ici {a : ℝ} : volume (Ici a) = ∞ := by rw [← measure_congr Ioi_ae_eq_Ici]; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
theorem volume_Iio {a : ℝ} : volume (Iio a) = ∞ :=
  top_unique <|
    le_of_tendsto' ENNReal.tendsto_nat_nhds_top fun n =>
      calc
                                                  /-
                                                    a : Real
                                                    n : Nat
                                                    ⊢ Eq (↑n) (MeasureTheory.MeasureSpace.volume (Set.Ioo (HSub.hSub a ↑n) a))
                                                  -/
        (n : ℝ≥0∞) = volume (Ioo (a - n) a) := by simp
                                                  /-
                                                    🎉 no goals
                                                  -/
        _ ≤ volume (Iio a) := measure_mono Ioo_subset_Iio_self


@[simp]
                                                      /-
                                                        a : Real
                                                        ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Iic a)) Top.top
                                                      -/
theorem volume_Iic {a : ℝ} : volume (Iic a) = ∞ := by rw [← measure_congr Iio_ae_eq_Iic]; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


instance locallyFinite_volume : IsLocallyFiniteMeasure (volume : Measure ℝ) :=
  ⟨fun x =>
    ⟨Ioo (x - 1) (x + 1),
      IsOpen.mem_nhds isOpen_Ioo ⟨sub_lt_self _ zero_lt_one, lt_add_of_pos_right _ zero_lt_one⟩, by
      /-
        ι : Type u_1
        inst✝ : Fintype ι
        x : Real
        ⊢ LT.lt (MeasureTheory.MeasureSpace.volume (Set.Ioo (HSub.hSub x 1) (HAdd.hAdd …
      -/
      simp only [Real.volume_Ioo, ENNReal.ofReal_lt_top]⟩⟩
      /-
        🎉 no goals
      -/


instance isFiniteMeasure_restrict_Icc (x y : ℝ) : IsFiniteMeasure (volume.restrict (Icc x y)) :=
      /-
        ι : Type u_1
        inst✝ : Fintype ι
        x y : Real
        ⊢ LT.lt ((MeasureTheory.MeasureSpace.volume.restrict (Set.Icc x y)) Set.univ)  …
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


instance isFiniteMeasure_restrict_Ico (x y : ℝ) : IsFiniteMeasure (volume.restrict (Ico x y)) :=
      /-
        ι : Type u_1
        inst✝ : Fintype ι
        x y : Real
        ⊢ LT.lt ((MeasureTheory.MeasureSpace.volume.restrict (Set.Ico x y)) Set.univ)  …
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


instance isFiniteMeasure_restrict_Ioc (x y : ℝ) : IsFiniteMeasure (volume.restrict (Ioc x y)) :=
      /-
        ι : Type u_1
        inst✝ : Fintype ι
        x y : Real
        ⊢ LT.lt ((MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc x y)) Set.univ)  …
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


instance isFiniteMeasure_restrict_Ioo (x y : ℝ) : IsFiniteMeasure (volume.restrict (Ioo x y)) :=
      /-
        ι : Type u_1
        inst✝ : Fintype ι
        x y : Real
        ⊢ LT.lt ((MeasureTheory.MeasureSpace.volume.restrict (Set.Ioo x y)) Set.univ)  …
      -/
  ⟨by simp⟩
      /-
        🎉 no goals
      -/


theorem volume_le_diam (s : Set ℝ) : volume s ≤ EMetric.diam s := by
  /-
    s : Set Real
    ⊢ LE.le (MeasureTheory.MeasureSpace.volume s) (EMetric.diam s)
  -/
  by_cases hs : Bornology.IsBounded s
    /-
      case pos
      s : Set Real
      hs : Bornology.IsBounded s
      ⊢ LE.le (MeasureTheory.MeasureSpace.volume s) (EMetric.diam s)
    -/
  · rw [Real.ediam_eq hs, ← volume_Icc]
    /-
      case pos
      s : Set Real
      hs : Bornology.IsBounded s
      ⊢ LE.le (MeasureTheory.MeasureSpace.volume s) (MeasureTheory.MeasureSpace.volu …
    -/
    exact volume.mono hs.subset_Icc_sInf_sSup
    /-
      🎉 no goals
    -/
    /-
      case neg
      s : Set Real
      hs : Not (Bornology.IsBounded s)
      ⊢ LE.le (MeasureTheory.MeasureSpace.volume s) (EMetric.diam s)
    -/
  · rw [Metric.ediam_of_unbounded hs]; exact le_top
                                       /-
                                         🎉 no goals
                                       -/


theorem _root_.Filter.Eventually.volume_pos_of_nhds_real {p : ℝ → Prop} {a : ℝ}
    (h : ∀ᶠ x in 𝓝 a, p x) : (0 : ℝ≥0∞) < volume { x | p x } := by
  /-
    p : Real → Prop
    a : Real
    h : Filter.Eventually (fun x => p x) (nhds a)
    ⊢ LT.lt 0 (MeasureTheory.MeasureSpace.volume (setOf fun x => p x))
  -/
  rcases h.exists_Ioo_subset with ⟨l, u, hx, hs⟩
  /-
    case intro.intro.intro
    p : Real → Prop
    a : Real
    h : Filter.Eventually (fun x => p x) (nhds a)
    l u : Real
    hx : Membership.mem (Set.Ioo l u) a
    hs : HasSubset.Subset (Set.Ioo l u) (setOf fun x => p x)
    ⊢ LT.lt 0 (MeasureTheory.MeasureSpace.volume (setOf fun x => p x))
  -/
  refine lt_of_lt_of_le ?_ (measure_mono hs)
  /-
    case intro.intro.intro
    p : Real → Prop
    a : Real
    h : Filter.Eventually (fun x => p x) (nhds a)
    l u : Real
    hx : Membership.mem (Set.Ioo l u) a
    hs : HasSubset.Subset (Set.Ioo l u) (setOf fun x => p x)
    ⊢ LT.lt 0 (MeasureTheory.MeasureSpace.volume (Set.Ioo l u))
  -/
  simpa [-mem_Ioo] using hx.1.trans hx.2
  /-
    🎉 no goals
  -/


theorem volume_Icc_pi {a b : ι → ℝ} : volume (Icc a b) = ∏ i, ENNReal.ofReal (b i - a i) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a b : ι → Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Icc a b)) (Finset.univ.prod fun i …
  -/
  rw [← pi_univ_Icc, volume_pi_pi]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a b : ι → Real
    ⊢ Eq (Finset.univ.prod fun i => MeasureTheory.MeasureSpace.volume (Set.Icc (a  …
  -/
  simp only [Real.volume_Icc]
  /-
    🎉 no goals
  -/


@[simp]
theorem volume_Icc_pi_toReal {a b : ι → ℝ} (h : a ≤ b) :
    (volume (Icc a b)).toReal = ∏ i, (b i - a i) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a b : ι → Real
    h : LE.le a b
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Icc a b)).toReal (Finset.univ.pro …
  -/
  simp only [volume_Icc_pi, ENNReal.toReal_prod, ENNReal.toReal_ofReal (sub_nonneg.2 (h _))]
  /-
    🎉 no goals
  -/


theorem volume_pi_Ioo {a b : ι → ℝ} :
    volume (pi univ fun i => Ioo (a i) (b i)) = ∏ i, ENNReal.ofReal (b i - a i) :=
  (measure_congr Measure.univ_pi_Ioo_ae_eq_Icc).trans volume_Icc_pi


@[simp]
theorem volume_pi_Ioo_toReal {a b : ι → ℝ} (h : a ≤ b) :
    (volume (pi univ fun i => Ioo (a i) (b i))).toReal = ∏ i, (b i - a i) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a b : ι → Real
    h : LE.le a b
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.univ.pi fun i => Set.Ioo (a i) (b …
  -/
  simp only [volume_pi_Ioo, ENNReal.toReal_prod, ENNReal.toReal_ofReal (sub_nonneg.2 (h _))]
  /-
    🎉 no goals
  -/


theorem volume_pi_Ioc {a b : ι → ℝ} :
    volume (pi univ fun i => Ioc (a i) (b i)) = ∏ i, ENNReal.ofReal (b i - a i) :=
  (measure_congr Measure.univ_pi_Ioc_ae_eq_Icc).trans volume_Icc_pi


@[simp]
theorem volume_pi_Ioc_toReal {a b : ι → ℝ} (h : a ≤ b) :
    (volume (pi univ fun i => Ioc (a i) (b i))).toReal = ∏ i, (b i - a i) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a b : ι → Real
    h : LE.le a b
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.univ.pi fun i => Set.Ioc (a i) (b …
  -/
  simp only [volume_pi_Ioc, ENNReal.toReal_prod, ENNReal.toReal_ofReal (sub_nonneg.2 (h _))]
  /-
    🎉 no goals
  -/


theorem volume_pi_Ico {a b : ι → ℝ} :
    volume (pi univ fun i => Ico (a i) (b i)) = ∏ i, ENNReal.ofReal (b i - a i) :=
  (measure_congr Measure.univ_pi_Ico_ae_eq_Icc).trans volume_Icc_pi


@[simp]
theorem volume_pi_Ico_toReal {a b : ι → ℝ} (h : a ≤ b) :
    (volume (pi univ fun i => Ico (a i) (b i))).toReal = ∏ i, (b i - a i) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a b : ι → Real
    h : LE.le a b
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.univ.pi fun i => Set.Ico (a i) (b …
  -/
  simp only [volume_pi_Ico, ENNReal.toReal_prod, ENNReal.toReal_ofReal (sub_nonneg.2 (h _))]
  /-
    🎉 no goals
  -/


@[simp]
nonrec theorem volume_pi_ball (a : ι → ℝ) {r : ℝ} (hr : 0 < r) :
    volume (Metric.ball a r) = ENNReal.ofReal ((2 * r) ^ Fintype.card ι) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a : ι → Real
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.ball a r)) (ENNReal.ofReal (HP …
  -/
  simp only [MeasureTheory.volume_pi_ball a hr, volume_ball, Finset.prod_const]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a : ι → Real
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (HPow.hPow (ENNReal.ofReal (HMul.hMul 2 r)) Finset.univ.card) (ENNReal.of …
  -/
  exact (ENNReal.ofReal_pow (mul_nonneg zero_le_two hr.le) _).symm
  /-
    🎉 no goals
  -/


@[simp]
nonrec theorem volume_pi_closedBall (a : ι → ℝ) {r : ℝ} (hr : 0 ≤ r) :
    volume (Metric.closedBall a r) = ENNReal.ofReal ((2 * r) ^ Fintype.card ι) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a : ι → Real
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Metric.closedBall a r)) (ENNReal.ofRe …
  -/
  simp only [MeasureTheory.volume_pi_closedBall a hr, volume_closedBall, Finset.prod_const]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    a : ι → Real
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (HPow.hPow (ENNReal.ofReal (HMul.hMul 2 r)) Finset.univ.card) (ENNReal.of …
  -/
  exact (ENNReal.ofReal_pow (mul_nonneg zero_le_two hr) _).symm
  /-
    🎉 no goals
  -/


theorem volume_pi_le_prod_diam (s : Set (ι → ℝ)) :
    volume s ≤ ∏ i : ι, EMetric.diam (Function.eval i '' s) :=
  calc
    volume s ≤ volume (pi univ fun i => closure (Function.eval i '' s)) :=
      volume.mono <|
        Subset.trans (subset_pi_eval_image univ s) <| pi_mono fun _ _ => subset_closure
    _ = ∏ i, volume (closure <| Function.eval i '' s) := volume_pi_pi _
    _ ≤ ∏ i : ι, EMetric.diam (Function.eval i '' s) :=
      Finset.prod_le_prod' fun _ _ => (volume_le_diam _).trans_eq (EMetric.diam_closure _)


theorem volume_pi_le_diam_pow (s : Set (ι → ℝ)) : volume s ≤ EMetric.diam s ^ Fintype.card ι :=
  calc
    volume s ≤ ∏ i : ι, EMetric.diam (Function.eval i '' s) := volume_pi_le_prod_diam s
    _ ≤ ∏ _i : ι, (1 : ℝ≥0) * EMetric.diam s :=
      (Finset.prod_le_prod' fun i _ => (LipschitzWith.eval i).ediam_image_le s)
    _ = EMetric.diam s ^ Fintype.card ι := by
      /-
        ι : Type u_1
        inst✝ : Fintype ι
        s : Set (ι → Real)
        ⊢ Eq (Finset.univ.prod fun _i => HMul.hMul (↑1) (EMetric.diam s)) (HPow.hPow ( …
      -/
      simp only [ENNReal.coe_one, one_mul, Finset.prod_const, Fintype.card]
      /-
        🎉 no goals
      -/


theorem smul_map_volume_mul_left {a : ℝ} (h : a ≠ 0) :
    ENNReal.ofReal |a| • Measure.map (a * ·) volume = volume := by
  /-
    a : Real
    h : Ne a 0
    ⊢ Eq (HSMul.hSMul (ENNReal.ofReal (abs a)) (MeasureTheory.Measure.map (fun x = …
  -/
  refine (Real.measure_ext_Ioo_rat fun p q => ?_).symm
  /-
    a : Real
    h : Ne a 0
    p q : Rat
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.Ioo ↑p ↑q)) ((HSMul.hSMul (ENNRea …
  -/
  cases' lt_or_gt_of_ne h with h h
  · simp only [Real.volume_Ioo, Measure.smul_apply, ← ENNReal.ofReal_mul (le_of_lt <| neg_pos.2 h),
      Measure.map_apply (measurable_const_mul a) measurableSet_Ioo, neg_sub_neg, neg_mul,
      preimage_const_mul_Ioo_of_neg _ _ h, abs_of_neg h, mul_sub, smul_eq_mul,
      mul_div_cancel₀ _ (ne_of_lt h)]
  · simp only [Real.volume_Ioo, Measure.smul_apply, ← ENNReal.ofReal_mul (le_of_lt h),
      Measure.map_apply (measurable_const_mul a) measurableSet_Ioo, preimage_const_mul_Ioo _ _ h,
      abs_of_pos h, mul_sub, mul_div_cancel₀ _ (ne_of_gt h), smul_eq_mul]


theorem map_volume_mul_left {a : ℝ} (h : a ≠ 0) :
    Measure.map (a * ·) volume = ENNReal.ofReal |a⁻¹| • volume := by
  conv_rhs =>
    rw [← Real.smul_map_volume_mul_left h, smul_smul, ← ENNReal.ofReal_mul (abs_nonneg _), ←
      abs_mul, inv_mul_cancel₀ h, abs_one, ENNReal.ofReal_one, one_smul]


@[simp]
theorem volume_preimage_mul_left {a : ℝ} (h : a ≠ 0) (s : Set ℝ) :
    volume ((a * ·) ⁻¹' s) = ENNReal.ofReal (abs a⁻¹) * volume s :=
  calc
    volume ((a * ·) ⁻¹' s) = Measure.map (a * ·) volume s :=
      ((Homeomorph.mulLeft₀ a h).toMeasurableEquiv.map_apply s).symm
                                                  /-
                                                    a : Real
                                                    h : Ne a 0
                                                    s : Set Real
                                                    ⊢ Eq ((MeasureTheory.Measure.map (fun x => HMul.hMul a x) MeasureTheory.Measur …
                                                  -/
    _ = ENNReal.ofReal (abs a⁻¹) * volume s := by rw [map_volume_mul_left h]; rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


theorem smul_map_volume_mul_right {a : ℝ} (h : a ≠ 0) :
    ENNReal.ofReal |a| • Measure.map (· * a) volume = volume := by
  /-
    a : Real
    h : Ne a 0
    ⊢ Eq (HSMul.hSMul (ENNReal.ofReal (abs a)) (MeasureTheory.Measure.map (fun x = …
  -/
  simpa only [mul_comm] using Real.smul_map_volume_mul_left h
  /-
    🎉 no goals
  -/


theorem map_volume_mul_right {a : ℝ} (h : a ≠ 0) :
    Measure.map (· * a) volume = ENNReal.ofReal |a⁻¹| • volume := by
  /-
    a : Real
    h : Ne a 0
    ⊢ Eq (MeasureTheory.Measure.map (fun x => HMul.hMul x a) MeasureTheory.Measure …
  -/
  simpa only [mul_comm] using Real.map_volume_mul_left h
  /-
    🎉 no goals
  -/


@[simp]
theorem volume_preimage_mul_right {a : ℝ} (h : a ≠ 0) (s : Set ℝ) :
    volume ((· * a) ⁻¹' s) = ENNReal.ofReal (abs a⁻¹) * volume s :=
  calc
    volume ((· * a) ⁻¹' s) = Measure.map (· * a) volume s :=
      ((Homeomorph.mulRight₀ a h).toMeasurableEquiv.map_apply s).symm
                                                  /-
                                                    a : Real
                                                    h : Ne a 0
                                                    s : Set Real
                                                    ⊢ Eq ((MeasureTheory.Measure.map (fun x => HMul.hMul x a) MeasureTheory.Measur …
                                                  -/
    _ = ENNReal.ofReal (abs a⁻¹) * volume s := by rw [map_volume_mul_right h]; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- A diagonal matrix rescales Lebesgue according to its determinant. This is a special case of
`Real.map_matrix_volume_pi_eq_smul_volume_pi`, that one should use instead (and whose proof
uses this particular case). -/
theorem smul_map_diagonal_volume_pi [DecidableEq ι] {D : ι → ℝ} (h : det (diagonal D) ≠ 0) :
    ENNReal.ofReal (abs (det (diagonal D))) • Measure.map (toLin' (diagonal D)) volume =
      volume := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    D : ι → Real
    h : Ne (Matrix.diagonal D).det 0
    ⊢ Eq (HSMul.hSMul (ENNReal.ofReal (abs (Matrix.diagonal D).det)) (MeasureTheor …
  -/
  refine (Measure.pi_eq fun s hs => ?_).symm
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    D : ι → Real
    h : Ne (Matrix.diagonal D).det 0
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    ⊢ Eq ((HSMul.hSMul (ENNReal.ofReal (abs (Matrix.diagonal D).det)) (MeasureTheo …
  -/
  simp only [det_diagonal, Measure.coe_smul, Algebra.id.smul_eq_mul, Pi.smul_apply]
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    D : ι → Real
    h : Ne (Matrix.diagonal D).det 0
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    ⊢ Eq (HMul.hMul (ENNReal.ofReal (abs (Finset.univ.prod fun i => D i))) ((Measu …
  -/
  rw [Measure.map_apply _ (MeasurableSet.univ_pi hs)]
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    D : ι → Real
    h : Ne (Matrix.diagonal D).det 0
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    ⊢ Eq (HMul.hMul (ENNReal.ofReal (abs (Finset.univ.prod fun i => D i))) (Measur …
  -/
  swap; · exact Continuous.measurable (LinearMap.continuous_on_pi _)
          /-
            🎉 no goals
          -/
  have :
    (Matrix.toLin' (diagonal D) ⁻¹' Set.pi Set.univ fun i : ι => s i) =
      Set.pi Set.univ fun i : ι => (D i * ·) ⁻¹' s i := by
    ext f
    simp only [LinearMap.coe_proj, Algebra.id.smul_eq_mul, LinearMap.smul_apply, mem_univ_pi,
      mem_preimage, LinearMap.pi_apply, diagonal_toLin']
  have B : ∀ i, ofReal (abs (D i)) * volume ((D i * ·) ⁻¹' s i) = volume (s i) := by
    intro i
    have A : D i ≠ 0 := by
      simp only [det_diagonal, Ne] at h
      exact Finset.prod_ne_zero_iff.1 h i (Finset.mem_univ i)
    rw [volume_preimage_mul_left A, ← mul_assoc, ← ENNReal.ofReal_mul (abs_nonneg _), ← abs_mul,
      mul_inv_cancel₀ A, abs_one, ENNReal.ofReal_one, one_mul]
  rw [this, volume_pi_pi, Finset.abs_prod,
    ENNReal.ofReal_prod_of_nonneg fun i _ => abs_nonneg (D i), ← Finset.prod_mul_distrib]
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    D : ι → Real
    h : Ne (Matrix.diagonal D).det 0
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    this : Eq (Set.preimage (⇑(Matrix.toLin' (Matrix.diagonal D))) (Set.univ.pi fu …
    B : ∀ (i : ι), Eq (HMul.hMul (ENNReal.ofReal (abs (D i))) (MeasureTheory.Measu …
    ⊢ Eq (Finset.univ.prod fun x => HMul.hMul (ENNReal.ofReal (abs (D x))) (Measur …
  -/
  simp only [B]
  /-
    🎉 no goals
  -/


/-- A transvection preserves Lebesgue measure. -/
theorem volume_preserving_transvectionStruct [DecidableEq ι] (t : TransvectionStruct ι ℝ) :
    /-
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      t : Matrix.TransvectionStruct ι Real
      ⊢ MeasureTheory.Measure (ι → Real)
    -/
    /-
      🎉 no goals
    -/
    MeasurePreserving (toLin' t.toMatrix) := by
    /-
      🎉 no goals
    -/
  /- We use `lmarginal` to conveniently use Fubini's theorem.
    Along the coordinate where there is a shearing, it acts like a
    translation, and therefore preserves Lebesgue. -/
  have ht : Measurable (toLin' t.toMatrix) :=
    (toLin' t.toMatrix).continuous_of_finiteDimensional.measurable
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    t : Matrix.TransvectionStruct ι Real
    ht : Measurable ⇑(Matrix.toLin' t.toMatrix)
    ⊢ MeasureTheory.MeasurePreserving (⇑(Matrix.toLin' t.toMatrix)) MeasureTheory. …
  -/
  refine ⟨ht, ?_⟩
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    t : Matrix.TransvectionStruct ι Real
    ht : Measurable ⇑(Matrix.toLin' t.toMatrix)
    ⊢ Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' t.toMatrix)) MeasureTheory.Me …
  -/
  refine (pi_eq fun s hs ↦ ?_).symm
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    t : Matrix.TransvectionStruct ι Real
    ht : Measurable ⇑(Matrix.toLin' t.toMatrix)
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    ⊢ Eq ((MeasureTheory.Measure.map (⇑(Matrix.toLin' t.toMatrix)) MeasureTheory.M …
  -/
  have h2s : MeasurableSet (univ.pi s) := .pi countable_univ fun i _ ↦ hs i
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    t : Matrix.TransvectionStruct ι Real
    ht : Measurable ⇑(Matrix.toLin' t.toMatrix)
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    h2s : MeasurableSet (Set.univ.pi s)
    ⊢ Eq ((MeasureTheory.Measure.map (⇑(Matrix.toLin' t.toMatrix)) MeasureTheory.M …
  -/
  simp_rw [← pi_pi, ← lintegral_indicator_one h2s]
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    t : Matrix.TransvectionStruct ι Real
    ht : Measurable ⇑(Matrix.toLin' t.toMatrix)
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    h2s : MeasurableSet (Set.univ.pi s)
    ⊢ Eq (MeasureTheory.lintegral (MeasureTheory.Measure.map (⇑(Matrix.toLin' t.to …
  -/
  rw [lintegral_map (measurable_one.indicator h2s) ht, volume_pi]
  refine lintegral_eq_of_lmarginal_eq {t.i} ((measurable_one.indicator h2s).comp ht)
    (measurable_one.indicator h2s) ?_
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    t : Matrix.TransvectionStruct ι Real
    ht : Measurable ⇑(Matrix.toLin' t.toMatrix)
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    h2s : MeasurableSet (Set.univ.pi s)
    ⊢ Eq (MeasureTheory.lmarginal (fun x => MeasureTheory.MeasureSpace.volume) (Si …
  -/
  simp_rw [lmarginal_singleton]
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    t : Matrix.TransvectionStruct ι Real
    ht : Measurable ⇑(Matrix.toLin' t.toMatrix)
    s : ι → Set Real
    hs : ∀ (i : ι), MeasurableSet (s i)
    h2s : MeasurableSet (Set.univ.pi s)
    ⊢ Eq (fun x => MeasureTheory.lintegral MeasureTheory.MeasureSpace.volume fun x …
  -/
  ext x
  cases t with | mk t_i t_j t_hij t_c =>
  simp [transvection, mulVec_stdBasisMatrix, t_hij.symm, ← Function.update_add,
    lintegral_add_right_eq_self fun xᵢ ↦ indicator (univ.pi s) 1 (Function.update x t_i xᵢ)]


/-- Any invertible matrix rescales Lebesgue measure through the absolute value of its
determinant. -/
theorem map_matrix_volume_pi_eq_smul_volume_pi [DecidableEq ι] {M : Matrix ι ι ℝ} (hM : det M ≠ 0) :
    Measure.map (toLin' M) volume = ENNReal.ofReal (abs (det M)⁻¹) • volume := by
  -- This follows from the cases we have already proved, of diagonal matrices and transvections,
  -- as these matrices generate all invertible matrices.
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    M : Matrix ι ι Real
    hM : Ne M.det 0
    ⊢ Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' M)) MeasureTheory.MeasureSpac …
  -/
  apply diagonal_transvection_induction_of_det_ne_zero _ M hM
    /-
      case hdiag
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      M : Matrix ι ι Real
      hM : Ne M.det 0
      ⊢ ∀ (D : ι → Real), Ne (Matrix.diagonal D).det 0 → Eq (MeasureTheory.Measure.m …
    -/
  · intro D hD
    /-
      case hdiag
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      M : Matrix ι ι Real
      hM : Ne M.det 0
      D : ι → Real
      hD : Ne (Matrix.diagonal D).det 0
      ⊢ Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' (Matrix.diagonal D))) Measure …
    -/
    conv_rhs => rw [← smul_map_diagonal_volume_pi hD]
    rw [smul_smul, ← ENNReal.ofReal_mul (abs_nonneg _), ← abs_mul, inv_mul_cancel₀ hD, abs_one,
      ENNReal.ofReal_one, one_smul]
    /-
      case htransvec
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      M : Matrix ι ι Real
      hM : Ne M.det 0
      ⊢ ∀ (t : Matrix.TransvectionStruct ι Real), Eq (MeasureTheory.Measure.map (⇑(M …
    -/
  · intro t
    simp_rw [Matrix.TransvectionStruct.det, _root_.inv_one, abs_one, ENNReal.ofReal_one, one_smul,
      (volume_preserving_transvectionStruct _).map_eq]
    /-
      case hmul
      ι : Type u_1
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      M : Matrix ι ι Real
      hM : Ne M.det 0
      ⊢ ∀ (A B : Matrix ι ι Real), Ne A.det 0 → Ne B.det 0 → Eq (MeasureTheory.Measu …
    -/
  · intro A B _ _ IHA IHB
    rw [toLin'_mul, det_mul, LinearMap.coe_comp, ← Measure.map_map, IHB, Measure.map_smul, IHA,
      smul_smul, ← ENNReal.ofReal_mul (abs_nonneg _), ← abs_mul, mul_comm, mul_inv]
      /-
        case hmul.hg
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        M : Matrix ι ι Real
        hM : Ne M.det 0
        A B : Matrix ι ι Real
        a✝¹ : Ne A.det 0
        a✝ : Ne B.det 0
        IHA : Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' A)) MeasureTheory.Measure …
        IHB : Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' B)) MeasureTheory.Measure …
        ⊢ Measurable ⇑(Matrix.toLin' A)
      -/
    · apply Continuous.measurable
      /-
        case hmul.hg.hf
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        M : Matrix ι ι Real
        hM : Ne M.det 0
        A B : Matrix ι ι Real
        a✝¹ : Ne A.det 0
        a✝ : Ne B.det 0
        IHA : Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' A)) MeasureTheory.Measure …
        IHB : Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' B)) MeasureTheory.Measure …
        ⊢ Continuous ⇑(Matrix.toLin' A)
      -/
      apply LinearMap.continuous_on_pi
      /-
        🎉 no goals
      -/
      /-
        case hmul.hf
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        M : Matrix ι ι Real
        hM : Ne M.det 0
        A B : Matrix ι ι Real
        a✝¹ : Ne A.det 0
        a✝ : Ne B.det 0
        IHA : Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' A)) MeasureTheory.Measure …
        IHB : Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' B)) MeasureTheory.Measure …
        ⊢ Measurable ⇑(Matrix.toLin' B)
      -/
    · apply Continuous.measurable
      /-
        case hmul.hf.hf
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        M : Matrix ι ι Real
        hM : Ne M.det 0
        A B : Matrix ι ι Real
        a✝¹ : Ne A.det 0
        a✝ : Ne B.det 0
        IHA : Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' A)) MeasureTheory.Measure …
        IHB : Eq (MeasureTheory.Measure.map (⇑(Matrix.toLin' B)) MeasureTheory.Measure …
        ⊢ Continuous ⇑(Matrix.toLin' B)
      -/
      apply LinearMap.continuous_on_pi
      /-
        🎉 no goals
      -/


/-- Any invertible linear map rescales Lebesgue measure through the absolute value of its
determinant. -/
theorem map_linearMap_volume_pi_eq_smul_volume_pi {f : (ι → ℝ) →ₗ[ℝ] ι → ℝ}
    (hf : LinearMap.det f ≠ 0) : Measure.map f volume =
      ENNReal.ofReal (abs (LinearMap.det f)⁻¹) • volume := by
  classical
    -- this is deduced from the matrix case
    let M := LinearMap.toMatrix' f
    have A : LinearMap.det f = det M := by simp only [M, LinearMap.det_toMatrix']
    have B : f = toLin' M := by simp only [M, toLin'_toMatrix']
    rw [A, B]
    apply map_matrix_volume_pi_eq_smul_volume_pi
    rwa [A] at hf


/-- The region between two real-valued functions on an arbitrary set. -/
def regionBetween (f g : α → ℝ) (s : Set α) : Set (α × ℝ) :=
  { p : α × ℝ | p.1 ∈ s ∧ p.2 ∈ Ioo (f p.1) (g p.1) }


theorem regionBetween_subset (f g : α → ℝ) (s : Set α) : regionBetween f g s ⊆ s ×ˢ univ := by
  /-
    α : Type u_1
    f g : α → Real
    s : Set α
    ⊢ HasSubset.Subset (regionBetween f g s) (SProd.sprod s Set.univ)
  -/
  simpa only [prod_univ, regionBetween, Set.preimage, setOf_subset_setOf] using fun a => And.left
  /-
    🎉 no goals
  -/


/-- The region between two measurable functions on a measurable set is measurable. -/
theorem measurableSet_regionBetween (hf : Measurable f) (hg : Measurable g) (hs : MeasurableSet s) :
    MeasurableSet (regionBetween f g s) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f g : α → Real
    s : Set α
    hf : Measurable f
    hg : Measurable g
    hs : MeasurableSet s
    ⊢ MeasurableSet (regionBetween f g s)
  -/
  dsimp only [regionBetween, Ioo, mem_setOf_eq, setOf_and]
  refine
    MeasurableSet.inter ?_
      ((measurableSet_lt (hf.comp measurable_fst) measurable_snd).inter
        (measurableSet_lt measurable_snd (hg.comp measurable_fst)))
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f g : α → Real
    s : Set α
    hf : Measurable f
    hg : Measurable g
    hs : MeasurableSet s
    ⊢ MeasurableSet (setOf fun a => Membership.mem s a.1)
  -/
  exact measurable_fst hs
  /-
    🎉 no goals
  -/


/-- The region between two measurable functions on a measurable set is measurable;
a version for the region together with the graph of the upper function. -/
theorem measurableSet_region_between_oc (hf : Measurable f) (hg : Measurable g)
    (hs : MeasurableSet s) :
    MeasurableSet { p : α × ℝ | p.fst ∈ s ∧ p.snd ∈ Ioc (f p.fst) (g p.fst) } := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f g : α → Real
    s : Set α
    hf : Measurable f
    hg : Measurable g
    hs : MeasurableSet s
    ⊢ MeasurableSet (setOf fun p => And (Membership.mem s p.1) (Membership.mem (Se …
  -/
  dsimp only [regionBetween, Ioc, mem_setOf_eq, setOf_and]
  refine
    MeasurableSet.inter ?_
      ((measurableSet_lt (hf.comp measurable_fst) measurable_snd).inter
        (measurableSet_le measurable_snd (hg.comp measurable_fst)))
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f g : α → Real
    s : Set α
    hf : Measurable f
    hg : Measurable g
    hs : MeasurableSet s
    ⊢ MeasurableSet (setOf fun a => Membership.mem s a.1)
  -/
  exact measurable_fst hs
  /-
    🎉 no goals
  -/


/-- The region between two measurable functions on a measurable set is measurable;
a version for the region together with the graph of the lower function. -/
theorem measurableSet_region_between_co (hf : Measurable f) (hg : Measurable g)
    (hs : MeasurableSet s) :
    MeasurableSet { p : α × ℝ | p.fst ∈ s ∧ p.snd ∈ Ico (f p.fst) (g p.fst) } := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f g : α → Real
    s : Set α
    hf : Measurable f
    hg : Measurable g
    hs : MeasurableSet s
    ⊢ MeasurableSet (setOf fun p => And (Membership.mem s p.1) (Membership.mem (Se …
  -/
  dsimp only [regionBetween, Ico, mem_setOf_eq, setOf_and]
  refine
    MeasurableSet.inter ?_
      ((measurableSet_le (hf.comp measurable_fst) measurable_snd).inter
        (measurableSet_lt measurable_snd (hg.comp measurable_fst)))
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f g : α → Real
    s : Set α
    hf : Measurable f
    hg : Measurable g
    hs : MeasurableSet s
    ⊢ MeasurableSet (setOf fun a => Membership.mem s a.1)
  -/
  exact measurable_fst hs
  /-
    🎉 no goals
  -/


/-- The region between two measurable functions on a measurable set is measurable;
a version for the region together with the graphs of both functions. -/
theorem measurableSet_region_between_cc (hf : Measurable f) (hg : Measurable g)
    (hs : MeasurableSet s) :
    MeasurableSet { p : α × ℝ | p.fst ∈ s ∧ p.snd ∈ Icc (f p.fst) (g p.fst) } := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f g : α → Real
    s : Set α
    hf : Measurable f
    hg : Measurable g
    hs : MeasurableSet s
    ⊢ MeasurableSet (setOf fun p => And (Membership.mem s p.1) (Membership.mem (Se …
  -/
  dsimp only [regionBetween, Icc, mem_setOf_eq, setOf_and]
  refine
    MeasurableSet.inter ?_
      ((measurableSet_le (hf.comp measurable_fst) measurable_snd).inter
        (measurableSet_le measurable_snd (hg.comp measurable_fst)))
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f g : α → Real
    s : Set α
    hf : Measurable f
    hg : Measurable g
    hs : MeasurableSet s
    ⊢ MeasurableSet (setOf fun a => Membership.mem s a.1)
  -/
  exact measurable_fst hs
  /-
    🎉 no goals
  -/


/-- The graph of a measurable function is a measurable set. -/
theorem measurableSet_graph (hf : Measurable f) :
    MeasurableSet { p : α × ℝ | p.snd = f p.fst } := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Real
    hf : Measurable f
    ⊢ MeasurableSet (setOf fun p => Eq p.2 (f p.1))
  -/
  simpa using measurableSet_region_between_cc hf hf MeasurableSet.univ
  /-
    🎉 no goals
  -/


theorem volume_regionBetween_eq_lintegral' (hf : Measurable f) (hg : Measurable g)
    (hs : MeasurableSet s) :
    μ.prod volume (regionBetween f g s) = ∫⁻ y in s, ENNReal.ofReal ((g - f) y) ∂μ := by
  classical
    rw [Measure.prod_apply]
    · have h :
        (fun x => volume { a | x ∈ s ∧ a ∈ Ioo (f x) (g x) }) =
          s.indicator fun x => ENNReal.ofReal (g x - f x) := by
        funext x
        rw [indicator_apply]
        split_ifs with h
        · have hx : { a | x ∈ s ∧ a ∈ Ioo (f x) (g x) } = Ioo (f x) (g x) := by simp [h, Ioo]
          simp only [hx, Real.volume_Ioo, sub_zero]
        · have hx : { a | x ∈ s ∧ a ∈ Ioo (f x) (g x) } = ∅ := by simp [h]
          simp only [hx, measure_empty]
      dsimp only [regionBetween, preimage_setOf_eq]
      rw [h, lintegral_indicator] <;> simp only [hs, Pi.sub_apply]
    · exact measurableSet_regionBetween hf hg hs


/-- The volume of the region between two almost everywhere measurable functions on a measurable set
    can be represented as a Lebesgue integral. -/
theorem volume_regionBetween_eq_lintegral [SFinite μ] (hf : AEMeasurable f (μ.restrict s))
    (hg : AEMeasurable g (μ.restrict s)) (hs : MeasurableSet s) :
    μ.prod volume (regionBetween f g s) = ∫⁻ y in s, ENNReal.ofReal ((g - f) y) ∂μ := by
  have h₁ :
    (fun y => ENNReal.ofReal ((g - f) y)) =ᵐ[μ.restrict s] fun y =>
      ENNReal.ofReal ((AEMeasurable.mk g hg - AEMeasurable.mk f hf) y) :=
    (hg.ae_eq_mk.sub hf.ae_eq_mk).fun_comp ENNReal.ofReal
  have h₂ :
    (μ.restrict s).prod volume (regionBetween f g s) =
      (μ.restrict s).prod volume
        (regionBetween (AEMeasurable.mk f hf) (AEMeasurable.mk g hg) s) := by
    apply measure_congr
    apply EventuallyEq.rfl.inter
    exact
      ((quasiMeasurePreserving_fst.ae_eq_comp hf.ae_eq_mk).comp₂ _ EventuallyEq.rfl).inter
        (EventuallyEq.rfl.comp₂ _ <| quasiMeasurePreserving_fst.ae_eq_comp hg.ae_eq_mk)
  rw [lintegral_congr_ae h₁, ←
    volume_regionBetween_eq_lintegral' hf.measurable_mk hg.measurable_mk hs]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    s : Set α
    inst✝ : MeasureTheory.SFinite μ
    hf : AEMeasurable f (μ.restrict s)
    hg : AEMeasurable g (μ.restrict s)
    hs : MeasurableSet s
    h₁ : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun y => ENNReal.ofReal ( …
    h₂ : Eq (((μ.restrict s).prod MeasureTheory.MeasureSpace.volume) (regionBetwee …
    ⊢ Eq ((μ.prod MeasureTheory.MeasureSpace.volume) (regionBetween f g s)) ((μ.pr …
  -/
  convert h₂ using 1
    /-
      case h.e'_2
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      s : Set α
      inst✝ : MeasureTheory.SFinite μ
      hf : AEMeasurable f (μ.restrict s)
      hg : AEMeasurable g (μ.restrict s)
      hs : MeasurableSet s
      h₁ : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun y => ENNReal.ofReal ( …
      h₂ : Eq (((μ.restrict s).prod MeasureTheory.MeasureSpace.volume) (regionBetwee …
      ⊢ Eq ((μ.prod MeasureTheory.MeasureSpace.volume) (regionBetween f g s)) (((μ.r …
    -/
  · rw [Measure.restrict_prod_eq_prod_univ]
    /-
      case h.e'_2
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      s : Set α
      inst✝ : MeasureTheory.SFinite μ
      hf : AEMeasurable f (μ.restrict s)
      hg : AEMeasurable g (μ.restrict s)
      hs : MeasurableSet s
      h₁ : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun y => ENNReal.ofReal ( …
      h₂ : Eq (((μ.restrict s).prod MeasureTheory.MeasureSpace.volume) (regionBetwee …
      ⊢ Eq ((μ.prod MeasureTheory.MeasureSpace.volume) (regionBetween f g s)) (((μ.p …
    -/
    exact (Measure.restrict_eq_self _ (regionBetween_subset f g s)).symm
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      α : Type u_1
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      s : Set α
      inst✝ : MeasureTheory.SFinite μ
      hf : AEMeasurable f (μ.restrict s)
      hg : AEMeasurable g (μ.restrict s)
      hs : MeasurableSet s
      h₁ : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun y => ENNReal.ofReal ( …
      h₂ : Eq (((μ.restrict s).prod MeasureTheory.MeasureSpace.volume) (regionBetwee …
      ⊢ Eq ((μ.prod MeasureTheory.MeasureSpace.volume) (regionBetween (AEMeasurable. …
    -/
  · rw [Measure.restrict_prod_eq_prod_univ]
    exact
      (Measure.restrict_eq_self _
          (regionBetween_subset (AEMeasurable.mk f hf) (AEMeasurable.mk g hg) s)).symm


/-- The region between two a.e.-measurable functions on a null-measurable set is null-measurable. -/
lemma nullMeasurableSet_regionBetween (μ : Measure α)
    {f g : α → ℝ} (f_mble : AEMeasurable f μ) (g_mble : AEMeasurable g μ)
    {s : Set α} (s_mble : NullMeasurableSet s μ) :
    NullMeasurableSet {p : α × ℝ | p.1 ∈ s ∧ p.snd ∈ Ioo (f p.fst) (g p.fst)} (μ.prod volume) := by
  refine NullMeasurableSet.inter
          (s_mble.preimage quasiMeasurePreserving_fst) (NullMeasurableSet.inter ?_ ?_)
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (fun p => LT.lt (f p.1) p.2) (μ.prod Measure …
    -/
  · exact nullMeasurableSet_lt (AEMeasurable.fst f_mble) measurable_snd.aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (fun p => LT.lt p.2 (g p.1)) (μ.prod Measure …
    -/
  · exact nullMeasurableSet_lt measurable_snd.aemeasurable (AEMeasurable.fst g_mble)
    /-
      🎉 no goals
    -/


/-- The region between two a.e.-measurable functions on a null-measurable set is null-measurable;
a version for the region together with the graph of the upper function. -/
lemma nullMeasurableSet_region_between_oc (μ : Measure α)
    {f g : α → ℝ} (f_mble : AEMeasurable f μ) (g_mble : AEMeasurable g μ)
    {s : Set α} (s_mble : NullMeasurableSet s μ) :
    NullMeasurableSet {p : α × ℝ | p.1 ∈ s ∧ p.snd ∈ Ioc (f p.fst) (g p.fst)} (μ.prod volume) := by
  refine NullMeasurableSet.inter
          (s_mble.preimage quasiMeasurePreserving_fst) (NullMeasurableSet.inter ?_ ?_)
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (fun p => LT.lt (f p.1) p.2) (μ.prod Measure …
    -/
  · exact nullMeasurableSet_lt (AEMeasurable.fst f_mble) measurable_snd.aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (fun p => LE.le p.2 (g p.1)) (μ.prod Measure …
    -/
  · change NullMeasurableSet {p : α × ℝ | p.snd ≤ g p.fst} (μ.prod volume)
    rw [show {p : α × ℝ | p.snd ≤ g p.fst} = {p : α × ℝ | g p.fst < p.snd}ᶜ by
          ext p
          simp only [mem_setOf_eq, mem_compl_iff, not_lt]]
    /-
      case refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (HasCompl.compl (setOf fun p => LT.lt (g p.1 …
    -/
    exact (nullMeasurableSet_lt (AEMeasurable.fst g_mble) measurable_snd.aemeasurable).compl
    /-
      🎉 no goals
    -/


/-- The region between two a.e.-measurable functions on a null-measurable set is null-measurable;
a version for the region together with the graph of the lower function. -/
lemma nullMeasurableSet_region_between_co (μ : Measure α)
    {f g : α → ℝ} (f_mble : AEMeasurable f μ) (g_mble : AEMeasurable g μ)
    {s : Set α} (s_mble : NullMeasurableSet s μ) :
    NullMeasurableSet {p : α × ℝ | p.1 ∈ s ∧ p.snd ∈ Ico (f p.fst) (g p.fst)} (μ.prod volume) := by
  refine NullMeasurableSet.inter
          (s_mble.preimage quasiMeasurePreserving_fst) (NullMeasurableSet.inter ?_ ?_)
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (fun p => LE.le (f p.1) p.2) (μ.prod Measure …
    -/
  · change NullMeasurableSet {p : α × ℝ | f p.fst ≤ p.snd} (μ.prod volume)
    rw [show {p : α × ℝ | f p.fst ≤ p.snd} = {p : α × ℝ | p.snd < f p.fst}ᶜ by
          ext p
          simp only [mem_setOf_eq, mem_compl_iff, not_lt]]
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (HasCompl.compl (setOf fun p => LT.lt p.2 (f …
    -/
    exact (nullMeasurableSet_lt measurable_snd.aemeasurable (AEMeasurable.fst f_mble)).compl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (fun p => LT.lt p.2 (g p.1)) (μ.prod Measure …
    -/
  · exact nullMeasurableSet_lt measurable_snd.aemeasurable (AEMeasurable.fst g_mble)
    /-
      🎉 no goals
    -/


/-- The region between two a.e.-measurable functions on a null-measurable set is null-measurable;
a version for the region together with the graphs of both functions. -/
lemma nullMeasurableSet_region_between_cc (μ : Measure α)
    {f g : α → ℝ} (f_mble : AEMeasurable f μ) (g_mble : AEMeasurable g μ)
    {s : Set α} (s_mble : NullMeasurableSet s μ) :
    NullMeasurableSet {p : α × ℝ | p.1 ∈ s ∧ p.snd ∈ Icc (f p.fst) (g p.fst)} (μ.prod volume) := by
  refine NullMeasurableSet.inter
          (s_mble.preimage quasiMeasurePreserving_fst) (NullMeasurableSet.inter ?_ ?_)
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (fun p => LE.le (f p.1) p.2) (μ.prod Measure …
    -/
  · change NullMeasurableSet {p : α × ℝ | f p.fst ≤ p.snd} (μ.prod volume)
    rw [show {p : α × ℝ | f p.fst ≤ p.snd} = {p : α × ℝ | p.snd < f p.fst}ᶜ by
          ext p
          simp only [mem_setOf_eq, mem_compl_iff, not_lt]]
    /-
      case refine_1
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (HasCompl.compl (setOf fun p => LT.lt p.2 (f …
    -/
    exact (nullMeasurableSet_lt measurable_snd.aemeasurable (AEMeasurable.fst f_mble)).compl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (fun p => LE.le p.2 (g p.1)) (μ.prod Measure …
    -/
  · change NullMeasurableSet {p : α × ℝ | p.snd ≤ g p.fst} (μ.prod volume)
    rw [show {p : α × ℝ | p.snd ≤ g p.fst} = {p : α × ℝ | g p.fst < p.snd}ᶜ by
          ext p
          simp only [mem_setOf_eq, mem_compl_iff, not_lt]]
    /-
      case refine_2
      α : Type u_1
      inst✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f g : α → Real
      f_mble : AEMeasurable f μ
      g_mble : AEMeasurable g μ
      s : Set α
      s_mble : MeasureTheory.NullMeasurableSet s μ
      ⊢ MeasureTheory.NullMeasurableSet (HasCompl.compl (setOf fun p => LT.lt (g p.1 …
    -/
    exact (nullMeasurableSet_lt (AEMeasurable.fst g_mble) measurable_snd.aemeasurable).compl
    /-
      🎉 no goals
    -/


/-- Consider a real set `s`. If a property is true almost everywhere in `s ∩ (a, b)` for
all `a, b ∈ s`, then it is true almost everywhere in `s`. Formulated with `μ.restrict`.
See also `ae_of_mem_of_ae_of_mem_inter_Ioo`. -/
theorem ae_restrict_of_ae_restrict_inter_Ioo {μ : Measure ℝ} [NoAtoms μ] {s : Set ℝ} {p : ℝ → Prop}
    (h : ∀ a b, a ∈ s → b ∈ s → a < b → ∀ᵐ x ∂μ.restrict (s ∩ Ioo a b), p x) :
    ∀ᵐ x ∂μ.restrict s, p x := by
  /- By second-countability, we cover `s` by countably many intervals `(a, b)` (except maybe for
    two endpoints, which don't matter since `μ` does not have any atom). -/
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))
  -/
  let T : s × s → Set ℝ := fun p => Ioo p.1 p.2
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))
  -/
  let u := ⋃ i : ↥s × ↥s, T i
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))
  -/
  have hfinite : (s \ u).Finite := s.finite_diff_iUnion_Ioo'
  obtain ⟨A, A_count, hA⟩ :
    ∃ A : Set (↥s × ↥s), A.Countable ∧ ⋃ i ∈ A, T i = ⋃ i : ↥s × ↥s, T i :=
    isOpen_iUnion_countable _ fun p => isOpen_Ioo
  have : s ⊆ s \ u ∪ ⋃ p ∈ A, s ∩ T p := by
    intro x hx
    by_cases h'x : x ∈ ⋃ i : ↥s × ↥s, T i
    · rw [← hA] at h'x
      obtain ⟨p, pA, xp⟩ : ∃ p : ↥s × ↥s, p ∈ A ∧ x ∈ T p := by
        simpa only [mem_iUnion, exists_prop, SetCoe.exists, exists_and_right] using h'x
      right
      exact mem_biUnion pA ⟨hx, xp⟩
    · exact Or.inl ⟨hx, h'x⟩
  /-
    case intro.intro
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    hfinite : (SDiff.sdiff s u).Finite
    A : Set (Prod ↑s ↑s)
    A_count : A.Countable
    hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
    this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
    ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict s))
  -/
  apply ae_restrict_of_ae_restrict_of_subset this
  /-
    case intro.intro
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    hfinite : (SDiff.sdiff s u).Finite
    A : Set (Prod ↑s ↑s)
    A_count : A.Countable
    hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
    this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
    ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Union.union  …
  -/
  rw [ae_restrict_union_iff, ae_restrict_biUnion_iff _ A_count]
  /-
    case intro.intro
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    hfinite : (SDiff.sdiff s u).Finite
    A : Set (Prod ↑s ↑s)
    A_count : A.Countable
    hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
    this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
    ⊢ And (Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (SDiff.s …
  -/
  constructor
    /-
      case intro.intro.left
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.NoAtoms μ
      s : Set Real
      p : Real → Prop
      h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
      T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
      u : Set Real := Set.iUnion fun i => T i
      hfinite : (SDiff.sdiff s u).Finite
      A : Set (Prod ↑s ↑s)
      A_count : A.Countable
      hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
      this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
      ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (SDiff.sdiff  …
    -/
  · have : μ.restrict (s \ u) = 0 := by simp only [restrict_eq_zero, hfinite.measure_zero]
    /-
      case intro.intro.left
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.NoAtoms μ
      s : Set Real
      p : Real → Prop
      h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
      T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
      u : Set Real := Set.iUnion fun i => T i
      hfinite : (SDiff.sdiff s u).Finite
      A : Set (Prod ↑s ↑s)
      A_count : A.Countable
      hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
      this✝ : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p => …
      this : Eq (μ.restrict (SDiff.sdiff s u)) 0
      ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (SDiff.sdiff  …
    -/
    simp only [this, ae_zero, eventually_bot]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.right
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.NoAtoms μ
      s : Set Real
      p : Real → Prop
      h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
      T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
      u : Set Real := Set.iUnion fun i => T i
      hfinite : (SDiff.sdiff s u).Finite
      A : Set (Prod ↑s ↑s)
      A_count : A.Countable
      hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
      this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
      ⊢ ∀ (i : Prod ↑s ↑s), Membership.mem A i → Filter.Eventually (fun x => p x) (M …
    -/
  · rintro ⟨⟨a, as⟩, ⟨b, bs⟩⟩ -
    /-
      case intro.intro.right.mk.mk.mk
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.NoAtoms μ
      s : Set Real
      p : Real → Prop
      h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
      T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
      u : Set Real := Set.iUnion fun i => T i
      hfinite : (SDiff.sdiff s u).Finite
      A : Set (Prod ↑s ↑s)
      A_count : A.Countable
      hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
      this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
      a : Real
      as : Membership.mem s a
      b : Real
      bs : Membership.mem s b
      ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Inter.inter  …
    -/
    dsimp [T]
    /-
      case intro.intro.right.mk.mk.mk
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.NoAtoms μ
      s : Set Real
      p : Real → Prop
      h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
      T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
      u : Set Real := Set.iUnion fun i => T i
      hfinite : (SDiff.sdiff s u).Finite
      A : Set (Prod ↑s ↑s)
      A_count : A.Countable
      hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
      this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
      a : Real
      as : Membership.mem s a
      b : Real
      bs : Membership.mem s b
      ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Inter.inter  …
    -/
    rcases le_or_lt b a with (hba | hab)
      /-
        case intro.intro.right.mk.mk.mk.inl
        μ : MeasureTheory.Measure Real
        inst✝ : MeasureTheory.NoAtoms μ
        s : Set Real
        p : Real → Prop
        h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
        T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
        u : Set Real := Set.iUnion fun i => T i
        hfinite : (SDiff.sdiff s u).Finite
        A : Set (Prod ↑s ↑s)
        A_count : A.Countable
        hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
        this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
        a : Real
        as : Membership.mem s a
        b : Real
        bs : Membership.mem s b
        hba : LE.le b a
        ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Inter.inter  …
      -/
    · simp only [Ioo_eq_empty_of_le hba, inter_empty, restrict_empty, ae_zero, eventually_bot]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.right.mk.mk.mk.inr
        μ : MeasureTheory.Measure Real
        inst✝ : MeasureTheory.NoAtoms μ
        s : Set Real
        p : Real → Prop
        h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
        T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
        u : Set Real := Set.iUnion fun i => T i
        hfinite : (SDiff.sdiff s u).Finite
        A : Set (Prod ↑s ↑s)
        A_count : A.Countable
        hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
        this : HasSubset.Subset s (Union.union (SDiff.sdiff s u) (Set.iUnion fun p =>  …
        a : Real
        as : Membership.mem s a
        b : Real
        bs : Membership.mem s b
        hab : LT.lt a b
        ⊢ Filter.Eventually (fun x => p x) (MeasureTheory.ae (μ.restrict (Inter.inter  …
      -/
    · exact h a b as bs hab
      /-
        🎉 no goals
      -/


/-- Consider a real set `s`. If a property is true almost everywhere in `s ∩ (a, b)` for
all `a, b ∈ s`, then it is true almost everywhere in `s`. Formulated with bare membership.
See also `ae_restrict_of_ae_restrict_inter_Ioo`. -/
theorem ae_of_mem_of_ae_of_mem_inter_Ioo {μ : Measure ℝ} [NoAtoms μ] {s : Set ℝ} {p : ℝ → Prop}
    (h : ∀ a b, a ∈ s → b ∈ s → a < b → ∀ᵐ x ∂μ, x ∈ s ∩ Ioo a b → p x) :
    ∀ᵐ x ∂μ, x ∈ s → p x := by
  /- By second-countability, we cover `s` by countably many intervals `(a, b)` (except maybe for
    two endpoints, which don't matter since `μ` does not have any atom). -/
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    ⊢ Filter.Eventually (fun x => Membership.mem s x → p x) (MeasureTheory.ae μ)
  -/
  let T : s × s → Set ℝ := fun p => Ioo p.1 p.2
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    ⊢ Filter.Eventually (fun x => Membership.mem s x → p x) (MeasureTheory.ae μ)
  -/
  let u := ⋃ i : ↥s × ↥s, T i
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    ⊢ Filter.Eventually (fun x => Membership.mem s x → p x) (MeasureTheory.ae μ)
  -/
  have hfinite : (s \ u).Finite := s.finite_diff_iUnion_Ioo'
  obtain ⟨A, A_count, hA⟩ :
    ∃ A : Set (↥s × ↥s), A.Countable ∧ ⋃ i ∈ A, T i = ⋃ i : ↥s × ↥s, T i :=
    isOpen_iUnion_countable _ fun p => isOpen_Ioo
  /-
    case intro.intro
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    hfinite : (SDiff.sdiff s u).Finite
    A : Set (Prod ↑s ↑s)
    A_count : A.Countable
    hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
    ⊢ Filter.Eventually (fun x => Membership.mem s x → p x) (MeasureTheory.ae μ)
  -/
  have M : ∀ᵐ x ∂μ, x ∉ s \ u := hfinite.countable.ae_not_mem _
  have M' : ∀ᵐ x ∂μ, ∀ (i : ↥s × ↥s), i ∈ A → x ∈ s ∩ T i → p x := by
    rw [ae_ball_iff A_count]
    rintro ⟨⟨a, as⟩, ⟨b, bs⟩⟩ -
    change ∀ᵐ x : ℝ ∂μ, x ∈ s ∩ Ioo a b → p x
    rcases le_or_lt b a with (hba | hab)
    · simp only [Ioo_eq_empty_of_le hba, inter_empty, IsEmpty.forall_iff, eventually_true,
        mem_empty_iff_false]
    · exact h a b as bs hab
  /-
    case intro.intro
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    hfinite : (SDiff.sdiff s u).Finite
    A : Set (Prod ↑s ↑s)
    A_count : A.Countable
    hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
    M : Filter.Eventually (fun x => Not (Membership.mem (SDiff.sdiff s u) x)) (Mea …
    M' : Filter.Eventually (fun x => ∀ (i : Prod ↑s ↑s), Membership.mem A i → Memb …
    ⊢ Filter.Eventually (fun x => Membership.mem s x → p x) (MeasureTheory.ae μ)
  -/
  filter_upwards [M, M'] with x hx h'x
  /-
    case h
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    hfinite : (SDiff.sdiff s u).Finite
    A : Set (Prod ↑s ↑s)
    A_count : A.Countable
    hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
    M : Filter.Eventually (fun x => Not (Membership.mem (SDiff.sdiff s u) x)) (Mea …
    M' : Filter.Eventually (fun x => ∀ (i : Prod ↑s ↑s), Membership.mem A i → Memb …
    x : Real
    hx : Not (Membership.mem (SDiff.sdiff s u) x)
    h'x : ∀ (i : Prod ↑s ↑s), Membership.mem A i → Membership.mem (Inter.inter s ( …
    ⊢ Membership.mem s x → p x
  -/
  intro xs
  /-
    case h
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    s : Set Real
    p : Real → Prop
    h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
    T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
    u : Set Real := Set.iUnion fun i => T i
    hfinite : (SDiff.sdiff s u).Finite
    A : Set (Prod ↑s ↑s)
    A_count : A.Countable
    hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
    M : Filter.Eventually (fun x => Not (Membership.mem (SDiff.sdiff s u) x)) (Mea …
    M' : Filter.Eventually (fun x => ∀ (i : Prod ↑s ↑s), Membership.mem A i → Memb …
    x : Real
    hx : Not (Membership.mem (SDiff.sdiff s u) x)
    h'x : ∀ (i : Prod ↑s ↑s), Membership.mem A i → Membership.mem (Inter.inter s ( …
    xs : Membership.mem s x
    ⊢ p x
  -/
  by_cases Hx : x ∈ ⋃ i : ↥s × ↥s, T i
    /-
      case pos
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.NoAtoms μ
      s : Set Real
      p : Real → Prop
      h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
      T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
      u : Set Real := Set.iUnion fun i => T i
      hfinite : (SDiff.sdiff s u).Finite
      A : Set (Prod ↑s ↑s)
      A_count : A.Countable
      hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
      M : Filter.Eventually (fun x => Not (Membership.mem (SDiff.sdiff s u) x)) (Mea …
      M' : Filter.Eventually (fun x => ∀ (i : Prod ↑s ↑s), Membership.mem A i → Memb …
      x : Real
      hx : Not (Membership.mem (SDiff.sdiff s u) x)
      h'x : ∀ (i : Prod ↑s ↑s), Membership.mem A i → Membership.mem (Inter.inter s ( …
      xs : Membership.mem s x
      Hx : Membership.mem (Set.iUnion fun i => T i) x
      ⊢ p x
    -/
  · rw [← hA] at Hx
    obtain ⟨p, pA, xp⟩ : ∃ p : ↥s × ↥s, p ∈ A ∧ x ∈ T p := by
      simpa only [mem_iUnion, exists_prop, SetCoe.exists, exists_and_right] using Hx
    /-
      case pos.intro.intro
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.NoAtoms μ
      s : Set Real
      p✝ : Real → Prop
      h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
      T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
      u : Set Real := Set.iUnion fun i => T i
      hfinite : (SDiff.sdiff s u).Finite
      A : Set (Prod ↑s ↑s)
      A_count : A.Countable
      hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
      M : Filter.Eventually (fun x => Not (Membership.mem (SDiff.sdiff s u) x)) (Mea …
      M' : Filter.Eventually (fun x => ∀ (i : Prod ↑s ↑s), Membership.mem A i → Memb …
      x : Real
      hx : Not (Membership.mem (SDiff.sdiff s u) x)
      h'x : ∀ (i : Prod ↑s ↑s), Membership.mem A i → Membership.mem (Inter.inter s ( …
      xs : Membership.mem s x
      Hx : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => T i) x
      p : Prod ↑s ↑s
      pA : Membership.mem A p
      xp : Membership.mem (T p) x
      ⊢ p✝ x
    -/
    apply h'x p pA ⟨xs, xp⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.NoAtoms μ
      s : Set Real
      p : Real → Prop
      h : ∀ (a b : Real), Membership.mem s a → Membership.mem s b → LT.lt a b → Filt …
      T : Prod ↑s ↑s → Set Real := fun p => Set.Ioo ↑p.1 ↑p.2
      u : Set Real := Set.iUnion fun i => T i
      hfinite : (SDiff.sdiff s u).Finite
      A : Set (Prod ↑s ↑s)
      A_count : A.Countable
      hA : Eq (Set.iUnion fun i => Set.iUnion fun h => T i) (Set.iUnion fun i => T i)
      M : Filter.Eventually (fun x => Not (Membership.mem (SDiff.sdiff s u) x)) (Mea …
      M' : Filter.Eventually (fun x => ∀ (i : Prod ↑s ↑s), Membership.mem A i → Memb …
      x : Real
      hx : Not (Membership.mem (SDiff.sdiff s u) x)
      h'x : ∀ (i : Prod ↑s ↑s), Membership.mem A i → Membership.mem (Inter.inter s ( …
      xs : Membership.mem s x
      Hx : Not (Membership.mem (Set.iUnion fun i => T i) x)
      ⊢ p x
    -/
  · exact False.elim (hx ⟨xs, Hx⟩)
    /-
      🎉 no goals
    -/

