/-- A function `f` is called *interval integrable* with respect to a measure `μ` on an unordered
interval `a..b` if it is integrable on both intervals `(a, b]` and `(b, a]`. One of these
intervals is always empty, so this property is equivalent to `f` being integrable on
`(min a b, max a b]`. -/
def IntervalIntegrable (f : ℝ → E) (μ : Measure ℝ) (a b : ℝ) : Prop :=
  IntegrableOn f (Ioc a b) μ ∧ IntegrableOn f (Ioc b a) μ


/-- A function is interval integrable with respect to a given measure `μ` on `a..b` if and
  only if it is integrable on `uIoc a b` with respect to `μ`. This is an equivalent
  definition of `IntervalIntegrable`. -/
theorem intervalIntegrable_iff : IntervalIntegrable f μ a b ↔ IntegrableOn f (Ι a b) μ := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    ⊢ Iff (IntervalIntegrable f μ a b) (MeasureTheory.IntegrableOn f (Set.uIoc a b …
  -/
  rw [uIoc_eq_union, integrableOn_union, IntervalIntegrable]
  /-
    🎉 no goals
  -/


/-- If a function is interval integrable with respect to a given measure `μ` on `a..b` then
  it is integrable on `uIoc a b` with respect to `μ`. -/
theorem IntervalIntegrable.def' (h : IntervalIntegrable f μ a b) : IntegrableOn f (Ι a b) μ :=
  intervalIntegrable_iff.mp h


theorem IntervalIntegrable.congr {g : ℝ → E} (hf : IntervalIntegrable f μ a b)
    (h : f =ᵐ[μ.restrict (Ι a b)] g) :
    IntervalIntegrable g μ a b := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    g : Real → E
    hf : IntervalIntegrable f μ a b
    h : (MeasureTheory.ae (μ.restrict (Set.uIoc a b))).EventuallyEq f g
    ⊢ IntervalIntegrable g μ a b
  -/
  rwa [intervalIntegrable_iff, ← integrableOn_congr_fun_ae h, ← intervalIntegrable_iff]
  /-
    🎉 no goals
  -/


theorem intervalIntegrable_iff_integrableOn_Ioc_of_le (hab : a ≤ b) :
    IntervalIntegrable f μ a b ↔ IntegrableOn f (Ioc a b) μ := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    ⊢ Iff (IntervalIntegrable f μ a b) (MeasureTheory.IntegrableOn f (Set.Ioc a b) …
  -/
  rw [intervalIntegrable_iff, uIoc_of_le hab]
  /-
    🎉 no goals
  -/


theorem intervalIntegrable_iff' [NoAtoms μ] :
    IntervalIntegrable f μ a b ↔ IntegrableOn f (uIcc a b) μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    ⊢ Iff (IntervalIntegrable f μ a b) (MeasureTheory.IntegrableOn f (Set.uIcc a b …
  -/
  rw [intervalIntegrable_iff, ← Icc_min_max, uIoc, integrableOn_Icc_iff_integrableOn_Ioc]
  /-
    🎉 no goals
  -/


theorem intervalIntegrable_iff_integrableOn_Icc_of_le {f : ℝ → E} {a b : ℝ} (hab : a ≤ b)
    {μ : Measure ℝ} [NoAtoms μ] : IntervalIntegrable f μ a b ↔ IntegrableOn f (Icc a b) μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hab : LE.le a b
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    ⊢ Iff (IntervalIntegrable f μ a b) (MeasureTheory.IntegrableOn f (Set.Icc a b) …
  -/
  rw [intervalIntegrable_iff_integrableOn_Ioc_of_le hab, integrableOn_Icc_iff_integrableOn_Ioc]
  /-
    🎉 no goals
  -/


theorem intervalIntegrable_iff_integrableOn_Ico_of_le [NoAtoms μ] (hab : a ≤ b) :
    IntervalIntegrable f μ a b ↔ IntegrableOn f (Ico a b) μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    hab : LE.le a b
    ⊢ Iff (IntervalIntegrable f μ a b) (MeasureTheory.IntegrableOn f (Set.Ico a b) …
  -/
  rw [intervalIntegrable_iff_integrableOn_Icc_of_le hab, integrableOn_Icc_iff_integrableOn_Ico]
  /-
    🎉 no goals
  -/


theorem intervalIntegrable_iff_integrableOn_Ioo_of_le [NoAtoms μ] (hab : a ≤ b) :
    IntervalIntegrable f μ a b ↔ IntegrableOn f (Ioo a b) μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.NoAtoms μ
    hab : LE.le a b
    ⊢ Iff (IntervalIntegrable f μ a b) (MeasureTheory.IntegrableOn f (Set.Ioo a b) …
  -/
  rw [intervalIntegrable_iff_integrableOn_Icc_of_le hab, integrableOn_Icc_iff_integrableOn_Ioo]
  /-
    🎉 no goals
  -/


/-- If a function is integrable with respect to a given measure `μ` then it is interval integrable
  with respect to `μ` on `uIcc a b`. -/
theorem MeasureTheory.Integrable.intervalIntegrable (hf : Integrable f μ) :
    IntervalIntegrable f μ a b :=
  ⟨hf.integrableOn, hf.integrableOn⟩


theorem MeasureTheory.IntegrableOn.intervalIntegrable (hf : IntegrableOn f [[a, b]] μ) :
    IntervalIntegrable f μ a b :=
  ⟨MeasureTheory.IntegrableOn.mono_set hf (Ioc_subset_Icc_self.trans Icc_subset_uIcc),
    MeasureTheory.IntegrableOn.mono_set hf (Ioc_subset_Icc_self.trans Icc_subset_uIcc')⟩


theorem intervalIntegrable_const_iff {c : E} :
    IntervalIntegrable (fun _ => c) μ a b ↔ c = 0 ∨ μ (Ι a b) < ∞ := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    a b : Real
    μ : MeasureTheory.Measure Real
    c : E
    ⊢ Iff (IntervalIntegrable (fun x => c) μ a b) (Or (Eq c 0) (LT.lt (μ (Set.uIoc …
  -/
  simp only [intervalIntegrable_iff, integrableOn_const]
  /-
    🎉 no goals
  -/


@[simp]
theorem intervalIntegrable_const [IsLocallyFiniteMeasure μ] {c : E} :
    IntervalIntegrable (fun _ => c) μ a b :=
  intervalIntegrable_const_iff.2 <| Or.inr measure_Ioc_lt_top


@[symm]
nonrec theorem symm (h : IntervalIntegrable f μ a b) : IntervalIntegrable f μ b a :=
  h.symm


@[refl, simp] -- Porting note: added `simp`
                                                /-
                                                  E : Type u_3
                                                  inst✝ : NormedAddCommGroup E
                                                  f : Real → E
                                                  a : Real
                                                  μ : MeasureTheory.Measure Real
                                                  ⊢ IntervalIntegrable f μ a a
                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
theorem refl : IntervalIntegrable f μ a a := by constructor <;> simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[trans]
theorem trans {a b c : ℝ} (hab : IntervalIntegrable f μ a b) (hbc : IntervalIntegrable f μ b c) :
    IntervalIntegrable f μ a c :=
  ⟨(hab.1.union hbc.1).mono_set Ioc_subset_Ioc_union_Ioc,
    (hbc.2.union hab.2).mono_set Ioc_subset_Ioc_union_Ioc⟩


theorem trans_iterate_Ico {a : ℕ → ℝ} {m n : ℕ} (hmn : m ≤ n)
    (hint : ∀ k ∈ Ico m n, IntervalIntegrable f μ (a k) (a <| k + 1)) :
    IntervalIntegrable f μ (a m) (a n) := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a : Nat → Real
    m n : Nat
    hmn : LE.le m n
    hint : ∀ (k : Nat), Membership.mem (Set.Ico m n) k → IntervalIntegrable f μ (a …
    ⊢ IntervalIntegrable f μ (a m) (a n)
  -/
  revert hint
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a : Nat → Real
    m n : Nat
    hmn : LE.le m n
    ⊢ (∀ (k : Nat), Membership.mem (Set.Ico m n) k → IntervalIntegrable f μ (a k)  …
  -/
  refine Nat.le_induction ?_ ?_ n hmn
    /-
      case refine_1
      E : Type u_3
      inst✝ : NormedAddCommGroup E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a : Nat → Real
      m n : Nat
      hmn : LE.le m n
      ⊢ (∀ (k : Nat), Membership.mem (Set.Ico m m) k → IntervalIntegrable f μ (a k)  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_3
      inst✝ : NormedAddCommGroup E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a : Nat → Real
      m n : Nat
      hmn : LE.le m n
      ⊢ ∀ (n : Nat), LE.le m n → ((∀ (k : Nat), Membership.mem (Set.Ico m n) k → Int …
    -/
  · intro p hp IH h
    /-
      case refine_2
      E : Type u_3
      inst✝ : NormedAddCommGroup E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a : Nat → Real
      m n : Nat
      hmn : LE.le m n
      p : Nat
      hp : LE.le m p
      IH : (∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a  …
      h : ∀ (k : Nat), Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k → IntervalIntegr …
      ⊢ IntervalIntegrable f μ (a m) (a (HAdd.hAdd p 1))
    -/
    exact (IH fun k hk => h k (Ico_subset_Ico_right p.le_succ hk)).trans (h p (by simp [hp]))
    /-
      🎉 no goals
    -/


theorem trans_iterate {a : ℕ → ℝ} {n : ℕ}
    (hint : ∀ k < n, IntervalIntegrable f μ (a k) (a <| k + 1)) :
    IntervalIntegrable f μ (a 0) (a n) :=
  trans_iterate_Ico bot_le fun k hk => hint k hk.2


theorem neg (h : IntervalIntegrable f μ a b) : IntervalIntegrable (-f) μ a b :=
  ⟨h.1.neg, h.2.neg⟩


theorem norm (h : IntervalIntegrable f μ a b) : IntervalIntegrable (fun x => ‖f x‖) μ a b :=
  ⟨h.1.norm, h.2.norm⟩


theorem intervalIntegrable_norm_iff {f : ℝ → E} {μ : Measure ℝ} {a b : ℝ}
    (hf : AEStronglyMeasurable f (μ.restrict (Ι a b))) :
    IntervalIntegrable (fun t => ‖f t‖) μ a b ↔ IntervalIntegrable f μ a b := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a b : Real
    hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.uIoc a b))
    ⊢ Iff (IntervalIntegrable (fun t => Norm.norm (f t)) μ a b) (IntervalIntegrabl …
  -/
  simp_rw [intervalIntegrable_iff, IntegrableOn]; exact integrable_norm_iff hf
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem abs {f : ℝ → ℝ} (h : IntervalIntegrable f μ a b) :
    IntervalIntegrable (fun x => |f x|) μ a b :=
  h.norm


theorem mono (hf : IntervalIntegrable f ν a b) (h1 : [[c, d]] ⊆ [[a, b]]) (h2 : μ ≤ ν) :
    IntervalIntegrable f μ c d :=
  intervalIntegrable_iff.mpr <| hf.def'.mono (uIoc_subset_uIoc_of_uIcc_subset_uIcc h1) h2


theorem mono_measure (hf : IntervalIntegrable f ν a b) (h : μ ≤ ν) : IntervalIntegrable f μ a b :=
  hf.mono Subset.rfl h


theorem mono_set (hf : IntervalIntegrable f μ a b) (h : [[c, d]] ⊆ [[a, b]]) :
    IntervalIntegrable f μ c d :=
  hf.mono h le_rfl


theorem mono_set_ae (hf : IntervalIntegrable f μ a b) (h : Ι c d ≤ᵐ[μ] Ι a b) :
    IntervalIntegrable f μ c d :=
  intervalIntegrable_iff.mpr <| hf.def'.mono_set_ae h


theorem mono_set' (hf : IntervalIntegrable f μ a b) (hsub : Ι c d ⊆ Ι a b) :
    IntervalIntegrable f μ c d :=
  hf.mono_set_ae <| Eventually.of_forall hsub


theorem mono_fun [NormedAddCommGroup F] {g : ℝ → F} (hf : IntervalIntegrable f μ a b)
    (hgm : AEStronglyMeasurable g (μ.restrict (Ι a b)))
    (hle : (fun x => ‖g x‖) ≤ᵐ[μ.restrict (Ι a b)] fun x => ‖f x‖) : IntervalIntegrable g μ a b :=
  intervalIntegrable_iff.2 <| hf.def'.integrable.mono hgm hle


theorem mono_fun' {g : ℝ → ℝ} (hg : IntervalIntegrable g μ a b)
    (hfm : AEStronglyMeasurable f (μ.restrict (Ι a b)))
    (hle : (fun x => ‖f x‖) ≤ᵐ[μ.restrict (Ι a b)] g) : IntervalIntegrable f μ a b :=
  intervalIntegrable_iff.2 <| hg.def'.integrable.mono' hfm hle


protected theorem aestronglyMeasurable (h : IntervalIntegrable f μ a b) :
    AEStronglyMeasurable f (μ.restrict (Ioc a b)) :=
  h.1.aestronglyMeasurable


protected theorem aestronglyMeasurable' (h : IntervalIntegrable f μ a b) :
    AEStronglyMeasurable f (μ.restrict (Ioc b a)) :=
  h.2.aestronglyMeasurable


theorem smul [NormedField 𝕜] [NormedSpace 𝕜 E] {f : ℝ → E} {a b : ℝ} {μ : Measure ℝ}
    (h : IntervalIntegrable f μ a b) (r : 𝕜) : IntervalIntegrable (r • f) μ a b :=
  ⟨h.1.smul r, h.2.smul r⟩


@[simp]
theorem add (hf : IntervalIntegrable f μ a b) (hg : IntervalIntegrable g μ a b) :
    IntervalIntegrable (fun x => f x + g x) μ a b :=
  ⟨hf.1.add hg.1, hf.2.add hg.2⟩


@[simp]
theorem sub (hf : IntervalIntegrable f μ a b) (hg : IntervalIntegrable g μ a b) :
    IntervalIntegrable (fun x => f x - g x) μ a b :=
  ⟨hf.1.sub hg.1, hf.2.sub hg.2⟩


theorem sum (s : Finset ι) {f : ι → ℝ → E} (h : ∀ i ∈ s, IntervalIntegrable (f i) μ a b) :
    IntervalIntegrable (∑ i ∈ s, f i) μ a b :=
  ⟨integrable_finset_sum' s fun i hi => (h i hi).1, integrable_finset_sum' s fun i hi => (h i hi).2⟩


theorem mul_continuousOn {f g : ℝ → A} (hf : IntervalIntegrable f μ a b)
    (hg : ContinuousOn g [[a, b]]) : IntervalIntegrable (fun x => f x * g x) μ a b := by
  /-
    A : Type u_5
    inst✝ : NormedRing A
    a b : Real
    μ : MeasureTheory.Measure Real
    f g : Real → A
    hf : IntervalIntegrable f μ a b
    hg : ContinuousOn g (Set.uIcc a b)
    ⊢ IntervalIntegrable (fun x => HMul.hMul (f x) (g x)) μ a b
  -/
  rw [intervalIntegrable_iff] at hf ⊢
  /-
    A : Type u_5
    inst✝ : NormedRing A
    a b : Real
    μ : MeasureTheory.Measure Real
    f g : Real → A
    hf : MeasureTheory.IntegrableOn f (Set.uIoc a b) μ
    hg : ContinuousOn g (Set.uIcc a b)
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (f x) (g x)) (Set.uIoc a b) μ
  -/
  exact hf.mul_continuousOn_of_subset hg measurableSet_Ioc isCompact_uIcc Ioc_subset_Icc_self
  /-
    🎉 no goals
  -/


theorem continuousOn_mul {f g : ℝ → A} (hf : IntervalIntegrable f μ a b)
    (hg : ContinuousOn g [[a, b]]) : IntervalIntegrable (fun x => g x * f x) μ a b := by
  /-
    A : Type u_5
    inst✝ : NormedRing A
    a b : Real
    μ : MeasureTheory.Measure Real
    f g : Real → A
    hf : IntervalIntegrable f μ a b
    hg : ContinuousOn g (Set.uIcc a b)
    ⊢ IntervalIntegrable (fun x => HMul.hMul (g x) (f x)) μ a b
  -/
  rw [intervalIntegrable_iff] at hf ⊢
  /-
    A : Type u_5
    inst✝ : NormedRing A
    a b : Real
    μ : MeasureTheory.Measure Real
    f g : Real → A
    hf : MeasureTheory.IntegrableOn f (Set.uIoc a b) μ
    hg : ContinuousOn g (Set.uIcc a b)
    ⊢ MeasureTheory.IntegrableOn (fun x => HMul.hMul (g x) (f x)) (Set.uIoc a b) μ
  -/
  exact hf.continuousOn_mul_of_subset hg isCompact_uIcc measurableSet_Ioc Ioc_subset_Icc_self
  /-
    🎉 no goals
  -/


@[simp]
theorem const_mul {f : ℝ → A} (hf : IntervalIntegrable f μ a b) (c : A) :
    IntervalIntegrable (fun x => c * f x) μ a b :=
  hf.continuousOn_mul continuousOn_const


@[simp]
theorem mul_const {f : ℝ → A} (hf : IntervalIntegrable f μ a b) (c : A) :
    IntervalIntegrable (fun x => f x * c) μ a b :=
  hf.mul_continuousOn continuousOn_const


@[simp]
theorem div_const {𝕜 : Type*} {f : ℝ → 𝕜} [NormedField 𝕜] (h : IntervalIntegrable f μ a b)
    (c : 𝕜) : IntervalIntegrable (fun x => f x / c) μ a b := by
  /-
    a b : Real
    μ : MeasureTheory.Measure Real
    𝕜 : Type u_6
    f : Real → 𝕜
    inst✝ : NormedField 𝕜
    h : IntervalIntegrable f μ a b
    c : 𝕜
    ⊢ IntervalIntegrable (fun x => HDiv.hDiv (f x) c) μ a b
  -/
  simpa only [div_eq_mul_inv] using mul_const h c⁻¹
  /-
    🎉 no goals
  -/


theorem comp_mul_left (hf : IntervalIntegrable f volume a b) (c : ℝ) :
    IntervalIntegrable (fun x => f (c * x)) volume (a / c) (b / c) := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    c : Real
    ⊢ IntervalIntegrable (fun x => f (HMul.hMul c x)) MeasureTheory.MeasureSpace.v …
  -/
  rcases eq_or_ne c 0 with (hc | hc); · rw [hc]; simp
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case inr
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    c : Real
    hc : Ne c 0
    ⊢ IntervalIntegrable (fun x => f (HMul.hMul c x)) MeasureTheory.MeasureSpace.v …
  -/
  rw [intervalIntegrable_iff'] at hf ⊢
  have A : MeasurableEmbedding fun x => x * c⁻¹ :=
    (Homeomorph.mulRight₀ _ (inv_ne_zero hc)).isClosedEmbedding.measurableEmbedding
  rw [← Real.smul_map_volume_mul_right (inv_ne_zero hc), IntegrableOn, Measure.restrict_smul,
    integrable_smul_measure (by simpa : ENNReal.ofReal |c⁻¹| ≠ 0) ENNReal.ofReal_ne_top,
    ← IntegrableOn, MeasurableEmbedding.integrableOn_map_iff A]
  /-
    case inr
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hf : MeasureTheory.IntegrableOn f (Set.uIcc a b) MeasureTheory.MeasureSpace.vo …
    c : Real
    hc : Ne c 0
    A : MeasurableEmbedding fun x => HMul.hMul x (Inv.inv c)
    ⊢ MeasureTheory.IntegrableOn (Function.comp (fun x => f (HMul.hMul c x)) fun x …
  -/
  convert hf using 1
    /-
      case h.e'_6
      E : Type u_3
      inst✝ : NormedAddCommGroup E
      f : Real → E
      a b : Real
      hf : MeasureTheory.IntegrableOn f (Set.uIcc a b) MeasureTheory.MeasureSpace.vo …
      c : Real
      hc : Ne c 0
      A : MeasurableEmbedding fun x => HMul.hMul x (Inv.inv c)
      ⊢ Eq (Function.comp (fun x => f (HMul.hMul c x)) fun x => HMul.hMul x (Inv.inv …
    -/
  · ext; simp only [comp_apply]; congr 1; field_simp
                                          /-
                                            🎉 no goals
                                          -/
    /-
      case h.e'_7
      E : Type u_3
      inst✝ : NormedAddCommGroup E
      f : Real → E
      a b : Real
      hf : MeasureTheory.IntegrableOn f (Set.uIcc a b) MeasureTheory.MeasureSpace.vo …
      c : Real
      hc : Ne c 0
      A : MeasurableEmbedding fun x => HMul.hMul x (Inv.inv c)
      ⊢ Eq (Set.preimage (fun x => HMul.hMul x (Inv.inv c)) (Set.uIcc (HDiv.hDiv a c …
    -/
  · rw [preimage_mul_const_uIcc (inv_ne_zero hc)]; field_simp [hc]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem comp_mul_left_iff {c : ℝ} (hc : c ≠ 0) :
    IntervalIntegrable (fun x ↦ f (c * x)) volume (a / c) (b / c) ↔
      IntervalIntegrable f volume a b :=
              /-
                E : Type u_3
                inst✝ : NormedAddCommGroup E
                f : Real → E
                a b c : Real
                hc : Ne c 0
                h : IntervalIntegrable (fun x => f (HMul.hMul c x)) MeasureTheory.MeasureSpace …
                ⊢ IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
              -/
  ⟨fun h ↦ by simpa [hc] using h.comp_mul_left c⁻¹, (comp_mul_left · c)⟩
              /-
                🎉 no goals
              -/


theorem comp_mul_right (hf : IntervalIntegrable f volume a b) (c : ℝ) :
    IntervalIntegrable (fun x => f (x * c)) volume (a / c) (b / c) := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    c : Real
    ⊢ IntervalIntegrable (fun x => f (HMul.hMul x c)) MeasureTheory.MeasureSpace.v …
  -/
  simpa only [mul_comm] using comp_mul_left hf c
  /-
    🎉 no goals
  -/


theorem comp_add_right (hf : IntervalIntegrable f volume a b) (c : ℝ) :
    IntervalIntegrable (fun x => f (x + c)) volume (a - c) (b - c) := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    c : Real
    ⊢ IntervalIntegrable (fun x => f (HAdd.hAdd x c)) MeasureTheory.MeasureSpace.v …
  -/
  wlog h : a ≤ b generalizing a b
    /-
      case inr
      E : Type u_3
      inst✝ : NormedAddCommGroup E
      f : Real → E
      a b : Real
      hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
      c : Real
      this : ∀ {a b : Real}, IntervalIntegrable f MeasureTheory.MeasureSpace.volume  …
      h : Not (LE.le a b)
      ⊢ IntervalIntegrable (fun x => f (HAdd.hAdd x c)) MeasureTheory.MeasureSpace.v …
    -/
  · exact IntervalIntegrable.symm (this hf.symm (le_of_not_le h))
    /-
      🎉 no goals
    -/
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a✝ b✝ c a b : Real
    hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    h : LE.le a b
    ⊢ IntervalIntegrable (fun x => f (HAdd.hAdd x c)) MeasureTheory.MeasureSpace.v …
  -/
  rw [intervalIntegrable_iff'] at hf ⊢
  have A : MeasurableEmbedding fun x => x + c :=
    (Homeomorph.addRight c).isClosedEmbedding.measurableEmbedding
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a✝ b✝ c a b : Real
    hf : MeasureTheory.IntegrableOn f (Set.uIcc a b) MeasureTheory.MeasureSpace.vo …
    h : LE.le a b
    A : MeasurableEmbedding fun x => HAdd.hAdd x c
    ⊢ MeasureTheory.IntegrableOn (fun x => f (HAdd.hAdd x c)) (Set.uIcc (HSub.hSub …
  -/
  rw [← map_add_right_eq_self volume c] at hf
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a✝ b✝ c a b : Real
    hf : MeasureTheory.IntegrableOn f (Set.uIcc a b) (MeasureTheory.Measure.map (f …
    h : LE.le a b
    A : MeasurableEmbedding fun x => HAdd.hAdd x c
    ⊢ MeasureTheory.IntegrableOn (fun x => f (HAdd.hAdd x c)) (Set.uIcc (HSub.hSub …
  -/
  convert (MeasurableEmbedding.integrableOn_map_iff A).mp hf using 1
  /-
    case h.e'_7
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a✝ b✝ c a b : Real
    hf : MeasureTheory.IntegrableOn f (Set.uIcc a b) (MeasureTheory.Measure.map (f …
    h : LE.le a b
    A : MeasurableEmbedding fun x => HAdd.hAdd x c
    ⊢ Eq (Set.uIcc (HSub.hSub a c) (HSub.hSub b c)) (Set.preimage (fun x => HAdd.h …
  -/
  rw [preimage_add_const_uIcc]
  /-
    🎉 no goals
  -/


theorem comp_add_left (hf : IntervalIntegrable f volume a b) (c : ℝ) :
    IntervalIntegrable (fun x => f (c + x)) volume (a - c) (b - c) := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    c : Real
    ⊢ IntervalIntegrable (fun x => f (HAdd.hAdd c x)) MeasureTheory.MeasureSpace.v …
  -/
  simpa only [add_comm] using IntervalIntegrable.comp_add_right hf c
  /-
    🎉 no goals
  -/


theorem comp_sub_right (hf : IntervalIntegrable f volume a b) (c : ℝ) :
    IntervalIntegrable (fun x => f (x - c)) volume (a + c) (b + c) := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    c : Real
    ⊢ IntervalIntegrable (fun x => f (HSub.hSub x c)) MeasureTheory.MeasureSpace.v …
  -/
  simpa only [sub_neg_eq_add] using IntervalIntegrable.comp_add_right hf (-c)
  /-
    🎉 no goals
  -/


theorem iff_comp_neg :
    IntervalIntegrable f volume a b ↔ IntervalIntegrable (fun x => f (-x)) volume (-a) (-b) := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    ⊢ Iff (IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b) (IntervalIn …
  -/
  rw [← comp_mul_left_iff (neg_ne_zero.2 one_ne_zero)]; simp [div_neg]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem comp_sub_left (hf : IntervalIntegrable f volume a b) (c : ℝ) :
    IntervalIntegrable (fun x => f (c - x)) volume (c - a) (c - b) := by
  /-
    E : Type u_3
    inst✝ : NormedAddCommGroup E
    f : Real → E
    a b : Real
    hf : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    c : Real
    ⊢ IntervalIntegrable (fun x => f (HSub.hSub c x)) MeasureTheory.MeasureSpace.v …
  -/
  simpa only [neg_sub, ← sub_eq_add_neg] using iff_comp_neg.mp (hf.comp_add_left c)
  /-
    🎉 no goals
  -/


theorem ContinuousOn.intervalIntegrable {u : ℝ → E} {a b : ℝ} (hu : ContinuousOn u (uIcc a b)) :
    IntervalIntegrable u μ a b :=
  (ContinuousOn.integrableOn_Icc hu).intervalIntegrable


theorem ContinuousOn.intervalIntegrable_of_Icc {u : ℝ → E} {a b : ℝ} (h : a ≤ b)
    (hu : ContinuousOn u (Icc a b)) : IntervalIntegrable u μ a b :=
  ContinuousOn.intervalIntegrable ((uIcc_of_le h).symm ▸ hu)


/-- A continuous function on `ℝ` is `IntervalIntegrable` with respect to any locally finite measure
`ν` on ℝ. -/
theorem Continuous.intervalIntegrable {u : ℝ → E} (hu : Continuous u) (a b : ℝ) :
    IntervalIntegrable u μ a b :=
  hu.continuousOn.intervalIntegrable


theorem MonotoneOn.intervalIntegrable {u : ℝ → E} {a b : ℝ} (hu : MonotoneOn u (uIcc a b)) :
    IntervalIntegrable u μ a b := by
  /-
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝² : ConditionallyCompleteLinearOrder E
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    u : Real → E
    a b : Real
    hu : MonotoneOn u (Set.uIcc a b)
    ⊢ IntervalIntegrable u μ a b
  -/
  rw [intervalIntegrable_iff]
  /-
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    μ : MeasureTheory.Measure Real
    inst✝³ : MeasureTheory.IsLocallyFiniteMeasure μ
    inst✝² : ConditionallyCompleteLinearOrder E
    inst✝¹ : OrderTopology E
    inst✝ : SecondCountableTopology E
    u : Real → E
    a b : Real
    hu : MonotoneOn u (Set.uIcc a b)
    ⊢ MeasureTheory.IntegrableOn u (Set.uIoc a b) μ
  -/
  exact (hu.integrableOn_isCompact isCompact_uIcc).mono_set Ioc_subset_Icc_self
  /-
    🎉 no goals
  -/


theorem AntitoneOn.intervalIntegrable {u : ℝ → E} {a b : ℝ} (hu : AntitoneOn u (uIcc a b)) :
    IntervalIntegrable u μ a b :=
  hu.dual_right.intervalIntegrable


theorem Monotone.intervalIntegrable {u : ℝ → E} {a b : ℝ} (hu : Monotone u) :
    IntervalIntegrable u μ a b :=
  (hu.monotoneOn _).intervalIntegrable


theorem Antitone.intervalIntegrable {u : ℝ → E} {a b : ℝ} (hu : Antitone u) :
    IntervalIntegrable u μ a b :=
  (hu.antitoneOn _).intervalIntegrable


/-- Let `l'` be a measurably generated filter; let `l` be a of filter such that each `s ∈ l'`
eventually includes `Ioc u v` as both `u` and `v` tend to `l`. Let `μ` be a measure finite at `l'`.

Suppose that `f : ℝ → E` has a finite limit at `l' ⊓ ae μ`. Then `f` is interval integrable on
`u..v` provided that both `u` and `v` tend to `l`.

Typeclass instances allow Lean to find `l'` based on `l` but not vice versa, so
`apply Tendsto.eventually_intervalIntegrable_ae` will generate goals `Filter ℝ` and
`TendstoIxxClass Ioc ?m_1 l'`. -/
theorem Filter.Tendsto.eventually_intervalIntegrable_ae {f : ℝ → E} {μ : Measure ℝ}
    {l l' : Filter ℝ} (hfm : StronglyMeasurableAtFilter f l' μ) [TendstoIxxClass Ioc l l']
    [IsMeasurablyGenerated l'] (hμ : μ.FiniteAtFilter l') {c : E} (hf : Tendsto f (l' ⊓ ae μ) (𝓝 c))
    {u v : ι → ℝ} {lt : Filter ι} (hu : Tendsto u lt l) (hv : Tendsto v lt l) :
    ∀ᶠ t in lt, IntervalIntegrable f μ (u t) (v t) :=
  have := (hf.integrableAtFilter_ae hfm hμ).eventually
  ((hu.Ioc hv).eventually this).and <| (hv.Ioc hu).eventually this


/-- Let `l'` be a measurably generated filter; let `l` be a of filter such that each `s ∈ l'`
eventually includes `Ioc u v` as both `u` and `v` tend to `l`. Let `μ` be a measure finite at `l'`.

Suppose that `f : ℝ → E` has a finite limit at `l`. Then `f` is interval integrable on `u..v`
provided that both `u` and `v` tend to `l`.

Typeclass instances allow Lean to find `l'` based on `l` but not vice versa, so
`apply Tendsto.eventually_intervalIntegrable` will generate goals `Filter ℝ` and
`TendstoIxxClass Ioc ?m_1 l'`. -/
theorem Filter.Tendsto.eventually_intervalIntegrable {f : ℝ → E} {μ : Measure ℝ} {l l' : Filter ℝ}
    (hfm : StronglyMeasurableAtFilter f l' μ) [TendstoIxxClass Ioc l l'] [IsMeasurablyGenerated l']
    (hμ : μ.FiniteAtFilter l') {c : E} (hf : Tendsto f l' (𝓝 c)) {u v : ι → ℝ} {lt : Filter ι}
    (hu : Tendsto u lt l) (hv : Tendsto v lt l) : ∀ᶠ t in lt, IntervalIntegrable f μ (u t) (v t) :=
  (hf.mono_left inf_le_left).eventually_intervalIntegrable_ae hfm hμ hu hv


/-- The interval integral `∫ x in a..b, f x ∂μ` is defined
as `∫ x in Ioc a b, f x ∂μ - ∫ x in Ioc b a, f x ∂μ`. If `a ≤ b`, then it equals
`∫ x in Ioc a b, f x ∂μ`, otherwise it equals `-∫ x in Ioc b a, f x ∂μ`. -/
def intervalIntegral (f : ℝ → E) (a b : ℝ) (μ : Measure ℝ) : E :=
  (∫ x in Ioc a b, f x ∂μ) - ∫ x in Ioc b a, f x ∂μ


@[inherit_doc intervalIntegral]
notation3"∫ "(...)" in "a".."b", "r:60:(scoped f => f)" ∂"μ:70 => intervalIntegral r a b μ


/-- The interval integral `∫ x in a..b, f x` is defined
as `∫ x in Ioc a b, f x - ∫ x in Ioc b a, f x`. If `a ≤ b`, then it equals
`∫ x in Ioc a b, f x`, otherwise it equals `-∫ x in Ioc b a, f x`. -/
notation3"∫ "(...)" in "a".."b", "r:60:(scoped f => intervalIntegral f a b volume) => r


@[simp]
                                                            /-
                                                              E : Type u_3
                                                              inst✝¹ : NormedAddCommGroup E
                                                              inst✝ : NormedSpace Real E
                                                              a b : Real
                                                              μ : MeasureTheory.Measure Real
                                                              ⊢ Eq (intervalIntegral (fun x => 0) a b μ) 0
                                                            -/
theorem integral_zero : (∫ _ in a..b, (0 : E) ∂μ) = 0 := by simp [intervalIntegral]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem integral_of_le (h : a ≤ b) : ∫ x in a..b, f x ∂μ = ∫ x in Ioc a b, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    h : LE.le a b
    ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (MeasureTheory.integral (μ.restri …
  -/
  simp [intervalIntegral, h]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_same : ∫ x in a..a, f x ∂μ = 0 :=
  sub_self _


theorem integral_symm (a b) : ∫ x in b..a, f x ∂μ = -∫ x in a..b, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a b : Real
    ⊢ Eq (intervalIntegral (fun x => f x) b a μ) (Neg.neg (intervalIntegral (fun x …
  -/
  simp only [intervalIntegral, neg_sub]
  /-
    🎉 no goals
  -/


theorem integral_of_ge (h : b ≤ a) : ∫ x in a..b, f x ∂μ = -∫ x in Ioc b a, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    h : LE.le b a
    ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (Neg.neg (MeasureTheory.integral  …
  -/
  simp only [integral_symm b, integral_of_le h]
  /-
    🎉 no goals
  -/


theorem intervalIntegral_eq_integral_uIoc (f : ℝ → E) (a b : ℝ) (μ : Measure ℝ) :
    ∫ x in a..b, f x ∂μ = (if a ≤ b then 1 else -1 : ℝ) • ∫ x in Ι a b, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (HSMul.hSMul (ite (LE.le a b) 1 ( …
  -/
  split_ifs with h
    /-
      case pos
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      a b : Real
      μ : MeasureTheory.Measure Real
      h : LE.le a b
      ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (HSMul.hSMul 1 (MeasureTheory.int …
    -/
  · simp only [integral_of_le h, uIoc_of_le h, one_smul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      a b : Real
      μ : MeasureTheory.Measure Real
      h : Not (LE.le a b)
      ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (HSMul.hSMul (-1) (MeasureTheory. …
    -/
  · simp only [integral_of_ge (not_le.1 h).le, uIoc_of_ge (not_le.1 h).le, neg_one_smul]
    /-
      🎉 no goals
    -/


theorem norm_intervalIntegral_eq (f : ℝ → E) (a b : ℝ) (μ : Measure ℝ) :
    ‖∫ x in a..b, f x ∂μ‖ = ‖∫ x in Ι a b, f x ∂μ‖ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    ⊢ Eq (Norm.norm (intervalIntegral (fun x => f x) a b μ)) (Norm.norm (MeasureTh …
  -/
  simp_rw [intervalIntegral_eq_integral_uIoc, norm_smul]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    μ : MeasureTheory.Measure Real
    ⊢ Eq (HMul.hMul (Norm.norm (ite (LE.le a b) 1 (-1))) (Norm.norm (MeasureTheory …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp only [norm_neg, norm_one, one_mul]
                /-
                  🎉 no goals
                -/


theorem abs_intervalIntegral_eq (f : ℝ → ℝ) (a b : ℝ) (μ : Measure ℝ) :
    |∫ x in a..b, f x ∂μ| = |∫ x in Ι a b, f x ∂μ| :=
  norm_intervalIntegral_eq f a b μ


theorem integral_cases (f : ℝ → E) (a b) :
    (∫ x in a..b, f x ∂μ) ∈ ({∫ x in Ι a b, f x ∂μ, -∫ x in Ι a b, f x ∂μ} : Set E) := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    a b : Real
    ⊢ Membership.mem (Insert.insert (MeasureTheory.integral (μ.restrict (Set.uIoc  …
  -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  rw [intervalIntegral_eq_integral_uIoc]; split_ifs <;> simp
                                                        /-
                                                          🎉 no goals
                                                        -/


nonrec theorem integral_undef (h : ¬IntervalIntegrable f μ a b) : ∫ x in a..b, f x ∂μ = 0 := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    h : Not (IntervalIntegrable f μ a b)
    ⊢ Eq (intervalIntegral (fun x => f x) a b μ) 0
  -/
  rw [intervalIntegrable_iff] at h
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    h : Not (MeasureTheory.IntegrableOn f (Set.uIoc a b) μ)
    ⊢ Eq (intervalIntegral (fun x => f x) a b μ) 0
  -/
  rw [intervalIntegral_eq_integral_uIoc, integral_undef h, smul_zero]
  /-
    🎉 no goals
  -/


theorem intervalIntegrable_of_integral_ne_zero {a b : ℝ} {f : ℝ → E} {μ : Measure ℝ}
    (h : (∫ x in a..b, f x ∂μ) ≠ 0) : IntervalIntegrable f μ a b :=
  not_imp_comm.1 integral_undef h


nonrec theorem integral_non_aestronglyMeasurable
    (hf : ¬AEStronglyMeasurable f (μ.restrict (Ι a b))) :
    ∫ x in a..b, f x ∂μ = 0 := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    hf : Not (MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.uIoc a b)))
    ⊢ Eq (intervalIntegral (fun x => f x) a b μ) 0
  -/
  rw [intervalIntegral_eq_integral_uIoc, integral_non_aestronglyMeasurable hf, smul_zero]
  /-
    🎉 no goals
  -/


theorem integral_non_aestronglyMeasurable_of_le (h : a ≤ b)
    (hf : ¬AEStronglyMeasurable f (μ.restrict (Ioc a b))) : ∫ x in a..b, f x ∂μ = 0 :=
                                          /-
                                            E : Type u_3
                                            inst✝¹ : NormedAddCommGroup E
                                            inst✝ : NormedSpace Real E
                                            a b : Real
                                            f : Real → E
                                            μ : MeasureTheory.Measure Real
                                            h : LE.le a b
                                            hf : Not (MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.Ioc a b)))
                                            ⊢ Not (MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.uIoc a b)))
                                          -/
  integral_non_aestronglyMeasurable <| by rwa [uIoc_of_le h]
                                          /-
                                            🎉 no goals
                                          -/


theorem norm_integral_min_max (f : ℝ → E) :
    ‖∫ x in min a b..max a b, f x ∂μ‖ = ‖∫ x in a..b, f x ∂μ‖ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    ⊢ Eq (Norm.norm (intervalIntegral (fun x => f x) (Min.min a b) (Max.max a b) μ …
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total a b <;> simp [*, integral_symm a b]
                         /-
                           🎉 no goals
                         -/


theorem norm_integral_eq_norm_integral_Ioc (f : ℝ → E) :
    ‖∫ x in a..b, f x ∂μ‖ = ‖∫ x in Ι a b, f x ∂μ‖ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    ⊢ Eq (Norm.norm (intervalIntegral (fun x => f x) a b μ)) (Norm.norm (MeasureTh …
  -/
  rw [← norm_integral_min_max, integral_of_le min_le_max, uIoc]
  /-
    🎉 no goals
  -/


theorem abs_integral_eq_abs_integral_uIoc (f : ℝ → ℝ) :
    |∫ x in a..b, f x ∂μ| = |∫ x in Ι a b, f x ∂μ| :=
  norm_integral_eq_norm_integral_Ioc f


theorem norm_integral_le_integral_norm_Ioc : ‖∫ x in a..b, f x ∂μ‖ ≤ ∫ x in Ι a b, ‖f x‖ ∂μ :=
  calc
    ‖∫ x in a..b, f x ∂μ‖ = ‖∫ x in Ι a b, f x ∂μ‖ := norm_integral_eq_norm_integral_Ioc f
    _ ≤ ∫ x in Ι a b, ‖f x‖ ∂μ := norm_integral_le_integral_norm f


theorem norm_integral_le_abs_integral_norm : ‖∫ x in a..b, f x ∂μ‖ ≤ |∫ x in a..b, ‖f x‖ ∂μ| := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    ⊢ LE.le (Norm.norm (intervalIntegral (fun x => f x) a b μ)) (abs (intervalInte …
  -/
  simp only [← Real.norm_eq_abs, norm_integral_eq_norm_integral_Ioc]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (μ.restrict (Set.uIoc a b)) fun x = …
  -/
  exact le_trans (norm_integral_le_integral_norm _) (le_abs_self _)
  /-
    🎉 no goals
  -/


theorem norm_integral_le_integral_norm (h : a ≤ b) :
    ‖∫ x in a..b, f x ∂μ‖ ≤ ∫ x in a..b, ‖f x‖ ∂μ :=
                                                    /-
                                                      E : Type u_3
                                                      inst✝¹ : NormedAddCommGroup E
                                                      inst✝ : NormedSpace Real E
                                                      a b : Real
                                                      f : Real → E
                                                      μ : MeasureTheory.Measure Real
                                                      h : LE.le a b
                                                      ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.uIoc a b)) fun x => Norm.norm (f …
                                                    -/
  norm_integral_le_integral_norm_Ioc.trans_eq <| by rw [uIoc_of_le h, integral_of_le h]
                                                    /-
                                                      🎉 no goals
                                                    -/


nonrec theorem norm_integral_le_of_norm_le {g : ℝ → ℝ} (h : ∀ᵐ t ∂μ.restrict <| Ι a b, ‖f t‖ ≤ g t)
    (hbound : IntervalIntegrable g μ a b) : ‖∫ t in a..b, f t ∂μ‖ ≤ |∫ t in a..b, g t ∂μ| := by
  simp_rw [norm_intervalIntegral_eq, abs_intervalIntegral_eq,
    abs_eq_self.mpr (integral_nonneg_of_ae <| h.mono fun _t ht => (norm_nonneg _).trans ht),
    norm_integral_le_of_norm_le hbound.def' h]


theorem norm_integral_le_of_norm_le_const_ae {a b C : ℝ} {f : ℝ → E}
    (h : ∀ᵐ x, x ∈ Ι a b → ‖f x‖ ≤ C) : ‖∫ x in a..b, f x‖ ≤ C * |b - a| := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b C : Real
    f : Real → E
    h : Filter.Eventually (fun x => Membership.mem (Set.uIoc a b) x → LE.le (Norm. …
    ⊢ LE.le (Norm.norm (intervalIntegral (fun x => f x) a b MeasureTheory.MeasureS …
  -/
  rw [norm_integral_eq_norm_integral_Ioc]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b C : Real
    f : Real → E
    h : Filter.Eventually (fun x => Membership.mem (Set.uIoc a b) x → LE.le (Norm. …
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume. …
  -/
  convert norm_setIntegral_le_of_norm_le_const_ae'' _ measurableSet_Ioc h using 1
    /-
      case h.e'_4
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b C : Real
      f : Real → E
      h : Filter.Eventually (fun x => Membership.mem (Set.uIoc a b) x → LE.le (Norm. …
      ⊢ Eq (HMul.hMul C (abs (HSub.hSub b a))) (HMul.hMul C (MeasureTheory.MeasureSp …
    -/
  · rw [Real.volume_Ioc, max_sub_min_eq_abs, ENNReal.toReal_ofReal (abs_nonneg _)]
    /-
      🎉 no goals
    -/
    /-
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b C : Real
      f : Real → E
      h : Filter.Eventually (fun x => Membership.mem (Set.uIoc a b) x → LE.le (Norm. …
      ⊢ LT.lt (MeasureTheory.MeasureSpace.volume (Set.Ioc (Min.min a b) (Max.max a b …
    -/
  · simp only [Real.volume_Ioc, ENNReal.ofReal_lt_top]
    /-
      🎉 no goals
    -/


theorem norm_integral_le_of_norm_le_const {a b C : ℝ} {f : ℝ → E} (h : ∀ x ∈ Ι a b, ‖f x‖ ≤ C) :
    ‖∫ x in a..b, f x‖ ≤ C * |b - a| :=
  norm_integral_le_of_norm_le_const_ae <| Eventually.of_forall h


@[simp]
nonrec theorem integral_add (hf : IntervalIntegrable f μ a b) (hg : IntervalIntegrable g μ a b) :
    ∫ x in a..b, f x + g x ∂μ = (∫ x in a..b, f x ∂μ) + ∫ x in a..b, g x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f g : Real → E
    μ : MeasureTheory.Measure Real
    hf : IntervalIntegrable f μ a b
    hg : IntervalIntegrable g μ a b
    ⊢ Eq (intervalIntegral (fun x => HAdd.hAdd (f x) (g x)) a b μ) (HAdd.hAdd (int …
  -/
  simp only [intervalIntegral_eq_integral_uIoc, integral_add hf.def' hg.def', smul_add]
  /-
    🎉 no goals
  -/


nonrec theorem integral_finset_sum {ι} {s : Finset ι} {f : ι → ℝ → E}
    (h : ∀ i ∈ s, IntervalIntegrable (f i) μ a b) :
    ∫ x in a..b, ∑ i ∈ s, f i x ∂μ = ∑ i ∈ s, ∫ x in a..b, f i x ∂μ := by
  simp only [intervalIntegral_eq_integral_uIoc, integral_finset_sum s fun i hi => (h i hi).def',
    Finset.smul_sum]


@[simp]
nonrec theorem integral_neg : ∫ x in a..b, -f x ∂μ = -∫ x in a..b, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    ⊢ Eq (intervalIntegral (fun x => Neg.neg (f x)) a b μ) (Neg.neg (intervalInteg …
  -/
                                              /-
                                                🎉 no goals
                                              -/
  simp only [intervalIntegral, integral_neg]; abel
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem integral_sub (hf : IntervalIntegrable f μ a b) (hg : IntervalIntegrable g μ a b) :
    ∫ x in a..b, f x - g x ∂μ = (∫ x in a..b, f x ∂μ) - ∫ x in a..b, g x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f g : Real → E
    μ : MeasureTheory.Measure Real
    hf : IntervalIntegrable f μ a b
    hg : IntervalIntegrable g μ a b
    ⊢ Eq (intervalIntegral (fun x => HSub.hSub (f x) (g x)) a b μ) (HSub.hSub (int …
  -/
  simpa only [sub_eq_add_neg] using (integral_add hf hg.neg).trans (congr_arg _ integral_neg)
  /-
    🎉 no goals
  -/


@[simp]
nonrec theorem integral_smul {𝕜 : Type*} [NontriviallyNormedField 𝕜] [NormedSpace 𝕜 E]
    [SMulCommClass ℝ 𝕜 E] (r : 𝕜) (f : ℝ → E) :
    ∫ x in a..b, r • f x ∂μ = r • ∫ x in a..b, f x ∂μ := by
  /-
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    𝕜 : Type u_6
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : SMulCommClass Real 𝕜 E
    r : 𝕜
    f : Real → E
    ⊢ Eq (intervalIntegral (fun x => HSMul.hSMul r (f x)) a b μ) (HSMul.hSMul r (i …
  -/
  simp only [intervalIntegral, integral_smul, smul_sub]
  /-
    🎉 no goals
  -/


@[simp]
nonrec theorem integral_smul_const [CompleteSpace E]
    {𝕜 : Type*} [RCLike 𝕜] [NormedSpace 𝕜 E] (f : ℝ → 𝕜) (c : E) :
    ∫ x in a..b, f x • c ∂μ = (∫ x in a..b, f x ∂μ) • c := by
  /-
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝² : CompleteSpace E
    𝕜 : Type u_6
    inst✝¹ : RCLike 𝕜
    inst✝ : NormedSpace 𝕜 E
    f : Real → 𝕜
    c : E
    ⊢ Eq (intervalIntegral (fun x => HSMul.hSMul (f x) c) a b μ) (HSMul.hSMul (int …
  -/
  simp only [intervalIntegral_eq_integral_uIoc, integral_smul_const, smul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_const_mul {𝕜 : Type*} [RCLike 𝕜] (r : 𝕜) (f : ℝ → 𝕜) :
    ∫ x in a..b, r * f x ∂μ = r * ∫ x in a..b, f x ∂μ :=
  integral_smul r f


@[simp]
theorem integral_mul_const {𝕜 : Type*} [RCLike 𝕜] (r : 𝕜) (f : ℝ → 𝕜) :
    ∫ x in a..b, f x * r ∂μ = (∫ x in a..b, f x ∂μ) * r := by
  /-
    a b : Real
    μ : MeasureTheory.Measure Real
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    r : 𝕜
    f : Real → 𝕜
    ⊢ Eq (intervalIntegral (fun x => HMul.hMul (f x) r) a b μ) (HMul.hMul (interva …
  -/
  simpa only [mul_comm r] using integral_const_mul r f
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_div {𝕜 : Type*} [RCLike 𝕜] (r : 𝕜) (f : ℝ → 𝕜) :
    ∫ x in a..b, f x / r ∂μ = (∫ x in a..b, f x ∂μ) / r := by
  /-
    a b : Real
    μ : MeasureTheory.Measure Real
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    r : 𝕜
    f : Real → 𝕜
    ⊢ Eq (intervalIntegral (fun x => HDiv.hDiv (f x) r) a b μ) (HDiv.hDiv (interva …
  -/
  simpa only [div_eq_mul_inv] using integral_mul_const r⁻¹ f
  /-
    🎉 no goals
  -/


theorem integral_const' [CompleteSpace E] (c : E) :
    ∫ _ in a..b, c ∂μ = ((μ <| Ioc a b).toReal - (μ <| Ioc b a).toReal) • c := by
  /-
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝ : CompleteSpace E
    c : E
    ⊢ Eq (intervalIntegral (fun x => c) a b μ) (HSMul.hSMul (HSub.hSub (μ (Set.Ioc …
  -/
  simp only [intervalIntegral, setIntegral_const, sub_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_const [CompleteSpace E] (c : E) : ∫ _ in a..b, c = (b - a) • c := by
  simp only [integral_const', Real.volume_Ioc, ENNReal.toReal_ofReal', ← neg_sub b,
    max_zero_sub_eq_self]


nonrec theorem integral_smul_measure (c : ℝ≥0∞) :
    ∫ x in a..b, f x ∂c • μ = c.toReal • ∫ x in a..b, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    c : ENNReal
    ⊢ Eq (intervalIntegral (fun x => f x) a b (HSMul.hSMul c μ)) (HSMul.hSMul c.to …
  -/
  simp only [intervalIntegral, Measure.restrict_smul, integral_smul_measure, smul_sub]
  /-
    🎉 no goals
  -/


nonrec theorem _root_.RCLike.intervalIntegral_ofReal {𝕜 : Type*} [RCLike 𝕜] {a b : ℝ}
    {μ : Measure ℝ} {f : ℝ → ℝ} : (∫ x in a..b, (f x : 𝕜) ∂μ) = ↑(∫ x in a..b, f x ∂μ) := by
  /-
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → Real
    ⊢ Eq (intervalIntegral (fun x => ↑(f x)) a b μ) ↑(intervalIntegral (fun x => f …
  -/
  simp only [intervalIntegral, integral_ofReal, RCLike.ofReal_sub]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-06")]
alias RCLike.interval_integral_ofReal := RCLike.intervalIntegral_ofReal


nonrec theorem integral_ofReal {a b : ℝ} {μ : Measure ℝ} {f : ℝ → ℝ} :
    (∫ x in a..b, (f x : ℂ) ∂μ) = ↑(∫ x in a..b, f x ∂μ) :=
  RCLike.intervalIntegral_ofReal


theorem _root_.ContinuousLinearMap.intervalIntegral_apply {a b : ℝ} {φ : ℝ → F →L[𝕜] E}
    (hφ : IntervalIntegrable φ μ a b) (v : F) :
    (∫ x in a..b, φ x ∂μ) v = ∫ x in a..b, φ x v ∂μ := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    inst✝³ : RCLike 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    a b : Real
    φ : Real → ContinuousLinearMap (RingHom.id 𝕜) F E
    hφ : IntervalIntegrable φ μ a b
    v : F
    ⊢ Eq ((intervalIntegral (fun x => φ x) a b μ) v) (intervalIntegral (fun x => ( …
  -/
  simp_rw [intervalIntegral_eq_integral_uIoc, ← integral_apply hφ.def' v, coe_smul', Pi.smul_apply]
  /-
    🎉 no goals
  -/


theorem _root_.ContinuousLinearMap.intervalIntegral_comp_comm [CompleteSpace E] (L : E →L[𝕜] F)
    (hf : IntervalIntegrable f μ a b) : (∫ x in a..b, L (f x) ∂μ) = L (∫ x in a..b, f x ∂μ) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    F : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    f : Real → E
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : NormedSpace Real F
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : IntervalIntegrable f μ a b
    ⊢ Eq (intervalIntegral (fun x => L (f x)) a b μ) (L (intervalIntegral (fun x = …
  -/
  simp_rw [intervalIntegral, L.integral_comp_comm hf.1, L.integral_comp_comm hf.2, L.map_sub]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_comp_mul_right (hc : c ≠ 0) :
    (∫ x in a..b, f (x * c)) = c⁻¹ • ∫ x in a * c..b * c, f x := by
  have A : MeasurableEmbedding fun x => x * c :=
    (Homeomorph.mulRight₀ c hc).isClosedEmbedding.measurableEmbedding
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    A : MeasurableEmbedding fun x => HMul.hMul x c
    ⊢ Eq (intervalIntegral (fun x => f (HMul.hMul x c)) a b MeasureTheory.MeasureS …
  -/
  conv_rhs => rw [← Real.smul_map_volume_mul_right hc]
  simp_rw [integral_smul_measure, intervalIntegral, A.setIntegral_map,
    ENNReal.toReal_ofReal (abs_nonneg c)]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    A : MeasurableEmbedding fun x => HMul.hMul x c
    ⊢ Eq (HSub.hSub (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
  -/
  cases' hc.lt_or_lt with h h
  · simp [h, mul_div_cancel_right₀, hc, abs_of_neg,
      Measure.restrict_congr_set (α := ℝ) (μ := volume) Ico_ae_eq_Ioc]
    /-
      case inr
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b c : Real
      f : Real → E
      hc : Ne c 0
      A : MeasurableEmbedding fun x => HMul.hMul x c
      h : LT.lt 0 c
      ⊢ Eq (HSub.hSub (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
    -/
  · simp [h, mul_div_cancel_right₀, hc, abs_of_pos]
    /-
      🎉 no goals
    -/


@[simp]
theorem smul_integral_comp_mul_right (c) :
    (c • ∫ x in a..b, f (x * c)) = ∫ x in a * c..b * c, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c : Real
    ⊢ Eq (HSMul.hSMul c (intervalIntegral (fun x => f (HMul.hMul x c)) a b Measure …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_mul_right]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_mul_left (hc : c ≠ 0) :
    (∫ x in a..b, f (c * x)) = c⁻¹ • ∫ x in c * a..c * b, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    ⊢ Eq (intervalIntegral (fun x => f (HMul.hMul c x)) a b MeasureTheory.MeasureS …
  -/
  simpa only [mul_comm c] using integral_comp_mul_right f hc
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_integral_comp_mul_left (c) :
    (c • ∫ x in a..b, f (c * x)) = ∫ x in c * a..c * b, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c : Real
    ⊢ Eq (HSMul.hSMul c (intervalIntegral (fun x => f (HMul.hMul c x)) a b Measure …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_mul_left]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_div (hc : c ≠ 0) :
    (∫ x in a..b, f (x / c)) = c • ∫ x in a / c..b / c, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    ⊢ Eq (intervalIntegral (fun x => f (HDiv.hDiv x c)) a b MeasureTheory.MeasureS …
  -/
  simpa only [inv_inv] using integral_comp_mul_right f (inv_ne_zero hc)
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_smul_integral_comp_div (c) :
    (c⁻¹ • ∫ x in a..b, f (x / c)) = ∫ x in a / c..b / c, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c : Real
    ⊢ Eq (HSMul.hSMul (Inv.inv c) (intervalIntegral (fun x => f (HDiv.hDiv x c)) a …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_div]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_add_right (d) : (∫ x in a..b, f (x + d)) = ∫ x in a + d..b + d, f x :=
  have A : MeasurableEmbedding fun x => x + d :=
    (Homeomorph.addRight d).isClosedEmbedding.measurableEmbedding
  calc
    (∫ x in a..b, f (x + d)) = ∫ x in a + d..b + d, f x ∂Measure.map (fun x => x + d) volume := by
      /-
        E : Type u_3
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        a b : Real
        f : Real → E
        d : Real
        A : MeasurableEmbedding fun x => HAdd.hAdd x d
        ⊢ Eq (intervalIntegral (fun x => f (HAdd.hAdd x d)) a b MeasureTheory.MeasureS …
      -/
      simp [intervalIntegral, A.setIntegral_map]
      /-
        🎉 no goals
      -/
                                       /-
                                         E : Type u_3
                                         inst✝¹ : NormedAddCommGroup E
                                         inst✝ : NormedSpace Real E
                                         a b : Real
                                         f : Real → E
                                         d : Real
                                         A : MeasurableEmbedding fun x => HAdd.hAdd x d
                                         ⊢ Eq (intervalIntegral (fun x => f x) (HAdd.hAdd a d) (HAdd.hAdd b d) (Measure …
                                       -/
    _ = ∫ x in a + d..b + d, f x := by rw [map_add_right_eq_self]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
nonrec theorem integral_comp_add_left (d) :
    (∫ x in a..b, f (d + x)) = ∫ x in d + a..d + b, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HAdd.hAdd d x)) a b MeasureTheory.MeasureS …
  -/
  simpa only [add_comm d] using integral_comp_add_right f d
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_comp_mul_add (hc : c ≠ 0) (d) :
    (∫ x in a..b, f (c * x + d)) = c⁻¹ • ∫ x in c * a + d..c * b + d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HAdd.hAdd (HMul.hMul c x) d)) a b MeasureT …
  -/
  rw [← integral_comp_add_right, ← integral_comp_mul_left _ hc]
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_integral_comp_mul_add (c d) :
    (c • ∫ x in a..b, f (c * x + d)) = ∫ x in c * a + d..c * b + d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c d : Real
    ⊢ Eq (HSMul.hSMul c (intervalIntegral (fun x => f (HAdd.hAdd (HMul.hMul c x) d …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_mul_add]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_add_mul (hc : c ≠ 0) (d) :
    (∫ x in a..b, f (d + c * x)) = c⁻¹ • ∫ x in d + c * a..d + c * b, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HAdd.hAdd d (HMul.hMul c x))) a b MeasureT …
  -/
  rw [← integral_comp_add_left, ← integral_comp_mul_left _ hc]
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_integral_comp_add_mul (c d) :
    (c • ∫ x in a..b, f (d + c * x)) = ∫ x in d + c * a..d + c * b, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c d : Real
    ⊢ Eq (HSMul.hSMul c (intervalIntegral (fun x => f (HAdd.hAdd d (HMul.hMul c x) …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_add_mul]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_div_add (hc : c ≠ 0) (d) :
    (∫ x in a..b, f (x / c + d)) = c • ∫ x in a / c + d..b / c + d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HAdd.hAdd (HDiv.hDiv x c) d)) a b MeasureT …
  -/
  simpa only [div_eq_inv_mul, inv_inv] using integral_comp_mul_add f (inv_ne_zero hc) d
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_smul_integral_comp_div_add (c d) :
    (c⁻¹ • ∫ x in a..b, f (x / c + d)) = ∫ x in a / c + d..b / c + d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c d : Real
    ⊢ Eq (HSMul.hSMul (Inv.inv c) (intervalIntegral (fun x => f (HAdd.hAdd (HDiv.h …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_div_add]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_add_div (hc : c ≠ 0) (d) :
    (∫ x in a..b, f (d + x / c)) = c • ∫ x in d + a / c..d + b / c, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HAdd.hAdd d (HDiv.hDiv x c))) a b MeasureT …
  -/
  simpa only [div_eq_inv_mul, inv_inv] using integral_comp_add_mul f (inv_ne_zero hc) d
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_smul_integral_comp_add_div (c d) :
    (c⁻¹ • ∫ x in a..b, f (d + x / c)) = ∫ x in d + a / c..d + b / c, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c d : Real
    ⊢ Eq (HSMul.hSMul (Inv.inv c) (intervalIntegral (fun x => f (HAdd.hAdd d (HDiv …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_add_div]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_mul_sub (hc : c ≠ 0) (d) :
    (∫ x in a..b, f (c * x - d)) = c⁻¹ • ∫ x in c * a - d..c * b - d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HSub.hSub (HMul.hMul c x) d)) a b MeasureT …
  -/
  simpa only [sub_eq_add_neg] using integral_comp_mul_add f hc (-d)
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_integral_comp_mul_sub (c d) :
    (c • ∫ x in a..b, f (c * x - d)) = ∫ x in c * a - d..c * b - d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c d : Real
    ⊢ Eq (HSMul.hSMul c (intervalIntegral (fun x => f (HSub.hSub (HMul.hMul c x) d …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_mul_sub]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_sub_mul (hc : c ≠ 0) (d) :
    (∫ x in a..b, f (d - c * x)) = c⁻¹ • ∫ x in d - c * b..d - c * a, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HSub.hSub d (HMul.hMul c x))) a b MeasureT …
  -/
  simp only [sub_eq_add_neg, neg_mul_eq_neg_mul]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HAdd.hAdd d (HMul.hMul (Neg.neg c) x))) a  …
  -/
  rw [integral_comp_add_mul f (neg_ne_zero.mpr hc) d, integral_symm]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (HSMul.hSMul (Inv.inv (Neg.neg c)) (Neg.neg (intervalIntegral (fun x => f …
  -/
  simp only [inv_neg, smul_neg, neg_neg, neg_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem smul_integral_comp_sub_mul (c d) :
    (c • ∫ x in a..b, f (d - c * x)) = ∫ x in d - c * b..d - c * a, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c d : Real
    ⊢ Eq (HSMul.hSMul c (intervalIntegral (fun x => f (HSub.hSub d (HMul.hMul c x) …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_sub_mul]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_div_sub (hc : c ≠ 0) (d) :
    (∫ x in a..b, f (x / c - d)) = c • ∫ x in a / c - d..b / c - d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HSub.hSub (HDiv.hDiv x c) d)) a b MeasureT …
  -/
  simpa only [div_eq_inv_mul, inv_inv] using integral_comp_mul_sub f (inv_ne_zero hc) d
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_smul_integral_comp_div_sub (c d) :
    (c⁻¹ • ∫ x in a..b, f (x / c - d)) = ∫ x in a / c - d..b / c - d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c d : Real
    ⊢ Eq (HSMul.hSMul (Inv.inv c) (intervalIntegral (fun x => f (HSub.hSub (HDiv.h …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_div_sub]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_sub_div (hc : c ≠ 0) (d) :
    (∫ x in a..b, f (d - x / c)) = c • ∫ x in d - b / c..d - a / c, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    hc : Ne c 0
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HSub.hSub d (HDiv.hDiv x c))) a b MeasureT …
  -/
  simpa only [div_eq_inv_mul, inv_inv] using integral_comp_sub_mul f (inv_ne_zero hc) d
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_smul_integral_comp_sub_div (c d) :
    (c⁻¹ • ∫ x in a..b, f (d - x / c)) = ∫ x in d - b / c..d - a / c, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    c d : Real
    ⊢ Eq (HSMul.hSMul (Inv.inv c) (intervalIntegral (fun x => f (HSub.hSub d (HDiv …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hc : c = 0 <;> simp [hc, integral_comp_sub_div]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem integral_comp_sub_right (d) : (∫ x in a..b, f (x - d)) = ∫ x in a - d..b - d, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HSub.hSub x d)) a b MeasureTheory.MeasureS …
  -/
  simpa only [sub_eq_add_neg] using integral_comp_add_right f (-d)
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_comp_sub_left (d) : (∫ x in a..b, f (d - x)) = ∫ x in d - b..d - a, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    d : Real
    ⊢ Eq (intervalIntegral (fun x => f (HSub.hSub d x)) a b MeasureTheory.MeasureS …
  -/
  simpa only [one_mul, one_smul, inv_one] using integral_comp_sub_mul f one_ne_zero d
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_comp_neg : (∫ x in a..b, f (-x)) = ∫ x in -b..-a, f x := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    ⊢ Eq (intervalIntegral (fun x => f (Neg.neg x)) a b MeasureTheory.MeasureSpace …
  -/
  simpa only [zero_sub] using integral_comp_sub_left f 0
  /-
    🎉 no goals
  -/


/-- If two functions are equal in the relevant interval, their interval integrals are also equal. -/
theorem integral_congr {a b : ℝ} (h : EqOn f g [[a, b]]) :
    ∫ x in a..b, f x ∂μ = ∫ x in a..b, g x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : Real → E
    μ : MeasureTheory.Measure Real
    a b : Real
    h : Set.EqOn f g (Set.uIcc a b)
    ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (intervalIntegral (fun x => g x)  …
  -/
  rcases le_total a b with hab | hab <;>
    simpa [hab, integral_of_le, integral_of_ge] using
      setIntegral_congr_fun measurableSet_Ioc (h.mono Ioc_subset_Icc_self)


theorem integral_add_adjacent_intervals_cancel (hab : IntervalIntegrable f μ a b)
    (hbc : IntervalIntegrable f μ b c) :
    (((∫ x in a..b, f x ∂μ) + ∫ x in b..c, f x ∂μ) + ∫ x in c..a, f x ∂μ) = 0 := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    hab : IntervalIntegrable f μ a b
    hbc : IntervalIntegrable f μ b c
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (intervalIntegral (fun x => f x) a b μ) (intervalIn …
  -/
  have hac := hab.trans hbc
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    hab : IntervalIntegrable f μ a b
    hbc : IntervalIntegrable f μ b c
    hac : IntervalIntegrable f μ a c
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (intervalIntegral (fun x => f x) a b μ) (intervalIn …
  -/
  simp only [intervalIntegral, sub_add_sub_comm, sub_eq_zero]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    hab : IntervalIntegrable f μ a b
    hbc : IntervalIntegrable f μ b c
    hac : IntervalIntegrable f μ a c
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (MeasureTheory.integral (μ.restrict (Set.Ioc a b))  …
  -/
  iterate 4 rw [← setIntegral_union]
    /-
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b c : Real
      f : Real → E
      μ : MeasureTheory.Measure Real
      hab : IntervalIntegrable f μ a b
      hbc : IntervalIntegrable f μ b c
      hac : IntervalIntegrable f μ a c
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union (Union.union (Set.Ioc a  …
    -/
  · suffices Ioc a b ∪ Ioc b c ∪ Ioc c a = Ioc b a ∪ Ioc c b ∪ Ioc a c by rw [this]
    rw [Ioc_union_Ioc_union_Ioc_cycle, union_right_comm, Ioc_union_Ioc_union_Ioc_cycle,
      min_left_comm, max_left_comm]
  all_goals
    simp [*, MeasurableSet.union, measurableSet_Ioc, Ioc_disjoint_Ioc_same,
      Ioc_disjoint_Ioc_same.symm, hab.1, hab.2, hbc.1, hbc.2, hac.1, hac.2]


theorem integral_add_adjacent_intervals (hab : IntervalIntegrable f μ a b)
    (hbc : IntervalIntegrable f μ b c) :
    ((∫ x in a..b, f x ∂μ) + ∫ x in b..c, f x ∂μ) = ∫ x in a..c, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b c : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    hab : IntervalIntegrable f μ a b
    hbc : IntervalIntegrable f μ b c
    ⊢ Eq (HAdd.hAdd (intervalIntegral (fun x => f x) a b μ) (intervalIntegral (fun …
  -/
  rw [← add_neg_eq_zero, ← integral_symm, integral_add_adjacent_intervals_cancel hab hbc]
  /-
    🎉 no goals
  -/


theorem sum_integral_adjacent_intervals_Ico {a : ℕ → ℝ} {m n : ℕ} (hmn : m ≤ n)
    (hint : ∀ k ∈ Ico m n, IntervalIntegrable f μ (a k) (a <| k + 1)) :
    ∑ k ∈ Finset.Ico m n, ∫ x in a k..a <| k + 1, f x ∂μ = ∫ x in a m..a n, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a : Nat → Real
    m n : Nat
    hmn : LE.le m n
    hint : ∀ (k : Nat), Membership.mem (Set.Ico m n) k → IntervalIntegrable f μ (a …
    ⊢ Eq ((Finset.Ico m n).sum fun k => intervalIntegral (fun x => f x) (a k) (a ( …
  -/
  revert hint
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a : Nat → Real
    m n : Nat
    hmn : LE.le m n
    ⊢ (∀ (k : Nat), Membership.mem (Set.Ico m n) k → IntervalIntegrable f μ (a k)  …
  -/
  refine Nat.le_induction ?_ ?_ n hmn
    /-
      case refine_1
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a : Nat → Real
      m n : Nat
      hmn : LE.le m n
      ⊢ (∀ (k : Nat), Membership.mem (Set.Ico m m) k → IntervalIntegrable f μ (a k)  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a : Nat → Real
      m n : Nat
      hmn : LE.le m n
      ⊢ ∀ (n : Nat), LE.le m n → ((∀ (k : Nat), Membership.mem (Set.Ico m n) k → Int …
    -/
  · intro p hmp IH h
    /-
      case refine_2
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a : Nat → Real
      m n : Nat
      hmn : LE.le m n
      p : Nat
      hmp : LE.le m p
      IH : (∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a  …
      h : ∀ (k : Nat), Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k → IntervalIntegr …
      ⊢ Eq ((Finset.Ico m (HAdd.hAdd p 1)).sum fun k => intervalIntegral (fun x => f …
    -/
    rw [Finset.sum_Ico_succ_top hmp, IH, integral_add_adjacent_intervals]
      /-
        case refine_2.hab
        E : Type u_3
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : Real → E
        μ : MeasureTheory.Measure Real
        a : Nat → Real
        m n : Nat
        hmn : LE.le m n
        p : Nat
        hmp : LE.le m p
        IH : (∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a  …
        h : ∀ (k : Nat), Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k → IntervalIntegr …
        ⊢ IntervalIntegrable f μ (a m) (a p)
      -/
    · refine IntervalIntegrable.trans_iterate_Ico hmp fun k hk => h k ?_
      /-
        case refine_2.hab
        E : Type u_3
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : Real → E
        μ : MeasureTheory.Measure Real
        a : Nat → Real
        m n : Nat
        hmn : LE.le m n
        p : Nat
        hmp : LE.le m p
        IH : (∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a  …
        h : ∀ (k : Nat), Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k → IntervalIntegr …
        k : Nat
        hk : Membership.mem (Set.Ico m p) k
        ⊢ Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k
      -/
      exact (Ico_subset_Ico le_rfl (Nat.le_succ _)) hk
      /-
        🎉 no goals
      -/
      /-
        case refine_2.hbc
        E : Type u_3
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : Real → E
        μ : MeasureTheory.Measure Real
        a : Nat → Real
        m n : Nat
        hmn : LE.le m n
        p : Nat
        hmp : LE.le m p
        IH : (∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a  …
        h : ∀ (k : Nat), Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k → IntervalIntegr …
        ⊢ IntervalIntegrable f μ (a p) (a (HAdd.hAdd p 1))
      -/
    · apply h
      /-
        case refine_2.hbc.a
        E : Type u_3
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : Real → E
        μ : MeasureTheory.Measure Real
        a : Nat → Real
        m n : Nat
        hmn : LE.le m n
        p : Nat
        hmp : LE.le m p
        IH : (∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a  …
        h : ∀ (k : Nat), Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k → IntervalIntegr …
        ⊢ Membership.mem (Set.Ico m (HAdd.hAdd p 1)) p
      -/
      simp [hmp]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        E : Type u_3
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : Real → E
        μ : MeasureTheory.Measure Real
        a : Nat → Real
        m n : Nat
        hmn : LE.le m n
        p : Nat
        hmp : LE.le m p
        IH : (∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a  …
        h : ∀ (k : Nat), Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k → IntervalIntegr …
        ⊢ ∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a k) ( …
      -/
    · intro k hk
      /-
        case refine_2
        E : Type u_3
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : Real → E
        μ : MeasureTheory.Measure Real
        a : Nat → Real
        m n : Nat
        hmn : LE.le m n
        p : Nat
        hmp : LE.le m p
        IH : (∀ (k : Nat), Membership.mem (Set.Ico m p) k → IntervalIntegrable f μ (a  …
        h : ∀ (k : Nat), Membership.mem (Set.Ico m (HAdd.hAdd p 1)) k → IntervalIntegr …
        k : Nat
        hk : Membership.mem (Set.Ico m p) k
        ⊢ IntervalIntegrable f μ (a k) (a (HAdd.hAdd k 1))
      -/
      exact h _ (Ico_subset_Ico_right p.le_succ hk)
      /-
        🎉 no goals
      -/


theorem sum_integral_adjacent_intervals {a : ℕ → ℝ} {n : ℕ}
    (hint : ∀ k < n, IntervalIntegrable f μ (a k) (a <| k + 1)) :
    ∑ k ∈ Finset.range n, ∫ x in a k..a <| k + 1, f x ∂μ = ∫ x in (a 0)..(a n), f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a : Nat → Real
    n : Nat
    hint : ∀ (k : Nat), LT.lt k n → IntervalIntegrable f μ (a k) (a (HAdd.hAdd k 1))
    ⊢ Eq ((Finset.range n).sum fun k => intervalIntegral (fun x => f x) (a k) (a ( …
  -/
  rw [← Nat.Ico_zero_eq_range]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a : Nat → Real
    n : Nat
    hint : ∀ (k : Nat), LT.lt k n → IntervalIntegrable f μ (a k) (a (HAdd.hAdd k 1))
    ⊢ Eq ((Finset.Ico 0 n).sum fun k => intervalIntegral (fun x => f x) (a k) (a ( …
  -/
  exact sum_integral_adjacent_intervals_Ico (zero_le n) fun k hk => hint k hk.2
  /-
    🎉 no goals
  -/


theorem integral_interval_sub_left (hab : IntervalIntegrable f μ a b)
    (hac : IntervalIntegrable f μ a c) :
    ((∫ x in a..b, f x ∂μ) - ∫ x in a..c, f x ∂μ) = ∫ x in c..b, f x ∂μ :=
  sub_eq_of_eq_add' <| Eq.symm <| integral_add_adjacent_intervals hac (hac.symm.trans hab)


theorem integral_interval_add_interval_comm (hab : IntervalIntegrable f μ a b)
    (hcd : IntervalIntegrable f μ c d) (hac : IntervalIntegrable f μ a c) :
    ((∫ x in a..b, f x ∂μ) + ∫ x in c..d, f x ∂μ) =
      (∫ x in a..d, f x ∂μ) + ∫ x in c..b, f x ∂μ := by
  rw [← integral_add_adjacent_intervals hac hcd, add_assoc, add_left_comm,
    integral_add_adjacent_intervals hac (hac.symm.trans hab), add_comm]


theorem integral_interval_sub_interval_comm (hab : IntervalIntegrable f μ a b)
    (hcd : IntervalIntegrable f μ c d) (hac : IntervalIntegrable f μ a c) :
    ((∫ x in a..b, f x ∂μ) - ∫ x in c..d, f x ∂μ) =
      (∫ x in a..c, f x ∂μ) - ∫ x in b..d, f x ∂μ := by
  simp only [sub_eq_add_neg, ← integral_symm,
    integral_interval_add_interval_comm hab hcd.symm (hac.trans hcd)]


theorem integral_interval_sub_interval_comm' (hab : IntervalIntegrable f μ a b)
    (hcd : IntervalIntegrable f μ c d) (hac : IntervalIntegrable f μ a c) :
    ((∫ x in a..b, f x ∂μ) - ∫ x in c..d, f x ∂μ) =
      (∫ x in d..b, f x ∂μ) - ∫ x in c..a, f x ∂μ := by
  rw [integral_interval_sub_interval_comm hab hcd hac, integral_symm b d, integral_symm a c,
    sub_neg_eq_add, sub_eq_neg_add]


theorem integral_Iic_sub_Iic (ha : IntegrableOn f (Iic a) μ) (hb : IntegrableOn f (Iic b) μ) :
    ((∫ x in Iic b, f x ∂μ) - ∫ x in Iic a, f x ∂μ) = ∫ x in a..b, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    ha : MeasureTheory.IntegrableOn f (Set.Iic a) μ
    hb : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    ⊢ Eq (HSub.hSub (MeasureTheory.integral (μ.restrict (Set.Iic b)) fun x => f x) …
  -/
  wlog hab : a ≤ b generalizing a b
    /-
      case inr
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      a b : Real
      f : Real → E
      μ : MeasureTheory.Measure Real
      ha : MeasureTheory.IntegrableOn f (Set.Iic a) μ
      hb : MeasureTheory.IntegrableOn f (Set.Iic b) μ
      this : ∀ {a b : Real}, MeasureTheory.IntegrableOn f (Set.Iic a) μ → MeasureThe …
      hab : Not (LE.le a b)
      ⊢ Eq (HSub.hSub (MeasureTheory.integral (μ.restrict (Set.Iic b)) fun x => f x) …
    -/
  · rw [integral_symm, ← this hb ha (le_of_not_le hab), neg_sub]
    /-
      🎉 no goals
    -/
  rw [sub_eq_iff_eq_add', integral_of_le hab, ← setIntegral_union (Iic_disjoint_Ioc le_rfl),
    Iic_union_Ioc_eq_Iic hab]
  /-
    case ht
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a✝ b✝ : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    a b : Real
    ha : MeasureTheory.IntegrableOn f (Set.Iic a) μ
    hb : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    hab : LE.le a b
    ⊢ MeasurableSet (Set.Ioc a b)
  -/
  exacts [measurableSet_Ioc, ha, hb.mono_set fun _ => And.right]
  /-
    🎉 no goals
  -/


theorem integral_Iic_add_Ioi (h_left : IntegrableOn f (Iic b) μ)
    (h_right : IntegrableOn f (Ioi b) μ) :
    (∫ x in Iic b, f x ∂μ) + (∫ x in Ioi b, f x ∂μ) = ∫ (x : ℝ), f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    h_left : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    h_right : MeasureTheory.IntegrableOn f (Set.Ioi b) μ
    ⊢ Eq (HAdd.hAdd (MeasureTheory.integral (μ.restrict (Set.Iic b)) fun x => f x) …
  -/
  convert (setIntegral_union (Iic_disjoint_Ioi <| Eq.le rfl) measurableSet_Ioi h_left h_right).symm
  /-
    case h.e'_3.h.e'_6
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    h_left : MeasureTheory.IntegrableOn f (Set.Iic b) μ
    h_right : MeasureTheory.IntegrableOn f (Set.Ioi b) μ
    ⊢ Eq μ (μ.restrict (Union.union (Set.Iic b) (Set.Ioi b)))
  -/
  rw [Iic_union_Ioi, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


theorem integral_Iio_add_Ici (h_left : IntegrableOn f (Iio b) μ)
    (h_right : IntegrableOn f (Ici b) μ) :
    (∫ x in Iio b, f x ∂μ) + (∫ x in Ici b, f x ∂μ) = ∫ (x : ℝ), f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    h_left : MeasureTheory.IntegrableOn f (Set.Iio b) μ
    h_right : MeasureTheory.IntegrableOn f (Set.Ici b) μ
    ⊢ Eq (HAdd.hAdd (MeasureTheory.integral (μ.restrict (Set.Iio b)) fun x => f x) …
  -/
  convert (setIntegral_union (Iio_disjoint_Ici <| Eq.le rfl) measurableSet_Ici h_left h_right).symm
  /-
    case h.e'_3.h.e'_6
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : Real
    f : Real → E
    μ : MeasureTheory.Measure Real
    h_left : MeasureTheory.IntegrableOn f (Set.Iio b) μ
    h_right : MeasureTheory.IntegrableOn f (Set.Ici b) μ
    ⊢ Eq μ (μ.restrict (Union.union (Set.Iio b) (Set.Ici b)))
  -/
  rw [Iio_union_Ici, Measure.restrict_univ]
  /-
    🎉 no goals
  -/


/-- If `μ` is a finite measure then `∫ x in a..b, c ∂μ = (μ (Iic b) - μ (Iic a)) • c`. -/
theorem integral_const_of_cdf [CompleteSpace E] [IsFiniteMeasure μ] (c : E) :
    ∫ _ in a..b, c ∂μ = ((μ (Iic b)).toReal - (μ (Iic a)).toReal) • c := by
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : E
    ⊢ Eq (intervalIntegral (fun x => c) a b μ) (HSMul.hSMul (HSub.hSub (μ (Set.Iic …
  -/
  simp only [sub_smul, ← setIntegral_const]
  /-
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    a b : Real
    μ : MeasureTheory.Measure Real
    inst✝¹ : CompleteSpace E
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    c : E
    ⊢ Eq (intervalIntegral (fun x => c) a b μ) (HSub.hSub (MeasureTheory.integral  …
  -/
  refine (integral_Iic_sub_Iic ?_ ?_).symm <;>
    /-
      case refine_1
      E : Type u_3
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      a b : Real
      μ : MeasureTheory.Measure Real
      inst✝¹ : CompleteSpace E
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      c : E
      ⊢ MeasureTheory.IntegrableOn (fun x => c) (Set.Iic a) μ
    -/
    /-
      🎉 no goals
    -/
    simp only [integrableOn_const, measure_lt_top, or_true]
    /-
      🎉 no goals
    -/


theorem integral_eq_integral_of_support_subset {a b} (h : support f ⊆ Ioc a b) :
    ∫ x in a..b, f x ∂μ = ∫ x, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a b : Real
    h : HasSubset.Subset (Function.support f) (Set.Ioc a b)
    ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (MeasureTheory.integral μ fun x = …
  -/
  rcases le_total a b with hab | hab
    /-
      case inl
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a b : Real
      h : HasSubset.Subset (Function.support f) (Set.Ioc a b)
      hab : LE.le a b
      ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (MeasureTheory.integral μ fun x = …
    -/
  · rw [integral_of_le hab, ← integral_indicator measurableSet_Ioc, indicator_eq_self.2 h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a b : Real
      h : HasSubset.Subset (Function.support f) (Set.Ioc a b)
      hab : LE.le b a
      ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (MeasureTheory.integral μ fun x = …
    -/
  · rw [Ioc_eq_empty hab.not_lt, subset_empty_iff, support_eq_empty_iff] at h
    /-
      case inr
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a b : Real
      h : Eq f 0
      hab : LE.le b a
      ⊢ Eq (intervalIntegral (fun x => f x) a b μ) (MeasureTheory.integral μ fun x = …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


theorem integral_congr_ae' (h : ∀ᵐ x ∂μ, x ∈ Ioc a b → f x = g x)
    (h' : ∀ᵐ x ∂μ, x ∈ Ioc b a → f x = g x) : ∫ x in a..b, f x ∂μ = ∫ x in a..b, g x ∂μ := by
  simp only [intervalIntegral, setIntegral_congr_ae measurableSet_Ioc h,
    setIntegral_congr_ae measurableSet_Ioc h']


theorem integral_congr_ae (h : ∀ᵐ x ∂μ, x ∈ Ι a b → f x = g x) :
    ∫ x in a..b, f x ∂μ = ∫ x in a..b, g x ∂μ :=
  integral_congr_ae' (ae_uIoc_iff.mp h).1 (ae_uIoc_iff.mp h).2


theorem integral_zero_ae (h : ∀ᵐ x ∂μ, x ∈ Ι a b → f x = 0) : ∫ x in a..b, f x ∂μ = 0 :=
  calc
    ∫ x in a..b, f x ∂μ = ∫ _ in a..b, 0 ∂μ := integral_congr_ae h
    _ = 0 := integral_zero


nonrec theorem integral_indicator {a₁ a₂ a₃ : ℝ} (h : a₂ ∈ Icc a₁ a₃) :
    ∫ x in a₁..a₃, indicator {x | x ≤ a₂} f x ∂μ = ∫ x in a₁..a₂, f x ∂μ := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a₁ a₂ a₃ : Real
    h : Membership.mem (Set.Icc a₁ a₃) a₂
    ⊢ Eq (intervalIntegral (fun x => (setOf fun x => LE.le x a₂).indicator f x) a₁ …
  -/
  have : {x | x ≤ a₂} ∩ Ioc a₁ a₃ = Ioc a₁ a₂ := Iic_inter_Ioc_of_le h.2
  rw [integral_of_le h.1, integral_of_le (h.1.trans h.2), integral_indicator,
    Measure.restrict_restrict, this]
    /-
      E : Type u_3
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : Real → E
      μ : MeasureTheory.Measure Real
      a₁ a₂ a₃ : Real
      h : Membership.mem (Set.Icc a₁ a₃) a₂
      this : Eq (Inter.inter (setOf fun x => LE.le x a₂) (Set.Ioc a₁ a₃)) (Set.Ioc a …
      ⊢ MeasurableSet (setOf fun x => LE.le x a₂)
    -/
  · exact measurableSet_Iic
    /-
      🎉 no goals
    -/
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    μ : MeasureTheory.Measure Real
    a₁ a₂ a₃ : Real
    h : Membership.mem (Set.Icc a₁ a₃) a₂
    this : Eq (Inter.inter (setOf fun x => LE.le x a₂) (Set.Ioc a₁ a₃)) (Set.Ioc a …
    ⊢ MeasurableSet (setOf fun x => LE.le x a₂)
  -/
  all_goals apply measurableSet_Iic
  /-
    🎉 no goals
  -/


theorem integral_eq_zero_iff_of_le_of_nonneg_ae (hab : a ≤ b) (hf : 0 ≤ᵐ[μ.restrict (Ioc a b)] f)
    (hfi : IntervalIntegrable f μ a b) :
    ∫ x in a..b, f x ∂μ = 0 ↔ f =ᵐ[μ.restrict (Ioc a b)] 0 := by
  /-
    f : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hf : (MeasureTheory.ae (μ.restrict (Set.Ioc a b))).EventuallyLE 0 f
    hfi : IntervalIntegrable f μ a b
    ⊢ Iff (Eq (intervalIntegral (fun x => f x) a b μ) 0) ((MeasureTheory.ae (μ.res …
  -/
  rw [integral_of_le hab, integral_eq_zero_iff_of_nonneg_ae hf hfi.1]
  /-
    🎉 no goals
  -/


theorem integral_eq_zero_iff_of_nonneg_ae (hf : 0 ≤ᵐ[μ.restrict (Ioc a b ∪ Ioc b a)] f)
    (hfi : IntervalIntegrable f μ a b) :
    ∫ x in a..b, f x ∂μ = 0 ↔ f =ᵐ[μ.restrict (Ioc a b ∪ Ioc b a)] 0 := by
  /-
    f : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hf : (MeasureTheory.ae (μ.restrict (Union.union (Set.Ioc a b) (Set.Ioc b a)))) …
    hfi : IntervalIntegrable f μ a b
    ⊢ Iff (Eq (intervalIntegral (fun x => f x) a b μ) 0) ((MeasureTheory.ae (μ.res …
  -/
  rcases le_total a b with hab | hab <;>
    /-
      case inl
      f : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hf : (MeasureTheory.ae (μ.restrict (Union.union (Set.Ioc a b) (Set.Ioc b a)))) …
      hfi : IntervalIntegrable f μ a b
      hab : LE.le a b
      ⊢ Iff (Eq (intervalIntegral (fun x => f x) a b μ) 0) ((MeasureTheory.ae (μ.res …
    -/
    simp only [Ioc_eq_empty hab.not_lt, empty_union, union_empty] at hf ⊢
    /-
      case inl
      f : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hfi : IntervalIntegrable f μ a b
      hab : LE.le a b
      hf : (MeasureTheory.ae (μ.restrict (Set.Ioc a b))).EventuallyLE 0 f
      ⊢ Iff (Eq (intervalIntegral (fun x => f x) a b μ) 0) ((MeasureTheory.ae (μ.res …
    -/
  · exact integral_eq_zero_iff_of_le_of_nonneg_ae hab hf hfi
    /-
      🎉 no goals
    -/
    /-
      case inr
      f : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hfi : IntervalIntegrable f μ a b
      hab : LE.le b a
      hf : (MeasureTheory.ae (μ.restrict (Set.Ioc b a))).EventuallyLE 0 f
      ⊢ Iff (Eq (intervalIntegral (fun x => f x) a b μ) 0) ((MeasureTheory.ae (μ.res …
    -/
  · rw [integral_symm, neg_eq_zero, integral_eq_zero_iff_of_le_of_nonneg_ae hab hf hfi.symm]
    /-
      🎉 no goals
    -/


/-- If `f` is nonnegative and integrable on the unordered interval `Set.uIoc a b`, then its
integral over `a..b` is positive if and only if `a < b` and the measure of
`Function.support f ∩ Set.Ioc a b` is positive. -/
theorem integral_pos_iff_support_of_nonneg_ae' (hf : 0 ≤ᵐ[μ.restrict (Ι a b)] f)
    (hfi : IntervalIntegrable f μ a b) :
    (0 < ∫ x in a..b, f x ∂μ) ↔ a < b ∧ 0 < μ (support f ∩ Ioc a b) := by
  /-
    f : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hf : (MeasureTheory.ae (μ.restrict (Set.uIoc a b))).EventuallyLE 0 f
    hfi : IntervalIntegrable f μ a b
    ⊢ Iff (LT.lt 0 (intervalIntegral (fun x => f x) a b μ)) (And (LT.lt a b) (LT.l …
  -/
  cases' lt_or_le a b with hab hba
    /-
      case inl
      f : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hf : (MeasureTheory.ae (μ.restrict (Set.uIoc a b))).EventuallyLE 0 f
      hfi : IntervalIntegrable f μ a b
      hab : LT.lt a b
      ⊢ Iff (LT.lt 0 (intervalIntegral (fun x => f x) a b μ)) (And (LT.lt a b) (LT.l …
    -/
  · rw [uIoc_of_le hab.le] at hf
    simp only [hab, true_and, integral_of_le hab.le,
      setIntegral_pos_iff_support_of_nonneg_ae hf hfi.1]
    /-
      case inr
      f : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hf : (MeasureTheory.ae (μ.restrict (Set.uIoc a b))).EventuallyLE 0 f
      hfi : IntervalIntegrable f μ a b
      hba : LE.le b a
      ⊢ Iff (LT.lt 0 (intervalIntegral (fun x => f x) a b μ)) (And (LT.lt a b) (LT.l …
    -/
  · suffices (∫ x in a..b, f x ∂μ) ≤ 0 by simp only [this.not_lt, hba.not_lt, false_and]
    /-
      case inr
      f : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hf : (MeasureTheory.ae (μ.restrict (Set.uIoc a b))).EventuallyLE 0 f
      hfi : IntervalIntegrable f μ a b
      hba : LE.le b a
      ⊢ LE.le (intervalIntegral (fun x => f x) a b μ) 0
    -/
    rw [integral_of_ge hba, neg_nonpos]
    /-
      case inr
      f : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hf : (MeasureTheory.ae (μ.restrict (Set.uIoc a b))).EventuallyLE 0 f
      hfi : IntervalIntegrable f μ a b
      hba : LE.le b a
      ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict (Set.Ioc b a)) fun x => f x)
    -/
    rw [uIoc_comm, uIoc_of_le hba] at hf
    /-
      case inr
      f : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hf : (MeasureTheory.ae (μ.restrict (Set.Ioc b a))).EventuallyLE 0 f
      hfi : IntervalIntegrable f μ a b
      hba : LE.le b a
      ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict (Set.Ioc b a)) fun x => f x)
    -/
    exact integral_nonneg_of_ae hf
    /-
      🎉 no goals
    -/


/-- If `f` is nonnegative a.e.-everywhere and it is integrable on the unordered interval
`Set.uIoc a b`, then its integral over `a..b` is positive if and only if `a < b` and the
measure of `Function.support f ∩ Set.Ioc a b` is positive. -/
theorem integral_pos_iff_support_of_nonneg_ae (hf : 0 ≤ᵐ[μ] f) (hfi : IntervalIntegrable f μ a b) :
    (0 < ∫ x in a..b, f x ∂μ) ↔ a < b ∧ 0 < μ (support f ∩ Ioc a b) :=
  integral_pos_iff_support_of_nonneg_ae' (ae_mono Measure.restrict_le_self hf) hfi


/-- If `f : ℝ → ℝ` is integrable on `(a, b]` for real numbers `a < b`, and positive on the interior
of the interval, then its integral over `a..b` is strictly positive. -/
theorem intervalIntegral_pos_of_pos_on {f : ℝ → ℝ} {a b : ℝ} (hfi : IntervalIntegrable f volume a b)
    (hpos : ∀ x : ℝ, x ∈ Ioo a b → 0 < f x) (hab : a < b) : 0 < ∫ x : ℝ in a..b, f x := by
  have hsupp : Ioo a b ⊆ support f ∩ Ioc a b := fun x hx =>
    ⟨mem_support.mpr (hpos x hx).ne', Ioo_subset_Ioc_self hx⟩
  have h₀ : 0 ≤ᵐ[volume.restrict (uIoc a b)] f := by
    rw [EventuallyLE, uIoc_of_le hab.le]
    refine ae_restrict_of_ae_eq_of_ae_restrict Ioo_ae_eq_Ioc ?_
    rw [ae_restrict_iff' measurableSet_Ioo]
    filter_upwards with x hx using (hpos x hx).le
  /-
    f : Real → Real
    a b : Real
    hfi : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    hpos : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → LT.lt 0 (f x)
    hab : LT.lt a b
    hsupp : HasSubset.Subset (Set.Ioo a b) (Inter.inter (Function.support f) (Set. …
    h₀ : (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.uIoc a …
    ⊢ LT.lt 0 (intervalIntegral (fun x => f x) a b MeasureTheory.MeasureSpace.volu …
  -/
  rw [integral_pos_iff_support_of_nonneg_ae' h₀ hfi]
  /-
    f : Real → Real
    a b : Real
    hfi : IntervalIntegrable f MeasureTheory.MeasureSpace.volume a b
    hpos : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → LT.lt 0 (f x)
    hab : LT.lt a b
    hsupp : HasSubset.Subset (Set.Ioo a b) (Inter.inter (Function.support f) (Set. …
    h₀ : (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.uIoc a …
    ⊢ And (LT.lt a b) (LT.lt 0 (MeasureTheory.MeasureSpace.volume (Inter.inter (Fu …
  -/
  exact ⟨hab, ((Measure.measure_Ioo_pos _).mpr hab).trans_le (measure_mono hsupp)⟩
  /-
    🎉 no goals
  -/


/-- If `f : ℝ → ℝ` is strictly positive everywhere, and integrable on `(a, b]` for real numbers
`a < b`, then its integral over `a..b` is strictly positive. (See `intervalIntegral_pos_of_pos_on`
for a version only assuming positivity of `f` on `(a, b)` rather than everywhere.) -/
theorem intervalIntegral_pos_of_pos {f : ℝ → ℝ} {a b : ℝ}
    (hfi : IntervalIntegrable f MeasureSpace.volume a b) (hpos : ∀ x, 0 < f x) (hab : a < b) :
    0 < ∫ x in a..b, f x :=
  intervalIntegral_pos_of_pos_on hfi (fun x _ => hpos x) hab


/-- If `f` and `g` are two functions that are interval integrable on `a..b`, `a ≤ b`,
`f x ≤ g x` for a.e. `x ∈ Set.Ioc a b`, and `f x < g x` on a subset of `Set.Ioc a b`
of nonzero measure, then `∫ x in a..b, f x ∂μ < ∫ x in a..b, g x ∂μ`. -/
theorem integral_lt_integral_of_ae_le_of_measure_setOf_lt_ne_zero (hab : a ≤ b)
    (hfi : IntervalIntegrable f μ a b) (hgi : IntervalIntegrable g μ a b)
    (hle : f ≤ᵐ[μ.restrict (Ioc a b)] g) (hlt : μ.restrict (Ioc a b) {x | f x < g x} ≠ 0) :
    (∫ x in a..b, f x ∂μ) < ∫ x in a..b, g x ∂μ := by
  rw [← sub_pos, ← integral_sub hgi hfi, integral_of_le hab,
    MeasureTheory.integral_pos_iff_support_of_nonneg_ae]
    /-
      f g : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hab : LE.le a b
      hfi : IntervalIntegrable f μ a b
      hgi : IntervalIntegrable g μ a b
      hle : (MeasureTheory.ae (μ.restrict (Set.Ioc a b))).EventuallyLE f g
      hlt : Ne ((μ.restrict (Set.Ioc a b)) (setOf fun x => LT.lt (f x) (g x))) 0
      ⊢ LT.lt 0 ((μ.restrict (Set.Ioc a b)) (Function.support fun x => HSub.hSub (g  …
    -/
  · refine pos_iff_ne_zero.2 (mt (measure_mono_null ?_) hlt)
    /-
      f g : Real → Real
      a b : Real
      μ : MeasureTheory.Measure Real
      hab : LE.le a b
      hfi : IntervalIntegrable f μ a b
      hgi : IntervalIntegrable g μ a b
      hle : (MeasureTheory.ae (μ.restrict (Set.Ioc a b))).EventuallyLE f g
      hlt : Ne ((μ.restrict (Set.Ioc a b)) (setOf fun x => LT.lt (f x) (g x))) 0
      ⊢ HasSubset.Subset (setOf fun x => LT.lt (f x) (g x)) (Function.support fun x  …
    -/
    exact fun x hx => (sub_pos.2 hx.out).ne'
    /-
      🎉 no goals
    -/
  /-
    case hf
    f g : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hfi : IntervalIntegrable f μ a b
    hgi : IntervalIntegrable g μ a b
    hle : (MeasureTheory.ae (μ.restrict (Set.Ioc a b))).EventuallyLE f g
    hlt : Ne ((μ.restrict (Set.Ioc a b)) (setOf fun x => LT.lt (f x) (g x))) 0
    ⊢ (MeasureTheory.ae (μ.restrict (Set.Ioc a b))).EventuallyLE 0 fun x => HSub.h …
  -/
  exacts [hle.mono fun x => sub_nonneg.2, hgi.1.sub hfi.1]
  /-
    🎉 no goals
  -/


/-- If `f` and `g` are continuous on `[a, b]`, `a < b`, `f x ≤ g x` on this interval, and
`f c < g c` at some point `c ∈ [a, b]`, then `∫ x in a..b, f x < ∫ x in a..b, g x`. -/
theorem integral_lt_integral_of_continuousOn_of_le_of_exists_lt {f g : ℝ → ℝ} {a b : ℝ}
    (hab : a < b) (hfc : ContinuousOn f (Icc a b)) (hgc : ContinuousOn g (Icc a b))
    (hle : ∀ x ∈ Ioc a b, f x ≤ g x) (hlt : ∃ c ∈ Icc a b, f c < g c) :
    (∫ x in a..b, f x) < ∫ x in a..b, g x := by
  apply integral_lt_integral_of_ae_le_of_measure_setOf_lt_ne_zero hab.le
    (hfc.intervalIntegrable_of_Icc hab.le) (hgc.intervalIntegrable_of_Icc hab.le)
  · simpa only [measurableSet_Ioc, ae_restrict_eq]
      using (ae_restrict_mem measurableSet_Ioc).mono hle
  /-
    case hlt
    f g : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hgc : ContinuousOn g (Set.Icc a b)
    hle : ∀ (x : Real), Membership.mem (Set.Ioc a b) x → LE.le (f x) (g x)
    hlt : Exists fun c => And (Membership.mem (Set.Icc a b) c) (LT.lt (f c) (g c))
    ⊢ Ne ((MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc a b)) (setOf fun x  …
  -/
  contrapose! hlt
  have h_eq : f =ᵐ[volume.restrict (Ioc a b)] g := by
    simp only [← not_le, ← ae_iff] at hlt
    exact EventuallyLE.antisymm ((ae_restrict_iff' measurableSet_Ioc).2 <|
      Eventually.of_forall hle) hlt
  /-
    case hlt
    f g : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hgc : ContinuousOn g (Set.Icc a b)
    hle : ∀ (x : Real), Membership.mem (Set.Ioc a b) x → LE.le (f x) (g x)
    hlt : Eq ((MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc a b)) (setOf fu …
    h_eq : (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc  …
    ⊢ ∀ (c : Real), Membership.mem (Set.Icc a b) c → LE.le (g c) (f c)
  -/
  rw [Measure.restrict_congr_set Ioc_ae_eq_Icc] at h_eq
  /-
    case hlt
    f g : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hgc : ContinuousOn g (Set.Icc a b)
    hle : ∀ (x : Real), Membership.mem (Set.Ioc a b) x → LE.le (f x) (g x)
    hlt : Eq ((MeasureTheory.MeasureSpace.volume.restrict (Set.Ioc a b)) (setOf fu …
    h_eq : (MeasureTheory.ae (MeasureTheory.MeasureSpace.volume.restrict (Set.Icc  …
    ⊢ ∀ (c : Real), Membership.mem (Set.Icc a b) c → LE.le (g c) (f c)
  -/
  exact fun c hc ↦ (Measure.eqOn_Icc_of_ae_eq volume hab.ne h_eq hfc hgc hc).ge
  /-
    🎉 no goals
  -/


theorem integral_nonneg_of_ae_restrict (hab : a ≤ b) (hf : 0 ≤ᵐ[μ.restrict (Icc a b)] f) :
    0 ≤ ∫ u in a..b, f u ∂μ := by
  /-
    f : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hf : (MeasureTheory.ae (μ.restrict (Set.Icc a b))).EventuallyLE 0 f
    ⊢ LE.le 0 (intervalIntegral (fun u => f u) a b μ)
  -/
  let H := ae_restrict_of_ae_restrict_of_subset Ioc_subset_Icc_self hf
  /-
    f : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hf : (MeasureTheory.ae (μ.restrict (Set.Icc a b))).EventuallyLE 0 f
    H : Filter.Eventually (fun x => LE.le (0 x) (f x)) (MeasureTheory.ae (μ.restri …
    ⊢ LE.le 0 (intervalIntegral (fun u => f u) a b μ)
  -/
  simpa only [integral_of_le hab] using setIntegral_nonneg_of_ae_restrict H
  /-
    🎉 no goals
  -/


theorem integral_nonneg_of_ae (hab : a ≤ b) (hf : 0 ≤ᵐ[μ] f) : 0 ≤ ∫ u in a..b, f u ∂μ :=
  integral_nonneg_of_ae_restrict hab <| ae_restrict_of_ae hf


theorem integral_nonneg_of_forall (hab : a ≤ b) (hf : ∀ u, 0 ≤ f u) : 0 ≤ ∫ u in a..b, f u ∂μ :=
  integral_nonneg_of_ae hab <| Eventually.of_forall hf


theorem integral_nonneg (hab : a ≤ b) (hf : ∀ u, u ∈ Icc a b → 0 ≤ f u) : 0 ≤ ∫ u in a..b, f u ∂μ :=
  integral_nonneg_of_ae_restrict hab <| (ae_restrict_iff' measurableSet_Icc).mpr <| ae_of_all μ hf


theorem abs_integral_le_integral_abs (hab : a ≤ b) :
    |∫ x in a..b, f x ∂μ| ≤ ∫ x in a..b, |f x| ∂μ := by
  /-
    f : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    ⊢ LE.le (abs (intervalIntegral (fun x => f x) a b μ)) (intervalIntegral (fun x …
  -/
  simpa only [← Real.norm_eq_abs] using norm_integral_le_integral_norm hab
  /-
    🎉 no goals
  -/


lemma integral_pos (hab : a < b)
    (hfc : ContinuousOn f (Icc a b)) (hle : ∀ x ∈ Ioc a b, 0 ≤ f x) (hlt : ∃ c ∈ Icc a b, 0 < f c) :
    0 < ∫ x in a..b, f x :=
  (integral_lt_integral_of_continuousOn_of_le_of_exists_lt hab
                                                  /-
                                                    f : Real → Real
                                                    a b : Real
                                                    hab : LT.lt a b
                                                    hfc : ContinuousOn f (Set.Icc a b)
                                                    hle : ∀ (x : Real), Membership.mem (Set.Ioc a b) x → LE.le 0 (f x)
                                                    hlt : Exists fun c => And (Membership.mem (Set.Icc a b) c) (LT.lt 0 (f c))
                                                    ⊢ Eq 0 (intervalIntegral (fun x => 0) a b MeasureTheory.MeasureSpace.volume)
                                                  -/
    continuousOn_const hfc hle hlt).trans_eq' (by simp)
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem integral_mono_interval {c d} (hca : c ≤ a) (hab : a ≤ b) (hbd : b ≤ d)
    (hf : 0 ≤ᵐ[μ.restrict (Ioc c d)] f) (hfi : IntervalIntegrable f μ c d) :
    (∫ x in a..b, f x ∂μ) ≤ ∫ x in c..d, f x ∂μ := by
  /-
    f : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    c d : Real
    hca : LE.le c a
    hab : LE.le a b
    hbd : LE.le b d
    hf : (MeasureTheory.ae (μ.restrict (Set.Ioc c d))).EventuallyLE 0 f
    hfi : IntervalIntegrable f μ c d
    ⊢ LE.le (intervalIntegral (fun x => f x) a b μ) (intervalIntegral (fun x => f  …
  -/
  rw [integral_of_le hab, integral_of_le (hca.trans (hab.trans hbd))]
  /-
    f : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    c d : Real
    hca : LE.le c a
    hab : LE.le a b
    hbd : LE.le b d
    hf : (MeasureTheory.ae (μ.restrict (Set.Ioc c d))).EventuallyLE 0 f
    hfi : IntervalIntegrable f μ c d
    ⊢ LE.le (MeasureTheory.integral (μ.restrict (Set.Ioc a b)) fun x => f x) (Meas …
  -/
  exact setIntegral_mono_set hfi.1 hf (Ioc_subset_Ioc hca hbd).eventuallyLE
  /-
    🎉 no goals
  -/


theorem abs_integral_mono_interval {c d} (h : Ι a b ⊆ Ι c d) (hf : 0 ≤ᵐ[μ.restrict (Ι c d)] f)
    (hfi : IntervalIntegrable f μ c d) : |∫ x in a..b, f x ∂μ| ≤ |∫ x in c..d, f x ∂μ| :=
  have hf' : 0 ≤ᵐ[μ.restrict (Ι a b)] f := ae_mono (Measure.restrict_mono h le_rfl) hf
  calc
    |∫ x in a..b, f x ∂μ| = |∫ x in Ι a b, f x ∂μ| := abs_integral_eq_abs_integral_uIoc f
    _ = ∫ x in Ι a b, f x ∂μ := abs_of_nonneg (MeasureTheory.integral_nonneg_of_ae hf')
    _ ≤ ∫ x in Ι c d, f x ∂μ := setIntegral_mono_set hfi.def' hf h.eventuallyLE
    _ ≤ |∫ x in Ι c d, f x ∂μ| := le_abs_self _
    _ = |∫ x in c..d, f x ∂μ| := (abs_integral_eq_abs_integral_uIoc f).symm


theorem integral_mono_ae_restrict (h : f ≤ᵐ[μ.restrict (Icc a b)] g) :
    (∫ u in a..b, f u ∂μ) ≤ ∫ u in a..b, g u ∂μ := by
  /-
    f g : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hf : IntervalIntegrable f μ a b
    hg : IntervalIntegrable g μ a b
    h : (MeasureTheory.ae (μ.restrict (Set.Icc a b))).EventuallyLE f g
    ⊢ LE.le (intervalIntegral (fun u => f u) a b μ) (intervalIntegral (fun u => g  …
  -/
  let H := h.filter_mono <| ae_mono <| Measure.restrict_mono Ioc_subset_Icc_self <| le_refl μ
  /-
    f g : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hf : IntervalIntegrable f μ a b
    hg : IntervalIntegrable g μ a b
    h : (MeasureTheory.ae (μ.restrict (Set.Icc a b))).EventuallyLE f g
    H : Filter.Eventually (fun x => LE.le (f x) (g x)) (MeasureTheory.ae (μ.restri …
    ⊢ LE.le (intervalIntegral (fun u => f u) a b μ) (intervalIntegral (fun u => g  …
  -/
  simpa only [integral_of_le hab] using setIntegral_mono_ae_restrict hf.1 hg.1 H
  /-
    🎉 no goals
  -/


theorem integral_mono_ae (h : f ≤ᵐ[μ] g) : (∫ u in a..b, f u ∂μ) ≤ ∫ u in a..b, g u ∂μ := by
  /-
    f g : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hf : IntervalIntegrable f μ a b
    hg : IntervalIntegrable g μ a b
    h : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ LE.le (intervalIntegral (fun u => f u) a b μ) (intervalIntegral (fun u => g  …
  -/
  simpa only [integral_of_le hab] using setIntegral_mono_ae hf.1 hg.1 h
  /-
    🎉 no goals
  -/


theorem integral_mono_on (h : ∀ x ∈ Icc a b, f x ≤ g x) :
    (∫ u in a..b, f u ∂μ) ≤ ∫ u in a..b, g u ∂μ := by
  /-
    f g : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hf : IntervalIntegrable f μ a b
    hg : IntervalIntegrable g μ a b
    h : ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (f x) (g x)
    ⊢ LE.le (intervalIntegral (fun u => f u) a b μ) (intervalIntegral (fun u => g  …
  -/
  let H x hx := h x <| Ioc_subset_Icc_self hx
  /-
    f g : Real → Real
    a b : Real
    μ : MeasureTheory.Measure Real
    hab : LE.le a b
    hf : IntervalIntegrable f μ a b
    hg : IntervalIntegrable g μ a b
    h : ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (f x) (g x)
    H : ∀ (x : Real), Membership.mem (Set.Ioc a b) x → LE.le (f x) (g x) := fun x  …
    ⊢ LE.le (intervalIntegral (fun u => f u) a b μ) (intervalIntegral (fun u => g  …
  -/
  simpa only [integral_of_le hab] using setIntegral_mono_on hf.1 hg.1 measurableSet_Ioc H
  /-
    🎉 no goals
  -/


theorem integral_mono (h : f ≤ g) : (∫ u in a..b, f u ∂μ) ≤ ∫ u in a..b, g u ∂μ :=
  integral_mono_ae hab hf hg <| ae_of_all _ h


theorem _root_.MeasureTheory.Integrable.hasSum_intervalIntegral (hfi : Integrable f μ) (y : ℝ) :
    HasSum (fun n : ℤ => ∫ x in y + n..y + n + 1, f x ∂μ) (∫ x, f x ∂μ) := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    hfi : MeasureTheory.Integrable f μ
    y : Real
    ⊢ HasSum (fun n => intervalIntegral (fun x => f x) (HAdd.hAdd y ↑n) (HAdd.hAdd …
  -/
  simp_rw [integral_of_le (le_add_of_nonneg_right zero_le_one)]
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure Real
    f : Real → E
    hfi : MeasureTheory.Integrable f μ
    y : Real
    ⊢ HasSum (fun n => MeasureTheory.integral (μ.restrict (Set.Ioc (HAdd.hAdd y ↑n …
  -/
  rw [← setIntegral_univ, ← iUnion_Ioc_add_intCast y]
  exact
    hasSum_integral_iUnion (fun i => measurableSet_Ioc) (pairwise_disjoint_Ioc_add_intCast y)
      hfi.integrableOn


                                                                                    /-
                                                                                      ι : Type u_1
                                                                                      𝕜 : Type u_2
                                                                                      E : Type u_3
                                                                                      F : Type u_4
                                                                                      A : Type u_5
                                                                                      inst✝¹ : NormedAddCommGroup E
                                                                                      inst✝ : NormedSpace Real E
                                                                                      μ : MeasureTheory.Measure Real
                                                                                      f : Real → E
                                                                                      ⊢ MeasureTheory.Measure Real
                                                                                    -/
theorem _root_.MeasureTheory.Integrable.hasSum_intervalIntegral_comp_add_int (hfi : Integrable f) :
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    HasSum (fun n : ℤ => ∫ x in (0 : ℝ)..(1 : ℝ), f (x + n)) (∫ x, f x) := by
  simpa only [integral_comp_add_right, zero_add, add_comm (1 : ℝ)] using
    hfi.hasSum_intervalIntegral 0


