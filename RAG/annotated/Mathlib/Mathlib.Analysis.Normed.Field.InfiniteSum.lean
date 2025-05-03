theorem Summable.mul_of_nonneg {f : ι → ℝ} {g : ι' → ℝ} (hf : Summable f) (hg : Summable g)
    (hf' : 0 ≤ f) (hg' : 0 ≤ g) : Summable fun x : ι × ι' => f x.1 * g x.2 :=
  (summable_prod_of_nonneg fun _ ↦ mul_nonneg (hf' _) (hg' _)).2 ⟨fun x ↦ hg.mul_left (f x),
       /-
         ι : Type u_2
         ι' : Type u_3
         f : ι → Real
         g : ι' → Real
         hf : Summable f
         hg : Summable g
         hf' : LE.le 0 f
         hg' : LE.le 0 g
         ⊢ Summable fun x => tsum fun y => HMul.hMul (f { fst := x, snd := y }.1) (g {  …
       -/
    by simpa only [hg.tsum_mul_left _] using hf.mul_right (∑' x, g x)⟩
       /-
         🎉 no goals
       -/


theorem Summable.mul_norm {f : ι → R} {g : ι' → R} (hf : Summable fun x => ‖f x‖)
    (hg : Summable fun x => ‖g x‖) : Summable fun x : ι × ι' => ‖f x.1 * g x.2‖ :=
  .of_nonneg_of_le (fun _ ↦ norm_nonneg _)
    (fun x => norm_mul_le (f x.1) (g x.2))
    (hf.mul_of_nonneg hg (fun x => norm_nonneg <| f x) fun x => norm_nonneg <| g x : _)


theorem summable_mul_of_summable_norm [CompleteSpace R] {f : ι → R} {g : ι' → R}
    (hf : Summable fun x => ‖f x‖) (hg : Summable fun x => ‖g x‖) :
    Summable fun x : ι × ι' => f x.1 * g x.2 :=
  (hf.mul_norm hg).of_norm


theorem summable_mul_of_summable_norm' {f : ι → R} {g : ι' → R}
    (hf : Summable fun x => ‖f x‖) (h'f : Summable f)
    (hg : Summable fun x => ‖g x‖) (h'g : Summable g) :
    Summable fun x : ι × ι' => f x.1 * g x.2 := by
  classical
  suffices HasSum (fun x : ι × ι' => f x.1 * g x.2) ((∑' i, f i) * (∑' j, g j)) from this.summable
  let s : Finset ι × Finset ι' → Finset (ι × ι') := fun p ↦ p.1 ×ˢ p.2
  apply hasSum_of_subseq_of_summable (hf.mul_norm hg) tendsto_finset_prod_atTop
  rw [← prod_atTop_atTop_eq]
  have := Tendsto.prod_map h'f.hasSum h'g.hasSum
  rw [← nhds_prod_eq] at this
  convert ((continuous_mul (M := R)).continuousAt
      (x := (∑' (i : ι), f i, ∑' (j : ι'), g j))).tendsto.comp this with p
  simp [s, sum_product, ← mul_sum, ← sum_mul]


/-- Product of two infinite sums indexed by arbitrary types.
    See also `tsum_mul_tsum` if `f` and `g` are *not* absolutely summable, and
    `tsum_mul_tsum_of_summable_norm'` when the space is not complete. -/
theorem tsum_mul_tsum_of_summable_norm [CompleteSpace R] {f : ι → R} {g : ι' → R}
    (hf : Summable fun x => ‖f x‖) (hg : Summable fun x => ‖g x‖) :
    ((∑' x, f x) * ∑' y, g y) = ∑' z : ι × ι', f z.1 * g z.2 :=
  tsum_mul_tsum hf.of_norm hg.of_norm (summable_mul_of_summable_norm hf hg)


theorem tsum_mul_tsum_of_summable_norm' {f : ι → R} {g : ι' → R}
    (hf : Summable fun x => ‖f x‖) (h'f : Summable f)
    (hg : Summable fun x => ‖g x‖) (h'g : Summable g) :
    ((∑' x, f x) * ∑' y, g y) = ∑' z : ι × ι', f z.1 * g z.2 :=
  tsum_mul_tsum h'f h'g (summable_mul_of_summable_norm' hf h'f hg h'g)


theorem summable_norm_sum_mul_antidiagonal_of_summable_norm {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (hg : Summable fun x => ‖g x‖) :
    Summable fun n => ‖∑ kl ∈ antidiagonal n, f kl.1 * g kl.2‖ := by
  have :=
    summable_sum_mul_antidiagonal_of_summable_mul
      (Summable.mul_of_nonneg hf hg (fun _ => norm_nonneg _) fun _ => norm_nonneg _)
  /-
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    hg : Summable fun x => Norm.norm (g x)
    this : Summable fun n => (Finset.HasAntidiagonal.antidiagonal n).sum fun kl => …
    ⊢ Summable fun n => Norm.norm ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
  -/
  refine this.of_nonneg_of_le (fun _ => norm_nonneg _) (fun n ↦ ?_)
  calc
    ‖∑ kl ∈ antidiagonal n, f kl.1 * g kl.2‖ ≤ ∑ kl ∈ antidiagonal n, ‖f kl.1 * g kl.2‖ :=
      norm_sum_le _ _
    _ ≤ ∑ kl ∈ antidiagonal n, ‖f kl.1‖ * ‖g kl.2‖ := by gcongr; apply norm_mul_le


theorem summable_sum_mul_antidiagonal_of_summable_norm' {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (h'f : Summable f)
    (hg : Summable fun x => ‖g x‖) (h'g : Summable g) :
    Summable fun n => ∑ kl ∈ antidiagonal n, f kl.1 * g kl.2 :=
  summable_sum_mul_antidiagonal_of_summable_mul (summable_mul_of_summable_norm' hf h'f hg h'g)


/-- The Cauchy product formula for the product of two infinite sums indexed by `ℕ`,
    expressed by summing on `Finset.antidiagonal`.
    See also `tsum_mul_tsum_eq_tsum_sum_antidiagonal` if `f` and `g` are
    *not* absolutely summable, and `tsum_mul_tsum_eq_tsum_sum_antidiagonal_of_summable_norm'`
    when the space is not complete. -/
theorem tsum_mul_tsum_eq_tsum_sum_antidiagonal_of_summable_norm [CompleteSpace R] {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (hg : Summable fun x => ‖g x‖) :
    ((∑' n, f n) * ∑' n, g n) = ∑' n, ∑ kl ∈ antidiagonal n, f kl.1 * g kl.2 :=
  tsum_mul_tsum_eq_tsum_sum_antidiagonal hf.of_norm hg.of_norm (summable_mul_of_summable_norm hf hg)


theorem tsum_mul_tsum_eq_tsum_sum_antidiagonal_of_summable_norm' {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (h'f : Summable f)
    (hg : Summable fun x => ‖g x‖) (h'g : Summable g) :
    ((∑' n, f n) * ∑' n, g n) = ∑' n, ∑ kl ∈ antidiagonal n, f kl.1 * g kl.2 :=
  tsum_mul_tsum_eq_tsum_sum_antidiagonal h'f h'g (summable_mul_of_summable_norm' hf h'f hg h'g)


theorem summable_norm_sum_mul_range_of_summable_norm {f g : ℕ → R} (hf : Summable fun x => ‖f x‖)
    (hg : Summable fun x => ‖g x‖) : Summable fun n => ‖∑ k ∈ range (n + 1), f k * g (n - k)‖ := by
  /-
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    hg : Summable fun x => Norm.norm (g x)
    ⊢ Summable fun n => Norm.norm ((Finset.range (HAdd.hAdd n 1)).sum fun k => HMu …
  -/
  simp_rw [← sum_antidiagonal_eq_sum_range_succ fun k l => f k * g l]
  /-
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    hg : Summable fun x => Norm.norm (g x)
    ⊢ Summable fun n => Norm.norm ((Finset.HasAntidiagonal.antidiagonal n).sum fun …
  -/
  exact summable_norm_sum_mul_antidiagonal_of_summable_norm hf hg
  /-
    🎉 no goals
  -/


theorem summable_sum_mul_range_of_summable_norm' {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (h'f : Summable f)
    (hg : Summable fun x => ‖g x‖) (h'g : Summable g) :
    Summable fun n => ∑ k ∈ range (n + 1), f k * g (n - k) := by
  /-
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    h'f : Summable f
    hg : Summable fun x => Norm.norm (g x)
    h'g : Summable g
    ⊢ Summable fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => HMul.hMul (f k …
  -/
  simp_rw [← sum_antidiagonal_eq_sum_range_succ fun k l => f k * g l]
  /-
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    h'f : Summable f
    hg : Summable fun x => Norm.norm (g x)
    h'g : Summable g
    ⊢ Summable fun n => (Finset.HasAntidiagonal.antidiagonal n).sum fun ij => HMul …
  -/
  exact summable_sum_mul_antidiagonal_of_summable_norm' hf h'f hg h'g
  /-
    🎉 no goals
  -/


/-- The Cauchy product formula for the product of two infinite sums indexed by `ℕ`,
    expressed by summing on `Finset.range`.
    See also `tsum_mul_tsum_eq_tsum_sum_range` if `f` and `g` are
    *not* absolutely summable, and `tsum_mul_tsum_eq_tsum_sum_range_of_summable_norm'` when the
    space is not complete. -/
theorem tsum_mul_tsum_eq_tsum_sum_range_of_summable_norm [CompleteSpace R] {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (hg : Summable fun x => ‖g x‖) :
    ((∑' n, f n) * ∑' n, g n) = ∑' n, ∑ k ∈ range (n + 1), f k * g (n - k) := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    hg : Summable fun x => Norm.norm (g x)
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun n => (Finse …
  -/
  simp_rw [← sum_antidiagonal_eq_sum_range_succ fun k l => f k * g l]
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    hg : Summable fun x => Norm.norm (g x)
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun n => (Finse …
  -/
  exact tsum_mul_tsum_eq_tsum_sum_antidiagonal_of_summable_norm hf hg
  /-
    🎉 no goals
  -/


theorem hasSum_sum_range_mul_of_summable_norm [CompleteSpace R] {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (hg : Summable fun x => ‖g x‖) :
    HasSum (fun n ↦ ∑ k ∈ range (n + 1), f k * g (n - k)) ((∑' n, f n) * ∑' n, g n) := by
  /-
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    hg : Summable fun x => Norm.norm (g x)
    ⊢ HasSum (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => HMul.hMul (f k) …
  -/
  convert (summable_norm_sum_mul_range_of_summable_norm hf hg).of_norm.hasSum
  /-
    case h.e'_6
    R : Type u_1
    inst✝¹ : NormedRing R
    inst✝ : CompleteSpace R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    hg : Summable fun x => Norm.norm (g x)
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun b => (Finse …
  -/
  exact tsum_mul_tsum_eq_tsum_sum_range_of_summable_norm hf hg
  /-
    🎉 no goals
  -/


theorem tsum_mul_tsum_eq_tsum_sum_range_of_summable_norm' {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (h'f : Summable f)
    (hg : Summable fun x => ‖g x‖) (h'g : Summable g) :
    ((∑' n, f n) * ∑' n, g n) = ∑' n, ∑ k ∈ range (n + 1), f k * g (n - k) := by
  /-
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    h'f : Summable f
    hg : Summable fun x => Norm.norm (g x)
    h'g : Summable g
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun n => (Finse …
  -/
  simp_rw [← sum_antidiagonal_eq_sum_range_succ fun k l => f k * g l]
  /-
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    h'f : Summable f
    hg : Summable fun x => Norm.norm (g x)
    h'g : Summable g
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun n => (Finse …
  -/
  exact tsum_mul_tsum_eq_tsum_sum_antidiagonal_of_summable_norm' hf h'f hg h'g
  /-
    🎉 no goals
  -/


theorem hasSum_sum_range_mul_of_summable_norm' {f g : ℕ → R}
    (hf : Summable fun x => ‖f x‖) (h'f : Summable f)
    (hg : Summable fun x => ‖g x‖) (h'g : Summable g) :
    HasSum (fun n ↦ ∑ k ∈ range (n + 1), f k * g (n - k)) ((∑' n, f n) * ∑' n, g n) := by
  /-
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    h'f : Summable f
    hg : Summable fun x => Norm.norm (g x)
    h'g : Summable g
    ⊢ HasSum (fun n => (Finset.range (HAdd.hAdd n 1)).sum fun k => HMul.hMul (f k) …
  -/
  convert (summable_sum_mul_range_of_summable_norm' hf h'f hg h'g).hasSum
  /-
    case h.e'_6
    R : Type u_1
    inst✝ : NormedRing R
    f g : Nat → R
    hf : Summable fun x => Norm.norm (f x)
    h'f : Summable f
    hg : Summable fun x => Norm.norm (g x)
    h'g : Summable g
    ⊢ Eq (HMul.hMul (tsum fun n => f n) (tsum fun n => g n)) (tsum fun b => (Finse …
  -/
  exact tsum_mul_tsum_eq_tsum_sum_range_of_summable_norm' hf h'f hg h'g
  /-
    🎉 no goals
  -/


lemma summable_of_absolute_convergence_real {f : ℕ → ℝ} :
    (∃ r, Tendsto (fun n ↦ ∑ i ∈ range n, |f i|) atTop (𝓝 r)) → Summable f
  | ⟨r, hr⟩ => by
    /-
      f : Nat → Real
      r : Real
      hr : Filter.Tendsto (fun n => (Finset.range n).sum fun i => abs (f i)) Filter. …
      ⊢ Summable f
    -/
    refine .of_norm ⟨r, (hasSum_iff_tendsto_nat_of_nonneg ?_ _).2 ?_⟩
      /-
        case refine_1
        f : Nat → Real
        r : Real
        hr : Filter.Tendsto (fun n => (Finset.range n).sum fun i => abs (f i)) Filter. …
        ⊢ ∀ (i : Nat), LE.le 0 (Norm.norm (f i))
      -/
    · exact fun i ↦ norm_nonneg _
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        f : Nat → Real
        r : Real
        hr : Filter.Tendsto (fun n => (Finset.range n).sum fun i => abs (f i)) Filter. …
        ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => Norm.norm (f i)) Filt …
      -/
    · simpa only using hr
      /-
        🎉 no goals
      -/

