theorem setOf_liouvilleWith_subset_aux :
    { x : ℝ | ∃ p > 2, LiouvilleWith p x } ⊆
      ⋃ m : ℤ, (· + (m : ℝ)) ⁻¹' ⋃ n > (0 : ℕ),
        { x : ℝ | ∃ᶠ b : ℕ in atTop, ∃ a ∈ Finset.Icc (0 : ℤ) b,
          |x - (a : ℤ) / b| < 1 / (b : ℝ) ^ (2 + 1 / n : ℝ) } := by
  /-
    ⊢ HasSubset.Subset (setOf fun x => Exists fun p => And (GT.gt p 2) (LiouvilleW …
  -/
  rintro x ⟨p, hp, hxp⟩
  /-
    case intro.intro
    x p : Real
    hp : GT.gt p 2
    hxp : LiouvilleWith p x
    ⊢ Membership.mem (Set.iUnion fun m => Set.preimage (fun x => HAdd.hAdd x ↑m) ( …
  -/
  rcases exists_nat_one_div_lt (sub_pos.2 hp) with ⟨n, hn⟩
  /-
    case intro.intro.intro
    x p : Real
    hp : GT.gt p 2
    hxp : LiouvilleWith p x
    n : Nat
    hn : LT.lt (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)) (HSub.hSub p 2)
    ⊢ Membership.mem (Set.iUnion fun m => Set.preimage (fun x => HAdd.hAdd x ↑m) ( …
  -/
  rw [lt_sub_iff_add_lt'] at hn
  suffices ∀ y : ℝ, LiouvilleWith p y → y ∈ Ico (0 : ℝ) 1 → ∃ᶠ b : ℕ in atTop,
      ∃ a ∈ Finset.Icc (0 : ℤ) b, |y - a / b| < 1 / (b : ℝ) ^ (2 + 1 / (n + 1 : ℕ) : ℝ) by
    simp only [mem_iUnion, mem_preimage]
    have hx : x + ↑(-⌊x⌋) ∈ Ico (0 : ℝ) 1 := by
      simp only [Int.floor_le, Int.lt_floor_add_one, add_neg_lt_iff_le_add', zero_add, and_self_iff,
        mem_Ico, Int.cast_neg, le_add_neg_iff_add_le]
    exact ⟨-⌊x⌋, n + 1, n.succ_pos, this _ (hxp.add_int _) hx⟩
  /-
    case intro.intro.intro
    x p : Real
    hp : GT.gt p 2
    hxp : LiouvilleWith p x
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    ⊢ ∀ (y : Real), LiouvilleWith p y → Membership.mem (Set.Ico 0 1) y → Filter.Fr …
  -/
  clear hxp x; intro x hxp hx01
  /-
    case intro.intro.intro
    p : Real
    hp : GT.gt p 2
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    x : Real
    hxp : LiouvilleWith p x
    hx01 : Membership.mem (Set.Ico 0 1) x
    ⊢ Filter.Frequently (fun b => Exists fun a => And (Membership.mem (Finset.Icc  …
  -/
  refine ((hxp.frequently_lt_rpow_neg hn).and_eventually (eventually_ge_atTop 1)).mono ?_
  /-
    case intro.intro.intro
    p : Real
    hp : GT.gt p 2
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    x : Real
    hxp : LiouvilleWith p x
    hx01 : Membership.mem (Set.Ico 0 1) x
    ⊢ ∀ (x_1 : Nat), And (Exists fun m => And (Ne x (HDiv.hDiv ↑m ↑x_1)) (LT.lt (a …
  -/
  rintro b ⟨⟨a, -, hlt⟩, hb⟩
  /-
    case intro.intro.intro.intro.intro.intro
    p : Real
    hp : GT.gt p 2
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    x : Real
    hxp : LiouvilleWith p x
    hx01 : Membership.mem (Set.Ico 0 1) x
    b : Nat
    hb : LE.le 1 b
    a : Int
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HPow.hPow (↑b) (Neg.neg (HA …
    ⊢ Exists fun a => And (Membership.mem (Finset.Icc 0 ↑b) a) (LT.lt (abs (HSub.h …
  -/
  rw [rpow_neg b.cast_nonneg, ← one_div, ← Nat.cast_succ] at hlt
  /-
    case intro.intro.intro.intro.intro.intro
    p : Real
    hp : GT.gt p 2
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    x : Real
    hxp : LiouvilleWith p x
    hx01 : Membership.mem (Set.Ico 0 1) x
    b : Nat
    hb : LE.le 1 b
    a : Int
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
    ⊢ Exists fun a => And (Membership.mem (Finset.Icc 0 ↑b) a) (LT.lt (abs (HSub.h …
  -/
  refine ⟨a, ?_, hlt⟩
  /-
    case intro.intro.intro.intro.intro.intro
    p : Real
    hp : GT.gt p 2
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    x : Real
    hxp : LiouvilleWith p x
    hx01 : Membership.mem (Set.Ico 0 1) x
    b : Nat
    hb : LE.le 1 b
    a : Int
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
    ⊢ Membership.mem (Finset.Icc 0 ↑b) a
  -/
  replace hb : (1 : ℝ) ≤ b := Nat.one_le_cast.2 hb
  /-
    case intro.intro.intro.intro.intro.intro
    p : Real
    hp : GT.gt p 2
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    x : Real
    hxp : LiouvilleWith p x
    hx01 : Membership.mem (Set.Ico 0 1) x
    b : Nat
    a : Int
    hlt : LT.lt (abs (HSub.hSub x (HDiv.hDiv ↑a ↑b))) (HDiv.hDiv 1 (HPow.hPow (↑b) …
    hb : LE.le 1 ↑b
    ⊢ Membership.mem (Finset.Icc 0 ↑b) a
  -/
  have hb0 : (0 : ℝ) < b := zero_lt_one.trans_le hb
  replace hlt : |x - a / b| < 1 / b := by
    refine hlt.trans_le (one_div_le_one_div_of_le hb0 ?_)
    calc
      (b : ℝ) = (b : ℝ) ^ (1 : ℝ) := (rpow_one _).symm
      _ ≤ (b : ℝ) ^ (2 + 1 / (n + 1 : ℕ) : ℝ) :=
        rpow_le_rpow_of_exponent_le hb (one_le_two.trans ?_)
    simpa using n.cast_add_one_pos.le
  rw [sub_div' _ _ _ hb0.ne', abs_div, abs_of_pos hb0, div_lt_div_iff_of_pos_right hb0,
    abs_sub_lt_iff, sub_lt_iff_lt_add, sub_lt_iff_lt_add, ← sub_lt_iff_lt_add'] at hlt
  rw [Finset.mem_Icc, ← Int.lt_add_one_iff, ← Int.lt_add_one_iff, ← neg_lt_iff_pos_add, add_comm, ←
    @Int.cast_lt ℝ, ← @Int.cast_lt ℝ]
  /-
    case intro.intro.intro.intro.intro.intro
    p : Real
    hp : GT.gt p 2
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    x : Real
    hxp : LiouvilleWith p x
    hx01 : Membership.mem (Set.Ico 0 1) x
    b : Nat
    a : Int
    hb : LE.le 1 ↑b
    hb0 : LT.lt 0 ↑b
    hlt : And (LT.lt (HSub.hSub (HMul.hMul x ↑b) 1) ↑a) (LT.lt (↑a) (HAdd.hAdd 1 ( …
    ⊢ And (LT.lt ↑(-1) ↑a) (LT.lt ↑a ↑(HAdd.hAdd 1 ↑b))
  -/
  push_cast
  /-
    case intro.intro.intro.intro.intro.intro
    p : Real
    hp : GT.gt p 2
    n : Nat
    hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
    x : Real
    hxp : LiouvilleWith p x
    hx01 : Membership.mem (Set.Ico 0 1) x
    b : Nat
    a : Int
    hb : LE.le 1 ↑b
    hb0 : LT.lt 0 ↑b
    hlt : And (LT.lt (HSub.hSub (HMul.hMul x ↑b) 1) ↑a) (LT.lt (↑a) (HAdd.hAdd 1 ( …
    ⊢ And (LT.lt (-1) ↑a) (LT.lt (↑a) (HAdd.hAdd 1 ↑b))
  -/
  refine ⟨lt_of_le_of_lt ?_ hlt.1, hlt.2.trans_le ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      p : Real
      hp : GT.gt p 2
      n : Nat
      hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
      x : Real
      hxp : LiouvilleWith p x
      hx01 : Membership.mem (Set.Ico 0 1) x
      b : Nat
      a : Int
      hb : LE.le 1 ↑b
      hb0 : LT.lt 0 ↑b
      hlt : And (LT.lt (HSub.hSub (HMul.hMul x ↑b) 1) ↑a) (LT.lt (↑a) (HAdd.hAdd 1 ( …
      ⊢ LE.le (-1) (HSub.hSub (HMul.hMul x ↑b) 1)
    -/
  · simp only [mul_nonneg hx01.left b.cast_nonneg, neg_le_sub_iff_le_add, le_add_iff_nonneg_left]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      p : Real
      hp : GT.gt p 2
      n : Nat
      hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
      x : Real
      hxp : LiouvilleWith p x
      hx01 : Membership.mem (Set.Ico 0 1) x
      b : Nat
      a : Int
      hb : LE.le 1 ↑b
      hb0 : LT.lt 0 ↑b
      hlt : And (LT.lt (HSub.hSub (HMul.hMul x ↑b) 1) ↑a) (LT.lt (↑a) (HAdd.hAdd 1 ( …
      ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul x ↑b)) (HAdd.hAdd 1 ↑b)
    -/
  · rw [add_le_add_iff_left]
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      p : Real
      hp : GT.gt p 2
      n : Nat
      hn : LT.lt (HAdd.hAdd 2 (HDiv.hDiv 1 (HAdd.hAdd (↑n) 1))) p
      x : Real
      hxp : LiouvilleWith p x
      hx01 : Membership.mem (Set.Ico 0 1) x
      b : Nat
      a : Int
      hb : LE.le 1 ↑b
      hb0 : LT.lt 0 ↑b
      hlt : And (LT.lt (HSub.hSub (HMul.hMul x ↑b) 1) ↑a) (LT.lt (↑a) (HAdd.hAdd 1 ( …
      ⊢ LE.le (HMul.hMul x ↑b) ↑b
    -/
    exact mul_le_of_le_one_left hb0.le hx01.2.le
    /-
      🎉 no goals
    -/


/-- The set of numbers satisfying the Liouville condition with some exponent `p > 2` has Lebesgue
measure zero. -/
@[simp]
theorem volume_iUnion_setOf_liouvilleWith :
    volume (⋃ (p : ℝ) (_hp : 2 < p), { x : ℝ | LiouvilleWith p x }) = 0 := by
  /-
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.iUnion fun p => Set.iUnion fun _h …
  -/
  simp only [← setOf_exists, exists_prop]
  /-
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => Exists fun i => And (L …
  -/
  refine measure_mono_null setOf_liouvilleWith_subset_aux ?_
  /-
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.iUnion fun m => Set.preimage (fun …
  -/
  rw [measure_iUnion_null_iff]; intro m; rw [measure_preimage_add_right]; clear m
  /-
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (Set.iUnion fun n => Set.iUnion fun h  …
  -/
  refine (measure_biUnion_null_iff <| to_countable _).2 fun n (hn : 1 ≤ n) => ?_
  /-
    n : Nat
    hn : LE.le 1 n
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => Filter.Frequently (fun …
  -/
  generalize hr : (2 + 1 / n : ℝ) = r
  /-
    n : Nat
    hn : LE.le 1 n
    r : Real
    hr : Eq (HAdd.hAdd 2 (HDiv.hDiv 1 ↑n)) r
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => Filter.Frequently (fun …
  -/
  replace hr : 2 < r := by simp [← hr, zero_lt_one.trans_le hn]
  /-
    n : Nat
    hn : LE.le 1 n
    r : Real
    hr : LT.lt 2 r
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => Filter.Frequently (fun …
  -/
  clear hn n
  /-
    r : Real
    hr : LT.lt 2 r
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => Filter.Frequently (fun …
  -/
  refine measure_setOf_frequently_eq_zero ?_
  /-
    r : Real
    hr : LT.lt 2 r
    ⊢ Ne (tsum fun i => MeasureTheory.MeasureSpace.volume (setOf fun x => Exists f …
  -/
  simp only [setOf_exists, ← exists_prop, ← Real.dist_eq, ← mem_ball, setOf_mem_eq]
  /-
    r : Real
    hr : LT.lt 2 r
    ⊢ Ne (tsum fun i => MeasureTheory.MeasureSpace.volume (Set.iUnion fun i_1 => S …
  -/
  set B : ℤ → ℕ → Set ℝ := fun a b => ball (a / b) (1 / (b : ℝ) ^ r)
  have hB : ∀ a b, volume (B a b) = ↑((2 : ℝ≥0) / (b : ℝ≥0) ^ r) := fun a b ↦ by
    rw [Real.volume_ball, mul_one_div, ← NNReal.coe_two, ← NNReal.coe_natCast, ← NNReal.coe_rpow,
      ← NNReal.coe_div, ENNReal.ofReal_coe_nnreal]
  have : ∀ b : ℕ, volume (⋃ a ∈ Finset.Icc (0 : ℤ) b, B a b) ≤
      ↑(2 * ((b : ℝ≥0) ^ (1 - r) + (b : ℝ≥0) ^ (-r))) := fun b ↦
    calc
      volume (⋃ a ∈ Finset.Icc (0 : ℤ) b, B a b) ≤ ∑ a ∈ Finset.Icc (0 : ℤ) b, volume (B a b) :=
        measure_biUnion_finset_le _ _
      _ = ↑((b + 1) * (2 / (b : ℝ≥0) ^ r)) := by
        simp only [hB, Int.card_Icc, Finset.sum_const, nsmul_eq_mul, sub_zero, ← Int.ofNat_succ,
          Int.toNat_natCast, ← Nat.cast_succ, ENNReal.coe_mul, ENNReal.coe_natCast]
      _ = _ := by
        have : 1 - r ≠ 0 := by linarith
        rw [ENNReal.coe_inj]
        simp [add_mul, div_eq_mul_inv, NNReal.rpow_neg, NNReal.rpow_sub' this, mul_add,
          mul_left_comm]
  /-
    r : Real
    hr : LT.lt 2 r
    B : Int → Nat → Set Real := fun a b => Metric.ball (HDiv.hDiv ↑a ↑b) (HDiv.hDi …
    hB : ∀ (a : Int) (b : Nat), Eq (MeasureTheory.MeasureSpace.volume (B a b)) ↑(H …
    this : ∀ (b : Nat), LE.le (MeasureTheory.MeasureSpace.volume (Set.iUnion fun a …
    ⊢ Ne (tsum fun i => MeasureTheory.MeasureSpace.volume (Set.iUnion fun i_1 => S …
  -/
  refine ne_top_of_le_ne_top (ENNReal.tsum_coe_ne_top_iff_summable.2 ?_) (ENNReal.tsum_le_tsum this)
  /-
    r : Real
    hr : LT.lt 2 r
    B : Int → Nat → Set Real := fun a b => Metric.ball (HDiv.hDiv ↑a ↑b) (HDiv.hDi …
    hB : ∀ (a : Int) (b : Nat), Eq (MeasureTheory.MeasureSpace.volume (B a b)) ↑(H …
    this : ∀ (b : Nat), LE.le (MeasureTheory.MeasureSpace.volume (Set.iUnion fun a …
    ⊢ Summable fun a => HMul.hMul 2 (HAdd.hAdd (HPow.hPow (↑a) (HSub.hSub 1 r)) (H …
  -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
  refine (Summable.add ?_ ?_).mul_left _ <;> simp only [NNReal.summable_rpow] <;> linarith
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem ae_not_liouvilleWith : ∀ᵐ x, ∀ p > (2 : ℝ), ¬LiouvilleWith p x := by
  simpa only [ae_iff, not_forall, Classical.not_not, setOf_exists] using
    volume_iUnion_setOf_liouvilleWith


theorem ae_not_liouville : ∀ᵐ x, ¬Liouville x :=
                                                    /-
                                                      x✝ : Real
                                                      h₁ : ∀ (p : Real), GT.gt p 2 → Not (LiouvilleWith p x✝)
                                                      h₂ : Liouville x✝
                                                      ⊢ GT.gt 3 2
                                                    -/
  ae_not_liouvilleWith.mono fun _ h₁ h₂ => h₁ 3 (by norm_num) (h₂.liouvilleWith 3)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The set of Liouville numbers has Lebesgue measure zero. -/
@[simp]
theorem volume_setOf_liouville : volume { x : ℝ | Liouville x } = 0 := by
  /-
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => Liouville x)) 0
  -/
  simpa only [ae_iff, Classical.not_not] using ae_not_liouville
  /-
    🎉 no goals
  -/


/-- The filters `residual ℝ` and `ae volume` are disjoint. This means that there exists a residual
set of Lebesgue measure zero (e.g., the set of Liouville numbers). -/
theorem Real.disjoint_residual_ae : Disjoint (residual ℝ) (ae volume) :=
  disjoint_of_disjoint_of_mem disjoint_compl_right eventually_residual_liouville ae_not_liouville

