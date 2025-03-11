theorem ne_zero_iff_exists_coeff_ne_zero_and_weight :
    f ≠ 0 ↔ (∃ n : ℕ, ∃ d : σ →₀ ℕ, coeff R d f ≠ 0 ∧ weight w d = n) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    ⊢ Iff (Ne f 0) (Exists fun n => Exists fun d => And (Ne ((MvPowerSeries.coeff  …
  -/
  refine not_iff_not.mp ?_
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    ⊢ Iff (Not (Ne f 0)) (Not (Exists fun n => Exists fun d => And (Ne ((MvPowerSe …
  -/
  simp only [ne_eq, not_not, not_exists, not_and, forall_apply_eq_imp_iff₂, imp_false]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    ⊢ Iff (Eq f 0) (∀ (a : Finsupp σ Nat), Eq ((MvPowerSeries.coeff R a) f) 0)
  -/
  exact MvPowerSeries.ext_iff
  /-
    🎉 no goals
  -/


/-- The weighted order of a mv_power_series -/
def weightedOrder (f : MvPowerSeries σ R) : ℕ∞ := by
  classical
  exact dite (f = 0) (fun _ => ⊤) fun h =>
    Nat.find ((ne_zero_iff_exists_coeff_ne_zero_and_weight w).mp h)


@[simp] theorem weightedOrder_zero : (0 : MvPowerSeries σ R).weightedOrder w = ⊤ := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    ⊢ Eq (MvPowerSeries.weightedOrder w 0) Top.top
  -/
  rw [weightedOrder, dif_pos rfl]
  /-
    🎉 no goals
  -/


theorem ne_zero_iff_weightedOrder_finite :
    f ≠ 0 ↔ (f.weightedOrder w).toNat = f.weightedOrder w := by
  simp only [weightedOrder, ne_eq, coe_toNat_eq_self, dite_eq_left_iff,
    ENat.coe_ne_top, imp_false, not_not]


/-- The `0` power series is the unique power series with infinite order.-/
@[simp]
theorem weightedOrder_eq_top_iff :
    f.weightedOrder w = ⊤ ↔ f = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    ⊢ Iff (Eq (MvPowerSeries.weightedOrder w f) Top.top) (Eq f 0)
  -/
  rw [← not_iff_not, ← ne_eq, ← ne_eq,   ne_zero_iff_weightedOrder_finite w, coe_toNat_eq_self]
  /-
    🎉 no goals
  -/


/-- If the order of a formal power series `f` is finite,
then some coefficient of weight equal to the order of `f` is nonzero.-/
theorem exists_coeff_ne_zero_and_weightedOrder
    (h : (toNat (f.weightedOrder w) : ℕ∞) = f.weightedOrder w) :
    ∃ d, coeff R d f ≠ 0 ∧ weight w d = f.weightedOrder w := by
  classical
  simp_rw [weightedOrder, dif_neg ((ne_zero_iff_weightedOrder_finite w).mpr h), Nat.cast_inj]
  generalize_proofs h1
  exact Nat.find_spec h1


/-- If the `d`th coefficient of a formal power series is nonzero,
then the weighted order of the power series is less than or equal to `weight d w`.-/
theorem weightedOrder_le {d : σ →₀ ℕ} (h : coeff R d f ≠ 0) :
    f.weightedOrder w ≤ weight w d := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Ne ((MvPowerSeries.coeff R d) f) 0
    ⊢ LE.le (MvPowerSeries.weightedOrder w f) ↑((Finsupp.weight w) d)
  -/
  rw [weightedOrder, dif_neg]
    /-
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      d : Finsupp σ Nat
      h : Ne ((MvPowerSeries.coeff R d) f) 0
      ⊢ LE.le ↑(Nat.find ⋯) ↑((Finsupp.weight w) d)
    -/
  · simp only [ne_eq, Nat.cast_le, Nat.find_le_iff]
    /-
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      d : Finsupp σ Nat
      h : Ne ((MvPowerSeries.coeff R d) f) 0
      ⊢ Exists fun m => And (LE.le m ((Finsupp.weight w) d)) (Exists fun d => And (N …
    -/
    exact ⟨weight w d, le_rfl, d, h, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case hnc
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      d : Finsupp σ Nat
      h : Ne ((MvPowerSeries.coeff R d) f) 0
      ⊢ Not (Eq f 0)
    -/
  · exact (f.ne_zero_iff_exists_coeff_ne_zero_and_weight w).mpr ⟨weight w d, d, h, rfl⟩
    /-
      🎉 no goals
    -/


/-- The `n`th coefficient of a formal power series is `0` if `n` is strictly
smaller than the order of the power series.-/
theorem coeff_eq_zero_of_lt_weightedOrder {d : σ →₀ ℕ} (h : (weight w d) < f.weightedOrder w) :
    coeff R d f = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w f)
    ⊢ Eq ((MvPowerSeries.coeff R d) f) 0
  -/
  contrapose! h; exact weightedOrder_le w h
                 /-
                   🎉 no goals
                 -/


/-- The order of a formal power series is at least `n` if
the `d`th coefficient is `0` for all `d` such that `weight w d < n`.-/
theorem nat_le_weightedOrder {n : ℕ} (h : ∀ d, weight w d < n → coeff R d f = 0) :
    n ≤ f.weightedOrder w := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    n : Nat
    h : ∀ (d : Finsupp σ Nat), LT.lt ((Finsupp.weight w) d) n → Eq ((MvPowerSeries …
    ⊢ LE.le (↑n) (MvPowerSeries.weightedOrder w f)
  -/
  by_contra! H
  have : (f.weightedOrder w).toNat = f.weightedOrder w := by
    rw [coe_toNat_eq_self]; exact ne_top_of_lt H
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    n : Nat
    h : ∀ (d : Finsupp σ Nat), LT.lt ((Finsupp.weight w) d) n → Eq ((MvPowerSeries …
    H : LT.lt (MvPowerSeries.weightedOrder w f) ↑n
    this : Eq (↑(MvPowerSeries.weightedOrder w f).toNat) (MvPowerSeries.weightedOr …
    ⊢ False
  -/
  obtain ⟨d, hfd, hd⟩ := exists_coeff_ne_zero_and_weightedOrder w this
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    n : Nat
    h : ∀ (d : Finsupp σ Nat), LT.lt ((Finsupp.weight w) d) n → Eq ((MvPowerSeries …
    H : LT.lt (MvPowerSeries.weightedOrder w f) ↑n
    this : Eq (↑(MvPowerSeries.weightedOrder w f).toNat) (MvPowerSeries.weightedOr …
    d : Finsupp σ Nat
    hfd : Ne ((MvPowerSeries.coeff R d) f) 0
    hd : Eq (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w f)
    ⊢ False
  -/
  rw [← hd, Nat.cast_lt] at H
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    n : Nat
    h : ∀ (d : Finsupp σ Nat), LT.lt ((Finsupp.weight w) d) n → Eq ((MvPowerSeries …
    this : Eq (↑(MvPowerSeries.weightedOrder w f).toNat) (MvPowerSeries.weightedOr …
    d : Finsupp σ Nat
    H : LT.lt ((Finsupp.weight w) d) n
    hfd : Ne ((MvPowerSeries.coeff R d) f) 0
    hd : Eq (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w f)
    ⊢ False
  -/
  exact hfd (h d H)
  /-
    🎉 no goals
  -/


/-- The order of a formal power series is at least `n` if
the `d`th coefficient is `0` for all `d` such that `weight w d < n`.-/
theorem le_weightedOrder {n : ℕ∞} (h : ∀ d : σ →₀ ℕ, weight w d < n → coeff R d f = 0) :
    n ≤ f.weightedOrder w := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    n : ENat
    h : ∀ (d : Finsupp σ Nat), LT.lt (↑((Finsupp.weight w) d)) n → Eq ((MvPowerSer …
    ⊢ LE.le n (MvPowerSeries.weightedOrder w f)
  -/
  cases n
    /-
      case top
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      h : ∀ (d : Finsupp σ Nat), LT.lt (↑((Finsupp.weight w) d)) Top.top → Eq ((MvPo …
      ⊢ LE.le Top.top (MvPowerSeries.weightedOrder w f)
    -/
  · rw [top_le_iff, weightedOrder_eq_top_iff]
    /-
      case top
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      h : ∀ (d : Finsupp σ Nat), LT.lt (↑((Finsupp.weight w) d)) Top.top → Eq ((MvPo …
      ⊢ Eq f 0
    -/
    ext d; exact h d (ENat.coe_lt_top _)
           /-
             🎉 no goals
           -/
    /-
      case coe
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      a✝ : Nat
      h : ∀ (d : Finsupp σ Nat), LT.lt ↑((Finsupp.weight w) d) ↑a✝ → Eq ((MvPowerSer …
      ⊢ LE.le (↑a✝) (MvPowerSeries.weightedOrder w f)
    -/
  · apply nat_le_weightedOrder;
    /-
      case coe.h
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      a✝ : Nat
      h : ∀ (d : Finsupp σ Nat), LT.lt ↑((Finsupp.weight w) d) ↑a✝ → Eq ((MvPowerSer …
      ⊢ ∀ (d : Finsupp σ Nat), LT.lt ((Finsupp.weight w) d) a✝ → Eq ((MvPowerSeries. …
    -/
    simpa only [ENat.some_eq_coe, Nat.cast_lt] using h
    /-
      🎉 no goals
    -/


/-- The order of a formal power series is exactly `n` if and only if some coefficient of weight `n`
is nonzero, and the `d`th coefficient is `0` for all `d` such that `weight w d < n`.-/
theorem weightedOrder_eq_nat {n : ℕ} :
    f.weightedOrder w = n ↔
      (∃ d, coeff R d f ≠ 0 ∧ weight w d = n) ∧ ∀ d, weight w d < n → coeff R d f = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f : MvPowerSeries σ R
    n : Nat
    ⊢ Iff (Eq (MvPowerSeries.weightedOrder w f) ↑n) (And (Exists fun d => And (Ne  …
  -/
  constructor
    /-
      case mp
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      n : Nat
      ⊢ Eq (MvPowerSeries.weightedOrder w f) ↑n → And (Exists fun d => And (Ne ((MvP …
    -/
  · intro h
    /-
      case mp
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      n : Nat
      h : Eq (MvPowerSeries.weightedOrder w f) ↑n
      ⊢ And (Exists fun d => And (Ne ((MvPowerSeries.coeff R d) f) 0) (Eq ((Finsupp. …
    -/
    obtain ⟨d, hd⟩ := f.exists_coeff_ne_zero_and_weightedOrder w (by simp only [h, toNat_coe])
    exact ⟨⟨d, by simpa [h, Nat.cast_inj, ne_eq] using hd⟩,
      fun e he ↦ f.coeff_eq_zero_of_lt_weightedOrder w (by simp only [h, Nat.cast_lt, he])⟩
    /-
      case mpr
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      n : Nat
      ⊢ And (Exists fun d => And (Ne ((MvPowerSeries.coeff R d) f) 0) (Eq ((Finsupp. …
    -/
  · rintro ⟨⟨d, hd', hd⟩, h⟩
    /-
      case mpr.intro.intro.intro
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f : MvPowerSeries σ R
      n : Nat
      h : ∀ (d : Finsupp σ Nat), LT.lt ((Finsupp.weight w) d) n → Eq ((MvPowerSeries …
      d : Finsupp σ Nat
      hd' : Ne ((MvPowerSeries.coeff R d) f) 0
      hd : Eq ((Finsupp.weight w) d) n
      ⊢ Eq (MvPowerSeries.weightedOrder w f) ↑n
    -/
    exact le_antisymm (hd.symm ▸ f.weightedOrder_le w hd') (nat_le_weightedOrder w h)
    /-
      🎉 no goals
    -/


/-- The weighted_order of the monomial `a*X^d` is infinite if `a = 0` and `weight w d` otherwise.-/
theorem weightedOrder_monomial {d : σ →₀ ℕ} {a : R} [Decidable (a = 0)] :
    weightedOrder w (monomial R d a) = if a = 0 then (⊤ : ℕ∞) else weight w d := by
  classical
  split_ifs with h
  · rw [h, weightedOrder_eq_top_iff, LinearMap.map_zero]
  · rw [weightedOrder_eq_nat]
    constructor
    · use d
      simp only [coeff_monomial_same, ne_eq, h, not_false_eq_true, and_self]
    · intro b hb
      rw [coeff_monomial, if_neg]
      intro h
      simp only [h, lt_self_iff_false] at hb


/-- The order of the monomial `a*X^n` is `n` if `a ≠ 0`.-/
theorem weightedOrder_monomial_of_ne_zero {d : σ →₀ ℕ} {a : R} (h : a ≠ 0) :
    weightedOrder w (monomial R d a) = weight w d := by
  classical
  rw [weightedOrder_monomial, if_neg h]



/-- The order of the sum of two formal power series is at least the minimum of their orders.-/
theorem min_weightedOrder_le_add :
    min (f.weightedOrder w) (g.weightedOrder w) ≤ (f + g).weightedOrder w := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f g : MvPowerSeries σ R
    ⊢ LE.le (Min.min (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrde …
  -/
  apply le_weightedOrder w
  simp (config := { contextual := true }) only
    [coeff_eq_zero_of_lt_weightedOrder w, lt_min_iff, map_add, add_zero,
      eq_self_iff_true, imp_true_iff]


private theorem weightedOrder_add_of_weightedOrder_lt.aux
    (H : f.weightedOrder w < g.weightedOrder w) :
    (f + g).weightedOrder w = f.weightedOrder w := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f g : MvPowerSeries σ R
    H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
    ⊢ Eq (MvPowerSeries.weightedOrder w (HAdd.hAdd f g)) (MvPowerSeries.weightedOr …
  -/
  obtain ⟨n, hn : (n : ℕ∞) = _⟩ := ENat.ne_top_iff_exists.mp (ne_top_of_lt H)
  /-
    case intro
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f g : MvPowerSeries σ R
    H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
    n : Nat
    hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
    ⊢ Eq (MvPowerSeries.weightedOrder w (HAdd.hAdd f g)) (MvPowerSeries.weightedOr …
  -/
  rw [← hn, weightedOrder_eq_nat]
  /-
    case intro
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f g : MvPowerSeries σ R
    H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
    n : Nat
    hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
    ⊢ And (Exists fun d => And (Ne ((MvPowerSeries.coeff R d) (HAdd.hAdd f g)) 0)  …
  -/
  obtain ⟨d, hd', hd⟩ := ((weightedOrder_eq_nat w).mp hn.symm).1
  /-
    case intro.intro.intro
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f g : MvPowerSeries σ R
    H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
    n : Nat
    hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
    d : Finsupp σ Nat
    hd' : Ne ((MvPowerSeries.coeff R d) f) 0
    hd : Eq ((Finsupp.weight w) d) n
    ⊢ And (Exists fun d => And (Ne ((MvPowerSeries.coeff R d) (HAdd.hAdd f g)) 0)  …
  -/
  constructor
    /-
      case intro.intro.intro.left
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      n : Nat
      hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
      d : Finsupp σ Nat
      hd' : Ne ((MvPowerSeries.coeff R d) f) 0
      hd : Eq ((Finsupp.weight w) d) n
      ⊢ Exists fun d => And (Ne ((MvPowerSeries.coeff R d) (HAdd.hAdd f g)) 0) (Eq ( …
    -/
  · refine ⟨d, ?_, hd⟩
    /-
      case intro.intro.intro.left
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      n : Nat
      hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
      d : Finsupp σ Nat
      hd' : Ne ((MvPowerSeries.coeff R d) f) 0
      hd : Eq ((Finsupp.weight w) d) n
      ⊢ Ne ((MvPowerSeries.coeff R d) (HAdd.hAdd f g)) 0
    -/
    rw [← hn, ← hd] at H
    /-
      case intro.intro.intro.left
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      n : Nat
      hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
      d : Finsupp σ Nat
      H : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w g)
      hd' : Ne ((MvPowerSeries.coeff R d) f) 0
      hd : Eq ((Finsupp.weight w) d) n
      ⊢ Ne ((MvPowerSeries.coeff R d) (HAdd.hAdd f g)) 0
    -/
    rw [(coeff _ _).map_add, coeff_eq_zero_of_lt_weightedOrder w H, add_zero]
    /-
      case intro.intro.intro.left
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      n : Nat
      hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
      d : Finsupp σ Nat
      H : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w g)
      hd' : Ne ((MvPowerSeries.coeff R d) f) 0
      hd : Eq ((Finsupp.weight w) d) n
      ⊢ Ne ((MvPowerSeries.coeff R d) f) 0
    -/
    exact hd'
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.right
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      n : Nat
      hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
      d : Finsupp σ Nat
      hd' : Ne ((MvPowerSeries.coeff R d) f) 0
      hd : Eq ((Finsupp.weight w) d) n
      ⊢ ∀ (d : Finsupp σ Nat), LT.lt ((Finsupp.weight w) d) n → Eq ((MvPowerSeries.c …
    -/
  · intro b hb
    suffices weight w b < weightedOrder w f by
      rw [(coeff _ _).map_add, coeff_eq_zero_of_lt_weightedOrder w this,
        coeff_eq_zero_of_lt_weightedOrder w (lt_trans this H), add_zero]
    /-
      case intro.intro.intro.right
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      n : Nat
      hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
      d : Finsupp σ Nat
      hd' : Ne ((MvPowerSeries.coeff R d) f) 0
      hd : Eq ((Finsupp.weight w) d) n
      b : Finsupp σ Nat
      hb : LT.lt ((Finsupp.weight w) b) n
      ⊢ LT.lt (↑((Finsupp.weight w) b)) (MvPowerSeries.weightedOrder w f)
    -/
    rw [← hn, Nat.cast_lt]
    /-
      case intro.intro.intro.right
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      H : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      n : Nat
      hn : Eq (↑n) (MvPowerSeries.weightedOrder w f)
      d : Finsupp σ Nat
      hd' : Ne ((MvPowerSeries.coeff R d) f) 0
      hd : Eq ((Finsupp.weight w) d) n
      b : Finsupp σ Nat
      hb : LT.lt ((Finsupp.weight w) b) n
      ⊢ LT.lt ((Finsupp.weight w) b) n
    -/
    exact hb
    /-
      🎉 no goals
    -/


/-- The weighted_order of the sum of two formal power series
 is the minimum of their orders if their orders differ.-/
theorem weightedOrder_add_of_weightedOrder_ne (h : f.weightedOrder w ≠ g.weightedOrder w) :
    weightedOrder w (f + g) = weightedOrder w f ⊓ weightedOrder w g := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f g : MvPowerSeries σ R
    h : Ne (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
    ⊢ Eq (MvPowerSeries.weightedOrder w (HAdd.hAdd f g)) (Min.min (MvPowerSeries.w …
  -/
  refine le_antisymm ?_ (min_weightedOrder_le_add w)
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    w : σ → Nat
    f g : MvPowerSeries σ R
    h : Ne (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
    ⊢ LE.le (MvPowerSeries.weightedOrder w (HAdd.hAdd f g)) (Min.min (MvPowerSerie …
  -/
  by_cases H₁ : f.weightedOrder w < g.weightedOrder w
    /-
      case pos
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      h : Ne (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      H₁ : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      ⊢ LE.le (MvPowerSeries.weightedOrder w (HAdd.hAdd f g)) (Min.min (MvPowerSerie …
    -/
  · simp only [le_inf_iff, weightedOrder_add_of_weightedOrder_lt.aux w H₁]
    /-
      case pos
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      h : Ne (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      H₁ : LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      ⊢ And (LE.le (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w  …
    -/
    exact ⟨le_rfl, le_of_lt H₁⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      f g : MvPowerSeries σ R
      h : Ne (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
      H₁ : Not (LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder …
      ⊢ LE.le (MvPowerSeries.weightedOrder w (HAdd.hAdd f g)) (Min.min (MvPowerSerie …
    -/
  · by_cases H₂ : g.weightedOrder w < f.weightedOrder w
      /-
        case pos
        σ : Type u_1
        R : Type u_2
        inst✝ : Semiring R
        w : σ → Nat
        f g : MvPowerSeries σ R
        h : Ne (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
        H₁ : Not (LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder …
        H₂ : LT.lt (MvPowerSeries.weightedOrder w g) (MvPowerSeries.weightedOrder w f)
        ⊢ LE.le (MvPowerSeries.weightedOrder w (HAdd.hAdd f g)) (Min.min (MvPowerSerie …
      -/
    · simp only [add_comm f g, le_inf_iff, weightedOrder_add_of_weightedOrder_lt.aux w H₂]
      /-
        case pos
        σ : Type u_1
        R : Type u_2
        inst✝ : Semiring R
        w : σ → Nat
        f g : MvPowerSeries σ R
        h : Ne (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
        H₁ : Not (LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder …
        H₂ : LT.lt (MvPowerSeries.weightedOrder w g) (MvPowerSeries.weightedOrder w f)
        ⊢ And (LE.le (MvPowerSeries.weightedOrder w g) (MvPowerSeries.weightedOrder w  …
      -/
      exact ⟨le_of_lt H₂, le_rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝ : Semiring R
        w : σ → Nat
        f g : MvPowerSeries σ R
        h : Ne (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder w g)
        H₁ : Not (LT.lt (MvPowerSeries.weightedOrder w f) (MvPowerSeries.weightedOrder …
        H₂ : Not (LT.lt (MvPowerSeries.weightedOrder w g) (MvPowerSeries.weightedOrder …
        ⊢ LE.le (MvPowerSeries.weightedOrder w (HAdd.hAdd f g)) (Min.min (MvPowerSerie …
      -/
    · exact absurd (le_antisymm (not_lt.1 H₂) (not_lt.1 H₁)) h
      /-
        🎉 no goals
      -/


/-- The weighted_order of the product of two formal power series
 is at least the sum of their orders.-/
theorem le_weightedOrder_mul :
    f.weightedOrder w + g.weightedOrder w ≤ weightedOrder w (f * g) := by
  classical
  apply le_weightedOrder
  intro d hd
  rw [coeff_mul, Finset.sum_eq_zero]
  rintro ⟨i, j⟩ hij
  by_cases hi : weight w i < f.weightedOrder w
  · rw [coeff_eq_zero_of_lt_weightedOrder w hi, MulZeroClass.zero_mul]
  · by_cases hj : weight w j < g.weightedOrder w
    · rw [coeff_eq_zero_of_lt_weightedOrder w hj, MulZeroClass.mul_zero]
    · rw [not_lt] at hi hj
      simp only [Finset.mem_antidiagonal] at hij
      exfalso
      apply ne_of_lt (lt_of_lt_of_le hd <| add_le_add hi hj)
      rw [← hij, map_add, Nat.cast_add]


alias weightedOrder_mul_ge := le_weightedOrder_mul


theorem coeff_mul_left_one_sub_of_lt_weightedOrder
    {d : σ →₀ ℕ} (h : (weight w d) < g.weightedOrder w) :
    coeff R d (f * (1 - g)) = coeff R d f := by
  /-
    σ : Type u_1
    w : σ → Nat
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w g)
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul f (HSub.hSub 1 g))) ((MvPowerSeries …
  -/
  simp only [mul_sub, mul_one, _root_.map_sub, sub_eq_self]
  /-
    σ : Type u_1
    w : σ → Nat
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w g)
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul f g)) 0
  -/
  apply coeff_eq_zero_of_lt_weightedOrder w
  /-
    σ : Type u_1
    w : σ → Nat
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w g)
    ⊢ LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w (HMul.hMul f  …
  -/
  exact lt_of_lt_of_le (lt_of_lt_of_le h le_add_self) (le_weightedOrder_mul w)
  /-
    🎉 no goals
  -/


theorem coeff_mul_right_one_sub_of_lt_weightedOrder
    {d : σ →₀ ℕ} (h : (weight w d) < g.weightedOrder w) :
    coeff R d ((1 - g) * f) = coeff R d f := by
  /-
    σ : Type u_1
    w : σ → Nat
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w g)
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul (HSub.hSub 1 g) f)) ((MvPowerSeries …
  -/
  simp only [sub_mul, one_mul, _root_.map_sub, sub_eq_self]
  /-
    σ : Type u_1
    w : σ → Nat
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w g)
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul g f)) 0
  -/
  apply coeff_eq_zero_of_lt_weightedOrder w
  /-
    σ : Type u_1
    w : σ → Nat
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w g)
    ⊢ LT.lt (↑((Finsupp.weight w) d)) (MvPowerSeries.weightedOrder w (HMul.hMul g  …
  -/
  apply lt_of_lt_of_le (lt_of_lt_of_le h le_self_add) (le_weightedOrder_mul w)
  /-
    🎉 no goals
  -/


theorem coeff_mul_prod_one_sub_of_lt_weightedOrder {R ι : Type*} [CommRing R] (d : σ →₀ ℕ)
    (s : Finset ι) (f : MvPowerSeries σ R) (g : ι → MvPowerSeries σ R) :
    (∀ i ∈ s, (weight w d) < weightedOrder w (g i)) →
      coeff R d (f * ∏ i in s, (1 - g i)) = coeff R d f := by
  classical
  induction s using Finset.induction_on with
  | empty => simp only [imp_true_iff, Finset.prod_empty, mul_one, eq_self_iff_true]
  | @insert a s ha ih =>
    intro h
    simp only [Finset.mem_insert, forall_eq_or_imp] at h
    rw [Finset.prod_insert ha, ← mul_assoc, mul_right_comm,
      coeff_mul_left_one_sub_of_lt_weightedOrder w h.1, ih h.2]


theorem eq_zero_iff_forall_coeff_eq_zero_and :
    f = 0 ↔ (∀ d : σ →₀ ℕ, coeff R d f = 0) :=
  MvPowerSeries.ext_iff


theorem ne_zero_iff_exists_coeff_ne_zero_and_degree :
    f ≠ 0 ↔ (∃ n : ℕ, ∃ d : σ →₀ ℕ, coeff R d f ≠ 0 ∧ degree d = n) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    ⊢ Iff (Ne f 0) (Exists fun n => Exists fun d => And (Ne ((MvPowerSeries.coeff  …
  -/
  simp_rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    ⊢ Iff (Ne f 0) (Exists fun n => Exists fun d => And (Ne ((MvPowerSeries.coeff  …
  -/
  exact ne_zero_iff_exists_coeff_ne_zero_and_weight (fun _ => 1)
  /-
    🎉 no goals
  -/


/-- The order of a mv_power_series -/
def order (f : MvPowerSeries σ R) : ℕ∞ := weightedOrder (fun _ => 1) f


@[simp]
theorem order_zero : (0 : MvPowerSeries σ R).order = ⊤ :=
  weightedOrder_zero _


theorem ne_zero_iff_order_finite : f ≠ 0 ↔ f.order.toNat = f.order :=
  ne_zero_iff_weightedOrder_finite 1


/-- The `0` power series is the unique power series with infinite order.-/
@[simp] theorem order_eq_top_iff : f.order = ⊤ ↔ f = 0 :=
  weightedOrder_eq_top_iff _


/-- If the order of a formal power series `f` is finite,
then some coefficient of degree the order of `f` is nonzero.-/
theorem exists_coeff_ne_zero_and_order (h : f.order.toNat = f.order) :
    ∃ d : σ →₀ ℕ, coeff R d f ≠ 0 ∧ degree d = f.order := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    h : Eq (↑f.order.toNat) f.order
    ⊢ Exists fun d => And (Ne ((MvPowerSeries.coeff R d) f) 0) (Eq (↑d.degree) f.o …
  -/
  simp_rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    h : Eq (↑f.order.toNat) f.order
    ⊢ Exists fun d => And (Ne ((MvPowerSeries.coeff R d) f) 0) (Eq (↑((Finsupp.wei …
  -/
  exact exists_coeff_ne_zero_and_weightedOrder _ h
  /-
    🎉 no goals
  -/


/-- If the `d`th coefficient of a formal power series is nonzero,
then the order of the power series is less than or equal to `degree d`. -/
theorem order_le {d : σ →₀ ℕ} (h : coeff R d f ≠ 0) : f.order ≤ degree d := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Ne ((MvPowerSeries.coeff R d) f) 0
    ⊢ LE.le f.order ↑d.degree
  -/
  rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : Ne ((MvPowerSeries.coeff R d) f) 0
    ⊢ LE.le f.order ↑((Finsupp.weight 1) d)
  -/
  exact weightedOrder_le _ h
  /-
    🎉 no goals
  -/


/-- The `n`th coefficient of a formal power series is `0` if `n` is strictly
smaller than the order of the power series.-/
theorem coeff_of_lt_order {d : σ →₀ ℕ} (h : degree d < f.order) :
    coeff R d f = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑d.degree) f.order
    ⊢ Eq ((MvPowerSeries.coeff R d) f) 0
  -/
  rw [degree_eq_weight_one] at h
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight 1) d)) f.order
    ⊢ Eq ((MvPowerSeries.coeff R d) f) 0
  -/
  exact coeff_eq_zero_of_lt_weightedOrder _ h
  /-
    🎉 no goals
  -/


/-- The order of a formal power series is at least `n` if
the `d`th coefficient is `0` for all `d` such that `degree d < n`.-/
theorem nat_le_order {n : ℕ} (h : ∀ d, degree d < n → coeff R d f = 0) :
    n ≤ f.order := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    n : Nat
    h : ∀ (d : Finsupp σ Nat), LT.lt d.degree n → Eq ((MvPowerSeries.coeff R d) f) 0
    ⊢ LE.le (↑n) f.order
  -/
  simp_rw [degree_eq_weight_one] at h
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    n : Nat
    h : ∀ (d : Finsupp σ Nat), LT.lt ((Finsupp.weight 1) d) n → Eq ((MvPowerSeries …
    ⊢ LE.le (↑n) f.order
  -/
  exact nat_le_weightedOrder _ h
  /-
    🎉 no goals
  -/


/-- The order of a formal power series is at least `n` if
the `d`th coefficient is `0` for all `d` such that `degree d < n`.-/
theorem le_order {n : ℕ∞} (h : ∀ d : σ →₀ ℕ, degree d < n → coeff R d f = 0) :
    n ≤ f.order := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    n : ENat
    h : ∀ (d : Finsupp σ Nat), LT.lt (↑d.degree) n → Eq ((MvPowerSeries.coeff R d) …
    ⊢ LE.le n f.order
  -/
  simp_rw [degree_eq_weight_one] at h
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    n : ENat
    h : ∀ (d : Finsupp σ Nat), LT.lt (↑((Finsupp.weight 1) d)) n → Eq ((MvPowerSer …
    ⊢ LE.le n f.order
  -/
  exact le_weightedOrder _ h
  /-
    🎉 no goals
  -/


/-- The order of a formal power series is exactly `n` some coefficient
of degree `n` is nonzero,
and the `d`th coefficient is `0` for all `d` such that `degree d < n`.-/
theorem order_eq_nat {n : ℕ} :
    f.order = n ↔
      (∃ d, coeff R d f ≠ 0 ∧ degree d = n) ∧ ∀ d, degree d < n → coeff R d f = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    n : Nat
    ⊢ Iff (Eq f.order ↑n) (And (Exists fun d => And (Ne ((MvPowerSeries.coeff R d) …
  -/
  simp_rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    f : MvPowerSeries σ R
    n : Nat
    ⊢ Iff (Eq f.order ↑n) (And (Exists fun d => And (Ne ((MvPowerSeries.coeff R d) …
  -/
  exact weightedOrder_eq_nat _
  /-
    🎉 no goals
  -/


/-- The order of the monomial `a*X^d` is infinite if `a = 0` and `degree d` otherwise.-/
theorem order_monomial {d : σ →₀ ℕ} {a : R} [Decidable (a = 0)] :
    order (monomial R d a) = if a = 0 then (⊤ : ℕ∞) else degree d := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    d : Finsupp σ Nat
    a : R
    inst✝ : Decidable (Eq a 0)
    ⊢ Eq ((MvPowerSeries.monomial R d) a).order (ite (Eq a 0) Top.top ↑d.degree)
  -/
  rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : Semiring R
    d : Finsupp σ Nat
    a : R
    inst✝ : Decidable (Eq a 0)
    ⊢ Eq ((MvPowerSeries.monomial R d) a).order (ite (Eq a 0) Top.top ↑((Finsupp.w …
  -/
  exact weightedOrder_monomial _
  /-
    🎉 no goals
  -/


/-- The order of the monomial `a*X^n` is `n` if `a ≠ 0`.-/
theorem order_monomial_of_ne_zero {d : σ →₀ ℕ} {a : R} (h : a ≠ 0) :
    order (monomial R d a) = degree d := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    d : Finsupp σ Nat
    a : R
    h : Ne a 0
    ⊢ Eq ((MvPowerSeries.monomial R d) a).order ↑d.degree
  -/
  rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    d : Finsupp σ Nat
    a : R
    h : Ne a 0
    ⊢ Eq ((MvPowerSeries.monomial R d) a).order ↑((Finsupp.weight 1) d)
  -/
  exact weightedOrder_monomial_of_ne_zero _ h
  /-
    🎉 no goals
  -/


/-- The order of the sum of two formal power series
 is at least the minimum of their orders.-/
theorem min_order_le_add : min f.order g.order ≤ (f + g).order :=
  min_weightedOrder_le_add _


/-- The order of the sum of two formal power series
 is the minimum of their orders if their orders differ.-/
theorem order_add_of_order_ne (h : f.order ≠ g.order) :
    order (f + g) = order f ⊓ order g :=
  weightedOrder_add_of_weightedOrder_ne _ h


/-- The order of the product of two formal power series
 is at least the sum of their orders.-/
theorem le_order_mul : f.order + g.order ≤ order (f * g) :=
  le_weightedOrder_mul _


alias order_mul_ge := le_order_mul


theorem coeff_mul_left_one_sub_of_lt_order (d : σ →₀ ℕ) (h : degree d < g.order) :
    coeff R d (f * (1 - g)) = coeff R d f := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑d.degree) g.order
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul f (HSub.hSub 1 g))) ((MvPowerSeries …
  -/
  rw [degree_eq_weight_one] at h
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight 1) d)) g.order
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul f (HSub.hSub 1 g))) ((MvPowerSeries …
  -/
  exact coeff_mul_left_one_sub_of_lt_weightedOrder _ h
  /-
    🎉 no goals
  -/


theorem coeff_mul_right_one_sub_of_lt_order (d : σ →₀ ℕ) (h : degree d < g.order) :
    coeff R d ((1 - g) * f) = coeff R d f := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑d.degree) g.order
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul (HSub.hSub 1 g) f)) ((MvPowerSeries …
  -/
  rw [degree_eq_weight_one] at h
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : Ring R
    f g : MvPowerSeries σ R
    d : Finsupp σ Nat
    h : LT.lt (↑((Finsupp.weight 1) d)) g.order
    ⊢ Eq ((MvPowerSeries.coeff R d) (HMul.hMul (HSub.hSub 1 g) f)) ((MvPowerSeries …
  -/
  exact coeff_mul_right_one_sub_of_lt_weightedOrder _ h
  /-
    🎉 no goals
  -/


theorem coeff_mul_prod_one_sub_of_lt_order {R ι : Type*} [CommRing R] (d : σ →₀ ℕ) (s : Finset ι)
    (f : MvPowerSeries σ R) (g : ι → MvPowerSeries σ R) :
    (∀ i ∈ s, degree d < order (g i)) → coeff R d (f * ∏ i in s, (1 - g i)) = coeff R d f := by
  /-
    σ : Type u_1
    R : Type u_4
    ι : Type u_5
    inst✝ : CommRing R
    d : Finsupp σ Nat
    s : Finset ι
    f : MvPowerSeries σ R
    g : ι → MvPowerSeries σ R
    ⊢ (∀ (i : ι), Membership.mem s i → LT.lt (↑d.degree) (g i).order) → Eq ((MvPow …
  -/
  rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_4
    ι : Type u_5
    inst✝ : CommRing R
    d : Finsupp σ Nat
    s : Finset ι
    f : MvPowerSeries σ R
    g : ι → MvPowerSeries σ R
    ⊢ (∀ (i : ι), Membership.mem s i → LT.lt (↑((Finsupp.weight 1) d)) (g i).order …
  -/
  exact coeff_mul_prod_one_sub_of_lt_weightedOrder _ d s f g
  /-
    🎉 no goals
  -/


/-- The weighted homogeneous components of an `MvPowerSeries f`. -/
def weightedHomogeneousComponent (p : ℕ) : MvPowerSeries σ R →ₗ[R] MvPowerSeries σ R
    where
  toFun f d := if weight w d = p then coeff R d f else 0
  map_add' f g := by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      p : Nat
      f g : MvPowerSeries σ R
      ⊢ Eq ((fun f d => ite (Eq ((Finsupp.weight w) d) p) ((MvPowerSeries.coeff R d) …
    -/
    ext d
    /-
      case h
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      p : Nat
      f g : MvPowerSeries σ R
      d : Finsupp σ Nat
      ⊢ Eq ((MvPowerSeries.coeff R d) ((fun f d => ite (Eq ((Finsupp.weight w) d) p) …
    -/
    simp only [map_add, coeff_apply, Pi.add_apply]
    /-
      case h
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      p : Nat
      f g : MvPowerSeries σ R
      d : Finsupp σ Nat
      ⊢ Eq (ite (Eq ((Finsupp.weight w) d) p) (HAdd.hAdd (f d) (g d)) 0) (HAdd.hAdd  …
    -/
    split_ifs with h
      /-
        case pos
        σ : Type u_1
        R : Type u_2
        inst✝ : Semiring R
        w : σ → Nat
        p : Nat
        f g : MvPowerSeries σ R
        d : Finsupp σ Nat
        h : Eq ((Finsupp.weight w) d) p
        ⊢ Eq (HAdd.hAdd (f d) (g d)) (HAdd.hAdd (f d) (g d))
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        σ : Type u_1
        R : Type u_2
        inst✝ : Semiring R
        w : σ → Nat
        p : Nat
        f g : MvPowerSeries σ R
        d : Finsupp σ Nat
        h : Not (Eq ((Finsupp.weight w) d) p)
        ⊢ Eq 0 (HAdd.hAdd 0 0)
      -/
    · rw [add_zero]
      /-
        🎉 no goals
      -/
  map_smul' a f := by
    /-
      σ : Type u_1
      R : Type u_2
      inst✝ : Semiring R
      w : σ → Nat
      p : Nat
      a : R
      f : MvPowerSeries σ R
      ⊢ Eq ({ toFun := fun f d => ite (Eq ((Finsupp.weight w) d) p) ((MvPowerSeries. …
    -/
    ext d
    simp only [id_eq, eq_mpr_eq_cast, AddHom.toFun_eq_coe, AddHom.coe_mk, map_smul,
      smul_eq_mul, RingHom.id_apply, coeff_apply, mul_ite, MulZeroClass.mul_zero]


theorem coeff_weightedHomogeneousComponent (p : ℕ) (d : σ →₀ ℕ) (f : MvPowerSeries σ R) :
    coeff R d (weightedHomogeneousComponent w p f) =
      if weight w d = p then coeff R d f else 0 :=
  rfl


/-- The homogeneous components of an `MvPowerSeries` -/
def homogeneousComponent (p : ℕ) : MvPowerSeries σ R →ₗ[R] MvPowerSeries σ R :=
  weightedHomogeneousComponent 1 p


theorem coeff_homogeneousComponent (p : ℕ) (d : σ →₀ ℕ) (f : MvPowerSeries σ R) :
    coeff R d (homogeneousComponent p f) =
      if degree d = p then coeff R d f else 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    p : Nat
    d : Finsupp σ Nat
    f : MvPowerSeries σ R
    ⊢ Eq ((MvPowerSeries.coeff R d) ((MvPowerSeries.homogeneousComponent p) f)) (i …
  -/
  rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝ : Semiring R
    p : Nat
    d : Finsupp σ Nat
    f : MvPowerSeries σ R
    ⊢ Eq ((MvPowerSeries.coeff R d) ((MvPowerSeries.homogeneousComponent p) f)) (i …
  -/
  exact coeff_weightedHomogeneousComponent 1 p d f
  /-
    🎉 no goals
  -/


