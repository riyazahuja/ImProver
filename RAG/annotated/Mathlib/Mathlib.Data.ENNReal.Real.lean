theorem toReal_add (ha : a ≠ ∞) (hb : b ≠ ∞) : (a + b).toReal = a.toReal + b.toReal := by
  /-
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b Top.top
    ⊢ Eq (HAdd.hAdd a b).toReal (HAdd.hAdd a.toReal b.toReal)
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro
    b : ENNReal
    hb : Ne b Top.top
    a : NNReal
    ⊢ Eq (HAdd.hAdd (↑a) b).toReal (HAdd.hAdd (↑a).toReal b.toReal)
  -/
  lift b to ℝ≥0 using hb
  /-
    case intro.intro
    a b : NNReal
    ⊢ Eq (HAdd.hAdd ↑a ↑b).toReal (HAdd.hAdd (↑a).toReal (↑b).toReal)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toReal_sub_of_le {a b : ℝ≥0∞} (h : b ≤ a) (ha : a ≠ ∞) :
    (a - b).toReal = a.toReal - b.toReal := by
  /-
    a b : ENNReal
    h : LE.le b a
    ha : Ne a Top.top
    ⊢ Eq (HSub.hSub a b).toReal (HSub.hSub a.toReal b.toReal)
  -/
  lift b to ℝ≥0 using ne_top_of_le_ne_top ha h
  /-
    case intro
    a : ENNReal
    ha : Ne a Top.top
    b : NNReal
    h : LE.le (↑b) a
    ⊢ Eq (HSub.hSub a ↑b).toReal (HSub.hSub a.toReal (↑b).toReal)
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro.intro
    b a : NNReal
    h : LE.le ↑b ↑a
    ⊢ Eq (HSub.hSub ↑a ↑b).toReal (HSub.hSub (↑a).toReal (↑b).toReal)
  -/
  simp only [← ENNReal.coe_sub, ENNReal.coe_toReal, NNReal.coe_sub (ENNReal.coe_le_coe.mp h)]
  /-
    🎉 no goals
  -/


theorem le_toReal_sub {a b : ℝ≥0∞} (hb : b ≠ ∞) : a.toReal - b.toReal ≤ (a - b).toReal := by
  /-
    a b : ENNReal
    hb : Ne b Top.top
    ⊢ LE.le (HSub.hSub a.toReal b.toReal) (HSub.hSub a b).toReal
  -/
  lift b to ℝ≥0 using hb
  /-
    case intro
    a : ENNReal
    b : NNReal
    ⊢ LE.le (HSub.hSub a.toReal (↑b).toReal) (HSub.hSub a ↑b).toReal
  -/
  induction a
    /-
      case intro.top
      b : NNReal
      ⊢ LE.le (HSub.hSub Top.top.toReal (↑b).toReal) (HSub.hSub Top.top ↑b).toReal
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.coe
      b x✝ : NNReal
      ⊢ LE.le (HSub.hSub (↑x✝).toReal (↑b).toReal) (HSub.hSub ↑x✝ ↑b).toReal
    -/
  · simp only [← coe_sub, NNReal.sub_def, Real.coe_toNNReal', coe_toReal]
    /-
      case intro.coe
      b x✝ : NNReal
      ⊢ LE.le (HSub.hSub ↑x✝ ↑b) (Max.max (HSub.hSub ↑x✝ ↑b) 0)
    -/
    exact le_max_left _ _
    /-
      🎉 no goals
    -/


theorem toReal_add_le : (a + b).toReal ≤ a.toReal + b.toReal :=
                        /-
                          a b : ENNReal
                          ha : Eq a Top.top
                          ⊢ LE.le (HAdd.hAdd a b).toReal (HAdd.hAdd a.toReal b.toReal)
                        -/
  if ha : a = ∞ then by simp only [ha, top_add, top_toReal, zero_add, toReal_nonneg]
                        /-
                          🎉 no goals
                        -/
  else
                          /-
                            a b : ENNReal
                            ha : Not (Eq a Top.top)
                            hb : Eq b Top.top
                            ⊢ LE.le (HAdd.hAdd a b).toReal (HAdd.hAdd a.toReal b.toReal)
                          -/
    if hb : b = ∞ then by simp only [hb, add_top, top_toReal, add_zero, toReal_nonneg]
                          /-
                            🎉 no goals
                          -/
    else le_of_eq (toReal_add ha hb)


theorem ofReal_add {p q : ℝ} (hp : 0 ≤ p) (hq : 0 ≤ q) :
    ENNReal.ofReal (p + q) = ENNReal.ofReal p + ENNReal.ofReal q := by
  rw [ENNReal.ofReal, ENNReal.ofReal, ENNReal.ofReal, ← coe_add, coe_inj,
    Real.toNNReal_add hp hq]


theorem ofReal_add_le {p q : ℝ} : ENNReal.ofReal (p + q) ≤ ENNReal.ofReal p + ENNReal.ofReal q :=
  coe_le_coe.2 Real.toNNReal_add_le


@[simp]
theorem toReal_le_toReal (ha : a ≠ ∞) (hb : b ≠ ∞) : a.toReal ≤ b.toReal ↔ a ≤ b := by
  /-
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b Top.top
    ⊢ Iff (LE.le a.toReal b.toReal) (LE.le a b)
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro
    b : ENNReal
    hb : Ne b Top.top
    a : NNReal
    ⊢ Iff (LE.le (↑a).toReal b.toReal) (LE.le (↑a) b)
  -/
  lift b to ℝ≥0 using hb
  /-
    case intro.intro
    a b : NNReal
    ⊢ Iff (LE.le (↑a).toReal (↑b).toReal) (LE.le ↑a ↑b)
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[gcongr]
theorem toReal_mono (hb : b ≠ ∞) (h : a ≤ b) : a.toReal ≤ b.toReal :=
  (toReal_le_toReal (ne_top_of_le_ne_top hb h) hb).2 h


theorem toReal_mono' (h : a ≤ b) (ht : b = ∞ → a = ∞) : a.toReal ≤ b.toReal := by
  /-
    a b : ENNReal
    h : LE.le a b
    ht : Eq b Top.top → Eq a Top.top
    ⊢ LE.le a.toReal b.toReal
  -/
  rcases eq_or_ne a ∞ with rfl | ha
    /-
      case inl
      b : ENNReal
      h : LE.le Top.top b
      ht : Eq b Top.top → Eq Top.top Top.top
      ⊢ LE.le Top.top.toReal b.toReal
    -/
  · exact toReal_nonneg
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : ENNReal
      h : LE.le a b
      ht : Eq b Top.top → Eq a Top.top
      ha : Ne a Top.top
      ⊢ LE.le a.toReal b.toReal
    -/
  · exact toReal_mono (mt ht ha) h
    /-
      🎉 no goals
    -/


@[simp]
theorem toReal_lt_toReal (ha : a ≠ ∞) (hb : b ≠ ∞) : a.toReal < b.toReal ↔ a < b := by
  /-
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b Top.top
    ⊢ Iff (LT.lt a.toReal b.toReal) (LT.lt a b)
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro
    b : ENNReal
    hb : Ne b Top.top
    a : NNReal
    ⊢ Iff (LT.lt (↑a).toReal b.toReal) (LT.lt (↑a) b)
  -/
  lift b to ℝ≥0 using hb
  /-
    case intro.intro
    a b : NNReal
    ⊢ Iff (LT.lt (↑a).toReal (↑b).toReal) (LT.lt ↑a ↑b)
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[gcongr]
theorem toReal_strict_mono (hb : b ≠ ∞) (h : a < b) : a.toReal < b.toReal :=
  (toReal_lt_toReal h.ne_top hb).2 h


@[gcongr]
theorem toNNReal_mono (hb : b ≠ ∞) (h : a ≤ b) : a.toNNReal ≤ b.toNNReal :=
  toReal_mono hb h


theorem le_toNNReal_of_coe_le (h : p ≤ a) (ha : a ≠ ∞) : p ≤ a.toNNReal :=
  @toNNReal_coe p ▸ toNNReal_mono ha h


/-- If `a ≤ b + c` and `a = ∞` whenever `b = ∞` or `c = ∞`, then
`ENNReal.toReal a ≤ ENNReal.toReal b + ENNReal.toReal c`. This lemma is useful to transfer
triangle-like inequalities from `ENNReal`s to `Real`s. -/
theorem toReal_le_add' (hle : a ≤ b + c) (hb : b = ∞ → a = ∞) (hc : c = ∞ → a = ∞) :
    a.toReal ≤ b.toReal + c.toReal := by
  /-
    a b c : ENNReal
    hle : LE.le a (HAdd.hAdd b c)
    hb : Eq b Top.top → Eq a Top.top
    hc : Eq c Top.top → Eq a Top.top
    ⊢ LE.le a.toReal (HAdd.hAdd b.toReal c.toReal)
  -/
  refine le_trans (toReal_mono' hle ?_) toReal_add_le
  /-
    a b c : ENNReal
    hle : LE.le a (HAdd.hAdd b c)
    hb : Eq b Top.top → Eq a Top.top
    hc : Eq c Top.top → Eq a Top.top
    ⊢ Eq (HAdd.hAdd b c) Top.top → Eq a Top.top
  -/
  simpa only [add_eq_top, or_imp] using And.intro hb hc
  /-
    🎉 no goals
  -/


/-- If `a ≤ b + c`, `b ≠ ∞`, and `c ≠ ∞`, then
`ENNReal.toReal a ≤ ENNReal.toReal b + ENNReal.toReal c`. This lemma is useful to transfer
triangle-like inequalities from `ENNReal`s to `Real`s. -/
theorem toReal_le_add (hle : a ≤ b + c) (hb : b ≠ ∞) (hc : c ≠ ∞) :
    a.toReal ≤ b.toReal + c.toReal :=
  toReal_le_add' hle (flip absurd hb) (flip absurd hc)


@[simp]
theorem toNNReal_le_toNNReal (ha : a ≠ ∞) (hb : b ≠ ∞) : a.toNNReal ≤ b.toNNReal ↔ a ≤ b :=
               /-
                 a b : ENNReal
                 ha : Ne a Top.top
                 hb : Ne b Top.top
                 h : LE.le a.toNNReal b.toNNReal
                 ⊢ LE.le a b
               -/
  ⟨fun h => by rwa [← coe_toNNReal ha, ← coe_toNNReal hb, coe_le_coe], toNNReal_mono hb⟩
               /-
                 🎉 no goals
               -/


@[gcongr]
theorem toNNReal_strict_mono (hb : b ≠ ∞) (h : a < b) : a.toNNReal < b.toNNReal := by
  /-
    a b : ENNReal
    hb : Ne b Top.top
    h : LT.lt a b
    ⊢ LT.lt a.toNNReal b.toNNReal
  -/
  simpa [← ENNReal.coe_lt_coe, hb, h.ne_top]
  /-
    🎉 no goals
  -/


@[simp]
theorem toNNReal_lt_toNNReal (ha : a ≠ ∞) (hb : b ≠ ∞) : a.toNNReal < b.toNNReal ↔ a < b :=
               /-
                 a b : ENNReal
                 ha : Ne a Top.top
                 hb : Ne b Top.top
                 h : LT.lt a.toNNReal b.toNNReal
                 ⊢ LT.lt a b
               -/
  ⟨fun h => by rwa [← coe_toNNReal ha, ← coe_toNNReal hb, coe_lt_coe], toNNReal_strict_mono hb⟩
               /-
                 🎉 no goals
               -/


theorem toNNReal_lt_of_lt_coe (h : a < p) : a.toNNReal < p :=
  @toNNReal_coe p ▸ toNNReal_strict_mono coe_ne_top h


theorem toReal_max (hr : a ≠ ∞) (hp : b ≠ ∞) :
    ENNReal.toReal (max a b) = max (ENNReal.toReal a) (ENNReal.toReal b) :=
  (le_total a b).elim
                 /-
                   a b : ENNReal
                   hr : Ne a Top.top
                   hp : Ne b Top.top
                   h : LE.le a b
                   ⊢ Eq (Max.max a b).toReal (Max.max a.toReal b.toReal)
                 -/
    (fun h => by simp only [h, ENNReal.toReal_mono hp h, max_eq_right]) fun h => by
                 /-
                   🎉 no goals
                 -/
    /-
      a b : ENNReal
      hr : Ne a Top.top
      hp : Ne b Top.top
      h : LE.le b a
      ⊢ Eq (Max.max a b).toReal (Max.max a.toReal b.toReal)
    -/
    simp only [h, ENNReal.toReal_mono hr h, max_eq_left]
    /-
      🎉 no goals
    -/


theorem toReal_min {a b : ℝ≥0∞} (hr : a ≠ ∞) (hp : b ≠ ∞) :
    ENNReal.toReal (min a b) = min (ENNReal.toReal a) (ENNReal.toReal b) :=
                                   /-
                                     a b : ENNReal
                                     hr : Ne a Top.top
                                     hp : Ne b Top.top
                                     h : LE.le a b
                                     ⊢ Eq (Min.min a b).toReal (Min.min a.toReal b.toReal)
                                   -/
  (le_total a b).elim (fun h => by simp only [h, ENNReal.toReal_mono hp h, min_eq_left])
                                   /-
                                     🎉 no goals
                                   -/
                /-
                  a b : ENNReal
                  hr : Ne a Top.top
                  hp : Ne b Top.top
                  h : LE.le b a
                  ⊢ Eq (Min.min a b).toReal (Min.min a.toReal b.toReal)
                -/
    fun h => by simp only [h, ENNReal.toReal_mono hr h, min_eq_right]
                /-
                  🎉 no goals
                -/


theorem toReal_sup {a b : ℝ≥0∞} : a ≠ ∞ → b ≠ ∞ → (a ⊔ b).toReal = a.toReal ⊔ b.toReal :=
  toReal_max


theorem toReal_inf {a b : ℝ≥0∞} : a ≠ ∞ → b ≠ ∞ → (a ⊓ b).toReal = a.toReal ⊓ b.toReal :=
  toReal_min


theorem toNNReal_pos_iff : 0 < a.toNNReal ↔ 0 < a ∧ a < ∞ := by
  /-
    a : ENNReal
    ⊢ Iff (LT.lt 0 a.toNNReal) (And (LT.lt 0 a) (LT.lt a Top.top))
  -/
                  /-
                    🎉 no goals
                  -/
  induction a <;> simp
                  /-
                    🎉 no goals
                  -/


theorem toNNReal_pos {a : ℝ≥0∞} (ha₀ : a ≠ 0) (ha_top : a ≠ ∞) : 0 < a.toNNReal :=
  toNNReal_pos_iff.mpr ⟨bot_lt_iff_ne_bot.mpr ha₀, lt_top_iff_ne_top.mpr ha_top⟩


theorem toReal_pos_iff : 0 < a.toReal ↔ 0 < a ∧ a < ∞ :=
  NNReal.coe_pos.trans toNNReal_pos_iff


theorem toReal_pos {a : ℝ≥0∞} (ha₀ : a ≠ 0) (ha_top : a ≠ ∞) : 0 < a.toReal :=
  toReal_pos_iff.mpr ⟨bot_lt_iff_ne_bot.mpr ha₀, lt_top_iff_ne_top.mpr ha_top⟩


@[gcongr, bound]
theorem ofReal_le_ofReal {p q : ℝ} (h : p ≤ q) : ENNReal.ofReal p ≤ ENNReal.ofReal q := by
  /-
    p q : Real
    h : LE.le p q
    ⊢ LE.le (ENNReal.ofReal p) (ENNReal.ofReal q)
  -/
  simp [ENNReal.ofReal, Real.toNNReal_le_toNNReal h]
  /-
    🎉 no goals
  -/


theorem ofReal_le_of_le_toReal {a : ℝ} {b : ℝ≥0∞} (h : a ≤ ENNReal.toReal b) :
    ENNReal.ofReal a ≤ b :=
  (ofReal_le_ofReal h).trans ofReal_toReal_le


@[simp]
theorem ofReal_le_ofReal_iff {p q : ℝ} (h : 0 ≤ q) :
    ENNReal.ofReal p ≤ ENNReal.ofReal q ↔ p ≤ q := by
  /-
    p q : Real
    h : LE.le 0 q
    ⊢ Iff (LE.le (ENNReal.ofReal p) (ENNReal.ofReal q)) (LE.le p q)
  -/
  rw [ENNReal.ofReal, ENNReal.ofReal, coe_le_coe, Real.toNNReal_le_toNNReal_iff h]
  /-
    🎉 no goals
  -/


lemma ofReal_le_ofReal_iff' {p q : ℝ} : ENNReal.ofReal p ≤ .ofReal q ↔ p ≤ q ∨ p ≤ 0 :=
  coe_le_coe.trans Real.toNNReal_le_toNNReal_iff'


lemma ofReal_lt_ofReal_iff' {p q : ℝ} : ENNReal.ofReal p < .ofReal q ↔ p < q ∧ 0 < q :=
  coe_lt_coe.trans Real.toNNReal_lt_toNNReal_iff'


@[simp]
theorem ofReal_eq_ofReal_iff {p q : ℝ} (hp : 0 ≤ p) (hq : 0 ≤ q) :
    ENNReal.ofReal p = ENNReal.ofReal q ↔ p = q := by
  /-
    p q : Real
    hp : LE.le 0 p
    hq : LE.le 0 q
    ⊢ Iff (Eq (ENNReal.ofReal p) (ENNReal.ofReal q)) (Eq p q)
  -/
  rw [ENNReal.ofReal, ENNReal.ofReal, coe_inj, Real.toNNReal_eq_toNNReal_iff hp hq]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofReal_lt_ofReal_iff {p q : ℝ} (h : 0 < q) :
    ENNReal.ofReal p < ENNReal.ofReal q ↔ p < q := by
  /-
    p q : Real
    h : LT.lt 0 q
    ⊢ Iff (LT.lt (ENNReal.ofReal p) (ENNReal.ofReal q)) (LT.lt p q)
  -/
  rw [ENNReal.ofReal, ENNReal.ofReal, coe_lt_coe, Real.toNNReal_lt_toNNReal_iff h]
  /-
    🎉 no goals
  -/


theorem ofReal_lt_ofReal_iff_of_nonneg {p q : ℝ} (hp : 0 ≤ p) :
    ENNReal.ofReal p < ENNReal.ofReal q ↔ p < q := by
  /-
    p q : Real
    hp : LE.le 0 p
    ⊢ Iff (LT.lt (ENNReal.ofReal p) (ENNReal.ofReal q)) (LT.lt p q)
  -/
  rw [ENNReal.ofReal, ENNReal.ofReal, coe_lt_coe, Real.toNNReal_lt_toNNReal_iff_of_nonneg hp]
  /-
    🎉 no goals
  -/


@[simp]
                                                                /-
                                                                  p : Real
                                                                  ⊢ Iff (LT.lt 0 (ENNReal.ofReal p)) (LT.lt 0 p)
                                                                -/
theorem ofReal_pos {p : ℝ} : 0 < ENNReal.ofReal p ↔ 0 < p := by simp [ENNReal.ofReal]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[bound] private alias ⟨_, Bound.ofReal_pos_of_pos⟩ := ofReal_pos


@[simp]
                                                                    /-
                                                                      p : Real
                                                                      ⊢ Iff (Eq (ENNReal.ofReal p) 0) (LE.le p 0)
                                                                    -/
theorem ofReal_eq_zero {p : ℝ} : ENNReal.ofReal p = 0 ↔ p ≤ 0 := by simp [ENNReal.ofReal]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem zero_eq_ofReal {p : ℝ} : 0 = ENNReal.ofReal p ↔ p ≤ 0 :=
  eq_comm.trans ofReal_eq_zero


alias ⟨_, ofReal_of_nonpos⟩ := ofReal_eq_zero


@[simp]
lemma ofReal_lt_natCast {p : ℝ} {n : ℕ} (hn : n ≠ 0) : ENNReal.ofReal p < n ↔ p < n := by
  /-
    p : Real
    n : Nat
    hn : Ne n 0
    ⊢ Iff (LT.lt (ENNReal.ofReal p) ↑n) (LT.lt p ↑n)
  -/
  exact mod_cast ofReal_lt_ofReal_iff (Nat.cast_pos.2 hn.bot_lt)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias ofReal_lt_nat_cast := ofReal_lt_natCast


@[simp]
lemma ofReal_lt_one {p : ℝ} : ENNReal.ofReal p < 1 ↔ p < 1 := by
  /-
    p : Real
    ⊢ Iff (LT.lt (ENNReal.ofReal p) 1) (LT.lt p 1)
  -/
  exact mod_cast ofReal_lt_natCast one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma ofReal_lt_ofNat {p : ℝ} {n : ℕ} [n.AtLeastTwo] :
    ENNReal.ofReal p < no_index (OfNat.ofNat n) ↔ p < OfNat.ofNat n :=
  ofReal_lt_natCast (NeZero.ne n)


@[simp]
lemma natCast_le_ofReal {n : ℕ} {p : ℝ} (hn : n ≠ 0) : n ≤ ENNReal.ofReal p ↔ n ≤ p := by
  /-
    n : Nat
    p : Real
    hn : Ne n 0
    ⊢ Iff (LE.le (↑n) (ENNReal.ofReal p)) (LE.le (↑n) p)
  -/
  simp only [← not_lt, ofReal_lt_natCast hn]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_le_ofReal := natCast_le_ofReal


@[simp]
lemma one_le_ofReal {p : ℝ} : 1 ≤ ENNReal.ofReal p ↔ 1 ≤ p := by
  /-
    p : Real
    ⊢ Iff (LE.le 1 (ENNReal.ofReal p)) (LE.le 1 p)
  -/
  exact mod_cast natCast_le_ofReal one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
lemma ofNat_le_ofReal {n : ℕ} [n.AtLeastTwo] {p : ℝ} :
    no_index (OfNat.ofNat n) ≤ ENNReal.ofReal p ↔ OfNat.ofNat n ≤ p :=
  natCast_le_ofReal (NeZero.ne n)


@[simp, norm_cast]
lemma ofReal_le_natCast {r : ℝ} {n : ℕ} : ENNReal.ofReal r ≤ n ↔ r ≤ n :=
  coe_le_coe.trans Real.toNNReal_le_natCast


@[deprecated (since := "2024-04-17")]
alias ofReal_le_nat_cast := ofReal_le_natCast


@[simp]
lemma ofReal_le_one {r : ℝ} : ENNReal.ofReal r ≤ 1 ↔ r ≤ 1 :=
  coe_le_coe.trans Real.toNNReal_le_one


@[simp]
lemma ofReal_le_ofNat {r : ℝ} {n : ℕ} [n.AtLeastTwo] :
    ENNReal.ofReal r ≤ no_index (OfNat.ofNat n) ↔ r ≤ OfNat.ofNat n :=
  ofReal_le_natCast


@[simp]
lemma natCast_lt_ofReal {n : ℕ} {r : ℝ} : n < ENNReal.ofReal r ↔ n < r :=
  coe_lt_coe.trans Real.natCast_lt_toNNReal


@[deprecated (since := "2024-04-17")]
alias nat_cast_lt_ofReal := natCast_lt_ofReal


@[simp]
lemma one_lt_ofReal {r : ℝ} : 1 < ENNReal.ofReal r ↔ 1 < r := coe_lt_coe.trans Real.one_lt_toNNReal


@[simp]
lemma ofNat_lt_ofReal {n : ℕ} [n.AtLeastTwo] {r : ℝ} :
    no_index (OfNat.ofNat n) < ENNReal.ofReal r ↔ OfNat.ofNat n < r :=
  natCast_lt_ofReal


@[simp]
lemma ofReal_eq_natCast {r : ℝ} {n : ℕ} (h : n ≠ 0) : ENNReal.ofReal r = n ↔ r = n :=
  ENNReal.coe_inj.trans <| Real.toNNReal_eq_natCast h


@[deprecated (since := "2024-04-17")]
alias ofReal_eq_nat_cast := ofReal_eq_natCast


@[simp]
lemma ofReal_eq_one {r : ℝ} : ENNReal.ofReal r = 1 ↔ r = 1 :=
  ENNReal.coe_inj.trans Real.toNNReal_eq_one


@[simp]
lemma ofReal_eq_ofNat {r : ℝ} {n : ℕ} [n.AtLeastTwo] :
    ENNReal.ofReal r = no_index (OfNat.ofNat n) ↔ r = OfNat.ofNat n :=
  ofReal_eq_natCast (NeZero.ne n)


theorem ofReal_sub (p : ℝ) {q : ℝ} (hq : 0 ≤ q) :
    ENNReal.ofReal (p - q) = ENNReal.ofReal p - ENNReal.ofReal q := by
  /-
    p q : Real
    hq : LE.le 0 q
    ⊢ Eq (ENNReal.ofReal (HSub.hSub p q)) (HSub.hSub (ENNReal.ofReal p) (ENNReal.o …
  -/
  obtain h | h := le_total p q
    /-
      case inl
      p q : Real
      hq : LE.le 0 q
      h : LE.le p q
      ⊢ Eq (ENNReal.ofReal (HSub.hSub p q)) (HSub.hSub (ENNReal.ofReal p) (ENNReal.o …
    -/
  · rw [ofReal_of_nonpos (sub_nonpos_of_le h), tsub_eq_zero_of_le (ofReal_le_ofReal h)]
    /-
      🎉 no goals
    -/
  /-
    case inr
    p q : Real
    hq : LE.le 0 q
    h : LE.le q p
    ⊢ Eq (ENNReal.ofReal (HSub.hSub p q)) (HSub.hSub (ENNReal.ofReal p) (ENNReal.o …
  -/
  refine ENNReal.eq_sub_of_add_eq ofReal_ne_top ?_
  /-
    case inr
    p q : Real
    hq : LE.le 0 q
    h : LE.le q p
    ⊢ Eq (HAdd.hAdd (ENNReal.ofReal (HSub.hSub p q)) (ENNReal.ofReal q)) (ENNReal. …
  -/
  rw [← ofReal_add (sub_nonneg_of_le h) hq, sub_add_cancel]
  /-
    🎉 no goals
  -/


theorem ofReal_le_iff_le_toReal {a : ℝ} {b : ℝ≥0∞} (hb : b ≠ ∞) :
    ENNReal.ofReal a ≤ b ↔ a ≤ ENNReal.toReal b := by
  /-
    a : Real
    b : ENNReal
    hb : Ne b Top.top
    ⊢ Iff (LE.le (ENNReal.ofReal a) b) (LE.le a b.toReal)
  -/
  lift b to ℝ≥0 using hb
  /-
    case intro
    a : Real
    b : NNReal
    ⊢ Iff (LE.le (ENNReal.ofReal a) ↑b) (LE.le a (↑b).toReal)
  -/
  simpa [ENNReal.ofReal, ENNReal.toReal] using Real.toNNReal_le_iff_le_coe
  /-
    🎉 no goals
  -/


theorem ofReal_lt_iff_lt_toReal {a : ℝ} {b : ℝ≥0∞} (ha : 0 ≤ a) (hb : b ≠ ∞) :
    ENNReal.ofReal a < b ↔ a < ENNReal.toReal b := by
  /-
    a : Real
    b : ENNReal
    ha : LE.le 0 a
    hb : Ne b Top.top
    ⊢ Iff (LT.lt (ENNReal.ofReal a) b) (LT.lt a b.toReal)
  -/
  lift b to ℝ≥0 using hb
  /-
    case intro
    a : Real
    ha : LE.le 0 a
    b : NNReal
    ⊢ Iff (LT.lt (ENNReal.ofReal a) ↑b) (LT.lt a (↑b).toReal)
  -/
  simpa [ENNReal.ofReal, ENNReal.toReal] using Real.toNNReal_lt_iff_lt_coe ha
  /-
    🎉 no goals
  -/


theorem ofReal_lt_coe_iff {a : ℝ} {b : ℝ≥0} (ha : 0 ≤ a) : ENNReal.ofReal a < b ↔ a < b :=
                                                      /-
                                                        a : Real
                                                        b : NNReal
                                                        ha : LE.le 0 a
                                                        ⊢ Iff (LT.lt a (↑b).toReal) (LT.lt a ↑b)
                                                      -/
  (ofReal_lt_iff_lt_toReal ha coe_ne_top).trans <| by rw [coe_toReal]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem le_ofReal_iff_toReal_le {a : ℝ≥0∞} {b : ℝ} (ha : a ≠ ∞) (hb : 0 ≤ b) :
    a ≤ ENNReal.ofReal b ↔ ENNReal.toReal a ≤ b := by
  /-
    a : ENNReal
    b : Real
    ha : Ne a Top.top
    hb : LE.le 0 b
    ⊢ Iff (LE.le a (ENNReal.ofReal b)) (LE.le a.toReal b)
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro
    b : Real
    hb : LE.le 0 b
    a : NNReal
    ⊢ Iff (LE.le (↑a) (ENNReal.ofReal b)) (LE.le (↑a).toReal b)
  -/
  simpa [ENNReal.ofReal, ENNReal.toReal] using Real.le_toNNReal_iff_coe_le hb
  /-
    🎉 no goals
  -/


theorem toReal_le_of_le_ofReal {a : ℝ≥0∞} {b : ℝ} (hb : 0 ≤ b) (h : a ≤ ENNReal.ofReal b) :
    ENNReal.toReal a ≤ b :=
  have ha : a ≠ ∞ := ne_top_of_le_ne_top ofReal_ne_top h
  (le_ofReal_iff_toReal_le ha hb).1 h


theorem lt_ofReal_iff_toReal_lt {a : ℝ≥0∞} {b : ℝ} (ha : a ≠ ∞) :
    a < ENNReal.ofReal b ↔ ENNReal.toReal a < b := by
  /-
    a : ENNReal
    b : Real
    ha : Ne a Top.top
    ⊢ Iff (LT.lt a (ENNReal.ofReal b)) (LT.lt a.toReal b)
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro
    b : Real
    a : NNReal
    ⊢ Iff (LT.lt (↑a) (ENNReal.ofReal b)) (LT.lt (↑a).toReal b)
  -/
  simpa [ENNReal.ofReal, ENNReal.toReal] using Real.lt_toNNReal_iff_coe_lt
  /-
    🎉 no goals
  -/


theorem toReal_lt_of_lt_ofReal {b : ℝ} (h : a < ENNReal.ofReal b) : ENNReal.toReal a < b :=
  (lt_ofReal_iff_toReal_lt h.ne_top).1 h


theorem ofReal_mul {p q : ℝ} (hp : 0 ≤ p) :
    ENNReal.ofReal (p * q) = ENNReal.ofReal p * ENNReal.ofReal q := by
  /-
    p q : Real
    hp : LE.le 0 p
    ⊢ Eq (ENNReal.ofReal (HMul.hMul p q)) (HMul.hMul (ENNReal.ofReal p) (ENNReal.o …
  -/
  simp only [ENNReal.ofReal, ← coe_mul, Real.toNNReal_mul hp]
  /-
    🎉 no goals
  -/


theorem ofReal_mul' {p q : ℝ} (hq : 0 ≤ q) :
    ENNReal.ofReal (p * q) = ENNReal.ofReal p * ENNReal.ofReal q := by
  /-
    p q : Real
    hq : LE.le 0 q
    ⊢ Eq (ENNReal.ofReal (HMul.hMul p q)) (HMul.hMul (ENNReal.ofReal p) (ENNReal.o …
  -/
  rw [mul_comm, ofReal_mul hq, mul_comm]
  /-
    🎉 no goals
  -/


theorem ofReal_pow {p : ℝ} (hp : 0 ≤ p) (n : ℕ) :
    ENNReal.ofReal (p ^ n) = ENNReal.ofReal p ^ n := by
  /-
    p : Real
    hp : LE.le 0 p
    n : Nat
    ⊢ Eq (ENNReal.ofReal (HPow.hPow p n)) (HPow.hPow (ENNReal.ofReal p) n)
  -/
  rw [ofReal_eq_coe_nnreal hp, ← coe_pow, ← ofReal_coe_nnreal, NNReal.coe_pow, NNReal.coe_mk]
  /-
    🎉 no goals
  -/


theorem ofReal_nsmul {x : ℝ} {n : ℕ} : ENNReal.ofReal (n • x) = n • ENNReal.ofReal x := by
  /-
    x : Real
    n : Nat
    ⊢ Eq (ENNReal.ofReal (HSMul.hSMul n x)) (HSMul.hSMul n (ENNReal.ofReal x))
  -/
  simp only [nsmul_eq_mul, ← ofReal_natCast n, ← ofReal_mul n.cast_nonneg]
  /-
    🎉 no goals
  -/


theorem ofReal_inv_of_pos {x : ℝ} (hx : 0 < x) : ENNReal.ofReal x⁻¹ = (ENNReal.ofReal x)⁻¹ := by
  rw [ENNReal.ofReal, ENNReal.ofReal, ← @coe_inv (Real.toNNReal x) (by simp [hx]), coe_inj,
    ← Real.toNNReal_inv]


theorem ofReal_div_of_pos {x y : ℝ} (hy : 0 < y) :
    ENNReal.ofReal (x / y) = ENNReal.ofReal x / ENNReal.ofReal y := by
  /-
    x y : Real
    hy : LT.lt 0 y
    ⊢ Eq (ENNReal.ofReal (HDiv.hDiv x y)) (HDiv.hDiv (ENNReal.ofReal x) (ENNReal.o …
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv, ofReal_mul' (inv_nonneg.2 hy.le), ofReal_inv_of_pos hy]
  /-
    🎉 no goals
  -/


@[simp]
theorem toNNReal_mul {a b : ℝ≥0∞} : (a * b).toNNReal = a.toNNReal * b.toNNReal :=
  WithTop.untop'_zero_mul a b


                                                                         /-
                                                                           a : ENNReal
                                                                           ⊢ Eq (HMul.hMul a Top.top).toNNReal 0
                                                                         -/
theorem toNNReal_mul_top (a : ℝ≥0∞) : ENNReal.toNNReal (a * ∞) = 0 := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


                                                                         /-
                                                                           a : ENNReal
                                                                           ⊢ Eq (HMul.hMul Top.top a).toNNReal 0
                                                                         -/
theorem toNNReal_top_mul (a : ℝ≥0∞) : ENNReal.toNNReal (∞ * a) = 0 := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem smul_toNNReal (a : ℝ≥0) (b : ℝ≥0∞) : (a • b).toNNReal = a * b.toNNReal := by
  /-
    a : NNReal
    b : ENNReal
    ⊢ Eq (HSMul.hSMul a b).toNNReal (HMul.hMul a b.toNNReal)
  -/
  change ((a : ℝ≥0∞) * b).toNNReal = a * b.toNNReal
  /-
    a : NNReal
    b : ENNReal
    ⊢ Eq (HMul.hMul (↑a) b).toNNReal (HMul.hMul a b.toNNReal)
  -/
  simp only [ENNReal.toNNReal_mul, ENNReal.toNNReal_coe]
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: upgrade to `→*₀`

/-- `ENNReal.toNNReal` as a `MonoidHom`. -/
def toNNRealHom : ℝ≥0∞ →* ℝ≥0 where
  toFun := ENNReal.toNNReal
  map_one' := toNNReal_coe _
  map_mul' _ _ := toNNReal_mul


@[simp]
theorem toNNReal_pow (a : ℝ≥0∞) (n : ℕ) : (a ^ n).toNNReal = a.toNNReal ^ n :=
  toNNRealHom.map_pow a n


@[simp]
theorem toNNReal_prod {ι : Type*} {s : Finset ι} {f : ι → ℝ≥0∞} :
    (∏ i ∈ s, f i).toNNReal = ∏ i ∈ s, (f i).toNNReal :=
  map_prod toNNRealHom _ _

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: upgrade to `→*₀`

/-- `ENNReal.toReal` as a `MonoidHom`. -/
def toRealHom : ℝ≥0∞ →* ℝ :=
  (NNReal.toRealHom : ℝ≥0 →* ℝ).comp toNNRealHom


@[simp]
theorem toReal_mul : (a * b).toReal = a.toReal * b.toReal :=
  toRealHom.map_mul a b


                                                                              /-
                                                                                a : ENNReal
                                                                                n : Nat
                                                                                ⊢ Eq (HSMul.hSMul n a).toReal (HSMul.hSMul n a.toReal)
                                                                              -/
theorem toReal_nsmul (a : ℝ≥0∞) (n : ℕ) : (n • a).toReal = n • a.toReal := by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem toReal_pow (a : ℝ≥0∞) (n : ℕ) : (a ^ n).toReal = a.toReal ^ n :=
  toRealHom.map_pow a n


@[simp]
theorem toReal_prod {ι : Type*} {s : Finset ι} {f : ι → ℝ≥0∞} :
    (∏ i ∈ s, f i).toReal = ∏ i ∈ s, (f i).toReal :=
  map_prod toRealHom _ _


theorem toReal_ofReal_mul (c : ℝ) (a : ℝ≥0∞) (h : 0 ≤ c) :
    ENNReal.toReal (ENNReal.ofReal c * a) = c * ENNReal.toReal a := by
  /-
    c : Real
    a : ENNReal
    h : LE.le 0 c
    ⊢ Eq (HMul.hMul (ENNReal.ofReal c) a).toReal (HMul.hMul c a.toReal)
  -/
  rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal h]
  /-
    🎉 no goals
  -/


theorem toReal_mul_top (a : ℝ≥0∞) : ENNReal.toReal (a * ∞) = 0 := by
  /-
    a : ENNReal
    ⊢ Eq (HMul.hMul a Top.top).toReal 0
  -/
  rw [toReal_mul, top_toReal, mul_zero]
  /-
    🎉 no goals
  -/


theorem toReal_top_mul (a : ℝ≥0∞) : ENNReal.toReal (∞ * a) = 0 := by
  /-
    a : ENNReal
    ⊢ Eq (HMul.hMul Top.top a).toReal 0
  -/
  rw [mul_comm]
  /-
    a : ENNReal
    ⊢ Eq (HMul.hMul a Top.top).toReal 0
  -/
  exact toReal_mul_top _
  /-
    🎉 no goals
  -/


theorem toReal_eq_toReal (ha : a ≠ ∞) (hb : b ≠ ∞) : a.toReal = b.toReal ↔ a = b := by
  /-
    a b : ENNReal
    ha : Ne a Top.top
    hb : Ne b Top.top
    ⊢ Iff (Eq a.toReal b.toReal) (Eq a b)
  -/
  lift a to ℝ≥0 using ha
  /-
    case intro
    b : ENNReal
    hb : Ne b Top.top
    a : NNReal
    ⊢ Iff (Eq (↑a).toReal b.toReal) (Eq (↑a) b)
  -/
  lift b to ℝ≥0 using hb
  /-
    case intro.intro
    a b : NNReal
    ⊢ Iff (Eq (↑a).toReal (↑b).toReal) (Eq ↑a ↑b)
  -/
  simp only [coe_inj, NNReal.coe_inj, coe_toReal]
  /-
    🎉 no goals
  -/


theorem toReal_smul (r : ℝ≥0) (s : ℝ≥0∞) : (r • s).toReal = r • s.toReal := by
  /-
    r : NNReal
    s : ENNReal
    ⊢ Eq (HSMul.hSMul r s).toReal (HSMul.hSMul r s.toReal)
  -/
  rw [ENNReal.smul_def, smul_eq_mul, toReal_mul, coe_toReal]
  /-
    r : NNReal
    s : ENNReal
    ⊢ Eq (HMul.hMul (↑r) s.toReal) (HSMul.hSMul r s.toReal)
  -/
  rfl
  /-
    🎉 no goals
  -/


protected theorem trichotomy (p : ℝ≥0∞) : p = 0 ∨ p = ∞ ∨ 0 < p.toReal := by
  /-
    p : ENNReal
    ⊢ Or (Eq p 0) (Or (Eq p Top.top) (LT.lt 0 p.toReal))
  -/
  simpa only [or_iff_not_imp_left] using toReal_pos
  /-
    🎉 no goals
  -/


protected theorem trichotomy₂ {p q : ℝ≥0∞} (hpq : p ≤ q) :
    p = 0 ∧ q = 0 ∨
      p = 0 ∧ q = ∞ ∨
        p = 0 ∧ 0 < q.toReal ∨
          p = ∞ ∧ q = ∞ ∨
            0 < p.toReal ∧ q = ∞ ∨ 0 < p.toReal ∧ 0 < q.toReal ∧ p.toReal ≤ q.toReal := by
  /-
    p q : ENNReal
    hpq : LE.le p q
    ⊢ Or (And (Eq p 0) (Eq q 0)) (Or (And (Eq p 0) (Eq q Top.top)) (Or (And (Eq p  …
  -/
  rcases eq_or_lt_of_le (bot_le : 0 ≤ p) with ((rfl : 0 = p) | (hp : 0 < p))
    /-
      case inl
      q : ENNReal
      hpq : LE.le 0 q
      ⊢ Or (And (Eq 0 0) (Eq q 0)) (Or (And (Eq 0 0) (Eq q Top.top)) (Or (And (Eq 0  …
    -/
  · simpa using q.trichotomy
    /-
      🎉 no goals
    -/
  /-
    case inr
    p q : ENNReal
    hpq : LE.le p q
    hp : LT.lt 0 p
    ⊢ Or (And (Eq p 0) (Eq q 0)) (Or (And (Eq p 0) (Eq q Top.top)) (Or (And (Eq p  …
  -/
  rcases eq_or_lt_of_le (le_top : q ≤ ∞) with (rfl | hq)
    /-
      case inr.inl
      p : ENNReal
      hp : LT.lt 0 p
      hpq : LE.le p Top.top
      ⊢ Or (And (Eq p 0) (Eq Top.top 0)) (Or (And (Eq p 0) (Eq Top.top Top.top)) (Or …
    -/
  · simpa using p.trichotomy
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    p q : ENNReal
    hpq : LE.le p q
    hp : LT.lt 0 p
    hq : LT.lt q Top.top
    ⊢ Or (And (Eq p 0) (Eq q 0)) (Or (And (Eq p 0) (Eq q Top.top)) (Or (And (Eq p  …
  -/
  repeat' right
  /-
    case inr.inr.h.h.h.h.h
    p q : ENNReal
    hpq : LE.le p q
    hp : LT.lt 0 p
    hq : LT.lt q Top.top
    ⊢ And (LT.lt 0 p.toReal) (And (LT.lt 0 q.toReal) (LE.le p.toReal q.toReal))
  -/
  have hq' : 0 < q := lt_of_lt_of_le hp hpq
  /-
    case inr.inr.h.h.h.h.h
    p q : ENNReal
    hpq : LE.le p q
    hp : LT.lt 0 p
    hq : LT.lt q Top.top
    hq' : LT.lt 0 q
    ⊢ And (LT.lt 0 p.toReal) (And (LT.lt 0 q.toReal) (LE.le p.toReal q.toReal))
  -/
  have hp' : p < ∞ := lt_of_le_of_lt hpq hq
  /-
    case inr.inr.h.h.h.h.h
    p q : ENNReal
    hpq : LE.le p q
    hp : LT.lt 0 p
    hq : LT.lt q Top.top
    hq' : LT.lt 0 q
    hp' : LT.lt p Top.top
    ⊢ And (LT.lt 0 p.toReal) (And (LT.lt 0 q.toReal) (LE.le p.toReal q.toReal))
  -/
  simp [ENNReal.toReal_mono hq.ne hpq, ENNReal.toReal_pos_iff, hp, hp', hq', hq]
  /-
    🎉 no goals
  -/


protected theorem dichotomy (p : ℝ≥0∞) [Fact (1 ≤ p)] : p = ∞ ∨ 1 ≤ p.toReal :=
  haveI : p = ⊤ ∨ 0 < p.toReal ∧ 1 ≤ p.toReal := by
    /-
      p : ENNReal
      inst✝ : Fact (LE.le 1 p)
      ⊢ Or (Eq p Top.top) (And (LT.lt 0 p.toReal) (LE.le 1 p.toReal))
    -/
    simpa using ENNReal.trichotomy₂ (Fact.out : 1 ≤ p)
    /-
      🎉 no goals
    -/
  this.imp_right fun h => h.2


theorem toReal_pos_iff_ne_top (p : ℝ≥0∞) [Fact (1 ≤ p)] : 0 < p.toReal ↔ p ≠ ∞ :=
  ⟨fun h hp =>
    have : (0 : ℝ) ≠ 0 := top_toReal ▸ (hp ▸ h.ne : 0 ≠ ∞.toReal)
    this rfl,
    fun h => zero_lt_one.trans_le (p.dichotomy.resolve_left h)⟩


@[simp] theorem toNNReal_inv (a : ℝ≥0∞) : a⁻¹.toNNReal = a.toNNReal⁻¹ := by
  /-
    a : ENNReal
    ⊢ Eq (Inv.inv a).toNNReal (Inv.inv a.toNNReal)
  -/
  induction' a with a; · simp
                         /-
                           🎉 no goals
                         -/
  /-
    case coe
    a : NNReal
    ⊢ Eq (Inv.inv ↑a).toNNReal (Inv.inv (↑a).toNNReal)
  -/
  rcases eq_or_ne a 0 with (rfl | ha); · simp
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case coe.inr
    a : NNReal
    ha : Ne a 0
    ⊢ Eq (Inv.inv ↑a).toNNReal (Inv.inv (↑a).toNNReal)
  -/
  rw [← coe_inv ha, toNNReal_coe, toNNReal_coe]
  /-
    🎉 no goals
  -/


@[simp] theorem toNNReal_div (a b : ℝ≥0∞) : (a / b).toNNReal = a.toNNReal / b.toNNReal := by
  /-
    a b : ENNReal
    ⊢ Eq (HDiv.hDiv a b).toNNReal (HDiv.hDiv a.toNNReal b.toNNReal)
  -/
  rw [div_eq_mul_inv, toNNReal_mul, toNNReal_inv, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


@[simp] theorem toReal_inv (a : ℝ≥0∞) : a⁻¹.toReal = a.toReal⁻¹ := by
  /-
    a : ENNReal
    ⊢ Eq (Inv.inv a).toReal (Inv.inv a.toReal)
  -/
  simp only [ENNReal.toReal, toNNReal_inv, NNReal.coe_inv]
  /-
    🎉 no goals
  -/


@[simp] theorem toReal_div (a b : ℝ≥0∞) : (a / b).toReal = a.toReal / b.toReal := by
  /-
    a b : ENNReal
    ⊢ Eq (HDiv.hDiv a b).toReal (HDiv.hDiv a.toReal b.toReal)
  -/
  rw [div_eq_mul_inv, toReal_mul, toReal_inv, div_eq_mul_inv]
  /-
    🎉 no goals
  -/


theorem ofReal_prod_of_nonneg {α : Type*} {s : Finset α} {f : α → ℝ} (hf : ∀ i, i ∈ s → 0 ≤ f i) :
    ENNReal.ofReal (∏ i ∈ s, f i) = ∏ i ∈ s, ENNReal.ofReal (f i) := by
  /-
    α : Type u_1
    s : Finset α
    f : α → Real
    hf : ∀ (i : α), Membership.mem s i → LE.le 0 (f i)
    ⊢ Eq (ENNReal.ofReal (s.prod fun i => f i)) (s.prod fun i => ENNReal.ofReal (f …
  -/
  simp_rw [ENNReal.ofReal, ← coe_finset_prod, coe_inj]
  /-
    α : Type u_1
    s : Finset α
    f : α → Real
    hf : ∀ (i : α), Membership.mem s i → LE.le 0 (f i)
    ⊢ Eq (s.prod fun i => f i).toNNReal (s.prod fun a => (f a).toNNReal)
  -/
  exact Real.toNNReal_prod_of_nonneg hf
  /-
    🎉 no goals
  -/


theorem toNNReal_iInf (hf : ∀ i, f i ≠ ∞) : (iInf f).toNNReal = ⨅ i, (f i).toNNReal := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    hf : ∀ (i : ι), Ne (f i) Top.top
    ⊢ Eq (iInf f).toNNReal (iInf fun i => (f i).toNNReal)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_1
      f : ι → ENNReal
      hf : ∀ (i : ι), Ne (f i) Top.top
      h✝ : IsEmpty ι
      ⊢ Eq (iInf f).toNNReal (iInf fun i => (f i).toNNReal)
    -/
  · rw [iInf_of_empty, top_toNNReal, NNReal.iInf_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_1
      f : ι → ENNReal
      hf : ∀ (i : ι), Ne (f i) Top.top
      h✝ : Nonempty ι
      ⊢ Eq (iInf f).toNNReal (iInf fun i => (f i).toNNReal)
    -/
  · lift f to ι → ℝ≥0 using hf
    /-
      case inr.intro
      ι : Sort u_1
      h✝ : Nonempty ι
      f : ι → NNReal
      ⊢ Eq (iInf fun i => ↑(f i)).toNNReal (iInf fun i => ((fun i => ↑(f i)) i).toNN …
    -/
    simp_rw [← coe_iInf, toNNReal_coe]
    /-
      🎉 no goals
    -/


theorem toNNReal_sInf (s : Set ℝ≥0∞) (hs : ∀ r ∈ s, r ≠ ∞) :
    (sInf s).toNNReal = sInf (ENNReal.toNNReal '' s) := by
  /-
    s : Set ENNReal
    hs : ∀ (r : ENNReal), Membership.mem s r → Ne r Top.top
    ⊢ Eq (InfSet.sInf s).toNNReal (InfSet.sInf (Set.image ENNReal.toNNReal s))
  -/
  have hf : ∀ i, ((↑) : s → ℝ≥0∞) i ≠ ∞ := fun ⟨r, rs⟩ => hs r rs
  -- Porting note: `← sInf_image'` had to be replaced by `← image_eq_range` as the lemmas are used
  -- in a different order.
  /-
    s : Set ENNReal
    hs : ∀ (r : ENNReal), Membership.mem s r → Ne r Top.top
    hf : ∀ (i : Subtype fun x => Membership.mem s x), Ne (↑i) Top.top
    ⊢ Eq (InfSet.sInf s).toNNReal (InfSet.sInf (Set.image ENNReal.toNNReal s))
  -/
  simpa only [← sInf_range, ← image_eq_range, Subtype.range_coe_subtype] using (toNNReal_iInf hf)
  /-
    🎉 no goals
  -/


theorem toNNReal_iSup (hf : ∀ i, f i ≠ ∞) : (iSup f).toNNReal = ⨆ i, (f i).toNNReal := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    hf : ∀ (i : ι), Ne (f i) Top.top
    ⊢ Eq (iSup f).toNNReal (iSup fun i => (f i).toNNReal)
  -/
  lift f to ι → ℝ≥0 using hf
  /-
    case intro
    ι : Sort u_1
    f : ι → NNReal
    ⊢ Eq (iSup fun i => ↑(f i)).toNNReal (iSup fun i => ((fun i => ↑(f i)) i).toNN …
  -/
  simp_rw [toNNReal_coe]
  /-
    case intro
    ι : Sort u_1
    f : ι → NNReal
    ⊢ Eq (iSup fun i => ↑(f i)).toNNReal (iSup fun i => f i)
  -/
  by_cases h : BddAbove (range f)
    /-
      case pos
      ι : Sort u_1
      f : ι → NNReal
      h : BddAbove (Set.range f)
      ⊢ Eq (iSup fun i => ↑(f i)).toNNReal (iSup fun i => f i)
    -/
  · rw [← coe_iSup h, toNNReal_coe]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Sort u_1
      f : ι → NNReal
      h : Not (BddAbove (Set.range f))
      ⊢ Eq (iSup fun i => ↑(f i)).toNNReal (iSup fun i => f i)
    -/
  · rw [NNReal.iSup_of_not_bddAbove h, iSup_coe_eq_top.2 h, top_toNNReal]
    /-
      🎉 no goals
    -/


theorem toNNReal_sSup (s : Set ℝ≥0∞) (hs : ∀ r ∈ s, r ≠ ∞) :
    (sSup s).toNNReal = sSup (ENNReal.toNNReal '' s) := by
  /-
    s : Set ENNReal
    hs : ∀ (r : ENNReal), Membership.mem s r → Ne r Top.top
    ⊢ Eq (SupSet.sSup s).toNNReal (SupSet.sSup (Set.image ENNReal.toNNReal s))
  -/
  have hf : ∀ i, ((↑) : s → ℝ≥0∞) i ≠ ∞ := fun ⟨r, rs⟩ => hs r rs
  -- Porting note: `← sSup_image'` had to be replaced by `← image_eq_range` as the lemmas are used
  -- in a different order.
  /-
    s : Set ENNReal
    hs : ∀ (r : ENNReal), Membership.mem s r → Ne r Top.top
    hf : ∀ (i : Subtype fun x => Membership.mem s x), Ne (↑i) Top.top
    ⊢ Eq (SupSet.sSup s).toNNReal (SupSet.sSup (Set.image ENNReal.toNNReal s))
  -/
  simpa only [← sSup_range, ← image_eq_range, Subtype.range_coe_subtype] using (toNNReal_iSup hf)
  /-
    🎉 no goals
  -/


theorem toReal_iInf (hf : ∀ i, f i ≠ ∞) : (iInf f).toReal = ⨅ i, (f i).toReal := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    hf : ∀ (i : ι), Ne (f i) Top.top
    ⊢ Eq (iInf f).toReal (iInf fun i => (f i).toReal)
  -/
  simp only [ENNReal.toReal, toNNReal_iInf hf, NNReal.coe_iInf]
  /-
    🎉 no goals
  -/


theorem toReal_sInf (s : Set ℝ≥0∞) (hf : ∀ r ∈ s, r ≠ ∞) :
    (sInf s).toReal = sInf (ENNReal.toReal '' s) := by
  /-
    s : Set ENNReal
    hf : ∀ (r : ENNReal), Membership.mem s r → Ne r Top.top
    ⊢ Eq (InfSet.sInf s).toReal (InfSet.sInf (Set.image ENNReal.toReal s))
  -/
  simp only [ENNReal.toReal, toNNReal_sInf s hf, NNReal.coe_sInf, Set.image_image]
  /-
    🎉 no goals
  -/


theorem toReal_iSup (hf : ∀ i, f i ≠ ∞) : (iSup f).toReal = ⨆ i, (f i).toReal := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    hf : ∀ (i : ι), Ne (f i) Top.top
    ⊢ Eq (iSup f).toReal (iSup fun i => (f i).toReal)
  -/
  simp only [ENNReal.toReal, toNNReal_iSup hf, NNReal.coe_iSup]
  /-
    🎉 no goals
  -/


theorem toReal_sSup (s : Set ℝ≥0∞) (hf : ∀ r ∈ s, r ≠ ∞) :
    (sSup s).toReal = sSup (ENNReal.toReal '' s) := by
  /-
    s : Set ENNReal
    hf : ∀ (r : ENNReal), Membership.mem s r → Ne r Top.top
    ⊢ Eq (SupSet.sSup s).toReal (SupSet.sSup (Set.image ENNReal.toReal s))
  -/
  simp only [ENNReal.toReal, toNNReal_sSup s hf, NNReal.coe_sSup, Set.image_image]
  /-
    🎉 no goals
  -/


@[simp] lemma ofReal_iInf [Nonempty ι] (f : ι → ℝ) :
    ENNReal.ofReal (⨅ i, f i) = ⨅ i, ENNReal.ofReal (f i) := by
  /-
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : ι → Real
    ⊢ Eq (ENNReal.ofReal (iInf fun i => f i)) (iInf fun i => ENNReal.ofReal (f i))
  -/
  obtain ⟨i, hi⟩ | h := em (∃ i, f i ≤ 0)
    /-
      case inl.intro
      ι : Sort u_1
      inst✝ : Nonempty ι
      f : ι → Real
      i : ι
      hi : LE.le (f i) 0
      ⊢ Eq (ENNReal.ofReal (iInf fun i => f i)) (iInf fun i => ENNReal.ofReal (f i))
    -/
  · rw [(iInf_eq_bot _).2 fun _ _ ↦ ⟨i, by simpa [ofReal_of_nonpos hi]⟩]
    /-
      case inl.intro
      ι : Sort u_1
      inst✝ : Nonempty ι
      f : ι → Real
      i : ι
      hi : LE.le (f i) 0
      ⊢ Eq (ENNReal.ofReal (iInf fun i => f i)) Bot.bot
    -/
    simp [Real.iInf_nonpos' ⟨i, hi⟩]
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : ι → Real
    h : Not (Exists fun i => LE.le (f i) 0)
    ⊢ Eq (ENNReal.ofReal (iInf fun i => f i)) (iInf fun i => ENNReal.ofReal (f i))
  -/
  replace h i : 0 ≤ f i := le_of_not_le fun hi ↦ h ⟨i, hi⟩
  /-
    case inr
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : ι → Real
    h : ∀ (i : ι), LE.le 0 (f i)
    ⊢ Eq (ENNReal.ofReal (iInf fun i => f i)) (iInf fun i => ENNReal.ofReal (f i))
  -/
  refine eq_of_forall_le_iff fun a ↦ ?_
  /-
    case inr
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : ι → Real
    h : ∀ (i : ι), LE.le 0 (f i)
    a : ENNReal
    ⊢ Iff (LE.le a (ENNReal.ofReal (iInf fun i => f i))) (LE.le a (iInf fun i => E …
  -/
  obtain rfl | ha := eq_or_ne a ∞
    /-
      case inr.inl
      ι : Sort u_1
      inst✝ : Nonempty ι
      f : ι → Real
      h : ∀ (i : ι), LE.le 0 (f i)
      ⊢ Iff (LE.le Top.top (ENNReal.ofReal (iInf fun i => f i))) (LE.le Top.top (iIn …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    ι : Sort u_1
    inst✝ : Nonempty ι
    f : ι → Real
    h : ∀ (i : ι), LE.le 0 (f i)
    a : ENNReal
    ha : Ne a Top.top
    ⊢ Iff (LE.le a (ENNReal.ofReal (iInf fun i => f i))) (LE.le a (iInf fun i => E …
  -/
  rw [le_iInf_iff, le_ofReal_iff_toReal_le ha, le_ciInf_iff ⟨0, by simpa [mem_lowerBounds]⟩]
    /-
      case inr.inr
      ι : Sort u_1
      inst✝ : Nonempty ι
      f : ι → Real
      h : ∀ (i : ι), LE.le 0 (f i)
      a : ENNReal
      ha : Ne a Top.top
      ⊢ Iff (∀ (i : ι), LE.le a.toReal (f i)) (∀ (i : ι), LE.le a (ENNReal.ofReal (f …
    -/
  · exact forall_congr' fun i ↦ (le_ofReal_iff_toReal_le ha (h _)).symm
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      ι : Sort u_1
      inst✝ : Nonempty ι
      f : ι → Real
      h : ∀ (i : ι), LE.le 0 (f i)
      a : ENNReal
      ha : Ne a Top.top
      ⊢ LE.le 0 (iInf fun i => f i)
    -/
  · exact Real.iInf_nonneg h
    /-
      🎉 no goals
    -/


theorem iInf_add : iInf f + a = ⨅ i, f i + a :=
  le_antisymm (le_iInf fun _ => add_le_add (iInf_le _ _) <| le_rfl)
    (tsub_le_iff_right.1 <| le_iInf fun _ => tsub_le_iff_right.2 <| iInf_le _ _)


theorem iSup_sub : (⨆ i, f i) - a = ⨆ i, f i - a :=
  le_antisymm (tsub_le_iff_right.2 <| iSup_le fun i => tsub_le_iff_right.1 <| le_iSup (f · - a) i)
    (iSup_le fun _ => tsub_le_tsub (le_iSup _ _) (le_refl a))


theorem sub_iInf : (a - ⨅ i, f i) = ⨆ i, a - f i := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    ⊢ Eq (HSub.hSub a (iInf fun i => f i)) (iSup fun i => HSub.hSub a (f i))
  -/
  refine eq_of_forall_ge_iff fun c => ?_
  /-
    ι : Sort u_1
    f : ι → ENNReal
    a c : ENNReal
    ⊢ Iff (LE.le (HSub.hSub a (iInf fun i => f i)) c) (LE.le (iSup fun i => HSub.h …
  -/
  rw [tsub_le_iff_right, add_comm, iInf_add]
  /-
    ι : Sort u_1
    f : ι → ENNReal
    a c : ENNReal
    ⊢ Iff (LE.le a (iInf fun i => HAdd.hAdd (f i) c)) (LE.le (iSup fun i => HSub.h …
  -/
  simp [tsub_le_iff_right, sub_eq_add_neg, add_comm]
  /-
    🎉 no goals
  -/


                                                                    /-
                                                                      a : ENNReal
                                                                      s : Set ENNReal
                                                                      ⊢ Eq (HAdd.hAdd (InfSet.sInf s) a) (iInf fun b => iInf fun h => HAdd.hAdd b a)
                                                                    -/
theorem sInf_add {s : Set ℝ≥0∞} : sInf s + a = ⨅ b ∈ s, b + a := by simp [sInf_eq_iInf, iInf_add]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem add_iInf {a : ℝ≥0∞} : a + iInf f = ⨅ b, a + f b := by
  /-
    ι : Sort u_1
    f : ι → ENNReal
    a : ENNReal
    ⊢ Eq (HAdd.hAdd a (iInf f)) (iInf fun b => HAdd.hAdd a (f b))
  -/
  rw [add_comm, iInf_add]; simp [add_comm]
                           /-
                             🎉 no goals
                           -/


theorem iInf_add_iInf (h : ∀ i j, ∃ k, f k + g k ≤ f i + g j) : iInf f + iInf g = ⨅ a, f a + g a :=
  suffices ⨅ a, f a + g a ≤ iInf f + iInf g from
    le_antisymm (le_iInf fun _ => add_le_add (iInf_le _ _) (iInf_le _ _)) this
  calc
    ⨅ a, f a + g a ≤ ⨅ (a) (a'), f a + g a' :=
      le_iInf₂ fun a a' => let ⟨k, h⟩ := h a a'; iInf_le_of_le k h
                              /-
                                ι : Sort u_1
                                f g : ι → ENNReal
                                h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f k) (g k)) (HAdd.hAdd (f i …
                                ⊢ Eq (iInf fun a => iInf fun a' => HAdd.hAdd (f a) (g a')) (HAdd.hAdd (iInf f) …
                              -/
    _ = iInf f + iInf g := by simp_rw [iInf_add, add_iInf]
                              /-
                                🎉 no goals
                              -/


theorem iInf_sum {α : Type*} {f : ι → α → ℝ≥0∞} {s : Finset α} [Nonempty ι]
    (h : ∀ (t : Finset α) (i j : ι), ∃ k, ∀ a ∈ t, f k a ≤ f i a ∧ f k a ≤ f j a) :
    ⨅ i, ∑ a ∈ s, f i a = ∑ a ∈ s, ⨅ i, f i a := by
  /-
    ι : Sort u_1
    α : Type u_2
    f : ι → α → ENNReal
    s : Finset α
    inst✝ : Nonempty ι
    h : ∀ (t : Finset α) (i j : ι), Exists fun k => ∀ (a : α), Membership.mem t a  …
    ⊢ Eq (iInf fun i => s.sum fun a => f i a) (s.sum fun a => iInf fun i => f i a)
  -/
  induction' s using Finset.cons_induction_on with a s ha ih
    /-
      case h₁
      ι : Sort u_1
      α : Type u_2
      f : ι → α → ENNReal
      inst✝ : Nonempty ι
      h : ∀ (t : Finset α) (i j : ι), Exists fun k => ∀ (a : α), Membership.mem t a  …
      ⊢ Eq (iInf fun i => EmptyCollection.emptyCollection.sum fun a => f i a) (Empty …
    -/
  · simp only [Finset.sum_empty, ciInf_const]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      ι : Sort u_1
      α : Type u_2
      f : ι → α → ENNReal
      inst✝ : Nonempty ι
      h : ∀ (t : Finset α) (i j : ι), Exists fun k => ∀ (a : α), Membership.mem t a  …
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      ih : Eq (iInf fun i => s.sum fun a => f i a) (s.sum fun a => iInf fun i => f i …
      ⊢ Eq (iInf fun i => (Finset.cons a s ha).sum fun a => f i a) ((Finset.cons a s …
    -/
  · simp only [Finset.sum_cons, ← ih]
    /-
      case h₂
      ι : Sort u_1
      α : Type u_2
      f : ι → α → ENNReal
      inst✝ : Nonempty ι
      h : ∀ (t : Finset α) (i j : ι), Exists fun k => ∀ (a : α), Membership.mem t a  …
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      ih : Eq (iInf fun i => s.sum fun a => f i a) (s.sum fun a => iInf fun i => f i …
      ⊢ Eq (iInf fun i => HAdd.hAdd (f i a) (s.sum fun a => f i a)) (HAdd.hAdd (iInf …
    -/
    refine (iInf_add_iInf fun i j => ?_).symm
    /-
      case h₂
      ι : Sort u_1
      α : Type u_2
      f : ι → α → ENNReal
      inst✝ : Nonempty ι
      h : ∀ (t : Finset α) (i j : ι), Exists fun k => ∀ (a : α), Membership.mem t a  …
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      ih : Eq (iInf fun i => s.sum fun a => f i a) (s.sum fun a => iInf fun i => f i …
      i j : ι
      ⊢ Exists fun k => LE.le (HAdd.hAdd (f k a) (s.sum fun a => f k a)) (HAdd.hAdd  …
    -/
    refine (h (Finset.cons a s ha) i j).imp fun k hk => ?_
    /-
      case h₂
      ι : Sort u_1
      α : Type u_2
      f : ι → α → ENNReal
      inst✝ : Nonempty ι
      h : ∀ (t : Finset α) (i j : ι), Exists fun k => ∀ (a : α), Membership.mem t a  …
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      ih : Eq (iInf fun i => s.sum fun a => f i a) (s.sum fun a => iInf fun i => f i …
      i j k : ι
      hk : ∀ (a_1 : α), Membership.mem (Finset.cons a s ha) a_1 → And (LE.le (f k a_ …
      ⊢ LE.le (HAdd.hAdd (f k a) (s.sum fun a => f k a)) (HAdd.hAdd (f i a) (s.sum f …
    -/
    rw [Finset.forall_mem_cons] at hk
    /-
      case h₂
      ι : Sort u_1
      α : Type u_2
      f : ι → α → ENNReal
      inst✝ : Nonempty ι
      h : ∀ (t : Finset α) (i j : ι), Exists fun k => ∀ (a : α), Membership.mem t a  …
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      ih : Eq (iInf fun i => s.sum fun a => f i a) (s.sum fun a => iInf fun i => f i …
      i j k : ι
      hk : And (And (LE.le (f k a) (f i a)) (LE.le (f k a) (f j a))) (∀ (x : α), Mem …
      ⊢ LE.le (HAdd.hAdd (f k a) (s.sum fun a => f k a)) (HAdd.hAdd (f i a) (s.sum f …
    -/
    exact add_le_add hk.1.1 (Finset.sum_le_sum fun a ha => (hk.2 a ha).2)
    /-
      🎉 no goals
    -/


theorem sup_eq_zero {a b : ℝ≥0∞} : a ⊔ b = 0 ↔ a = 0 ∧ b = 0 :=
  sup_eq_bot_iff


@[deprecated (since := "2024-04-05")] alias iSup_coe_nat := iSup_natCast


/-- Extension for the `positivity` tactic: `ENNReal.ofReal`. -/
@[positivity ENNReal.ofReal _]
def evalENNRealOfReal : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ≥0∞), ~q(ENNReal.ofReal $a) =>
    let ra ← core q(inferInstance) q(inferInstance) a
    assertInstancesCommute
    match ra with
    | .positive pa => pure (.positive q(Iff.mpr (@ENNReal.ofReal_pos $a) $pa))
    | _ => pure .none
  | _, _, _ => throwError "not ENNReal.ofReal"

