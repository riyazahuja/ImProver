/-- Two real exponents `p, q` are conjugate if they are `> 1` and satisfy the equality
`1/p + 1/q = 1`. This condition shows up in many theorems in analysis, notably related to `L^p`
norms. -/
@[mk_iff]
structure IsConjExponent (p q : ℝ) : Prop where
  one_lt : 1 < p
  inv_add_inv_conj : p⁻¹ + q⁻¹ = 1


/-- The conjugate exponent of `p` is `q = p/(p-1)`, so that `1/p + 1/q = 1`. -/
def conjExponent (p : ℝ) : ℝ := p / (p - 1)


theorem pos : 0 < p := lt_trans zero_lt_one h.one_lt


theorem nonneg : 0 ≤ p := le_of_lt h.pos


theorem ne_zero : p ≠ 0 := ne_of_gt h.pos


theorem sub_one_pos : 0 < p - 1 := sub_pos.2 h.one_lt


theorem sub_one_ne_zero : p - 1 ≠ 0 := ne_of_gt h.sub_one_pos


protected lemma inv_pos : 0 < p⁻¹ := inv_pos.2 h.pos

protected lemma inv_nonneg : 0 ≤ p⁻¹ := h.inv_pos.le

protected lemma inv_ne_zero : p⁻¹ ≠ 0 := h.inv_pos.ne'


theorem one_div_pos : 0 < 1 / p := _root_.one_div_pos.2 h.pos


theorem one_div_nonneg : 0 ≤ 1 / p := le_of_lt h.one_div_pos


theorem one_div_ne_zero : 1 / p ≠ 0 := ne_of_gt h.one_div_pos


theorem conj_eq : q = p / (p - 1) := by
  /-
    p q : Real
    h : p.IsConjExponent q
    ⊢ Eq q (HDiv.hDiv p (HSub.hSub p 1))
  -/
  have := h.inv_add_inv_conj
  /-
    p q : Real
    h : p.IsConjExponent q
    this : Eq (HAdd.hAdd (Inv.inv p) (Inv.inv q)) 1
    ⊢ Eq q (HDiv.hDiv p (HSub.hSub p 1))
  -/
  rw [← eq_sub_iff_add_eq', inv_eq_iff_eq_inv] at this
  /-
    p q : Real
    h : p.IsConjExponent q
    this : Eq q (Inv.inv (HSub.hSub 1 (Inv.inv p)))
    ⊢ Eq q (HDiv.hDiv p (HSub.hSub p 1))
  -/
  field_simp [this, h.ne_zero]
  /-
    🎉 no goals
  -/


lemma conjExponent_eq : conjExponent p = q := h.conj_eq.symm


lemma one_sub_inv : 1 - p⁻¹ = q⁻¹ := sub_eq_of_eq_add' h.inv_add_inv_conj.symm

                                         /-
                                           p q : Real
                                           h : p.IsConjExponent q
                                           ⊢ Eq (HSub.hSub (Inv.inv p) 1) (Neg.neg (Inv.inv q))
                                         -/
lemma inv_sub_one : p⁻¹ - 1 = -q⁻¹ := by rw [← h.inv_add_inv_conj, sub_add_cancel_left]
                                         /-
                                           🎉 no goals
                                         -/


theorem sub_one_mul_conj : (p - 1) * q = p :=
  mul_comm q (p - 1) ▸ (eq_div_iff h.sub_one_ne_zero).1 h.conj_eq


theorem mul_eq_add : p * q = p + q := by
  /-
    p q : Real
    h : p.IsConjExponent q
    ⊢ Eq (HMul.hMul p q) (HAdd.hAdd p q)
  -/
  simpa only [sub_mul, sub_eq_iff_eq_add, one_mul] using h.sub_one_mul_conj
  /-
    🎉 no goals
  -/


@[symm] protected lemma symm : q.IsConjExponent p where
               /-
                 p q : Real
                 h : p.IsConjExponent q
                 ⊢ LT.lt 1 q
               -/
  one_lt := by simpa only [h.conj_eq] using (one_lt_div h.sub_one_pos).mpr (sub_one_lt p)
               /-
                 🎉 no goals
               -/
                         /-
                           p q : Real
                           h : p.IsConjExponent q
                           ⊢ Eq (HAdd.hAdd (Inv.inv q) (Inv.inv p)) 1
                         -/
  inv_add_inv_conj := by simpa [add_comm] using h.inv_add_inv_conj
                         /-
                           🎉 no goals
                         -/


theorem div_conj_eq_sub_one : p / q = p - 1 := by
  /-
    p q : Real
    h : p.IsConjExponent q
    ⊢ Eq (HDiv.hDiv p q) (HSub.hSub p 1)
  -/
  field_simp [h.symm.ne_zero]
  /-
    p q : Real
    h : p.IsConjExponent q
    ⊢ Eq p (HMul.hMul (HSub.hSub p 1) q)
  -/
  rw [h.sub_one_mul_conj]
  /-
    🎉 no goals
  -/


theorem inv_add_inv_conj_ennreal : (ENNReal.ofReal p)⁻¹ + (ENNReal.ofReal q)⁻¹ = 1 := by
  rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_inv_of_pos h.pos,
    ← ENNReal.ofReal_inv_of_pos h.symm.pos, ← ENNReal.ofReal_add h.inv_nonneg h.symm.inv_nonneg,
    h.inv_add_inv_conj]


protected lemma inv_inv (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : a⁻¹.IsConjExponent b⁻¹ :=
                            /-
                              a b : Real
                              ha : LT.lt 0 a
                              hb : LT.lt 0 b
                              hab : Eq (HAdd.hAdd a b) 1
                              ⊢ LT.lt a 1
                            -/
                            /-
                              🎉 no goals
                            -/
  ⟨(one_lt_inv₀ ha).2 <| by linarith, by simpa only [inv_inv]⟩
                                         /-
                                           🎉 no goals
                                         -/


lemma inv_one_sub_inv (ha₀ : 0 < a) (ha₁ : a < 1) : a⁻¹.IsConjExponent (1 - a)⁻¹ :=
  .inv_inv ha₀ (sub_pos_of_lt ha₁) <| add_tsub_cancel_of_le ha₁.le


lemma one_sub_inv_inv (ha₀ : 0 < a) (ha₁ : a < 1) : (1 - a)⁻¹.IsConjExponent a⁻¹ :=
  (inv_one_sub_inv ha₀ ha₁).symm


lemma isConjExponent_comm : p.IsConjExponent q ↔ q.IsConjExponent p := ⟨.symm, .symm⟩


lemma isConjExponent_iff_eq_conjExponent (hp : 1 < p) : p.IsConjExponent q ↔ q = p / (p - 1) :=
                                           /-
                                             p q : Real
                                             hp : LT.lt 1 p
                                             h : Eq q (HDiv.hDiv p (HSub.hSub p 1))
                                             ⊢ Eq (HAdd.hAdd (Inv.inv p) (Inv.inv q)) 1
                                           -/
  ⟨IsConjExponent.conj_eq, fun h ↦ ⟨hp, by field_simp [h]⟩⟩
                                           /-
                                             🎉 no goals
                                           -/


lemma IsConjExponent.conjExponent (h : 1 < p) : p.IsConjExponent (conjExponent p) :=
  (isConjExponent_iff_eq_conjExponent h).2 rfl


lemma isConjExponent_one_div (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
                                         /-
                                           a b : Real
                                           ha : LT.lt 0 a
                                           hb : LT.lt 0 b
                                           hab : Eq (HAdd.hAdd a b) 1
                                           ⊢ (HDiv.hDiv 1 a).IsConjExponent (HDiv.hDiv 1 b)
                                         -/
    (1 / a).IsConjExponent (1 / b) := by simpa using IsConjExponent.inv_inv ha hb hab
                                         /-
                                           🎉 no goals
                                         -/


/-- Two nonnegative real exponents `p, q` are conjugate if they are `> 1` and satisfy the equality
`1/p + 1/q = 1`. This condition shows up in many theorems in analysis, notably related to `L^p`
norms. -/
@[mk_iff]
structure IsConjExponent (p q : ℝ≥0) : Prop where
  one_lt : 1 < p
  inv_add_inv_conj : p⁻¹ + q⁻¹ = 1


/-- The conjugate exponent of `p` is `q = p/(p-1)`, so that `1/p + 1/q = 1`. -/
noncomputable def conjExponent (p : ℝ≥0) : ℝ≥0 := p / (p - 1)


@[simp, norm_cast] lemma isConjExponent_coe : (p : ℝ).IsConjExponent q ↔ p.IsConjExponent q := by
  /-
    p q : NNReal
    ⊢ Iff ((↑p).IsConjExponent ↑q) (p.IsConjExponent q)
  -/
  simp [Real.isConjExponent_iff, isConjExponent_iff]; norm_cast; simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


alias ⟨_, IsConjExponent.coe⟩ := isConjExponent_coe


lemma one_le : 1 ≤ p := h.one_lt.le

lemma pos : 0 < p := zero_lt_one.trans h.one_lt

lemma ne_zero : p ≠ 0 := h.pos.ne'


lemma sub_one_pos : 0 < p - 1 := tsub_pos_of_lt h.one_lt

lemma sub_one_ne_zero : p - 1 ≠ 0 := h.sub_one_pos.ne'


lemma inv_pos : 0 < p⁻¹ := _root_.inv_pos.2 h.pos

lemma inv_ne_zero : p⁻¹ ≠ 0 := h.inv_pos.ne'


lemma one_sub_inv : 1 - p⁻¹ = q⁻¹ := tsub_eq_of_eq_add_rev h.inv_add_inv_conj.symm


lemma conj_eq : q = p / (p - 1) := by
  /-
    p q : NNReal
    h : p.IsConjExponent q
    ⊢ Eq q (HDiv.hDiv p (HSub.hSub p 1))
  -/
  simpa only [← coe_one, ← NNReal.coe_sub h.one_le, ← NNReal.coe_div, coe_inj] using h.coe.conj_eq
  /-
    🎉 no goals
  -/


lemma sub_one_mul_conj : (p - 1) * q = p :=
  mul_comm q (p - 1) ▸ (eq_div_iff h.sub_one_ne_zero).1 h.conj_eq


lemma mul_eq_add : p * q = p + q := by
  /-
    p q : NNReal
    h : p.IsConjExponent q
    ⊢ Eq (HMul.hMul p q) (HAdd.hAdd p q)
  -/
  simpa only [← NNReal.coe_mul, ← NNReal.coe_add, NNReal.coe_inj] using h.coe.mul_eq_add
  /-
    🎉 no goals
  -/


@[symm]
protected lemma symm : q.IsConjExponent p where
  one_lt := by
    /-
      p q : NNReal
      h : p.IsConjExponent q
      ⊢ LT.lt 1 q
    -/
    rw [h.conj_eq]
    /-
      p q : NNReal
      h : p.IsConjExponent q
      ⊢ LT.lt 1 (HDiv.hDiv p (HSub.hSub p 1))
    -/
    exact (one_lt_div h.sub_one_pos).mpr (tsub_lt_self h.pos zero_lt_one)
    /-
      🎉 no goals
    -/
                         /-
                           p q : NNReal
                           h : p.IsConjExponent q
                           ⊢ Eq (HAdd.hAdd (Inv.inv q) (Inv.inv p)) 1
                         -/
  inv_add_inv_conj := by simpa [add_comm] using h.inv_add_inv_conj
                         /-
                           🎉 no goals
                         -/


                                                /-
                                                  p q : NNReal
                                                  h : p.IsConjExponent q
                                                  ⊢ Eq (HDiv.hDiv p q) (HSub.hSub p 1)
                                                -/
lemma div_conj_eq_sub_one : p / q = p - 1 := by field_simp [h.symm.ne_zero]; rw [h.sub_one_mul_conj]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


                                                              /-
                                                                p q : NNReal
                                                                h : p.IsConjExponent q
                                                                ⊢ Eq (HAdd.hAdd ↑(Inv.inv p) ↑(Inv.inv q)) 1
                                                              -/
lemma inv_add_inv_conj_ennreal : (p⁻¹ + q⁻¹ : ℝ≥0∞) = 1 := by norm_cast; exact h.inv_add_inv_conj
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


protected lemma inv_inv (ha : a ≠ 0) (hb : b ≠ 0) (hab : a + b = 1) :
    a⁻¹.IsConjExponent b⁻¹ :=
                                   /-
                                     a b : NNReal
                                     ha : Ne a 0
                                     hb : Ne b 0
                                     hab : Eq (HAdd.hAdd a b) 1
                                     ⊢ LT.lt a 1
                                   -/
  ⟨(one_lt_inv₀ ha.bot_lt).2 <| by rw [← hab]; exact lt_add_of_pos_right _ hb.bot_lt, by
                                               /-
                                                 🎉 no goals
                                               -/
    /-
      a b : NNReal
      ha : Ne a 0
      hb : Ne b 0
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ Eq (HAdd.hAdd (Inv.inv (Inv.inv a)) (Inv.inv (Inv.inv b))) 1
    -/
    simpa only [inv_inv] using hab⟩
    /-
      🎉 no goals
    -/


lemma inv_one_sub_inv (ha₀ : a ≠ 0) (ha₁ : a < 1) : a⁻¹.IsConjExponent (1 - a)⁻¹ :=
  .inv_inv ha₀ (tsub_pos_of_lt ha₁).ne' <| add_tsub_cancel_of_le ha₁.le


lemma one_sub_inv_inv (ha₀ : a ≠ 0) (ha₁ : a < 1) : (1 - a)⁻¹.IsConjExponent a⁻¹ :=
  (inv_one_sub_inv ha₀ ha₁).symm


lemma isConjExponent_iff_eq_conjExponent (h : 1 < p) : p.IsConjExponent q ↔ q = p / (p - 1) := by
  rw [← isConjExponent_coe, Real.isConjExponent_iff_eq_conjExponent (mod_cast h), ← coe_inj,
    NNReal.coe_div, NNReal.coe_sub h.le, coe_one]


protected lemma IsConjExponent.conjExponent (h : 1 < p) : p.IsConjExponent (conjExponent p) :=
  (isConjExponent_iff_eq_conjExponent h).2 rfl


protected lemma Real.IsConjExponent.toNNReal {p q : ℝ} (hpq : p.IsConjExponent q) :
    p.toNNReal.IsConjExponent q.toNNReal where
               /-
                 p q : Real
                 hpq : p.IsConjExponent q
                 ⊢ LT.lt 1 p.toNNReal
               -/
  one_lt := by simpa using hpq.one_lt
               /-
                 🎉 no goals
               -/
  inv_add_inv_conj := by rw [← toNNReal_inv, ← toNNReal_inv, ← toNNReal_add hpq.inv_nonneg
    hpq.symm.inv_nonneg, hpq.inv_add_inv_conj, toNNReal_one]


/-- Two extended nonnegative real exponents `p, q` are conjugate and satisfy the equality
`1/p + 1/q = 1`. This condition shows up in many theorems in analysis, notably related to `L^p`
norms. Note that we permit one of the exponents to be `∞` and the other `1`. -/
@[mk_iff]
structure IsConjExponent (p q : ℝ≥0∞) : Prop where
  inv_add_inv_conj : p⁻¹ + q⁻¹ = 1


/-- The conjugate exponent of `p` is `q = 1 + (p - 1)⁻¹`, so that `1/p + 1/q = 1`. -/
noncomputable def conjExponent (p : ℝ≥0∞) : ℝ≥0∞ := 1 + (p - 1)⁻¹


lemma coe_conjExponent {p : ℝ≥0} (hp : 1 < p) : p.conjExponent = conjExponent p := by
  /-
    p : NNReal
    hp : LT.lt 1 p
    ⊢ Eq (↑p.conjExponent) (↑p).conjExponent
  -/
  rw [NNReal.conjExponent, conjExponent]
  /-
    p : NNReal
    hp : LT.lt 1 p
    ⊢ Eq (↑(HDiv.hDiv p (HSub.hSub p 1))) (HAdd.hAdd 1 (Inv.inv (HSub.hSub (↑p) 1)))
  -/
  norm_cast
  /-
    p : NNReal
    hp : LT.lt 1 p
    ⊢ Eq (↑(HDiv.hDiv p (HSub.hSub p 1))) (HAdd.hAdd 1 (Inv.inv ↑(HSub.hSub p 1)))
  -/
  rw [← coe_inv (tsub_pos_of_lt hp).ne']
  /-
    p : NNReal
    hp : LT.lt 1 p
    ⊢ Eq (↑(HDiv.hDiv p (HSub.hSub p 1))) (HAdd.hAdd 1 ↑(Inv.inv (HSub.hSub p 1)))
  -/
  norm_cast
  /-
    p : NNReal
    hp : LT.lt 1 p
    ⊢ Eq (HDiv.hDiv p (HSub.hSub p 1)) (HAdd.hAdd 1 (Inv.inv (HSub.hSub p 1)))
  -/
  field_simp [(tsub_pos_of_lt hp).ne']
  /-
    p : NNReal
    hp : LT.lt 1 p
    ⊢ Eq p (HAdd.hAdd (HSub.hSub p 1) 1)
  -/
  rw [tsub_add_cancel_of_le hp.le]
  /-
    🎉 no goals
  -/


@[simp, norm_cast] lemma isConjExponent_coe {p q : ℝ≥0} :
    IsConjExponent p q ↔ p.IsConjExponent q := by
  /-
    p q : NNReal
    ⊢ Iff ((↑p).IsConjExponent ↑q) (p.IsConjExponent q)
  -/
  simp only [isConjExponent_iff, NNReal.isConjExponent_iff]
  /-
    p q : NNReal
    ⊢ Iff (Eq (HAdd.hAdd (Inv.inv ↑p) (Inv.inv ↑q)) 1) (And (LT.lt 1 p) (Eq (HAdd. …
  -/
  refine ⟨fun h ↦ ⟨?_, ?_⟩, ?_⟩
    /-
      case refine_1
      p q : NNReal
      h : Eq (HAdd.hAdd (Inv.inv ↑p) (Inv.inv ↑q)) 1
      ⊢ LT.lt 1 p
    -/
  · simpa using (ENNReal.lt_add_right (fun hp ↦ by simp [hp] at h) <| by simp).trans_eq h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      p q : NNReal
      h : Eq (HAdd.hAdd (Inv.inv ↑p) (Inv.inv ↑q)) 1
      ⊢ Eq (HAdd.hAdd (Inv.inv p) (Inv.inv q)) 1
    -/
  · rw [← coe_inv, ← coe_inv] at h
      /-
        case refine_2
        p q : NNReal
        h : Eq (HAdd.hAdd ↑(Inv.inv p) ↑(Inv.inv q)) 1
        ⊢ Eq (HAdd.hAdd (Inv.inv p) (Inv.inv q)) 1
      -/
    · norm_cast at h
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      p q : NNReal
      h : Eq (HAdd.hAdd (↑(Inv.inv p)) (Inv.inv ↑q)) 1
      ⊢ Ne q 0
    -/
    all_goals rintro rfl; simp at h
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      p q : NNReal
      ⊢ And (LT.lt 1 p) (Eq (HAdd.hAdd (Inv.inv p) (Inv.inv q)) 1) → Eq (HAdd.hAdd ( …
    -/
  · rintro ⟨hp, h⟩
    /-
      case refine_3.intro
      p q : NNReal
      hp : LT.lt 1 p
      h : Eq (HAdd.hAdd (Inv.inv p) (Inv.inv q)) 1
      ⊢ Eq (HAdd.hAdd (Inv.inv ↑p) (Inv.inv ↑q)) 1
    -/
    rw [← coe_inv (zero_lt_one.trans hp).ne', ← coe_inv, ← coe_add, h, coe_one]
    /-
      case refine_3.intro
      p q : NNReal
      hp : LT.lt 1 p
      h : Eq (HAdd.hAdd (Inv.inv p) (Inv.inv q)) 1
      ⊢ Ne q 0
    -/
    rintro rfl
    /-
      case refine_3.intro
      p : NNReal
      hp : LT.lt 1 p
      h : Eq (HAdd.hAdd (Inv.inv p) (Inv.inv 0)) 1
      ⊢ False
    -/
    simp [hp.ne'] at h
    /-
      🎉 no goals
    -/


alias ⟨_, _root_.NNReal.IsConjExponent.coe_ennreal⟩ := isConjExponent_coe


protected lemma conjExponent (hp : 1 ≤ p) : p.IsConjExponent (conjExponent p) := by
  /-
    p : ENNReal
    hp : LE.le 1 p
    ⊢ p.IsConjExponent p.conjExponent
  -/
  have : p ≠ 0 := (zero_lt_one.trans_le hp).ne'
  /-
    p : ENNReal
    hp : LE.le 1 p
    this : Ne p 0
    ⊢ p.IsConjExponent p.conjExponent
  -/
  rw [isConjExponent_iff, conjExponent, add_comm]
  /-
    p : ENNReal
    hp : LE.le 1 p
    this : Ne p 0
    ⊢ Eq (HAdd.hAdd (Inv.inv (HAdd.hAdd 1 (Inv.inv (HSub.hSub p 1)))) (Inv.inv p)) 1
  -/
  refine (AddLECancellable.eq_tsub_iff_add_eq_of_le (α := ℝ≥0∞) (by simpa) (by simpa)).1 ?_
  /-
    p : ENNReal
    hp : LE.le 1 p
    this : Ne p 0
    ⊢ Eq (Inv.inv (HAdd.hAdd 1 (Inv.inv (HSub.hSub p 1)))) (HSub.hSub 1 (Inv.inv p))
  -/
  rw [inv_eq_iff_eq_inv]
  /-
    p : ENNReal
    hp : LE.le 1 p
    this : Ne p 0
    ⊢ Eq (HAdd.hAdd 1 (Inv.inv (HSub.hSub p 1))) (Inv.inv (HSub.hSub 1 (Inv.inv p)))
  -/
  obtain rfl | hp₁ := hp.eq_or_lt
    /-
      case inl
      hp : LE.le 1 1
      this : Ne 1 0
      ⊢ Eq (HAdd.hAdd 1 (Inv.inv (HSub.hSub 1 1))) (Inv.inv (HSub.hSub 1 (Inv.inv 1)))
    -/
  · simp [tsub_eq_zero_of_le]
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : ENNReal
    hp : LE.le 1 p
    this : Ne p 0
    hp₁ : LT.lt 1 p
    ⊢ Eq (HAdd.hAdd 1 (Inv.inv (HSub.hSub p 1))) (Inv.inv (HSub.hSub 1 (Inv.inv p)))
  -/
  obtain rfl | hp := eq_or_ne p ∞
    /-
      case inr.inl
      hp : LE.le 1 Top.top
      this : Ne Top.top 0
      hp₁ : LT.lt 1 Top.top
      ⊢ Eq (HAdd.hAdd 1 (Inv.inv (HSub.hSub Top.top 1))) (Inv.inv (HSub.hSub 1 (Inv. …
    -/
  · simp
    /-
      🎉 no goals
    -/
  calc
    1 + (p - 1)⁻¹ = (p - 1 + 1) / (p - 1) := by
      rw [ENNReal.add_div, ENNReal.div_self ((tsub_pos_of_lt hp₁).ne') (sub_ne_top hp), one_div]
    _ = (1 - p⁻¹)⁻¹ := by
      rw [tsub_add_cancel_of_le, ← inv_eq_iff_eq_inv, div_eq_mul_inv, ENNReal.mul_inv, inv_inv,
        ENNReal.mul_sub, ENNReal.inv_mul_cancel, mul_one] <;> simp [*]


@[symm]
protected lemma symm : q.IsConjExponent p where
                         /-
                           p q : ENNReal
                           h : p.IsConjExponent q
                           ⊢ Eq (HAdd.hAdd (Inv.inv q) (Inv.inv p)) 1
                         -/
  inv_add_inv_conj := by simpa [add_comm] using h.inv_add_inv_conj
                         /-
                           🎉 no goals
                         -/


lemma one_le : 1 ≤ p := ENNReal.inv_le_one.1 <| by
  /-
    p q : ENNReal
    h : p.IsConjExponent q
    ⊢ LE.le (Inv.inv p) 1
  -/
  rw [← add_zero p⁻¹, ← h.inv_add_inv_conj]; gcongr; positivity
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma pos : 0 < p := zero_lt_one.trans_le h.one_le

lemma one_sub_inv : 1 - p⁻¹ = q⁻¹ :=
  ENNReal.sub_eq_of_eq_add_rev' one_ne_top h.inv_add_inv_conj.symm


lemma conjExponent_eq : conjExponent p = q := by
  /-
    p q : ENNReal
    h : p.IsConjExponent q
    ⊢ Eq p.conjExponent q
  -/
  have hp : 1 ≤ p := h.one_le
  /-
    p q : ENNReal
    h : p.IsConjExponent q
    hp : LE.le 1 p
    ⊢ Eq p.conjExponent q
  -/
  have : p⁻¹ ≠ ∞ := by simpa using h.ne_zero
  simpa [ENNReal.add_right_inj, *] using
    (IsConjExponent.conjExponent hp).inv_add_inv_conj.trans h.inv_add_inv_conj.symm


lemma conj_eq : q = 1 + (p - 1)⁻¹ := h.conjExponent_eq.symm


lemma mul_eq_add : p * q = p + q := by
  /-
    p q : ENNReal
    h : p.IsConjExponent q
    ⊢ Eq (HMul.hMul p q) (HAdd.hAdd p q)
  -/
  obtain rfl | hp := eq_or_ne p ∞
    /-
      case inl
      q : ENNReal
      h : Top.top.IsConjExponent q
      ⊢ Eq (HMul.hMul Top.top q) (HAdd.hAdd Top.top q)
    -/
  · simp [h.symm.ne_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    p q : ENNReal
    h : p.IsConjExponent q
    hp : Ne p Top.top
    ⊢ Eq (HMul.hMul p q) (HAdd.hAdd p q)
  -/
  obtain rfl | hq := eq_or_ne q ∞
    /-
      case inr.inl
      p : ENNReal
      hp : Ne p Top.top
      h : p.IsConjExponent Top.top
      ⊢ Eq (HMul.hMul p Top.top) (HAdd.hAdd p Top.top)
    -/
  · simp [h.ne_zero]
    /-
      🎉 no goals
    -/
  rw [← mul_one (_ * _), ← h.inv_add_inv_conj, mul_add, mul_right_comm,
    ENNReal.mul_inv_cancel h.ne_zero hp, one_mul, mul_assoc,
    ENNReal.mul_inv_cancel h.symm.ne_zero hq, mul_one, add_comm]


lemma div_conj_eq_sub_one : p / q = p - 1 := by
  /-
    p q : ENNReal
    h : p.IsConjExponent q
    ⊢ Eq (HDiv.hDiv p q) (HSub.hSub p 1)
  -/
  obtain rfl | hq := eq_or_ne q ∞
    /-
      case inl
      p : ENNReal
      h : p.IsConjExponent Top.top
      ⊢ Eq (HDiv.hDiv p Top.top) (HSub.hSub p 1)
    -/
  · simp [h.symm.conj_eq, tsub_eq_zero_of_le]
    /-
      🎉 no goals
    -/
  /-
    case inr
    p q : ENNReal
    h : p.IsConjExponent q
    hq : Ne q Top.top
    ⊢ Eq (HDiv.hDiv p q) (HSub.hSub p 1)
  -/
  refine ENNReal.eq_sub_of_add_eq one_ne_top ?_
  rw [← ENNReal.div_self h.symm.ne_zero hq, ← ENNReal.add_div, ← h.mul_eq_add, mul_div_assoc,
    ENNReal.div_self h.symm.ne_zero hq, mul_one]


protected lemma inv_inv (hab : a + b = 1) : a⁻¹.IsConjExponent b⁻¹ where
                         /-
                           a b : ENNReal
                           hab : Eq (HAdd.hAdd a b) 1
                           ⊢ Eq (HAdd.hAdd (Inv.inv (Inv.inv a)) (Inv.inv (Inv.inv b))) 1
                         -/
  inv_add_inv_conj := by simpa only [inv_inv] using hab
                         /-
                           🎉 no goals
                         -/


lemma inv_one_sub_inv (ha : a ≤ 1) : a⁻¹.IsConjExponent (1 - a)⁻¹ :=
  .inv_inv <| add_tsub_cancel_of_le ha


lemma one_sub_inv_inv (ha : a ≤ 1) : (1 - a)⁻¹.IsConjExponent a⁻¹ := (inv_one_sub_inv ha).symm


                                          /-
                                            ⊢ Eq (HAdd.hAdd (Inv.inv Top.top) (Inv.inv 1)) 1
                                          -/
lemma top_one : IsConjExponent ∞ 1 := ⟨by simp⟩
                                          /-
                                            🎉 no goals
                                          -/

                                          /-
                                            ⊢ Eq (HAdd.hAdd (Inv.inv 1) (Inv.inv Top.top)) 1
                                          -/
lemma one_top : IsConjExponent 1 ∞ := ⟨by simp⟩
                                          /-
                                            🎉 no goals
                                          -/


lemma isConjExponent_iff_eq_conjExponent (hp : 1 ≤ p) : p.IsConjExponent q ↔ q = 1 + (p - 1)⁻¹ :=
                         /-
                           p q : ENNReal
                           hp : LE.le 1 p
                           ⊢ Eq q (HAdd.hAdd 1 (Inv.inv (HSub.hSub p 1))) → p.IsConjExponent q
                         -/
  ⟨fun h ↦ h.conj_eq, by rintro rfl; exact .conjExponent hp⟩
                                     /-
                                       🎉 no goals
                                     -/


