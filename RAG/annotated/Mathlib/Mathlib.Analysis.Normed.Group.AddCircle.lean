instance : NormedAddCommGroup (AddCircle p) :=
  AddSubgroup.normedAddCommGroupQuotient _


@[simp]
theorem norm_coe_mul (x : ℝ) (t : ℝ) :
    ‖(↑(t * x) : AddCircle (t * p))‖ = |t| * ‖(x : AddCircle p)‖ := by
  have aux : ∀ {a b c : ℝ}, a ∈ zmultiples b → c * a ∈ zmultiples (c * b) := fun {a b c} h => by
    simp only [mem_zmultiples_iff] at h ⊢
    obtain ⟨n, rfl⟩ := h
    exact ⟨n, (mul_smul_comm n c b).symm⟩
  /-
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ⊢ Eq (Norm.norm ↑(HMul.hMul t x)) (HMul.hMul (abs t) (Norm.norm ↑x))
  -/
  rcases eq_or_ne t 0 with (rfl | ht); · simp
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ht : Ne t 0
    ⊢ Eq (Norm.norm ↑(HMul.hMul t x)) (HMul.hMul (abs t) (Norm.norm ↑x))
  -/
  have ht' : |t| ≠ 0 := (not_congr abs_eq_zero).mpr ht
  /-
    case inr
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ht : Ne t 0
    ht' : Ne (abs t) 0
    ⊢ Eq (Norm.norm ↑(HMul.hMul t x)) (HMul.hMul (abs t) (Norm.norm ↑x))
  -/
  simp only [quotient_norm_eq, Real.norm_eq_abs]
  /-
    case inr
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ht : Ne t 0
    ht' : Ne (abs t) 0
    ⊢ Eq (InfSet.sInf (Set.image (fun a => abs a) (setOf fun m => Eq ↑m ↑(HMul.hMu …
  -/
  conv_rhs => rw [← smul_eq_mul, ← Real.sInf_smul_of_nonneg (abs_nonneg t)]
  /-
    case inr
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ht : Ne t 0
    ht' : Ne (abs t) 0
    ⊢ Eq (InfSet.sInf (Set.image (fun a => abs a) (setOf fun m => Eq ↑m ↑(HMul.hMu …
  -/
  simp only [QuotientAddGroup.mk'_apply, QuotientAddGroup.eq_iff_sub_mem]
  /-
    case inr
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ht : Ne t 0
    ht' : Ne (abs t) 0
    ⊢ Eq (InfSet.sInf (Set.image (fun a => abs a) (setOf fun m => Membership.mem ( …
  -/
  congr 1
  /-
    case inr.e_a
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ht : Ne t 0
    ht' : Ne (abs t) 0
    ⊢ Eq (Set.image (fun a => abs a) (setOf fun m => Membership.mem (AddSubgroup.z …
  -/
  ext z
  /-
    case inr.e_a.h
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ht : Ne t 0
    ht' : Ne (abs t) 0
    z : Real
    ⊢ Iff (Membership.mem (Set.image (fun a => abs a) (setOf fun m => Membership.m …
  -/
  rw [mem_smul_set_iff_inv_smul_mem₀ ht']
  show
    (∃ y, y - t * x ∈ zmultiples (t * p) ∧ |y| = z) ↔ ∃ w, w - x ∈ zmultiples p ∧ |w| = |t|⁻¹ * z
  /-
    case inr.e_a.h
    p x t : Real
    aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
    ht : Ne t 0
    ht' : Ne (abs t) 0
    z : Real
    ⊢ Iff (Exists fun y => And (Membership.mem (AddSubgroup.zmultiples (HMul.hMul  …
  -/
  constructor
    /-
      case inr.e_a.h.mp
      p x t : Real
      aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
      ht : Ne t 0
      ht' : Ne (abs t) 0
      z : Real
      ⊢ (Exists fun y => And (Membership.mem (AddSubgroup.zmultiples (HMul.hMul t p) …
    -/
  · rintro ⟨y, hy, rfl⟩
    /-
      case inr.e_a.h.mp.intro.intro
      p x t : Real
      aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
      ht : Ne t 0
      ht' : Ne (abs t) 0
      y : Real
      hy : Membership.mem (AddSubgroup.zmultiples (HMul.hMul t p)) (HSub.hSub y (HMu …
      ⊢ Exists fun w => And (Membership.mem (AddSubgroup.zmultiples p) (HSub.hSub w  …
    -/
    refine ⟨t⁻¹ * y, ?_, by rw [abs_mul, abs_inv]⟩
    /-
      case inr.e_a.h.mp.intro.intro
      p x t : Real
      aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
      ht : Ne t 0
      ht' : Ne (abs t) 0
      y : Real
      hy : Membership.mem (AddSubgroup.zmultiples (HMul.hMul t p)) (HSub.hSub y (HMu …
      ⊢ Membership.mem (AddSubgroup.zmultiples p) (HSub.hSub (HMul.hMul (Inv.inv t)  …
    -/
    rw [← inv_mul_cancel_left₀ ht x, ← inv_mul_cancel_left₀ ht p, ← mul_sub]
    /-
      case inr.e_a.h.mp.intro.intro
      p x t : Real
      aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
      ht : Ne t 0
      ht' : Ne (abs t) 0
      y : Real
      hy : Membership.mem (AddSubgroup.zmultiples (HMul.hMul t p)) (HSub.hSub y (HMu …
      ⊢ Membership.mem (AddSubgroup.zmultiples (HMul.hMul (Inv.inv t) (HMul.hMul t p …
    -/
    exact aux hy
    /-
      🎉 no goals
    -/
    /-
      case inr.e_a.h.mpr
      p x t : Real
      aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
      ht : Ne t 0
      ht' : Ne (abs t) 0
      z : Real
      ⊢ (Exists fun w => And (Membership.mem (AddSubgroup.zmultiples p) (HSub.hSub w …
    -/
  · rintro ⟨w, hw, hw'⟩
    /-
      case inr.e_a.h.mpr.intro.intro
      p x t : Real
      aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
      ht : Ne t 0
      ht' : Ne (abs t) 0
      z w : Real
      hw : Membership.mem (AddSubgroup.zmultiples p) (HSub.hSub w x)
      hw' : Eq (abs w) (HMul.hMul (Inv.inv (abs t)) z)
      ⊢ Exists fun y => And (Membership.mem (AddSubgroup.zmultiples (HMul.hMul t p)) …
    -/
    refine ⟨t * w, ?_, by rw [← (eq_inv_mul_iff_mul_eq₀ ht').mp hw', abs_mul]⟩
    /-
      case inr.e_a.h.mpr.intro.intro
      p x t : Real
      aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
      ht : Ne t 0
      ht' : Ne (abs t) 0
      z w : Real
      hw : Membership.mem (AddSubgroup.zmultiples p) (HSub.hSub w x)
      hw' : Eq (abs w) (HMul.hMul (Inv.inv (abs t)) z)
      ⊢ Membership.mem (AddSubgroup.zmultiples (HMul.hMul t p)) (HSub.hSub (HMul.hMu …
    -/
    rw [← mul_sub]
    /-
      case inr.e_a.h.mpr.intro.intro
      p x t : Real
      aux : ∀ {a b c : Real}, Membership.mem (AddSubgroup.zmultiples b) a → Membersh …
      ht : Ne t 0
      ht' : Ne (abs t) 0
      z w : Real
      hw : Membership.mem (AddSubgroup.zmultiples p) (HSub.hSub w x)
      hw' : Eq (abs w) (HMul.hMul (Inv.inv (abs t)) z)
      ⊢ Membership.mem (AddSubgroup.zmultiples (HMul.hMul t p)) (HMul.hMul t (HSub.h …
    -/
    exact aux hw
    /-
      🎉 no goals
    -/


theorem norm_neg_period (x : ℝ) : ‖(x : AddCircle (-p))‖ = ‖(x : AddCircle p)‖ := by
  suffices ‖(↑(-1 * x) : AddCircle (-1 * p))‖ = ‖(x : AddCircle p)‖ by
    rw [← this, neg_one_mul]
    simp
  /-
    p x : Real
    ⊢ Eq (Norm.norm ↑(HMul.hMul (-1) x)) (Norm.norm ↑x)
  -/
  simp only [norm_coe_mul, abs_neg, abs_one, one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_eq_of_zero {x : ℝ} : ‖(x : AddCircle (0 : ℝ))‖ = |x| := by
  suffices { y : ℝ | (y : AddCircle (0 : ℝ)) = (x : AddCircle (0 : ℝ)) } = {x} by
    rw [quotient_norm_eq, this, image_singleton, Real.norm_eq_abs, csInf_singleton]
  /-
    x : Real
    ⊢ Eq (setOf fun y => Eq ↑y ↑x) (Singleton.singleton x)
  -/
  ext y
  /-
    case h
    x y : Real
    ⊢ Iff (Membership.mem (setOf fun y => Eq ↑y ↑x) y) (Membership.mem (Singleton. …
  -/
  simp [QuotientAddGroup.eq_iff_sub_mem, mem_zmultiples_iff, sub_eq_zero]
  /-
    🎉 no goals
  -/


theorem norm_eq {x : ℝ} : ‖(x : AddCircle p)‖ = |x - round (p⁻¹ * x) * p| := by
  suffices ∀ x : ℝ, ‖(x : AddCircle (1 : ℝ))‖ = |x - round x| by
    rcases eq_or_ne p 0 with (rfl | hp)
    · simp
    have hx := norm_coe_mul p x p⁻¹
    rw [abs_inv, eq_inv_mul_iff_mul_eq₀ ((not_congr abs_eq_zero).mpr hp)] at hx
    rw [← hx, inv_mul_cancel₀ hp, this, ← abs_mul, mul_sub, mul_inv_cancel_left₀ hp, mul_comm p]
  /-
    p x : Real
    ⊢ ∀ (x : Real), Eq (Norm.norm ↑x) (abs (HSub.hSub x ↑(round x)))
  -/
  clear! x p
  /-
    ⊢ ∀ (x : Real), Eq (Norm.norm ↑x) (abs (HSub.hSub x ↑(round x)))
  -/
  intros x
  /-
    x : Real
    ⊢ Eq (Norm.norm ↑x) (abs (HSub.hSub x ↑(round x)))
  -/
  rw [quotient_norm_eq, abs_sub_round_eq_min]
  have h₁ : BddBelow (abs '' { m : ℝ | (m : AddCircle (1 : ℝ)) = x }) :=
    ⟨0, by simp [mem_lowerBounds]⟩
  /-
    x : Real
    h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
    ⊢ Eq (InfSet.sInf (Set.image Norm.norm (setOf fun m => Eq ↑m ↑x))) (Min.min (I …
  -/
  have h₂ : (abs '' { m : ℝ | (m : AddCircle (1 : ℝ)) = x }).Nonempty := ⟨|x|, ⟨x, rfl, rfl⟩⟩
  /-
    x : Real
    h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
    h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
    ⊢ Eq (InfSet.sInf (Set.image Norm.norm (setOf fun m => Eq ↑m ↑x))) (Min.min (I …
  -/
  apply le_antisymm
    /-
      case a
      x : Real
      h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
      h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
      ⊢ LE.le (InfSet.sInf (Set.image Norm.norm (setOf fun m => Eq ↑m ↑x))) (Min.min …
    -/
  · simp_rw [Real.norm_eq_abs, csInf_le_iff h₁ h₂, le_min_iff]
    /-
      case a
      x : Real
      h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
      h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
      ⊢ ∀ (b : Real), Membership.mem (lowerBounds (Set.image abs (setOf fun m => Eq  …
    -/
    intro b h
    refine
      ⟨mem_lowerBounds.1 h _ ⟨fract x, ?_, abs_fract⟩,
        mem_lowerBounds.1 h _ ⟨fract x - 1, ?_, by rw [abs_sub_comm, abs_one_sub_fract]⟩⟩
    · simp only [mem_setOf, fract, sub_eq_self, QuotientAddGroup.mk_sub,
        QuotientAddGroup.eq_zero_iff, intCast_mem_zmultiples_one]
    · simp only [mem_setOf, fract, sub_eq_self, QuotientAddGroup.mk_sub,
        QuotientAddGroup.eq_zero_iff, intCast_mem_zmultiples_one, sub_sub,
        (by norm_cast : (⌊x⌋ : ℝ) + 1 = (↑(⌊x⌋ + 1) : ℝ))]
    /-
      case a
      x : Real
      h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
      h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
      ⊢ LE.le (Min.min (Int.fract x) (HSub.hSub 1 (Int.fract x))) (InfSet.sInf (Set. …
    -/
  · simp only [QuotientAddGroup.mk'_apply, Real.norm_eq_abs, le_csInf_iff h₁ h₂]
    /-
      case a
      x : Real
      h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
      h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
      ⊢ ∀ (b : Real), Membership.mem (Set.image abs (setOf fun m => Eq ↑m ↑x)) b → L …
    -/
    rintro b' ⟨b, hb, rfl⟩
    simp only [mem_setOf, QuotientAddGroup.eq_iff_sub_mem, mem_zmultiples_iff,
      smul_one_eq_cast] at hb
    /-
      case a.intro.intro
      x : Real
      h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
      h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
      b : Real
      hb : Exists fun k => Eq (↑k) (HSub.hSub b x)
      ⊢ LE.le (Min.min (Int.fract x) (HSub.hSub 1 (Int.fract x))) (abs b)
    -/
    obtain ⟨z, hz⟩ := hb
    /-
      case a.intro.intro.intro
      x : Real
      h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
      h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
      b : Real
      z : Int
      hz : Eq (↑z) (HSub.hSub b x)
      ⊢ LE.le (Min.min (Int.fract x) (HSub.hSub 1 (Int.fract x))) (abs b)
    -/
    rw [(by rw [hz]; abel : x = b - z), fract_sub_int, ← abs_sub_round_eq_min]
    /-
      case a.intro.intro.intro
      x : Real
      h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
      h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
      b : Real
      z : Int
      hz : Eq (↑z) (HSub.hSub b x)
      ⊢ LE.le (abs (HSub.hSub b ↑(round b))) (abs b)
    -/
    convert round_le b 0
    /-
      case h.e'_4.h.e'_4
      x : Real
      h₁ : BddBelow (Set.image abs (setOf fun m => Eq ↑m ↑x))
      h₂ : (Set.image abs (setOf fun m => Eq ↑m ↑x)).Nonempty
      b : Real
      z : Int
      hz : Eq (↑z) (HSub.hSub b x)
      ⊢ Eq b (HSub.hSub b ↑0)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem norm_eq' (hp : 0 < p) {x : ℝ} : ‖(x : AddCircle p)‖ = p * |p⁻¹ * x - round (p⁻¹ * x)| := by
  conv_rhs =>
    congr
    rw [← abs_eq_self.mpr hp.le]
  /-
    p : Real
    hp : LT.lt 0 p
    x : Real
    ⊢ Eq (Norm.norm ↑x) (HMul.hMul (abs p) (abs (HSub.hSub (HMul.hMul (Inv.inv p)  …
  -/
  rw [← abs_mul, mul_sub, mul_inv_cancel_left₀ hp.ne.symm, norm_eq, mul_comm p]
  /-
    🎉 no goals
  -/


theorem norm_le_half_period {x : AddCircle p} (hp : p ≠ 0) : ‖x‖ ≤ |p| / 2 := by
  /-
    p : Real
    x : AddCircle p
    hp : Ne p 0
    ⊢ LE.le (Norm.norm x) (HDiv.hDiv (abs p) 2)
  -/
  obtain ⟨x⟩ := x
  /-
    case mk
    p : Real
    x✝ : AddCircle p
    hp : Ne p 0
    x : Real
    ⊢ LE.le (Norm.norm (Quot.mk (⇑(QuotientAddGroup.leftRel (AddSubgroup.zmultiple …
  -/
  change ‖(x : AddCircle p)‖ ≤ |p| / 2
  rw [norm_eq, ← mul_le_mul_left (abs_pos.mpr (inv_ne_zero hp)), ← abs_mul, mul_sub, mul_left_comm,
    ← mul_div_assoc, ← abs_mul, inv_mul_cancel₀ hp, mul_one, abs_one]
  /-
    case mk
    p : Real
    x✝ : AddCircle p
    hp : Ne p 0
    x : Real
    ⊢ LE.le (abs (HSub.hSub (HMul.hMul (Inv.inv p) x) ↑(round (HMul.hMul (Inv.inv  …
  -/
  exact abs_sub_round (p⁻¹ * x)
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_half_period_eq : ‖(↑(p / 2) : AddCircle p)‖ = |p| / 2 := by
  /-
    p : Real
    ⊢ Eq (Norm.norm ↑(HDiv.hDiv p 2)) (HDiv.hDiv (abs p) 2)
  -/
  rcases eq_or_ne p 0 with (rfl | hp); · simp
                                         /-
                                           🎉 no goals
                                         -/
  rw [norm_eq, ← mul_div_assoc, inv_mul_cancel₀ hp, one_div, round_two_inv, Int.cast_one,
    one_mul, (by linarith : p / 2 - p = -(p / 2)), abs_neg, abs_div, abs_two]


theorem norm_coe_eq_abs_iff {x : ℝ} (hp : p ≠ 0) : ‖(x : AddCircle p)‖ = |x| ↔ |x| ≤ |p| / 2 := by
  /-
    p x : Real
    hp : Ne p 0
    ⊢ Iff (Eq (Norm.norm ↑x) (abs x)) (LE.le (abs x) (HDiv.hDiv (abs p) 2))
  -/
  refine ⟨fun hx => hx ▸ norm_le_half_period p hp, fun hx => ?_⟩
  suffices ∀ p : ℝ, 0 < p → |x| ≤ p / 2 → ‖(x : AddCircle p)‖ = |x| by
    -- Porting note: replaced `lt_trichotomy` which had trouble substituting `p = 0`.
    rcases hp.symm.lt_or_lt with (hp | hp)
    · rw [abs_eq_self.mpr hp.le] at hx
      exact this p hp hx
    · rw [← norm_neg_period]
      rw [abs_eq_neg_self.mpr hp.le] at hx
      exact this (-p) (neg_pos.mpr hp) hx
  /-
    p x : Real
    hp : Ne p 0
    hx : LE.le (abs x) (HDiv.hDiv (abs p) 2)
    ⊢ ∀ (p : Real), LT.lt 0 p → LE.le (abs x) (HDiv.hDiv p 2) → Eq (Norm.norm ↑x)  …
  -/
  clear hx
  /-
    p x : Real
    hp : Ne p 0
    ⊢ ∀ (p : Real), LT.lt 0 p → LE.le (abs x) (HDiv.hDiv p 2) → Eq (Norm.norm ↑x)  …
  -/
  intro p hp hx
  /-
    p✝ x : Real
    hp✝ : Ne p✝ 0
    p : Real
    hp : LT.lt 0 p
    hx : LE.le (abs x) (HDiv.hDiv p 2)
    ⊢ Eq (Norm.norm ↑x) (abs x)
  -/
  rcases eq_or_ne x (p / (2 : ℝ)) with (rfl | hx')
    /-
      case inl
      p✝ : Real
      hp✝ : Ne p✝ 0
      p : Real
      hp : LT.lt 0 p
      hx : LE.le (abs (HDiv.hDiv p 2)) (HDiv.hDiv p 2)
      ⊢ Eq (Norm.norm ↑(HDiv.hDiv p 2)) (abs (HDiv.hDiv p 2))
    -/
  · simp [abs_div, abs_two]
    /-
      🎉 no goals
    -/
  /-
    case inr
    p✝ x : Real
    hp✝ : Ne p✝ 0
    p : Real
    hp : LT.lt 0 p
    hx : LE.le (abs x) (HDiv.hDiv p 2)
    hx' : Ne x (HDiv.hDiv p 2)
    ⊢ Eq (Norm.norm ↑x) (abs x)
  -/
  suffices round (p⁻¹ * x) = 0 by simp [norm_eq, this]
  /-
    case inr
    p✝ x : Real
    hp✝ : Ne p✝ 0
    p : Real
    hp : LT.lt 0 p
    hx : LE.le (abs x) (HDiv.hDiv p 2)
    hx' : Ne x (HDiv.hDiv p 2)
    ⊢ Eq (round (HMul.hMul (Inv.inv p) x)) 0
  -/
  rw [round_eq_zero_iff]
  /-
    case inr
    p✝ x : Real
    hp✝ : Ne p✝ 0
    p : Real
    hp : LT.lt 0 p
    hx : LE.le (abs x) (HDiv.hDiv p 2)
    hx' : Ne x (HDiv.hDiv p 2)
    ⊢ Membership.mem (Set.Ico (Neg.neg (1 / 2)) (1 / 2)) (HMul.hMul (Inv.inv p) x)
  -/
  obtain ⟨hx₁, hx₂⟩ := abs_le.mp hx
  /-
    case inr.intro
    p✝ x : Real
    hp✝ : Ne p✝ 0
    p : Real
    hp : LT.lt 0 p
    hx : LE.le (abs x) (HDiv.hDiv p 2)
    hx' : Ne x (HDiv.hDiv p 2)
    hx₁ : LE.le (Neg.neg (HDiv.hDiv p 2)) x
    hx₂ : LE.le x (HDiv.hDiv p 2)
    ⊢ Membership.mem (Set.Ico (Neg.neg (1 / 2)) (1 / 2)) (HMul.hMul (Inv.inv p) x)
  -/
  replace hx₂ := Ne.lt_of_le hx' hx₂
  /-
    case inr.intro
    p✝ x : Real
    hp✝ : Ne p✝ 0
    p : Real
    hp : LT.lt 0 p
    hx : LE.le (abs x) (HDiv.hDiv p 2)
    hx' : Ne x (HDiv.hDiv p 2)
    hx₁ : LE.le (Neg.neg (HDiv.hDiv p 2)) x
    hx₂ : LT.lt x (HDiv.hDiv p 2)
    ⊢ Membership.mem (Set.Ico (Neg.neg (1 / 2)) (1 / 2)) (HMul.hMul (Inv.inv p) x)
  -/
  constructor
  · rwa [← mul_le_mul_left hp, ← mul_assoc, mul_inv_cancel₀ hp.ne.symm, one_mul, mul_neg, ←
      mul_div_assoc, mul_one]
  · rwa [← mul_lt_mul_left hp, ← mul_assoc, mul_inv_cancel₀ hp.ne.symm, one_mul, ← mul_div_assoc,
      mul_one]


theorem closedBall_eq_univ_of_half_period_le (hp : p ≠ 0) (x : AddCircle p) {ε : ℝ}
    (hε : |p| / 2 ≤ ε) : closedBall x ε = univ :=
  eq_univ_iff_forall.mpr fun x => by
    /-
      p : Real
      hp : Ne p 0
      x✝ : AddCircle p
      ε : Real
      hε : LE.le (HDiv.hDiv (abs p) 2) ε
      x : AddCircle p
      ⊢ Membership.mem (Metric.closedBall x✝ ε) x
    -/
    simpa only [mem_closedBall, dist_eq_norm] using (norm_le_half_period p hp).trans hε
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_real_preimage_closedBall_period_zero (x ε : ℝ) :
    (↑) ⁻¹' closedBall (x : AddCircle (0 : ℝ)) ε = closedBall x ε := by
  /-
    x ε : Real
    ⊢ Eq (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) ε)) (Metric.clo …
  -/
  ext y
  -- Porting note: squeezed the simp
  simp only [Set.mem_preimage, dist_eq_norm, AddCircle.norm_eq_of_zero, iff_self,
    ← QuotientAddGroup.mk_sub, Metric.mem_closedBall, Real.norm_eq_abs]


theorem coe_real_preimage_closedBall_eq_iUnion (x ε : ℝ) :
    (↑) ⁻¹' closedBall (x : AddCircle p) ε = ⋃ z : ℤ, closedBall (x + z • p) ε := by
  /-
    p x ε : Real
    ⊢ Eq (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) ε)) (Set.iUnion …
  -/
  rcases eq_or_ne p 0 with (rfl | hp)
    /-
      case inl
      x ε : Real
      ⊢ Eq (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) ε)) (Set.iUnion …
    -/
  · simp [iUnion_const]
    /-
      🎉 no goals
    -/
  /-
    case inr
    p x ε : Real
    hp : Ne p 0
    ⊢ Eq (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) ε)) (Set.iUnion …
  -/
  ext y
  simp only [dist_eq_norm, mem_preimage, mem_closedBall, zsmul_eq_mul, mem_iUnion, Real.norm_eq_abs,
    ← QuotientAddGroup.mk_sub, norm_eq, ← sub_sub]
  /-
    case inr.h
    p x ε : Real
    hp : Ne p 0
    y : Real
    ⊢ Iff (LE.le (abs (HSub.hSub (HSub.hSub y x) (HMul.hMul (↑(round (HMul.hMul (I …
  -/
  refine ⟨fun h => ⟨round (p⁻¹ * (y - x)), h⟩, ?_⟩
  /-
    case inr.h
    p x ε : Real
    hp : Ne p 0
    y : Real
    ⊢ (Exists fun i => LE.le (abs (HSub.hSub (HSub.hSub y x) (HMul.hMul (↑i) p)))  …
  -/
  rintro ⟨n, hn⟩
  rw [← mul_le_mul_left (abs_pos.mpr <| inv_ne_zero hp), ← abs_mul, mul_sub, mul_comm _ p,
    inv_mul_cancel_left₀ hp] at hn ⊢
  /-
    case inr.h.intro
    p x ε : Real
    hp : Ne p 0
    y : Real
    n : Int
    hn : LE.le (abs (HSub.hSub (HMul.hMul (Inv.inv p) (HSub.hSub y x)) ↑n)) (HMul. …
    ⊢ LE.le (abs (HSub.hSub (HMul.hMul (Inv.inv p) (HSub.hSub y x)) ↑(round (HMul. …
  -/
  exact (round_le (p⁻¹ * (y - x)) n).trans hn
  /-
    🎉 no goals
  -/


theorem coe_real_preimage_closedBall_inter_eq {x ε : ℝ} (s : Set ℝ)
    (hs : s ⊆ closedBall x (|p| / 2)) :
    (↑) ⁻¹' closedBall (x : AddCircle p) ε ∩ s = if ε < |p| / 2 then closedBall x ε ∩ s else s := by
  /-
    p x ε : Real
    s : Set Real
    hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs p) 2))
    ⊢ Eq (Inter.inter (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) ε) …
  -/
  rcases le_or_lt (|p| / 2) ε with hε | hε
    /-
      case inl
      p x ε : Real
      s : Set Real
      hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs p) 2))
      hε : LE.le (HDiv.hDiv (abs p) 2) ε
      ⊢ Eq (Inter.inter (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) ε) …
    -/
  · rcases eq_or_ne p 0 with (rfl | hp)
      /-
        case inl.inl
        x ε : Real
        s : Set Real
        hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs 0) 2))
        hε : LE.le (HDiv.hDiv (abs 0) 2) ε
        ⊢ Eq (Inter.inter (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) ε) …
      -/
    · simp only [abs_zero, zero_div] at hε
      simp only [not_lt.mpr hε, coe_real_preimage_closedBall_period_zero, abs_zero, zero_div,
        if_false, inter_eq_right]
      /-
        case inl.inl
        x ε : Real
        s : Set Real
        hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs 0) 2))
        hε : LE.le 0 ε
        ⊢ HasSubset.Subset s (Metric.closedBall x ε)
      -/
      exact hs.trans (closedBall_subset_closedBall <| by simp [hε])
      /-
        🎉 no goals
      -/
    -- Porting note: was
    -- simp [closedBall_eq_univ_of_half_period_le p hp (↑x) hε, not_lt.mpr hε]
    /-
      case inl.inr
      p x ε : Real
      s : Set Real
      hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs p) 2))
      hε : LE.le (HDiv.hDiv (abs p) 2) ε
      hp : Ne p 0
      ⊢ Eq (Inter.inter (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) ε) …
    -/
    simp only [not_lt.mpr hε, ite_false, inter_eq_right]
    /-
      case inl.inr
      p x ε : Real
      s : Set Real
      hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs p) 2))
      hε : LE.le (HDiv.hDiv (abs p) 2) ε
      hp : Ne p 0
      ⊢ HasSubset.Subset s (Set.preimage QuotientAddGroup.mk (Metric.closedBall (↑x) …
    -/
    rw [closedBall_eq_univ_of_half_period_le p hp (↑x : ℝ ⧸ zmultiples p) hε, preimage_univ]
    /-
      case inl.inr
      p x ε : Real
      s : Set Real
      hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs p) 2))
      hε : LE.le (HDiv.hDiv (abs p) 2) ε
      hp : Ne p 0
      ⊢ HasSubset.Subset s Set.univ
    -/
    apply subset_univ
    /-
      🎉 no goals
    -/
  · suffices ∀ z : ℤ, closedBall (x + z • p) ε ∩ s = if z = 0 then closedBall x ε ∩ s else ∅ by
      simp [-zsmul_eq_mul, ← QuotientAddGroup.mk_zero, coe_real_preimage_closedBall_eq_iUnion,
        iUnion_inter, iUnion_ite, this, hε]
    /-
      case inr
      p x ε : Real
      s : Set Real
      hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs p) 2))
      hε : LT.lt ε (HDiv.hDiv (abs p) 2)
      ⊢ ∀ (z : Int), Eq (Inter.inter (Metric.closedBall (HAdd.hAdd x (HSMul.hSMul z  …
    -/
    intro z
    /-
      case inr
      p x ε : Real
      s : Set Real
      hs : HasSubset.Subset s (Metric.closedBall x (HDiv.hDiv (abs p) 2))
      hε : LT.lt ε (HDiv.hDiv (abs p) 2)
      z : Int
      ⊢ Eq (Inter.inter (Metric.closedBall (HAdd.hAdd x (HSMul.hSMul z p)) ε) s) (it …
    -/
    simp only [Real.closedBall_eq_Icc, zero_sub, zero_add] at hs ⊢
    /-
      case inr
      p x ε : Real
      s : Set Real
      hε : LT.lt ε (HDiv.hDiv (abs p) 2)
      z : Int
      hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
      ⊢ Eq (Inter.inter (Set.Icc (HSub.hSub (HAdd.hAdd x (HSMul.hSMul z p)) ε) (HAdd …
    -/
    rcases eq_or_ne z 0 with (rfl | hz)
      /-
        case inr.inl
        p x ε : Real
        s : Set Real
        hε : LT.lt ε (HDiv.hDiv (abs p) 2)
        hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
        ⊢ Eq (Inter.inter (Set.Icc (HSub.hSub (HAdd.hAdd x (HSMul.hSMul 0 p)) ε) (HAdd …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      p x ε : Real
      s : Set Real
      hε : LT.lt ε (HDiv.hDiv (abs p) 2)
      z : Int
      hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
      hz : Ne z 0
      ⊢ Eq (Inter.inter (Set.Icc (HSub.hSub (HAdd.hAdd x (HSMul.hSMul z p)) ε) (HAdd …
    -/
    simp only [hz, zsmul_eq_mul, if_false, eq_empty_iff_forall_not_mem]
    /-
      case inr.inr
      p x ε : Real
      s : Set Real
      hε : LT.lt ε (HDiv.hDiv (abs p) 2)
      z : Int
      hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
      hz : Ne z 0
      ⊢ ∀ (x_1 : Real), Not (Membership.mem (Inter.inter (Set.Icc (HSub.hSub (HAdd.h …
    -/
    rintro y ⟨⟨hy₁, hy₂⟩, hy₀⟩
    /-
      case inr.inr.intro.intro
      p x ε : Real
      s : Set Real
      hε : LT.lt ε (HDiv.hDiv (abs p) 2)
      z : Int
      hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
      hz : Ne z 0
      y : Real
      hy₀ : Membership.mem s y
      hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
      hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
      ⊢ False
    -/
    obtain ⟨hy₃, hy₄⟩ := hs hy₀
    /-
      case inr.inr.intro.intro.intro
      p x ε : Real
      s : Set Real
      hε : LT.lt ε (HDiv.hDiv (abs p) 2)
      z : Int
      hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
      hz : Ne z 0
      y : Real
      hy₀ : Membership.mem s y
      hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
      hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
      hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
      hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
      ⊢ False
    -/
    rcases lt_trichotomy 0 p with (hp | (rfl : 0 = p) | hp)
      /-
        case inr.inr.intro.intro.intro.inl
        p x ε : Real
        s : Set Real
        hε : LT.lt ε (HDiv.hDiv (abs p) 2)
        z : Int
        hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
        hz : Ne z 0
        y : Real
        hy₀ : Membership.mem s y
        hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
        hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
        hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
        hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
        hp : LT.lt 0 p
        ⊢ False
      -/
    · cases' Int.cast_le_neg_one_or_one_le_cast_of_ne_zero ℝ hz with hz' hz'
        /-
          case inr.inr.intro.intro.intro.inl.inl
          p x ε : Real
          s : Set Real
          hε : LT.lt ε (HDiv.hDiv (abs p) 2)
          z : Int
          hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
          hz : Ne z 0
          y : Real
          hy₀ : Membership.mem s y
          hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
          hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
          hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
          hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
          hp : LT.lt 0 p
          hz' : LE.le (↑z) (-1)
          ⊢ False
        -/
      · have : ↑z * p ≤ -p := by nlinarith
        /-
          case inr.inr.intro.intro.intro.inl.inl
          p x ε : Real
          s : Set Real
          hε : LT.lt ε (HDiv.hDiv (abs p) 2)
          z : Int
          hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
          hz : Ne z 0
          y : Real
          hy₀ : Membership.mem s y
          hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
          hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
          hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
          hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
          hp : LT.lt 0 p
          hz' : LE.le (↑z) (-1)
          this : LE.le (HMul.hMul (↑z) p) (Neg.neg p)
          ⊢ False
        -/
        linarith [abs_eq_self.mpr hp.le]
        /-
          🎉 no goals
        -/
        /-
          case inr.inr.intro.intro.intro.inl.inr
          p x ε : Real
          s : Set Real
          hε : LT.lt ε (HDiv.hDiv (abs p) 2)
          z : Int
          hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
          hz : Ne z 0
          y : Real
          hy₀ : Membership.mem s y
          hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
          hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
          hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
          hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
          hp : LT.lt 0 p
          hz' : LE.le 1 ↑z
          ⊢ False
        -/
      · have : p ≤ ↑z * p := by nlinarith
        /-
          case inr.inr.intro.intro.intro.inl.inr
          p x ε : Real
          s : Set Real
          hε : LT.lt ε (HDiv.hDiv (abs p) 2)
          z : Int
          hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
          hz : Ne z 0
          y : Real
          hy₀ : Membership.mem s y
          hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
          hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
          hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
          hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
          hp : LT.lt 0 p
          hz' : LE.le 1 ↑z
          this : LE.le p (HMul.hMul (↑z) p)
          ⊢ False
        -/
        linarith [abs_eq_self.mpr hp.le]
        /-
          🎉 no goals
        -/
      /-
        case inr.inr.intro.intro.intro.inr.inl
        x ε : Real
        s : Set Real
        z : Int
        hz : Ne z 0
        y : Real
        hy₀ : Membership.mem s y
        hε : LT.lt ε (HDiv.hDiv (abs 0) 2)
        hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs 0) 2)) (HAdd.hAd …
        hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) 0)) ε) y
        hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) 0)) ε)
        hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs 0) 2)) y
        hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs 0) 2))
        ⊢ False
      -/
    · simp only [mul_zero, add_zero, abs_zero, zero_div] at hy₁ hy₂ hε
      /-
        case inr.inr.intro.intro.intro.inr.inl
        x ε : Real
        s : Set Real
        z : Int
        hz : Ne z 0
        y : Real
        hy₀ : Membership.mem s y
        hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs 0) 2)) (HAdd.hAd …
        hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs 0) 2)) y
        hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs 0) 2))
        hy₁ : LE.le (HSub.hSub x ε) y
        hy₂ : LE.le y (HAdd.hAdd x ε)
        hε : LT.lt ε 0
        ⊢ False
      -/
      linarith
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.intro.intro.intro.inr.inr
        p x ε : Real
        s : Set Real
        hε : LT.lt ε (HDiv.hDiv (abs p) 2)
        z : Int
        hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
        hz : Ne z 0
        y : Real
        hy₀ : Membership.mem s y
        hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
        hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
        hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
        hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
        hp : LT.lt p 0
        ⊢ False
      -/
    · cases' Int.cast_le_neg_one_or_one_le_cast_of_ne_zero ℝ hz with hz' hz'
        /-
          case inr.inr.intro.intro.intro.inr.inr.inl
          p x ε : Real
          s : Set Real
          hε : LT.lt ε (HDiv.hDiv (abs p) 2)
          z : Int
          hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
          hz : Ne z 0
          y : Real
          hy₀ : Membership.mem s y
          hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
          hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
          hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
          hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
          hp : LT.lt p 0
          hz' : LE.le (↑z) (-1)
          ⊢ False
        -/
      · have : -p ≤ ↑z * p := by nlinarith
        /-
          case inr.inr.intro.intro.intro.inr.inr.inl
          p x ε : Real
          s : Set Real
          hε : LT.lt ε (HDiv.hDiv (abs p) 2)
          z : Int
          hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
          hz : Ne z 0
          y : Real
          hy₀ : Membership.mem s y
          hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
          hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
          hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
          hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
          hp : LT.lt p 0
          hz' : LE.le (↑z) (-1)
          this : LE.le (Neg.neg p) (HMul.hMul (↑z) p)
          ⊢ False
        -/
        linarith [abs_eq_neg_self.mpr hp.le]
        /-
          🎉 no goals
        -/
        /-
          case inr.inr.intro.intro.intro.inr.inr.inr
          p x ε : Real
          s : Set Real
          hε : LT.lt ε (HDiv.hDiv (abs p) 2)
          z : Int
          hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
          hz : Ne z 0
          y : Real
          hy₀ : Membership.mem s y
          hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
          hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
          hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
          hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
          hp : LT.lt p 0
          hz' : LE.le 1 ↑z
          ⊢ False
        -/
      · have : ↑z * p ≤ p := by nlinarith
        /-
          case inr.inr.intro.intro.intro.inr.inr.inr
          p x ε : Real
          s : Set Real
          hε : LT.lt ε (HDiv.hDiv (abs p) 2)
          z : Int
          hs : HasSubset.Subset s (Set.Icc (HSub.hSub x (HDiv.hDiv (abs p) 2)) (HAdd.hAd …
          hz : Ne z 0
          y : Real
          hy₀ : Membership.mem s y
          hy₁ : LE.le (HSub.hSub (HAdd.hAdd x (HMul.hMul (↑z) p)) ε) y
          hy₂ : LE.le y (HAdd.hAdd (HAdd.hAdd x (HMul.hMul (↑z) p)) ε)
          hy₃ : LE.le (HSub.hSub x (HDiv.hDiv (abs p) 2)) y
          hy₄ : LE.le y (HAdd.hAdd x (HDiv.hDiv (abs p) 2))
          hp : LT.lt p 0
          hz' : LE.le 1 ↑z
          this : LE.le (HMul.hMul (↑z) p) p
          ⊢ False
        -/
        linarith [abs_eq_neg_self.mpr hp.le]
        /-
          🎉 no goals
        -/


theorem norm_div_natCast {m n : ℕ} :
    ‖(↑(↑m / ↑n * p) : AddCircle p)‖ = p * (↑(min (m % n) (n - m % n)) / n) := by
  /-
    p : Real
    hp : Fact (LT.lt 0 p)
    m n : Nat
    ⊢ Eq (Norm.norm ↑(HMul.hMul (HDiv.hDiv ↑m ↑n) p)) (HMul.hMul p (HDiv.hDiv ↑(Mi …
  -/
  have : p⁻¹ * (↑m / ↑n * p) = ↑m / ↑n := by rw [mul_comm _ p, inv_mul_cancel_left₀ hp.out.ne.symm]
  /-
    p : Real
    hp : Fact (LT.lt 0 p)
    m n : Nat
    this : Eq (HMul.hMul (Inv.inv p) (HMul.hMul (HDiv.hDiv ↑m ↑n) p)) (HDiv.hDiv ↑ …
    ⊢ Eq (Norm.norm ↑(HMul.hMul (HDiv.hDiv ↑m ↑n) p)) (HMul.hMul p (HDiv.hDiv ↑(Mi …
  -/
  rw [norm_eq' p hp.out, this, abs_sub_round_div_natCast_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias norm_div_nat_cast := norm_div_natCast


theorem exists_norm_eq_of_isOfFinAddOrder {u : AddCircle p} (hu : IsOfFinAddOrder u) :
    ∃ k : ℕ, ‖u‖ = p * (k / addOrderOf u) := by
  /-
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    hu : IsOfFinAddOrder u
    ⊢ Exists fun k => Eq (Norm.norm u) (HMul.hMul p (HDiv.hDiv ↑k ↑(addOrderOf u)))
  -/
  let n := addOrderOf u
  /-
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    hu : IsOfFinAddOrder u
    n : Nat := addOrderOf u
    ⊢ Exists fun k => Eq (Norm.norm u) (HMul.hMul p (HDiv.hDiv ↑k ↑(addOrderOf u)))
  -/
  change ∃ k : ℕ, ‖u‖ = p * (k / n)
  /-
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    hu : IsOfFinAddOrder u
    n : Nat := addOrderOf u
    ⊢ Exists fun k => Eq (Norm.norm u) (HMul.hMul p (HDiv.hDiv ↑k ↑n))
  -/
  obtain ⟨m, -, -, hm⟩ := exists_gcd_eq_one_of_isOfFinAddOrder hu
  /-
    case intro.intro.intro
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    hu : IsOfFinAddOrder u
    n : Nat := addOrderOf u
    m : Nat
    hm : Eq (↑(HMul.hMul (HDiv.hDiv ↑m ↑(addOrderOf u)) p)) u
    ⊢ Exists fun k => Eq (Norm.norm u) (HMul.hMul p (HDiv.hDiv ↑k ↑n))
  -/
  refine ⟨min (m % n) (n - m % n), ?_⟩
  /-
    case intro.intro.intro
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    hu : IsOfFinAddOrder u
    n : Nat := addOrderOf u
    m : Nat
    hm : Eq (↑(HMul.hMul (HDiv.hDiv ↑m ↑(addOrderOf u)) p)) u
    ⊢ Eq (Norm.norm u) (HMul.hMul p (HDiv.hDiv ↑(Min.min (HMod.hMod m n) (HSub.hSu …
  -/
  rw [← hm, norm_div_natCast]
  /-
    🎉 no goals
  -/


theorem le_add_order_smul_norm_of_isOfFinAddOrder {u : AddCircle p} (hu : IsOfFinAddOrder u)
    (hu' : u ≠ 0) : p ≤ addOrderOf u • ‖u‖ := by
  /-
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    hu : IsOfFinAddOrder u
    hu' : Ne u 0
    ⊢ LE.le p (HSMul.hSMul (addOrderOf u) (Norm.norm u))
  -/
  obtain ⟨n, hn⟩ := exists_norm_eq_of_isOfFinAddOrder hu
  replace hu : (addOrderOf u : ℝ) ≠ 0 := by
    norm_cast
    exact (addOrderOf_pos_iff.mpr hu).ne'
  /-
    case intro
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    hu' : Ne u 0
    n : Nat
    hn : Eq (Norm.norm u) (HMul.hMul p (HDiv.hDiv ↑n ↑(addOrderOf u)))
    hu : Ne (↑(addOrderOf u)) 0
    ⊢ LE.le p (HSMul.hSMul (addOrderOf u) (Norm.norm u))
  -/
  conv_lhs => rw [← mul_one p]
  rw [hn, nsmul_eq_mul, ← mul_assoc, mul_comm _ p, mul_assoc, mul_div_cancel₀ _ hu,
    mul_le_mul_left hp.out, Nat.one_le_cast, Nat.one_le_iff_ne_zero]
  /-
    case intro
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    hu' : Ne u 0
    n : Nat
    hn : Eq (Norm.norm u) (HMul.hMul p (HDiv.hDiv ↑n ↑(addOrderOf u)))
    hu : Ne (↑(addOrderOf u)) 0
    ⊢ Ne n 0
  -/
  contrapose! hu'
  /-
    case intro
    p : Real
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    n : Nat
    hn : Eq (Norm.norm u) (HMul.hMul p (HDiv.hDiv ↑n ↑(addOrderOf u)))
    hu : Ne (↑(addOrderOf u)) 0
    hu' : Eq n 0
    ⊢ Eq u 0
  -/
  simpa only [hu', Nat.cast_zero, zero_div, mul_zero, norm_eq_zero] using hn
  /-
    🎉 no goals
  -/


                                                                      /-
                                                                        x : Real
                                                                        ⊢ Eq (Norm.norm ↑x) (abs (HSub.hSub x ↑(round x)))
                                                                      -/
theorem norm_eq {x : ℝ} : ‖(x : UnitAddCircle)‖ = |x - round x| := by simp [AddCircle.norm_eq]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


