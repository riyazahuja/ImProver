/-- If `r` is a positive constant, `fun x ↦ r * f x` tends to infinity along a filter
if and only if `f` tends to infinity along the same filter. -/
theorem tendsto_const_mul_atTop_of_pos (hr : 0 < r) :
    Tendsto (fun x => r * f x) l atTop ↔ Tendsto f l atTop :=
  ⟨fun h => h.atTop_of_const_mul hr, fun h =>
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      inst✝ : LinearOrderedSemifield α
                                                      l : Filter β
                                                      f : β → α
                                                      r : α
                                                      hr : LT.lt 0 r
                                                      h : Filter.Tendsto f l Filter.atTop
                                                      ⊢ Filter.Tendsto (fun x => HMul.hMul (Inv.inv r) (HMul.hMul r (f x))) l Filter …
                                                    -/
    Tendsto.atTop_of_const_mul (inv_pos.2 hr) <| by simpa only [inv_mul_cancel_left₀ hr.ne'] ⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- If `r` is a positive constant, `fun x ↦ f x * r` tends to infinity along a filter
if and only if `f` tends to infinity along the same filter. -/
theorem tendsto_mul_const_atTop_of_pos (hr : 0 < r) :
    Tendsto (fun x => f x * r) l atTop ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedSemifield α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt 0 r
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atTop) (Filter.Ten …
  -/
  simpa only [mul_comm] using tendsto_const_mul_atTop_of_pos hr
  /-
    🎉 no goals
  -/


/-- If `r` is a positive constant, `x ↦ f x / r` tends to infinity along a filter
if and only if `f` tends to infinity along the same filter. -/
lemma tendsto_div_const_atTop_of_pos (hr : 0 < r) :
    Tendsto (fun x ↦ f x / r) l atTop ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedSemifield α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt 0 r
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atTop) (Filter.Ten …
  -/
  simpa only [div_eq_mul_inv] using tendsto_mul_const_atTop_of_pos (inv_pos.2 hr)
  /-
    🎉 no goals
  -/


/-- If `f` tends to infinity along a nontrivial filter `l`, then
`fun x ↦ r * f x` tends to infinity if and only if `0 < r. `-/
theorem tendsto_const_mul_atTop_iff_pos [NeBot l] (h : Tendsto f l atTop) :
    Tendsto (fun x => r * f x) l atTop ↔ 0 < r := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atTop
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop) (LT.lt 0 r)
  -/
  refine ⟨fun hrf => not_le.mp fun hr => ?_, fun hr => (tendsto_const_mul_atTop_of_pos hr).mpr h⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atTop
    hrf : Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop
    hr : LE.le r 0
    ⊢ False
  -/
  rcases ((h.eventually_ge_atTop 0).and (hrf.eventually_gt_atTop 0)).exists with ⟨x, hx, hrx⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atTop
    hrf : Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop
    hr : LE.le r 0
    x : β
    hx : LE.le 0 (f x)
    hrx : LT.lt 0 (HMul.hMul r (f x))
    ⊢ False
  -/
  exact (mul_nonpos_of_nonpos_of_nonneg hr hx).not_lt hrx
  /-
    🎉 no goals
  -/


/-- If `f` tends to infinity along a nontrivial filter `l`, then
`fun x ↦ f x * r` tends to infinity if and only if `0 < r. `-/
theorem tendsto_mul_const_atTop_iff_pos [NeBot l] (h : Tendsto f l atTop) :
    Tendsto (fun x => f x * r) l atTop ↔ 0 < r := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atTop
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atTop) (LT.lt 0 r)
  -/
  simp only [mul_comm _ r, tendsto_const_mul_atTop_iff_pos h]
  /-
    🎉 no goals
  -/


/-- If `f` tends to infinity along a nontrivial filter `l`, then
`x ↦ f x * r` tends to infinity if and only if `0 < r. `-/
lemma tendsto_div_const_atTop_iff_pos [NeBot l] (h : Tendsto f l atTop) :
    Tendsto (fun x ↦ f x / r) l atTop ↔ 0 < r := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedSemifield α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atTop
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atTop) (LT.lt 0 r)
  -/
  simp only [div_eq_mul_inv, tendsto_mul_const_atTop_iff_pos h, inv_pos]
  /-
    🎉 no goals
  -/


/-- If `f` tends to infinity along a filter, then `f` multiplied by a positive
constant (on the left) also tends to infinity. For a version working in `ℕ` or `ℤ`, use
`Filter.Tendsto.const_mul_atTop'` instead. -/
theorem Tendsto.const_mul_atTop (hr : 0 < r) (hf : Tendsto f l atTop) :
    Tendsto (fun x => r * f x) l atTop :=
  (tendsto_const_mul_atTop_of_pos hr).2 hf


/-- If a function `f` tends to infinity along a filter, then `f` multiplied by a positive
constant (on the right) also tends to infinity. For a version working in `ℕ` or `ℤ`, use
`Filter.Tendsto.atTop_mul_const'` instead. -/
theorem Tendsto.atTop_mul_const (hr : 0 < r) (hf : Tendsto f l atTop) :
    Tendsto (fun x => f x * r) l atTop :=
  (tendsto_mul_const_atTop_of_pos hr).2 hf


/-- If a function `f` tends to infinity along a filter, then `f` divided by a positive
constant also tends to infinity. -/
theorem Tendsto.atTop_div_const (hr : 0 < r) (hf : Tendsto f l atTop) :
    Tendsto (fun x => f x / r) l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedSemifield α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt 0 r
    hf : Filter.Tendsto f l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atTop
  -/
  simpa only [div_eq_mul_inv] using hf.atTop_mul_const (inv_pos.2 hr)
  /-
    🎉 no goals
  -/


theorem tendsto_const_mul_pow_atTop (hn : n ≠ 0) (hc : 0 < c) :
    Tendsto (fun x => c * x ^ n) atTop atTop :=
  Tendsto.const_mul_atTop hc (tendsto_pow_atTop hn)


theorem tendsto_const_mul_pow_atTop_iff :
    Tendsto (fun x => c * x ^ n) atTop atTop ↔ n ≠ 0 ∧ 0 < c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    c : α
    n : Nat
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop Filt …
  -/
  refine ⟨fun h => ⟨?_, ?_⟩, fun h => tendsto_const_mul_pow_atTop h.1 h.2⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      c : α
      n : Nat
      h : Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop Filter. …
      ⊢ Ne n 0
    -/
  · rintro rfl
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      c : α
      h : Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x 0)) Filter.atTop Filter. …
      ⊢ False
    -/
    simp only [pow_zero, not_tendsto_const_atTop] at h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      c : α
      n : Nat
      h : Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop Filter. …
      ⊢ LT.lt 0 c
    -/
  · rcases ((h.eventually_gt_atTop 0).and (eventually_ge_atTop 0)).exists with ⟨k, hck, hk⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      inst✝ : LinearOrderedSemifield α
      c : α
      n : Nat
      h : Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop Filter. …
      k : α
      hck : LT.lt 0 (HMul.hMul c (HPow.hPow k n))
      hk : LE.le 0 k
      ⊢ LT.lt 0 c
    -/
    exact pos_of_mul_pos_left hck (pow_nonneg hk _)
    /-
      🎉 no goals
    -/


lemma tendsto_zpow_atTop_atTop {n : ℤ} (hn : 0 < n) : Tendsto (fun x : α ↦ x ^ n) atTop atTop := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedSemifield α
    n : Int
    hn : LT.lt 0 n
    ⊢ Filter.Tendsto (fun x => HPow.hPow x n) Filter.atTop Filter.atTop
  -/
  lift n to ℕ using hn.le; simp [(Int.ofNat_pos.mp hn).ne']
                           /-
                             🎉 no goals
                           -/


/-- If `r` is a positive constant, `fun x ↦ r * f x` tends to negative infinity along a filter
if and only if `f` tends to negative infinity along the same filter. -/
theorem tendsto_const_mul_atBot_of_pos (hr : 0 < r) :
    Tendsto (fun x => r * f x) l atBot ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt 0 r
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atBot) (Filter.Ten …
  -/
  simpa only [← mul_neg, ← tendsto_neg_atTop_iff] using tendsto_const_mul_atTop_of_pos hr
  /-
    🎉 no goals
  -/


/-- If `r` is a positive constant, `fun x ↦ f x * r` tends to negative infinity along a filter
if and only if `f` tends to negative infinity along the same filter. -/
theorem tendsto_mul_const_atBot_of_pos (hr : 0 < r) :
    Tendsto (fun x => f x * r) l atBot ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt 0 r
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atBot) (Filter.Ten …
  -/
  simpa only [mul_comm] using tendsto_const_mul_atBot_of_pos hr
  /-
    🎉 no goals
  -/


/-- If `r` is a positive constant, `fun x ↦ f x / r` tends to negative infinity along a filter
if and only if `f` tends to negative infinity along the same filter. -/
lemma tendsto_div_const_atBot_of_pos (hr : 0 < r) :
    Tendsto (fun x ↦ f x / r) l atBot ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt 0 r
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atBot) (Filter.Ten …
  -/
  simp [div_eq_mul_inv, tendsto_mul_const_atBot_of_pos, hr]
  /-
    🎉 no goals
  -/


/-- If `r` is a negative constant, `fun x ↦ r * f x` tends to infinity along a filter `l`
if and only if `f` tends to negative infinity along `l`. -/
theorem tendsto_const_mul_atTop_of_neg (hr : r < 0) :
    Tendsto (fun x => r * f x) l atTop ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt r 0
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop) (Filter.Ten …
  -/
  simpa only [neg_mul, tendsto_neg_atBot_iff] using tendsto_const_mul_atBot_of_pos (neg_pos.2 hr)
  /-
    🎉 no goals
  -/


/-- If `r` is a negative constant, `fun x ↦ f x * r` tends to infinity along a filter `l`
if and only if `f` tends to negative infinity along `l`. -/
theorem tendsto_mul_const_atTop_of_neg (hr : r < 0) :
    Tendsto (fun x => f x * r) l atTop ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt r 0
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atTop) (Filter.Ten …
  -/
  simpa only [mul_comm] using tendsto_const_mul_atTop_of_neg hr
  /-
    🎉 no goals
  -/


/-- If `r` is a negative constant, `fun x ↦ f x / r` tends to infinity along a filter `l`
if and only if `f` tends to negative infinity along `l`. -/
lemma tendsto_div_const_atTop_of_neg (hr : r < 0) :
    Tendsto (fun x ↦ f x / r) l atTop ↔ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt r 0
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atTop) (Filter.Ten …
  -/
  simp [div_eq_mul_inv, tendsto_mul_const_atTop_of_neg, hr]
  /-
    🎉 no goals
  -/


/-- If `r` is a negative constant, `fun x ↦ r * f x` tends to negative infinity along a filter `l`
if and only if `f` tends to infinity along `l`. -/
theorem tendsto_const_mul_atBot_of_neg (hr : r < 0) :
    Tendsto (fun x => r * f x) l atBot ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt r 0
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atBot) (Filter.Ten …
  -/
  simpa only [neg_mul, tendsto_neg_atTop_iff] using tendsto_const_mul_atTop_of_pos (neg_pos.2 hr)
  /-
    🎉 no goals
  -/


/-- If `r` is a negative constant, `fun x ↦ f x * r` tends to negative infinity along a filter `l`
if and only if `f` tends to infinity along `l`. -/
theorem tendsto_mul_const_atBot_of_neg (hr : r < 0) :
    Tendsto (fun x => f x * r) l atBot ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt r 0
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atBot) (Filter.Ten …
  -/
  simpa only [mul_comm] using tendsto_const_mul_atBot_of_neg hr
  /-
    🎉 no goals
  -/


/-- If `r` is a negative constant, `fun x ↦ f x / r` tends to negative infinity along a filter `l`
if and only if `f` tends to infinity along `l`. -/
lemma tendsto_div_const_atBot_of_neg (hr : r < 0) :
    Tendsto (fun x ↦ f x / r) l atBot ↔ Tendsto f l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    hr : LT.lt r 0
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atBot) (Filter.Ten …
  -/
  simp [div_eq_mul_inv, tendsto_mul_const_atBot_of_neg, hr]
  /-
    🎉 no goals
  -/


/-- The function `fun x ↦ r * f x` tends to infinity along a nontrivial filter
if and only if `r > 0` and `f` tends to infinity or `r < 0` and `f` tends to negative infinity. -/
theorem tendsto_const_mul_atTop_iff [NeBot l] :
    Tendsto (fun x => r * f x) l atTop ↔ 0 < r ∧ Tendsto f l atTop ∨ r < 0 ∧ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop) (Or (And (L …
  -/
  rcases lt_trichotomy r 0 with (hr | rfl | hr)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrderedField α
      l : Filter β
      f : β → α
      r : α
      inst✝ : l.NeBot
      hr : LT.lt r 0
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop) (Or (And (L …
    -/
  · simp [hr, hr.not_lt, tendsto_const_mul_atTop_of_neg]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrderedField α
      l : Filter β
      f : β → α
      inst✝ : l.NeBot
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul 0 (f x)) l Filter.atTop) (Or (And (L …
    -/
  · simp [not_tendsto_const_atTop]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrderedField α
      l : Filter β
      f : β → α
      r : α
      inst✝ : l.NeBot
      hr : LT.lt 0 r
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop) (Or (And (L …
    -/
  · simp [hr, hr.not_lt, tendsto_const_mul_atTop_of_pos]
    /-
      🎉 no goals
    -/


/-- The function `fun x ↦ f x * r` tends to infinity along a nontrivial filter
if and only if `r > 0` and `f` tends to infinity or `r < 0` and `f` tends to negative infinity. -/
theorem tendsto_mul_const_atTop_iff [NeBot l] :
    Tendsto (fun x => f x * r) l atTop ↔ 0 < r ∧ Tendsto f l atTop ∨ r < 0 ∧ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atTop) (Or (And (L …
  -/
  simp only [mul_comm _ r, tendsto_const_mul_atTop_iff]
  /-
    🎉 no goals
  -/


/-- The function `fun x ↦ f x / r` tends to infinity along a nontrivial filter
if and only if `r > 0` and `f` tends to infinity or `r < 0` and `f` tends to negative infinity. -/
lemma tendsto_div_const_atTop_iff [NeBot l] :
    Tendsto (fun x ↦ f x / r) l atTop ↔ 0 < r ∧ Tendsto f l atTop ∨ r < 0 ∧ Tendsto f l atBot := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atTop) (Or (And (L …
  -/
  simp [div_eq_mul_inv, tendsto_mul_const_atTop_iff]
  /-
    🎉 no goals
  -/


/-- The function `fun x ↦ r * f x` tends to negative infinity along a nontrivial filter
if and only if `r > 0` and `f` tends to negative infinity or `r < 0` and `f` tends to infinity. -/
theorem tendsto_const_mul_atBot_iff [NeBot l] :
    Tendsto (fun x => r * f x) l atBot ↔ 0 < r ∧ Tendsto f l atBot ∨ r < 0 ∧ Tendsto f l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atBot) (Or (And (L …
  -/
  simp only [← tendsto_neg_atTop_iff, ← mul_neg, tendsto_const_mul_atTop_iff, neg_neg]
  /-
    🎉 no goals
  -/


/-- The function `fun x ↦ f x * r` tends to negative infinity along a nontrivial filter
if and only if `r > 0` and `f` tends to negative infinity or `r < 0` and `f` tends to infinity. -/
theorem tendsto_mul_const_atBot_iff [NeBot l] :
    Tendsto (fun x => f x * r) l atBot ↔ 0 < r ∧ Tendsto f l atBot ∨ r < 0 ∧ Tendsto f l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atBot) (Or (And (L …
  -/
  simp only [mul_comm _ r, tendsto_const_mul_atBot_iff]
  /-
    🎉 no goals
  -/


/-- The function `fun x ↦ f x / r` tends to negative infinity along a nontrivial filter
if and only if `r > 0` and `f` tends to negative infinity or `r < 0` and `f` tends to infinity. -/
lemma tendsto_div_const_atBot_iff [NeBot l] :
    Tendsto (fun x ↦ f x / r) l atBot ↔ 0 < r ∧ Tendsto f l atBot ∨ r < 0 ∧ Tendsto f l atTop := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atBot) (Or (And (L …
  -/
  simp [div_eq_mul_inv, tendsto_mul_const_atBot_iff]
  /-
    🎉 no goals
  -/


/-- If `f` tends to negative infinity along a nontrivial filter `l`,
then `fun x ↦ r * f x` tends to infinity if and only if `r < 0. `-/
theorem tendsto_const_mul_atTop_iff_neg [NeBot l] (h : Tendsto f l atBot) :
    Tendsto (fun x => r * f x) l atTop ↔ r < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atBot
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atTop) (LT.lt r 0)
  -/
  simp [tendsto_const_mul_atTop_iff, h, h.not_tendsto disjoint_atBot_atTop]
  /-
    🎉 no goals
  -/


/-- If `f` tends to negative infinity along a nontrivial filter `l`,
then `fun x ↦ f x * r` tends to infinity if and only if `r < 0. `-/
theorem tendsto_mul_const_atTop_iff_neg [NeBot l] (h : Tendsto f l atBot) :
    Tendsto (fun x => f x * r) l atTop ↔ r < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atBot
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atTop) (LT.lt r 0)
  -/
  simp only [mul_comm _ r, tendsto_const_mul_atTop_iff_neg h]
  /-
    🎉 no goals
  -/


/-- If `f` tends to negative infinity along a nontrivial filter `l`,
then `fun x ↦ f x / r` tends to infinity if and only if `r < 0. `-/
lemma tendsto_div_const_atTop_iff_neg [NeBot l] (h : Tendsto f l atBot) :
    Tendsto (fun x ↦ f x / r) l atTop ↔ r < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atBot
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atTop) (LT.lt r 0)
  -/
  simp [div_eq_mul_inv, tendsto_mul_const_atTop_iff_neg h]
  /-
    🎉 no goals
  -/


/-- If `f` tends to negative infinity along a nontrivial filter `l`, then
`fun x ↦ r * f x` tends to negative infinity if and only if `0 < r. `-/
theorem tendsto_const_mul_atBot_iff_pos [NeBot l] (h : Tendsto f l atBot) :
    Tendsto (fun x => r * f x) l atBot ↔ 0 < r := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atBot
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atBot) (LT.lt 0 r)
  -/
  simp [tendsto_const_mul_atBot_iff, h, h.not_tendsto disjoint_atBot_atTop]
  /-
    🎉 no goals
  -/


/-- If `f` tends to negative infinity along a nontrivial filter `l`, then
`fun x ↦ f x * r` tends to negative infinity if and only if `0 < r. `-/
theorem tendsto_mul_const_atBot_iff_pos [NeBot l] (h : Tendsto f l atBot) :
    Tendsto (fun x => f x * r) l atBot ↔ 0 < r := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atBot
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atBot) (LT.lt 0 r)
  -/
  simp only [mul_comm _ r, tendsto_const_mul_atBot_iff_pos h]
  /-
    🎉 no goals
  -/


/-- If `f` tends to negative infinity along a nontrivial filter `l`, then
`fun x ↦ f x / r` tends to negative infinity if and only if `0 < r. `-/
lemma tendsto_div_const_atBot_iff_pos [NeBot l] (h : Tendsto f l atBot) :
    Tendsto (fun x ↦ f x / r) l atBot ↔ 0 < r := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atBot
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atBot) (LT.lt 0 r)
  -/
  simp [div_eq_mul_inv, tendsto_mul_const_atBot_iff_pos h]
  /-
    🎉 no goals
  -/


/-- If `f` tends to infinity along a nontrivial filter,
`fun x ↦ r * f x` tends to negative infinity if and only if `r < 0. `-/
theorem tendsto_const_mul_atBot_iff_neg [NeBot l] (h : Tendsto f l atTop) :
    Tendsto (fun x => r * f x) l atBot ↔ r < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atTop
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul r (f x)) l Filter.atBot) (LT.lt r 0)
  -/
  simp [tendsto_const_mul_atBot_iff, h, h.not_tendsto disjoint_atTop_atBot]
  /-
    🎉 no goals
  -/


/-- If `f` tends to infinity along a nontrivial filter,
`fun x ↦ f x * r` tends to negative infinity if and only if `r < 0. `-/
theorem tendsto_mul_const_atBot_iff_neg [NeBot l] (h : Tendsto f l atTop) :
    Tendsto (fun x => f x * r) l atBot ↔ r < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atTop
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul (f x) r) l Filter.atBot) (LT.lt r 0)
  -/
  simp only [mul_comm _ r, tendsto_const_mul_atBot_iff_neg h]
  /-
    🎉 no goals
  -/


/-- If `f` tends to infinity along a nontrivial filter,
`fun x ↦ f x / r` tends to negative infinity if and only if `r < 0. `-/
lemma tendsto_div_const_atBot_iff_neg [NeBot l] (h : Tendsto f l atTop) :
    Tendsto (fun x ↦ f x / r) l atBot ↔ r < 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrderedField α
    l : Filter β
    f : β → α
    r : α
    inst✝ : l.NeBot
    h : Filter.Tendsto f l Filter.atTop
    ⊢ Iff (Filter.Tendsto (fun x => HDiv.hDiv (f x) r) l Filter.atBot) (LT.lt r 0)
  -/
  simp [div_eq_mul_inv, tendsto_mul_const_atBot_iff_neg h]
  /-
    🎉 no goals
  -/


/-- If a function `f` tends to infinity along a filter,
then `f` multiplied by a negative constant (on the left) tends to negative infinity. -/
theorem Tendsto.const_mul_atTop_of_neg (hr : r < 0) (hf : Tendsto f l atTop) :
    Tendsto (fun x => r * f x) l atBot :=
  (tendsto_const_mul_atBot_of_neg hr).2 hf


/-- If a function `f` tends to infinity along a filter,
then `f` multiplied by a negative constant (on the right) tends to negative infinity. -/
theorem Tendsto.atTop_mul_const_of_neg (hr : r < 0) (hf : Tendsto f l atTop) :
    Tendsto (fun x => f x * r) l atBot :=
  (tendsto_mul_const_atBot_of_neg hr).2 hf


/-- If a function `f` tends to infinity along a filter,
then `f` divided by a negative constant tends to negative infinity. -/
lemma Tendsto.atTop_div_const_of_neg (hr : r < 0) (hf : Tendsto f l atTop) :
    Tendsto (fun x ↦ f x / r) l atBot := (tendsto_div_const_atBot_of_neg hr).2 hf


/-- If a function `f` tends to negative infinity along a filter, then `f` multiplied by
a positive constant (on the left) also tends to negative infinity. -/
theorem Tendsto.const_mul_atBot (hr : 0 < r) (hf : Tendsto f l atBot) :
    Tendsto (fun x => r * f x) l atBot :=
  (tendsto_const_mul_atBot_of_pos hr).2 hf


/-- If a function `f` tends to negative infinity along a filter, then `f` multiplied by
a positive constant (on the right) also tends to negative infinity. -/
theorem Tendsto.atBot_mul_const (hr : 0 < r) (hf : Tendsto f l atBot) :
    Tendsto (fun x => f x * r) l atBot :=
  (tendsto_mul_const_atBot_of_pos hr).2 hf


/-- If a function `f` tends to negative infinity along a filter, then `f` divided by
a positive constant also tends to negative infinity. -/
theorem Tendsto.atBot_div_const (hr : 0 < r) (hf : Tendsto f l atBot) :
    Tendsto (fun x => f x / r) l atBot := (tendsto_div_const_atBot_of_pos hr).2 hf


/-- If a function `f` tends to negative infinity along a filter,
then `f` multiplied by a negative constant (on the left) tends to positive infinity. -/
theorem Tendsto.const_mul_atBot_of_neg (hr : r < 0) (hf : Tendsto f l atBot) :
    Tendsto (fun x => r * f x) l atTop :=
  (tendsto_const_mul_atTop_of_neg hr).2 hf


/-- If a function tends to negative infinity along a filter,
then `f` multiplied by a negative constant (on the right) tends to positive infinity. -/
theorem Tendsto.atBot_mul_const_of_neg (hr : r < 0) (hf : Tendsto f l atBot) :
    Tendsto (fun x => f x * r) l atTop :=
  (tendsto_mul_const_atTop_of_neg hr).2 hf


theorem tendsto_neg_const_mul_pow_atTop {c : α} {n : ℕ} (hn : n ≠ 0) (hc : c < 0) :
    Tendsto (fun x => c * x ^ n) atTop atBot :=
  (tendsto_pow_atTop hn).const_mul_atTop_of_neg hc


theorem tendsto_const_mul_pow_atBot_iff {c : α} {n : ℕ} :
    Tendsto (fun x => c * x ^ n) atTop atBot ↔ n ≠ 0 ∧ c < 0 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedField α
    c : α
    n : Nat
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop Filt …
  -/
  simp only [← tendsto_neg_atTop_iff, ← neg_mul, tendsto_const_mul_pow_atTop_iff, neg_pos]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-06")]
alias Tendsto.neg_const_mul_atTop := Tendsto.const_mul_atTop_of_neg


@[deprecated (since := "2024-05-06")]
alias Tendsto.atTop_mul_neg_const := Tendsto.atTop_mul_const_of_neg


@[deprecated (since := "2024-05-06")]
alias Tendsto.neg_const_mul_atBot := Tendsto.const_mul_atBot_of_neg


@[deprecated (since := "2024-05-06")]
alias Tendsto.atBot_mul_neg_const := Tendsto.atBot_mul_const_of_neg


