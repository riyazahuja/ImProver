theorem zero_cpow_eq_nhds {b : ℂ} (hb : b ≠ 0) : (fun x : ℂ => (0 : ℂ) ^ x) =ᶠ[𝓝 b] 0 := by
  suffices ∀ᶠ x : ℂ in 𝓝 b, x ≠ 0 from
    this.mono fun x hx ↦ by
      dsimp only
      rw [zero_cpow hx, Pi.zero_apply]
  /-
    b : Complex
    hb : Ne b 0
    ⊢ Filter.Eventually (fun x => Ne x 0) (nhds b)
  -/
  exact IsOpen.eventually_mem isOpen_ne hb
  /-
    🎉 no goals
  -/


theorem cpow_eq_nhds {a b : ℂ} (ha : a ≠ 0) :
    (fun x => x ^ b) =ᶠ[𝓝 a] fun x => exp (log x * b) := by
  suffices ∀ᶠ x : ℂ in 𝓝 a, x ≠ 0 from
    this.mono fun x hx ↦ by
      dsimp only
      rw [cpow_def_of_ne_zero hx]
  /-
    a b : Complex
    ha : Ne a 0
    ⊢ Filter.Eventually (fun x => Ne x 0) (nhds a)
  -/
  exact IsOpen.eventually_mem isOpen_ne ha
  /-
    🎉 no goals
  -/


theorem cpow_eq_nhds' {p : ℂ × ℂ} (hp_fst : p.fst ≠ 0) :
    (fun x => x.1 ^ x.2) =ᶠ[𝓝 p] fun x => exp (log x.1 * x.2) := by
  suffices ∀ᶠ x : ℂ × ℂ in 𝓝 p, x.1 ≠ 0 from
    this.mono fun x hx ↦ by
      dsimp only
      rw [cpow_def_of_ne_zero hx]
  /-
    p : Prod Complex Complex
    hp_fst : Ne p.1 0
    ⊢ Filter.Eventually (fun x => Ne x.1 0) (nhds p)
  -/
  refine IsOpen.eventually_mem ?_ hp_fst
  /-
    p : Prod Complex Complex
    hp_fst : Ne p.1 0
    ⊢ IsOpen fun x => Eq x.1 0 → False
  -/
  change IsOpen { x : ℂ × ℂ | x.1 = 0 }ᶜ
  /-
    p : Prod Complex Complex
    hp_fst : Ne p.1 0
    ⊢ IsOpen (HasCompl.compl (setOf fun x => Eq x.1 0))
  -/
  rw [isOpen_compl_iff]
  /-
    p : Prod Complex Complex
    hp_fst : Ne p.1 0
    ⊢ IsClosed (setOf fun x => Eq x.1 0)
  -/
  exact isClosed_eq continuous_fst continuous_const
  /-
    🎉 no goals
  -/

-- Continuity of `fun x => a ^ x`: union of these two lemmas is optimal.

theorem continuousAt_const_cpow {a b : ℂ} (ha : a ≠ 0) : ContinuousAt (fun x : ℂ => a ^ x) b := by
  have cpow_eq : (fun x : ℂ => a ^ x) = fun x => exp (log a * x) := by
    ext1 b
    rw [cpow_def_of_ne_zero ha]
  /-
    a b : Complex
    ha : Ne a 0
    cpow_eq : Eq (fun x => HPow.hPow a x) fun x => Complex.exp (HMul.hMul (Complex …
    ⊢ ContinuousAt (fun x => HPow.hPow a x) b
  -/
  rw [cpow_eq]
  /-
    a b : Complex
    ha : Ne a 0
    cpow_eq : Eq (fun x => HPow.hPow a x) fun x => Complex.exp (HMul.hMul (Complex …
    ⊢ ContinuousAt (fun x => Complex.exp (HMul.hMul (Complex.log a) x)) b
  -/
  exact continuous_exp.continuousAt.comp (ContinuousAt.mul continuousAt_const continuousAt_id)
  /-
    🎉 no goals
  -/


theorem continuousAt_const_cpow' {a b : ℂ} (h : b ≠ 0) : ContinuousAt (fun x : ℂ => a ^ x) b := by
  /-
    a b : Complex
    h : Ne b 0
    ⊢ ContinuousAt (fun x => HPow.hPow a x) b
  -/
  by_cases ha : a = 0
    /-
      case pos
      a b : Complex
      h : Ne b 0
      ha : Eq a 0
      ⊢ ContinuousAt (fun x => HPow.hPow a x) b
    -/
  · rw [ha, continuousAt_congr (zero_cpow_eq_nhds h)]
    /-
      case pos
      a b : Complex
      h : Ne b 0
      ha : Eq a 0
      ⊢ ContinuousAt 0 b
    -/
    exact continuousAt_const
    /-
      🎉 no goals
    -/
    /-
      case neg
      a b : Complex
      h : Ne b 0
      ha : Not (Eq a 0)
      ⊢ ContinuousAt (fun x => HPow.hPow a x) b
    -/
  · exact continuousAt_const_cpow ha
    /-
      🎉 no goals
    -/


/-- The function `z ^ w` is continuous in `(z, w)` provided that `z` does not belong to the interval
`(-∞, 0]` on the real line. See also `Complex.continuousAt_cpow_zero_of_re_pos` for a version that
works for `z = 0` but assumes `0 < re w`. -/
theorem continuousAt_cpow {p : ℂ × ℂ} (hp_fst : p.fst ∈ slitPlane) :
    ContinuousAt (fun x : ℂ × ℂ => x.1 ^ x.2) p := by
  /-
    p : Prod Complex Complex
    hp_fst : Membership.mem Complex.slitPlane p.1
    ⊢ ContinuousAt (fun x => HPow.hPow x.1 x.2) p
  -/
  rw [continuousAt_congr (cpow_eq_nhds' <| slitPlane_ne_zero hp_fst)]
  /-
    p : Prod Complex Complex
    hp_fst : Membership.mem Complex.slitPlane p.1
    ⊢ ContinuousAt (fun x => Complex.exp (HMul.hMul (Complex.log x.1) x.2)) p
  -/
  refine continuous_exp.continuousAt.comp ?_
  exact
    ContinuousAt.mul
      (ContinuousAt.comp (continuousAt_clog hp_fst) continuous_fst.continuousAt)
      continuous_snd.continuousAt


theorem continuousAt_cpow_const {a b : ℂ} (ha : a ∈ slitPlane) :
    ContinuousAt (· ^ b) a :=
  Tendsto.comp (@continuousAt_cpow (a, b) ha) (continuousAt_id.prod continuousAt_const)


theorem Filter.Tendsto.cpow {l : Filter α} {f g : α → ℂ} {a b : ℂ} (hf : Tendsto f l (𝓝 a))
    (hg : Tendsto g l (𝓝 b)) (ha : a ∈ slitPlane) :
    Tendsto (fun x => f x ^ g x) l (𝓝 (a ^ b)) :=
  (@continuousAt_cpow (a, b) ha).tendsto.comp (hf.prod_mk_nhds hg)


theorem Filter.Tendsto.const_cpow {l : Filter α} {f : α → ℂ} {a b : ℂ} (hf : Tendsto f l (𝓝 b))
    (h : a ≠ 0 ∨ b ≠ 0) : Tendsto (fun x => a ^ f x) l (𝓝 (a ^ b)) := by
  cases h with
  | inl h => exact (continuousAt_const_cpow h).tendsto.comp hf
  | inr h => exact (continuousAt_const_cpow' h).tendsto.comp hf


nonrec theorem ContinuousWithinAt.cpow (hf : ContinuousWithinAt f s a)
    (hg : ContinuousWithinAt g s a) (h0 : f a ∈ slitPlane) :
    ContinuousWithinAt (fun x => f x ^ g x) s a :=
  hf.cpow hg h0


nonrec theorem ContinuousWithinAt.const_cpow {b : ℂ} (hf : ContinuousWithinAt f s a)
    (h : b ≠ 0 ∨ f a ≠ 0) : ContinuousWithinAt (fun x => b ^ f x) s a :=
  hf.const_cpow h


nonrec theorem ContinuousAt.cpow (hf : ContinuousAt f a) (hg : ContinuousAt g a)
    (h0 : f a ∈ slitPlane) : ContinuousAt (fun x => f x ^ g x) a :=
  hf.cpow hg h0


nonrec theorem ContinuousAt.const_cpow {b : ℂ} (hf : ContinuousAt f a) (h : b ≠ 0 ∨ f a ≠ 0) :
    ContinuousAt (fun x => b ^ f x) a :=
  hf.const_cpow h


theorem ContinuousOn.cpow (hf : ContinuousOn f s) (hg : ContinuousOn g s)
    (h0 : ∀ a ∈ s, f a ∈ slitPlane) : ContinuousOn (fun x => f x ^ g x) s := fun a ha =>
  (hf a ha).cpow (hg a ha) (h0 a ha)


theorem ContinuousOn.const_cpow {b : ℂ} (hf : ContinuousOn f s) (h : b ≠ 0 ∨ ∀ a ∈ s, f a ≠ 0) :
    ContinuousOn (fun x => b ^ f x) s := fun a ha => (hf a ha).const_cpow (h.imp id fun h => h a ha)


theorem Continuous.cpow (hf : Continuous f) (hg : Continuous g)
    (h0 : ∀ a, f a ∈ slitPlane) : Continuous fun x => f x ^ g x :=
  continuous_iff_continuousAt.2 fun a => hf.continuousAt.cpow hg.continuousAt (h0 a)


theorem Continuous.const_cpow {b : ℂ} (hf : Continuous f) (h : b ≠ 0 ∨ ∀ a, f a ≠ 0) :
    Continuous fun x => b ^ f x :=
  continuous_iff_continuousAt.2 fun a => hf.continuousAt.const_cpow <| h.imp id fun h => h a


theorem ContinuousOn.cpow_const {b : ℂ} (hf : ContinuousOn f s)
    (h : ∀ a : α, a ∈ s → f a ∈ slitPlane) : ContinuousOn (fun x => f x ^ b) s :=
  hf.cpow continuousOn_const h


@[fun_prop]
lemma continuous_const_cpow (z : ℂ) [NeZero z] : Continuous fun s : ℂ ↦ z ^ s :=
  continuous_id.const_cpow (.inl <| NeZero.ne z)


theorem continuousAt_const_rpow {a b : ℝ} (h : a ≠ 0) : ContinuousAt (a ^ ·) b := by
  /-
    a b : Real
    h : Ne a 0
    ⊢ ContinuousAt (fun x => HPow.hPow a x) b
  -/
  simp only [rpow_def]
  /-
    a b : Real
    h : Ne a 0
    ⊢ ContinuousAt (fun x => (HPow.hPow ↑a ↑x).re) b
  -/
  refine Complex.continuous_re.continuousAt.comp ?_
  /-
    a b : Real
    h : Ne a 0
    ⊢ ContinuousAt (fun x => HPow.hPow ↑a ↑x) b
  -/
  refine (continuousAt_const_cpow ?_).comp Complex.continuous_ofReal.continuousAt
  /-
    a b : Real
    h : Ne a 0
    ⊢ Ne (↑a) 0
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem continuousAt_const_rpow' {a b : ℝ} (h : b ≠ 0) : ContinuousAt (a ^ ·) b := by
  /-
    a b : Real
    h : Ne b 0
    ⊢ ContinuousAt (fun x => HPow.hPow a x) b
  -/
  simp only [rpow_def]
  /-
    a b : Real
    h : Ne b 0
    ⊢ ContinuousAt (fun x => (HPow.hPow ↑a ↑x).re) b
  -/
  refine Complex.continuous_re.continuousAt.comp ?_
  /-
    a b : Real
    h : Ne b 0
    ⊢ ContinuousAt (fun x => HPow.hPow ↑a ↑x) b
  -/
  refine (continuousAt_const_cpow' ?_).comp Complex.continuous_ofReal.continuousAt
  /-
    a b : Real
    h : Ne b 0
    ⊢ Ne (↑b) 0
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem rpow_eq_nhds_of_neg {p : ℝ × ℝ} (hp_fst : p.fst < 0) :
    (fun x : ℝ × ℝ => x.1 ^ x.2) =ᶠ[𝓝 p] fun x => exp (log x.1 * x.2) * cos (x.2 * π) := by
  suffices ∀ᶠ x : ℝ × ℝ in 𝓝 p, x.1 < 0 from
    this.mono fun x hx ↦ by
      dsimp only
      rw [rpow_def_of_neg hx]
  /-
    p : Prod Real Real
    hp_fst : LT.lt p.1 0
    ⊢ Filter.Eventually (fun x => LT.lt x.1 0) (nhds p)
  -/
  exact IsOpen.eventually_mem (isOpen_lt continuous_fst continuous_const) hp_fst
  /-
    🎉 no goals
  -/


theorem rpow_eq_nhds_of_pos {p : ℝ × ℝ} (hp_fst : 0 < p.fst) :
    (fun x : ℝ × ℝ => x.1 ^ x.2) =ᶠ[𝓝 p] fun x => exp (log x.1 * x.2) := by
  suffices ∀ᶠ x : ℝ × ℝ in 𝓝 p, 0 < x.1 from
    this.mono fun x hx ↦ by
      dsimp only
      rw [rpow_def_of_pos hx]
  /-
    p : Prod Real Real
    hp_fst : LT.lt 0 p.1
    ⊢ Filter.Eventually (fun x => LT.lt 0 x.1) (nhds p)
  -/
  exact IsOpen.eventually_mem (isOpen_lt continuous_const continuous_fst) hp_fst
  /-
    🎉 no goals
  -/


theorem continuousAt_rpow_of_ne (p : ℝ × ℝ) (hp : p.1 ≠ 0) :
    ContinuousAt (fun p : ℝ × ℝ => p.1 ^ p.2) p := by
  /-
    p : Prod Real Real
    hp : Ne p.1 0
    ⊢ ContinuousAt (fun p => HPow.hPow p.1 p.2) p
  -/
  rw [ne_iff_lt_or_gt] at hp
  cases hp with
  | inl hp =>
    rw [continuousAt_congr (rpow_eq_nhds_of_neg hp)]
    refine ContinuousAt.mul ?_ (continuous_cos.continuousAt.comp ?_)
    · refine continuous_exp.continuousAt.comp (ContinuousAt.mul ?_ continuous_snd.continuousAt)
      refine (continuousAt_log ?_).comp continuous_fst.continuousAt
      exact hp.ne
    · exact continuous_snd.continuousAt.mul continuousAt_const
  | inr hp =>
    rw [continuousAt_congr (rpow_eq_nhds_of_pos hp)]
    refine continuous_exp.continuousAt.comp (ContinuousAt.mul ?_ continuous_snd.continuousAt)
    refine (continuousAt_log ?_).comp continuous_fst.continuousAt
    exact hp.lt.ne.symm


theorem continuousAt_rpow_of_pos (p : ℝ × ℝ) (hp : 0 < p.2) :
    ContinuousAt (fun p : ℝ × ℝ => p.1 ^ p.2) p := by
  /-
    p : Prod Real Real
    hp : LT.lt 0 p.2
    ⊢ ContinuousAt (fun p => HPow.hPow p.1 p.2) p
  -/
  cases' p with x y
  /-
    case mk
    x y : Real
    hp : LT.lt 0 { fst := x, snd := y }.2
    ⊢ ContinuousAt (fun p => HPow.hPow p.1 p.2) { fst := x, snd := y }
  -/
  dsimp only at hp
  /-
    case mk
    x y : Real
    hp : LT.lt 0 y
    ⊢ ContinuousAt (fun p => HPow.hPow p.1 p.2) { fst := x, snd := y }
  -/
  obtain hx | rfl := ne_or_eq x 0
    /-
      case mk.inl
      x y : Real
      hp : LT.lt 0 y
      hx : Ne x 0
      ⊢ ContinuousAt (fun p => HPow.hPow p.1 p.2) { fst := x, snd := y }
    -/
  · exact continuousAt_rpow_of_ne (x, y) hx
    /-
      🎉 no goals
    -/
  have A : Tendsto (fun p : ℝ × ℝ => exp (log p.1 * p.2)) (𝓝[≠] 0 ×ˢ 𝓝 y) (𝓝 0) :=
    tendsto_exp_atBot.comp
      ((tendsto_log_nhdsWithin_zero.comp tendsto_fst).atBot_mul hp tendsto_snd)
  have B : Tendsto (fun p : ℝ × ℝ => p.1 ^ p.2) (𝓝[≠] 0 ×ˢ 𝓝 y) (𝓝 0) :=
    squeeze_zero_norm (fun p => abs_rpow_le_exp_log_mul p.1 p.2) A
  have C : Tendsto (fun p : ℝ × ℝ => p.1 ^ p.2) (𝓝[{0}] 0 ×ˢ 𝓝 y) (pure 0) := by
    rw [nhdsWithin_singleton, tendsto_pure, pure_prod, eventually_map]
    exact (lt_mem_nhds hp).mono fun y hy => zero_rpow hy.ne'
  simpa only [← sup_prod, ← nhdsWithin_union, compl_union_self, nhdsWithin_univ, nhds_prod_eq,
    ContinuousAt, zero_rpow hp.ne'] using B.sup (C.mono_right (pure_le_nhds _))


theorem continuousAt_rpow (p : ℝ × ℝ) (h : p.1 ≠ 0 ∨ 0 < p.2) :
    ContinuousAt (fun p : ℝ × ℝ => p.1 ^ p.2) p :=
  h.elim (fun h => continuousAt_rpow_of_ne p h) fun h => continuousAt_rpow_of_pos p h


@[fun_prop]
theorem continuousAt_rpow_const (x : ℝ) (q : ℝ) (h : x ≠ 0 ∨ 0 ≤ q) :
    ContinuousAt (fun x : ℝ => x ^ q) x := by
  /-
    x q : Real
    h : Or (Ne x 0) (LE.le 0 q)
    ⊢ ContinuousAt (fun x => HPow.hPow x q) x
  -/
· rw [le_iff_lt_or_eq, ← or_assoc] at h
  /-
    x q : Real
    h : Or (Or (Ne x 0) (LT.lt 0 q)) (Eq 0 q)
    ⊢ ContinuousAt (fun x => HPow.hPow x q) x
  -/
  obtain h|rfl := h
    /-
      case inl
      x q : Real
      h : Or (Ne x 0) (LT.lt 0 q)
      ⊢ ContinuousAt (fun x => HPow.hPow x q) x
    -/
  · exact (continuousAt_rpow (x, q) h).comp₂ continuousAt_id continuousAt_const
    /-
      🎉 no goals
    -/
    /-
      case inr
      x : Real
      ⊢ ContinuousAt (fun x => HPow.hPow x 0) x
    -/
  · simp_rw [rpow_zero]; exact continuousAt_const
                         /-
                           🎉 no goals
                         -/


@[fun_prop]
theorem continuous_rpow_const {q : ℝ} (h : 0 ≤ q) : Continuous (fun x : ℝ => x ^ q) :=
  continuous_iff_continuousAt.mpr fun x ↦ continuousAt_rpow_const x q (.inr h)

theorem Filter.Tendsto.rpow {l : Filter α} {f g : α → ℝ} {x y : ℝ} (hf : Tendsto f l (𝓝 x))
    (hg : Tendsto g l (𝓝 y)) (h : x ≠ 0 ∨ 0 < y) : Tendsto (fun t => f t ^ g t) l (𝓝 (x ^ y)) :=
  (Real.continuousAt_rpow (x, y) h).tendsto.comp (hf.prod_mk_nhds hg)


theorem Filter.Tendsto.rpow_const {l : Filter α} {f : α → ℝ} {x p : ℝ} (hf : Tendsto f l (𝓝 x))
    (h : x ≠ 0 ∨ 0 ≤ p) : Tendsto (fun a => f a ^ p) l (𝓝 (x ^ p)) :=
                             /-
                               α : Type u_1
                               l : Filter α
                               f : α → Real
                               x p : Real
                               hf : Filter.Tendsto f l (nhds x)
                               h : Or (Ne x 0) (LE.le 0 p)
                               h0 : Eq 0 p
                               ⊢ Filter.Tendsto (fun a => HPow.hPow (f a) 0) l (nhds (HPow.hPow x 0))
                             -/
  if h0 : 0 = p then h0 ▸ by simp [tendsto_const_nhds]
                             /-
                               🎉 no goals
                             -/
  else hf.rpow tendsto_const_nhds (h.imp id fun h' => h'.lt_of_ne h0)


nonrec theorem ContinuousAt.rpow (hf : ContinuousAt f x) (hg : ContinuousAt g x)
    (h : f x ≠ 0 ∨ 0 < g x) : ContinuousAt (fun t => f t ^ g t) x :=
  hf.rpow hg h


nonrec theorem ContinuousWithinAt.rpow (hf : ContinuousWithinAt f s x)
    (hg : ContinuousWithinAt g s x) (h : f x ≠ 0 ∨ 0 < g x) :
    ContinuousWithinAt (fun t => f t ^ g t) s x :=
  hf.rpow hg h


theorem ContinuousOn.rpow (hf : ContinuousOn f s) (hg : ContinuousOn g s)
    (h : ∀ x ∈ s, f x ≠ 0 ∨ 0 < g x) : ContinuousOn (fun t => f t ^ g t) s := fun t ht =>
  (hf t ht).rpow (hg t ht) (h t ht)


theorem Continuous.rpow (hf : Continuous f) (hg : Continuous g) (h : ∀ x, f x ≠ 0 ∨ 0 < g x) :
    Continuous fun x => f x ^ g x :=
  continuous_iff_continuousAt.2 fun x => hf.continuousAt.rpow hg.continuousAt (h x)


nonrec theorem ContinuousWithinAt.rpow_const (hf : ContinuousWithinAt f s x) (h : f x ≠ 0 ∨ 0 ≤ p) :
    ContinuousWithinAt (fun x => f x ^ p) s x :=
  hf.rpow_const h


nonrec theorem ContinuousAt.rpow_const (hf : ContinuousAt f x) (h : f x ≠ 0 ∨ 0 ≤ p) :
    ContinuousAt (fun x => f x ^ p) x :=
  hf.rpow_const h


theorem ContinuousOn.rpow_const (hf : ContinuousOn f s) (h : ∀ x ∈ s, f x ≠ 0 ∨ 0 ≤ p) :
    ContinuousOn (fun x => f x ^ p) s := fun x hx => (hf x hx).rpow_const (h x hx)


theorem Continuous.rpow_const (hf : Continuous f) (h : ∀ x, f x ≠ 0 ∨ 0 ≤ p) :
    Continuous fun x => f x ^ p :=
  continuous_iff_continuousAt.2 fun x => hf.continuousAt.rpow_const (h x)


/-- See also `continuousAt_cpow` and `Complex.continuousAt_cpow_of_re_pos`. -/
theorem continuousAt_cpow_zero_of_re_pos {z : ℂ} (hz : 0 < z.re) :
    ContinuousAt (fun x : ℂ × ℂ => x.1 ^ x.2) (0, z) := by
  /-
    z : Complex
    hz : LT.lt 0 z.re
    ⊢ ContinuousAt (fun x => HPow.hPow x.1 x.2) { fst := 0, snd := z }
  -/
  have hz₀ : z ≠ 0 := ne_of_apply_ne re hz.ne'
  /-
    z : Complex
    hz : LT.lt 0 z.re
    hz₀ : Ne z 0
    ⊢ ContinuousAt (fun x => HPow.hPow x.1 x.2) { fst := 0, snd := z }
  -/
  rw [ContinuousAt, zero_cpow hz₀, tendsto_zero_iff_norm_tendsto_zero]
  /-
    z : Complex
    hz : LT.lt 0 z.re
    hz₀ : Ne z 0
    ⊢ Filter.Tendsto (fun x => Norm.norm (HPow.hPow x.1 x.2)) (nhds { fst := 0, sn …
  -/
  refine squeeze_zero (fun _ => norm_nonneg _) (fun _ => abs_cpow_le _ _) ?_
  /-
    z : Complex
    hz : LT.lt 0 z.re
    hz₀ : Ne z 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow (Complex.abs x.1) x.2.re) (Rea …
  -/
  simp only [div_eq_mul_inv, ← Real.exp_neg]
  /-
    z : Complex
    hz : LT.lt 0 z.re
    hz₀ : Ne z 0
    ⊢ Filter.Tendsto (fun x => HMul.hMul (HPow.hPow (Complex.abs x.1) x.2.re) (Rea …
  -/
  refine Tendsto.zero_mul_isBoundedUnder_le ?_ ?_
  · convert
        (continuous_fst.norm.tendsto ((0 : ℂ), z)).rpow
          ((continuous_re.comp continuous_snd).tendsto _) _ <;>
      /-
        case h.e'_5.h.e'_3
        z : Complex
        hz : LT.lt 0 z.re
        hz₀ : Ne z 0
        ⊢ Eq 0 (HPow.hPow (Norm.norm { fst := 0, snd := z }.1) (Function.comp Complex. …
      -/
      /-
        🎉 no goals
      -/
      simp [hz, Real.zero_rpow hz.ne']
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      z : Complex
      hz : LT.lt 0 z.re
      hz₀ : Ne z 0
      ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (nhds { fst := 0, snd := z  …
    -/
  · simp only [Function.comp_def, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    /-
      case refine_2
      z : Complex
      hz : LT.lt 0 z.re
      hz₀ : Ne z 0
      ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (nhds { fst := 0, snd := z  …
    -/
    rcases exists_gt |im z| with ⟨C, hC⟩
    /-
      case refine_2.intro
      z : Complex
      hz : LT.lt 0 z.re
      hz₀ : Ne z 0
      C : Real
      hC : LT.lt (_root_.abs z.im) C
      ⊢ Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (nhds { fst := 0, snd := z  …
    -/
    refine ⟨Real.exp (π * C), eventually_map.2 ?_⟩
    refine
      (((continuous_im.comp continuous_snd).abs.tendsto (_, z)).eventually (gt_mem_nhds hC)).mono
        fun z hz => Real.exp_le_exp.2 <| (neg_le_abs _).trans ?_
    /-
      case refine_2.intro
      z✝ : Complex
      hz✝ : LT.lt 0 z✝.re
      hz₀ : Ne z✝ 0
      C : Real
      hC : LT.lt (_root_.abs z✝.im) C
      z : Prod Complex Complex
      hz : LT.lt (_root_.abs (Function.comp Complex.im Prod.snd z)) C
      ⊢ LE.le (_root_.abs (HMul.hMul z.1.arg z.2.im)) (HMul.hMul Real.pi C)
    -/
    rw [_root_.abs_mul]
    exact
      mul_le_mul (abs_le.2 ⟨(neg_pi_lt_arg _).le, arg_le_pi _⟩) hz.le (_root_.abs_nonneg _)
        Real.pi_pos.le


open ComplexOrder in
/-- See also `continuousAt_cpow` for a version that assumes `p.1 ≠ 0` but makes no
assumptions about `p.2`. -/
theorem continuousAt_cpow_of_re_pos {p : ℂ × ℂ} (h₁ : 0 ≤ p.1.re ∨ p.1.im ≠ 0) (h₂ : 0 < p.2.re) :
    ContinuousAt (fun x : ℂ × ℂ => x.1 ^ x.2) p := by
  /-
    p : Prod Complex Complex
    h₁ : Or (LE.le 0 p.1.re) (Ne p.1.im 0)
    h₂ : LT.lt 0 p.2.re
    ⊢ ContinuousAt (fun x => HPow.hPow x.1 x.2) p
  -/
  cases' p with z w
  rw [← not_lt_zero_iff, lt_iff_le_and_ne, not_and_or, Ne, Classical.not_not,
    not_le_zero_iff] at h₁
  /-
    case mk
    z w : Complex
    h₁ : Or (Or (LT.lt 0 { fst := z, snd := w }.1.re) (Ne { fst := z, snd := w }.1 …
    h₂ : LT.lt 0 { fst := z, snd := w }.2.re
    ⊢ ContinuousAt (fun x => HPow.hPow x.1 x.2) { fst := z, snd := w }
  -/
  rcases h₁ with (h₁ | (rfl : z = 0))
  /-
    case mk.inl
    z w : Complex
    h₂ : LT.lt 0 { fst := z, snd := w }.2.re
    h₁ : Or (LT.lt 0 { fst := z, snd := w }.1.re) (Ne { fst := z, snd := w }.1.im 0)
    ⊢ ContinuousAt (fun x => HPow.hPow x.1 x.2) { fst := z, snd := w }
  -/
  exacts [continuousAt_cpow h₁, continuousAt_cpow_zero_of_re_pos h₂]
  /-
    🎉 no goals
  -/


/-- See also `continuousAt_cpow_const` for a version that assumes `z ≠ 0` but makes no
assumptions about `w`. -/
theorem continuousAt_cpow_const_of_re_pos {z w : ℂ} (hz : 0 ≤ re z ∨ im z ≠ 0) (hw : 0 < re w) :
    ContinuousAt (fun x => x ^ w) z :=
  Tendsto.comp (@continuousAt_cpow_of_re_pos (z, w) hz hw) (continuousAt_id.prod continuousAt_const)


/-- Continuity of `(x, y) ↦ x ^ y` as a function on `ℝ × ℂ`. -/
theorem continuousAt_ofReal_cpow (x : ℝ) (y : ℂ) (h : 0 < y.re ∨ x ≠ 0) :
    ContinuousAt (fun p => (p.1 : ℂ) ^ p.2 : ℝ × ℂ → ℂ) (x, y) := by
  /-
    x : Real
    y : Complex
    h : Or (LT.lt 0 y.re) (Ne x 0)
    ⊢ ContinuousAt (fun p => HPow.hPow (↑p.1) p.2) { fst := x, snd := y }
  -/
  rcases lt_trichotomy (0 : ℝ) x with (hx | rfl | hx)
  · -- x > 0 : easy case
    /-
      case inl
      x : Real
      y : Complex
      h : Or (LT.lt 0 y.re) (Ne x 0)
      hx : LT.lt 0 x
      ⊢ ContinuousAt (fun p => HPow.hPow (↑p.1) p.2) { fst := x, snd := y }
    -/
    have : ContinuousAt (fun p => ⟨↑p.1, p.2⟩ : ℝ × ℂ → ℂ × ℂ) (x, y) := by fun_prop
    /-
      case inl
      x : Real
      y : Complex
      h : Or (LT.lt 0 y.re) (Ne x 0)
      hx : LT.lt 0 x
      this : ContinuousAt (fun p => { fst := ↑p.1, snd := p.2 }) { fst := x, snd :=  …
      ⊢ ContinuousAt (fun p => HPow.hPow (↑p.1) p.2) { fst := x, snd := y }
    -/
    refine (continuousAt_cpow (Or.inl ?_)).comp this
    /-
      case inl
      x : Real
      y : Complex
      h : Or (LT.lt 0 y.re) (Ne x 0)
      hx : LT.lt 0 x
      this : ContinuousAt (fun p => { fst := ↑p.1, snd := p.2 }) { fst := x, snd :=  …
      ⊢ LT.lt 0 { fst := ↑{ fst := x, snd := y }.1, snd := { fst := x, snd := y }.2  …
    -/
    rwa [ofReal_re]
    /-
      🎉 no goals
    -/
  · -- x = 0 : reduce to continuousAt_cpow_zero_of_re_pos
    have A : ContinuousAt (fun p => p.1 ^ p.2 : ℂ × ℂ → ℂ) ⟨↑(0 : ℝ), y⟩ := by
      rw [ofReal_zero]
      apply continuousAt_cpow_zero_of_re_pos
      tauto
    /-
      case inr.inl
      y : Complex
      h : Or (LT.lt 0 y.re) (Ne 0 0)
      A : ContinuousAt (fun p => HPow.hPow p.1 p.2) { fst := ↑0, snd := y }
      ⊢ ContinuousAt (fun p => HPow.hPow (↑p.1) p.2) { fst := 0, snd := y }
    -/
    have B : ContinuousAt (fun p => ⟨↑p.1, p.2⟩ : ℝ × ℂ → ℂ × ℂ) ⟨0, y⟩ := by fun_prop
    /-
      case inr.inl
      y : Complex
      h : Or (LT.lt 0 y.re) (Ne 0 0)
      A : ContinuousAt (fun p => HPow.hPow p.1 p.2) { fst := ↑0, snd := y }
      B : ContinuousAt (fun p => { fst := ↑p.1, snd := p.2 }) { fst := 0, snd := y }
      ⊢ ContinuousAt (fun p => HPow.hPow (↑p.1) p.2) { fst := 0, snd := y }
    -/
    exact A.comp_of_eq B rfl
    /-
      🎉 no goals
    -/
  · -- x < 0 : difficult case
    suffices ContinuousAt (fun p => (-(p.1 : ℂ)) ^ p.2 * exp (π * I * p.2) : ℝ × ℂ → ℂ) (x, y) by
      refine this.congr (eventually_of_mem (prod_mem_nhds (Iio_mem_nhds hx) univ_mem) ?_)
      exact fun p hp => (ofReal_cpow_of_nonpos (le_of_lt hp.1) p.2).symm
    /-
      case inr.inr
      x : Real
      y : Complex
      h : Or (LT.lt 0 y.re) (Ne x 0)
      hx : LT.lt x 0
      ⊢ ContinuousAt (fun p => HMul.hMul (HPow.hPow (Neg.neg ↑p.1) p.2) (Complex.exp …
    -/
    have A : ContinuousAt (fun p => ⟨-↑p.1, p.2⟩ : ℝ × ℂ → ℂ × ℂ) (x, y) := by fun_prop
    /-
      case inr.inr
      x : Real
      y : Complex
      h : Or (LT.lt 0 y.re) (Ne x 0)
      hx : LT.lt x 0
      A : ContinuousAt (fun p => { fst := Neg.neg ↑p.1, snd := p.2 }) { fst := x, sn …
      ⊢ ContinuousAt (fun p => HMul.hMul (HPow.hPow (Neg.neg ↑p.1) p.2) (Complex.exp …
    -/
    apply ContinuousAt.mul
      /-
        case inr.inr.hf
        x : Real
        y : Complex
        h : Or (LT.lt 0 y.re) (Ne x 0)
        hx : LT.lt x 0
        A : ContinuousAt (fun p => { fst := Neg.neg ↑p.1, snd := p.2 }) { fst := x, sn …
        ⊢ ContinuousAt (fun x => HPow.hPow (Neg.neg ↑x.1) x.2) { fst := x, snd := y }
      -/
    · refine (continuousAt_cpow (Or.inl ?_)).comp A
      /-
        case inr.inr.hf
        x : Real
        y : Complex
        h : Or (LT.lt 0 y.re) (Ne x 0)
        hx : LT.lt x 0
        A : ContinuousAt (fun p => { fst := Neg.neg ↑p.1, snd := p.2 }) { fst := x, sn …
        ⊢ LT.lt 0 { fst := Neg.neg ↑{ fst := x, snd := y }.1, snd := { fst := x, snd : …
      -/
      rwa [neg_re, ofReal_re, neg_pos]
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.hg
        x : Real
        y : Complex
        h : Or (LT.lt 0 y.re) (Ne x 0)
        hx : LT.lt x 0
        A : ContinuousAt (fun p => { fst := Neg.neg ↑p.1, snd := p.2 }) { fst := x, sn …
        ⊢ ContinuousAt (fun x => Complex.exp (HMul.hMul (HMul.hMul (↑Real.pi) Complex. …
      -/
    · exact (continuous_exp.comp (continuous_const.mul continuous_snd)).continuousAt
      /-
        🎉 no goals
      -/


theorem continuousAt_ofReal_cpow_const (x : ℝ) (y : ℂ) (h : 0 < y.re ∨ x ≠ 0) :
    ContinuousAt (fun a => (a : ℂ) ^ y : ℝ → ℂ) x := by
  exact ContinuousAt.comp (x := x) (continuousAt_ofReal_cpow x y h)
          ((continuous_id (X := ℝ)).prod_mk (continuous_const (y := y))).continuousAt


theorem continuous_ofReal_cpow_const {y : ℂ} (hs : 0 < y.re) :
    Continuous (fun x => (x : ℂ) ^ y : ℝ → ℂ) :=
  continuous_iff_continuousAt.mpr fun x => continuousAt_ofReal_cpow_const x y (Or.inl hs)


theorem continuousAt_rpow {x : ℝ≥0} {y : ℝ} (h : x ≠ 0 ∨ 0 < y) :
    ContinuousAt (fun p : ℝ≥0 × ℝ => p.1 ^ p.2) (x, y) := by
  have :
    (fun p : ℝ≥0 × ℝ => p.1 ^ p.2) =
      Real.toNNReal ∘ (fun p : ℝ × ℝ => p.1 ^ p.2) ∘ fun p : ℝ≥0 × ℝ => (p.1.1, p.2) := by
    ext p
    erw [coe_rpow, Real.coe_toNNReal _ (Real.rpow_nonneg p.1.2 _)]
    rfl
  /-
    x : NNReal
    y : Real
    h : Or (Ne x 0) (LT.lt 0 y)
    this : Eq (fun p => HPow.hPow p.1 p.2) (Function.comp Real.toNNReal (Function. …
    ⊢ ContinuousAt (fun p => HPow.hPow p.1 p.2) { fst := x, snd := y }
  -/
  rw [this]
  /-
    x : NNReal
    y : Real
    h : Or (Ne x 0) (LT.lt 0 y)
    this : Eq (fun p => HPow.hPow p.1 p.2) (Function.comp Real.toNNReal (Function. …
    ⊢ ContinuousAt (Function.comp Real.toNNReal (Function.comp (fun p => HPow.hPow …
  -/
  refine continuous_real_toNNReal.continuousAt.comp (ContinuousAt.comp ?_ ?_)
    /-
      case refine_1
      x : NNReal
      y : Real
      h : Or (Ne x 0) (LT.lt 0 y)
      this : Eq (fun p => HPow.hPow p.1 p.2) (Function.comp Real.toNNReal (Function. …
      ⊢ ContinuousAt (fun p => HPow.hPow p.1 p.2) { fst := ↑{ fst := x, snd := y }.1 …
    -/
  · apply Real.continuousAt_rpow
    /-
      case refine_1.h
      x : NNReal
      y : Real
      h : Or (Ne x 0) (LT.lt 0 y)
      this : Eq (fun p => HPow.hPow p.1 p.2) (Function.comp Real.toNNReal (Function. …
      ⊢ Or (Ne { fst := ↑{ fst := x, snd := y }.1, snd := { fst := x, snd := y }.2 } …
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      x : NNReal
      y : Real
      h : Or (Ne x 0) (LT.lt 0 y)
      this : Eq (fun p => HPow.hPow p.1 p.2) (Function.comp Real.toNNReal (Function. …
      ⊢ ContinuousAt (fun p => { fst := ↑p.1, snd := p.2 }) { fst := x, snd := y }
    -/
  · exact ((continuous_subtype_val.comp continuous_fst).prod_mk continuous_snd).continuousAt
    /-
      🎉 no goals
    -/


theorem eventually_pow_one_div_le (x : ℝ≥0) {y : ℝ≥0} (hy : 1 < y) :
    ∀ᶠ n : ℕ in atTop, x ^ (1 / n : ℝ) ≤ y := by
  /-
    x y : NNReal
    hy : LT.lt 1 y
    ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow x (HDiv.hDiv 1 ↑n)) y) Filter.a …
  -/
  obtain ⟨m, hm⟩ := add_one_pow_unbounded_of_pos x (tsub_pos_of_lt hy)
  /-
    case intro
    x y : NNReal
    hy : LT.lt 1 y
    m : Nat
    hm : LT.lt x (HPow.hPow (HAdd.hAdd (HSub.hSub y 1) 1) m)
    ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow x (HDiv.hDiv 1 ↑n)) y) Filter.a …
  -/
  rw [tsub_add_cancel_of_le hy.le] at hm
  /-
    case intro
    x y : NNReal
    hy : LT.lt 1 y
    m : Nat
    hm : LT.lt x (HPow.hPow y m)
    ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow x (HDiv.hDiv 1 ↑n)) y) Filter.a …
  -/
  refine eventually_atTop.2 ⟨m + 1, fun n hn => ?_⟩
  /-
    case intro
    x y : NNReal
    hy : LT.lt 1 y
    m : Nat
    hm : LT.lt x (HPow.hPow y m)
    n : Nat
    hn : GE.ge n (HAdd.hAdd m 1)
    ⊢ LE.le (HPow.hPow x (HDiv.hDiv 1 ↑n)) y
  -/
  simp only [one_div]
  simpa only [NNReal.rpow_inv_le_iff (Nat.cast_pos.2 <| m.succ_pos.trans_le hn),
    NNReal.rpow_natCast] using hm.le.trans (pow_right_mono₀ hy.le (m.le_succ.trans hn))


theorem Filter.Tendsto.nnrpow {α : Type*} {f : Filter α} {u : α → ℝ≥0} {v : α → ℝ} {x : ℝ≥0}
    {y : ℝ} (hx : Tendsto u f (𝓝 x)) (hy : Tendsto v f (𝓝 y)) (h : x ≠ 0 ∨ 0 < y) :
    Tendsto (fun a => u a ^ v a) f (𝓝 (x ^ y)) :=
  Tendsto.comp (NNReal.continuousAt_rpow h) (hx.prod_mk_nhds hy)


theorem continuousAt_rpow_const {x : ℝ≥0} {y : ℝ} (h : x ≠ 0 ∨ 0 ≤ y) :
    ContinuousAt (fun z => z ^ y) x :=
  h.elim (fun h => tendsto_id.nnrpow tendsto_const_nhds (Or.inl h)) fun h =>
                                     /-
                                       x : NNReal
                                       y : Real
                                       h✝¹ : Or (Ne x 0) (LE.le 0 y)
                                       h✝ : LE.le 0 y
                                       h : Eq 0 y
                                       ⊢ ContinuousAt (fun z => HPow.hPow z 0) x
                                     -/
    h.eq_or_lt.elim (fun h => h ▸ by simp only [rpow_zero, continuousAt_const]) fun h =>
                                     /-
                                       🎉 no goals
                                     -/
      tendsto_id.nnrpow tendsto_const_nhds (Or.inr h)


@[fun_prop]
theorem continuous_rpow_const {y : ℝ} (h : 0 ≤ y) : Continuous fun x : ℝ≥0 => x ^ y :=
  continuous_iff_continuousAt.2 fun _ => continuousAt_rpow_const (Or.inr h)


@[fun_prop]
theorem continuousOn_rpow_const_compl_zero {r : ℝ} :
    ContinuousOn (fun z : ℝ≥0 => z ^ r) {0}ᶜ :=
  fun _ h => ContinuousAt.continuousWithinAt <| NNReal.continuousAt_rpow_const (.inl h)

-- even though this follows from `ContinuousOn.mono` and the previous lemma, we include it for
-- automation purposes with `fun_prop`, because the side goal `0 ∉ s ∨ 0 ≤ r` is often easy to check

@[fun_prop]
theorem continuousOn_rpow_const {r : ℝ} {s : Set ℝ≥0}
    (h : 0 ∉ s ∨ 0 ≤ r) : ContinuousOn (fun z : ℝ≥0 => z ^ r) s :=
                                                    /-
                                                      r : Real
                                                      s : Set NNReal
                                                      h : Or (Not (Membership.mem s 0)) (LE.le 0 r)
                                                      x✝ : Not (Membership.mem s 0)
                                                      ⊢ ContinuousOn (fun z => HPow.hPow z r) (HasCompl.compl (Singleton.singleton 0))
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  h.elim (fun _ ↦ ContinuousOn.mono (s := {0}ᶜ) (by fun_prop) (by aesop))
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (NNReal.continuous_rpow_const · |>.continuousOn)


theorem eventually_pow_one_div_le {x : ℝ≥0∞} (hx : x ≠ ∞) {y : ℝ≥0∞} (hy : 1 < y) :
    ∀ᶠ n : ℕ in atTop, x ^ (1 / n : ℝ) ≤ y := by
  /-
    x : ENNReal
    hx : Ne x Top.top
    y : ENNReal
    hy : LT.lt 1 y
    ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow x (HDiv.hDiv 1 ↑n)) y) Filter.a …
  -/
  lift x to ℝ≥0 using hx
  /-
    case intro
    y : ENNReal
    hy : LT.lt 1 y
    x : NNReal
    ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow (↑x) (HDiv.hDiv 1 ↑n)) y) Filte …
  -/
  by_cases h : y = ∞
    /-
      case pos
      y : ENNReal
      hy : LT.lt 1 y
      x : NNReal
      h : Eq y Top.top
      ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow (↑x) (HDiv.hDiv 1 ↑n)) y) Filte …
    -/
  · exact Eventually.of_forall fun n => h.symm ▸ le_top
    /-
      🎉 no goals
    -/
    /-
      case neg
      y : ENNReal
      hy : LT.lt 1 y
      x : NNReal
      h : Not (Eq y Top.top)
      ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow (↑x) (HDiv.hDiv 1 ↑n)) y) Filte …
    -/
  · lift y to ℝ≥0 using h
    /-
      case neg.intro
      x y : NNReal
      hy : LT.lt 1 ↑y
      ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow (↑x) (HDiv.hDiv 1 ↑n)) ↑y) Filt …
    -/
    have := NNReal.eventually_pow_one_div_le x (mod_cast hy : 1 < y)
    /-
      case neg.intro
      x y : NNReal
      hy : LT.lt 1 ↑y
      this : Filter.Eventually (fun n => LE.le (HPow.hPow x (HDiv.hDiv 1 ↑n)) y) Fil …
      ⊢ Filter.Eventually (fun n => LE.le (HPow.hPow (↑x) (HDiv.hDiv 1 ↑n)) ↑y) Filt …
    -/
    refine this.congr (Eventually.of_forall fun n => ?_)
    /-
      case neg.intro
      x y : NNReal
      hy : LT.lt 1 ↑y
      this : Filter.Eventually (fun n => LE.le (HPow.hPow x (HDiv.hDiv 1 ↑n)) y) Fil …
      n : Nat
      ⊢ Iff (LE.le (HPow.hPow x (HDiv.hDiv 1 ↑n)) y) (LE.le (HPow.hPow (↑x) (HDiv.hD …
    -/
    rw [← coe_rpow_of_nonneg x (by positivity : 0 ≤ (1 / n : ℝ)), coe_le_coe]
    /-
      🎉 no goals
    -/


private theorem continuousAt_rpow_const_of_pos {x : ℝ≥0∞} {y : ℝ} (h : 0 < y) :
    ContinuousAt (fun a : ℝ≥0∞ => a ^ y) x := by
  /-
    x : ENNReal
    y : Real
    h : LT.lt 0 y
    ⊢ ContinuousAt (fun a => HPow.hPow a y) x
  -/
  by_cases hx : x = ⊤
    /-
      case pos
      x : ENNReal
      y : Real
      h : LT.lt 0 y
      hx : Eq x Top.top
      ⊢ ContinuousAt (fun a => HPow.hPow a y) x
    -/
  · rw [hx, ContinuousAt]
    /-
      case pos
      x : ENNReal
      y : Real
      h : LT.lt 0 y
      hx : Eq x Top.top
      ⊢ Filter.Tendsto (fun a => HPow.hPow a y) (nhds Top.top) (nhds (HPow.hPow Top. …
    -/
    convert ENNReal.tendsto_rpow_at_top h
    /-
      case h.e'_5.h.e'_3
      x : ENNReal
      y : Real
      h : LT.lt 0 y
      hx : Eq x Top.top
      ⊢ Eq (HPow.hPow Top.top y) Top.top
    -/
    simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : ENNReal
    y : Real
    h : LT.lt 0 y
    hx : Not (Eq x Top.top)
    ⊢ ContinuousAt (fun a => HPow.hPow a y) x
  -/
  lift x to ℝ≥0 using hx
  /-
    case neg.intro
    y : Real
    h : LT.lt 0 y
    x : NNReal
    ⊢ ContinuousAt (fun a => HPow.hPow a y) ↑x
  -/
  rw [continuousAt_coe_iff]
  /-
    case neg.intro
    y : Real
    h : LT.lt 0 y
    x : NNReal
    ⊢ ContinuousAt (Function.comp (fun a => HPow.hPow a y) ENNReal.ofNNReal) x
  -/
  convert continuous_coe.continuousAt.comp (NNReal.continuousAt_rpow_const (Or.inr h.le)) using 1
  /-
    case h.e'_5
    y : Real
    h : LT.lt 0 y
    x : NNReal
    ⊢ Eq (Function.comp (fun a => HPow.hPow a y) ENNReal.ofNNReal) (Function.comp  …
  -/
  ext1 x
  /-
    case h.e'_5.h
    y : Real
    h : LT.lt 0 y
    x✝ x : NNReal
    ⊢ Eq (Function.comp (fun a => HPow.hPow a y) ENNReal.ofNNReal x) (Function.com …
  -/
  simp [← coe_rpow_of_nonneg _ h.le]
  /-
    🎉 no goals
  -/


@[continuity, fun_prop]
theorem continuous_rpow_const {y : ℝ} : Continuous fun a : ℝ≥0∞ => a ^ y := by
  /-
    y : Real
    ⊢ Continuous fun a => HPow.hPow a y
  -/
  refine continuous_iff_continuousAt.2 fun x => ?_
  /-
    y : Real
    x : ENNReal
    ⊢ ContinuousAt (fun a => HPow.hPow a y) x
  -/
  rcases lt_trichotomy (0 : ℝ) y with (hy | rfl | hy)
    /-
      case inl
      y : Real
      x : ENNReal
      hy : LT.lt 0 y
      ⊢ ContinuousAt (fun a => HPow.hPow a y) x
    -/
  · exact continuousAt_rpow_const_of_pos hy
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      x : ENNReal
      ⊢ ContinuousAt (fun a => HPow.hPow a 0) x
    -/
  · simp only [rpow_zero]
    /-
      case inr.inl
      x : ENNReal
      ⊢ ContinuousAt (fun a => 1) x
    -/
    exact continuousAt_const
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      y : Real
      x : ENNReal
      hy : LT.lt y 0
      ⊢ ContinuousAt (fun a => HPow.hPow a y) x
    -/
  · obtain ⟨z, hz⟩ : ∃ z, y = -z := ⟨-y, (neg_neg _).symm⟩
    /-
      case inr.inr.intro
      y : Real
      x : ENNReal
      hy : LT.lt y 0
      z : Real
      hz : Eq y (Neg.neg z)
      ⊢ ContinuousAt (fun a => HPow.hPow a y) x
    -/
    have z_pos : 0 < z := by simpa [hz] using hy
    /-
      case inr.inr.intro
      y : Real
      x : ENNReal
      hy : LT.lt y 0
      z : Real
      hz : Eq y (Neg.neg z)
      z_pos : LT.lt 0 z
      ⊢ ContinuousAt (fun a => HPow.hPow a y) x
    -/
    simp_rw [hz, rpow_neg]
    /-
      case inr.inr.intro
      y : Real
      x : ENNReal
      hy : LT.lt y 0
      z : Real
      hz : Eq y (Neg.neg z)
      z_pos : LT.lt 0 z
      ⊢ ContinuousAt (fun a => Inv.inv (HPow.hPow a z)) x
    -/
    exact continuous_inv.continuousAt.comp (continuousAt_rpow_const_of_pos z_pos)
    /-
      🎉 no goals
    -/


theorem tendsto_const_mul_rpow_nhds_zero_of_pos {c : ℝ≥0∞} (hc : c ≠ ∞) {y : ℝ} (hy : 0 < y) :
    Tendsto (fun x : ℝ≥0∞ => c * x ^ y) (𝓝 0) (𝓝 0) := by
  /-
    c : ENNReal
    hc : Ne c Top.top
    y : Real
    hy : LT.lt 0 y
    ⊢ Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x y)) (nhds 0) (nhds 0)
  -/
  convert ENNReal.Tendsto.const_mul (ENNReal.continuous_rpow_const.tendsto 0) _
    /-
      case h.e'_5.h.e'_3
      c : ENNReal
      hc : Ne c Top.top
      y : Real
      hy : LT.lt 0 y
      ⊢ Eq 0 (HMul.hMul c (HPow.hPow 0 y))
    -/
  · simp [hy]
    /-
      🎉 no goals
    -/
    /-
      case convert_3
      c : ENNReal
      hc : Ne c Top.top
      y : Real
      hy : LT.lt 0 y
      ⊢ Or (Ne (HPow.hPow 0 y) 0) (Ne c Top.top)
    -/
  · exact Or.inr hc
    /-
      🎉 no goals
    -/


theorem Filter.Tendsto.ennrpow_const {α : Type*} {f : Filter α} {m : α → ℝ≥0∞} {a : ℝ≥0∞} (r : ℝ)
    (hm : Tendsto m f (𝓝 a)) : Tendsto (fun x => m x ^ r) f (𝓝 (a ^ r)) :=
  (ENNReal.continuous_rpow_const.tendsto a).comp hm

