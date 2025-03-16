/-- General fencing theorem for continuous functions with an estimate on the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `f a ≤ B a`;
* `B` has right derivative `B'` at every point of `[a, b)`;
* for each `x ∈ [a, b)` the right-side limit inferior of `(f z - f x) / (z - x)`
  is bounded above by a function `f'`;
* we have `f' x < B' x` whenever `f x = B x`.

Then `f x ≤ B x` everywhere on `[a, b]`. -/
theorem image_le_of_liminf_slope_right_lt_deriv_boundary' {f f' : ℝ → ℝ} {a b : ℝ}
    (hf : ContinuousOn f (Icc a b))
    -- `hf'` actually says `liminf (f z - f x) / (z - x) ≤ f' x`
    (hf' : ∀ x ∈ Ico a b, ∀ r, f' x < r → ∃ᶠ z in 𝓝[>] x, slope f x z < r)
    {B B' : ℝ → ℝ} (ha : f a ≤ B a) (hB : ContinuousOn B (Icc a b))
    (hB' : ∀ x ∈ Ico a b, HasDerivWithinAt B (B' x) (Ici x) x)
    (bound : ∀ x ∈ Ico a b, f x = B x → f' x < B' x) : ∀ ⦃x⦄, x ∈ Icc a b → f x ≤ B x := by
  /-
    f f' : Real → Real
    a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    B B' : Real → Real
    ha : LE.le (f a) (B a)
    hB : ContinuousOn B (Set.Icc a b)
    hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
    ⊢ ∀ ⦃x : Real⦄, Membership.mem (Set.Icc a b) x → LE.le (f x) (B x)
  -/
  change Icc a b ⊆ { x | f x ≤ B x }
  /-
    f f' : Real → Real
    a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    B B' : Real → Real
    ha : LE.le (f a) (B a)
    hB : ContinuousOn B (Set.Icc a b)
    hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
    ⊢ HasSubset.Subset (Set.Icc a b) (setOf fun x => LE.le (f x) (B x))
  -/
  set s := { x | f x ≤ B x } ∩ Icc a b
  /-
    f f' : Real → Real
    a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    B B' : Real → Real
    ha : LE.le (f a) (B a)
    hB : ContinuousOn B (Set.Icc a b)
    hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
    s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
    ⊢ HasSubset.Subset (Set.Icc a b) (setOf fun x => LE.le (f x) (B x))
  -/
  have A : ContinuousOn (fun x => (f x, B x)) (Icc a b) := hf.prod hB
  have : IsClosed s := by
    simp only [s, inter_comm]
    exact A.preimage_isClosed_of_isClosed isClosed_Icc OrderClosedTopology.isClosed_le'
  /-
    f f' : Real → Real
    a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    B B' : Real → Real
    ha : LE.le (f a) (B a)
    hB : ContinuousOn B (Set.Icc a b)
    hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
    s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
    A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
    this : IsClosed s
    ⊢ HasSubset.Subset (Set.Icc a b) (setOf fun x => LE.le (f x) (B x))
  -/
  apply this.Icc_subset_of_forall_exists_gt ha
  /-
    f f' : Real → Real
    a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    B B' : Real → Real
    ha : LE.le (f a) (B a)
    hB : ContinuousOn B (Set.Icc a b)
    hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
    s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
    A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
    this : IsClosed s
    ⊢ ∀ (x : Real), Membership.mem (Inter.inter (setOf fun x => LE.le (f x) (B x)) …
  -/
  rintro x ⟨hxB : f x ≤ B x, xab⟩ y hy
  /-
    case intro
    f f' : Real → Real
    a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
    B B' : Real → Real
    ha : LE.le (f a) (B a)
    hB : ContinuousOn B (Set.Icc a b)
    hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
    s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
    A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
    this : IsClosed s
    x : Real
    hxB : LE.le (f x) (B x)
    xab : Membership.mem (Set.Ico a b) x
    y : Real
    hy : Membership.mem (Set.Ioi x) y
    ⊢ (Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Ioc x y)).Nonempty
  -/
  cases' hxB.lt_or_eq with hxB hxB
  · -- If `f x < B x`, then all we need is continuity of both sides
    /-
      case intro.inl
      f f' : Real → Real
      a b : Real
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
      B B' : Real → Real
      ha : LE.le (f a) (B a)
      hB : ContinuousOn B (Set.Icc a b)
      hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
      bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
      s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
      A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
      this : IsClosed s
      x : Real
      hxB✝ : LE.le (f x) (B x)
      xab : Membership.mem (Set.Ico a b) x
      y : Real
      hy : Membership.mem (Set.Ioi x) y
      hxB : LT.lt (f x) (B x)
      ⊢ (Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Ioc x y)).Nonempty
    -/
    refine nonempty_of_mem (inter_mem ?_ (Ioc_mem_nhdsGT hy))
    have : ∀ᶠ x in 𝓝[Icc a b] x, f x < B x :=
      A x (Ico_subset_Icc_self xab) (IsOpen.mem_nhds (isOpen_lt continuous_fst continuous_snd) hxB)
    /-
      case intro.inl
      f f' : Real → Real
      a b : Real
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
      B B' : Real → Real
      ha : LE.le (f a) (B a)
      hB : ContinuousOn B (Set.Icc a b)
      hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
      bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
      s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
      A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
      this✝ : IsClosed s
      x : Real
      hxB✝ : LE.le (f x) (B x)
      xab : Membership.mem (Set.Ico a b) x
      y : Real
      hy : Membership.mem (Set.Ioi x) y
      hxB : LT.lt (f x) (B x)
      this : Filter.Eventually (fun x => LT.lt (f x) (B x)) (nhdsWithin x (Set.Icc a …
      ⊢ Membership.mem (nhdsWithin x (Set.Ioi x)) (setOf fun x => LE.le (f x) (B x))
    -/
    have : ∀ᶠ x in 𝓝[>] x, f x < B x := nhdsWithin_le_of_mem (Icc_mem_nhdsGT_of_mem xab) this
    /-
      case intro.inl
      f f' : Real → Real
      a b : Real
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
      B B' : Real → Real
      ha : LE.le (f a) (B a)
      hB : ContinuousOn B (Set.Icc a b)
      hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
      bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
      s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
      A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
      this✝¹ : IsClosed s
      x : Real
      hxB✝ : LE.le (f x) (B x)
      xab : Membership.mem (Set.Ico a b) x
      y : Real
      hy : Membership.mem (Set.Ioi x) y
      hxB : LT.lt (f x) (B x)
      this✝ : Filter.Eventually (fun x => LT.lt (f x) (B x)) (nhdsWithin x (Set.Icc  …
      this : Filter.Eventually (fun x => LT.lt (f x) (B x)) (nhdsWithin x (Set.Ioi x))
      ⊢ Membership.mem (nhdsWithin x (Set.Ioi x)) (setOf fun x => LE.le (f x) (B x))
    -/
    exact this.mono fun y => le_of_lt
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      f f' : Real → Real
      a b : Real
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
      B B' : Real → Real
      ha : LE.le (f a) (B a)
      hB : ContinuousOn B (Set.Icc a b)
      hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
      bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
      s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
      A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
      this : IsClosed s
      x : Real
      hxB✝ : LE.le (f x) (B x)
      xab : Membership.mem (Set.Ico a b) x
      y : Real
      hy : Membership.mem (Set.Ioi x) y
      hxB : Eq (f x) (B x)
      ⊢ (Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Ioc x y)).Nonempty
    -/
  · rcases exists_between (bound x xab hxB) with ⟨r, hfr, hrB⟩
    /-
      case intro.inr.intro.intro
      f f' : Real → Real
      a b : Real
      hf : ContinuousOn f (Set.Icc a b)
      hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (f' x …
      B B' : Real → Real
      ha : LE.le (f a) (B a)
      hB : ContinuousOn B (Set.Icc a b)
      hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
      bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
      s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
      A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
      this : IsClosed s
      x : Real
      hxB✝ : LE.le (f x) (B x)
      xab : Membership.mem (Set.Ico a b) x
      y : Real
      hy : Membership.mem (Set.Ioi x) y
      hxB : Eq (f x) (B x)
      r : Real
      hfr : LT.lt (f' x) r
      hrB : LT.lt r (B' x)
      ⊢ (Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Ioc x y)).Nonempty
    -/
    specialize hf' x xab r hfr
    have HB : ∀ᶠ z in 𝓝[>] x, r < slope B x z :=
      (hasDerivWithinAt_iff_tendsto_slope' <| lt_irrefl x).1 (hB' x xab).Ioi_of_Ici
        (Ioi_mem_nhds hrB)
    obtain ⟨z, hfz, hzB, hz⟩ : ∃ z, slope f x z < r ∧ r < slope B x z ∧ z ∈ Ioc x y :=
      hf'.and_eventually (HB.and (Ioc_mem_nhdsGT hy)) |>.exists
    /-
      case intro.inr.intro.intro.intro.intro.intro
      f f' : Real → Real
      a b : Real
      hf : ContinuousOn f (Set.Icc a b)
      B B' : Real → Real
      ha : LE.le (f a) (B a)
      hB : ContinuousOn B (Set.Icc a b)
      hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
      bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
      s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
      A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
      this : IsClosed s
      x : Real
      hxB✝ : LE.le (f x) (B x)
      xab : Membership.mem (Set.Ico a b) x
      y : Real
      hy : Membership.mem (Set.Ioi x) y
      hxB : Eq (f x) (B x)
      r : Real
      hfr : LT.lt (f' x) r
      hrB : LT.lt r (B' x)
      hf' : Filter.Frequently (fun z => LT.lt (slope f x z) r) (nhdsWithin x (Set.Io …
      HB : Filter.Eventually (fun z => LT.lt r (slope B x z)) (nhdsWithin x (Set.Ioi …
      z : Real
      hfz : LT.lt (slope f x z) r
      hzB : LT.lt r (slope B x z)
      hz : Membership.mem (Set.Ioc x y) z
      ⊢ (Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Ioc x y)).Nonempty
    -/
    refine ⟨z, ?_, hz⟩
    /-
      case intro.inr.intro.intro.intro.intro.intro
      f f' : Real → Real
      a b : Real
      hf : ContinuousOn f (Set.Icc a b)
      B B' : Real → Real
      ha : LE.le (f a) (B a)
      hB : ContinuousOn B (Set.Icc a b)
      hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
      bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → Eq (f x) (B x) → LT.lt  …
      s : Set Real := Inter.inter (setOf fun x => LE.le (f x) (B x)) (Set.Icc a b)
      A : ContinuousOn (fun x => { fst := f x, snd := B x }) (Set.Icc a b)
      this : IsClosed s
      x : Real
      hxB✝ : LE.le (f x) (B x)
      xab : Membership.mem (Set.Ico a b) x
      y : Real
      hy : Membership.mem (Set.Ioi x) y
      hxB : Eq (f x) (B x)
      r : Real
      hfr : LT.lt (f' x) r
      hrB : LT.lt r (B' x)
      hf' : Filter.Frequently (fun z => LT.lt (slope f x z) r) (nhdsWithin x (Set.Io …
      HB : Filter.Eventually (fun z => LT.lt r (slope B x z)) (nhdsWithin x (Set.Ioi …
      z : Real
      hfz : LT.lt (slope f x z) r
      hzB : LT.lt r (slope B x z)
      hz : Membership.mem (Set.Ioc x y) z
      ⊢ Membership.mem (setOf fun x => LE.le (f x) (B x)) z
    -/
    have := (hfz.trans hzB).le
    rwa [slope_def_field, slope_def_field, div_le_div_iff_of_pos_right (sub_pos.2 hz.1), hxB,
      sub_le_sub_iff_right] at this


/-- General fencing theorem for continuous functions with an estimate on the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `f a ≤ B a`;
* `B` has derivative `B'` everywhere on `ℝ`;
* for each `x ∈ [a, b)` the right-side limit inferior of `(f z - f x) / (z - x)`
  is bounded above by a function `f'`;
* we have `f' x < B' x` whenever `f x = B x`.

Then `f x ≤ B x` everywhere on `[a, b]`. -/
theorem image_le_of_liminf_slope_right_lt_deriv_boundary {f f' : ℝ → ℝ} {a b : ℝ}
    (hf : ContinuousOn f (Icc a b))
    -- `hf'` actually says `liminf (f z - f x) / (z - x) ≤ f' x`
    (hf' : ∀ x ∈ Ico a b, ∀ r, f' x < r → ∃ᶠ z in 𝓝[>] x, slope f x z < r)
    {B B' : ℝ → ℝ} (ha : f a ≤ B a) (hB : ∀ x, HasDerivAt B (B' x) x)
    (bound : ∀ x ∈ Ico a b, f x = B x → f' x < B' x) : ∀ ⦃x⦄, x ∈ Icc a b → f x ≤ B x :=
  image_le_of_liminf_slope_right_lt_deriv_boundary' hf hf' ha
    (fun x _ => (hB x).continuousAt.continuousWithinAt) (fun x _ => (hB x).hasDerivWithinAt) bound


/-- General fencing theorem for continuous functions with an estimate on the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `f a ≤ B a`;
* `B` has right derivative `B'` at every point of `[a, b)`;
* for each `x ∈ [a, b)` the right-side limit inferior of `(f z - f x) / (z - x)`
  is bounded above by `B'`.

Then `f x ≤ B x` everywhere on `[a, b]`. -/
theorem image_le_of_liminf_slope_right_le_deriv_boundary {f : ℝ → ℝ} {a b : ℝ}
    (hf : ContinuousOn f (Icc a b)) {B B' : ℝ → ℝ} (ha : f a ≤ B a) (hB : ContinuousOn B (Icc a b))
    (hB' : ∀ x ∈ Ico a b, HasDerivWithinAt B (B' x) (Ici x) x)
    -- `bound` actually says `liminf (f z - f x) / (z - x) ≤ B' x`
    (bound : ∀ x ∈ Ico a b, ∀ r, B' x < r → ∃ᶠ z in 𝓝[>] x, slope f x z < r) :
    ∀ ⦃x⦄, x ∈ Icc a b → f x ≤ B x := by
  have Hr : ∀ x ∈ Icc a b, ∀ r > 0, f x ≤ B x + r * (x - a) := fun x hx r hr => by
    apply image_le_of_liminf_slope_right_lt_deriv_boundary' hf bound
    · rwa [sub_self, mul_zero, add_zero]
    · exact hB.add (continuousOn_const.mul (continuousOn_id.sub continuousOn_const))
    · intro x hx
      exact (hB' x hx).add (((hasDerivWithinAt_id x (Ici x)).sub_const a).const_mul r)
    · intro x _ _
      rw [mul_one]
      exact (lt_add_iff_pos_right _).2 hr
    exact hx
  /-
    f : Real → Real
    a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    B B' : Real → Real
    ha : LE.le (f a) (B a)
    hB : ContinuousOn B (Set.Icc a b)
    hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (B' …
    Hr : ∀ (x : Real), Membership.mem (Set.Icc a b) x → ∀ (r : Real), GT.gt r 0 →  …
    ⊢ ∀ ⦃x : Real⦄, Membership.mem (Set.Icc a b) x → LE.le (f x) (B x)
  -/
  intro x hx
  have : ContinuousWithinAt (fun r => B x + r * (x - a)) (Ioi 0) 0 :=
    continuousWithinAt_const.add (continuousWithinAt_id.mul continuousWithinAt_const)
  /-
    f : Real → Real
    a b : Real
    hf : ContinuousOn f (Set.Icc a b)
    B B' : Real → Real
    ha : LE.le (f a) (B a)
    hB : ContinuousOn B (Set.Icc a b)
    hB' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt B (B' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → ∀ (r : Real), LT.lt (B' …
    Hr : ∀ (x : Real), Membership.mem (Set.Icc a b) x → ∀ (r : Real), GT.gt r 0 →  …
    x : Real
    hx : Membership.mem (Set.Icc a b) x
    this : ContinuousWithinAt (fun r => HAdd.hAdd (B x) (HMul.hMul r (HSub.hSub x  …
    ⊢ LE.le (f x) (B x)
  -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  convert continuousWithinAt_const.closure_le _ this (Hr x hx) using 1 <;> simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- General fencing theorem for continuous functions with an estimate on the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `f a ≤ B a`;
* `B` has right derivative `B'` at every point of `[a, b)`;
* `f` has right derivative `f'` at every point of `[a, b)`;
* we have `f' x < B' x` whenever `f x = B x`.

Then `f x ≤ B x` everywhere on `[a, b]`. -/
theorem image_le_of_deriv_right_lt_deriv_boundary' {f f' : ℝ → ℝ} {a b : ℝ}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    {B B' : ℝ → ℝ} (ha : f a ≤ B a) (hB : ContinuousOn B (Icc a b))
    (hB' : ∀ x ∈ Ico a b, HasDerivWithinAt B (B' x) (Ici x) x)
    (bound : ∀ x ∈ Ico a b, f x = B x → f' x < B' x) : ∀ ⦃x⦄, x ∈ Icc a b → f x ≤ B x :=
  image_le_of_liminf_slope_right_lt_deriv_boundary' hf
    (fun x hx _ hr => (hf' x hx).liminf_right_slope_le hr) ha hB hB' bound


/-- General fencing theorem for continuous functions with an estimate on the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `f a ≤ B a`;
* `B` has derivative `B'` everywhere on `ℝ`;
* `f` has right derivative `f'` at every point of `[a, b)`;
* we have `f' x < B' x` whenever `f x = B x`.

Then `f x ≤ B x` everywhere on `[a, b]`. -/
theorem image_le_of_deriv_right_lt_deriv_boundary {f f' : ℝ → ℝ} {a b : ℝ}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    {B B' : ℝ → ℝ} (ha : f a ≤ B a) (hB : ∀ x, HasDerivAt B (B' x) x)
    (bound : ∀ x ∈ Ico a b, f x = B x → f' x < B' x) : ∀ ⦃x⦄, x ∈ Icc a b → f x ≤ B x :=
  image_le_of_deriv_right_lt_deriv_boundary' hf hf' ha
    (fun x _ => (hB x).continuousAt.continuousWithinAt) (fun x _ => (hB x).hasDerivWithinAt) bound


/-- General fencing theorem for continuous functions with an estimate on the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `f a ≤ B a`;
* `B` has derivative `B'` everywhere on `ℝ`;
* `f` has right derivative `f'` at every point of `[a, b)`;
* we have `f' x ≤ B' x` on `[a, b)`.

Then `f x ≤ B x` everywhere on `[a, b]`. -/
theorem image_le_of_deriv_right_le_deriv_boundary {f f' : ℝ → ℝ} {a b : ℝ}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    {B B' : ℝ → ℝ} (ha : f a ≤ B a) (hB : ContinuousOn B (Icc a b))
    (hB' : ∀ x ∈ Ico a b, HasDerivWithinAt B (B' x) (Ici x) x)
    (bound : ∀ x ∈ Ico a b, f' x ≤ B' x) : ∀ ⦃x⦄, x ∈ Icc a b → f x ≤ B x :=
  image_le_of_liminf_slope_right_le_deriv_boundary hf ha hB hB' fun x hx _ hr =>
    (hf' x hx).liminf_right_slope_le (lt_of_le_of_lt (bound x hx) hr)


/-- General fencing theorem for continuous functions with an estimate on the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `‖f a‖ ≤ B a`;
* `B` has right derivative at every point of `[a, b)`;
* for each `x ∈ [a, b)` the right-side limit inferior of `(‖f z‖ - ‖f x‖) / (z - x)`
  is bounded above by a function `f'`;
* we have `f' x < B' x` whenever `‖f x‖ = B x`.

Then `‖f x‖ ≤ B x` everywhere on `[a, b]`. -/
theorem image_norm_le_of_liminf_right_slope_norm_lt_deriv_boundary {E : Type*}
    [NormedAddCommGroup E] {f : ℝ → E} {f' : ℝ → ℝ} (hf : ContinuousOn f (Icc a b))
    -- `hf'` actually says `liminf (‖f z‖ - ‖f x‖) / (z - x) ≤ f' x`
    (hf' : ∀ x ∈ Ico a b, ∀ r, f' x < r → ∃ᶠ z in 𝓝[>] x, slope (norm ∘ f) x z < r)
    {B B' : ℝ → ℝ} (ha : ‖f a‖ ≤ B a) (hB : ContinuousOn B (Icc a b))
    (hB' : ∀ x ∈ Ico a b, HasDerivWithinAt B (B' x) (Ici x) x)
    (bound : ∀ x ∈ Ico a b, ‖f x‖ = B x → f' x < B' x) : ∀ ⦃x⦄, x ∈ Icc a b → ‖f x‖ ≤ B x :=
  image_le_of_liminf_slope_right_lt_deriv_boundary' (continuous_norm.comp_continuousOn hf) hf' ha hB
    hB' bound


/-- General fencing theorem for continuous functions with an estimate on the norm of the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `‖f a‖ ≤ B a`;
* `f` and `B` have right derivatives `f'` and `B'` respectively at every point of `[a, b)`;
* the norm of `f'` is strictly less than `B'` whenever `‖f x‖ = B x`.

Then `‖f x‖ ≤ B x` everywhere on `[a, b]`. We use one-sided derivatives in the assumptions
to make this theorem work for piecewise differentiable functions.
-/
theorem image_norm_le_of_norm_deriv_right_lt_deriv_boundary' {f' : ℝ → E}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    {B B' : ℝ → ℝ} (ha : ‖f a‖ ≤ B a) (hB : ContinuousOn B (Icc a b))
    (hB' : ∀ x ∈ Ico a b, HasDerivWithinAt B (B' x) (Ici x) x)
    (bound : ∀ x ∈ Ico a b, ‖f x‖ = B x → ‖f' x‖ < B' x) : ∀ ⦃x⦄, x ∈ Icc a b → ‖f x‖ ≤ B x :=
  image_norm_le_of_liminf_right_slope_norm_lt_deriv_boundary hf
    (fun x hx _ hr => (hf' x hx).liminf_right_slope_norm_le hr) ha hB hB' bound


/-- General fencing theorem for continuous functions with an estimate on the norm of the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `‖f a‖ ≤ B a`;
* `f` has right derivative `f'` at every point of `[a, b)`;
* `B` has derivative `B'` everywhere on `ℝ`;
* the norm of `f'` is strictly less than `B'` whenever `‖f x‖ = B x`.

Then `‖f x‖ ≤ B x` everywhere on `[a, b]`. We use one-sided derivatives in the assumptions
to make this theorem work for piecewise differentiable functions.
-/
theorem image_norm_le_of_norm_deriv_right_lt_deriv_boundary {f' : ℝ → E}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    {B B' : ℝ → ℝ} (ha : ‖f a‖ ≤ B a) (hB : ∀ x, HasDerivAt B (B' x) x)
    (bound : ∀ x ∈ Ico a b, ‖f x‖ = B x → ‖f' x‖ < B' x) : ∀ ⦃x⦄, x ∈ Icc a b → ‖f x‖ ≤ B x :=
  image_norm_le_of_norm_deriv_right_lt_deriv_boundary' hf hf' ha
    (fun x _ => (hB x).continuousAt.continuousWithinAt) (fun x _ => (hB x).hasDerivWithinAt) bound


/-- General fencing theorem for continuous functions with an estimate on the norm of the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `‖f a‖ ≤ B a`;
* `f` and `B` have right derivatives `f'` and `B'` respectively at every point of `[a, b)`;
* we have `‖f' x‖ ≤ B x` everywhere on `[a, b)`.

Then `‖f x‖ ≤ B x` everywhere on `[a, b]`. We use one-sided derivatives in the assumptions
to make this theorem work for piecewise differentiable functions.
-/
theorem image_norm_le_of_norm_deriv_right_le_deriv_boundary' {f' : ℝ → E}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    {B B' : ℝ → ℝ} (ha : ‖f a‖ ≤ B a) (hB : ContinuousOn B (Icc a b))
    (hB' : ∀ x ∈ Ico a b, HasDerivWithinAt B (B' x) (Ici x) x)
    (bound : ∀ x ∈ Ico a b, ‖f' x‖ ≤ B' x) : ∀ ⦃x⦄, x ∈ Icc a b → ‖f x‖ ≤ B x :=
  image_le_of_liminf_slope_right_le_deriv_boundary (continuous_norm.comp_continuousOn hf) ha hB hB'
    fun x hx _ hr => (hf' x hx).liminf_right_slope_norm_le ((bound x hx).trans_lt hr)


/-- General fencing theorem for continuous functions with an estimate on the norm of the derivative.
Let `f` and `B` be continuous functions on `[a, b]` such that

* `‖f a‖ ≤ B a`;
* `f` has right derivative `f'` at every point of `[a, b)`;
* `B` has derivative `B'` everywhere on `ℝ`;
* we have `‖f' x‖ ≤ B x` everywhere on `[a, b)`.

Then `‖f x‖ ≤ B x` everywhere on `[a, b]`. We use one-sided derivatives in the assumptions
to make this theorem work for piecewise differentiable functions.
-/
theorem image_norm_le_of_norm_deriv_right_le_deriv_boundary {f' : ℝ → E}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    {B B' : ℝ → ℝ} (ha : ‖f a‖ ≤ B a) (hB : ∀ x, HasDerivAt B (B' x) x)
    (bound : ∀ x ∈ Ico a b, ‖f' x‖ ≤ B' x) : ∀ ⦃x⦄, x ∈ Icc a b → ‖f x‖ ≤ B x :=
  image_norm_le_of_norm_deriv_right_le_deriv_boundary' hf hf' ha
    (fun x _ => (hB x).continuousAt.continuousWithinAt) (fun x _ => (hB x).hasDerivWithinAt) bound


/-- A function on `[a, b]` with the norm of the right derivative bounded by `C`
satisfies `‖f x - f a‖ ≤ C * (x - a)`. -/
theorem norm_image_sub_le_of_norm_deriv_right_le_segment {f' : ℝ → E} {C : ℝ}
    (hf : ContinuousOn f (Icc a b)) (hf' : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    (bound : ∀ x ∈ Ico a b, ‖f' x‖ ≤ C) : ∀ x ∈ Icc a b, ‖f x - f a‖ ≤ C * (x - a) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    f' : Real → E
    C : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt f (f' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (f' x) …
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.norm (HSub.hSub ( …
  -/
  let g x := f x - f a
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    f' : Real → E
    C : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt f (f' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (f' x) …
    g : Real → E := fun x => HSub.hSub (f x) (f a)
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.norm (HSub.hSub ( …
  -/
  have hg : ContinuousOn g (Icc a b) := hf.sub continuousOn_const
  have hg' : ∀ x ∈ Ico a b, HasDerivWithinAt g (f' x) (Ici x) x := by
    intro x hx
    simpa using (hf' x hx).sub (hasDerivWithinAt_const _ _ _)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    f' : Real → E
    C : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt f (f' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (f' x) …
    g : Real → E := fun x => HSub.hSub (f x) (f a)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt g (f' x) …
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.norm (HSub.hSub ( …
  -/
  let B x := C * (x - a)
  have hB : ∀ x, HasDerivAt B C x := by
    intro x
    simpa using (hasDerivAt_const x C).mul ((hasDerivAt_id x).sub (hasDerivAt_const x a))
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    f' : Real → E
    C : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt f (f' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (f' x) …
    g : Real → E := fun x => HSub.hSub (f x) (f a)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt g (f' x) …
    B : Real → Real := fun x => HMul.hMul C (HSub.hSub x a)
    hB : ∀ (x : Real), HasDerivAt B C x
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.norm (HSub.hSub ( …
  -/
  convert image_norm_le_of_norm_deriv_right_le_deriv_boundary hg hg' _ hB bound
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    f' : Real → E
    C : Real
    hf : ContinuousOn f (Set.Icc a b)
    hf' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt f (f' x) …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (f' x) …
    g : Real → E := fun x => HSub.hSub (f x) (f a)
    hg : ContinuousOn g (Set.Icc a b)
    hg' : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt g (f' x) …
    B : Real → Real := fun x => HMul.hMul C (HSub.hSub x a)
    hB : ∀ (x : Real), HasDerivAt B C x
    ⊢ LE.le (Norm.norm (g a)) (B a)
  -/
  simp only [g, B]; rw [sub_self, norm_zero, sub_self, mul_zero]
                    /-
                      🎉 no goals
                    -/


/-- A function on `[a, b]` with the norm of the derivative within `[a, b]`
bounded by `C` satisfies `‖f x - f a‖ ≤ C * (x - a)`, `HasDerivWithinAt`
version. -/
theorem norm_image_sub_le_of_norm_deriv_le_segment' {f' : ℝ → E} {C : ℝ}
    (hf : ∀ x ∈ Icc a b, HasDerivWithinAt f (f' x) (Icc a b) x)
    (bound : ∀ x ∈ Ico a b, ‖f' x‖ ≤ C) : ∀ x ∈ Icc a b, ‖f x - f a‖ ≤ C * (x - a) := by
  refine
    norm_image_sub_le_of_norm_deriv_right_le_segment (fun x hx => (hf x hx).continuousWithinAt)
      (fun x hx => ?_) bound
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    f' : Real → E
    C : Real
    hf : ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt f (f' x)  …
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (f' x) …
    x : Real
    hx : Membership.mem (Set.Ico a b) x
    ⊢ HasDerivWithinAt f (f' x) (Set.Ici x) x
  -/
  exact (hf x <| Ico_subset_Icc_self hx).mono_of_mem_nhdsWithin (Icc_mem_nhdsGE_of_mem hx)
  /-
    🎉 no goals
  -/


/-- A function on `[a, b]` with the norm of the derivative within `[a, b]`
bounded by `C` satisfies `‖f x - f a‖ ≤ C * (x - a)`, `derivWithin`
version. -/
theorem norm_image_sub_le_of_norm_deriv_le_segment {C : ℝ} (hf : DifferentiableOn ℝ f (Icc a b))
    (bound : ∀ x ∈ Ico a b, ‖derivWithin f (Icc a b) x‖ ≤ C) :
    ∀ x ∈ Icc a b, ‖f x - f a‖ ≤ C * (x - a) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b C : Real
    hf : DifferentiableOn Real f (Set.Icc a b)
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (deriv …
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.norm (HSub.hSub ( …
  -/
  refine norm_image_sub_le_of_norm_deriv_le_segment' ?_ bound
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b C : Real
    hf : DifferentiableOn Real f (Set.Icc a b)
    bound : ∀ (x : Real), Membership.mem (Set.Ico a b) x → LE.le (Norm.norm (deriv …
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → HasDerivWithinAt f (derivWith …
  -/
  exact fun x hx => (hf x hx).hasDerivWithinAt
  /-
    🎉 no goals
  -/


/-- A function on `[0, 1]` with the norm of the derivative within `[0, 1]`
bounded by `C` satisfies `‖f 1 - f 0‖ ≤ C`, `HasDerivWithinAt`
version. -/
theorem norm_image_sub_le_of_norm_deriv_le_segment_01' {f' : ℝ → E} {C : ℝ}
    (hf : ∀ x ∈ Icc (0 : ℝ) 1, HasDerivWithinAt f (f' x) (Icc (0 : ℝ) 1) x)
    (bound : ∀ x ∈ Ico (0 : ℝ) 1, ‖f' x‖ ≤ C) : ‖f 1 - f 0‖ ≤ C := by
  simpa only [sub_zero, mul_one] using
    norm_image_sub_le_of_norm_deriv_le_segment' hf bound 1 (right_mem_Icc.2 zero_le_one)


/-- A function on `[0, 1]` with the norm of the derivative within `[0, 1]`
bounded by `C` satisfies `‖f 1 - f 0‖ ≤ C`, `derivWithin` version. -/
theorem norm_image_sub_le_of_norm_deriv_le_segment_01 {C : ℝ}
    (hf : DifferentiableOn ℝ f (Icc (0 : ℝ) 1))
    (bound : ∀ x ∈ Ico (0 : ℝ) 1, ‖derivWithin f (Icc (0 : ℝ) 1) x‖ ≤ C) : ‖f 1 - f 0‖ ≤ C := by
  simpa only [sub_zero, mul_one] using
    norm_image_sub_le_of_norm_deriv_le_segment hf bound 1 (right_mem_Icc.2 zero_le_one)


theorem constant_of_has_deriv_right_zero (hcont : ContinuousOn f (Icc a b))
    (hderiv : ∀ x ∈ Ico a b, HasDerivWithinAt f 0 (Ici x) x) : ∀ x ∈ Icc a b, f x = f a := by
  have : ∀ x ∈ Icc a b, ‖f x - f a‖ ≤ 0 * (x - a) := fun x hx =>
    norm_image_sub_le_of_norm_deriv_right_le_segment hcont hderiv (fun _ _ => norm_zero.le) x hx
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    hcont : ContinuousOn f (Set.Icc a b)
    hderiv : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt f 0 ( …
    this : ∀ (x : Real), Membership.mem (Set.Icc a b) x → LE.le (Norm.norm (HSub.h …
    ⊢ ∀ (x : Real), Membership.mem (Set.Icc a b) x → Eq (f x) (f a)
  -/
  simpa only [zero_mul, norm_le_zero_iff, sub_eq_zero] using this
  /-
    🎉 no goals
  -/


theorem constant_of_derivWithin_zero (hdiff : DifferentiableOn ℝ f (Icc a b))
    (hderiv : ∀ x ∈ Ico a b, derivWithin f (Icc a b) x = 0) : ∀ x ∈ Icc a b, f x = f a := by
  have H : ∀ x ∈ Ico a b, ‖derivWithin f (Icc a b) x‖ ≤ 0 := by
    simpa only [norm_le_zero_iff] using fun x hx => hderiv x hx
  simpa only [zero_mul, norm_le_zero_iff, sub_eq_zero] using fun x hx =>
    norm_image_sub_le_of_norm_deriv_le_segment hdiff H x hx


/-- If two continuous functions on `[a, b]` have the same right derivative and are equal at `a`,
  then they are equal everywhere on `[a, b]`. -/
theorem eq_of_has_deriv_right_eq (derivf : ∀ x ∈ Ico a b, HasDerivWithinAt f (f' x) (Ici x) x)
    (derivg : ∀ x ∈ Ico a b, HasDerivWithinAt g (f' x) (Ici x) x) (fcont : ContinuousOn f (Icc a b))
    (gcont : ContinuousOn g (Icc a b)) (hi : f a = g a) : ∀ y ∈ Icc a b, f y = g y := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : Real → E
    a b : Real
    f' g : Real → E
    derivf : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt f (f' …
    derivg : ∀ (x : Real), Membership.mem (Set.Ico a b) x → HasDerivWithinAt g (f' …
    fcont : ContinuousOn f (Set.Icc a b)
    gcont : ContinuousOn g (Set.Icc a b)
    hi : Eq (f a) (g a)
    ⊢ ∀ (y : Real), Membership.mem (Set.Icc a b) y → Eq (f y) (g y)
  -/
  simp only [← @sub_eq_zero _ _ (f _)] at hi ⊢
  exact hi ▸ constant_of_has_deriv_right_zero (fcont.sub gcont) fun y hy => by
    simpa only [sub_self] using (derivf y hy).sub (derivg y hy)


/-- If two differentiable functions on `[a, b]` have the same derivative within `[a, b]` everywhere
  on `[a, b)` and are equal at `a`, then they are equal everywhere on `[a, b]`. -/
theorem eq_of_derivWithin_eq (fdiff : DifferentiableOn ℝ f (Icc a b))
    (gdiff : DifferentiableOn ℝ g (Icc a b))
    (hderiv : EqOn (derivWithin f (Icc a b)) (derivWithin g (Icc a b)) (Ico a b)) (hi : f a = g a) :
    ∀ y ∈ Icc a b, f y = g y := by
  have A : ∀ y ∈ Ico a b, HasDerivWithinAt f (derivWithin f (Icc a b) y) (Ici y) y := fun y hy =>
    (fdiff y (mem_Icc_of_Ico hy)).hasDerivWithinAt.mono_of_mem_nhdsWithin
    (Icc_mem_nhdsGE_of_mem hy)
  have B : ∀ y ∈ Ico a b, HasDerivWithinAt g (derivWithin g (Icc a b) y) (Ici y) y := fun y hy =>
    (gdiff y (mem_Icc_of_Ico hy)).hasDerivWithinAt.mono_of_mem_nhdsWithin
    (Icc_mem_nhdsGE_of_mem hy)
  exact eq_of_has_deriv_right_eq A (fun y hy => (hderiv hy).symm ▸ B y hy) fdiff.continuousOn
    gdiff.continuousOn hi


instance (priority := 100) : PathConnectedSpace 𝕜 := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f g : E → G
    C : Real
    s : Set E
    x y : E
    f' g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    φ : ContinuousLinearMap (RingHom.id 𝕜) E G
    ⊢ PathConnectedSpace 𝕜
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    F : Type u_2
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace Real F
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f g : E → G
    C : Real
    s : Set E
    x y : E
    f' g' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    φ : ContinuousLinearMap (RingHom.id 𝕜) E G
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ PathConnectedSpace 𝕜
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The mean value theorem on a convex set: if the derivative of a function is bounded by `C`, then
the function is `C`-Lipschitz. Version with `HasFDerivWithinAt`. -/
theorem norm_image_sub_le_of_norm_hasFDerivWithin_le
    (hf : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (bound : ∀ x ∈ s, ‖f' x‖ ≤ C) (hs : Convex ℝ s)
    (xs : x ∈ s) (ys : y ∈ s) : ‖f y - f x‖ ≤ C * ‖y - x‖ := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    C : Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm (f' x)) C
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    ⊢ LE.le (Norm.norm (HSub.hSub (f y) (f x))) (HMul.hMul C (Norm.norm (HSub.hSub …
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    C : Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm (f' x)) C
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ LE.le (Norm.norm (HSub.hSub (f y) (f x))) (HMul.hMul C (Norm.norm (HSub.hSub …
  -/
  letI : NormedSpace ℝ G := RestrictScalars.normedSpace ℝ 𝕜 G
  /- By composition with `AffineMap.lineMap x y`, we reduce to a statement for functions defined
    on `[0,1]`, for which it is proved in `norm_image_sub_le_of_norm_deriv_le_segment`.
    We just have to check the differentiability of the composition and bounds on its derivative,
    which is straightforward but tedious for lack of automation. -/
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    C : Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm (f' x)) C
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this : NormedSpace Real G := RestrictScalars.normedSpace Real 𝕜 G
    ⊢ LE.le (Norm.norm (HSub.hSub (f y) (f x))) (HMul.hMul C (Norm.norm (HSub.hSub …
  -/
  set g := (AffineMap.lineMap x y : ℝ → E)
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    C : Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm (f' x)) C
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this : NormedSpace Real G := RestrictScalars.normedSpace Real 𝕜 G
    g : Real → E := ⇑(AffineMap.lineMap x y)
    ⊢ LE.le (Norm.norm (HSub.hSub (f y) (f x))) (HMul.hMul C (Norm.norm (HSub.hSub …
  -/
  have segm : MapsTo g (Icc 0 1 : Set ℝ) s := hs.mapsTo_lineMap xs ys
  have hD : ∀ t ∈ Icc (0 : ℝ) 1,
      HasDerivWithinAt (f ∘ g) (f' (g t) (y - x)) (Icc 0 1) t := fun t ht => by
    simpa using ((hf (g t) (segm ht)).restrictScalars ℝ).comp_hasDerivWithinAt _
      AffineMap.hasDerivWithinAt_lineMap segm
  have bound : ∀ t ∈ Ico (0 : ℝ) 1, ‖f' (g t) (y - x)‖ ≤ C * ‖y - x‖ := fun t ht =>
    le_of_opNorm_le _ (bound _ <| segm <| Ico_subset_Icc_self ht) _
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    C : Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound✝ : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm (f' x)) C
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    this✝ : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    this : NormedSpace Real G := RestrictScalars.normedSpace Real 𝕜 G
    g : Real → E := ⇑(AffineMap.lineMap x y)
    segm : Set.MapsTo g (Set.Icc 0 1) s
    hD : ∀ (t : Real), Membership.mem (Set.Icc 0 1) t → HasDerivWithinAt (Function …
    bound : ∀ (t : Real), Membership.mem (Set.Ico 0 1) t → LE.le (Norm.norm ((f' ( …
    ⊢ LE.le (Norm.norm (HSub.hSub (f y) (f x))) (HMul.hMul C (Norm.norm (HSub.hSub …
  -/
  simpa [g] using norm_image_sub_le_of_norm_deriv_le_segment_01' hD bound
  /-
    🎉 no goals
  -/


/-- The mean value theorem on a convex set: if the derivative of a function is bounded by `C` on
`s`, then the function is `C`-Lipschitz on `s`. Version with `HasFDerivWithinAt` and
`LipschitzOnWith`. -/
theorem lipschitzOnWith_of_nnnorm_hasFDerivWithin_le {C : ℝ≥0}
    (hf : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (bound : ∀ x ∈ s, ‖f' x‖₊ ≤ C)
    (hs : Convex ℝ s) : LipschitzOnWith C f s := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    C : NNReal
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound : ∀ (x : E), Membership.mem s x → LE.le (NNNorm.nnnorm (f' x)) C
    hs : Convex Real s
    ⊢ LipschitzOnWith C f s
  -/
  rw [lipschitzOnWith_iff_norm_sub_le]
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    C : NNReal
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound : ∀ (x : E), Membership.mem s x → LE.le (NNNorm.nnnorm (f' x)) C
    hs : Convex Real s
    ⊢ ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃y : E⦄, Membership.mem s y → LE.le (Norm. …
  -/
  intro x x_in y y_in
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    s : Set E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    C : NNReal
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound : ∀ (x : E), Membership.mem s x → LE.le (NNNorm.nnnorm (f' x)) C
    hs : Convex Real s
    x : E
    x_in : Membership.mem s x
    y : E
    y_in : Membership.mem s y
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (f y))) (HMul.hMul (↑C) (Norm.norm (HSub.h …
  -/
  exact hs.norm_image_sub_le_of_norm_hasFDerivWithin_le hf bound y_in x_in
  /-
    🎉 no goals
  -/


/-- Let `s` be a convex set in a real normed vector space `E`, let `f : E → G` be a function
differentiable within `s` in a neighborhood of `x : E` with derivative `f'`. Suppose that `f'` is
continuous within `s` at `x`. Then for any number `K : ℝ≥0` larger than `‖f' x‖₊`, `f` is
`K`-Lipschitz on some neighborhood of `x` within `s`. See also
`Convex.exists_nhdsWithin_lipschitzOnWith_of_hasFDerivWithinAt` for a version that claims
existence of `K` instead of an explicit estimate. -/
theorem exists_nhdsWithin_lipschitzOnWith_of_hasFDerivWithinAt_of_nnnorm_lt (hs : Convex ℝ s)
    {f : E → G} (hder : ∀ᶠ y in 𝓝[s] x, HasFDerivWithinAt f (f' y) s y)
    (hcont : ContinuousWithinAt f' s x) (K : ℝ≥0) (hK : ‖f' x‖₊ < K) :
    ∃ t ∈ 𝓝[s] x, LipschitzOnWith K f t := by
  obtain ⟨ε, ε0, hε⟩ : ∃ ε > 0,
      ball x ε ∩ s ⊆ { y | HasFDerivWithinAt f (f' y) s y ∧ ‖f' y‖₊ < K } :=
    mem_nhdsWithin_iff.1 (hder.and <| hcont.nnnorm.eventually (gt_mem_nhds hK))
  /-
    case intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    x : E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    hs : Convex Real s
    f : E → G
    hder : Filter.Eventually (fun y => HasFDerivWithinAt f (f' y) s y) (nhdsWithin …
    hcont : ContinuousWithinAt f' s x
    K : NNReal
    hK : LT.lt (NNNorm.nnnorm (f' x)) K
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Inter.inter (Metric.ball x ε) s) (setOf fun y => And (H …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (LipschitzOnWith K f …
  -/
  rw [inter_comm] at hε
  /-
    case intro.intro
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    s : Set E
    x : E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    hs : Convex Real s
    f : E → G
    hder : Filter.Eventually (fun y => HasFDerivWithinAt f (f' y) s y) (nhdsWithin …
    hcont : ContinuousWithinAt f' s x
    K : NNReal
    hK : LT.lt (NNNorm.nnnorm (f' x)) K
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Inter.inter s (Metric.ball x ε)) (setOf fun y => And (H …
    ⊢ Exists fun t => And (Membership.mem (nhdsWithin x s) t) (LipschitzOnWith K f …
  -/
  refine ⟨s ∩ ball x ε, inter_mem_nhdsWithin _ (ball_mem_nhds _ ε0), ?_⟩
  exact
    (hs.inter (convex_ball _ _)).lipschitzOnWith_of_nnnorm_hasFDerivWithin_le
      (fun y hy => (hε hy).1.mono inter_subset_left) fun y hy => (hε hy).2.le


/-- Let `s` be a convex set in a real normed vector space `E`, let `f : E → G` be a function
differentiable within `s` in a neighborhood of `x : E` with derivative `f'`. Suppose that `f'` is
continuous within `s` at `x`. Then for any number `K : ℝ≥0` larger than `‖f' x‖₊`, `f` is Lipschitz
on some neighborhood of `x` within `s`. See also
`Convex.exists_nhdsWithin_lipschitzOnWith_of_hasFDerivWithinAt_of_nnnorm_lt` for a version
with an explicit estimate on the Lipschitz constant. -/
theorem exists_nhdsWithin_lipschitzOnWith_of_hasFDerivWithinAt (hs : Convex ℝ s) {f : E → G}
    (hder : ∀ᶠ y in 𝓝[s] x, HasFDerivWithinAt f (f' y) s y) (hcont : ContinuousWithinAt f' s x) :
    ∃ K, ∃ t ∈ 𝓝[s] x, LipschitzOnWith K f t :=
  (exists_gt _).imp <|
    hs.exists_nhdsWithin_lipschitzOnWith_of_hasFDerivWithinAt_of_nnnorm_lt hder hcont


/-- The mean value theorem on a convex set: if the derivative of a function within this set is
bounded by `C`, then the function is `C`-Lipschitz. Version with `fderivWithin`. -/
theorem norm_image_sub_le_of_norm_fderivWithin_le (hf : DifferentiableOn 𝕜 f s)
    (bound : ∀ x ∈ s, ‖fderivWithin 𝕜 f s x‖ ≤ C) (hs : Convex ℝ s) (xs : x ∈ s) (ys : y ∈ s) :
    ‖f y - f x‖ ≤ C * ‖y - x‖ :=
  hs.norm_image_sub_le_of_norm_hasFDerivWithin_le (fun x hx => (hf x hx).hasFDerivWithinAt) bound
    xs ys


/-- The mean value theorem on a convex set: if the derivative of a function is bounded by `C` on
`s`, then the function is `C`-Lipschitz on `s`. Version with `fderivWithin` and
`LipschitzOnWith`. -/
theorem lipschitzOnWith_of_nnnorm_fderivWithin_le {C : ℝ≥0} (hf : DifferentiableOn 𝕜 f s)
    (bound : ∀ x ∈ s, ‖fderivWithin 𝕜 f s x‖₊ ≤ C) (hs : Convex ℝ s) : LipschitzOnWith C f s :=
  hs.lipschitzOnWith_of_nnnorm_hasFDerivWithin_le (fun x hx => (hf x hx).hasFDerivWithinAt) bound


/-- The mean value theorem on a convex set: if the derivative of a function is bounded by `C`,
then the function is `C`-Lipschitz. Version with `fderiv`. -/
theorem norm_image_sub_le_of_norm_fderiv_le (hf : ∀ x ∈ s, DifferentiableAt 𝕜 f x)
    (bound : ∀ x ∈ s, ‖fderiv 𝕜 f x‖ ≤ C) (hs : Convex ℝ s) (xs : x ∈ s) (ys : y ∈ s) :
    ‖f y - f x‖ ≤ C * ‖y - x‖ :=
  hs.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun x hx => (hf x hx).hasFDerivAt.hasFDerivWithinAt) bound xs ys


/-- The mean value theorem on a convex set: if the derivative of a function is bounded by `C` on
`s`, then the function is `C`-Lipschitz on `s`. Version with `fderiv` and `LipschitzOnWith`. -/
theorem lipschitzOnWith_of_nnnorm_fderiv_le {C : ℝ≥0} (hf : ∀ x ∈ s, DifferentiableAt 𝕜 f x)
    (bound : ∀ x ∈ s, ‖fderiv 𝕜 f x‖₊ ≤ C) (hs : Convex ℝ s) : LipschitzOnWith C f s :=
  hs.lipschitzOnWith_of_nnnorm_hasFDerivWithin_le
    (fun x hx => (hf x hx).hasFDerivAt.hasFDerivWithinAt) bound


/-- The mean value theorem: if the derivative of a function is bounded by `C`, then the function is
`C`-Lipschitz. Version with `fderiv` and `LipschitzWith`. -/
theorem _root_.lipschitzWith_of_nnnorm_fderiv_le
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] {f : E → G}
    {C : ℝ≥0} (hf : Differentiable 𝕜 f)
    (bound : ∀ x, ‖fderiv 𝕜 f x‖₊ ≤ C) : LipschitzWith C f := by
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → G
    C : NNReal
    hf : Differentiable 𝕜 f
    bound : ∀ (x : E), LE.le (NNNorm.nnnorm (fderiv 𝕜 f x)) C
    ⊢ LipschitzWith C f
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → G
    C : NNReal
    hf : Differentiable 𝕜 f
    bound : ∀ (x : E), LE.le (NNNorm.nnnorm (fderiv 𝕜 f x)) C
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ LipschitzWith C f
  -/
  let A : NormedSpace ℝ E := RestrictScalars.normedSpace ℝ 𝕜 E
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → G
    C : NNReal
    hf : Differentiable 𝕜 f
    bound : ∀ (x : E), LE.le (NNNorm.nnnorm (fderiv 𝕜 f x)) C
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    A : NormedSpace Real E := RestrictScalars.normedSpace Real 𝕜 E
    ⊢ LipschitzWith C f
  -/
  rw [← lipschitzOnWith_univ]
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → G
    C : NNReal
    hf : Differentiable 𝕜 f
    bound : ∀ (x : E), LE.le (NNNorm.nnnorm (fderiv 𝕜 f x)) C
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    A : NormedSpace Real E := RestrictScalars.normedSpace Real 𝕜 E
    ⊢ LipschitzOnWith C f Set.univ
  -/
  exact lipschitzOnWith_of_nnnorm_fderiv_le (fun x _ ↦ hf x) (fun x _ ↦ bound x) convex_univ
  /-
    🎉 no goals
  -/


/-- Variant of the mean value inequality on a convex set, using a bound on the difference between
the derivative and a fixed linear map, rather than a bound on the derivative itself. Version with
`HasFDerivWithinAt`. -/
theorem norm_image_sub_le_of_norm_hasFDerivWithin_le'
    (hf : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (bound : ∀ x ∈ s, ‖f' x - φ‖ ≤ C)
    (hs : Convex ℝ s) (xs : x ∈ s) (ys : y ∈ s) : ‖f y - f x - φ (y - x)‖ ≤ C * ‖y - x‖ := by
  /- We subtract `φ` to define a new function `g` for which `g' = 0`, for which the previous theorem
    applies, `Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le`. Then, we just need to glue
    together the pieces, expressing back `f` in terms of `g`. -/
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    C : Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E G
    φ : ContinuousLinearMap (RingHom.id 𝕜) E G
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    bound : ∀ (x : E), Membership.mem s x → LE.le (Norm.norm (HSub.hSub (f' x) φ)) C
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f y) (f x)) (φ (HSub.hSub y x)))) (H …
  -/
  let g y := f y - φ y
  have hg : ∀ x ∈ s, HasFDerivWithinAt g (f' x - φ) s x := fun x xs =>
    (hf x xs).sub φ.hasFDerivWithinAt
  calc
    ‖f y - f x - φ (y - x)‖ = ‖f y - f x - (φ y - φ x)‖ := by simp
    _ = ‖f y - φ y - (f x - φ x)‖ := by congr 1; abel
    _ = ‖g y - g x‖ := by simp [g]
    _ ≤ C * ‖y - x‖ := Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le hg bound hs xs ys


/-- Variant of the mean value inequality on a convex set. Version with `fderivWithin`. -/
theorem norm_image_sub_le_of_norm_fderivWithin_le' (hf : DifferentiableOn 𝕜 f s)
    (bound : ∀ x ∈ s, ‖fderivWithin 𝕜 f s x - φ‖ ≤ C) (hs : Convex ℝ s) (xs : x ∈ s) (ys : y ∈ s) :
    ‖f y - f x - φ (y - x)‖ ≤ C * ‖y - x‖ :=
  hs.norm_image_sub_le_of_norm_hasFDerivWithin_le' (fun x hx => (hf x hx).hasFDerivWithinAt) bound
    xs ys


/-- Variant of the mean value inequality on a convex set. Version with `fderiv`. -/
theorem norm_image_sub_le_of_norm_fderiv_le' (hf : ∀ x ∈ s, DifferentiableAt 𝕜 f x)
    (bound : ∀ x ∈ s, ‖fderiv 𝕜 f x - φ‖ ≤ C) (hs : Convex ℝ s) (xs : x ∈ s) (ys : y ∈ s) :
    ‖f y - f x - φ (y - x)‖ ≤ C * ‖y - x‖ :=
  hs.norm_image_sub_le_of_norm_hasFDerivWithin_le'
    (fun x hx => (hf x hx).hasFDerivAt.hasFDerivWithinAt) bound xs ys


/-- If a function has zero Fréchet derivative at every point of a convex set,
then it is a constant on this set. -/
theorem is_const_of_fderivWithin_eq_zero (hs : Convex ℝ s) (hf : DifferentiableOn 𝕜 f s)
    (hf' : ∀ x ∈ s, fderivWithin 𝕜 f s x = 0) (hx : x ∈ s) (hy : y ∈ s) : f x = f y := by
  have bound : ∀ x ∈ s, ‖fderivWithin 𝕜 f s x‖ ≤ 0 := fun x hx => by
    simp only [hf' x hx, norm_zero, le_rfl]
  simpa only [(dist_eq_norm _ _).symm, zero_mul, dist_le_zero, eq_comm] using
    hs.norm_image_sub_le_of_norm_fderivWithin_le hf bound hx hy


theorem _root_.is_const_of_fderiv_eq_zero
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] {f : E → G}
    (hf : Differentiable 𝕜 f) (hf' : ∀ x, fderiv 𝕜 f x = 0)
    (x y : E) : f x = f y := by
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → G
    hf : Differentiable 𝕜 f
    hf' : ∀ (x : E), Eq (fderiv 𝕜 f x) 0
    x y : E
    ⊢ Eq (f x) (f y)
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : E → G
    hf : Differentiable 𝕜 f
    hf' : ∀ (x : E), Eq (fderiv 𝕜 f x) 0
    x y : E
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ Eq (f x) (f y)
  -/
  let A : NormedSpace ℝ E := RestrictScalars.normedSpace ℝ 𝕜 E
  exact convex_univ.is_const_of_fderivWithin_eq_zero hf.differentiableOn
    (fun x _ => by rw [fderivWithin_univ]; exact hf' x) trivial trivial


/-- If two functions have equal Fréchet derivatives at every point of a convex set, and are equal at
one point in that set, then they are equal on that set. -/
theorem eqOn_of_fderivWithin_eq (hs : Convex ℝ s) (hf : DifferentiableOn 𝕜 f s)
    (hg : DifferentiableOn 𝕜 g s) (hs' : UniqueDiffOn 𝕜 s)
    (hf' : ∀ x ∈ s, fderivWithin 𝕜 f s x = fderivWithin 𝕜 g s x) (hx : x ∈ s) (hfgx : f x = g x) :
    s.EqOn f g := fun y hy => by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f g : E → G
    s : Set E
    x : E
    hs : Convex Real s
    hf : DifferentiableOn 𝕜 f s
    hg : DifferentiableOn 𝕜 g s
    hs' : UniqueDiffOn 𝕜 s
    hf' : ∀ (x : E), Membership.mem s x → Eq (fderivWithin 𝕜 f s x) (fderivWithin  …
    hx : Membership.mem s x
    hfgx : Eq (f x) (g x)
    y : E
    hy : Membership.mem s y
    ⊢ Eq (f y) (g y)
  -/
  suffices f x - g x = f y - g y by rwa [hfgx, sub_self, eq_comm, sub_eq_zero] at this
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f g : E → G
    s : Set E
    x : E
    hs : Convex Real s
    hf : DifferentiableOn 𝕜 f s
    hg : DifferentiableOn 𝕜 g s
    hs' : UniqueDiffOn 𝕜 s
    hf' : ∀ (x : E), Membership.mem s x → Eq (fderivWithin 𝕜 f s x) (fderivWithin  …
    hx : Membership.mem s x
    hfgx : Eq (f x) (g x)
    y : E
    hy : Membership.mem s y
    ⊢ Eq (HSub.hSub (f x) (g x)) (HSub.hSub (f y) (g y))
  -/
  refine hs.is_const_of_fderivWithin_eq_zero (hf.sub hg) (fun z hz => ?_) hx hy
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : IsRCLikeNormedField 𝕜
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f g : E → G
    s : Set E
    x : E
    hs : Convex Real s
    hf : DifferentiableOn 𝕜 f s
    hg : DifferentiableOn 𝕜 g s
    hs' : UniqueDiffOn 𝕜 s
    hf' : ∀ (x : E), Membership.mem s x → Eq (fderivWithin 𝕜 f s x) (fderivWithin  …
    hx : Membership.mem s x
    hfgx : Eq (f x) (g x)
    y : E
    hy : Membership.mem s y
    z : E
    hz : Membership.mem s z
    ⊢ Eq (fderivWithin 𝕜 (fun y => HSub.hSub (f y) (g y)) s z) 0
  -/
  rw [fderivWithin_sub (hs' _ hz) (hf _ hz) (hg _ hz), sub_eq_zero, hf' _ hz]
  /-
    🎉 no goals
  -/


theorem _root_.eq_of_fderiv_eq
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] {f g : E → G}
    (hf : Differentiable 𝕜 f) (hg : Differentiable 𝕜 g)
    (hf' : ∀ x, fderiv 𝕜 f x = fderiv 𝕜 g x) (x : E) (hfgx : f x = g x) : f = g := by
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : E → G
    hf : Differentiable 𝕜 f
    hg : Differentiable 𝕜 g
    hf' : ∀ (x : E), Eq (fderiv 𝕜 f x) (fderiv 𝕜 g x)
    x : E
    hfgx : Eq (f x) (g x)
    ⊢ Eq f g
  -/
  letI : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : E → G
    hf : Differentiable 𝕜 f
    hg : Differentiable 𝕜 g
    hf' : ∀ (x : E), Eq (fderiv 𝕜 f x) (fderiv 𝕜 g x)
    x : E
    hfgx : Eq (f x) (g x)
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    ⊢ Eq f g
  -/
  let A : NormedSpace ℝ E := RestrictScalars.normedSpace ℝ 𝕜 E
  /-
    𝕜 : Type u_3
    G : Type u_4
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : IsRCLikeNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : E → G
    hf : Differentiable 𝕜 f
    hg : Differentiable 𝕜 g
    hf' : ∀ (x : E), Eq (fderiv 𝕜 f x) (fderiv 𝕜 g x)
    x : E
    hfgx : Eq (f x) (g x)
    this : RCLike 𝕜 := IsRCLikeNormedField.rclike 𝕜
    A : NormedSpace Real E := RestrictScalars.normedSpace Real 𝕜 E
    ⊢ Eq f g
  -/
  suffices Set.univ.EqOn f g from funext fun x => this <| mem_univ x
  exact convex_univ.eqOn_of_fderivWithin_eq hf.differentiableOn hg.differentiableOn
    uniqueDiffOn_univ (fun x _ => by simpa using hf' _) (mem_univ _) hfgx


/-- The mean value theorem on a convex set in dimension 1: if the derivative of a function is
bounded by `C`, then the function is `C`-Lipschitz. Version with `HasDerivWithinAt`. -/
theorem norm_image_sub_le_of_norm_hasDerivWithin_le {C : ℝ}
    (hf : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) (bound : ∀ x ∈ s, ‖f' x‖ ≤ C) (hs : Convex ℝ s)
    (xs : x ∈ s) (ys : y ∈ s) : ‖f y - f x‖ ≤ C * ‖y - x‖ :=
  Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le (fun x hx => (hf x hx).hasFDerivWithinAt)
                              /-
                                𝕜 : Type u_3
                                G : Type u_4
                                inst✝² : RCLike 𝕜
                                inst✝¹ : NormedAddCommGroup G
                                inst✝ : NormedSpace 𝕜 G
                                f f' : 𝕜 → G
                                s : Set 𝕜
                                x✝ y : 𝕜
                                C : Real
                                hf : ∀ (x : 𝕜), Membership.mem s x → HasDerivWithinAt f (f' x) s x
                                bound : ∀ (x : 𝕜), Membership.mem s x → LE.le (Norm.norm (f' x)) C
                                hs : Convex Real s
                                xs : Membership.mem s x✝
                                ys : Membership.mem s y
                                x : 𝕜
                                hx : Membership.mem s x
                                ⊢ LE.le (Norm.norm (ContinuousLinearMap.smulRight 1 (f' x))) (Norm.norm (f' x))
                              -/
    (fun x hx => le_trans (by simp) (bound x hx)) hs xs ys
                              /-
                                🎉 no goals
                              -/


/-- The mean value theorem on a convex set in dimension 1: if the derivative of a function is
bounded by `C` on `s`, then the function is `C`-Lipschitz on `s`.
Version with `HasDerivWithinAt` and `LipschitzOnWith`. -/
theorem lipschitzOnWith_of_nnnorm_hasDerivWithin_le {C : ℝ≥0} (hs : Convex ℝ s)
    (hf : ∀ x ∈ s, HasDerivWithinAt f (f' x) s x) (bound : ∀ x ∈ s, ‖f' x‖₊ ≤ C) :
    LipschitzOnWith C f s :=
  Convex.lipschitzOnWith_of_nnnorm_hasFDerivWithin_le (fun x hx => (hf x hx).hasFDerivWithinAt)
                              /-
                                𝕜 : Type u_3
                                G : Type u_4
                                inst✝² : RCLike 𝕜
                                inst✝¹ : NormedAddCommGroup G
                                inst✝ : NormedSpace 𝕜 G
                                f f' : 𝕜 → G
                                s : Set 𝕜
                                C : NNReal
                                hs : Convex Real s
                                hf : ∀ (x : 𝕜), Membership.mem s x → HasDerivWithinAt f (f' x) s x
                                bound : ∀ (x : 𝕜), Membership.mem s x → LE.le (NNNorm.nnnorm (f' x)) C
                                x : 𝕜
                                hx : Membership.mem s x
                                ⊢ LE.le (NNNorm.nnnorm (ContinuousLinearMap.smulRight 1 (f' x))) (NNNorm.nnnor …
                              -/
    (fun x hx => le_trans (by simp) (bound x hx)) hs
                              /-
                                🎉 no goals
                              -/


/-- The mean value theorem on a convex set in dimension 1: if the derivative of a function within
this set is bounded by `C`, then the function is `C`-Lipschitz. Version with `derivWithin` -/
theorem norm_image_sub_le_of_norm_derivWithin_le {C : ℝ} (hf : DifferentiableOn 𝕜 f s)
    (bound : ∀ x ∈ s, ‖derivWithin f s x‖ ≤ C) (hs : Convex ℝ s) (xs : x ∈ s) (ys : y ∈ s) :
    ‖f y - f x‖ ≤ C * ‖y - x‖ :=
  hs.norm_image_sub_le_of_norm_hasDerivWithin_le (fun x hx => (hf x hx).hasDerivWithinAt) bound xs
    ys


/-- The mean value theorem on a convex set in dimension 1: if the derivative of a function is
bounded by `C` on `s`, then the function is `C`-Lipschitz on `s`.
Version with `derivWithin` and `LipschitzOnWith`. -/
theorem lipschitzOnWith_of_nnnorm_derivWithin_le {C : ℝ≥0} (hs : Convex ℝ s)
    (hf : DifferentiableOn 𝕜 f s) (bound : ∀ x ∈ s, ‖derivWithin f s x‖₊ ≤ C) :
    LipschitzOnWith C f s :=
  hs.lipschitzOnWith_of_nnnorm_hasDerivWithin_le (fun x hx => (hf x hx).hasDerivWithinAt) bound


/-- The mean value theorem on a convex set in dimension 1: if the derivative of a function is
bounded by `C`, then the function is `C`-Lipschitz. Version with `deriv`. -/
theorem norm_image_sub_le_of_norm_deriv_le {C : ℝ} (hf : ∀ x ∈ s, DifferentiableAt 𝕜 f x)
    (bound : ∀ x ∈ s, ‖deriv f x‖ ≤ C) (hs : Convex ℝ s) (xs : x ∈ s) (ys : y ∈ s) :
    ‖f y - f x‖ ≤ C * ‖y - x‖ :=
  hs.norm_image_sub_le_of_norm_hasDerivWithin_le
    (fun x hx => (hf x hx).hasDerivAt.hasDerivWithinAt) bound xs ys


/-- The mean value theorem on a convex set in dimension 1: if the derivative of a function is
bounded by `C` on `s`, then the function is `C`-Lipschitz on `s`.
Version with `deriv` and `LipschitzOnWith`. -/
theorem lipschitzOnWith_of_nnnorm_deriv_le {C : ℝ≥0} (hf : ∀ x ∈ s, DifferentiableAt 𝕜 f x)
    (bound : ∀ x ∈ s, ‖deriv f x‖₊ ≤ C) (hs : Convex ℝ s) : LipschitzOnWith C f s :=
  hs.lipschitzOnWith_of_nnnorm_hasDerivWithin_le
    (fun x hx => (hf x hx).hasDerivAt.hasDerivWithinAt) bound


/-- The mean value theorem set in dimension 1: if the derivative of a function is bounded by `C`,
then the function is `C`-Lipschitz. Version with `deriv` and `LipschitzWith`. -/
theorem _root_.lipschitzWith_of_nnnorm_deriv_le {C : ℝ≥0} (hf : Differentiable 𝕜 f)
    (bound : ∀ x, ‖deriv f x‖₊ ≤ C) : LipschitzWith C f :=
  lipschitzOnWith_univ.1 <|
    convex_univ.lipschitzOnWith_of_nnnorm_deriv_le (fun x _ => hf x) fun x _ => bound x


/-- If `f : 𝕜 → G`, `𝕜 = R` or `𝕜 = ℂ`, is differentiable everywhere and its derivative equal zero,
then it is a constant function. -/
theorem _root_.is_const_of_deriv_eq_zero (hf : Differentiable 𝕜 f) (hf' : ∀ x, deriv f x = 0)
    (x y : 𝕜) : f x = f y :=
                                             /-
                                               𝕜 : Type u_3
                                               G : Type u_4
                                               inst✝² : RCLike 𝕜
                                               inst✝¹ : NormedAddCommGroup G
                                               inst✝ : NormedSpace 𝕜 G
                                               f : 𝕜 → G
                                               hf : Differentiable 𝕜 f
                                               hf' : ∀ (x : 𝕜), Eq (deriv f x) 0
                                               x y z : 𝕜
                                               ⊢ Eq (fderiv 𝕜 f z) 0
                                             -/
  is_const_of_fderiv_eq_zero hf (fun z => by ext; simp [← deriv_fderiv, hf']) _ _
                                                  /-
                                                    🎉 no goals
                                                  -/


include hab hfc hff' hgc hgg' in
/-- Cauchy's **Mean Value Theorem**, `HasDerivAt` version. -/
theorem exists_ratio_hasDerivAt_eq_ratio_slope :
    ∃ c ∈ Ioo a b, (g b - g a) * f' c = (f b - f a) * g' c := by
  /-
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    g g' : Real → Real
    hgc : ContinuousOn g (Set.Icc a b)
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  let h x := (g b - g a) * f x - (f b - f a) * g x
  /-
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    g g' : Real → Real
    hgc : ContinuousOn g (Set.Icc a b)
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    h : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub (g b) (g a)) (f x) …
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  have hI : h a = h b := by simp only [h]; ring
  /-
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    g g' : Real → Real
    hgc : ContinuousOn g (Set.Icc a b)
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    h : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub (g b) (g a)) (f x) …
    hI : Eq (h a) (h b)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  let h' x := (g b - g a) * f' x - (f b - f a) * g' x
  have hhh' : ∀ x ∈ Ioo a b, HasDerivAt h (h' x) x := fun x hx =>
    ((hff' x hx).const_mul (g b - g a)).sub ((hgg' x hx).const_mul (f b - f a))
  have hhc : ContinuousOn h (Icc a b) :=
    (continuousOn_const.mul hfc).sub (continuousOn_const.mul hgc)
  /-
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    g g' : Real → Real
    hgc : ContinuousOn g (Set.Icc a b)
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    h : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub (g b) (g a)) (f x) …
    hI : Eq (h a) (h b)
    h' : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub (g b) (g a)) (f'  …
    hhh' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt h (h' x) x
    hhc : ContinuousOn h (Set.Icc a b)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  rcases exists_hasDerivAt_eq_zero hab hhc hI hhh' with ⟨c, cmem, hc⟩
  /-
    case intro.intro
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    g g' : Real → Real
    hgc : ContinuousOn g (Set.Icc a b)
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    h : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub (g b) (g a)) (f x) …
    hI : Eq (h a) (h b)
    h' : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub (g b) (g a)) (f'  …
    hhh' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt h (h' x) x
    hhc : ContinuousOn h (Set.Icc a b)
    c : Real
    cmem : Membership.mem (Set.Ioo a b) c
    hc : Eq (h' c) 0
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  exact ⟨c, cmem, sub_eq_zero.1 hc⟩
  /-
    🎉 no goals
  -/


include hab in
/-- Cauchy's **Mean Value Theorem**, extended `HasDerivAt` version. -/
theorem exists_ratio_hasDerivAt_eq_ratio_slope' {lfa lga lfb lgb : ℝ}
    (hff' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x) (hgg' : ∀ x ∈ Ioo a b, HasDerivAt g (g' x) x)
    (hfa : Tendsto f (𝓝[>] a) (𝓝 lfa)) (hga : Tendsto g (𝓝[>] a) (𝓝 lga))
    (hfb : Tendsto f (𝓝[<] b) (𝓝 lfb)) (hgb : Tendsto g (𝓝[<] b) (𝓝 lgb)) :
    ∃ c ∈ Ioo a b, (lgb - lga) * f' c = (lfb - lfa) * g' c := by
  /-
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    g g' : Real → Real
    lfa lga lfb lgb : Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds lfa)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds lga)
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lfb)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds lgb)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  let h x := (lgb - lga) * f x - (lfb - lfa) * g x
  have hha : Tendsto h (𝓝[>] a) (𝓝 <| lgb * lfa - lfb * lga) := by
    have : Tendsto h (𝓝[>] a) (𝓝 <| (lgb - lga) * lfa - (lfb - lfa) * lga) :=
      (tendsto_const_nhds.mul hfa).sub (tendsto_const_nhds.mul hga)
    convert this using 2
    ring
  have hhb : Tendsto h (𝓝[<] b) (𝓝 <| lgb * lfa - lfb * lga) := by
    have : Tendsto h (𝓝[<] b) (𝓝 <| (lgb - lga) * lfb - (lfb - lfa) * lgb) :=
      (tendsto_const_nhds.mul hfb).sub (tendsto_const_nhds.mul hgb)
    convert this using 2
    ring
  /-
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    g g' : Real → Real
    lfa lga lfb lgb : Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds lfa)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds lga)
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lfb)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds lgb)
    h : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub lgb lga) (f x)) (H …
    hha : Filter.Tendsto h (nhdsWithin a (Set.Ioi a)) (nhds (HSub.hSub (HMul.hMul  …
    hhb : Filter.Tendsto h (nhdsWithin b (Set.Iio b)) (nhds (HSub.hSub (HMul.hMul  …
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  let h' x := (lgb - lga) * f' x - (lfb - lfa) * g' x
  have hhh' : ∀ x ∈ Ioo a b, HasDerivAt h (h' x) x := by
    intro x hx
    exact ((hff' x hx).const_mul _).sub ((hgg' x hx).const_mul _)
  /-
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    g g' : Real → Real
    lfa lga lfb lgb : Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds lfa)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds lga)
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lfb)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds lgb)
    h : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub lgb lga) (f x)) (H …
    hha : Filter.Tendsto h (nhdsWithin a (Set.Ioi a)) (nhds (HSub.hSub (HMul.hMul  …
    hhb : Filter.Tendsto h (nhdsWithin b (Set.Iio b)) (nhds (HSub.hSub (HMul.hMul  …
    h' : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub lgb lga) (f' x))  …
    hhh' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt h (h' x) x
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  rcases exists_hasDerivAt_eq_zero' hab hha hhb hhh' with ⟨c, cmem, hc⟩
  /-
    case intro.intro
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    g g' : Real → Real
    lfa lga lfb lgb : Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds lfa)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds lga)
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lfb)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds lgb)
    h : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub lgb lga) (f x)) (H …
    hha : Filter.Tendsto h (nhdsWithin a (Set.Ioi a)) (nhds (HSub.hSub (HMul.hMul  …
    hhb : Filter.Tendsto h (nhdsWithin b (Set.Iio b)) (nhds (HSub.hSub (HMul.hMul  …
    h' : Real → Real := fun x => HSub.hSub (HMul.hMul (HSub.hSub lgb lga) (f' x))  …
    hhh' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt h (h' x) x
    c : Real
    cmem : Membership.mem (Set.Ioo a b) c
    hc : Eq (h' c) 0
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (HMul.hMul (HSub.hS …
  -/
  exact ⟨c, cmem, sub_eq_zero.1 hc⟩
  /-
    🎉 no goals
  -/


include hab hfc hff' in
/-- Lagrange's Mean Value Theorem, `HasDerivAt` version -/
theorem exists_hasDerivAt_eq_slope : ∃ c ∈ Ioo a b, f' c = (f b - f a) / (b - a) := by
  obtain ⟨c, cmem, hc⟩ : ∃ c ∈ Ioo a b, (b - a) * f' c = (f b - f a) * 1 :=
    exists_ratio_hasDerivAt_eq_ratio_slope f f' hab hfc hff' id 1 continuousOn_id
      fun x _ => hasDerivAt_id x
  /-
    case intro.intro
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    c : Real
    cmem : Membership.mem (Set.Ioo a b) c
    hc : Eq (HMul.hMul (HSub.hSub b a) (f' c)) (HMul.hMul (HSub.hSub (f b) (f a)) 1)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (f' c) (HDiv.hDiv ( …
  -/
  use c, cmem
  /-
    case right
    f f' : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    c : Real
    cmem : Membership.mem (Set.Ioo a b) c
    hc : Eq (HMul.hMul (HSub.hSub b a) (f' c)) (HMul.hMul (HSub.hSub (f b) (f a)) 1)
    ⊢ Eq (f' c) (HDiv.hDiv (HSub.hSub (f b) (f a)) (HSub.hSub b a))
  -/
  rwa [mul_one, mul_comm, ← eq_div_iff (sub_ne_zero.2 hab.ne')] at hc
  /-
    🎉 no goals
  -/


include hab hfc hgc hgd hfd in
/-- Cauchy's Mean Value Theorem, `deriv` version. -/
theorem exists_ratio_deriv_eq_ratio_slope :
    ∃ c ∈ Ioo a b, (g b - g a) * deriv f c = (f b - f a) * deriv g c :=
  exists_ratio_hasDerivAt_eq_ratio_slope f (deriv f) hab hfc
    (fun x hx => ((hfd x hx).differentiableAt <| IsOpen.mem_nhds isOpen_Ioo hx).hasDerivAt) g
    (deriv g) hgc fun x hx =>
    ((hgd x hx).differentiableAt <| IsOpen.mem_nhds isOpen_Ioo hx).hasDerivAt


include hab in
/-- Cauchy's Mean Value Theorem, extended `deriv` version. -/
theorem exists_ratio_deriv_eq_ratio_slope' {lfa lga lfb lgb : ℝ}
    (hdf : DifferentiableOn ℝ f <| Ioo a b) (hdg : DifferentiableOn ℝ g <| Ioo a b)
    (hfa : Tendsto f (𝓝[>] a) (𝓝 lfa)) (hga : Tendsto g (𝓝[>] a) (𝓝 lga))
    (hfb : Tendsto f (𝓝[<] b) (𝓝 lfb)) (hgb : Tendsto g (𝓝[<] b) (𝓝 lgb)) :
    ∃ c ∈ Ioo a b, (lgb - lga) * deriv f c = (lfb - lfa) * deriv g c :=
  exists_ratio_hasDerivAt_eq_ratio_slope' _ _ hab _ _
    (fun x hx => ((hdf x hx).differentiableAt <| Ioo_mem_nhds hx.1 hx.2).hasDerivAt)
    (fun x hx => ((hdg x hx).differentiableAt <| Ioo_mem_nhds hx.1 hx.2).hasDerivAt) hfa hga hfb hgb


include hab hfc hfd in
/-- Lagrange's **Mean Value Theorem**, `deriv` version. -/
theorem exists_deriv_eq_slope : ∃ c ∈ Ioo a b, deriv f c = (f b - f a) / (b - a) :=
  exists_hasDerivAt_eq_slope f (deriv f) hab hfc fun x hx =>
    ((hfd x hx).differentiableAt <| IsOpen.mem_nhds isOpen_Ioo hx).hasDerivAt


include hab hfc hfd in
/-- Lagrange's **Mean Value Theorem**, `deriv` version. -/
theorem exists_deriv_eq_slope' : ∃ c ∈ Ioo a b, deriv f c = slope f a b := by
  /-
    f : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hfd : DifferentiableOn Real f (Set.Ioo a b)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (deriv f c) (slope  …
  -/
  rw [slope_def_field]
  /-
    f : Real → Real
    a b : Real
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hfd : DifferentiableOn Real f (Set.Ioo a b)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (deriv f c) (HDiv.h …
  -/
  exact exists_deriv_eq_slope f hab hfc hfd
  /-
    🎉 no goals
  -/


/-- A real function whose derivative tends to infinity from the right at a point is not
differentiable on the right at that point -/
theorem not_differentiableWithinAt_of_deriv_tendsto_atTop_Ioi (f : ℝ → ℝ) {a : ℝ}
    (hf : Tendsto (deriv f) (𝓝[>] a) atTop) : ¬ DifferentiableWithinAt ℝ f (Ioi a) a := by
  replace hf : Tendsto (derivWithin f (Ioi a)) (𝓝[>] a) atTop := by
    refine hf.congr' ?_
    filter_upwards [eventually_mem_nhdsWithin] with x hx
    have : Ioi a ∈ 𝓝 x := by simp [← mem_interior_iff_mem_nhds, hx]
    exact (derivWithin_of_mem_nhds this).symm
  /-
    f : Real → Real
    a : Real
    hf : Filter.Tendsto (derivWithin f (Set.Ioi a)) (nhdsWithin a (Set.Ioi a)) Fil …
    ⊢ Not (DifferentiableWithinAt Real f (Set.Ioi a) a)
  -/
  by_cases hcont_at_a : ContinuousWithinAt f (Ici a) a
  case neg =>
    intro hcontra
    have := hcontra.continuousWithinAt
    rw [← ContinuousWithinAt.diff_iff this] at hcont_at_a
    simp at hcont_at_a
  case pos =>
    intro hdiff
    replace hdiff := hdiff.hasDerivWithinAt
    rw [hasDerivWithinAt_iff_tendsto_slope, Set.diff_singleton_eq_self not_mem_Ioi_self] at hdiff
    have h₀ : ∀ᶠ b in 𝓝[>] a,
        ∀ x ∈ Ioc a b, max (derivWithin f (Ioi a) a + 1) 0 < derivWithin f (Ioi a) x := by
      rw [(nhdsGT_basis a).eventually_iff]
      rw [(nhdsGT_basis a).tendsto_left_iff] at hf
      obtain ⟨b, hab, hb⟩ := hf (Ioi (max (derivWithin f (Ioi a) a + 1) 0)) (Ioi_mem_atTop _)
      refine ⟨b, hab, fun x hx z hz => ?_⟩
      simp only [MapsTo, mem_Ioo, mem_Ioi, and_imp] at hb
      exact hb hz.1 <| hz.2.trans_lt hx.2
    have h₁ : ∀ᶠ b in 𝓝[>] a, slope f a b < derivWithin f (Ioi a) a + 1 := by
      rw [(nhds_basis_Ioo _).tendsto_right_iff] at hdiff
      specialize hdiff ⟨derivWithin f (Ioi a) a - 1, derivWithin f (Ioi a) a + 1⟩ <| by simp
      filter_upwards [hdiff] with z hz using hz.2
    have hcontra : ∀ᶠ _ in 𝓝[>] a, False := by
      filter_upwards [h₀, h₁, eventually_mem_nhdsWithin] with b hb hslope (hab : a < b)
      have hdiff' : DifferentiableOn ℝ f (Ioc a b) := fun z hz => by
        refine DifferentiableWithinAt.mono (t := Ioi a) ?_ Ioc_subset_Ioi_self
        have : derivWithin f (Ioi a) z ≠ 0 := ne_of_gt <| by
          simp_all only [mem_Ioo, and_imp, mem_Ioc, max_lt_iff]
        exact differentiableWithinAt_of_derivWithin_ne_zero this
      have hcont_Ioc : ∀ z ∈ Ioc a b, ContinuousWithinAt f (Icc a b) z := by
        intro z hz''
        refine (hdiff'.continuousOn z hz'').mono_of_mem_nhdsWithin ?_
        have hfinal : 𝓝[Ioc a b] z = 𝓝[Icc a b] z := by
          refine nhdsWithin_eq_nhdsWithin' (s := Ioi a) (Ioi_mem_nhds hz''.1) ?_
          simp only [Ioc_inter_Ioi, le_refl, sup_of_le_left]
          ext y
          exact ⟨fun h => ⟨mem_Icc_of_Ioc h, mem_of_mem_inter_left h⟩, fun ⟨H1, H2⟩ => ⟨H2, H1.2⟩⟩
        rw [← hfinal]
        exact self_mem_nhdsWithin
      have hcont : ContinuousOn f (Icc a b) := by
        intro z hz
        by_cases hz' : z = a
        · rw [hz']
          exact hcont_at_a.mono Icc_subset_Ici_self
        · exact hcont_Ioc z ⟨lt_of_le_of_ne hz.1 (Ne.symm hz'), hz.2⟩
      obtain ⟨x, hx₁, hx₂⟩ :=
        exists_deriv_eq_slope' f hab hcont (hdiff'.mono (Ioo_subset_Ioc_self))
      specialize hb x ⟨hx₁.1, le_of_lt hx₁.2⟩
      replace hx₂ : derivWithin f (Ioi a) x = slope f a b := by
        have : Ioi a ∈ 𝓝 x := by simp [← mem_interior_iff_mem_nhds, hx₁.1]
        rwa [derivWithin_of_mem_nhds this]
      rw [hx₂, max_lt_iff] at hb
      linarith
    simp [Filter.eventually_false_iff_eq_bot, ← not_mem_closure_iff_nhdsWithin_eq_bot] at hcontra


/-- A real function whose derivative tends to minus infinity from the right at a point is not
differentiable on the right at that point -/
theorem not_differentiableWithinAt_of_deriv_tendsto_atBot_Ioi (f : ℝ → ℝ) {a : ℝ}
    (hf : Tendsto (deriv f) (𝓝[>] a) atBot) : ¬ DifferentiableWithinAt ℝ f (Ioi a) a := by
  /-
    f : Real → Real
    a : Real
    hf : Filter.Tendsto (deriv f) (nhdsWithin a (Set.Ioi a)) Filter.atBot
    ⊢ Not (DifferentiableWithinAt Real f (Set.Ioi a) a)
  -/
  intro h
  have hf' : Tendsto (deriv (-f)) (𝓝[>] a) atTop := by
    rw [Pi.neg_def, deriv.neg']
    exact tendsto_neg_atBot_atTop.comp hf
  /-
    f : Real → Real
    a : Real
    hf : Filter.Tendsto (deriv f) (nhdsWithin a (Set.Ioi a)) Filter.atBot
    h : DifferentiableWithinAt Real f (Set.Ioi a) a
    hf' : Filter.Tendsto (deriv (Neg.neg f)) (nhdsWithin a (Set.Ioi a)) Filter.atTop
    ⊢ False
  -/
  exact not_differentiableWithinAt_of_deriv_tendsto_atTop_Ioi (-f) hf' h.neg
  /-
    🎉 no goals
  -/


/-- A real function whose derivative tends to minus infinity from the left at a point is not
differentiable on the left at that point -/
theorem not_differentiableWithinAt_of_deriv_tendsto_atBot_Iio (f : ℝ → ℝ) {a : ℝ}
    (hf : Tendsto (deriv f) (𝓝[<] a) atBot) : ¬ DifferentiableWithinAt ℝ f (Iio a) a := by
  /-
    f : Real → Real
    a : Real
    hf : Filter.Tendsto (deriv f) (nhdsWithin a (Set.Iio a)) Filter.atBot
    ⊢ Not (DifferentiableWithinAt Real f (Set.Iio a) a)
  -/
  let f' := f ∘ Neg.neg
  have hderiv : deriv f' =ᶠ[𝓝[>] (-a)] -(deriv f ∘ Neg.neg) := by
    rw [atBot_basis.tendsto_right_iff] at hf
    specialize hf (-1) trivial
    rw [(nhdsLT_basis a).eventually_iff] at hf
    rw [EventuallyEq, (nhdsGT_basis (-a)).eventually_iff]
    obtain ⟨b, hb₁, hb₂⟩ := hf
    refine ⟨-b, by linarith, fun x hx => ?_⟩
    simp only [Pi.neg_apply, Function.comp_apply]
    suffices deriv f' x = deriv f (-x) * deriv (Neg.neg : ℝ → ℝ) x by simpa using this
    refine deriv_comp x (differentiableAt_of_deriv_ne_zero ?_) (by fun_prop)
    rw [mem_Ioo] at hx
    have h₁ : -x ∈ Ioo b a := ⟨by linarith, by linarith⟩
    have h₂ : deriv f (-x) ≤ -1 := hb₂ h₁
    exact ne_of_lt (by linarith)
  have hmain : ¬ DifferentiableWithinAt ℝ f' (Ioi (-a)) (-a) := by
    refine not_differentiableWithinAt_of_deriv_tendsto_atTop_Ioi f' <| Tendsto.congr' hderiv.symm ?_
    refine Tendsto.comp (g := -deriv f) ?_ tendsto_neg_nhdsGT_neg
    exact Tendsto.comp (g := Neg.neg) tendsto_neg_atBot_atTop hf
  /-
    f : Real → Real
    a : Real
    hf : Filter.Tendsto (deriv f) (nhdsWithin a (Set.Iio a)) Filter.atBot
    f' : Real → Real := Function.comp f Neg.neg
    hderiv : (nhdsWithin (Neg.neg a) (Set.Ioi (Neg.neg a))).EventuallyEq (deriv f' …
    hmain : Not (DifferentiableWithinAt Real f' (Set.Ioi (Neg.neg a)) (Neg.neg a))
    ⊢ Not (DifferentiableWithinAt Real f (Set.Iio a) a)
  -/
  intro h
  have : DifferentiableWithinAt ℝ f' (Ioi (-a)) (-a) := by
    refine DifferentiableWithinAt.comp (g := f) (f := Neg.neg) (t := Iio a) (-a) ?_ ?_ ?_
    · simp [h]
    · fun_prop
    · intro x
      simp [neg_lt]
  /-
    f : Real → Real
    a : Real
    hf : Filter.Tendsto (deriv f) (nhdsWithin a (Set.Iio a)) Filter.atBot
    f' : Real → Real := Function.comp f Neg.neg
    hderiv : (nhdsWithin (Neg.neg a) (Set.Ioi (Neg.neg a))).EventuallyEq (deriv f' …
    hmain : Not (DifferentiableWithinAt Real f' (Set.Ioi (Neg.neg a)) (Neg.neg a))
    h : DifferentiableWithinAt Real f (Set.Iio a) a
    this : DifferentiableWithinAt Real f' (Set.Ioi (Neg.neg a)) (Neg.neg a)
    ⊢ False
  -/
  exact hmain this
  /-
    🎉 no goals
  -/


/-- A real function whose derivative tends to infinity from the left at a point is not
differentiable on the left at that point -/
theorem not_differentiableWithinAt_of_deriv_tendsto_atTop_Iio (f : ℝ → ℝ) {a : ℝ}
    (hf : Tendsto (deriv f) (𝓝[<] a) atTop) : ¬ DifferentiableWithinAt ℝ f (Iio a) a := by
  /-
    f : Real → Real
    a : Real
    hf : Filter.Tendsto (deriv f) (nhdsWithin a (Set.Iio a)) Filter.atTop
    ⊢ Not (DifferentiableWithinAt Real f (Set.Iio a) a)
  -/
  intro h
  have hf' : Tendsto (deriv (-f)) (𝓝[<] a) atBot := by
    rw [Pi.neg_def, deriv.neg']
    exact tendsto_neg_atTop_atBot.comp hf
  /-
    f : Real → Real
    a : Real
    hf : Filter.Tendsto (deriv f) (nhdsWithin a (Set.Iio a)) Filter.atTop
    h : DifferentiableWithinAt Real f (Set.Iio a) a
    hf' : Filter.Tendsto (deriv (Neg.neg f)) (nhdsWithin a (Set.Iio a)) Filter.atBot
    ⊢ False
  -/
  exact not_differentiableWithinAt_of_deriv_tendsto_atBot_Iio (-f) hf' h.neg
  /-
    🎉 no goals
  -/


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `C < f'`, then
`f` grows faster than `C * x` on `D`, i.e., `C * (y - x) < f y - f x` whenever `x, y ∈ D`,
`x < y`. -/
theorem Convex.mul_sub_lt_image_sub_of_lt_deriv {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : DifferentiableOn ℝ f (interior D)) {C}
    (hf'_gt : ∀ x ∈ interior D, C < deriv f x) :
    ∀ᵉ (x ∈ D) (y ∈ D), x < y → C * (y - x) < f y - f x := by
  /-
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_gt : ∀ (x : Real), Membership.mem (interior D) x → LT.lt C (deriv f x)
    ⊢ ∀ (x : Real), Membership.mem D x → ∀ (y : Real), Membership.mem D y → LT.lt  …
  -/
  intro x hx y hy hxy
  /-
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_gt : ∀ (x : Real), Membership.mem (interior D) x → LT.lt C (deriv f x)
    x : Real
    hx : Membership.mem D x
    y : Real
    hy : Membership.mem D y
    hxy : LT.lt x y
    ⊢ LT.lt (HMul.hMul C (HSub.hSub y x)) (HSub.hSub (f y) (f x))
  -/
  have hxyD : Icc x y ⊆ D := hD.ordConnected.out hx hy
  have hxyD' : Ioo x y ⊆ interior D :=
    subset_sUnion_of_mem ⟨isOpen_Ioo, Ioo_subset_Icc_self.trans hxyD⟩
  obtain ⟨a, a_mem, ha⟩ : ∃ a ∈ Ioo x y, deriv f a = (f y - f x) / (y - x) :=
    exists_deriv_eq_slope f hxy (hf.mono hxyD) (hf'.mono hxyD')
  /-
    case intro.intro
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_gt : ∀ (x : Real), Membership.mem (interior D) x → LT.lt C (deriv f x)
    x : Real
    hx : Membership.mem D x
    y : Real
    hy : Membership.mem D y
    hxy : LT.lt x y
    hxyD : HasSubset.Subset (Set.Icc x y) D
    hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
    a : Real
    a_mem : Membership.mem (Set.Ioo x y) a
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    ⊢ LT.lt (HMul.hMul C (HSub.hSub y x)) (HSub.hSub (f y) (f x))
  -/
  have : C < (f y - f x) / (y - x) := ha ▸ hf'_gt _ (hxyD' a_mem)
  /-
    case intro.intro
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_gt : ∀ (x : Real), Membership.mem (interior D) x → LT.lt C (deriv f x)
    x : Real
    hx : Membership.mem D x
    y : Real
    hy : Membership.mem D y
    hxy : LT.lt x y
    hxyD : HasSubset.Subset (Set.Icc x y) D
    hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
    a : Real
    a_mem : Membership.mem (Set.Ioo x y) a
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    this : LT.lt C (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    ⊢ LT.lt (HMul.hMul C (HSub.hSub y x)) (HSub.hSub (f y) (f x))
  -/
  exact (lt_div_iff₀ (sub_pos.2 hxy)).1 this
  /-
    🎉 no goals
  -/


/-- Let `f : ℝ → ℝ` be a differentiable function. If `C < f'`, then `f` grows faster than
`C * x`, i.e., `C * (y - x) < f y - f x` whenever `x < y`. -/
theorem mul_sub_lt_image_sub_of_lt_deriv {f : ℝ → ℝ} (hf : Differentiable ℝ f) {C}
    (hf'_gt : ∀ x, C < deriv f x) ⦃x y⦄ (hxy : x < y) : C * (y - x) < f y - f x :=
  convex_univ.mul_sub_lt_image_sub_of_lt_deriv hf.continuous.continuousOn hf.differentiableOn
    (fun x _ => hf'_gt x) x trivial y trivial hxy


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `C ≤ f'`, then
`f` grows at least as fast as `C * x` on `D`, i.e., `C * (y - x) ≤ f y - f x` whenever `x, y ∈ D`,
`x ≤ y`. -/
theorem Convex.mul_sub_le_image_sub_of_le_deriv {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : DifferentiableOn ℝ f (interior D)) {C}
    (hf'_ge : ∀ x ∈ interior D, C ≤ deriv f x) :
    ∀ᵉ (x ∈ D) (y ∈ D), x ≤ y → C * (y - x) ≤ f y - f x := by
  /-
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_ge : ∀ (x : Real), Membership.mem (interior D) x → LE.le C (deriv f x)
    ⊢ ∀ (x : Real), Membership.mem D x → ∀ (y : Real), Membership.mem D y → LE.le  …
  -/
  intro x hx y hy hxy
  /-
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_ge : ∀ (x : Real), Membership.mem (interior D) x → LE.le C (deriv f x)
    x : Real
    hx : Membership.mem D x
    y : Real
    hy : Membership.mem D y
    hxy : LE.le x y
    ⊢ LE.le (HMul.hMul C (HSub.hSub y x)) (HSub.hSub (f y) (f x))
  -/
  cases' eq_or_lt_of_le hxy with hxy' hxy'
    /-
      case inl
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : DifferentiableOn Real f (interior D)
      C : Real
      hf'_ge : ∀ (x : Real), Membership.mem (interior D) x → LE.le C (deriv f x)
      x : Real
      hx : Membership.mem D x
      y : Real
      hy : Membership.mem D y
      hxy : LE.le x y
      hxy' : Eq x y
      ⊢ LE.le (HMul.hMul C (HSub.hSub y x)) (HSub.hSub (f y) (f x))
    -/
  · rw [hxy', sub_self, sub_self, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_ge : ∀ (x : Real), Membership.mem (interior D) x → LE.le C (deriv f x)
    x : Real
    hx : Membership.mem D x
    y : Real
    hy : Membership.mem D y
    hxy : LE.le x y
    hxy' : LT.lt x y
    ⊢ LE.le (HMul.hMul C (HSub.hSub y x)) (HSub.hSub (f y) (f x))
  -/
  have hxyD : Icc x y ⊆ D := hD.ordConnected.out hx hy
  have hxyD' : Ioo x y ⊆ interior D :=
    subset_sUnion_of_mem ⟨isOpen_Ioo, Ioo_subset_Icc_self.trans hxyD⟩
  obtain ⟨a, a_mem, ha⟩ : ∃ a ∈ Ioo x y, deriv f a = (f y - f x) / (y - x) :=
    exists_deriv_eq_slope f hxy' (hf.mono hxyD) (hf'.mono hxyD')
  /-
    case inr.intro.intro
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_ge : ∀ (x : Real), Membership.mem (interior D) x → LE.le C (deriv f x)
    x : Real
    hx : Membership.mem D x
    y : Real
    hy : Membership.mem D y
    hxy : LE.le x y
    hxy' : LT.lt x y
    hxyD : HasSubset.Subset (Set.Icc x y) D
    hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
    a : Real
    a_mem : Membership.mem (Set.Ioo x y) a
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    ⊢ LE.le (HMul.hMul C (HSub.hSub y x)) (HSub.hSub (f y) (f x))
  -/
  have : C ≤ (f y - f x) / (y - x) := ha ▸ hf'_ge _ (hxyD' a_mem)
  /-
    case inr.intro.intro
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : DifferentiableOn Real f (interior D)
    C : Real
    hf'_ge : ∀ (x : Real), Membership.mem (interior D) x → LE.le C (deriv f x)
    x : Real
    hx : Membership.mem D x
    y : Real
    hy : Membership.mem D y
    hxy : LE.le x y
    hxy' : LT.lt x y
    hxyD : HasSubset.Subset (Set.Icc x y) D
    hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
    a : Real
    a_mem : Membership.mem (Set.Ioo x y) a
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    this : LE.le C (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    ⊢ LE.le (HMul.hMul C (HSub.hSub y x)) (HSub.hSub (f y) (f x))
  -/
  exact (le_div_iff₀ (sub_pos.2 hxy')).1 this
  /-
    🎉 no goals
  -/


/-- Let `f : ℝ → ℝ` be a differentiable function. If `C ≤ f'`, then `f` grows at least as fast
as `C * x`, i.e., `C * (y - x) ≤ f y - f x` whenever `x ≤ y`. -/
theorem mul_sub_le_image_sub_of_le_deriv {f : ℝ → ℝ} (hf : Differentiable ℝ f) {C}
    (hf'_ge : ∀ x, C ≤ deriv f x) ⦃x y⦄ (hxy : x ≤ y) : C * (y - x) ≤ f y - f x :=
  convex_univ.mul_sub_le_image_sub_of_le_deriv hf.continuous.continuousOn hf.differentiableOn
    (fun x _ => hf'_ge x) x trivial y trivial hxy


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f' < C`, then
`f` grows slower than `C * x` on `D`, i.e., `f y - f x < C * (y - x)` whenever `x, y ∈ D`,
`x < y`. -/
theorem Convex.image_sub_lt_mul_sub_of_deriv_lt {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : DifferentiableOn ℝ f (interior D)) {C}
    (lt_hf' : ∀ x ∈ interior D, deriv f x < C) (x : ℝ) (hx : x ∈ D) (y : ℝ) (hy : y ∈ D)
    (hxy : x < y) : f y - f x < C * (y - x) :=
  have hf'_gt : ∀ x ∈ interior D, -C < deriv (fun y => -f y) x := fun x hx => by
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : DifferentiableOn Real f (interior D)
      C : Real
      lt_hf' : ∀ (x : Real), Membership.mem (interior D) x → LT.lt (deriv f x) C
      x✝ : Real
      hx✝ : Membership.mem D x✝
      y : Real
      hy : Membership.mem D y
      hxy : LT.lt x✝ y
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LT.lt (Neg.neg C) (deriv (fun y => Neg.neg (f y)) x)
    -/
    rw [deriv.neg, neg_lt_neg_iff]
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : DifferentiableOn Real f (interior D)
      C : Real
      lt_hf' : ∀ (x : Real), Membership.mem (interior D) x → LT.lt (deriv f x) C
      x✝ : Real
      hx✝ : Membership.mem D x✝
      y : Real
      hy : Membership.mem D y
      hxy : LT.lt x✝ y
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LT.lt (deriv f x) C
    -/
    exact lt_hf' x hx
    /-
      🎉 no goals
    -/
     /-
       D : Set Real
       hD : Convex Real D
       f : Real → Real
       hf : ContinuousOn f D
       hf' : DifferentiableOn Real f (interior D)
       C : Real
       lt_hf' : ∀ (x : Real), Membership.mem (interior D) x → LT.lt (deriv f x) C
       x : Real
       hx : Membership.mem D x
       y : Real
       hy : Membership.mem D y
       hxy : LT.lt x y
       hf'_gt : ∀ (x : Real), Membership.mem (interior D) x → LT.lt (Neg.neg C) (deri …
       ⊢ LT.lt (HSub.hSub (f y) (f x)) (HMul.hMul C (HSub.hSub y x))
     -/
  by linarith [hD.mul_sub_lt_image_sub_of_lt_deriv hf.neg hf'.neg hf'_gt x hx y hy hxy]
     /-
       🎉 no goals
     -/


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f' < C`, then `f` grows slower than
`C * x` on `D`, i.e., `f y - f x < C * (y - x)` whenever `x < y`. -/
theorem image_sub_lt_mul_sub_of_deriv_lt {f : ℝ → ℝ} (hf : Differentiable ℝ f) {C}
    (lt_hf' : ∀ x, deriv f x < C) ⦃x y⦄ (hxy : x < y) : f y - f x < C * (y - x) :=
  convex_univ.image_sub_lt_mul_sub_of_deriv_lt hf.continuous.continuousOn hf.differentiableOn
    (fun x _ => lt_hf' x) x trivial y trivial hxy


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f' ≤ C`, then
`f` grows at most as fast as `C * x` on `D`, i.e., `f y - f x ≤ C * (y - x)` whenever `x, y ∈ D`,
`x ≤ y`. -/
theorem Convex.image_sub_le_mul_sub_of_deriv_le {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : DifferentiableOn ℝ f (interior D)) {C}
    (le_hf' : ∀ x ∈ interior D, deriv f x ≤ C) (x : ℝ) (hx : x ∈ D) (y : ℝ) (hy : y ∈ D)
    (hxy : x ≤ y) : f y - f x ≤ C * (y - x) :=
  have hf'_ge : ∀ x ∈ interior D, -C ≤ deriv (fun y => -f y) x := fun x hx => by
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : DifferentiableOn Real f (interior D)
      C : Real
      le_hf' : ∀ (x : Real), Membership.mem (interior D) x → LE.le (deriv f x) C
      x✝ : Real
      hx✝ : Membership.mem D x✝
      y : Real
      hy : Membership.mem D y
      hxy : LE.le x✝ y
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LE.le (Neg.neg C) (deriv (fun y => Neg.neg (f y)) x)
    -/
    rw [deriv.neg, neg_le_neg_iff]
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : DifferentiableOn Real f (interior D)
      C : Real
      le_hf' : ∀ (x : Real), Membership.mem (interior D) x → LE.le (deriv f x) C
      x✝ : Real
      hx✝ : Membership.mem D x✝
      y : Real
      hy : Membership.mem D y
      hxy : LE.le x✝ y
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LE.le (deriv f x) C
    -/
    exact le_hf' x hx
    /-
      🎉 no goals
    -/
     /-
       D : Set Real
       hD : Convex Real D
       f : Real → Real
       hf : ContinuousOn f D
       hf' : DifferentiableOn Real f (interior D)
       C : Real
       le_hf' : ∀ (x : Real), Membership.mem (interior D) x → LE.le (deriv f x) C
       x : Real
       hx : Membership.mem D x
       y : Real
       hy : Membership.mem D y
       hxy : LE.le x y
       hf'_ge : ∀ (x : Real), Membership.mem (interior D) x → LE.le (Neg.neg C) (deri …
       ⊢ LE.le (HSub.hSub (f y) (f x)) (HMul.hMul C (HSub.hSub y x))
     -/
  by linarith [hD.mul_sub_le_image_sub_of_le_deriv hf.neg hf'.neg hf'_ge x hx y hy hxy]
     /-
       🎉 no goals
     -/


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f' ≤ C`, then `f` grows at most as fast
as `C * x`, i.e., `f y - f x ≤ C * (y - x)` whenever `x ≤ y`. -/
theorem image_sub_le_mul_sub_of_deriv_le {f : ℝ → ℝ} (hf : Differentiable ℝ f) {C}
    (le_hf' : ∀ x, deriv f x ≤ C) ⦃x y⦄ (hxy : x ≤ y) : f y - f x ≤ C * (y - x) :=
  convex_univ.image_sub_le_mul_sub_of_deriv_le hf.continuous.continuousOn hf.differentiableOn
    (fun x _ => le_hf' x) x trivial y trivial hxy


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f'` is positive, then
`f` is a strictly monotone function on `D`.
Note that we don't require differentiability explicitly as it already implied by the derivative
being strictly positive. -/
theorem strictMonoOn_of_deriv_pos {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, 0 < deriv f x) : StrictMonoOn f D := by
  /-
    D : Set Real
    hD : Convex Real D
    f : Real → Real
    hf : ContinuousOn f D
    hf' : ∀ (x : Real), Membership.mem (interior D) x → LT.lt 0 (deriv f x)
    ⊢ StrictMonoOn f D
  -/
  intro x hx y hy
  have : DifferentiableOn ℝ f (interior D) := fun z hz =>
    (differentiableAt_of_deriv_ne_zero (hf' z hz).ne').differentiableWithinAt
  simpa only [zero_mul, sub_pos] using
    hD.mul_sub_lt_image_sub_of_lt_deriv hf this hf' x hx y hy


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f'` is positive, then
`f` is a strictly monotone function.
Note that we don't require differentiability explicitly as it already implied by the derivative
being strictly positive. -/
theorem strictMono_of_deriv_pos {f : ℝ → ℝ} (hf' : ∀ x, 0 < deriv f x) : StrictMono f :=
  strictMonoOn_univ.1 <| strictMonoOn_of_deriv_pos convex_univ (fun z _ =>
    (differentiableAt_of_deriv_ne_zero (hf' z).ne').differentiableWithinAt.continuousWithinAt)
    fun x _ => hf' x


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f'` is strictly positive,
then `f` is a strictly monotone function on `D`. -/
lemma strictMonoOn_of_hasDerivWithinAt_pos {D : Set ℝ} (hD : Convex ℝ D) {f f' : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, HasDerivWithinAt f (f' x) (interior D) x)
    (hf'₀ : ∀ x ∈ interior D, 0 < f' x) : StrictMonoOn f D :=
  strictMonoOn_of_deriv_pos hD hf fun x hx ↦ by
    /-
      D : Set Real
      hD : Convex Real D
      f f' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'₀ : ∀ (x : Real), Membership.mem (interior D) x → LT.lt 0 (f' x)
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LT.lt 0 (deriv f x)
    -/
    rw [deriv_eqOn isOpen_interior hf' hx]; exact hf'₀ _ hx
                                            /-
                                              🎉 no goals
                                            -/


@[deprecated (since := "2024-03-02")]
alias StrictMonoOn_of_hasDerivWithinAt_pos := strictMonoOn_of_hasDerivWithinAt_pos


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f'` is strictly positive, then
`f` is a strictly monotone function. -/
lemma strictMono_of_hasDerivAt_pos {f f' : ℝ → ℝ} (hf : ∀ x, HasDerivAt f (f' x) x)
    (hf' : ∀ x, 0 < f' x) : StrictMono f :=
                                     /-
                                       f f' : Real → Real
                                       hf : ∀ (x : Real), HasDerivAt f (f' x) x
                                       hf' : ∀ (x : Real), LT.lt 0 (f' x)
                                       x : Real
                                       ⊢ LT.lt 0 (deriv f x)
                                     -/
  strictMono_of_deriv_pos fun x ↦ by rw [(hf _).deriv]; exact hf' _
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f'` is nonnegative, then
`f` is a monotone function on `D`. -/
theorem monotoneOn_of_deriv_nonneg {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : DifferentiableOn ℝ f (interior D))
    (hf'_nonneg : ∀ x ∈ interior D, 0 ≤ deriv f x) : MonotoneOn f D := fun x hx y hy hxy => by
  simpa only [zero_mul, sub_nonneg] using
    hD.mul_sub_le_image_sub_of_le_deriv hf hf' hf'_nonneg x hx y hy hxy


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f'` is nonnegative, then
`f` is a monotone function. -/
theorem monotone_of_deriv_nonneg {f : ℝ → ℝ} (hf : Differentiable ℝ f) (hf' : ∀ x, 0 ≤ deriv f x) :
    Monotone f :=
  monotoneOn_univ.1 <|
    monotoneOn_of_deriv_nonneg convex_univ hf.continuous.continuousOn hf.differentiableOn fun x _ =>
      hf' x


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f'` is nonnegative, then
`f` is a monotone function on `D`. -/
lemma monotoneOn_of_hasDerivWithinAt_nonneg {D : Set ℝ} (hD : Convex ℝ D) {f f' : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, HasDerivWithinAt f (f' x) (interior D) x)
    (hf'₀ : ∀ x ∈ interior D, 0 ≤ f' x) : MonotoneOn f D :=
  monotoneOn_of_deriv_nonneg hD hf (fun _ hx ↦ (hf' _ hx).differentiableWithinAt) fun x hx ↦ by
    /-
      D : Set Real
      hD : Convex Real D
      f f' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f' x)
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LE.le 0 (deriv f x)
    -/
    rw [deriv_eqOn isOpen_interior hf' hx]; exact hf'₀ _ hx
                                            /-
                                              🎉 no goals
                                            -/


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f'` is nonnegative, then
`f` is a monotone function. -/
lemma monotone_of_hasDerivAt_nonneg {f f' : ℝ → ℝ} (hf : ∀ x, HasDerivAt f (f' x) x)
    (hf' : 0 ≤ f') : Monotone f :=
  monotone_of_deriv_nonneg (fun _ ↦ (hf _).differentiableAt) fun x ↦ by
    /-
      f f' : Real → Real
      hf : ∀ (x : Real), HasDerivAt f (f' x) x
      hf' : LE.le 0 f'
      x : Real
      ⊢ LE.le 0 (deriv f x)
    -/
    rw [(hf _).deriv]; exact hf' _
                       /-
                         🎉 no goals
                       -/


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f'` is negative, then
`f` is a strictly antitone function on `D`. -/
theorem strictAntiOn_of_deriv_neg {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, deriv f x < 0) : StrictAntiOn f D :=
  fun x hx y => by
  simpa only [zero_mul, sub_lt_zero] using
    hD.image_sub_lt_mul_sub_of_deriv_lt hf
      (fun z hz => (differentiableAt_of_deriv_ne_zero (hf' z hz).ne).differentiableWithinAt) hf' x
      hx y


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f'` is negative, then
`f` is a strictly antitone function.
Note that we don't require differentiability explicitly as it already implied by the derivative
being strictly negative. -/
theorem strictAnti_of_deriv_neg {f : ℝ → ℝ} (hf' : ∀ x, deriv f x < 0) : StrictAnti f :=
  strictAntiOn_univ.1 <| strictAntiOn_of_deriv_neg convex_univ
      (fun z _ =>
        (differentiableAt_of_deriv_ne_zero (hf' z).ne).differentiableWithinAt.continuousWithinAt)
      fun x _ => hf' x


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f'` is strictly positive,
then `f` is a strictly monotone function on `D`. -/
lemma strictAntiOn_of_hasDerivWithinAt_neg {D : Set ℝ} (hD : Convex ℝ D) {f f' : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, HasDerivWithinAt f (f' x) (interior D) x)
    (hf'₀ : ∀ x ∈ interior D, f' x < 0) : StrictAntiOn f D :=
  strictAntiOn_of_deriv_neg hD hf fun x hx ↦ by
    /-
      D : Set Real
      hD : Convex Real D
      f f' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'₀ : ∀ (x : Real), Membership.mem (interior D) x → LT.lt (f' x) 0
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LT.lt (deriv f x) 0
    -/
    rw [deriv_eqOn isOpen_interior hf' hx]; exact hf'₀ _ hx
                                            /-
                                              🎉 no goals
                                            -/


@[deprecated (since := "2024-03-02")]
alias StrictAntiOn_of_hasDerivWithinAt_pos := strictAntiOn_of_hasDerivWithinAt_neg


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f'` is strictly positive, then
`f` is a strictly monotone function. -/
lemma strictAnti_of_hasDerivAt_neg {f f' : ℝ → ℝ} (hf : ∀ x, HasDerivAt f (f' x) x)
    (hf' : ∀ x, f' x < 0) : StrictAnti f :=
                                     /-
                                       f f' : Real → Real
                                       hf : ∀ (x : Real), HasDerivAt f (f' x) x
                                       hf' : ∀ (x : Real), LT.lt (f' x) 0
                                       x : Real
                                       ⊢ LT.lt (deriv f x) 0
                                     -/
  strictAnti_of_deriv_neg fun x ↦ by rw [(hf _).deriv]; exact hf' _
                                                        /-
                                                          🎉 no goals
                                                        -/


@[deprecated (since := "2024-03-02")]
alias strictAnti_of_hasDerivAt_pos := strictAnti_of_hasDerivAt_neg


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f'` is nonpositive, then
`f` is an antitone function on `D`. -/
theorem antitoneOn_of_deriv_nonpos {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : DifferentiableOn ℝ f (interior D))
    (hf'_nonpos : ∀ x ∈ interior D, deriv f x ≤ 0) : AntitoneOn f D := fun x hx y hy hxy => by
  simpa only [zero_mul, sub_nonpos] using
    hD.image_sub_le_mul_sub_of_deriv_le hf hf' hf'_nonpos x hx y hy hxy


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f'` is nonpositive, then
`f` is an antitone function. -/
theorem antitone_of_deriv_nonpos {f : ℝ → ℝ} (hf : Differentiable ℝ f) (hf' : ∀ x, deriv f x ≤ 0) :
    Antitone f :=
  antitoneOn_univ.1 <|
    antitoneOn_of_deriv_nonpos convex_univ hf.continuous.continuousOn hf.differentiableOn fun x _ =>
      hf' x


/-- Let `f` be a function continuous on a convex (or, equivalently, connected) subset `D`
of the real line. If `f` is differentiable on the interior of `D` and `f'` is nonpositive, then
`f` is an antitone function on `D`. -/
lemma antitoneOn_of_hasDerivWithinAt_nonpos {D : Set ℝ} (hD : Convex ℝ D) {f f' : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, HasDerivWithinAt f (f' x) (interior D) x)
    (hf'₀ : ∀ x ∈ interior D, f' x ≤ 0) : AntitoneOn f D :=
  antitoneOn_of_deriv_nonpos hD hf (fun _ hx ↦ (hf' _ hx).differentiableWithinAt) fun x hx ↦ by
    /-
      D : Set Real
      hD : Convex Real D
      f f' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f' x) 0
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LE.le (deriv f x) 0
    -/
    rw [deriv_eqOn isOpen_interior hf' hx]; exact hf'₀ _ hx
                                            /-
                                              🎉 no goals
                                            -/


/-- Let `f : ℝ → ℝ` be a differentiable function. If `f'` is nonpositive, then `f` is an antitone
function. -/
lemma antitone_of_hasDerivAt_nonpos {f f' : ℝ → ℝ} (hf : ∀ x, HasDerivAt f (f' x) x)
    (hf' : f' ≤ 0) : Antitone f :=
  antitone_of_deriv_nonpos (fun _ ↦ (hf _).differentiableAt) fun x ↦ by
    /-
      f f' : Real → Real
      hf : ∀ (x : Real), HasDerivAt f (f' x) x
      hf' : LE.le f' 0
      x : Real
      ⊢ LE.le (deriv f x) 0
    -/
    rw [(hf _).deriv]; exact hf' _
                       /-
                         🎉 no goals
                       -/


/-- Lagrange's **Mean Value Theorem**, applied to convex domains. -/
theorem domain_mvt {f : E → ℝ} {s : Set E} {x y : E} {f' : E → E →L[ℝ] ℝ}
    (hf : ∀ x ∈ s, HasFDerivWithinAt f (f' x) s x) (hs : Convex ℝ s) (xs : x ∈ s) (ys : y ∈ s) :
    ∃ z ∈ segment ℝ x y, f y - f x = f' z (y - x) := by
  -- Use `g = AffineMap.lineMap x y` to parametrize the segment
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id Real) E Real
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    ⊢ Exists fun z => And (Membership.mem (segment Real x y) z) (Eq (HSub.hSub (f  …
  -/
  set g : ℝ → E := fun t => AffineMap.lineMap x y t
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id Real) E Real
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    g : Real → E := fun t => (AffineMap.lineMap x y) t
    ⊢ Exists fun z => And (Membership.mem (segment Real x y) z) (Eq (HSub.hSub (f  …
  -/
  set I := Icc (0 : ℝ) 1
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id Real) E Real
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    g : Real → E := fun t => (AffineMap.lineMap x y) t
    I : Set Real := Set.Icc 0 1
    ⊢ Exists fun z => And (Membership.mem (segment Real x y) z) (Eq (HSub.hSub (f  …
  -/
  have hsub : Ioo (0 : ℝ) 1 ⊆ I := Ioo_subset_Icc_self
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id Real) E Real
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    g : Real → E := fun t => (AffineMap.lineMap x y) t
    I : Set Real := Set.Icc 0 1
    hsub : HasSubset.Subset (Set.Ioo 0 1) I
    ⊢ Exists fun z => And (Membership.mem (segment Real x y) z) (Eq (HSub.hSub (f  …
  -/
  have hmaps : MapsTo g I s := hs.mapsTo_lineMap xs ys
  -- The one-variable function `f ∘ g` has derivative `f' (g t) (y - x)` at each `t ∈ I`
  have hfg : ∀ t ∈ I, HasDerivWithinAt (f ∘ g) (f' (g t) (y - x)) I t := fun t ht =>
    (hf _ (hmaps ht)).comp_hasDerivWithinAt t AffineMap.hasDerivWithinAt_lineMap hmaps
  -- apply 1-variable mean value theorem to pullback
  have hMVT : ∃ t ∈ Ioo (0 : ℝ) 1, f' (g t) (y - x) = (f (g 1) - f (g 0)) / (1 - 0) := by
    refine exists_hasDerivAt_eq_slope (f ∘ g) _ (by norm_num) ?_ ?_
    · exact fun t Ht => (hfg t Ht).continuousWithinAt
    · exact fun t Ht => (hfg t <| hsub Ht).hasDerivAt (Icc_mem_nhds Ht.1 Ht.2)
  -- reinterpret on domain
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id Real) E Real
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    g : Real → E := fun t => (AffineMap.lineMap x y) t
    I : Set Real := Set.Icc 0 1
    hsub : HasSubset.Subset (Set.Ioo 0 1) I
    hmaps : Set.MapsTo g I s
    hfg : ∀ (t : Real), Membership.mem I t → HasDerivWithinAt (Function.comp f g)  …
    hMVT : Exists fun t => And (Membership.mem (Set.Ioo 0 1) t) (Eq ((f' (g t)) (H …
    ⊢ Exists fun z => And (Membership.mem (segment Real x y) z) (Eq (HSub.hSub (f  …
  -/
  rcases hMVT with ⟨t, Ht, hMVT'⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id Real) E Real
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    g : Real → E := fun t => (AffineMap.lineMap x y) t
    I : Set Real := Set.Icc 0 1
    hsub : HasSubset.Subset (Set.Ioo 0 1) I
    hmaps : Set.MapsTo g I s
    hfg : ∀ (t : Real), Membership.mem I t → HasDerivWithinAt (Function.comp f g)  …
    t : Real
    Ht : Membership.mem (Set.Ioo 0 1) t
    hMVT' : Eq ((f' (g t)) (HSub.hSub y x)) (HDiv.hDiv (HSub.hSub (f (g 1)) (f (g  …
    ⊢ Exists fun z => And (Membership.mem (segment Real x y) z) (Eq (HSub.hSub (f  …
  -/
  rw [segment_eq_image_lineMap, exists_mem_image]
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id Real) E Real
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    g : Real → E := fun t => (AffineMap.lineMap x y) t
    I : Set Real := Set.Icc 0 1
    hsub : HasSubset.Subset (Set.Ioo 0 1) I
    hmaps : Set.MapsTo g I s
    hfg : ∀ (t : Real), Membership.mem I t → HasDerivWithinAt (Function.comp f g)  …
    t : Real
    Ht : Membership.mem (Set.Ioo 0 1) t
    hMVT' : Eq ((f' (g t)) (HSub.hSub y x)) (HDiv.hDiv (HSub.hSub (f (g 1)) (f (g  …
    ⊢ Exists fun x_1 => And (Membership.mem (Set.Icc 0 1) x_1) (Eq (HSub.hSub (f y …
  -/
  refine ⟨t, hsub Ht, ?_⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    s : Set E
    x y : E
    f' : E → ContinuousLinearMap (RingHom.id Real) E Real
    hf : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    hs : Convex Real s
    xs : Membership.mem s x
    ys : Membership.mem s y
    g : Real → E := fun t => (AffineMap.lineMap x y) t
    I : Set Real := Set.Icc 0 1
    hsub : HasSubset.Subset (Set.Ioo 0 1) I
    hmaps : Set.MapsTo g I s
    hfg : ∀ (t : Real), Membership.mem I t → HasDerivWithinAt (Function.comp f g)  …
    t : Real
    Ht : Membership.mem (Set.Ioo 0 1) t
    hMVT' : Eq ((f' (g t)) (HSub.hSub y x)) (HDiv.hDiv (HSub.hSub (f (g 1)) (f (g  …
    ⊢ Eq (HSub.hSub (f y) (f x)) ((f' ((AffineMap.lineMap x y) t)) (HSub.hSub y x))
  -/
  simpa [g] using hMVT'.symm
  /-
    🎉 no goals
  -/


/-- Over the reals or the complexes, a continuously differentiable function is strictly
differentiable. -/
theorem hasStrictFDerivAt_of_hasFDerivAt_of_continuousAt
    (hder : ∀ᶠ y in 𝓝 x, HasFDerivAt f (f' y) y) (hcont : ContinuousAt f' x) :
    HasStrictFDerivAt f (f' x) x := by
  -- turn little-o definition of strict_fderiv into an epsilon-delta statement
  /-
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    ⊢ HasStrictFDerivAt f (f' x) x
  -/
  rw [hasStrictFDerivAt_iff_isLittleO, isLittleO_iff]
  /-
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm (HS …
  -/
  refine fun c hc => Metric.eventually_nhds_iff_ball.mpr ?_
  -- the correct ε is the modulus of continuity of f'
  /-
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    c : Real
    hc : LT.lt 0 c
    ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (y : Prod G G), Membership.mem (Metric.ba …
  -/
  rcases Metric.mem_nhds_iff.mp (inter_mem hder (hcont <| ball_mem_nhds _ hc)) with ⟨ε, ε0, hε⟩
  /-
    case intro.intro
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    c : Real
    hc : LT.lt 0 c
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (Inter.inter (setOf fun x => (fun y => …
    ⊢ Exists fun ε => And (GT.gt ε 0) (∀ (y : Prod G G), Membership.mem (Metric.ba …
  -/
  refine ⟨ε, ε0, ?_⟩
  -- simplify formulas involving the product E × E
  /-
    case intro.intro
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    c : Real
    hc : LT.lt 0 c
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (Inter.inter (setOf fun x => (fun y => …
    ⊢ ∀ (y : Prod G G), Membership.mem (Metric.ball { fst := x, snd := x } ε) y →  …
  -/
  rintro ⟨a, b⟩ h
  /-
    case intro.intro.mk
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    c : Real
    hc : LT.lt 0 c
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (Inter.inter (setOf fun x => (fun y => …
    a b : G
    h : Membership.mem (Metric.ball { fst := x, snd := x } ε) { fst := a, snd := b }
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f { fst := a, snd := b }.1) (f { fst …
  -/
  rw [← ball_prod_same, prod_mk_mem_set_prod_eq] at h
  -- exploit the choice of ε as the modulus of continuity of f'
  have hf' : ∀ x' ∈ ball x ε, ‖f' x' - f' x‖ ≤ c := fun x' H' => by
    rw [← dist_eq_norm]
    exact le_of_lt (hε H').2
  -- apply mean value theorem
  /-
    case intro.intro.mk
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    c : Real
    hc : LT.lt 0 c
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (Inter.inter (setOf fun x => (fun y => …
    a b : G
    h : And (Membership.mem (Metric.ball x ε) a) (Membership.mem (Metric.ball x ε) …
    hf' : ∀ (x' : G), Membership.mem (Metric.ball x ε) x' → LE.le (Norm.norm (HSub …
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f { fst := a, snd := b }.1) (f { fst …
  -/
  letI : NormedSpace ℝ G := RestrictScalars.normedSpace ℝ 𝕜 G
  /-
    case intro.intro.mk
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    c : Real
    hc : LT.lt 0 c
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (Inter.inter (setOf fun x => (fun y => …
    a b : G
    h : And (Membership.mem (Metric.ball x ε) a) (Membership.mem (Metric.ball x ε) …
    hf' : ∀ (x' : G), Membership.mem (Metric.ball x ε) x' → LE.le (Norm.norm (HSub …
    this : NormedSpace Real G := RestrictScalars.normedSpace Real 𝕜 G
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (f { fst := a, snd := b }.1) (f { fst …
  -/
  refine (convex_ball _ _).norm_image_sub_le_of_norm_hasFDerivWithin_le' ?_ hf' h.2 h.1
  /-
    case intro.intro.mk
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    G : Type u_4
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    H : Type u_5
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    f : G → H
    f' : G → ContinuousLinearMap (RingHom.id 𝕜) G H
    x : G
    hder : Filter.Eventually (fun y => HasFDerivAt f (f' y) y) (nhds x)
    hcont : ContinuousAt f' x
    c : Real
    hc : LT.lt 0 c
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (Inter.inter (setOf fun x => (fun y => …
    a b : G
    h : And (Membership.mem (Metric.ball x ε) a) (Membership.mem (Metric.ball x ε) …
    hf' : ∀ (x' : G), Membership.mem (Metric.ball x ε) x' → LE.le (Norm.norm (HSub …
    this : NormedSpace Real G := RestrictScalars.normedSpace Real 𝕜 G
    ⊢ ∀ (x_1 : G), Membership.mem (Metric.ball x ε) x_1 → HasFDerivWithinAt f (f'  …
  -/
  exact fun y hy => (hε hy).1.hasFDerivWithinAt
  /-
    🎉 no goals
  -/


/-- Over the reals or the complexes, a continuously differentiable function is strictly
differentiable. -/
theorem hasStrictDerivAt_of_hasDerivAt_of_continuousAt {f f' : 𝕜 → G} {x : 𝕜}
    (hder : ∀ᶠ y in 𝓝 x, HasDerivAt f (f' y) y) (hcont : ContinuousAt f' x) :
    HasStrictDerivAt f (f' x) x :=
  hasStrictFDerivAt_of_hasFDerivAt_of_continuousAt (hder.mono fun _ hy => hy.hasFDerivAt) <|
    (smulRightL 𝕜 𝕜 G 1).continuous.continuousAt.comp hcont


