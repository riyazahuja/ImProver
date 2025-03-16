/-- "Positive" tangent cone to `s` at `x`; the only difference from `tangentConeAt`
is that we require `c n → ∞` instead of `‖c n‖ → ∞`. One can think about `posTangentConeAt`
as `tangentConeAt NNReal` but we have no theory of normed semifields yet. -/
def posTangentConeAt (s : Set E) (x : E) : Set E :=
  { y : E | ∃ (c : ℕ → ℝ) (d : ℕ → E), (∀ᶠ n in atTop, x + d n ∈ s) ∧
    Tendsto c atTop atTop ∧ Tendsto (fun n => c n • d n) atTop (𝓝 y) }


theorem posTangentConeAt_mono : Monotone fun s => posTangentConeAt s a := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : E
    ⊢ Monotone fun s => posTangentConeAt s a
  -/
  rintro s t hst y ⟨c, d, hd, hc, hcd⟩
  /-
    case intro.intro.intro.intro
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : E
    s t : Set E
    hst : LE.le s t
    y : E
    c : Nat → Real
    d : Nat → E
    hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd a (d n))) Filter. …
    hc : Filter.Tendsto c Filter.atTop Filter.atTop
    hcd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    ⊢ Membership.mem ((fun s => posTangentConeAt s a) t) y
  -/
  exact ⟨c, d, mem_of_superset hd fun h hn => hst hn, hc, hcd⟩
  /-
    🎉 no goals
  -/


theorem mem_posTangentConeAt_of_frequently_mem (h : ∃ᶠ t : ℝ in 𝓝[>] 0, x + t • y ∈ s) :
    y ∈ posTangentConeAt s x := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : Filter.Frequently (fun t => Membership.mem s (HAdd.hAdd x (HSMul.hSMul t y …
    ⊢ Membership.mem (posTangentConeAt s x) y
  -/
  obtain ⟨a, ha, has⟩ := Filter.exists_seq_forall_of_frequently h
  /-
    case intro.intro
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : Filter.Frequently (fun t => Membership.mem s (HAdd.hAdd x (HSMul.hSMul t y …
    a : Nat → Real
    ha : Filter.Tendsto a Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    has : ∀ (n : Nat), Membership.mem s (HAdd.hAdd x (HSMul.hSMul (a n) y))
    ⊢ Membership.mem (posTangentConeAt s x) y
  -/
  refine ⟨a⁻¹, (a · • y), Eventually.of_forall has, tendsto_inv_nhdsGT_zero.comp ha, ?_⟩
  /-
    case intro.intro
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : Filter.Frequently (fun t => Membership.mem s (HAdd.hAdd x (HSMul.hSMul t y …
    a : Nat → Real
    ha : Filter.Tendsto a Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    has : ∀ (n : Nat), Membership.mem s (HAdd.hAdd x (HSMul.hSMul (a n) y))
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv a n) ((fun x => HSMul.hSMul (a …
  -/
  refine tendsto_const_nhds.congr' ?_
  /-
    case intro.intro
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : Filter.Frequently (fun t => Membership.mem s (HAdd.hAdd x (HSMul.hSMul t y …
    a : Nat → Real
    ha : Filter.Tendsto a Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    has : ∀ (n : Nat), Membership.mem s (HAdd.hAdd x (HSMul.hSMul (a n) y))
    ⊢ Filter.atTop.EventuallyEq (fun x => y) fun n => HSMul.hSMul (Inv.inv a n) (( …
  -/
  filter_upwards [(tendsto_nhdsWithin_iff.1 ha).2] with n (hn : 0 < a n)
  /-
    case h
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : Filter.Frequently (fun t => Membership.mem s (HAdd.hAdd x (HSMul.hSMul t y …
    a : Nat → Real
    ha : Filter.Tendsto a Filter.atTop (nhdsWithin 0 (Set.Ioi 0))
    has : ∀ (n : Nat), Membership.mem s (HAdd.hAdd x (HSMul.hSMul (a n) y))
    n : Nat
    hn : LT.lt 0 (a n)
    ⊢ Eq y (HSMul.hSMul (Inv.inv a n) (HSMul.hSMul (a n) y))
  -/
  simp [ne_of_gt hn]
  /-
    🎉 no goals
  -/


/-- If `[x -[ℝ] x + y] ⊆ s`, then `y` belongs to the positive tangnet cone of `s`.

Before 2024-07-13, this lemma used to be called `mem_posTangentConeAt_of_segment_subset`.
See also `sub_mem_posTangentConeAt_of_segment_subset`
for the lemma that used to be called `mem_posTangentConeAt_of_segment_subset`. -/
theorem mem_posTangentConeAt_of_segment_subset (h : [x -[ℝ] x + y] ⊆ s) :
    y ∈ posTangentConeAt s x := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : HasSubset.Subset (segment Real x (HAdd.hAdd x y)) s
    ⊢ Membership.mem (posTangentConeAt s x) y
  -/
  refine mem_posTangentConeAt_of_frequently_mem (Eventually.frequently ?_)
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : HasSubset.Subset (segment Real x (HAdd.hAdd x y)) s
    ⊢ Filter.Eventually (fun x_1 => Membership.mem s (HAdd.hAdd x (HSMul.hSMul x_1 …
  -/
  rw [eventually_nhdsWithin_iff]
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : HasSubset.Subset (segment Real x (HAdd.hAdd x y)) s
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Ioi 0) x_1 → Membership.me …
  -/
  filter_upwards [ge_mem_nhds one_pos] with t ht₁ ht₀
  /-
    case h
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : HasSubset.Subset (segment Real x (HAdd.hAdd x y)) s
    t : Real
    ht₁ : LE.le t 1
    ht₀ : Membership.mem (Set.Ioi 0) t
    ⊢ Membership.mem s (HAdd.hAdd x (HSMul.hSMul t y))
  -/
  apply h
  /-
    case h.a
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : HasSubset.Subset (segment Real x (HAdd.hAdd x y)) s
    t : Real
    ht₁ : LE.le t 1
    ht₀ : Membership.mem (Set.Ioi 0) t
    ⊢ Membership.mem (segment Real x (HAdd.hAdd x y)) (HAdd.hAdd x (HSMul.hSMul t  …
  -/
  rw [segment_eq_image', add_sub_cancel_left]
  /-
    case h.a
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x y : E
    h : HasSubset.Subset (segment Real x (HAdd.hAdd x y)) s
    t : Real
    ht₁ : LE.le t 1
    ht₀ : Membership.mem (Set.Ioi 0) t
    ⊢ Membership.mem (Set.image (fun θ => HAdd.hAdd x (HSMul.hSMul θ y)) (Set.Icc  …
  -/
  exact mem_image_of_mem _ ⟨le_of_lt ht₀, ht₁⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-13")] -- cleanup docstrings when we drop this alias
alias mem_posTangentConeAt_of_segment_subset' := mem_posTangentConeAt_of_segment_subset


theorem sub_mem_posTangentConeAt_of_segment_subset (h : segment ℝ x y ⊆ s) :
    y - x ∈ posTangentConeAt s x :=
                                               /-
                                                 E : Type u
                                                 inst✝¹ : NormedAddCommGroup E
                                                 inst✝ : NormedSpace Real E
                                                 s : Set E
                                                 x y : E
                                                 h : HasSubset.Subset (segment Real x y) s
                                                 ⊢ HasSubset.Subset (segment Real x (HAdd.hAdd x (HSub.hSub y x))) s
                                               -/
  mem_posTangentConeAt_of_segment_subset <| by rwa [add_sub_cancel]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem posTangentConeAt_univ : posTangentConeAt univ a = univ :=
  eq_univ_of_forall fun _ => mem_posTangentConeAt_of_segment_subset (subset_univ _)


/-- If `f` has a local max on `s` at `a`, `f'` is the derivative of `f` at `a` within `s`, and
`y` belongs to the positive tangent cone of `s` at `a`, then `f' y ≤ 0`. -/
theorem IsLocalMaxOn.hasFDerivWithinAt_nonpos (h : IsLocalMaxOn f s a)
    (hf : HasFDerivWithinAt f f' s a) (hy : y ∈ posTangentConeAt s a) : f' y ≤ 0 := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    a y : E
    h : IsLocalMaxOn f s a
    hf : HasFDerivWithinAt f f' s a
    hy : Membership.mem (posTangentConeAt s a) y
    ⊢ LE.le (f' y) 0
  -/
  rcases hy with ⟨c, d, hd, hc, hcd⟩
  /-
    case intro.intro.intro.intro
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    a y : E
    h : IsLocalMaxOn f s a
    hf : HasFDerivWithinAt f f' s a
    c : Nat → Real
    d : Nat → E
    hd : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd a (d n))) Filter. …
    hc : Filter.Tendsto c Filter.atTop Filter.atTop
    hcd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    ⊢ LE.le (f' y) 0
  -/
  have hc' : Tendsto (‖c ·‖) atTop atTop := tendsto_abs_atTop_atTop.comp hc
  suffices ∀ᶠ n in atTop, c n • (f (a + d n) - f a) ≤ 0 from
    le_of_tendsto (hf.lim atTop hd hc' hcd) this
  replace hd : Tendsto (fun n => a + d n) atTop (𝓝[s] (a + 0)) :=
    tendsto_nhdsWithin_iff.2 ⟨tendsto_const_nhds.add (tangentConeAt.lim_zero _ hc' hcd), hd⟩
  /-
    case intro.intro.intro.intro
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    a y : E
    h : IsLocalMaxOn f s a
    hf : HasFDerivWithinAt f f' s a
    c : Nat → Real
    d : Nat → E
    hc : Filter.Tendsto c Filter.atTop Filter.atTop
    hcd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    hc' : Filter.Tendsto (fun x => Norm.norm (c x)) Filter.atTop Filter.atTop
    hd : Filter.Tendsto (fun n => HAdd.hAdd a (d n)) Filter.atTop (nhdsWithin (HAd …
    ⊢ Filter.Eventually (fun n => LE.le (HSMul.hSMul (c n) (HSub.hSub (f (HAdd.hAd …
  -/
  rw [add_zero] at hd
  /-
    case intro.intro.intro.intro
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    a y : E
    h : IsLocalMaxOn f s a
    hf : HasFDerivWithinAt f f' s a
    c : Nat → Real
    d : Nat → E
    hc : Filter.Tendsto c Filter.atTop Filter.atTop
    hcd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    hc' : Filter.Tendsto (fun x => Norm.norm (c x)) Filter.atTop Filter.atTop
    hd : Filter.Tendsto (fun n => HAdd.hAdd a (d n)) Filter.atTop (nhdsWithin a s)
    ⊢ Filter.Eventually (fun n => LE.le (HSMul.hSMul (c n) (HSub.hSub (f (HAdd.hAd …
  -/
  filter_upwards [hd.eventually h, hc.eventually_ge_atTop 0] with n hfn hcn
  /-
    case h
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    a y : E
    h : IsLocalMaxOn f s a
    hf : HasFDerivWithinAt f f' s a
    c : Nat → Real
    d : Nat → E
    hc : Filter.Tendsto c Filter.atTop Filter.atTop
    hcd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds y)
    hc' : Filter.Tendsto (fun x => Norm.norm (c x)) Filter.atTop Filter.atTop
    hd : Filter.Tendsto (fun n => HAdd.hAdd a (d n)) Filter.atTop (nhdsWithin a s)
    n : Nat
    hfn : LE.le (f (HAdd.hAdd a (d n))) (f a)
    hcn : LE.le 0 (c n)
    ⊢ LE.le (HSMul.hSMul (c n) (HSub.hSub (f (HAdd.hAdd a (d n))) (f a))) 0
  -/
  exact mul_nonpos_of_nonneg_of_nonpos hcn (sub_nonpos.2 hfn)
  /-
    🎉 no goals
  -/


/-- If `f` has a local max on `s` at `a` and `y` belongs to the positive tangent cone
of `s` at `a`, then `f' y ≤ 0`. -/
theorem IsLocalMaxOn.fderivWithin_nonpos (h : IsLocalMaxOn f s a)
    (hy : y ∈ posTangentConeAt s a) : (fderivWithin ℝ f s a : E → ℝ) y ≤ 0 := by
  classical
  exact
    if hf : DifferentiableWithinAt ℝ f s a then h.hasFDerivWithinAt_nonpos hf.hasFDerivWithinAt hy
    else by rw [fderivWithin_zero_of_not_differentiableWithinAt hf]; rfl


/-- If `f` has a local max on `s` at `a`, `f'` is a derivative of `f` at `a` within `s`, and
both `y` and `-y` belong to the positive tangent cone of `s` at `a`, then `f' y ≤ 0`. -/
theorem IsLocalMaxOn.hasFDerivWithinAt_eq_zero (h : IsLocalMaxOn f s a)
    (hf : HasFDerivWithinAt f f' s a) (hy : y ∈ posTangentConeAt s a)
    (hy' : -y ∈ posTangentConeAt s a) : f' y = 0 :=
                                                       /-
                                                         E : Type u
                                                         inst✝¹ : NormedAddCommGroup E
                                                         inst✝ : NormedSpace Real E
                                                         f : E → Real
                                                         f' : ContinuousLinearMap (RingHom.id Real) E Real
                                                         s : Set E
                                                         a y : E
                                                         h : IsLocalMaxOn f s a
                                                         hf : HasFDerivWithinAt f f' s a
                                                         hy : Membership.mem (posTangentConeAt s a) y
                                                         hy' : Membership.mem (posTangentConeAt s a) (Neg.neg y)
                                                         ⊢ LE.le 0 (f' y)
                                                       -/
  le_antisymm (h.hasFDerivWithinAt_nonpos hf hy) <| by simpa using h.hasFDerivWithinAt_nonpos hf hy'
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- If `f` has a local max on `s` at `a` and both `y` and `-y` belong to the positive tangent cone
of `s` at `a`, then `f' y = 0`. -/
theorem IsLocalMaxOn.fderivWithin_eq_zero (h : IsLocalMaxOn f s a)
    (hy : y ∈ posTangentConeAt s a) (hy' : -y ∈ posTangentConeAt s a) :
    (fderivWithin ℝ f s a : E → ℝ) y = 0 := by
  classical
  exact if hf : DifferentiableWithinAt ℝ f s a then
    h.hasFDerivWithinAt_eq_zero hf.hasFDerivWithinAt hy hy'
  else by rw [fderivWithin_zero_of_not_differentiableWithinAt hf]; rfl


/-- If `f` has a local min on `s` at `a`, `f'` is the derivative of `f` at `a` within `s`, and
`y` belongs to the positive tangent cone of `s` at `a`, then `0 ≤ f' y`. -/
theorem IsLocalMinOn.hasFDerivWithinAt_nonneg (h : IsLocalMinOn f s a)
    (hf : HasFDerivWithinAt f f' s a) (hy : y ∈ posTangentConeAt s a) : 0 ≤ f' y := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    a y : E
    h : IsLocalMinOn f s a
    hf : HasFDerivWithinAt f f' s a
    hy : Membership.mem (posTangentConeAt s a) y
    ⊢ LE.le 0 (f' y)
  -/
  simpa using h.neg.hasFDerivWithinAt_nonpos hf.neg hy
  /-
    🎉 no goals
  -/


/-- If `f` has a local min on `s` at `a` and `y` belongs to the positive tangent cone
of `s` at `a`, then `0 ≤ f' y`. -/
theorem IsLocalMinOn.fderivWithin_nonneg (h : IsLocalMinOn f s a)
    (hy : y ∈ posTangentConeAt s a) : (0 : ℝ) ≤ (fderivWithin ℝ f s a : E → ℝ) y := by
  classical
  exact
    if hf : DifferentiableWithinAt ℝ f s a then h.hasFDerivWithinAt_nonneg hf.hasFDerivWithinAt hy
    else by rw [fderivWithin_zero_of_not_differentiableWithinAt hf]; rfl


/-- If `f` has a local max on `s` at `a`, `f'` is a derivative of `f` at `a` within `s`, and
both `y` and `-y` belong to the positive tangent cone of `s` at `a`, then `f' y ≤ 0`. -/
theorem IsLocalMinOn.hasFDerivWithinAt_eq_zero (h : IsLocalMinOn f s a)
    (hf : HasFDerivWithinAt f f' s a) (hy : y ∈ posTangentConeAt s a)
    (hy' : -y ∈ posTangentConeAt s a) : f' y = 0 := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    s : Set E
    a y : E
    h : IsLocalMinOn f s a
    hf : HasFDerivWithinAt f f' s a
    hy : Membership.mem (posTangentConeAt s a) y
    hy' : Membership.mem (posTangentConeAt s a) (Neg.neg y)
    ⊢ Eq (f' y) 0
  -/
  simpa using h.neg.hasFDerivWithinAt_eq_zero hf.neg hy hy'
  /-
    🎉 no goals
  -/


/-- If `f` has a local min on `s` at `a` and both `y` and `-y` belong to the positive tangent cone
of `s` at `a`, then `f' y = 0`. -/
theorem IsLocalMinOn.fderivWithin_eq_zero (h : IsLocalMinOn f s a)
    (hy : y ∈ posTangentConeAt s a) (hy' : -y ∈ posTangentConeAt s a) :
    (fderivWithin ℝ f s a : E → ℝ) y = 0 := by
  classical
  exact if hf : DifferentiableWithinAt ℝ f s a then
    h.hasFDerivWithinAt_eq_zero hf.hasFDerivWithinAt hy hy'
  else by rw [fderivWithin_zero_of_not_differentiableWithinAt hf]; rfl


/-- **Fermat's Theorem**: the derivative of a function at a local minimum equals zero. -/
theorem IsLocalMin.hasFDerivAt_eq_zero (h : IsLocalMin f a) (hf : HasFDerivAt f f' a) : f' = 0 := by
  /-
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    a : E
    h : IsLocalMin f a
    hf : HasFDerivAt f f' a
    ⊢ Eq f' 0
  -/
  ext y
  /-
    case h
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : E → Real
    f' : ContinuousLinearMap (RingHom.id Real) E Real
    a : E
    h : IsLocalMin f a
    hf : HasFDerivAt f f' a
    y : E
    ⊢ Eq (f' y) (0 y)
  -/
  apply (h.on univ).hasFDerivWithinAt_eq_zero hf.hasFDerivWithinAt <;>
      /-
        case h.hy
        E : Type u
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : E → Real
        f' : ContinuousLinearMap (RingHom.id Real) E Real
        a : E
        h : IsLocalMin f a
        hf : HasFDerivAt f f' a
        y : E
        ⊢ Membership.mem (posTangentConeAt Set.univ a) y
      -/
      rw [posTangentConeAt_univ] <;>
    /-
      case h.hy
      E : Type u
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : E → Real
      f' : ContinuousLinearMap (RingHom.id Real) E Real
      a : E
      h : IsLocalMin f a
      hf : HasFDerivAt f f' a
      y : E
      ⊢ Membership.mem Set.univ y
    -/
    /-
      🎉 no goals
    -/
    apply mem_univ
    /-
      🎉 no goals
    -/


/-- **Fermat's Theorem**: the derivative of a function at a local minimum equals zero. -/
theorem IsLocalMin.fderiv_eq_zero (h : IsLocalMin f a) : fderiv ℝ f a = 0 := by
  classical
  exact if hf : DifferentiableAt ℝ f a then h.hasFDerivAt_eq_zero hf.hasFDerivAt
  else fderiv_zero_of_not_differentiableAt hf


/-- **Fermat's Theorem**: the derivative of a function at a local maximum equals zero. -/
theorem IsLocalMax.hasFDerivAt_eq_zero (h : IsLocalMax f a) (hf : HasFDerivAt f f' a) : f' = 0 :=
  neg_eq_zero.1 <| h.neg.hasFDerivAt_eq_zero hf.neg


/-- **Fermat's Theorem**: the derivative of a function at a local maximum equals zero. -/
theorem IsLocalMax.fderiv_eq_zero (h : IsLocalMax f a) : fderiv ℝ f a = 0 := by
  classical
  exact if hf : DifferentiableAt ℝ f a then h.hasFDerivAt_eq_zero hf.hasFDerivAt
  else fderiv_zero_of_not_differentiableAt hf


/-- **Fermat's Theorem**: the derivative of a function at a local extremum equals zero. -/
theorem IsLocalExtr.hasFDerivAt_eq_zero (h : IsLocalExtr f a) : HasFDerivAt f f' a → f' = 0 :=
  h.elim IsLocalMin.hasFDerivAt_eq_zero IsLocalMax.hasFDerivAt_eq_zero


/-- **Fermat's Theorem**: the derivative of a function at a local extremum equals zero. -/
theorem IsLocalExtr.fderiv_eq_zero (h : IsLocalExtr f a) : fderiv ℝ f a = 0 :=
  h.elim IsLocalMin.fderiv_eq_zero IsLocalMax.fderiv_eq_zero


lemma one_mem_posTangentConeAt_iff_mem_closure :
    1 ∈ posTangentConeAt s a ↔ a ∈ closure (Ioi a ∩ s) := by
  /-
    s : Set Real
    a : Real
    ⊢ Iff (Membership.mem (posTangentConeAt s a) 1) (Membership.mem (closure (Inte …
  -/
  constructor
    /-
      case mp
      s : Set Real
      a : Real
      ⊢ Membership.mem (posTangentConeAt s a) 1 → Membership.mem (closure (Inter.int …
    -/
  · rintro ⟨c, d, hs, hc, hcd⟩
    have : Tendsto (a + d ·) atTop (𝓝 a) := by
      simpa only [add_zero] using tendsto_const_nhds.add
        (tangentConeAt.lim_zero _ (tendsto_abs_atTop_atTop.comp hc) hcd)
    /-
      case mp.intro.intro.intro.intro
      s : Set Real
      a : Real
      c d : Nat → Real
      hs : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd a (d n))) Filter. …
      hc : Filter.Tendsto c Filter.atTop Filter.atTop
      hcd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds 1)
      this : Filter.Tendsto (fun x => HAdd.hAdd a (d x)) Filter.atTop (nhds a)
      ⊢ Membership.mem (closure (Inter.inter (Set.Ioi a) s)) a
    -/
    apply mem_closure_of_tendsto this
    filter_upwards [hc.eventually_gt_atTop 0, hcd.eventually (lt_mem_nhds one_pos), hs]
      with n hcn hcdn hdn
    /-
      case h
      s : Set Real
      a : Real
      c d : Nat → Real
      hs : Filter.Eventually (fun n => Membership.mem s (HAdd.hAdd a (d n))) Filter. …
      hc : Filter.Tendsto c Filter.atTop Filter.atTop
      hcd : Filter.Tendsto (fun n => HSMul.hSMul (c n) (d n)) Filter.atTop (nhds 1)
      this : Filter.Tendsto (fun x => HAdd.hAdd a (d x)) Filter.atTop (nhds a)
      n : Nat
      hcn : LT.lt 0 (c n)
      hcdn : LT.lt 0 (HSMul.hSMul (c n) (d n))
      hdn : Membership.mem s (HAdd.hAdd a (d n))
      ⊢ Membership.mem (Inter.inter (Set.Ioi a) s) (HAdd.hAdd a (d n))
    -/
    simp_all
    /-
      🎉 no goals
    -/
    /-
      case mpr
      s : Set Real
      a : Real
      ⊢ Membership.mem (closure (Inter.inter (Set.Ioi a) s)) a → Membership.mem (pos …
    -/
  · intro h
    /-
      case mpr
      s : Set Real
      a : Real
      h : Membership.mem (closure (Inter.inter (Set.Ioi a) s)) a
      ⊢ Membership.mem (posTangentConeAt s a) 1
    -/
    apply mem_posTangentConeAt_of_frequently_mem
    /-
      case mpr.h
      s : Set Real
      a : Real
      h : Membership.mem (closure (Inter.inter (Set.Ioi a) s)) a
      ⊢ Filter.Frequently (fun t => Membership.mem s (HAdd.hAdd a (HSMul.hSMul t 1)) …
    -/
    rw [mem_closure_iff_frequently, ← map_add_left_nhds_zero, frequently_map] at h
    /-
      case mpr.h
      s : Set Real
      a : Real
      h : Filter.Frequently (fun a_1 => Membership.mem (Inter.inter (Set.Ioi a) s) ( …
      ⊢ Filter.Frequently (fun t => Membership.mem s (HAdd.hAdd a (HSMul.hSMul t 1)) …
    -/
    simpa [nhdsWithin, frequently_inf_principal] using h
    /-
      🎉 no goals
    -/


lemma one_mem_posTangentConeAt_iff_frequently :
    1 ∈ posTangentConeAt s a ↔ ∃ᶠ x in 𝓝[>] a, x ∈ s := by
  rw [one_mem_posTangentConeAt_iff_mem_closure, mem_closure_iff_frequently,
    frequently_nhdsWithin_iff, inter_comm]
  /-
    s : Set Real
    a : Real
    ⊢ Iff (Filter.Frequently (fun x => Membership.mem (Inter.inter s (Set.Ioi a))  …
  -/
  simp_rw [mem_inter_iff]
  /-
    🎉 no goals
  -/


/-- **Fermat's Theorem**: the derivative of a function at a local minimum equals zero. -/
theorem IsLocalMin.hasDerivAt_eq_zero (h : IsLocalMin f a) (hf : HasDerivAt f f' a) : f' = 0 := by
  /-
    f : Real → Real
    f' a : Real
    h : IsLocalMin f a
    hf : HasDerivAt f f' a
    ⊢ Eq f' 0
  -/
  simpa using DFunLike.congr_fun (h.hasFDerivAt_eq_zero (hasDerivAt_iff_hasFDerivAt.1 hf)) 1
  /-
    🎉 no goals
  -/


/-- **Fermat's Theorem**: the derivative of a function at a local minimum equals zero. -/
theorem IsLocalMin.deriv_eq_zero (h : IsLocalMin f a) : deriv f a = 0 := by
  classical
  exact if hf : DifferentiableAt ℝ f a then h.hasDerivAt_eq_zero hf.hasDerivAt
  else deriv_zero_of_not_differentiableAt hf


/-- **Fermat's Theorem**: the derivative of a function at a local maximum equals zero. -/
theorem IsLocalMax.hasDerivAt_eq_zero (h : IsLocalMax f a) (hf : HasDerivAt f f' a) : f' = 0 :=
  neg_eq_zero.1 <| h.neg.hasDerivAt_eq_zero hf.neg


/-- **Fermat's Theorem**: the derivative of a function at a local maximum equals zero. -/
theorem IsLocalMax.deriv_eq_zero (h : IsLocalMax f a) : deriv f a = 0 := by
  classical
  exact if hf : DifferentiableAt ℝ f a then h.hasDerivAt_eq_zero hf.hasDerivAt
  else deriv_zero_of_not_differentiableAt hf


/-- **Fermat's Theorem**: the derivative of a function at a local extremum equals zero. -/
theorem IsLocalExtr.hasDerivAt_eq_zero (h : IsLocalExtr f a) : HasDerivAt f f' a → f' = 0 :=
  h.elim IsLocalMin.hasDerivAt_eq_zero IsLocalMax.hasDerivAt_eq_zero


/-- **Fermat's Theorem**: the derivative of a function at a local extremum equals zero. -/
theorem IsLocalExtr.deriv_eq_zero (h : IsLocalExtr f a) : deriv f a = 0 :=
  h.elim IsLocalMin.deriv_eq_zero IsLocalMax.deriv_eq_zero


