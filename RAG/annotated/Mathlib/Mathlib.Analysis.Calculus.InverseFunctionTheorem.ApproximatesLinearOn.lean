/-- We say that `f` approximates a continuous linear map `f'` on `s` with constant `c`,
if `‖f x - f y - f' (x - y)‖ ≤ c * ‖x - y‖` whenever `x, y ∈ s`.

This predicate is defined to facilitate the splitting of the inverse function theorem into small
lemmas. Some of these lemmas can be useful, e.g., to prove that the inverse function is defined
on a specific set. -/
def ApproximatesLinearOn (f : E → F) (f' : E →L[𝕜] F) (s : Set E) (c : ℝ≥0) : Prop :=
  ∀ x ∈ s, ∀ y ∈ s, ‖f x - f y - f' (x - y)‖ ≤ c * ‖x - y‖


@[simp]
theorem approximatesLinearOn_empty (f : E → F) (f' : E →L[𝕜] F) (c : ℝ≥0) :
                                        /-
                                          𝕜 : Type u_1
                                          inst✝⁴ : NontriviallyNormedField 𝕜
                                          E : Type u_2
                                          inst✝³ : NormedAddCommGroup E
                                          inst✝² : NormedSpace 𝕜 E
                                          F : Type u_3
                                          inst✝¹ : NormedAddCommGroup F
                                          inst✝ : NormedSpace 𝕜 F
                                          f : E → F
                                          f' : ContinuousLinearMap (RingHom.id 𝕜) E F
                                          c : NNReal
                                          ⊢ ApproximatesLinearOn f f' EmptyCollection.emptyCollection c
                                        -/
    ApproximatesLinearOn f f' ∅ c := by simp [ApproximatesLinearOn]
                                        /-
                                          🎉 no goals
                                        -/


theorem mono_num (hc : c ≤ c') (hf : ApproximatesLinearOn f f' s c) :
    ApproximatesLinearOn f f' s c' := fun x hx y hy =>
  le_trans (hf x hx y hy) (mul_le_mul_of_nonneg_right hc <| norm_nonneg _)


theorem mono_set (hst : s ⊆ t) (hf : ApproximatesLinearOn f f' t c) :
    ApproximatesLinearOn f f' s c := fun x hx y hy => hf x (hst hx) y (hst hy)


theorem approximatesLinearOn_iff_lipschitzOnWith {f : E → F} {f' : E →L[𝕜] F} {s : Set E}
    {c : ℝ≥0} : ApproximatesLinearOn f f' s c ↔ LipschitzOnWith c (f - ⇑f') s := by
  have : ∀ x y, f x - f y - f' (x - y) = (f - f') x - (f - f') y := fun x y ↦ by
    simp only [map_sub, Pi.sub_apply]; abel
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    this : ∀ (x y : E), Eq (HSub.hSub (HSub.hSub (f x) (f y)) (f' (HSub.hSub x y)) …
    ⊢ Iff (ApproximatesLinearOn f f' s c) (LipschitzOnWith c (HSub.hSub f ⇑f') s)
  -/
  simp only [this, lipschitzOnWith_iff_norm_sub_le, ApproximatesLinearOn]
  /-
    🎉 no goals
  -/


alias ⟨lipschitzOnWith, _root_.LipschitzOnWith.approximatesLinearOn⟩ :=
  approximatesLinearOn_iff_lipschitzOnWith


theorem lipschitz_sub (hf : ApproximatesLinearOn f f' s c) :
    LipschitzWith c fun x : s => f x - f' x :=
  hf.lipschitzOnWith.to_restrict


protected theorem lipschitz (hf : ApproximatesLinearOn f f' s c) :
    LipschitzWith (‖f'‖₊ + c) (s.restrict f) := by
  simpa only [restrict_apply, add_sub_cancel] using
    (f'.lipschitz.restrict s).add hf.lipschitz_sub


protected theorem continuous (hf : ApproximatesLinearOn f f' s c) : Continuous (s.restrict f) :=
  hf.lipschitz.continuous


protected theorem continuousOn (hf : ApproximatesLinearOn f f' s c) : ContinuousOn f s :=
  continuousOn_iff_continuous_restrict.2 hf.continuous


/-- If a function is linearly approximated by a continuous linear map with a (possibly nonlinear)
right inverse, then it is locally onto: a ball of an explicit radius is included in the image
of the map. -/
theorem surjOn_closedBall_of_nonlinearRightInverse
    (hf : ApproximatesLinearOn f f' s c)
    (f'symm : f'.NonlinearRightInverse) {ε : ℝ} {b : E} (ε0 : 0 ≤ ε) (hε : closedBall b ε ⊆ s) :
    SurjOn f (closedBall b ε) (closedBall (f b) (((f'symm.nnnorm : ℝ)⁻¹ - c) * ε)) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    ⊢ Set.SurjOn f (Metric.closedBall b ε) (Metric.closedBall (f b) (HMul.hMul (HS …
  -/
  intro y hy
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  rcases le_or_lt (f'symm.nnnorm : ℝ)⁻¹ c with hc | hc
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      inst✝ : CompleteSpace E
      s : Set E
      c : NNReal
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      hf : ApproximatesLinearOn f f' s c
      f'symm : f'.NonlinearRightInverse
      ε : Real
      b : E
      ε0 : LE.le 0 ε
      hε : HasSubset.Subset (Metric.closedBall b ε) s
      y : F
      hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
      hc : LE.le (Inv.inv ↑f'symm.nnnorm) ↑c
      ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
    -/
  · refine ⟨b, by simp [ε0], ?_⟩
    have : dist y (f b) ≤ 0 :=
      (mem_closedBall.1 hy).trans (mul_nonpos_of_nonpos_of_nonneg (by linarith) ε0)
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      inst✝ : CompleteSpace E
      s : Set E
      c : NNReal
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      hf : ApproximatesLinearOn f f' s c
      f'symm : f'.NonlinearRightInverse
      ε : Real
      b : E
      ε0 : LE.le 0 ε
      hε : HasSubset.Subset (Metric.closedBall b ε) s
      y : F
      hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
      hc : LE.le (Inv.inv ↑f'symm.nnnorm) ↑c
      this : LE.le (Dist.dist y (f b)) 0
      ⊢ Eq (f b) y
    -/
    simp only [dist_le_zero] at this
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      inst✝ : CompleteSpace E
      s : Set E
      c : NNReal
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      hf : ApproximatesLinearOn f f' s c
      f'symm : f'.NonlinearRightInverse
      ε : Real
      b : E
      ε0 : LE.le 0 ε
      hε : HasSubset.Subset (Metric.closedBall b ε) s
      y : F
      hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
      hc : LE.le (Inv.inv ↑f'symm.nnnorm) ↑c
      this : Eq y (f b)
      ⊢ Eq (f b) y
    -/
    rw [this]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  have If' : (0 : ℝ) < f'symm.nnnorm := by rw [← inv_pos]; exact (NNReal.coe_nonneg _).trans_lt hc
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  have Icf' : (c : ℝ) * f'symm.nnnorm < 1 := by rwa [inv_eq_one_div, lt_div_iff₀ If'] at hc
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    Icf' : LT.lt (HMul.hMul ↑c ↑f'symm.nnnorm) 1
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  have Jf' : (f'symm.nnnorm : ℝ) ≠ 0 := ne_of_gt If'
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    Icf' : LT.lt (HMul.hMul ↑c ↑f'symm.nnnorm) 1
    Jf' : Ne (↑f'symm.nnnorm) 0
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  have Jcf' : (1 : ℝ) - c * f'symm.nnnorm ≠ 0 := by apply ne_of_gt; linarith
  /- We have to show that `y` can be written as `f x` for some `x ∈ closedBall b ε`.
    The idea of the proof is to apply the Banach contraction principle to the map
    `g : x ↦ x + f'symm (y - f x)`, as a fixed point of this map satisfies `f x = y`.
    When `f'symm` is a genuine linear inverse, `g` is a contracting map. In our case, since `f'symm`
    is nonlinear, this map is not contracting (it is not even continuous), but still the proof of
    the contraction theorem holds: `uₙ = gⁿ b` is a Cauchy sequence, converging exponentially fast
    to the desired point `x`. Instead of appealing to general results, we check this by hand.

    The main point is that `f (u n)` becomes exponentially close to `y`, and therefore
    `dist (u (n+1)) (u n)` becomes exponentally small, making it possible to get an inductive
    bound on `dist (u n) b`, from which one checks that `u n` stays in the ball on which one has a
    control. Therefore, the bound can be checked at the next step, and so on inductively.
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    Icf' : LT.lt (HMul.hMul ↑c ↑f'symm.nnnorm) 1
    Jf' : Ne (↑f'symm.nnnorm) 0
    Jcf' : Ne (HSub.hSub 1 (HMul.hMul ↑c ↑f'symm.nnnorm)) 0
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  set g := fun x => x + f'symm (y - f x) with hg
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    Icf' : LT.lt (HMul.hMul ↑c ↑f'symm.nnnorm) 1
    Jf' : Ne (↑f'symm.nnnorm) 0
    Jcf' : Ne (HSub.hSub 1 (HMul.hMul ↑c ↑f'symm.nnnorm)) 0
    g : E → E := fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    hg : Eq g fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  set u := fun n : ℕ => g^[n] b with hu
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    Icf' : LT.lt (HMul.hMul ↑c ↑f'symm.nnnorm) 1
    Jf' : Ne (↑f'symm.nnnorm) 0
    Jcf' : Ne (HSub.hSub 1 (HMul.hMul ↑c ↑f'symm.nnnorm)) 0
    g : E → E := fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    hg : Eq g fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    u : Nat → E := fun n => Nat.iterate g n b
    hu : Eq u fun n => Nat.iterate g n b
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  have usucc : ∀ n, u (n + 1) = g (u n) := by simp [hu, ← iterate_succ_apply' g _ b]
  -- First bound: if `f z` is close to `y`, then `g z` is close to `z` (i.e., almost a fixed point).
  have A : ∀ z, dist (g z) z ≤ f'symm.nnnorm * dist (f z) y := by
    intro z
    rw [dist_eq_norm, hg, add_sub_cancel_left, dist_eq_norm']
    exact f'symm.bound _
  -- Second bound: if `z` and `g z` are in the set with good control, then `f (g z)` becomes closer
  -- to `y` than `f z` was (this uses the linear approximation property, and is the reason for the
  -- choice of the formula for `g`).
  have B :
    ∀ z ∈ closedBall b ε,
      g z ∈ closedBall b ε → dist (f (g z)) y ≤ c * f'symm.nnnorm * dist (f z) y := by
    intro z hz hgz
    set v := f'symm (y - f z)
    calc
      dist (f (g z)) y = ‖f (z + v) - y‖ := by rw [dist_eq_norm]
      _ = ‖f (z + v) - f z - f' v + f' v - (y - f z)‖ := by congr 1; abel
      _ = ‖f (z + v) - f z - f' (z + v - z)‖ := by
        simp only [v, ContinuousLinearMap.NonlinearRightInverse.right_inv, add_sub_cancel_left,
          sub_add_cancel]
      _ ≤ c * ‖z + v - z‖ := hf _ (hε hgz) _ (hε hz)
      _ ≤ c * (f'symm.nnnorm * dist (f z) y) := by
        gcongr
        simpa [dist_eq_norm'] using f'symm.bound (y - f z)
      _ = c * f'symm.nnnorm * dist (f z) y := by ring
  -- Third bound: a complicated bound on `dist w b` (that will show up in the induction) is enough
  -- to check that `w` is in the ball on which one has controls. Will be used to check that `u n`
  -- belongs to this ball for all `n`.
  have C : ∀ (n : ℕ) (w : E), dist w b ≤ f'symm.nnnorm * (1 - ((c : ℝ) * f'symm.nnnorm) ^ n) /
      (1 - c * f'symm.nnnorm) * dist (f b) y → w ∈ closedBall b ε := fun n w hw ↦ by
    apply hw.trans
    rw [div_mul_eq_mul_div, div_le_iff₀]; swap; · linarith
    calc
      (f'symm.nnnorm : ℝ) * (1 - ((c : ℝ) * f'symm.nnnorm) ^ n) * dist (f b) y =
          f'symm.nnnorm * dist (f b) y * (1 - ((c : ℝ) * f'symm.nnnorm) ^ n) := by
        ring
      _ ≤ f'symm.nnnorm * dist (f b) y * 1 := by
        gcongr
        rw [sub_le_self_iff]
        positivity
      _ ≤ f'symm.nnnorm * (((f'symm.nnnorm : ℝ)⁻¹ - c) * ε) := by
        rw [mul_one]
        gcongr
        exact mem_closedBall'.1 hy
      _ = ε * (1 - c * f'symm.nnnorm) := by field_simp; ring

  /- Main inductive control: `f (u n)` becomes exponentially close to `y`, and therefore
    `dist (u (n+1)) (u n)` becomes exponentally small, making it possible to get an inductive
    bound on `dist (u n) b`, from which one checks that `u n` remains in the ball on which we
    have estimates. -/
  have D : ∀ n : ℕ, dist (f (u n)) y ≤ ((c : ℝ) * f'symm.nnnorm) ^ n * dist (f b) y ∧
      dist (u n) b ≤ f'symm.nnnorm * (1 - ((c : ℝ) * f'symm.nnnorm) ^ n) /
        (1 - (c : ℝ) * f'symm.nnnorm) * dist (f b) y := fun n ↦ by
    induction' n with n IH; · simp [hu, le_refl]
    rw [usucc]
    have Ign : dist (g (u n)) b ≤ f'symm.nnnorm * (1 - ((c : ℝ) * f'symm.nnnorm) ^ n.succ) /
        (1 - c * f'symm.nnnorm) * dist (f b) y :=
      calc
        dist (g (u n)) b ≤ dist (g (u n)) (u n) + dist (u n) b := dist_triangle _ _ _
        _ ≤ f'symm.nnnorm * dist (f (u n)) y + dist (u n) b := add_le_add (A _) le_rfl
        _ ≤ f'symm.nnnorm * (((c : ℝ) * f'symm.nnnorm) ^ n * dist (f b) y) +
              f'symm.nnnorm * (1 - ((c : ℝ) * f'symm.nnnorm) ^ n) / (1 - c * f'symm.nnnorm) *
                dist (f b) y := by
                  gcongr
                  · exact IH.1
                  · exact IH.2
        _ = f'symm.nnnorm * (1 - ((c : ℝ) * f'symm.nnnorm) ^ n.succ) /
              (1 - (c : ℝ) * f'symm.nnnorm) * dist (f b) y := by
          field_simp [Jcf', pow_succ]; ring
    refine ⟨?_, Ign⟩
    calc
      dist (f (g (u n))) y ≤ c * f'symm.nnnorm * dist (f (u n)) y :=
        B _ (C n _ IH.2) (C n.succ _ Ign)
      _ ≤ (c : ℝ) * f'symm.nnnorm * (((c : ℝ) * f'symm.nnnorm) ^ n * dist (f b) y) := by
        gcongr
        apply IH.1
      _ = ((c : ℝ) * f'symm.nnnorm) ^ n.succ * dist (f b) y := by simp only [pow_succ']; ring
  -- Deduce from the inductive bound that `uₙ` is a Cauchy sequence, therefore converging.
  have : CauchySeq u := by
    refine cauchySeq_of_le_geometric _ (↑f'symm.nnnorm * dist (f b) y) Icf' fun n ↦ ?_
    calc
      dist (u n) (u (n + 1)) = dist (g (u n)) (u n) := by rw [usucc, dist_comm]
      _ ≤ f'symm.nnnorm * dist (f (u n)) y := A _
      _ ≤ f'symm.nnnorm * (((c : ℝ) * f'symm.nnnorm) ^ n * dist (f b) y) := by
        gcongr
        exact (D n).1
      _ = f'symm.nnnorm * dist (f b) y * ((c : ℝ) * f'symm.nnnorm) ^ n := by ring
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    Icf' : LT.lt (HMul.hMul ↑c ↑f'symm.nnnorm) 1
    Jf' : Ne (↑f'symm.nnnorm) 0
    Jcf' : Ne (HSub.hSub 1 (HMul.hMul ↑c ↑f'symm.nnnorm)) 0
    g : E → E := fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    hg : Eq g fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    u : Nat → E := fun n => Nat.iterate g n b
    hu : Eq u fun n => Nat.iterate g n b
    usucc : ∀ (n : Nat), Eq (u (HAdd.hAdd n 1)) (g (u n))
    A : ∀ (z : E), LE.le (Dist.dist (g z) z) (HMul.hMul (↑f'symm.nnnorm) (Dist.dis …
    B : ∀ (z : E), Membership.mem (Metric.closedBall b ε) z → Membership.mem (Metr …
    C : ∀ (n : Nat) (w : E), LE.le (Dist.dist w b) (HMul.hMul (HDiv.hDiv (HMul.hMu …
    D : ∀ (n : Nat), And (LE.le (Dist.dist (f (u n)) y) (HMul.hMul (HPow.hPow (HMu …
    this : CauchySeq u
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  obtain ⟨x, hx⟩ : ∃ x, Tendsto u atTop (𝓝 x) := cauchySeq_tendsto_of_complete this
  -- As all the `uₙ` belong to the ball `closedBall b ε`, so does their limit `x`.
  have xmem : x ∈ closedBall b ε :=
    isClosed_ball.mem_of_tendsto hx (Eventually.of_forall fun n => C n _ (D n).2)
  /-
    case inr.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    Icf' : LT.lt (HMul.hMul ↑c ↑f'symm.nnnorm) 1
    Jf' : Ne (↑f'symm.nnnorm) 0
    Jcf' : Ne (HSub.hSub 1 (HMul.hMul ↑c ↑f'symm.nnnorm)) 0
    g : E → E := fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    hg : Eq g fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    u : Nat → E := fun n => Nat.iterate g n b
    hu : Eq u fun n => Nat.iterate g n b
    usucc : ∀ (n : Nat), Eq (u (HAdd.hAdd n 1)) (g (u n))
    A : ∀ (z : E), LE.le (Dist.dist (g z) z) (HMul.hMul (↑f'symm.nnnorm) (Dist.dis …
    B : ∀ (z : E), Membership.mem (Metric.closedBall b ε) z → Membership.mem (Metr …
    C : ∀ (n : Nat) (w : E), LE.le (Dist.dist w b) (HMul.hMul (HDiv.hDiv (HMul.hMu …
    D : ∀ (n : Nat), And (LE.le (Dist.dist (f (u n)) y) (HMul.hMul (HPow.hPow (HMu …
    this : CauchySeq u
    x : E
    hx : Filter.Tendsto u Filter.atTop (nhds x)
    xmem : Membership.mem (Metric.closedBall b ε) x
    ⊢ Membership.mem (Set.image f (Metric.closedBall b ε)) y
  -/
  refine ⟨x, xmem, ?_⟩
  -- It remains to check that `f x = y`. This follows from continuity of `f` on `closedBall b ε`
  -- and from the fact that `f uₙ` is converging to `y` by construction.
  have hx' : Tendsto u atTop (𝓝[closedBall b ε] x) := by
    simp only [nhdsWithin, tendsto_inf, hx, true_and, tendsto_principal]
    exact Eventually.of_forall fun n => C n _ (D n).2
  have T1 : Tendsto (f ∘ u) atTop (𝓝 (f x)) :=
    (hf.continuousOn.mono hε x xmem).tendsto.comp hx'
  have T2 : Tendsto (f ∘ u) atTop (𝓝 y) := by
    rw [tendsto_iff_dist_tendsto_zero]
    refine squeeze_zero (fun _ => dist_nonneg) (fun n => (D n).1) ?_
    simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one (by positivity) Icf').mul tendsto_const_nhds
  /-
    case inr.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    ε : Real
    b : E
    ε0 : LE.le 0 ε
    hε : HasSubset.Subset (Metric.closedBall b ε) s
    y : F
    hy : Membership.mem (Metric.closedBall (f b) (HMul.hMul (HSub.hSub (Inv.inv ↑f …
    hc : LT.lt (↑c) (Inv.inv ↑f'symm.nnnorm)
    If' : LT.lt 0 ↑f'symm.nnnorm
    Icf' : LT.lt (HMul.hMul ↑c ↑f'symm.nnnorm) 1
    Jf' : Ne (↑f'symm.nnnorm) 0
    Jcf' : Ne (HSub.hSub 1 (HMul.hMul ↑c ↑f'symm.nnnorm)) 0
    g : E → E := fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    hg : Eq g fun x => HAdd.hAdd x (f'symm.toFun (HSub.hSub y (f x)))
    u : Nat → E := fun n => Nat.iterate g n b
    hu : Eq u fun n => Nat.iterate g n b
    usucc : ∀ (n : Nat), Eq (u (HAdd.hAdd n 1)) (g (u n))
    A : ∀ (z : E), LE.le (Dist.dist (g z) z) (HMul.hMul (↑f'symm.nnnorm) (Dist.dis …
    B : ∀ (z : E), Membership.mem (Metric.closedBall b ε) z → Membership.mem (Metr …
    C : ∀ (n : Nat) (w : E), LE.le (Dist.dist w b) (HMul.hMul (HDiv.hDiv (HMul.hMu …
    D : ∀ (n : Nat), And (LE.le (Dist.dist (f (u n)) y) (HMul.hMul (HPow.hPow (HMu …
    this : CauchySeq u
    x : E
    hx : Filter.Tendsto u Filter.atTop (nhds x)
    xmem : Membership.mem (Metric.closedBall b ε) x
    hx' : Filter.Tendsto u Filter.atTop (nhdsWithin x (Metric.closedBall b ε))
    T1 : Filter.Tendsto (Function.comp f u) Filter.atTop (nhds (f x))
    T2 : Filter.Tendsto (Function.comp f u) Filter.atTop (nhds y)
    ⊢ Eq (f x) y
  -/
  exact tendsto_nhds_unique T1 T2
  /-
    🎉 no goals
  -/


theorem open_image (hf : ApproximatesLinearOn f f' s c) (f'symm : f'.NonlinearRightInverse)
    (hs : IsOpen s) (hc : Subsingleton F ∨ c < f'symm.nnnorm⁻¹) : IsOpen (f '' s) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    hs : IsOpen s
    hc : Or (Subsingleton F) (LT.lt c (Inv.inv f'symm.nnnorm))
    ⊢ IsOpen (Set.image f s)
  -/
  cases' hc with hE hc
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      inst✝ : CompleteSpace E
      s : Set E
      c : NNReal
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      hf : ApproximatesLinearOn f f' s c
      f'symm : f'.NonlinearRightInverse
      hs : IsOpen s
      hE : Subsingleton F
      ⊢ IsOpen (Set.image f s)
    -/
  · exact isOpen_discrete _
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    hs : IsOpen s
    hc : LT.lt c (Inv.inv f'symm.nnnorm)
    ⊢ IsOpen (Set.image f s)
  -/
  simp only [isOpen_iff_mem_nhds, nhds_basis_closedBall.mem_iff, forall_mem_image] at hs ⊢
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    hc : LT.lt c (Inv.inv f'symm.nnnorm)
    hs : ∀ (x : E), Membership.mem s x → Exists fun i => And (LT.lt 0 i) (HasSubse …
    ⊢ ∀ ⦃x : E⦄, Membership.mem s x → Exists fun i => And (LT.lt 0 i) (HasSubset.S …
  -/
  intro x hx
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    hc : LT.lt c (Inv.inv f'symm.nnnorm)
    hs : ∀ (x : E), Membership.mem s x → Exists fun i => And (LT.lt 0 i) (HasSubse …
    x : E
    hx : Membership.mem s x
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.closedBall (f x) i …
  -/
  rcases hs x hx with ⟨ε, ε0, hε⟩
  /-
    case inr.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    hc : LT.lt c (Inv.inv f'symm.nnnorm)
    hs : ∀ (x : E), Membership.mem s x → Exists fun i => And (LT.lt 0 i) (HasSubse …
    x : E
    hx : Membership.mem s x
    ε : Real
    ε0 : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) s
    ⊢ Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.closedBall (f x) i …
  -/
  refine ⟨(f'symm.nnnorm⁻¹ - c) * ε, mul_pos (sub_pos.2 hc) ε0, ?_⟩
  /-
    case inr.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    hc : LT.lt c (Inv.inv f'symm.nnnorm)
    hs : ∀ (x : E), Membership.mem s x → Exists fun i => And (LT.lt 0 i) (HasSubse …
    x : E
    hx : Membership.mem s x
    ε : Real
    ε0 : LT.lt 0 ε
    hε : HasSubset.Subset (Metric.closedBall x ε) s
    ⊢ HasSubset.Subset (Metric.closedBall (f x) (HMul.hMul (HSub.hSub ↑(Inv.inv f' …
  -/
  exact (hf.surjOn_closedBall_of_nonlinearRightInverse f'symm (le_of_lt ε0) hε).mono hε Subset.rfl
  /-
    🎉 no goals
  -/


theorem image_mem_nhds (hf : ApproximatesLinearOn f f' s c) (f'symm : f'.NonlinearRightInverse)
    {x : E} (hs : s ∈ 𝓝 x) (hc : Subsingleton F ∨ c < f'symm.nnnorm⁻¹) : f '' s ∈ 𝓝 (f x) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    x : E
    hs : Membership.mem (nhds x) s
    hc : Or (Subsingleton F) (LT.lt c (Inv.inv f'symm.nnnorm))
    ⊢ Membership.mem (nhds (f x)) (Set.image f s)
  -/
  obtain ⟨t, hts, ht, xt⟩ : ∃ t, t ⊆ s ∧ IsOpen t ∧ x ∈ t := _root_.mem_nhds_iff.1 hs
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    x : E
    hs : Membership.mem (nhds x) s
    hc : Or (Subsingleton F) (LT.lt c (Inv.inv f'symm.nnnorm))
    t : Set E
    hts : HasSubset.Subset t s
    ht : IsOpen t
    xt : Membership.mem t x
    ⊢ Membership.mem (nhds (f x)) (Set.image f s)
  -/
  have := IsOpen.mem_nhds ((hf.mono_set hts).open_image f'symm ht hc) (mem_image_of_mem _ xt)
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    x : E
    hs : Membership.mem (nhds x) s
    hc : Or (Subsingleton F) (LT.lt c (Inv.inv f'symm.nnnorm))
    t : Set E
    hts : HasSubset.Subset t s
    ht : IsOpen t
    xt : Membership.mem t x
    this : Membership.mem (nhds (f x)) (Set.image f t)
    ⊢ Membership.mem (nhds (f x)) (Set.image f s)
  -/
  exact mem_of_superset this (image_subset _ hts)
  /-
    🎉 no goals
  -/


theorem map_nhds_eq (hf : ApproximatesLinearOn f f' s c) (f'symm : f'.NonlinearRightInverse) {x : E}
    (hs : s ∈ 𝓝 x) (hc : Subsingleton F ∨ c < f'symm.nnnorm⁻¹) : map f (𝓝 x) = 𝓝 (f x) := by
  refine
    le_antisymm ((hf.continuousOn x (mem_of_mem_nhds hs)).continuousAt hs) (le_map fun t ht => ?_)
  have : f '' (s ∩ t) ∈ 𝓝 (f x) :=
    (hf.mono_set inter_subset_left).image_mem_nhds f'symm (inter_mem hs ht) hc
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : CompleteSpace E
    s : Set E
    c : NNReal
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    hf : ApproximatesLinearOn f f' s c
    f'symm : f'.NonlinearRightInverse
    x : E
    hs : Membership.mem (nhds x) s
    hc : Or (Subsingleton F) (LT.lt c (Inv.inv f'symm.nnnorm))
    t : Set E
    ht : Membership.mem (nhds x) t
    this : Membership.mem (nhds (f x)) (Set.image f (Inter.inter s t))
    ⊢ Membership.mem (nhds (f x)) (Set.image f t)
  -/
  exact mem_of_superset this (image_subset _ inter_subset_right)
  /-
    🎉 no goals
  -/


local notation "N" => ‖(f'.symm : F →L[𝕜] E)‖₊


protected theorem antilipschitz (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) : AntilipschitzWith (N⁻¹ - c)⁻¹ (s.restrict f) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    ⊢ AntilipschitzWith (Inv.inv (HSub.hSub (Inv.inv (NNNorm.nnnorm ↑f'.symm)) c)) …
  -/
  cases' hc with hE hc
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      s : Set E
      c : NNReal
      hf : ApproximatesLinearOn f (↑f') s c
      hE : Subsingleton E
      ⊢ AntilipschitzWith (Inv.inv (HSub.hSub (Inv.inv (NNNorm.nnnorm ↑f'.symm)) c)) …
    -/
  · exact AntilipschitzWith.of_subsingleton
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm))
    ⊢ AntilipschitzWith (Inv.inv (HSub.hSub (Inv.inv (NNNorm.nnnorm ↑f'.symm)) c)) …
  -/
  convert (f'.antilipschitz.restrict s).add_lipschitzWith hf.lipschitz_sub hc
  /-
    case h.e'_6.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm))
    x✝ : ↑s
    ⊢ Eq (s.restrict f x✝) (HAdd.hAdd (s.restrict (⇑f') x✝) (HSub.hSub (f ↑x✝) (↑f …
  -/
  simp [restrict]
  /-
    🎉 no goals
  -/


protected theorem injective (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) : Injective (s.restrict f) :=
  (hf.antilipschitz hc).injective


protected theorem injOn (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) : InjOn f s :=
  injOn_iff_injective.2 <| hf.injective hc


protected theorem surjective [CompleteSpace E] (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) univ c)
    (hc : Subsingleton E ∨ c < N⁻¹) : Surjective f := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    c : NNReal
    inst✝ : CompleteSpace E
    hf : ApproximatesLinearOn f (↑f') Set.univ c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    ⊢ Function.Surjective f
  -/
  cases' hc with hE hc
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      c : NNReal
      inst✝ : CompleteSpace E
      hf : ApproximatesLinearOn f (↑f') Set.univ c
      hE : Subsingleton E
      ⊢ Function.Surjective f
    -/
  · haveI : Subsingleton F := (Equiv.subsingleton_congr f'.toEquiv).1 hE
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      c : NNReal
      inst✝ : CompleteSpace E
      hf : ApproximatesLinearOn f (↑f') Set.univ c
      hE : Subsingleton E
      this : Subsingleton F
      ⊢ Function.Surjective f
    -/
    exact surjective_to_subsingleton _
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      c : NNReal
      inst✝ : CompleteSpace E
      hf : ApproximatesLinearOn f (↑f') Set.univ c
      hc : LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm))
      ⊢ Function.Surjective f
    -/
  · apply forall_of_forall_mem_closedBall (fun y : F => ∃ a, f a = y) (f 0) _
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      c : NNReal
      inst✝ : CompleteSpace E
      hf : ApproximatesLinearOn f (↑f') Set.univ c
      hc : LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm))
      ⊢ Filter.Frequently (fun R => ∀ (y : F), Membership.mem (Metric.closedBall (f  …
    -/
    have hc' : (0 : ℝ) < N⁻¹ - c := by rw [sub_pos]; exact hc
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      c : NNReal
      inst✝ : CompleteSpace E
      hf : ApproximatesLinearOn f (↑f') Set.univ c
      hc : LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm))
      hc' : LT.lt 0 (HSub.hSub ↑(Inv.inv (NNNorm.nnnorm ↑f'.symm)) ↑c)
      ⊢ Filter.Frequently (fun R => ∀ (y : F), Membership.mem (Metric.closedBall (f  …
    -/
    let p : ℝ → Prop := fun R => closedBall (f 0) R ⊆ Set.range f
    have hp : ∀ᶠ r : ℝ in atTop, p ((N⁻¹ - c) * r) := by
      have hr : ∀ᶠ r : ℝ in atTop, 0 ≤ r := eventually_ge_atTop 0
      refine hr.mono fun r hr => Subset.trans ?_ (image_subset_range f (closedBall 0 r))
      refine hf.surjOn_closedBall_of_nonlinearRightInverse f'.toNonlinearRightInverse hr ?_
      exact subset_univ _
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      c : NNReal
      inst✝ : CompleteSpace E
      hf : ApproximatesLinearOn f (↑f') Set.univ c
      hc : LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm))
      hc' : LT.lt 0 (HSub.hSub ↑(Inv.inv (NNNorm.nnnorm ↑f'.symm)) ↑c)
      p : Real → Prop := fun R => HasSubset.Subset (Metric.closedBall (f 0) R) (Set. …
      hp : Filter.Eventually (fun r => p (HMul.hMul (HSub.hSub ↑(Inv.inv (NNNorm.nnn …
      ⊢ Filter.Frequently (fun R => ∀ (y : F), Membership.mem (Metric.closedBall (f  …
    -/
    refine ((tendsto_id.const_mul_atTop hc').frequently hp.frequently).mono ?_
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      c : NNReal
      inst✝ : CompleteSpace E
      hf : ApproximatesLinearOn f (↑f') Set.univ c
      hc : LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm))
      hc' : LT.lt 0 (HSub.hSub ↑(Inv.inv (NNNorm.nnnorm ↑f'.symm)) ↑c)
      p : Real → Prop := fun R => HasSubset.Subset (Metric.closedBall (f 0) R) (Set. …
      hp : Filter.Eventually (fun r => p (HMul.hMul (HSub.hSub ↑(Inv.inv (NNNorm.nnn …
      ⊢ ∀ (x : Real), p x → ∀ (y : F), Membership.mem (Metric.closedBall (f 0) x) y  …
    -/
    exact fun R h y hy => h hy
    /-
      🎉 no goals
    -/


/-- A map approximating a linear equivalence on a set defines a partial equivalence on this set.
Should not be used outside of this file, because it is superseded by `toPartialHomeomorph` below.

This is a first step towards the inverse function. -/
def toPartialEquiv (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) : PartialEquiv E F :=
  (hf.injOn hc).toPartialEquiv _ _


/-- The inverse function is continuous on `f '' s`.
Use properties of `PartialHomeomorph` instead. -/
theorem inverse_continuousOn (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) : ContinuousOn (hf.toPartialEquiv hc).symm (f '' s) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    ⊢ ContinuousOn (↑(hf.toPartialEquiv hc).symm) (Set.image f s)
  -/
  apply continuousOn_iff_continuous_restrict.2
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    ⊢ Continuous ((Set.image f s).restrict ↑(hf.toPartialEquiv hc).symm)
  -/
  refine ((hf.antilipschitz hc).to_rightInvOn' ?_ (hf.toPartialEquiv hc).right_inv').continuous
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    ⊢ Set.MapsTo (hf.toPartialEquiv hc).invFun (hf.toPartialEquiv hc).target s
  -/
  exact fun x hx => (hf.toPartialEquiv hc).map_target hx
  /-
    🎉 no goals
  -/


/-- The inverse function is approximated linearly on `f '' s` by `f'.symm`. -/
theorem to_inv (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c) (hc : Subsingleton E ∨ c < N⁻¹) :
    ApproximatesLinearOn (hf.toPartialEquiv hc).symm (f'.symm : F →L[𝕜] E) (f '' s)
      (N * (N⁻¹ - c)⁻¹ * c) := fun x hx y hy ↦ by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    x : F
    hx : Membership.mem (Set.image f s) x
    y : F
    hy : Membership.mem (Set.image f s) y
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (↑(hf.toPartialEquiv hc).symm x) (↑(h …
  -/
  set A := hf.toPartialEquiv hc
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    x : F
    hx : Membership.mem (Set.image f s) x
    y : F
    hy : Membership.mem (Set.image f s) y
    A : PartialEquiv E F := hf.toPartialEquiv hc
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (↑A.symm x) (↑A.symm y)) (↑f'.symm (H …
  -/
  have Af : ∀ z, A z = f z := fun z => rfl
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    x : F
    hx : Membership.mem (Set.image f s) x
    y : F
    hy : Membership.mem (Set.image f s) y
    A : PartialEquiv E F := hf.toPartialEquiv hc
    Af : ∀ (z : E), Eq (↑A z) (f z)
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (↑A.symm x) (↑A.symm y)) (↑f'.symm (H …
  -/
  rcases (mem_image _ _ _).1 hx with ⟨x', x's, rfl⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    y : F
    hy : Membership.mem (Set.image f s) y
    A : PartialEquiv E F := hf.toPartialEquiv hc
    Af : ∀ (z : E), Eq (↑A z) (f z)
    x' : E
    x's : Membership.mem s x'
    hx : Membership.mem (Set.image f s) (f x')
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (↑A.symm (f x')) (↑A.symm y)) (↑f'.sy …
  -/
  rcases (mem_image _ _ _).1 hy with ⟨y', y's, rfl⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    hf : ApproximatesLinearOn f (↑f') s c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    A : PartialEquiv E F := hf.toPartialEquiv hc
    Af : ∀ (z : E), Eq (↑A z) (f z)
    x' : E
    x's : Membership.mem s x'
    hx : Membership.mem (Set.image f s) (f x')
    y' : E
    y's : Membership.mem s y'
    hy : Membership.mem (Set.image f s) (f y')
    ⊢ LE.le (Norm.norm (HSub.hSub (HSub.hSub (↑A.symm (f x')) (↑A.symm (f y'))) (↑ …
  -/
  rw [← Af x', ← Af y', A.left_inv x's, A.left_inv y's]
  calc
    ‖x' - y' - f'.symm (A x' - A y')‖ ≤ N * ‖f' (x' - y' - f'.symm (A x' - A y'))‖ :=
      (f' : E →L[𝕜] F).bound_of_antilipschitz f'.antilipschitz _
    _ = N * ‖A y' - A x' - f' (y' - x')‖ := by
      congr 2
      simp only [ContinuousLinearEquiv.apply_symm_apply, ContinuousLinearEquiv.map_sub]
      abel
    _ ≤ N * (c * ‖y' - x'‖) := mul_le_mul_of_nonneg_left (hf _ y's _ x's) (NNReal.coe_nonneg _)
    _ ≤ N * (c * (((N⁻¹ - c)⁻¹ : ℝ≥0) * ‖A y' - A x'‖)) := by
      gcongr
      rw [← dist_eq_norm, ← dist_eq_norm]
      exact (hf.antilipschitz hc).le_mul_dist ⟨y', y's⟩ ⟨x', x's⟩
    _ = (N * (N⁻¹ - c)⁻¹ * c : ℝ≥0) * ‖A x' - A y'‖ := by
      simp only [norm_sub_rev, NNReal.coe_mul]; ring


/-- Given a function `f` that approximates a linear equivalence on an open set `s`,
returns a partial homeomorphism with `toFun = f` and `source = s`. -/
def toPartialHomeomorph (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) (hs : IsOpen s) : PartialHomeomorph E F where
  toPartialEquiv := hf.toPartialEquiv hc
  open_source := hs
  open_target := hf.open_image f'.toNonlinearRightInverse hs <| by
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      ε : Real
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      s : Set E
      c : NNReal
      inst✝ : CompleteSpace E
      hf : ApproximatesLinearOn f (↑f') s c
      hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
      hs : IsOpen s
      ⊢ Or (Subsingleton F) (LT.lt c (Inv.inv f'.toNonlinearRightInverse.nnnorm))
    -/
    rwa [f'.toEquiv.subsingleton_congr] at hc
    /-
      🎉 no goals
    -/
  continuousOn_toFun := hf.continuousOn
  continuousOn_invFun := hf.inverse_continuousOn hc


@[simp]
theorem toPartialHomeomorph_coe (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) (hs : IsOpen s) :
    (hf.toPartialHomeomorph f s hc hs : E → F) = f :=
  rfl


@[simp]
theorem toPartialHomeomorph_source (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) (hs : IsOpen s) :
    (hf.toPartialHomeomorph f s hc hs).source = s :=
  rfl


@[simp]
theorem toPartialHomeomorph_target (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) (hs : IsOpen s) :
    (hf.toPartialHomeomorph f s hc hs).target = f '' s :=
  rfl


/-- A function `f` that approximates a linear equivalence on the whole space is a homeomorphism. -/
def toHomeomorph (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) univ c)
    (hc : Subsingleton E ∨ c < N⁻¹) : E ≃ₜ F := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    ε : Real
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    inst✝ : CompleteSpace E
    hf : ApproximatesLinearOn f (↑f') Set.univ c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    ⊢ Homeomorph E F
  -/
  refine (hf.toPartialHomeomorph _ _ hc isOpen_univ).toHomeomorphOfSourceEqUnivTargetEqUniv rfl ?_
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    ε : Real
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    inst✝ : CompleteSpace E
    hf : ApproximatesLinearOn f (↑f') Set.univ c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    ⊢ Eq (ApproximatesLinearOn.toPartialHomeomorph f Set.univ hf hc ⋯).target Set. …
  -/
  rw [toPartialHomeomorph_target, image_univ, range_eq_univ]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    ε : Real
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    s : Set E
    c : NNReal
    inst✝ : CompleteSpace E
    hf : ApproximatesLinearOn f (↑f') Set.univ c
    hc : Or (Subsingleton E) (LT.lt c (Inv.inv (NNNorm.nnnorm ↑f'.symm)))
    ⊢ Function.Surjective f
  -/
  exact hf.surjective hc
  /-
    🎉 no goals
  -/


theorem closedBall_subset_target (hf : ApproximatesLinearOn f (f' : E →L[𝕜] F) s c)
    (hc : Subsingleton E ∨ c < N⁻¹) (hs : IsOpen s) {b : E} (ε0 : 0 ≤ ε) (hε : closedBall b ε ⊆ s) :
    closedBall (f b) ((N⁻¹ - c) * ε) ⊆ (hf.toPartialHomeomorph f s hc hs).target :=
  (hf.surjOn_closedBall_of_nonlinearRightInverse f'.toNonlinearRightInverse ε0 hε).mono hε
    Subset.rfl


