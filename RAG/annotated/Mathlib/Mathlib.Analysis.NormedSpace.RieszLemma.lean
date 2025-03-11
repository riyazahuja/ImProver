/-- Riesz's lemma, which usually states that it is possible to find a
vector with norm 1 whose distance to a closed proper subspace is
arbitrarily close to 1. The statement here is in terms of multiples of
norms, since in general the existence of an element of norm exactly 1
is not guaranteed. For a variant giving an element with norm in `[1, R]`, see
`riesz_lemma_of_norm_lt`. -/
theorem riesz_lemma {F : Subspace 𝕜 E} (hFc : IsClosed (F : Set E)) (hF : ∃ x : E, x ∉ F) {r : ℝ}
    (hr : r < 1) : ∃ x₀ : E, x₀ ∉ F ∧ ∀ y ∈ F, r * ‖x₀‖ ≤ ‖x₀ - y‖ := by
  classical
    obtain ⟨x, hx⟩ : ∃ x : E, x ∉ F := hF
    let d := Metric.infDist x F
    have hFn : (F : Set E).Nonempty := ⟨_, F.zero_mem⟩
    have hdp : 0 < d :=
      lt_of_le_of_ne Metric.infDist_nonneg fun heq =>
        hx ((hFc.mem_iff_infDist_zero hFn).2 heq.symm)
    let r' := max r 2⁻¹
    have hr' : r' < 1 := by
      simp only [r', max_lt_iff, hr, true_and]
      norm_num
    have hlt : 0 < r' := lt_of_lt_of_le (by norm_num) (le_max_right r 2⁻¹)
    have hdlt : d < d / r' := (lt_div_iff₀ hlt).mpr ((mul_lt_iff_lt_one_right hdp).2 hr')
    obtain ⟨y₀, hy₀F, hxy₀⟩ : ∃ y ∈ F, dist x y < d / r' := (Metric.infDist_lt_iff hFn).mp hdlt
    have x_ne_y₀ : x - y₀ ∉ F := by
      by_contra h
      have : x - y₀ + y₀ ∈ F := F.add_mem h hy₀F
      simp only [neg_add_cancel_right, sub_eq_add_neg] at this
      exact hx this
    refine ⟨x - y₀, x_ne_y₀, fun y hy => le_of_lt ?_⟩
    have hy₀y : y₀ + y ∈ F := F.add_mem hy₀F hy
    calc
      r * ‖x - y₀‖ ≤ r' * ‖x - y₀‖ := by gcongr; apply le_max_left
      _ < d := by
        rw [← dist_eq_norm]
        exact (lt_div_iff₀' hlt).1 hxy₀
      _ ≤ dist x (y₀ + y) := Metric.infDist_le_dist_of_mem hy₀y
      _ = ‖x - y₀ - y‖ := by rw [sub_sub, dist_eq_norm]


/--
A version of Riesz lemma: given a strict closed subspace `F`, one may find an element of norm `≤ R`
which is at distance at least `1` of every element of `F`. Here, `R` is any given constant
strictly larger than the norm of an element of norm `> 1`. For a version without an `R`, see
`riesz_lemma`.

Since we are considering a general nontrivially normed field, there may be a gap in possible norms
(for instance no element of norm in `(1,2)`). Hence, we can not allow `R` arbitrarily close to `1`,
and require `R > ‖c‖` for some `c : 𝕜` with norm `> 1`.
-/
theorem riesz_lemma_of_norm_lt {c : 𝕜} (hc : 1 < ‖c‖) {R : ℝ} (hR : ‖c‖ < R) {F : Subspace 𝕜 E}
    (hFc : IsClosed (F : Set E)) (hF : ∃ x : E, x ∉ F) :
    ∃ x₀ : E, ‖x₀‖ ≤ R ∧ ∀ y ∈ F, 1 ≤ ‖x₀ - y‖ := by
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    F : Subspace 𝕜 E
    hFc : IsClosed ↑F
    hF : Exists fun x => Not (Membership.mem F x)
    ⊢ Exists fun x₀ => And (LE.le (Norm.norm x₀) R) (∀ (y : E), Membership.mem F y …
  -/
  have Rpos : 0 < R := (norm_nonneg _).trans_lt hR
  have : ‖c‖ / R < 1 := by
    rw [div_lt_iff₀ Rpos]
    simpa using hR
  /-
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    F : Subspace 𝕜 E
    hFc : IsClosed ↑F
    hF : Exists fun x => Not (Membership.mem F x)
    Rpos : LT.lt 0 R
    this : LT.lt (HDiv.hDiv (Norm.norm c) R) 1
    ⊢ Exists fun x₀ => And (LE.le (Norm.norm x₀) R) (∀ (y : E), Membership.mem F y …
  -/
  rcases riesz_lemma hFc hF this with ⟨x, xF, hx⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    F : Subspace 𝕜 E
    hFc : IsClosed ↑F
    hF : Exists fun x => Not (Membership.mem F x)
    Rpos : LT.lt 0 R
    this : LT.lt (HDiv.hDiv (Norm.norm c) R) 1
    x : E
    xF : Not (Membership.mem F x)
    hx : ∀ (y : E), Membership.mem F y → LE.le (HMul.hMul (HDiv.hDiv (Norm.norm c) …
    ⊢ Exists fun x₀ => And (LE.le (Norm.norm x₀) R) (∀ (y : E), Membership.mem F y …
  -/
  have x0 : x ≠ 0 := fun H => by simp [H] at xF
  obtain ⟨d, d0, dxlt, ledx, -⟩ :
    ∃ d : 𝕜, d ≠ 0 ∧ ‖d • x‖ < R ∧ R / ‖c‖ ≤ ‖d • x‖ ∧ ‖d‖⁻¹ ≤ R⁻¹ * ‖c‖ * ‖x‖ :=
    rescale_to_shell hc Rpos x0
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    F : Subspace 𝕜 E
    hFc : IsClosed ↑F
    hF : Exists fun x => Not (Membership.mem F x)
    Rpos : LT.lt 0 R
    this : LT.lt (HDiv.hDiv (Norm.norm c) R) 1
    x : E
    xF : Not (Membership.mem F x)
    hx : ∀ (y : E), Membership.mem F y → LE.le (HMul.hMul (HDiv.hDiv (Norm.norm c) …
    x0 : Ne x 0
    d : 𝕜
    d0 : Ne d 0
    dxlt : LT.lt (Norm.norm (HSMul.hSMul d x)) R
    ledx : LE.le (HDiv.hDiv R (Norm.norm c)) (Norm.norm (HSMul.hSMul d x))
    ⊢ Exists fun x₀ => And (LE.le (Norm.norm x₀) R) (∀ (y : E), Membership.mem F y …
  -/
  refine ⟨d • x, dxlt.le, fun y hy => ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    F : Subspace 𝕜 E
    hFc : IsClosed ↑F
    hF : Exists fun x => Not (Membership.mem F x)
    Rpos : LT.lt 0 R
    this : LT.lt (HDiv.hDiv (Norm.norm c) R) 1
    x : E
    xF : Not (Membership.mem F x)
    hx : ∀ (y : E), Membership.mem F y → LE.le (HMul.hMul (HDiv.hDiv (Norm.norm c) …
    x0 : Ne x 0
    d : 𝕜
    d0 : Ne d 0
    dxlt : LT.lt (Norm.norm (HSMul.hSMul d x)) R
    ledx : LE.le (HDiv.hDiv R (Norm.norm c)) (Norm.norm (HSMul.hSMul d x))
    y : E
    hy : Membership.mem F y
    ⊢ LE.le 1 (Norm.norm (HSub.hSub (HSMul.hSMul d x) y))
  -/
  set y' := d⁻¹ • y
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝² : NormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    F : Subspace 𝕜 E
    hFc : IsClosed ↑F
    hF : Exists fun x => Not (Membership.mem F x)
    Rpos : LT.lt 0 R
    this : LT.lt (HDiv.hDiv (Norm.norm c) R) 1
    x : E
    xF : Not (Membership.mem F x)
    hx : ∀ (y : E), Membership.mem F y → LE.le (HMul.hMul (HDiv.hDiv (Norm.norm c) …
    x0 : Ne x 0
    d : 𝕜
    d0 : Ne d 0
    dxlt : LT.lt (Norm.norm (HSMul.hSMul d x)) R
    ledx : LE.le (HDiv.hDiv R (Norm.norm c)) (Norm.norm (HSMul.hSMul d x))
    y : E
    hy : Membership.mem F y
    y' : E := HSMul.hSMul (Inv.inv d) y
    ⊢ LE.le 1 (Norm.norm (HSub.hSub (HSMul.hSMul d x) y))
  -/
  have yy' : y = d • y' := by simp [y', smul_smul, mul_inv_cancel₀ d0]
  calc
    1 = ‖c‖ / R * (R / ‖c‖) := by field_simp [Rpos.ne', (zero_lt_one.trans hc).ne']
    _ ≤ ‖c‖ / R * ‖d • x‖ := by gcongr
    _ = ‖d‖ * (‖c‖ / R * ‖x‖) := by
      simp only [norm_smul]
      ring
    _ ≤ ‖d‖ * ‖x - y'‖ := by gcongr; exact hx y' (by simp [y', Submodule.smul_mem _ _ hy])
    _ = ‖d • x - y‖ := by rw [yy', ← smul_sub, norm_smul]


theorem Metric.closedBall_infDist_compl_subset_closure {x : F} {s : Set F} (hx : x ∈ s) :
    closedBall x (infDist x sᶜ) ⊆ closure s := by
  /-
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x : F
    s : Set F
    hx : Membership.mem s x
    ⊢ HasSubset.Subset (Metric.closedBall x (Metric.infDist x (HasCompl.compl s))) …
  -/
  rcases eq_or_ne (infDist x sᶜ) 0 with h₀ | h₀
    /-
      case inl
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace Real F
      x : F
      s : Set F
      hx : Membership.mem s x
      h₀ : Eq (Metric.infDist x (HasCompl.compl s)) 0
      ⊢ HasSubset.Subset (Metric.closedBall x (Metric.infDist x (HasCompl.compl s))) …
    -/
  · rw [h₀, closedBall_zero']
    /-
      case inl
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace Real F
      x : F
      s : Set F
      hx : Membership.mem s x
      h₀ : Eq (Metric.infDist x (HasCompl.compl s)) 0
      ⊢ HasSubset.Subset (closure (Singleton.singleton x)) (closure s)
    -/
    exact closure_mono (singleton_subset_iff.2 hx)
    /-
      🎉 no goals
    -/
    /-
      case inr
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace Real F
      x : F
      s : Set F
      hx : Membership.mem s x
      h₀ : Ne (Metric.infDist x (HasCompl.compl s)) 0
      ⊢ HasSubset.Subset (Metric.closedBall x (Metric.infDist x (HasCompl.compl s))) …
    -/
  · rw [← closure_ball x h₀]
    /-
      case inr
      F : Type u_3
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace Real F
      x : F
      s : Set F
      hx : Membership.mem s x
      h₀ : Ne (Metric.infDist x (HasCompl.compl s)) 0
      ⊢ HasSubset.Subset (closure (Metric.ball x (Metric.infDist x (HasCompl.compl s …
    -/
    exact closure_mono ball_infDist_compl_subset
    /-
      🎉 no goals
    -/

