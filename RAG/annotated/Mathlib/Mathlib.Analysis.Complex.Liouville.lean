local postfix:100 "̂" => UniformSpace.Completion


/-- If `f` is complex differentiable on an open disc with center `c` and radius `R > 0` and is
continuous on its closure, then `f' c` can be represented as an integral over the corresponding
circle.

TODO: add a version for `w ∈ Metric.ball c R`.

TODO: add a version for higher derivatives. -/
theorem deriv_eq_smul_circleIntegral [CompleteSpace F] {R : ℝ} {c : ℂ} {f : ℂ → F} (hR : 0 < R)
    (hf : DiffContOnCl ℂ f (ball c R)) :
    deriv f c = (2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - c) ^ (-2 : ℤ) • f z := by
  /-
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    R : Real
    c : Complex
    f : Complex → F
    hR : LT.lt 0 R
    hf : DiffContOnCl Complex f (Metric.ball c R)
    ⊢ Eq (deriv f c) (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Compl …
  -/
  lift R to ℝ≥0 using hR.le
  /-
    case intro
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    c : Complex
    f : Complex → F
    R : NNReal
    hR : LT.lt 0 ↑R
    hf : DiffContOnCl Complex f (Metric.ball c ↑R)
    ⊢ Eq (deriv f c) (HSMul.hSMul (Inv.inv (HMul.hMul (HMul.hMul 2 ↑Real.pi) Compl …
  -/
  refine (hf.hasFPowerSeriesOnBall hR).hasFPowerSeriesAt.deriv.trans ?_
  /-
    case intro
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : CompleteSpace F
    c : Complex
    f : Complex → F
    R : NNReal
    hR : LT.lt 0 ↑R
    hf : DiffContOnCl Complex f (Metric.ball c ↑R)
    ⊢ Eq ((cauchyPowerSeries f c (↑R) 1) fun x => 1) (HSMul.hSMul (Inv.inv (HMul.h …
  -/
  simp only [cauchyPowerSeries_apply, one_div, zpow_neg, pow_one, smul_smul, zpow_two, mul_inv]
  /-
    🎉 no goals
  -/


theorem norm_deriv_le_aux [CompleteSpace F] {c : ℂ} {R C : ℝ} {f : ℂ → F} (hR : 0 < R)
    (hf : DiffContOnCl ℂ f (ball c R)) (hC : ∀ z ∈ sphere c R, ‖f z‖ ≤ C) :
    ‖deriv f c‖ ≤ C / R := by
  have : ∀ z ∈ sphere c R, ‖(z - c) ^ (-2 : ℤ) • f z‖ ≤ C / (R * R) :=
    fun z (hz : abs (z - c) = R) => by
    simpa [-mul_inv_rev, norm_smul, hz, zpow_two, ← div_eq_inv_mul] using
      (div_le_div_iff_of_pos_right (mul_pos hR hR)).2 (hC z hz)
  calc
    ‖deriv f c‖ = ‖(2 * π * I : ℂ)⁻¹ • ∮ z in C(c, R), (z - c) ^ (-2 : ℤ) • f z‖ :=
      congr_arg norm (deriv_eq_smul_circleIntegral hR hf)
    _ ≤ R * (C / (R * R)) :=
      (circleIntegral.norm_two_pi_i_inv_smul_integral_le_of_norm_le_const hR.le this)
    _ = C / R := by rw [mul_div_left_comm, div_self_mul_self', div_eq_mul_inv]


/-- If `f` is complex differentiable on an open disc of radius `R > 0`, is continuous on its
closure, and its values on the boundary circle of this disc are bounded from above by `C`, then the
norm of its derivative at the center is at most `C / R`. -/
theorem norm_deriv_le_of_forall_mem_sphere_norm_le {c : ℂ} {R C : ℝ} {f : ℂ → F} (hR : 0 < R)
    (hd : DiffContOnCl ℂ f (ball c R)) (hC : ∀ z ∈ sphere c R, ‖f z‖ ≤ C) :
    ‖deriv f c‖ ≤ C / R := by
  /-
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    c : Complex
    R C : Real
    f : Complex → F
    hR : LT.lt 0 R
    hd : DiffContOnCl Complex f (Metric.ball c R)
    hC : ∀ (z : Complex), Membership.mem (Metric.sphere c R) z → LE.le (Norm.norm  …
    ⊢ LE.le (Norm.norm (deriv f c)) (HDiv.hDiv C R)
  -/
  set e : F →L[ℂ] F̂ := UniformSpace.Completion.toComplL
  have : HasDerivAt (e ∘ f) (e (deriv f c)) c :=
    e.hasFDerivAt.comp_hasDerivAt c
      (hd.differentiableAt isOpen_ball <| mem_ball_self hR).hasDerivAt
  calc
    ‖deriv f c‖ = ‖deriv (e ∘ f) c‖ := by
      rw [this.deriv]
      exact (UniformSpace.Completion.norm_coe _).symm
    _ ≤ C / R :=
      norm_deriv_le_aux hR (e.differentiable.comp_diffContOnCl hd) fun z hz =>
        (UniformSpace.Completion.norm_coe _).trans_le (hC z hz)


/-- An auxiliary lemma for Liouville's theorem `Differentiable.apply_eq_apply_of_bounded`. -/
theorem liouville_theorem_aux {f : ℂ → F} (hf : Differentiable ℂ f) (hb : IsBounded (range f))
    (z w : ℂ) : f z = f w := by
  /-
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    hf : Differentiable Complex f
    hb : Bornology.IsBounded (Set.range f)
    z w : Complex
    ⊢ Eq (f z) (f w)
  -/
  suffices ∀ c, deriv f c = 0 from is_const_of_deriv_eq_zero hf this z w
  /-
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    hf : Differentiable Complex f
    hb : Bornology.IsBounded (Set.range f)
    z w : Complex
    ⊢ ∀ (c : Complex), Eq (deriv f c) 0
  -/
  clear z w; intro c
  obtain ⟨C, C₀, hC⟩ : ∃ C > (0 : ℝ), ∀ z, ‖f z‖ ≤ C := by
    rcases isBounded_iff_forall_norm_le.1 hb with ⟨C, hC⟩
    exact
      ⟨max C 1, lt_max_iff.2 (Or.inr zero_lt_one), fun z =>
        (hC (f z) (mem_range_self _)).trans (le_max_left _ _)⟩
  /-
    case intro.intro
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : Complex → F
    hf : Differentiable Complex f
    hb : Bornology.IsBounded (Set.range f)
    c : Complex
    C : Real
    C₀ : GT.gt C 0
    hC : ∀ (z : Complex), LE.le (Norm.norm (f z)) C
    ⊢ Eq (deriv f c) 0
  -/
  refine norm_le_zero_iff.1 (le_of_forall_le_of_dense fun ε ε₀ => ?_)
  calc
    ‖deriv f c‖ ≤ C / (C / ε) :=
      norm_deriv_le_of_forall_mem_sphere_norm_le (div_pos C₀ ε₀) hf.diffContOnCl fun z _ => hC z
    _ = ε := div_div_cancel₀ C₀.lt.ne'


/-- **Liouville's theorem**: a complex differentiable bounded function `f : E → F` is a constant. -/
theorem apply_eq_apply_of_bounded {f : E → F} (hf : Differentiable ℂ f) (hb : IsBounded (range f))
    (z w : E) : f z = f w := by
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    hf : Differentiable Complex f
    hb : Bornology.IsBounded (Set.range f)
    z w : E
    ⊢ Eq (f z) (f w)
  -/
  set g : ℂ → F := f ∘ fun t : ℂ => t • (w - z) + z
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    hf : Differentiable Complex f
    hb : Bornology.IsBounded (Set.range f)
    z w : E
    g : Complex → F := Function.comp f fun t => HAdd.hAdd (HSMul.hSMul t (HSub.hSu …
    ⊢ Eq (f z) (f w)
  -/
  suffices g 0 = g 1 by simpa [g]
  /-
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Complex F
    f : E → F
    hf : Differentiable Complex f
    hb : Bornology.IsBounded (Set.range f)
    z w : E
    g : Complex → F := Function.comp f fun t => HAdd.hAdd (HSMul.hSMul t (HSub.hSu …
    ⊢ Eq (g 0) (g 1)
  -/
  apply liouville_theorem_aux
  exacts [hf.comp ((differentiable_id.smul_const (w - z)).add_const z),
    hb.subset (range_comp_subset_range _ _)]


/-- **Liouville's theorem**: a complex differentiable bounded function is a constant. -/
theorem exists_const_forall_eq_of_bounded {f : E → F} (hf : Differentiable ℂ f)
    (hb : IsBounded (range f)) : ∃ c, ∀ z, f z = c :=
  ⟨f 0, fun _ => hf.apply_eq_apply_of_bounded hb _ _⟩


/-- **Liouville's theorem**: a complex differentiable bounded function is a constant. -/
theorem exists_eq_const_of_bounded {f : E → F} (hf : Differentiable ℂ f)
    (hb : IsBounded (range f)) : ∃ c, f = const E c :=
  (hf.exists_const_forall_eq_of_bounded hb).imp fun _ => funext


/-- A corollary of Liouville's theorem where the function tends to a finite value at infinity
(i.e., along `Filter.cocompact`, which in proper spaces coincides with `Bornology.cobounded`). -/
theorem eq_const_of_tendsto_cocompact [Nontrivial E] {f : E → F} (hf : Differentiable ℂ f) {c : F}
    (hb : Tendsto f (cocompact E) (𝓝 c)) : f = Function.const E c := by
  have h_bdd : Bornology.IsBounded (Set.range f) := by
    obtain ⟨s, hs, hs_bdd⟩ := Metric.exists_isBounded_image_of_tendsto hb
    obtain ⟨t, ht, hts⟩ := mem_cocompact.mp hs
    apply ht.image hf.continuous |>.isBounded.union hs_bdd |>.subset
    simpa [Set.image_union, Set.image_univ] using Set.image_subset _ <| calc
      Set.univ = t ∪ tᶜ := t.union_compl_self.symm
      _        ⊆ t ∪ s  := by gcongr
  /-
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    hf : Differentiable Complex f
    c : F
    hb : Filter.Tendsto f (Filter.cocompact E) (nhds c)
    h_bdd : Bornology.IsBounded (Set.range f)
    ⊢ Eq f (Function.const E c)
  -/
  obtain ⟨c', hc'⟩ := hf.exists_eq_const_of_bounded h_bdd
  /-
    case intro
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    hf : Differentiable Complex f
    c : F
    hb : Filter.Tendsto f (Filter.cocompact E) (nhds c)
    h_bdd : Bornology.IsBounded (Set.range f)
    c' : F
    hc' : Eq f (Function.const E c')
    ⊢ Eq f (Function.const E c)
  -/
  convert hc'
  /-
    case h.e'_3.h.e'_3
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Complex E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace Complex F
    inst✝ : Nontrivial E
    f : E → F
    hf : Differentiable Complex f
    c : F
    hb : Filter.Tendsto f (Filter.cocompact E) (nhds c)
    h_bdd : Bornology.IsBounded (Set.range f)
    c' : F
    hc' : Eq f (Function.const E c')
    ⊢ Eq c c'
  -/
  exact tendsto_nhds_unique hb (by simpa [hc'] using tendsto_const_nhds)
  /-
    🎉 no goals
  -/


/-- A corollary of Liouville's theorem where the function tends to a finite value at infinity
(i.e., along `Filter.cocompact`, which in proper spaces coincides with `Bornology.cobounded`). -/
theorem apply_eq_of_tendsto_cocompact [Nontrivial E] {f : E → F} (hf : Differentiable ℂ f) {c : F}
    (x : E) (hb : Tendsto f (cocompact E) (𝓝 c)) : f x = c :=
  congr($(hf.eq_const_of_tendsto_cocompact hb) x)


