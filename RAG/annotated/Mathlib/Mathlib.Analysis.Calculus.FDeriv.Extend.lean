/-- If a function `f` is differentiable in a convex open set and continuous on its closure, and its
derivative converges to a limit `f'` at a point on the boundary, then `f` is differentiable there
with derivative `f'`. -/
theorem hasFDerivWithinAt_closure_of_tendsto_fderiv {f : E → F} {s : Set E} {x : E} {f' : E →L[ℝ] F}
    (f_diff : DifferentiableOn ℝ f s) (s_conv : Convex ℝ s) (s_open : IsOpen s)
    (f_cont : ∀ y ∈ closure s, ContinuousWithinAt f s y)
    (h : Tendsto (fun y => fderiv ℝ f y) (𝓝[s] x) (𝓝 f')) :
    HasFDerivWithinAt f f' (closure s) x := by
  classical
    -- one can assume without loss of generality that `x` belongs to the closure of `s`, as the
    -- statement is empty otherwise
    by_cases hx : x ∉ closure s
    · rw [← closure_closure] at hx; exact hasFDerivWithinAt_of_nmem_closure hx
    push_neg at hx
    rw [HasFDerivWithinAt, hasFDerivAtFilter_iff_isLittleO, Asymptotics.isLittleO_iff]
    /- One needs to show that `‖f y - f x - f' (y - x)‖ ≤ ε ‖y - x‖` for `y` close to `x` in
      `closure s`, where `ε` is an arbitrary positive constant. By continuity of the functions, it
      suffices to prove this for nearby points inside `s`. In a neighborhood of `x`, the derivative
      of `f` is arbitrarily close to `f'` by assumption. The mean value inequality completes the
      proof. -/
    intro ε ε_pos
    obtain ⟨δ, δ_pos, hδ⟩ : ∃ δ > 0, ∀ y ∈ s, dist y x < δ → ‖fderiv ℝ f y - f'‖ < ε := by
      simpa [dist_zero_right] using tendsto_nhdsWithin_nhds.1 h ε ε_pos
    set B := ball x δ
    suffices ∀ y ∈ B ∩ closure s, ‖f y - f x - (f' y - f' x)‖ ≤ ε * ‖y - x‖ from
      mem_nhdsWithin_iff.2 ⟨δ, δ_pos, fun y hy => by simpa using this y hy⟩
    suffices
      ∀ p : E × E,
        p ∈ closure ((B ∩ s) ×ˢ (B ∩ s)) → ‖f p.2 - f p.1 - (f' p.2 - f' p.1)‖ ≤ ε * ‖p.2 - p.1‖ by
      rw [closure_prod_eq] at this
      intro y y_in
      apply this ⟨x, y⟩
      have : B ∩ closure s ⊆ closure (B ∩ s) := isOpen_ball.inter_closure
      exact ⟨this ⟨mem_ball_self δ_pos, hx⟩, this y_in⟩
    have key : ∀ p : E × E, p ∈ (B ∩ s) ×ˢ (B ∩ s) →
          ‖f p.2 - f p.1 - (f' p.2 - f' p.1)‖ ≤ ε * ‖p.2 - p.1‖ := by
      rintro ⟨u, v⟩ ⟨u_in, v_in⟩
      have conv : Convex ℝ (B ∩ s) := (convex_ball _ _).inter s_conv
      have diff : DifferentiableOn ℝ f (B ∩ s) := f_diff.mono inter_subset_right
      have bound : ∀ z ∈ B ∩ s, ‖fderivWithin ℝ f (B ∩ s) z - f'‖ ≤ ε := by
        intro z z_in
        have h := hδ z
        have : fderivWithin ℝ f (B ∩ s) z = fderiv ℝ f z := by
          have op : IsOpen (B ∩ s) := isOpen_ball.inter s_open
          rw [DifferentiableAt.fderivWithin _ (op.uniqueDiffOn z z_in)]
          exact (diff z z_in).differentiableAt (IsOpen.mem_nhds op z_in)
        rw [← this] at h
        exact le_of_lt (h z_in.2 z_in.1)
      simpa using conv.norm_image_sub_le_of_norm_fderivWithin_le' diff bound u_in v_in
    rintro ⟨u, v⟩ uv_in
    have f_cont' : ∀ y ∈ closure s, ContinuousWithinAt (f -  ⇑f') s y := by
      intro y y_in
      exact Tendsto.sub (f_cont y y_in) f'.cont.continuousWithinAt
    refine ContinuousWithinAt.closure_le uv_in ?_ ?_ key
    all_goals
      -- common start for both continuity proofs
      have : (B ∩ s) ×ˢ (B ∩ s) ⊆ s ×ˢ s := by gcongr <;> exact inter_subset_right
      obtain ⟨u_in, v_in⟩ : u ∈ closure s ∧ v ∈ closure s := by
        simpa [closure_prod_eq] using closure_mono this uv_in
      apply ContinuousWithinAt.mono _ this
      simp only [ContinuousWithinAt]
    · rw [nhdsWithin_prod_eq]
      have : ∀ u v, f v - f u - (f' v - f' u) = f v - f' v - (f u - f' u) := by intros; abel
      simp only [this]
      exact
        Tendsto.comp continuous_norm.continuousAt
          ((Tendsto.comp (f_cont' v v_in) tendsto_snd).sub <|
            Tendsto.comp (f_cont' u u_in) tendsto_fst)
    · apply tendsto_nhdsWithin_of_tendsto_nhds
      rw [nhds_prod_eq]
      exact
        tendsto_const_nhds.mul
          (Tendsto.comp continuous_norm.continuousAt <| tendsto_snd.sub tendsto_fst)


@[deprecated (since := "2024-07-10")] alias has_fderiv_at_boundary_of_tendsto_fderiv :=
  hasFDerivWithinAt_closure_of_tendsto_fderiv


/-- If a function is differentiable on the right of a point `a : ℝ`, continuous at `a`, and
its derivative also converges at `a`, then `f` is differentiable on the right at `a`. -/
theorem hasDerivWithinAt_Ici_of_tendsto_deriv {s : Set ℝ} {e : E} {a : ℝ} {f : ℝ → E}
    (f_diff : DifferentiableOn ℝ f s) (f_lim : ContinuousWithinAt f s a) (hs : s ∈ 𝓝[>] a)
    (f_lim' : Tendsto (fun x => deriv f x) (𝓝[>] a) (𝓝 e)) : HasDerivWithinAt f e (Ici a) a := by
  /- This is a specialization of `hasFDerivWithinAt_closure_of_tendsto_fderiv`. To be in the
    setting of this theorem, we need to work on an open interval with closure contained in
    `s ∪ {a}`, that we call `t = (a, b)`. Then, we check all the assumptions of this theorem and
    we apply it. -/
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Ioi a)) (nhds e)
    ⊢ HasDerivWithinAt f e (Set.Ici a) a
  -/
  obtain ⟨b, ab : a < b, sab : Ioc a b ⊆ s⟩ := mem_nhdsGT_iff_exists_Ioc_subset.1 hs
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Ioi a)) (nhds e)
    b : Real
    ab : LT.lt a b
    sab : HasSubset.Subset (Set.Ioc a b) s
    ⊢ HasDerivWithinAt f e (Set.Ici a) a
  -/
  let t := Ioo a b
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Ioi a)) (nhds e)
    b : Real
    ab : LT.lt a b
    sab : HasSubset.Subset (Set.Ioc a b) s
    t : Set Real := Set.Ioo a b
    ⊢ HasDerivWithinAt f e (Set.Ici a) a
  -/
  have ts : t ⊆ s := Subset.trans Ioo_subset_Ioc_self sab
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Ioi a)) (nhds e)
    b : Real
    ab : LT.lt a b
    sab : HasSubset.Subset (Set.Ioc a b) s
    t : Set Real := Set.Ioo a b
    ts : HasSubset.Subset t s
    ⊢ HasDerivWithinAt f e (Set.Ici a) a
  -/
  have t_diff : DifferentiableOn ℝ f t := f_diff.mono ts
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Ioi a)) (nhds e)
    b : Real
    ab : LT.lt a b
    sab : HasSubset.Subset (Set.Ioc a b) s
    t : Set Real := Set.Ioo a b
    ts : HasSubset.Subset t s
    t_diff : DifferentiableOn Real f t
    ⊢ HasDerivWithinAt f e (Set.Ici a) a
  -/
  have t_conv : Convex ℝ t := convex_Ioo a b
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Ioi a)) (nhds e)
    b : Real
    ab : LT.lt a b
    sab : HasSubset.Subset (Set.Ioc a b) s
    t : Set Real := Set.Ioo a b
    ts : HasSubset.Subset t s
    t_diff : DifferentiableOn Real f t
    t_conv : Convex Real t
    ⊢ HasDerivWithinAt f e (Set.Ici a) a
  -/
  have t_open : IsOpen t := isOpen_Ioo
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Ioi a)) (nhds e)
    b : Real
    ab : LT.lt a b
    sab : HasSubset.Subset (Set.Ioc a b) s
    t : Set Real := Set.Ioo a b
    ts : HasSubset.Subset t s
    t_diff : DifferentiableOn Real f t
    t_conv : Convex Real t
    t_open : IsOpen t
    ⊢ HasDerivWithinAt f e (Set.Ici a) a
  -/
  have t_closure : closure t = Icc a b := closure_Ioo ab.ne
  have t_cont : ∀ y ∈ closure t, ContinuousWithinAt f t y := by
    rw [t_closure]
    intro y hy
    by_cases h : y = a
    · rw [h]; exact f_lim.mono ts
    · have : y ∈ s := sab ⟨lt_of_le_of_ne hy.1 (Ne.symm h), hy.2⟩
      exact (f_diff.continuousOn y this).mono ts
  have t_diff' : Tendsto (fun x => fderiv ℝ f x) (𝓝[t] a) (𝓝 (smulRight (1 : ℝ →L[ℝ] ℝ) e)) := by
    simp only [deriv_fderiv.symm]
    exact Tendsto.comp
      (isBoundedBilinearMap_smulRight : IsBoundedBilinearMap ℝ _).continuous_right.continuousAt
      (tendsto_nhdsWithin_mono_left Ioo_subset_Ioi_self f_lim')
  -- now we can apply `hasFDerivWithinAt_closure_of_tendsto_fderiv`
  have : HasDerivWithinAt f e (Icc a b) a := by
    rw [hasDerivWithinAt_iff_hasFDerivWithinAt, ← t_closure]
    exact hasFDerivWithinAt_closure_of_tendsto_fderiv t_diff t_conv t_open t_cont t_diff'
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Ioi a)) (nhds e)
    b : Real
    ab : LT.lt a b
    sab : HasSubset.Subset (Set.Ioc a b) s
    t : Set Real := Set.Ioo a b
    ts : HasSubset.Subset t s
    t_diff : DifferentiableOn Real f t
    t_conv : Convex Real t
    t_open : IsOpen t
    t_closure : Eq (closure t) (Set.Icc a b)
    t_cont : ∀ (y : Real), Membership.mem (closure t) y → ContinuousWithinAt f t y
    t_diff' : Filter.Tendsto (fun x => fderiv Real f x) (nhdsWithin a t) (nhds (Co …
    this : HasDerivWithinAt f e (Set.Icc a b) a
    ⊢ HasDerivWithinAt f e (Set.Ici a) a
  -/
  exact this.mono_of_mem_nhdsWithin (Icc_mem_nhdsGE ab)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-10")] alias has_deriv_at_interval_left_endpoint_of_tendsto_deriv :=
  hasDerivWithinAt_Ici_of_tendsto_deriv


/-- If a function is differentiable on the left of a point `a : ℝ`, continuous at `a`, and
its derivative also converges at `a`, then `f` is differentiable on the left at `a`. -/
theorem hasDerivWithinAt_Iic_of_tendsto_deriv {s : Set ℝ} {e : E} {a : ℝ}
    {f : ℝ → E} (f_diff : DifferentiableOn ℝ f s) (f_lim : ContinuousWithinAt f s a)
    (hs : s ∈ 𝓝[<] a) (f_lim' : Tendsto (fun x => deriv f x) (𝓝[<] a) (𝓝 e)) :
    HasDerivWithinAt f e (Iic a) a := by
  /- This is a specialization of `hasFDerivWithinAt_closure_of_tendsto_fderiv`. To be in the
    setting of this theorem, we need to work on an open interval with closure contained in
    `s ∪ {a}`, that we call `t = (b, a)`. Then, we check all the assumptions of this theorem and we
    apply it. -/
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Iio a)) (nhds e)
    ⊢ HasDerivWithinAt f e (Set.Iic a) a
  -/
  obtain ⟨b, ba, sab⟩ : ∃ b ∈ Iio a, Ico b a ⊆ s := mem_nhdsLT_iff_exists_Ico_subset.1 hs
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Iio a)) (nhds e)
    b : Real
    ba : Membership.mem (Set.Iio a) b
    sab : HasSubset.Subset (Set.Ico b a) s
    ⊢ HasDerivWithinAt f e (Set.Iic a) a
  -/
  let t := Ioo b a
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Iio a)) (nhds e)
    b : Real
    ba : Membership.mem (Set.Iio a) b
    sab : HasSubset.Subset (Set.Ico b a) s
    t : Set Real := Set.Ioo b a
    ⊢ HasDerivWithinAt f e (Set.Iic a) a
  -/
  have ts : t ⊆ s := Subset.trans Ioo_subset_Ico_self sab
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Iio a)) (nhds e)
    b : Real
    ba : Membership.mem (Set.Iio a) b
    sab : HasSubset.Subset (Set.Ico b a) s
    t : Set Real := Set.Ioo b a
    ts : HasSubset.Subset t s
    ⊢ HasDerivWithinAt f e (Set.Iic a) a
  -/
  have t_diff : DifferentiableOn ℝ f t := f_diff.mono ts
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Iio a)) (nhds e)
    b : Real
    ba : Membership.mem (Set.Iio a) b
    sab : HasSubset.Subset (Set.Ico b a) s
    t : Set Real := Set.Ioo b a
    ts : HasSubset.Subset t s
    t_diff : DifferentiableOn Real f t
    ⊢ HasDerivWithinAt f e (Set.Iic a) a
  -/
  have t_conv : Convex ℝ t := convex_Ioo b a
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Iio a)) (nhds e)
    b : Real
    ba : Membership.mem (Set.Iio a) b
    sab : HasSubset.Subset (Set.Ico b a) s
    t : Set Real := Set.Ioo b a
    ts : HasSubset.Subset t s
    t_diff : DifferentiableOn Real f t
    t_conv : Convex Real t
    ⊢ HasDerivWithinAt f e (Set.Iic a) a
  -/
  have t_open : IsOpen t := isOpen_Ioo
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Iio a)) (nhds e)
    b : Real
    ba : Membership.mem (Set.Iio a) b
    sab : HasSubset.Subset (Set.Ico b a) s
    t : Set Real := Set.Ioo b a
    ts : HasSubset.Subset t s
    t_diff : DifferentiableOn Real f t
    t_conv : Convex Real t
    t_open : IsOpen t
    ⊢ HasDerivWithinAt f e (Set.Iic a) a
  -/
  have t_closure : closure t = Icc b a := closure_Ioo (ne_of_lt ba)
  have t_cont : ∀ y ∈ closure t, ContinuousWithinAt f t y := by
    rw [t_closure]
    intro y hy
    by_cases h : y = a
    · rw [h]; exact f_lim.mono ts
    · have : y ∈ s := sab ⟨hy.1, lt_of_le_of_ne hy.2 h⟩
      exact (f_diff.continuousOn y this).mono ts
  have t_diff' : Tendsto (fun x => fderiv ℝ f x) (𝓝[t] a) (𝓝 (smulRight (1 : ℝ →L[ℝ] ℝ) e)) := by
    simp only [deriv_fderiv.symm]
    exact Tendsto.comp
      (isBoundedBilinearMap_smulRight : IsBoundedBilinearMap ℝ _).continuous_right.continuousAt
      (tendsto_nhdsWithin_mono_left Ioo_subset_Iio_self f_lim')
  -- now we can apply `hasFDerivWithinAt_closure_of_tendsto_fderiv`
  have : HasDerivWithinAt f e (Icc b a) a := by
    rw [hasDerivWithinAt_iff_hasFDerivWithinAt, ← t_closure]
    exact hasFDerivWithinAt_closure_of_tendsto_fderiv t_diff t_conv t_open t_cont t_diff'
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set Real
    e : E
    a : Real
    f : Real → E
    f_diff : DifferentiableOn Real f s
    f_lim : ContinuousWithinAt f s a
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    f_lim' : Filter.Tendsto (fun x => deriv f x) (nhdsWithin a (Set.Iio a)) (nhds e)
    b : Real
    ba : Membership.mem (Set.Iio a) b
    sab : HasSubset.Subset (Set.Ico b a) s
    t : Set Real := Set.Ioo b a
    ts : HasSubset.Subset t s
    t_diff : DifferentiableOn Real f t
    t_conv : Convex Real t
    t_open : IsOpen t
    t_closure : Eq (closure t) (Set.Icc b a)
    t_cont : ∀ (y : Real), Membership.mem (closure t) y → ContinuousWithinAt f t y
    t_diff' : Filter.Tendsto (fun x => fderiv Real f x) (nhdsWithin a t) (nhds (Co …
    this : HasDerivWithinAt f e (Set.Icc b a) a
    ⊢ HasDerivWithinAt f e (Set.Iic a) a
  -/
  exact this.mono_of_mem_nhdsWithin (Icc_mem_nhdsLE ba)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-10")] alias has_deriv_at_interval_right_endpoint_of_tendsto_deriv :=
  hasDerivWithinAt_Iic_of_tendsto_deriv


/-- If a real function `f` has a derivative `g` everywhere but at a point, and `f` and `g` are
continuous at this point, then `g` is also the derivative of `f` at this point. -/
theorem hasDerivAt_of_hasDerivAt_of_ne {f g : ℝ → E} {x : ℝ}
    (f_diff : ∀ y ≠ x, HasDerivAt f (g y) y) (hf : ContinuousAt f x)
    (hg : ContinuousAt g x) : HasDerivAt f (g x) x := by
  have A : HasDerivWithinAt f (g x) (Ici x) x := by
    have diff : DifferentiableOn ℝ f (Ioi x) := fun y hy =>
      (f_diff y (ne_of_gt hy)).differentiableAt.differentiableWithinAt
    -- next line is the nontrivial bit of this proof, appealing to differentiability
    -- extension results.
    apply
      hasDerivWithinAt_Ici_of_tendsto_deriv diff hf.continuousWithinAt
        self_mem_nhdsWithin
    have : Tendsto g (𝓝[>] x) (𝓝 (g x)) := tendsto_inf_left hg
    apply this.congr' _
    apply mem_of_superset self_mem_nhdsWithin fun y hy => _
    intros y hy
    exact (f_diff y (ne_of_gt hy)).deriv.symm
  have B : HasDerivWithinAt f (g x) (Iic x) x := by
    have diff : DifferentiableOn ℝ f (Iio x) := fun y hy =>
      (f_diff y (ne_of_lt hy)).differentiableAt.differentiableWithinAt
    -- next line is the nontrivial bit of this proof, appealing to differentiability
    -- extension results.
    apply
      hasDerivWithinAt_Iic_of_tendsto_deriv diff hf.continuousWithinAt
        self_mem_nhdsWithin
    have : Tendsto g (𝓝[<] x) (𝓝 (g x)) := tendsto_inf_left hg
    apply this.congr' _
    apply mem_of_superset self_mem_nhdsWithin fun y hy => _
    intros y hy
    exact (f_diff y (ne_of_lt hy)).deriv.symm
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : Real → E
    x : Real
    f_diff : ∀ (y : Real), Ne y x → HasDerivAt f (g y) y
    hf : ContinuousAt f x
    hg : ContinuousAt g x
    A : HasDerivWithinAt f (g x) (Set.Ici x) x
    B : HasDerivWithinAt f (g x) (Set.Iic x) x
    ⊢ HasDerivAt f (g x) x
  -/
  simpa using B.union A
  /-
    🎉 no goals
  -/


/-- If a real function `f` has a derivative `g` everywhere but at a point, and `f` and `g` are
continuous at this point, then `g` is the derivative of `f` everywhere. -/
theorem hasDerivAt_of_hasDerivAt_of_ne' {f g : ℝ → E} {x : ℝ}
    (f_diff : ∀ y ≠ x, HasDerivAt f (g y) y) (hf : ContinuousAt f x)
    (hg : ContinuousAt g x) (y : ℝ) : HasDerivAt f (g y) y := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f g : Real → E
    x : Real
    f_diff : ∀ (y : Real), Ne y x → HasDerivAt f (g y) y
    hf : ContinuousAt f x
    hg : ContinuousAt g x
    y : Real
    ⊢ HasDerivAt f (g y) y
  -/
  rcases eq_or_ne y x with (rfl | hne)
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : Real → E
      y : Real
      f_diff : ∀ (y_1 : Real), Ne y_1 y → HasDerivAt f (g y_1) y_1
      hf : ContinuousAt f y
      hg : ContinuousAt g y
      ⊢ HasDerivAt f (g y) y
    -/
  · exact hasDerivAt_of_hasDerivAt_of_ne f_diff hf hg
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f g : Real → E
      x : Real
      f_diff : ∀ (y : Real), Ne y x → HasDerivAt f (g y) y
      hf : ContinuousAt f x
      hg : ContinuousAt g x
      y : Real
      hne : Ne y x
      ⊢ HasDerivAt f (g y) y
    -/
  · exact f_diff y hne
    /-
      🎉 no goals
    -/

