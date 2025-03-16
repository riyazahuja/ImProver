/-- Auxiliary lemma for the divergence theorem. -/
theorem norm_volume_sub_integral_face_upper_sub_lower_smul_le {f : (Fin (n + 1) → ℝ) → E}
    {f' : (Fin (n + 1) → ℝ) →L[ℝ] E} (hfc : ContinuousOn f (Box.Icc I)) {x : Fin (n + 1) → ℝ}
    (hxI : x ∈ (Box.Icc I)) {a : E} {ε : ℝ} (h0 : 0 < ε)
    (hε : ∀ y ∈ (Box.Icc I), ‖f y - a - f' (y - x)‖ ≤ ε * ‖y - x‖) {c : ℝ≥0}
    (hc : I.distortion ≤ c) :
    ‖(∏ j, (I.upper j - I.lower j)) • f' (Pi.single i 1) -
      (integral (I.face i) ⊥ (f ∘ i.insertNth (α := fun _ ↦ ℝ) (I.upper i)) BoxAdditiveMap.volume -
        integral (I.face i) ⊥ (f ∘ i.insertNth (α := fun _ ↦ ℝ) (I.lower i))
          BoxAdditiveMap.volume)‖ ≤
      2 * ε * c * ∏ j, (I.upper j - I.lower j) := by
  -- Porting note: Lean fails to find `α` in the next line
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    i : Fin (HAdd.hAdd n 1)
    f : (Fin (HAdd.hAdd n 1) → Real) → E
    f' : ContinuousLinearMap (RingHom.id Real) (Fin (HAdd.hAdd n 1) → Real) E
    hfc : ContinuousOn f (BoxIntegral.Box.Icc I)
    x : Fin (HAdd.hAdd n 1) → Real
    hxI : Membership.mem (BoxIntegral.Box.Icc I) x
    a : E
    ε : Real
    h0 : LT.lt 0 ε
    hε : ∀ (y : Fin (HAdd.hAdd n 1) → Real), Membership.mem (BoxIntegral.Box.Icc I …
    c : NNReal
    hc : LE.le I.distortion c
    ⊢ LE.le (Norm.norm (HSub.hSub (HSMul.hSMul (Finset.univ.prod fun j => HSub.hSu …
  -/
  set e : ℝ → (Fin n → ℝ) → (Fin (n + 1) → ℝ) := i.insertNth (α := fun _ ↦ ℝ)
  /- **Plan of the proof**. The difference of the integrals of the affine function
    `fun y ↦ a + f' (y - x)` over the faces `x i = I.upper i` and `x i = I.lower i` is equal to the
    volume of `I` multiplied by `f' (Pi.single i 1)`, so it suffices to show that the integral of
    `f y - a - f' (y - x)` over each of these faces is less than or equal to `ε * c * vol I`. We
    integrate a function of the norm `≤ ε * diam I.Icc` over a box of volume
    `∏ j ≠ i, (I.upper j - I.lower j)`. Since `diam I.Icc ≤ c * (I.upper i - I.lower i)`, we get the
    required estimate. -/
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    i : Fin (HAdd.hAdd n 1)
    f : (Fin (HAdd.hAdd n 1) → Real) → E
    f' : ContinuousLinearMap (RingHom.id Real) (Fin (HAdd.hAdd n 1) → Real) E
    hfc : ContinuousOn f (BoxIntegral.Box.Icc I)
    x : Fin (HAdd.hAdd n 1) → Real
    hxI : Membership.mem (BoxIntegral.Box.Icc I) x
    a : E
    ε : Real
    h0 : LT.lt 0 ε
    hε : ∀ (y : Fin (HAdd.hAdd n 1) → Real), Membership.mem (BoxIntegral.Box.Icc I …
    c : NNReal
    hc : LE.le I.distortion c
    e : Real → (Fin n → Real) → Fin (HAdd.hAdd n 1) → Real := i.insertNth
    ⊢ LE.le (Norm.norm (HSub.hSub (HSMul.hSMul (Finset.univ.prod fun j => HSub.hSu …
  -/
  have Hl : I.lower i ∈ Icc (I.lower i) (I.upper i) := Set.left_mem_Icc.2 (I.lower_le_upper i)
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    i : Fin (HAdd.hAdd n 1)
    f : (Fin (HAdd.hAdd n 1) → Real) → E
    f' : ContinuousLinearMap (RingHom.id Real) (Fin (HAdd.hAdd n 1) → Real) E
    hfc : ContinuousOn f (BoxIntegral.Box.Icc I)
    x : Fin (HAdd.hAdd n 1) → Real
    hxI : Membership.mem (BoxIntegral.Box.Icc I) x
    a : E
    ε : Real
    h0 : LT.lt 0 ε
    hε : ∀ (y : Fin (HAdd.hAdd n 1) → Real), Membership.mem (BoxIntegral.Box.Icc I …
    c : NNReal
    hc : LE.le I.distortion c
    e : Real → (Fin n → Real) → Fin (HAdd.hAdd n 1) → Real := i.insertNth
    Hl : Membership.mem (Set.Icc (I.lower i) (I.upper i)) (I.lower i)
    ⊢ LE.le (Norm.norm (HSub.hSub (HSMul.hSMul (Finset.univ.prod fun j => HSub.hSu …
  -/
  have Hu : I.upper i ∈ Icc (I.lower i) (I.upper i) := Set.right_mem_Icc.2 (I.lower_le_upper i)
  have Hi : ∀ x ∈ Icc (I.lower i) (I.upper i),
      Integrable.{0, u, u} (I.face i) ⊥ (f ∘ e x) BoxAdditiveMap.volume := fun x hx =>
    integrable_of_continuousOn _ (Box.continuousOn_face_Icc hfc hx) volume
  /- We start with an estimate: the difference of the values of `f` at the corresponding points
    of the faces `x i = I.lower i` and `x i = I.upper i` is `(2 * ε * diam I.Icc)`-close to the
    value of `f'` on `Pi.single i (I.upper i - I.lower i) = lᵢ • eᵢ`, where
    `lᵢ = I.upper i - I.lower i` is the length of `i`-th edge of `I` and `eᵢ = Pi.single i 1` is the
    `i`-th unit vector. -/
  have : ∀ y ∈ Box.Icc (I.face i),
      ‖f' (Pi.single i (I.upper i - I.lower i)) -
          (f (e (I.upper i) y) - f (e (I.lower i) y))‖ ≤
        2 * ε * diam (Box.Icc I) := fun y hy ↦ by
    set g := fun y => f y - a - f' (y - x) with hg
    change ∀ y ∈ (Box.Icc I), ‖g y‖ ≤ ε * ‖y - x‖ at hε
    clear_value g; obtain rfl : f = fun y => a + f' (y - x) + g y := by simp [hg]
    convert_to ‖g (e (I.lower i) y) - g (e (I.upper i) y)‖ ≤ _
    · congr 1
      have := Fin.insertNth_sub_same (α := fun _ ↦ ℝ) i (I.upper i) (I.lower i) y
      simp only [← this, f'.map_sub]; abel
    · have : ∀ z ∈ Icc (I.lower i) (I.upper i), e z y ∈ (Box.Icc I) := fun z hz =>
        I.mapsTo_insertNth_face_Icc hz hy
      replace hε : ∀ y ∈ (Box.Icc I), ‖g y‖ ≤ ε * diam (Box.Icc I) := by
        intro y hy
        refine (hε y hy).trans (mul_le_mul_of_nonneg_left ?_ h0.le)
        rw [← dist_eq_norm]
        exact dist_le_diam_of_mem I.isCompact_Icc.isBounded hy hxI
      rw [two_mul, add_mul]
      exact norm_sub_le_of_le (hε _ (this _ Hl)) (hε _ (this _ Hu))
  calc
    ‖(∏ j, (I.upper j - I.lower j)) • f' (Pi.single i 1) -
            (integral (I.face i) ⊥ (f ∘ e (I.upper i)) BoxAdditiveMap.volume -
              integral (I.face i) ⊥ (f ∘ e (I.lower i)) BoxAdditiveMap.volume)‖ =
        ‖integral.{0, u, u} (I.face i) ⊥
            (fun x : Fin n → ℝ =>
              f' (Pi.single i (I.upper i - I.lower i)) -
                (f (e (I.upper i) x) - f (e (I.lower i) x)))
            BoxAdditiveMap.volume‖ := by
      rw [← integral_sub (Hi _ Hu) (Hi _ Hl), ← Box.volume_face_mul i, mul_smul, ← Box.volume_apply,
        ← BoxAdditiveMap.toSMul_apply, ← integral_const, ← BoxAdditiveMap.volume,
        ← integral_sub (integrable_const _) ((Hi _ Hu).sub (Hi _ Hl))]
      simp only [(· ∘ ·), Pi.sub_def, ← f'.map_smul, ← Pi.single_smul', smul_eq_mul, mul_one]
    _ ≤ (volume (I.face i : Set (Fin n → ℝ))).toReal * (2 * ε * c * (I.upper i - I.lower i)) := by
      -- The hard part of the estimate was done above, here we just replace `diam I.Icc`
      -- with `c * (I.upper i - I.lower i)`
      refine norm_integral_le_of_le_const (fun y hy => (this y hy).trans ?_) volume
      rw [mul_assoc (2 * ε)]
      gcongr
      exact I.diam_Icc_le_of_distortion_le i hc
    _ = 2 * ε * c * ∏ j, (I.upper j - I.lower j) := by
      rw [← Measure.toBoxAdditive_apply, Box.volume_apply, ← I.volume_face_mul i]
      ac_rfl


/-- If `f : ℝⁿ⁺¹ → E` is differentiable on a closed rectangular box `I` with derivative `f'`, then
the partial derivative `fun x ↦ f' x (Pi.single i 1)` is Henstock-Kurzweil integrable with integral
equal to the difference of integrals of `f` over the faces `x i = I.upper i` and `x i = I.lower i`.

More precisely, we use a non-standard generalization of the Henstock-Kurzweil integral and
we allow `f` to be non-differentiable (but still continuous) at a countable set of points.

TODO: If `n > 0`, then the condition at `x ∈ s` can be replaced by a much weaker estimate but this
requires either better integrability theorems, or usage of a filter depending on the countable set
`s` (we need to ensure that none of the faces of a partition contain a point from `s`). -/
theorem hasIntegral_GP_pderiv (f : (Fin (n + 1) → ℝ) → E)
    (f' : (Fin (n + 1) → ℝ) → (Fin (n + 1) → ℝ) →L[ℝ] E) (s : Set (Fin (n + 1) → ℝ))
    (hs : s.Countable) (Hs : ∀ x ∈ s, ContinuousWithinAt f (Box.Icc I) x)
    (Hd : ∀ x ∈ (Box.Icc I) \ s, HasFDerivWithinAt f (f' x) (Box.Icc I) x) (i : Fin (n + 1)) :
    HasIntegral.{0, u, u} I GP (fun x => f' x (Pi.single i 1)) BoxAdditiveMap.volume
      (integral.{0, u, u} (I.face i) GP (fun x => f (i.insertNth (I.upper i) x))
          BoxAdditiveMap.volume -
        integral.{0, u, u} (I.face i) GP (fun x => f (i.insertNth (I.lower i) x))
          BoxAdditiveMap.volume) := by
  /- Note that `f` is continuous on `I.Icc`, hence it is integrable on the faces of all boxes
    `J ≤ I`, thus the difference of integrals over `x i = J.upper i` and `x i = J.lower i` is a
    box-additive function of `J ≤ I`. -/
  have Hc : ContinuousOn f (Box.Icc I) := fun x hx ↦ by
    by_cases hxs : x ∈ s
    exacts [Hs x hxs, (Hd x ⟨hx, hxs⟩).continuousWithinAt]
  set fI : ℝ → Box (Fin n) → E := fun y J =>
    integral.{0, u, u} J GP (fun x => f (i.insertNth y x)) BoxAdditiveMap.volume
  set fb : Icc (I.lower i) (I.upper i) → Fin n →ᵇᵃ[↑(I.face i)] E := fun x =>
    (integrable_of_continuousOn GP (Box.continuousOn_face_Icc Hc x.2) volume).toBoxAdditive
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    i : Fin (HAdd.hAdd n 1)
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
    fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
    ⊢ BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => (f' x)  …
  -/
  set F : Fin (n + 1) →ᵇᵃ[I] E := BoxAdditiveMap.upperSubLower I i fI fb fun x _ J => rfl
  -- Thus our statement follows from some local estimates.
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    i : Fin (HAdd.hAdd n 1)
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
    fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
    F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
    ⊢ BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => (f' x)  …
  -/
  change HasIntegral I GP (fun x => f' x (Pi.single i 1)) _ (F I)
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    i : Fin (HAdd.hAdd n 1)
    Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
    fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
    fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
    F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
    ⊢ BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => (f' x)  …
  -/
  refine HasIntegral.of_le_Henstock_of_forall_isLittleO gp_le ?_ ?_ _ s hs ?_ ?_
  ·-- We use the volume as an upper estimate.
    /-
      case refine_1
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      ⊢ BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) Real ↑I
    -/
    exact (volume : Measure (Fin (n + 1) → ℝ)).toBoxAdditive.restrict _ le_top
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      ⊢ ∀ (J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))), LE.le 0 ((MeasureTheory.Measu …
    -/
  · exact fun J => ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      ⊢ ∀ (c : NNReal) (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter …
    -/
  · intro c x hx ε ε0
    /- Near `x ∈ s` we choose `δ` so that both vectors are small. `volume J • eᵢ` is small because
        `volume J ≤ (2 * δ) ^ (n + 1)` is small, and the difference of the integrals is small
        because each of the integrals is close to `volume (J.face i) • f x`.
        TODO: there should be a shorter and more readable way to formalize this simple proof. -/
    have : ∀ᶠ δ in 𝓝[>] (0 : ℝ), δ ∈ Ioc (0 : ℝ) (1 / 2) ∧
        (∀ᵉ (y₁ ∈ closedBall x δ ∩ (Box.Icc I)) (y₂ ∈ closedBall x δ ∩ (Box.Icc I)),
              ‖f y₁ - f y₂‖ ≤ ε / 2) ∧ (2 * δ) ^ (n + 1) * ‖f' x (Pi.single i 1)‖ ≤ ε / 2 := by
      refine .and (Ioc_mem_nhdsGT one_half_pos) (.and ?_ ?_)
      · rcases ((nhdsWithin_hasBasis nhds_basis_closedBall _).tendsto_iff nhds_basis_closedBall).1
            (Hs x hx.2) _ (half_pos <| half_pos ε0) with ⟨δ₁, δ₁0, hδ₁⟩
        filter_upwards [Ioc_mem_nhdsGT δ₁0] with δ hδ y₁ hy₁ y₂ hy₂
        have : closedBall x δ ∩ (Box.Icc I) ⊆ closedBall x δ₁ ∩ (Box.Icc I) := by gcongr; exact hδ.2
        rw [← dist_eq_norm]
        calc
          dist (f y₁) (f y₂) ≤ dist (f y₁) (f x) + dist (f y₂) (f x) := dist_triangle_right _ _ _
          _ ≤ ε / 2 / 2 + ε / 2 / 2 := add_le_add (hδ₁ _ <| this hy₁) (hδ₁ _ <| this hy₂)
          _ = ε / 2 := add_halves _
      · have : ContinuousWithinAt (fun δ : ℝ => (2 * δ) ^ (n + 1) * ‖f' x (Pi.single i 1)‖)
            (Ioi 0) 0 := ((continuousWithinAt_id.const_mul _).pow _).mul_const _
        refine this.eventually (ge_mem_nhds ?_)
        simpa using half_pos ε0
    /-
      case refine_3
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
      ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (J : BoxIntegral.Box (Fin (HAdd.hAdd n 1) …
    -/
    rcases this.exists with ⟨δ, ⟨hδ0, hδ12⟩, hdfδ, hδ⟩
    /-
      case refine_3.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
      δ : Real
      hδ0 : LT.lt 0 δ
      hδ12 : LE.le δ (1 / 2)
      hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
      hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
      ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (J : BoxIntegral.Box (Fin (HAdd.hAdd n 1) …
    -/
    refine ⟨δ, hδ0, fun J hJI hJδ _ _ => add_halves ε ▸ ?_⟩
    /-
      case refine_3.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
      δ : Real
      hδ0 : LT.lt 0 δ
      hδ12 : LE.le δ (1 / 2)
      hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
      hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
      J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      hJI : LE.le J I
      hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
      x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
      x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
      ⊢ LE.le (Dist.dist ((BoxIntegral.BoxAdditiveMap.volume J) ((f' x) (Pi.single i …
    -/
    have Hl : J.lower i ∈ Icc (J.lower i) (J.upper i) := Set.left_mem_Icc.2 (J.lower_le_upper i)
    /-
      case refine_3.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
      δ : Real
      hδ0 : LT.lt 0 δ
      hδ12 : LE.le δ (1 / 2)
      hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
      hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
      J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      hJI : LE.le J I
      hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
      x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
      x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
      Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
      ⊢ LE.le (Dist.dist ((BoxIntegral.BoxAdditiveMap.volume J) ((f' x) (Pi.single i …
    -/
    have Hu : J.upper i ∈ Icc (J.lower i) (J.upper i) := Set.right_mem_Icc.2 (J.lower_le_upper i)
    have Hi : ∀ x ∈ Icc (J.lower i) (J.upper i),
        Integrable.{0, u, u} (J.face i) GP (fun y => f (i.insertNth x y))
          BoxAdditiveMap.volume := fun x hx =>
      integrable_of_continuousOn _ (Box.continuousOn_face_Icc (Hc.mono <| Box.le_iff_Icc.1 hJI) hx)
        volume
    /-
      case refine_3.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
      δ : Real
      hδ0 : LT.lt 0 δ
      hδ12 : LE.le δ (1 / 2)
      hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
      hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
      J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      hJI : LE.le J I
      hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
      x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
      x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
      Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
      Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
      Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
      ⊢ LE.le (Dist.dist ((BoxIntegral.BoxAdditiveMap.volume J) ((f' x) (Pi.single i …
    -/
    have hJδ' : Box.Icc J ⊆ closedBall x δ ∩ (Box.Icc I) := subset_inter hJδ (Box.le_iff_Icc.1 hJI)
    have Hmaps : ∀ z ∈ Icc (J.lower i) (J.upper i),
        MapsTo (i.insertNth z) (Box.Icc (J.face i)) (closedBall x δ ∩ (Box.Icc I)) := fun z hz =>
      (J.mapsTo_insertNth_face_Icc hz).mono Subset.rfl hJδ'
    /-
      case refine_3.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
      δ : Real
      hδ0 : LT.lt 0 δ
      hδ12 : LE.le δ (1 / 2)
      hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
      hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
      J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      hJI : LE.le J I
      hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
      x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
      x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
      Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
      Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
      Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
      hJδ' : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (Metric.closedBal …
      Hmaps : ∀ (z : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) z → Set …
      ⊢ LE.le (Dist.dist ((BoxIntegral.BoxAdditiveMap.volume J) ((f' x) (Pi.single i …
    -/
    simp only [dist_eq_norm]; dsimp [F]
    /-
      case refine_3.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
      δ : Real
      hδ0 : LT.lt 0 δ
      hδ12 : LE.le δ (1 / 2)
      hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
      hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
      J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      hJI : LE.le J I
      hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
      x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
      x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
      Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
      Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
      Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
      hJδ' : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (Metric.closedBal …
      Hmaps : ∀ (z : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) z → Set …
      ⊢ LE.le (Norm.norm (HSub.hSub ((BoxIntegral.BoxAdditiveMap.volume J) ((f' x) ( …
    -/
    rw [← integral_sub (Hi _ Hu) (Hi _ Hl)]
    /-
      case refine_3.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
      δ : Real
      hδ0 : LT.lt 0 δ
      hδ12 : LE.le δ (1 / 2)
      hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
      hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
      J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      hJI : LE.le J I
      hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
      x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
      x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
      Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
      Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
      Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
      hJδ' : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (Metric.closedBal …
      Hmaps : ∀ (z : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) z → Set …
      ⊢ LE.le (Norm.norm (HSub.hSub ((BoxIntegral.BoxAdditiveMap.volume J) ((f' x) ( …
    -/
    refine (norm_sub_le _ _).trans (add_le_add ?_ ?_)
      /-
        case refine_3.intro.intro.intro.intro.refine_1
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        n : Nat
        inst✝ : CompleteSpace E
        I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        f : (Fin (HAdd.hAdd n 1) → Real) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
        i : Fin (HAdd.hAdd n 1)
        Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
        fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
        fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
        F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
        c : NNReal
        x : Fin (HAdd.hAdd n 1) → Real
        hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
        ε : Real
        ε0 : GT.gt ε 0
        this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
        δ : Real
        hδ0 : LT.lt 0 δ
        hδ12 : LE.le δ (1 / 2)
        hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
        hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hJI : LE.le J I
        hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
        x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
        x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
        Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
        Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
        Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
        hJδ' : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (Metric.closedBal …
        Hmaps : ∀ (z : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) z → Set …
        ⊢ LE.le (Norm.norm ((BoxIntegral.BoxAdditiveMap.volume J) ((f' x) (Pi.single i …
      -/
    · simp_rw [BoxAdditiveMap.volume_apply, norm_smul, Real.norm_eq_abs, abs_prod]
      /-
        case refine_3.intro.intro.intro.intro.refine_1
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        n : Nat
        inst✝ : CompleteSpace E
        I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        f : (Fin (HAdd.hAdd n 1) → Real) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
        i : Fin (HAdd.hAdd n 1)
        Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
        fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
        fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
        F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
        c : NNReal
        x : Fin (HAdd.hAdd n 1) → Real
        hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
        ε : Real
        ε0 : GT.gt ε 0
        this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
        δ : Real
        hδ0 : LT.lt 0 δ
        hδ12 : LE.le δ (1 / 2)
        hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
        hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hJI : LE.le J I
        hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
        x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
        x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
        Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
        Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
        Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
        hJδ' : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (Metric.closedBal …
        Hmaps : ∀ (z : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) z → Set …
        ⊢ LE.le (HMul.hMul (Finset.univ.prod fun x => abs (HSub.hSub (J.upper x) (J.lo …
      -/
      refine (mul_le_mul_of_nonneg_right ?_ <| norm_nonneg _).trans hδ
      have : ∀ j, |J.upper j - J.lower j| ≤ 2 * δ := fun j ↦
        calc
          dist (J.upper j) (J.lower j) ≤ dist J.upper J.lower := dist_le_pi_dist _ _ _
          _ ≤ dist J.upper x + dist J.lower x := dist_triangle_right _ _ _
          _ ≤ δ + δ := add_le_add (hJδ J.upper_mem_Icc) (hJδ J.lower_mem_Icc)
          _ = 2 * δ := (two_mul δ).symm
      calc
        ∏ j, |J.upper j - J.lower j| ≤ ∏ j : Fin (n + 1), 2 * δ :=
          prod_le_prod (fun _ _ => abs_nonneg _) fun j _ => this j
        _ = (2 * δ) ^ (n + 1) := by simp
    · refine (norm_integral_le_of_le_const (fun y hy => hdfδ _ (Hmaps _ Hu hy) _
        (Hmaps _ Hl hy)) volume).trans ?_
      /-
        case refine_3.intro.intro.intro.intro.refine_2
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        n : Nat
        inst✝ : CompleteSpace E
        I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        f : (Fin (HAdd.hAdd n 1) → Real) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
        i : Fin (HAdd.hAdd n 1)
        Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
        fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
        fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
        F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
        c : NNReal
        x : Fin (HAdd.hAdd n 1) → Real
        hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
        ε : Real
        ε0 : GT.gt ε 0
        this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
        δ : Real
        hδ0 : LT.lt 0 δ
        hδ12 : LE.le δ (1 / 2)
        hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
        hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hJI : LE.le J I
        hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
        x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
        x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
        Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
        Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
        Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
        hJδ' : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (Metric.closedBal …
        Hmaps : ∀ (z : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) z → Set …
        ⊢ LE.le (HMul.hMul (MeasureTheory.MeasureSpace.volume ↑(J.face i)).toReal (HDi …
      -/
      refine (mul_le_mul_of_nonneg_right ?_ (half_pos ε0).le).trans_eq (one_mul _)
      /-
        case refine_3.intro.intro.intro.intro.refine_2
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        n : Nat
        inst✝ : CompleteSpace E
        I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        f : (Fin (HAdd.hAdd n 1) → Real) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
        i : Fin (HAdd.hAdd n 1)
        Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
        fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
        fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
        F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
        c : NNReal
        x : Fin (HAdd.hAdd n 1) → Real
        hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
        ε : Real
        ε0 : GT.gt ε 0
        this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
        δ : Real
        hδ0 : LT.lt 0 δ
        hδ12 : LE.le δ (1 / 2)
        hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
        hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hJI : LE.le J I
        hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
        x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
        x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
        Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
        Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
        Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
        hJδ' : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (Metric.closedBal …
        Hmaps : ∀ (z : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) z → Set …
        ⊢ LE.le (MeasureTheory.MeasureSpace.volume ↑(J.face i)).toReal 1
      -/
      rw [Box.coe_eq_pi, Real.volume_pi_Ioc_toReal (Box.lower_le_upper _)]
      /-
        case refine_3.intro.intro.intro.intro.refine_2
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        n : Nat
        inst✝ : CompleteSpace E
        I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        f : (Fin (HAdd.hAdd n 1) → Real) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
        i : Fin (HAdd.hAdd n 1)
        Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
        fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
        fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
        F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
        c : NNReal
        x : Fin (HAdd.hAdd n 1) → Real
        hx : Membership.mem (Inter.inter (BoxIntegral.Box.Icc I) s) x
        ε : Real
        ε0 : GT.gt ε 0
        this : Filter.Eventually (fun δ => And (Membership.mem (Set.Ioc 0 (1 / 2)) δ)  …
        δ : Real
        hδ0 : LT.lt 0 δ
        hδ12 : LE.le δ (1 / 2)
        hdfδ : ∀ (y₁ : Fin (HAdd.hAdd n 1) → Real), Membership.mem (Inter.inter (Metri …
        hδ : LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 δ) (HAdd.hAdd n 1)) (Norm.norm ( …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hJI : LE.le J I
        hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
        x✝¹ : Membership.mem (BoxIntegral.Box.Icc J) x
        x✝ : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.disto …
        Hl : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.lower i)
        Hu : Membership.mem (Set.Icc (J.lower i) (J.upper i)) (J.upper i)
        Hi : ∀ (x : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) x → BoxInt …
        hJδ' : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (Metric.closedBal …
        Hmaps : ∀ (z : Real), Membership.mem (Set.Icc (J.lower i) (J.upper i)) z → Set …
        ⊢ LE.le (Finset.univ.prod fun i_1 => HSub.hSub ((J.face i).upper i_1) ((J.face …
      -/
      refine prod_le_one (fun _ _ => sub_nonneg.2 <| Box.lower_le_upper _ _) fun j _ => ?_
      calc
        J.upper (i.succAbove j) - J.lower (i.succAbove j) ≤
            dist (J.upper (i.succAbove j)) (J.lower (i.succAbove j)) :=
          le_abs_self _
        _ ≤ dist J.upper J.lower := dist_le_pi_dist J.upper J.lower (i.succAbove j)
        _ ≤ dist J.upper x + dist J.lower x := dist_triangle_right _ _ _
        _ ≤ δ + δ := add_le_add (hJδ J.upper_mem_Icc) (hJδ J.lower_mem_Icc)
        _ ≤ 1 / 2 + 1 / 2 := by gcongr
        _ = 1 := add_halves 1
    /-
      case refine_4
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      ⊢ ∀ (c : NNReal) (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff …
    -/
  · intro c x hx ε ε0
    /- At a point `x ∉ s`, we unfold the definition of Fréchet differentiability, then use
        an estimate we proved earlier in this file. -/
    /-
      case refine_4
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (J : BoxIntegral.Box (Fin (HAdd.hAdd n 1) …
    -/
    rcases exists_pos_mul_lt ε0 (2 * c) with ⟨ε', ε'0, hlt⟩
    rcases (nhdsWithin_hasBasis nhds_basis_closedBall _).mem_iff.1
      ((Hd x hx).isLittleO.def ε'0) with ⟨δ, δ0, Hδ⟩
    /-
      case refine_4.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      ε' : Real
      ε'0 : LT.lt 0 ε'
      hlt : LT.lt (HMul.hMul (HMul.hMul 2 ↑c) ε') ε
      δ : Real
      δ0 : LT.lt 0 δ
      Hδ : HasSubset.Subset (Inter.inter (Metric.closedBall x δ) (BoxIntegral.Box.Ic …
      ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (J : BoxIntegral.Box (Fin (HAdd.hAdd n 1) …
    -/
    refine ⟨δ, δ0, fun J hle hJδ hxJ hJc => ?_⟩
    /-
      case refine_4.intro.intro.intro.intro
      E : Type u
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      n : Nat
      inst✝ : CompleteSpace E
      I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      f : (Fin (HAdd.hAdd n 1) → Real) → E
      f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
      s : Set (Fin (HAdd.hAdd n 1) → Real)
      hs : s.Countable
      Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
      Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
      i : Fin (HAdd.hAdd n 1)
      Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
      fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
      fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
      F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
      c : NNReal
      x : Fin (HAdd.hAdd n 1) → Real
      hx : Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) s) x
      ε : Real
      ε0 : GT.gt ε 0
      ε' : Real
      ε'0 : LT.lt 0 ε'
      hlt : LT.lt (HMul.hMul (HMul.hMul 2 ↑c) ε') ε
      δ : Real
      δ0 : LT.lt 0 δ
      Hδ : HasSubset.Subset (Inter.inter (Metric.closedBall x δ) (BoxIntegral.Box.Ic …
      J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
      hle : LE.le J I
      hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
      hxJ : Membership.mem (BoxIntegral.Box.Icc J) x
      hJc : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.dist …
      ⊢ LE.le (Dist.dist ((BoxIntegral.BoxAdditiveMap.volume J) ((f' x) (Pi.single i …
    -/
    simp only [BoxAdditiveMap.volume_apply, Box.volume_apply, dist_eq_norm]
    refine (norm_volume_sub_integral_face_upper_sub_lower_smul_le _
      (Hc.mono <| Box.le_iff_Icc.1 hle) hxJ ε'0 (fun y hy => Hδ ?_) (hJc rfl)).trans ?_
      /-
        case refine_4.intro.intro.intro.intro.refine_1
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        n : Nat
        inst✝ : CompleteSpace E
        I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        f : (Fin (HAdd.hAdd n 1) → Real) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
        i : Fin (HAdd.hAdd n 1)
        Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
        fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
        fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
        F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
        c : NNReal
        x : Fin (HAdd.hAdd n 1) → Real
        hx : Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) s) x
        ε : Real
        ε0 : GT.gt ε 0
        ε' : Real
        ε'0 : LT.lt 0 ε'
        hlt : LT.lt (HMul.hMul (HMul.hMul 2 ↑c) ε') ε
        δ : Real
        δ0 : LT.lt 0 δ
        Hδ : HasSubset.Subset (Inter.inter (Metric.closedBall x δ) (BoxIntegral.Box.Ic …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hle : LE.le J I
        hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
        hxJ : Membership.mem (BoxIntegral.Box.Icc J) x
        hJc : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.dist …
        y : Fin (HAdd.hAdd n 1) → Real
        hy : Membership.mem (BoxIntegral.Box.Icc J) y
        ⊢ Membership.mem (Inter.inter (Metric.closedBall x δ) (BoxIntegral.Box.Icc I)) y
      -/
    · exact ⟨hJδ hy, Box.le_iff_Icc.1 hle hy⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_4.intro.intro.intro.intro.refine_2
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        n : Nat
        inst✝ : CompleteSpace E
        I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        f : (Fin (HAdd.hAdd n 1) → Real) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
        i : Fin (HAdd.hAdd n 1)
        Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
        fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
        fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
        F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
        c : NNReal
        x : Fin (HAdd.hAdd n 1) → Real
        hx : Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) s) x
        ε : Real
        ε0 : GT.gt ε 0
        ε' : Real
        ε'0 : LT.lt 0 ε'
        hlt : LT.lt (HMul.hMul (HMul.hMul 2 ↑c) ε') ε
        δ : Real
        δ0 : LT.lt 0 δ
        Hδ : HasSubset.Subset (Inter.inter (Metric.closedBall x δ) (BoxIntegral.Box.Ic …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hle : LE.le J I
        hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
        hxJ : Membership.mem (BoxIntegral.Box.Icc J) x
        hJc : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.dist …
        ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul 2 ε') ↑c) (Finset.univ.prod fun j =>  …
      -/
    · rw [mul_right_comm (2 : ℝ), ← Box.volume_apply]
      /-
        case refine_4.intro.intro.intro.intro.refine_2
        E : Type u
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        n : Nat
        inst✝ : CompleteSpace E
        I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        f : (Fin (HAdd.hAdd n 1) → Real) → E
        f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
        s : Set (Fin (HAdd.hAdd n 1) → Real)
        hs : s.Countable
        Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
        Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
        i : Fin (HAdd.hAdd n 1)
        Hc : ContinuousOn f (BoxIntegral.Box.Icc I)
        fI : Real → BoxIntegral.Box (Fin n) → E := fun y J => BoxIntegral.integral J B …
        fb : ↑(Set.Icc (I.lower i) (I.upper i)) → BoxIntegral.BoxAdditiveMap (Fin n) E …
        F : BoxIntegral.BoxAdditiveMap (Fin (HAdd.hAdd n 1)) E ↑I := BoxIntegral.BoxAd …
        c : NNReal
        x : Fin (HAdd.hAdd n 1) → Real
        hx : Membership.mem (SDiff.sdiff (BoxIntegral.Box.Icc I) s) x
        ε : Real
        ε0 : GT.gt ε 0
        ε' : Real
        ε'0 : LT.lt 0 ε'
        hlt : LT.lt (HMul.hMul (HMul.hMul 2 ↑c) ε') ε
        δ : Real
        δ0 : LT.lt 0 δ
        Hδ : HasSubset.Subset (Inter.inter (Metric.closedBall x δ) (BoxIntegral.Box.Ic …
        J : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
        hle : LE.le J I
        hJδ : HasSubset.Subset (BoxIntegral.Box.Icc J) (Metric.closedBall x δ)
        hxJ : Membership.mem (BoxIntegral.Box.Icc J) x
        hJc : Eq BoxIntegral.IntegrationParams.GP.bDistortion Bool.true → LE.le J.dist …
        ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑c) ε') (MeasureTheory.MeasureSpace …
      -/
      exact mul_le_mul_of_nonneg_right hlt.le ENNReal.toReal_nonneg
      /-
        🎉 no goals
      -/


/-- Divergence theorem for a Henstock-Kurzweil style integral.

If `f : ℝⁿ⁺¹ → Eⁿ⁺¹` is differentiable on a closed rectangular box `I` with derivative `f'`, then
the divergence `∑ i, f' x (Pi.single i 1) i` is Henstock-Kurzweil integrable with integral equal to
the sum of integrals of `f` over the faces of `I` taken with appropriate signs.

More precisely, we use a non-standard generalization of the Henstock-Kurzweil integral and
we allow `f` to be non-differentiable (but still continuous) at a countable set of points. -/
theorem hasIntegral_GP_divergence_of_forall_hasDerivWithinAt
    (f : (Fin (n + 1) → ℝ) → Fin (n + 1) → E)
    (f' : (Fin (n + 1) → ℝ) → (Fin (n + 1) → ℝ) →L[ℝ] (Fin (n + 1) → E))
    (s : Set (Fin (n + 1) → ℝ)) (hs : s.Countable)
    (Hs : ∀ x ∈ s, ContinuousWithinAt f (Box.Icc I) x)
    (Hd : ∀ x ∈ (Box.Icc I) \ s, HasFDerivWithinAt f (f' x) (Box.Icc I) x) :
    HasIntegral.{0, u, u} I GP (fun x => ∑ i, f' x (Pi.single i 1) i) BoxAdditiveMap.volume
      (∑ i,
        (integral.{0, u, u} (I.face i) GP (fun x => f (i.insertNth (I.upper i) x) i)
            BoxAdditiveMap.volume -
          integral.{0, u, u} (I.face i) GP (fun x => f (i.insertNth (I.lower i) x) i)
            BoxAdditiveMap.volume)) := by
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    ⊢ BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => Finset. …
  -/
  refine HasIntegral.sum fun i _ => ?_
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ContinuousWithin …
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    i : Fin (HAdd.hAdd n 1)
    x✝ : Membership.mem Finset.univ i
    ⊢ BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => (f' x)  …
  -/
  simp only [hasFDerivWithinAt_pi', continuousWithinAt_pi] at Hd Hs
  /-
    E : Type u
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    n : Nat
    inst✝ : CompleteSpace E
    I : BoxIntegral.Box (Fin (HAdd.hAdd n 1))
    f : (Fin (HAdd.hAdd n 1) → Real) → Fin (HAdd.hAdd n 1) → E
    f' : (Fin (HAdd.hAdd n 1) → Real) → ContinuousLinearMap (RingHom.id Real) (Fin …
    s : Set (Fin (HAdd.hAdd n 1) → Real)
    hs : s.Countable
    i : Fin (HAdd.hAdd n 1)
    x✝ : Membership.mem Finset.univ i
    Hd : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem (SDiff.sdiff (BoxInteg …
    Hs : ∀ (x : Fin (HAdd.hAdd n 1) → Real), Membership.mem s x → ∀ (i : Fin (HAdd …
    ⊢ BoxIntegral.HasIntegral I BoxIntegral.IntegrationParams.GP (fun x => (f' x)  …
  -/
  exact hasIntegral_GP_pderiv I _ _ s hs (fun x hx => Hs x hx i) (fun x hx => Hd x hx i) i
  /-
    🎉 no goals
  -/


